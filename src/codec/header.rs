//! HEC0 wire-format header: serialize/deserialize the type tag and pipeline
//! description that prefix every self-describing codec buffer.
//!
//! ## Wire layout (frozen — bump magic to `HEC1` if layout changes)
//!
//! ```text
//! [0..4]   magic        = b"HEC0"          (4 bytes)
//! [4..5]   header_len   u8                 (bytes from [5] to end of header)
//! [5..6]   data_type    u8                 (DataType discriminant — frozen table below)
//! [6..7]   stage_count  u8
//! [7..]    stage_blocks (stage_count of):
//!            [0..1]    coder_id_len  u8
//!            [1..N]    coder_id      bytes (UTF-8)
//!            [N..N+1]  param_count   u8
//!            per param:
//!              [0..1]    key_len  u8
//!              [1..K]    key      bytes (UTF-8)
//!              [K..K+1]  val_len  u8
//!              [K+1..]   val      bytes (UTF-8 / JSON value)
//! [after header] value_count  varint (LEB128)
//! [after that]   encoded body (pipeline output)
//! ```
//!
//! ## DataType discriminant table (frozen)
//!
//! | byte | DataType |
//! |------|----------|
//! | 1    | I8       |
//! | 2    | I16      |
//! | 3    | I32      |
//! | 4    | I64      |
//! | 5    | U8       |
//! | 6    | U16      |
//! | 7    | U32      |
//! | 8    | U64      |
//! | 9    | F32      |
//! | 10   | F64      |
//! | 11   | Bytes    |

use crate::core::coder::DataType;
use crate::core::error::{HeliumError, Result};
use crate::core::registry::CoderSpec;

/// Magic bytes identifying the HEC0 self-describing codec format.
pub(crate) const HEC0_MAGIC: &[u8; 4] = b"HEC0";

/// Convert a `DataType` to its wire-format discriminant byte.
///
/// This table is wire-format-frozen: values must never change once any byte
/// produced with this format has been shipped.
pub(crate) fn data_type_to_byte(dt: DataType) -> u8 {
    match dt {
        DataType::I8 => 1,
        DataType::I16 => 2,
        DataType::I32 => 3,
        DataType::I64 => 4,
        DataType::U8 => 5,
        DataType::U16 => 6,
        DataType::U32 => 7,
        DataType::U64 => 8,
        DataType::F32 => 9,
        DataType::F64 => 10,
        DataType::Bytes => 11,
    }
}

/// Convert a wire-format discriminant byte to a `DataType`.
pub(crate) fn byte_to_data_type(b: u8) -> Result<DataType> {
    match b {
        1 => Ok(DataType::I8),
        2 => Ok(DataType::I16),
        3 => Ok(DataType::I32),
        4 => Ok(DataType::I64),
        5 => Ok(DataType::U8),
        6 => Ok(DataType::U16),
        7 => Ok(DataType::U32),
        8 => Ok(DataType::U64),
        9 => Ok(DataType::F32),
        10 => Ok(DataType::F64),
        11 => Ok(DataType::Bytes),
        other => Err(HeliumError::Format(format!(
            "HEC0: unknown DataType discriminant byte {other}"
        ))),
    }
}

/// Parsed representation of the HEC0 header.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Hec0Header {
    pub data_type: DataType,
    pub stages: Vec<CoderSpec>,
}

/// Serialize a `Hec0Header` plus `value_count` into a byte buffer.
///
/// Produces: `HEC0_MAGIC | header_len | <type byte> | <stage_count> | <stages…> | <value_count LEB128>`
pub(crate) fn write_header(header: &Hec0Header, value_count: u64) -> Result<Vec<u8>> {
    // Build the type+pipeline block first so we can measure its length.
    let mut block: Vec<u8> = Vec::new();
    block.push(data_type_to_byte(header.data_type));

    let stage_count = header.stages.len();
    if stage_count > 255 {
        return Err(HeliumError::Format(format!(
            "HEC0: too many pipeline stages ({stage_count}); max 255"
        )));
    }
    block.push(stage_count as u8);

    for spec in &header.stages {
        let id_bytes = spec.id.as_bytes();
        if id_bytes.len() > 255 {
            return Err(HeliumError::Format(format!(
                "HEC0: coder id '{}' is too long (max 255 bytes)",
                spec.id
            )));
        }
        block.push(id_bytes.len() as u8);
        block.extend_from_slice(id_bytes);

        let param_count = spec.params.len();
        if param_count > 255 {
            return Err(HeliumError::Format(format!(
                "HEC0: coder '{}' has too many params (max 255)",
                spec.id
            )));
        }
        block.push(param_count as u8);

        // Sort params for deterministic output.
        let mut params: Vec<(&String, &serde_json::Value)> = spec.params.iter().collect();
        params.sort_by_key(|(k, _)| k.as_str());

        for (key, val) in params {
            let key_bytes = key.as_bytes();
            if key_bytes.len() > 255 {
                return Err(HeliumError::Format(format!(
                    "HEC0: param key '{key}' is too long (max 255 bytes)"
                )));
            }
            block.push(key_bytes.len() as u8);
            block.extend_from_slice(key_bytes);

            let val_str = val.to_string();
            let val_bytes = val_str.as_bytes();
            if val_bytes.len() > 255 {
                return Err(HeliumError::Format(format!(
                    "HEC0: param value for key '{key}' is too long (max 255 bytes)"
                )));
            }
            block.push(val_bytes.len() as u8);
            block.extend_from_slice(val_bytes);
        }
    }

    let header_len = block.len();
    if header_len > 255 {
        return Err(HeliumError::Format(format!(
            "HEC0: header block is too large ({header_len} bytes); max 255"
        )));
    }

    let mut out: Vec<u8> = Vec::with_capacity(4 + 1 + header_len + 10);
    out.extend_from_slice(HEC0_MAGIC);
    out.push(header_len as u8);
    out.extend_from_slice(&block);

    // Write value_count as LEB128.
    write_leb128_u64(&mut out, value_count);

    Ok(out)
}

/// Parse a `Hec0Header` and `value_count` from the start of `bytes`.
///
/// Returns `(header, value_count, body_offset)` where `body_offset` is the
/// index into `bytes` where the encoded column body starts.
pub(crate) fn read_header(bytes: &[u8]) -> Result<(Hec0Header, u64, usize)> {
    if bytes.len() < 4 {
        return Err(HeliumError::Format(
            "HEC0: buffer too short to contain magic".into(),
        ));
    }
    if &bytes[0..4] != HEC0_MAGIC {
        return Err(HeliumError::Format(format!(
            "HEC0: bad magic bytes {:?}; expected {:?}",
            &bytes[0..4],
            HEC0_MAGIC
        )));
    }

    if bytes.len() < 5 {
        return Err(HeliumError::Format(
            "HEC0: buffer truncated at header_len".into(),
        ));
    }
    let header_len = bytes[4] as usize;

    // The block starts at byte 5 and is `header_len` bytes long.
    let block_end = 5 + header_len;
    if bytes.len() < block_end {
        return Err(HeliumError::Format(format!(
            "HEC0: buffer too short for header block (need {} bytes after magic+len, have {})",
            header_len,
            bytes.len() - 5
        )));
    }
    let block = &bytes[5..block_end];

    if block.len() < 2 {
        return Err(HeliumError::Format(
            "HEC0: header block too short (need data_type + stage_count)".into(),
        ));
    }

    let data_type = byte_to_data_type(block[0])?;
    let stage_count = block[1] as usize;

    let mut pos = 2usize;
    let mut stages: Vec<CoderSpec> = Vec::with_capacity(stage_count);

    for stage_idx in 0..stage_count {
        if pos >= block.len() {
            return Err(HeliumError::Format(format!(
                "HEC0: truncated at stage {stage_idx} coder_id_len"
            )));
        }
        let id_len = block[pos] as usize;
        pos += 1;
        if pos + id_len > block.len() {
            return Err(HeliumError::Format(format!(
                "HEC0: truncated at stage {stage_idx} coder_id"
            )));
        }
        let id = std::str::from_utf8(&block[pos..pos + id_len]).map_err(|e| {
            HeliumError::Format(format!(
                "HEC0: coder id at stage {stage_idx} is not UTF-8: {e}"
            ))
        })?;
        pos += id_len;

        let mut spec = CoderSpec::new(id);

        if pos >= block.len() {
            return Err(HeliumError::Format(format!(
                "HEC0: truncated at stage {stage_idx} param_count"
            )));
        }
        let param_count = block[pos] as usize;
        pos += 1;

        for param_idx in 0..param_count {
            if pos >= block.len() {
                return Err(HeliumError::Format(format!(
                    "HEC0: truncated at stage {stage_idx} param {param_idx} key_len"
                )));
            }
            let key_len = block[pos] as usize;
            pos += 1;
            if pos + key_len > block.len() {
                return Err(HeliumError::Format(format!(
                    "HEC0: truncated at stage {stage_idx} param {param_idx} key"
                )));
            }
            let key = std::str::from_utf8(&block[pos..pos + key_len]).map_err(|e| {
                HeliumError::Format(format!(
                    "HEC0: param key at stage {stage_idx} param {param_idx} is not UTF-8: {e}"
                ))
            })?;
            pos += key_len;

            if pos >= block.len() {
                return Err(HeliumError::Format(format!(
                    "HEC0: truncated at stage {stage_idx} param {param_idx} val_len"
                )));
            }
            let val_len = block[pos] as usize;
            pos += 1;
            if pos + val_len > block.len() {
                return Err(HeliumError::Format(format!(
                    "HEC0: truncated at stage {stage_idx} param {param_idx} val"
                )));
            }
            let val_str = std::str::from_utf8(&block[pos..pos + val_len]).map_err(|e| {
                HeliumError::Format(format!(
                    "HEC0: param value at stage {stage_idx} param {param_idx} is not UTF-8: {e}"
                ))
            })?;
            pos += val_len;

            let val: serde_json::Value = serde_json::from_str(val_str).map_err(|e| {
                HeliumError::Format(format!(
                    "HEC0: param value '{val_str}' at stage {stage_idx} param {param_idx} is not valid JSON: {e}"
                ))
            })?;
            spec.params.insert(key.to_string(), val);
        }

        stages.push(spec);
    }

    // After the block, read value_count as LEB128.
    let (value_count, leb_bytes) = read_leb128_u64(&bytes[block_end..]).map_err(|e| {
        HeliumError::Format(format!("HEC0: failed to read value_count LEB128: {e}"))
    })?;

    let body_offset = block_end + leb_bytes;

    Ok((Hec0Header { data_type, stages }, value_count, body_offset))
}

// ---------------------------------------------------------------------------
// LEB128 helpers (u64, unsigned only — not shared with the coder)
// ---------------------------------------------------------------------------

fn write_leb128_u64(out: &mut Vec<u8>, mut value: u64) {
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        if value == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

/// Returns `(value, bytes_consumed)`.
fn read_leb128_u64(bytes: &[u8]) -> std::result::Result<(u64, usize), String> {
    let mut result: u64 = 0;
    let mut shift: u32 = 0;
    let mut i = 0usize;
    loop {
        if i >= bytes.len() {
            return Err("unterminated LEB128 sequence".into());
        }
        let b = bytes[i];
        i += 1;
        result |= ((b & 0x7f) as u64) << shift;
        if b & 0x80 == 0 {
            return Ok((result, i));
        }
        shift += 7;
        if shift >= 64 {
            return Err("LEB128 value exceeds 64 bits".into());
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_type_discriminants_round_trip() {
        let types = [
            DataType::I8,
            DataType::I16,
            DataType::I32,
            DataType::I64,
            DataType::U8,
            DataType::U16,
            DataType::U32,
            DataType::U64,
            DataType::F32,
            DataType::F64,
            DataType::Bytes,
        ];
        for dt in types {
            let b = data_type_to_byte(dt);
            assert_eq!(byte_to_data_type(b).unwrap(), dt);
        }
    }

    #[test]
    fn discriminant_values_are_stable() {
        // These are wire-format-frozen: if any assertion here fails, the format
        // has been broken — do not change these values.
        assert_eq!(data_type_to_byte(DataType::I8), 1);
        assert_eq!(data_type_to_byte(DataType::I16), 2);
        assert_eq!(data_type_to_byte(DataType::I32), 3);
        assert_eq!(data_type_to_byte(DataType::I64), 4);
        assert_eq!(data_type_to_byte(DataType::U8), 5);
        assert_eq!(data_type_to_byte(DataType::U16), 6);
        assert_eq!(data_type_to_byte(DataType::U32), 7);
        assert_eq!(data_type_to_byte(DataType::U64), 8);
        assert_eq!(data_type_to_byte(DataType::F32), 9);
        assert_eq!(data_type_to_byte(DataType::F64), 10);
        assert_eq!(data_type_to_byte(DataType::Bytes), 11);
    }

    #[test]
    fn unknown_discriminant_errors() {
        assert!(byte_to_data_type(0).is_err());
        assert!(byte_to_data_type(12).is_err());
        assert!(byte_to_data_type(255).is_err());
    }

    #[test]
    fn header_round_trip_no_params() {
        let header = Hec0Header {
            data_type: DataType::I64,
            stages: vec![
                CoderSpec::new("delta"),
                CoderSpec::new("leb128"),
                CoderSpec::new("zstd"),
            ],
        };
        let buf = write_header(&header, 42).unwrap();

        // Magic check.
        assert_eq!(&buf[0..4], b"HEC0");

        let (parsed, count, body_off) = read_header(&buf).unwrap();
        assert_eq!(parsed, header);
        assert_eq!(count, 42);
        // Body is at body_off, nothing after it in this test.
        assert_eq!(body_off, buf.len());
    }

    #[test]
    fn header_round_trip_with_params() {
        let mut spec = CoderSpec::new("zstd");
        spec.params
            .insert("level".into(), serde_json::Value::from(6i32));

        let header = Hec0Header {
            data_type: DataType::F64,
            stages: vec![CoderSpec::new("gorilla"), spec],
        };
        let buf = write_header(&header, 1000).unwrap();
        let (parsed, count, _body_off) = read_header(&buf).unwrap();
        assert_eq!(parsed, header);
        assert_eq!(count, 1000);
    }

    #[test]
    fn header_zero_stages() {
        let header = Hec0Header {
            data_type: DataType::Bytes,
            stages: vec![],
        };
        let buf = write_header(&header, 0).unwrap();
        let (parsed, count, _) = read_header(&buf).unwrap();
        assert_eq!(parsed.data_type, DataType::Bytes);
        assert!(parsed.stages.is_empty());
        assert_eq!(count, 0);
    }

    #[test]
    fn bad_magic_errors() {
        let buf = b"FOOO\x01\x03\x00";
        let err = read_header(buf).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("magic") || msg.contains("HEC0"),
            "error should mention magic or HEC0, got: {msg}"
        );
    }

    #[test]
    fn truncated_buffer_errors() {
        // Only magic, no header_len.
        let err = read_header(b"HEC0").unwrap_err();
        assert!(
            err.to_string().contains("HEC0")
                || err.to_string().contains("truncated")
                || err.to_string().contains("header_len")
        );

        // Magic + header_len but no block.
        let err2 = read_header(b"HEC0\x05").unwrap_err();
        assert!(err2.to_string().contains("HEC0") || err2.to_string().contains("short"));
    }

    #[test]
    fn leb128_value_count_survives_large_values() {
        let header = Hec0Header {
            data_type: DataType::U32,
            stages: vec![CoderSpec::new("zstd")],
        };
        // Use a value that needs more than one LEB128 byte.
        let count = 100_000u64;
        let buf = write_header(&header, count).unwrap();
        let (_, parsed_count, _) = read_header(&buf).unwrap();
        assert_eq!(parsed_count, count);
    }
}
