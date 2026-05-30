//! Bit-packing coders — fixed and auto widths.
//!
//! Accepts any integer type (signed or unsigned). Values must be
//! non-negative at encode time; negative values are rejected with
//! `CoderFailed`. Decode preserves the original type, so compositions like
//! `deltamin(I64) → bitpack_auto(I64)` — where deltamin guarantees
//! non-negative output — work naturally.
//!
//! Output format (LSB-first packed bits):
//!
//! - `bitpack_fixed`: `[8B count u64 LE][packed bits]`
//! - `bitpack_auto`:  `[8B count u64 LE][1B width][packed bits]`

use crate::core::coder::{
    AccessPattern, BlockCoder, Coder, CoderKind, ColumnData, DataType, NonBlockCoder,
};
use crate::core::error::{HeliumError, Result};

const MAX_WIDTH: u32 = 64;

/// Fixed-width bit-packing. Width is declared in the schema and every value
/// must fit in `width` bits. Non-block.
pub struct BitpackFixed {
    data_type: DataType,
    width: u32,
}

impl BitpackFixed {
    /// Create a new fixed-width bitpacker.
    ///
    /// `data_type` must be an integer type; `width` must be ≤ the type's bit width.
    pub fn new(data_type: DataType, width: u32) -> Result<Self> {
        if !data_type.is_integer() {
            return Err(HeliumError::InvalidParam {
                coder: "bitpack_fixed".into(),
                param: "<input_type>".into(),
                reason: format!("bitpack_fixed only supports integer types, got {data_type:?}"),
            });
        }
        if width > MAX_WIDTH {
            return Err(HeliumError::CoderFailed {
                coder: "bitpack_fixed".into(),
                reason: format!("width {width} exceeds max {MAX_WIDTH}"),
            });
        }
        let max_width = (data_type
            .byte_width()
            .ok_or_else(|| HeliumError::InvalidParam {
                coder: "bitpack_fixed".into(),
                param: "<input_type>".into(),
                reason: format!("no byte width for {data_type:?}"),
            })?
            * 8) as u32;
        if width > max_width {
            return Err(HeliumError::CoderFailed {
                coder: "bitpack_fixed".into(),
                reason: format!("width {width} exceeds {data_type:?}'s {max_width} bits"),
            });
        }
        Ok(Self { data_type, width })
    }

    /// Returns the bit-packing width (number of bits per value).
    pub fn width(&self) -> u32 {
        self.width
    }
}

impl Coder for BitpackFixed {
    fn id(&self) -> &'static str {
        "bitpack_fixed"
    }
    fn kind(&self) -> CoderKind {
        CoderKind::NonBlock
    }
    fn accepted_input_type(&self) -> DataType {
        self.data_type
    }
    fn produced_output_type(&self) -> DataType {
        DataType::Bytes
    }
    fn access_pattern(&self) -> AccessPattern {
        // i-th value lives at bit offset i * width in the packed region;
        // reading it is O(1) given the 8-byte count header is skipped.
        AccessPattern::RandomAccess
    }
}

impl NonBlockCoder for BitpackFixed {
    fn encode(&self, input: &ColumnData) -> Result<ColumnData> {
        let values = to_u64_vec(self.data_type, input, self.id())?;
        let max_allowed: u64 = if self.width == 0 {
            0
        } else if self.width == 64 {
            u64::MAX
        } else {
            (1u64 << self.width) - 1
        };
        for &v in &values {
            if v > max_allowed {
                return Err(HeliumError::CoderFailed {
                    coder: self.id().into(),
                    reason: format!("value {v} exceeds {}-bit range", self.width),
                });
            }
        }
        let mut out = Vec::with_capacity(8 + packed_byte_len(values.len(), self.width));
        out.extend_from_slice(&(values.len() as u64).to_le_bytes());
        pack_into(&values, self.width, &mut out);
        Ok(ColumnData::Bytes(out))
    }

    fn decode(&self, input: &ColumnData) -> Result<ColumnData> {
        let ColumnData::Bytes(bytes) = input else {
            return Err(HeliumError::RuntimeType {
                coder: self.id().into(),
                expected: DataType::Bytes,
            });
        };
        let (count, rest) = read_count(bytes, self.id())?;
        let unpacked = unpack(rest, count, self.width, self.id())?;
        from_u64_vec(self.data_type, unpacked, self.id())
    }
}

/// Auto-width bit-packing. Scans the column for max, derives width, stores
/// width in output header. Block.
pub struct BitpackAuto {
    data_type: DataType,
}

impl BitpackAuto {
    /// Create a new auto-width bitpacker for an integer `data_type`.
    pub fn new(data_type: DataType) -> Result<Self> {
        if !data_type.is_integer() {
            return Err(HeliumError::InvalidParam {
                coder: "bitpack_auto".into(),
                param: "<input_type>".into(),
                reason: format!("bitpack_auto only supports integer types, got {data_type:?}"),
            });
        }
        Ok(Self { data_type })
    }
}

impl Coder for BitpackAuto {
    fn id(&self) -> &'static str {
        "bitpack_auto"
    }
    fn kind(&self) -> CoderKind {
        CoderKind::Block
    }
    fn accepted_input_type(&self) -> DataType {
        self.data_type
    }
    fn produced_output_type(&self) -> DataType {
        DataType::Bytes
    }
    fn access_pattern(&self) -> AccessPattern {
        // Width is stored in the 9-byte header; after reading it once,
        // i-th value is O(1) at bit offset i * width.
        AccessPattern::RandomAccess
    }
}

impl BlockCoder for BitpackAuto {
    fn encode_block(&self, input: &ColumnData) -> Result<ColumnData> {
        let values = to_u64_vec(self.data_type, input, self.id())?;
        let max = values.iter().copied().max().unwrap_or(0);
        let width = if max == 0 {
            0
        } else {
            u64::BITS - max.leading_zeros()
        };
        let mut out = Vec::with_capacity(9 + packed_byte_len(values.len(), width));
        out.extend_from_slice(&(values.len() as u64).to_le_bytes());
        out.push(width as u8);
        pack_into(&values, width, &mut out);
        Ok(ColumnData::Bytes(out))
    }

    fn decode_block(&self, input: &ColumnData) -> Result<ColumnData> {
        let ColumnData::Bytes(bytes) = input else {
            return Err(HeliumError::RuntimeType {
                coder: self.id().into(),
                expected: DataType::Bytes,
            });
        };
        let (count, rest) = read_count(bytes, self.id())?;
        if rest.is_empty() {
            return Err(HeliumError::Corrupted {
                coder: self.id().into(),
                reason: "missing width byte".into(),
            });
        }
        let width = rest[0] as u32;
        if width > MAX_WIDTH {
            return Err(HeliumError::Corrupted {
                coder: self.id().into(),
                reason: format!("width {width} exceeds max {MAX_WIDTH}"),
            });
        }
        let unpacked = unpack(&rest[1..], count, width, self.id())?;
        from_u64_vec(self.data_type, unpacked, self.id())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn check_non_negative<T: Copy + PartialOrd + Default + std::fmt::Display>(
    xs: &[T],
    coder_id: &'static str,
) -> Result<()> {
    if let Some(&neg) = xs.iter().find(|&&v| v < T::default()) {
        return Err(HeliumError::CoderFailed {
            coder: coder_id.into(),
            reason: format!("negative input value {neg} not supported"),
        });
    }
    Ok(())
}

fn to_u64_vec(dt: DataType, input: &ColumnData, coder_id: &'static str) -> Result<Vec<u64>> {
    Ok(match (dt, input) {
        (DataType::I8, ColumnData::I8(xs)) => {
            check_non_negative(xs, coder_id)?;
            xs.iter().map(|&x| x as u64).collect()
        }
        (DataType::I16, ColumnData::I16(xs)) => {
            check_non_negative(xs, coder_id)?;
            xs.iter().map(|&x| x as u64).collect()
        }
        (DataType::I32, ColumnData::I32(xs)) => {
            check_non_negative(xs, coder_id)?;
            xs.iter().map(|&x| x as u64).collect()
        }
        (DataType::I64, ColumnData::I64(xs)) => {
            check_non_negative(xs, coder_id)?;
            xs.iter().map(|&x| x as u64).collect()
        }
        (DataType::U8, ColumnData::U8(xs)) => xs.iter().map(|&x| x as u64).collect(),
        (DataType::U16, ColumnData::U16(xs)) => xs.iter().map(|&x| x as u64).collect(),
        (DataType::U32, ColumnData::U32(xs)) => xs.iter().map(|&x| x as u64).collect(),
        (DataType::U64, ColumnData::U64(xs)) => xs.clone(),
        _ => {
            return Err(HeliumError::RuntimeType {
                coder: coder_id.into(),
                expected: dt,
            });
        }
    })
}

fn from_u64_vec(dt: DataType, xs: Vec<u64>, coder_id: &'static str) -> Result<ColumnData> {
    fn to<T>(xs: Vec<u64>, coder_id: &'static str, tname: &'static str) -> Result<Vec<T>>
    where
        T: TryFrom<u64>,
    {
        let mut out = Vec::with_capacity(xs.len());
        for v in xs {
            out.push(T::try_from(v).map_err(|_| HeliumError::Corrupted {
                coder: coder_id.into(),
                reason: format!("value {v} overflows {tname}"),
            })?);
        }
        Ok(out)
    }
    Ok(match dt {
        DataType::I8 => ColumnData::I8(to::<i8>(xs, coder_id, "i8")?),
        DataType::I16 => ColumnData::I16(to::<i16>(xs, coder_id, "i16")?),
        DataType::I32 => ColumnData::I32(to::<i32>(xs, coder_id, "i32")?),
        DataType::I64 => ColumnData::I64(to::<i64>(xs, coder_id, "i64")?),
        DataType::U8 => ColumnData::U8(to::<u8>(xs, coder_id, "u8")?),
        DataType::U16 => ColumnData::U16(to::<u16>(xs, coder_id, "u16")?),
        DataType::U32 => ColumnData::U32(to::<u32>(xs, coder_id, "u32")?),
        DataType::U64 => ColumnData::U64(xs),
        _ => {
            return Err(HeliumError::RuntimeType {
                coder: coder_id.into(),
                expected: dt,
            });
        }
    })
}

fn packed_byte_len(count: usize, width: u32) -> usize {
    (count * width as usize).div_ceil(8)
}

fn pack_into(xs: &[u64], width: u32, out: &mut Vec<u8>) {
    if width == 0 || xs.is_empty() {
        return;
    }
    let start = out.len();
    out.resize(start + packed_byte_len(xs.len(), width), 0);
    let mut bit_pos = 0usize;
    for &v in xs {
        for b in 0..width {
            if (v >> b) & 1 != 0 {
                out[start + bit_pos / 8] |= 1 << (bit_pos % 8);
            }
            bit_pos += 1;
        }
    }
}

fn read_count<'a>(bytes: &'a [u8], coder_id: &'static str) -> Result<(usize, &'a [u8])> {
    if bytes.len() < 8 {
        return Err(HeliumError::Corrupted {
            coder: coder_id.into(),
            reason: format!("header truncated: got {} bytes, need 8", bytes.len()),
        });
    }
    // SAFETY: bytes.len() >= 8 is checked above; bytes[..8] is exactly 8 bytes.
    let count = u64::from_le_bytes(bytes[..8].try_into().map_err(|_| HeliumError::Corrupted {
        coder: coder_id.into(),
        reason: "header read failed".into(),
    })?) as usize;
    Ok((count, &bytes[8..]))
}

fn unpack(bytes: &[u8], count: usize, width: u32, coder_id: &'static str) -> Result<Vec<u64>> {
    if width == 0 {
        return Ok(vec![0; count]);
    }
    let needed = packed_byte_len(count, width);
    if bytes.len() < needed {
        return Err(HeliumError::Corrupted {
            coder: coder_id.into(),
            reason: format!("expected {needed} packed bytes, got {}", bytes.len()),
        });
    }
    let mut out = Vec::with_capacity(count);
    let mut bit_pos = 0usize;
    for _ in 0..count {
        let mut v: u64 = 0;
        for b in 0..width {
            let byte = bytes[bit_pos / 8];
            let bit = (byte >> (bit_pos % 8)) & 1;
            v |= (bit as u64) << b;
            bit_pos += 1;
        }
        out.push(v);
    }
    Ok(out)
}
