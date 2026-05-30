//! Gorilla XOR float compression (Facebook TSDB paper, 2015).
//!
//! For each float, XOR with the previous value. If zero, emit `0`. Otherwise
//! emit `1` + either a reuse-previous-block header (`0` + meaningful bits) or
//! a new-block header (`1` + leading-zeros + meaningful-count + meaningful
//! bits). Small floats that drift slowly compress to a handful of bits per
//! value.
//!
//! This is a non-block coder — only the previous value's bit pattern and the
//! previous meaningful-bit window are kept as state.

use crate::coders::bitstream::{BitReader, BitWriter};
use crate::core::coder::{Coder, CoderKind, ColumnData, DataType, NonBlockCoder};
use crate::core::error::{HeliumError, Result};

/// Gorilla XOR encoder for `f32` / `f64` columns.
pub struct GorillaXor {
    data_type: DataType,
}

impl GorillaXor {
    /// Create a new Gorilla XOR coder for `F32` or `F64` data.
    pub fn new(data_type: DataType) -> Result<Self> {
        if !data_type.is_float() {
            return Err(HeliumError::InvalidParam {
                coder: "gorilla".into(),
                param: "<input_type>".into(),
                reason: format!("gorilla only supports float types, got {data_type:?}"),
            });
        }
        Ok(Self { data_type })
    }
}

impl Coder for GorillaXor {
    fn id(&self) -> &'static str {
        "gorilla"
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
}

impl NonBlockCoder for GorillaXor {
    fn encode(&self, input: &ColumnData) -> Result<ColumnData> {
        Ok(ColumnData::Bytes(match (self.data_type, input) {
            (DataType::F32, ColumnData::F32(xs)) => encode_f32(xs),
            (DataType::F64, ColumnData::F64(xs)) => encode_f64(xs),
            _ => {
                return Err(HeliumError::RuntimeType {
                    coder: self.id().into(),
                    expected: self.data_type,
                });
            }
        }))
    }

    fn decode(&self, input: &ColumnData) -> Result<ColumnData> {
        let ColumnData::Bytes(src) = input else {
            return Err(HeliumError::RuntimeType {
                coder: self.id().into(),
                expected: DataType::Bytes,
            });
        };
        Ok(match self.data_type {
            DataType::F32 => ColumnData::F32(decode_f32(src)?),
            DataType::F64 => ColumnData::F64(decode_f64(src)?),
            _ => unreachable!("validated in new()"),
        })
    }
}

// ---------------------------------------------------------------------------
// f64 codec — 6-bit leading-zero field, 6-bit meaningful-count-minus-1 field.
// ---------------------------------------------------------------------------

const F64_TOTAL_BITS: u32 = 64;
const F64_LZ_FIELD_BITS: u32 = 6;
const F64_MF_FIELD_BITS: u32 = 6;

const F32_TOTAL_BITS: u32 = 32;
const F32_LZ_FIELD_BITS: u32 = 5;
const F32_MF_FIELD_BITS: u32 = 5;

fn encode_f64(xs: &[f64]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(xs.len() as u64).to_le_bytes());
    if xs.is_empty() {
        return out;
    }
    let mut bw = BitWriter::new();
    let mut prev_bits = xs[0].to_bits();
    bw.write_bits(prev_bits, F64_TOTAL_BITS);

    let mut prev_lz: u32 = F64_TOTAL_BITS + 1; // sentinel = no prev block
    let mut prev_tz: u32 = 0;

    for &v in &xs[1..] {
        let cur_bits = v.to_bits();
        let xor = prev_bits ^ cur_bits;
        if xor == 0 {
            bw.write_bit(false);
        } else {
            bw.write_bit(true);
            let lz = xor.leading_zeros();
            let tz = xor.trailing_zeros();

            if prev_lz <= F64_TOTAL_BITS && lz >= prev_lz && tz >= prev_tz {
                bw.write_bit(false);
                let mf = F64_TOTAL_BITS - prev_lz - prev_tz;
                bw.write_bits(xor >> prev_tz, mf);
            } else {
                bw.write_bit(true);
                bw.write_bits(lz as u64, F64_LZ_FIELD_BITS);
                let mf = F64_TOTAL_BITS - lz - tz;
                bw.write_bits((mf - 1) as u64, F64_MF_FIELD_BITS);
                bw.write_bits(xor >> tz, mf);
                prev_lz = lz;
                prev_tz = tz;
            }
        }
        prev_bits = cur_bits;
    }
    out.extend_from_slice(&bw.finish());
    out
}

fn decode_f64(bytes: &[u8]) -> Result<Vec<f64>> {
    let (count, rest) = read_count(bytes, "gorilla")?;
    if count == 0 {
        return Ok(Vec::new());
    }
    let mut br = BitReader::new(rest);
    let mut prev_bits = br
        .read_bits(F64_TOTAL_BITS)
        .ok_or_else(|| truncated("gorilla"))?;
    let mut out = Vec::with_capacity(count);
    out.push(f64::from_bits(prev_bits));

    let mut prev_lz: u32 = 0;
    let mut prev_tz: u32 = 0;
    let mut has_prev_block = false;

    for _ in 1..count {
        let bit0 = br.read_bit().ok_or_else(|| truncated("gorilla"))?;
        let xor: u64 = if !bit0 {
            0
        } else {
            let bit1 = br.read_bit().ok_or_else(|| truncated("gorilla"))?;
            if !bit1 {
                if !has_prev_block {
                    return Err(HeliumError::Corrupted {
                        coder: "gorilla".into(),
                        reason: "reuse-previous-block header before any block defined".into(),
                    });
                }
                let mf = F64_TOTAL_BITS - prev_lz - prev_tz;
                let bits = br.read_bits(mf).ok_or_else(|| truncated("gorilla"))?;
                bits << prev_tz
            } else {
                let lz = br
                    .read_bits(F64_LZ_FIELD_BITS)
                    .ok_or_else(|| truncated("gorilla"))? as u32;
                let mf_minus_1 = br
                    .read_bits(F64_MF_FIELD_BITS)
                    .ok_or_else(|| truncated("gorilla"))? as u32;
                let mf = mf_minus_1 + 1;
                if lz + mf > F64_TOTAL_BITS {
                    return Err(HeliumError::Corrupted {
                        coder: "gorilla".into(),
                        reason: format!("invalid block geometry lz={lz} mf={mf}"),
                    });
                }
                let bits = br.read_bits(mf).ok_or_else(|| truncated("gorilla"))?;
                prev_lz = lz;
                prev_tz = F64_TOTAL_BITS - lz - mf;
                has_prev_block = true;
                bits << prev_tz
            }
        };
        prev_bits ^= xor;
        out.push(f64::from_bits(prev_bits));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// f32 codec — same algorithm with 32-bit values and 5-bit header fields.
// ---------------------------------------------------------------------------

fn encode_f32(xs: &[f32]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(xs.len() as u64).to_le_bytes());
    if xs.is_empty() {
        return out;
    }
    let mut bw = BitWriter::new();
    let mut prev_bits = xs[0].to_bits();
    bw.write_bits(prev_bits as u64, F32_TOTAL_BITS);

    let mut prev_lz: u32 = F32_TOTAL_BITS + 1;
    let mut prev_tz: u32 = 0;

    for &v in &xs[1..] {
        let cur_bits = v.to_bits();
        let xor = prev_bits ^ cur_bits;
        if xor == 0 {
            bw.write_bit(false);
        } else {
            bw.write_bit(true);
            let lz = xor.leading_zeros();
            let tz = xor.trailing_zeros();

            if prev_lz <= F32_TOTAL_BITS && lz >= prev_lz && tz >= prev_tz {
                bw.write_bit(false);
                let mf = F32_TOTAL_BITS - prev_lz - prev_tz;
                bw.write_bits((xor >> prev_tz) as u64, mf);
            } else {
                bw.write_bit(true);
                bw.write_bits(lz as u64, F32_LZ_FIELD_BITS);
                let mf = F32_TOTAL_BITS - lz - tz;
                bw.write_bits((mf - 1) as u64, F32_MF_FIELD_BITS);
                bw.write_bits((xor >> tz) as u64, mf);
                prev_lz = lz;
                prev_tz = tz;
            }
        }
        prev_bits = cur_bits;
    }
    out.extend_from_slice(&bw.finish());
    out
}

fn decode_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    let (count, rest) = read_count(bytes, "gorilla")?;
    if count == 0 {
        return Ok(Vec::new());
    }
    let mut br = BitReader::new(rest);
    let mut prev_bits = br
        .read_bits(F32_TOTAL_BITS)
        .ok_or_else(|| truncated("gorilla"))? as u32;
    let mut out = Vec::with_capacity(count);
    out.push(f32::from_bits(prev_bits));

    let mut prev_lz: u32 = 0;
    let mut prev_tz: u32 = 0;
    let mut has_prev_block = false;

    for _ in 1..count {
        let bit0 = br.read_bit().ok_or_else(|| truncated("gorilla"))?;
        let xor: u32 = if !bit0 {
            0
        } else {
            let bit1 = br.read_bit().ok_or_else(|| truncated("gorilla"))?;
            if !bit1 {
                if !has_prev_block {
                    return Err(HeliumError::Corrupted {
                        coder: "gorilla".into(),
                        reason: "reuse-previous-block header before any block defined".into(),
                    });
                }
                let mf = F32_TOTAL_BITS - prev_lz - prev_tz;
                let bits = br.read_bits(mf).ok_or_else(|| truncated("gorilla"))? as u32;
                bits << prev_tz
            } else {
                let lz = br
                    .read_bits(F32_LZ_FIELD_BITS)
                    .ok_or_else(|| truncated("gorilla"))? as u32;
                let mf_minus_1 = br
                    .read_bits(F32_MF_FIELD_BITS)
                    .ok_or_else(|| truncated("gorilla"))? as u32;
                let mf = mf_minus_1 + 1;
                if lz + mf > F32_TOTAL_BITS {
                    return Err(HeliumError::Corrupted {
                        coder: "gorilla".into(),
                        reason: format!("invalid block geometry lz={lz} mf={mf}"),
                    });
                }
                let bits = br.read_bits(mf).ok_or_else(|| truncated("gorilla"))? as u32;
                prev_lz = lz;
                prev_tz = F32_TOTAL_BITS - lz - mf;
                has_prev_block = true;
                bits << prev_tz
            }
        };
        prev_bits ^= xor;
        out.push(f32::from_bits(prev_bits));
    }
    Ok(out)
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

fn truncated(coder_id: &'static str) -> HeliumError {
    HeliumError::Corrupted {
        coder: coder_id.into(),
        reason: "bitstream truncated".into(),
    }
}
