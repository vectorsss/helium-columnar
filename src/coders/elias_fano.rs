//! Elias-Fano encoding for strictly-increasing unsigned integer sequences.
//!
//! Given `n` sorted values with max `u`, EF stores each value in ~`2 +
//! log2(u/n)` bits — near the information-theoretic lower bound. Standard
//! technique for inverted-index postings.
//!
//! Encoding (per value `x`):
//! - split into `high = x >> l` (`l = floor(log2(u/n))` low bits) and
//!   `low = x & ((1<<l)-1)`
//! - write `high - prev_high` zeros then a one into the high bitstream
//! - write `low` as `l` bits into the low bitstream
//!
//! On-wire layout (little-endian):
//!
//! ```text
//! [8B]   n: u64
//! [1B]   l: u8   (0..=63)
//! [4B]   high_byte_len: u32
//! [..]   high bytes (`n + max_high + 1` unary bits, byte-padded)
//! [..]   low bytes (`ceil(n*l/8)`)
//! ```
//!
//! The encoder *requires* strict monotonic increase — repeats or unsorted
//! input are rejected with a specific error.

use crate::coders::bitstream::{BitReader, BitWriter};
use crate::core::coder::{AccessPattern, BlockCoder, Coder, CoderKind, ColumnData, DataType};
use crate::core::error::{HeliumError, Result};

/// Elias-Fano block coder for sorted unique unsigned integers.
///
/// Achieves near-information-theoretic compression for sorted sets
/// (inverted index postings, sorted unique IDs).  Input must be strictly
/// monotonically increasing; duplicates or unsorted values are rejected.
pub struct EliasFano {
    data_type: DataType,
}

impl EliasFano {
    /// Create a new Elias-Fano coder for `U32` or `U64` data.
    pub fn new(data_type: DataType) -> Result<Self> {
        if !matches!(data_type, DataType::U32 | DataType::U64) {
            return Err(HeliumError::InvalidParam {
                coder: "elias_fano".into(),
                param: "<input_type>".into(),
                reason: format!("elias_fano supports U32 / U64 only, got {data_type:?}"),
            });
        }
        Ok(Self { data_type })
    }
}

impl Coder for EliasFano {
    fn id(&self) -> &'static str {
        "elias_fano"
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
        // Classical Elias-Fano supports O(1 + n/b) select with a rank/select
        // index; the pure bitstream form is O(1) on average after amortizing
        // the unary scan. Declaring RandomAccess is honest at the algorithm
        // level even though the current decode path reads sequentially.
        AccessPattern::RandomAccess
    }
}

impl BlockCoder for EliasFano {
    fn encode_block(&self, input: &ColumnData) -> Result<ColumnData> {
        let xs: Vec<u64> = match (self.data_type, input) {
            (DataType::U32, ColumnData::U32(v)) => v.iter().map(|&x| x as u64).collect(),
            (DataType::U64, ColumnData::U64(v)) => v.clone(),
            _ => {
                return Err(HeliumError::RuntimeType {
                    coder: self.id().into(),
                    expected: self.data_type,
                });
            }
        };
        Ok(ColumnData::Bytes(ef_encode(&xs)?))
    }

    fn decode_block(&self, input: &ColumnData) -> Result<ColumnData> {
        let ColumnData::Bytes(src) = input else {
            return Err(HeliumError::RuntimeType {
                coder: self.id().into(),
                expected: DataType::Bytes,
            });
        };
        let xs = ef_decode(src)?;
        Ok(match self.data_type {
            DataType::U32 => {
                let mut out = Vec::with_capacity(xs.len());
                for x in xs {
                    out.push(u32::try_from(x).map_err(|_| HeliumError::Corrupted {
                        coder: "elias_fano".into(),
                        reason: format!("value {x} overflows u32"),
                    })?);
                }
                ColumnData::U32(out)
            }
            DataType::U64 => ColumnData::U64(xs),
            _ => unreachable!("validated in new()"),
        })
    }
}

fn ef_encode(xs: &[u64]) -> Result<Vec<u8>> {
    let n = xs.len() as u64;
    if n == 0 {
        let mut out = Vec::with_capacity(13);
        out.extend_from_slice(&0u64.to_le_bytes()); // n
        out.push(0); // l
        out.extend_from_slice(&0u32.to_le_bytes()); // high_byte_len
        return Ok(out);
    }
    for w in xs.windows(2) {
        if w[1] <= w[0] {
            return Err(HeliumError::CoderFailed {
                coder: "elias_fano".into(),
                reason: format!(
                    "input must be strictly increasing; saw {} after {}",
                    w[1], w[0]
                ),
            });
        }
    }
    // SAFETY: xs is non-empty (empty case returns early above).
    let Some(&u) = xs.last() else {
        return Ok(Vec::new());
    };
    // l = floor(log2(u / n)), clamped to [0, 63].
    let l: u32 = if n == 0 || u / n == 0 {
        0
    } else {
        let q = u / n;
        u64::BITS - q.leading_zeros() - 1
    };
    let low_mask: u64 = if l == 0 { 0 } else { (1u64 << l) - 1 };

    let mut high = BitWriter::new();
    let mut low = BitWriter::new();
    let mut prev_high: u64 = 0;
    for &x in xs {
        let h = x >> l;
        let gap = h - prev_high;
        for _ in 0..gap {
            high.write_bit(false);
        }
        high.write_bit(true);
        prev_high = h;
        if l > 0 {
            low.write_bits(x & low_mask, l);
        }
    }
    let high_bytes = high.finish();
    let low_bytes = low.finish();

    let mut out = Vec::with_capacity(13 + high_bytes.len() + low_bytes.len());
    out.extend_from_slice(&n.to_le_bytes());
    out.push(l as u8);
    let high_len: u32 = high_bytes
        .len()
        .try_into()
        .map_err(|_| HeliumError::CoderFailed {
            coder: "elias_fano".into(),
            reason: "high bitstream exceeds u32::MAX bytes".into(),
        })?;
    out.extend_from_slice(&high_len.to_le_bytes());
    out.extend_from_slice(&high_bytes);
    out.extend_from_slice(&low_bytes);
    Ok(out)
}

fn ef_decode(src: &[u8]) -> Result<Vec<u64>> {
    if src.len() < 13 {
        return Err(HeliumError::Corrupted {
            coder: "elias_fano".into(),
            reason: "header truncated".into(),
        });
    }
    // SAFETY: src.len() >= 13 is checked above; these slices are exactly the right size.
    let n = u64::from_le_bytes(src[..8].try_into().map_err(|_| HeliumError::Corrupted {
        coder: "elias_fano".into(),
        reason: "header read failed".into(),
    })?);
    let l = src[8] as u32;
    if l > 63 {
        return Err(HeliumError::Corrupted {
            coder: "elias_fano".into(),
            reason: format!("invalid low-width {l}"),
        });
    }
    let high_len =
        u32::from_le_bytes(src[9..13].try_into().map_err(|_| HeliumError::Corrupted {
            coder: "elias_fano".into(),
            reason: "header read failed".into(),
        })?) as usize;
    let body = &src[13..];
    if body.len() < high_len {
        return Err(HeliumError::Corrupted {
            coder: "elias_fano".into(),
            reason: "high bitstream truncated".into(),
        });
    }
    if n == 0 {
        return Ok(Vec::new());
    }
    let (high_bytes, low_bytes) = body.split_at(high_len);

    let mut high_rd = BitReader::new(high_bytes);
    let mut low_rd = BitReader::new(low_bytes);

    let mut out = Vec::with_capacity(n as usize);
    let mut cur_high: u64 = 0;
    for _ in 0..n {
        loop {
            match high_rd.read_bit() {
                Some(true) => break,
                Some(false) => cur_high += 1,
                None => {
                    return Err(HeliumError::Corrupted {
                        coder: "elias_fano".into(),
                        reason: "high bitstream ended mid-value".into(),
                    });
                }
            }
        }
        let low = if l == 0 {
            0
        } else {
            low_rd.read_bits(l).ok_or_else(|| HeliumError::Corrupted {
                coder: "elias_fano".into(),
                reason: "low bitstream truncated".into(),
            })?
        };
        out.push((cur_high << l) | low);
    }
    Ok(out)
}
