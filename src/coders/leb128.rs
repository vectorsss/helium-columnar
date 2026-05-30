use crate::core::coder::{Coder, CoderKind, ColumnData, DataType, NonBlockCoder};
use crate::core::error::{HeliumError, Result};

/// LEB128 variable-length integer encoder.
///
/// - Signed types (`i8..i64`): zigzag-transformed, then LEB128.
/// - Unsigned types (`u8..u64`): LEB128 directly.
///
/// Output is always `DataType::Bytes`.
pub struct Leb128 {
    data_type: DataType,
}

impl Leb128 {
    /// Create a new LEB128 coder for `data_type` (must be an integer type).
    pub fn new(data_type: DataType) -> Result<Self> {
        if !data_type.is_integer() {
            return Err(HeliumError::InvalidParam {
                coder: "leb128".into(),
                param: "<input_type>".into(),
                reason: format!("leb128 only supports integer types, got {data_type:?}"),
            });
        }
        Ok(Self { data_type })
    }
}

impl Coder for Leb128 {
    fn id(&self) -> &'static str {
        "leb128"
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

impl NonBlockCoder for Leb128 {
    fn encode(&self, input: &ColumnData) -> Result<ColumnData> {
        let mut out = Vec::with_capacity(input.len() * 2);
        match (self.data_type, input) {
            (DataType::I8, ColumnData::I8(xs)) => {
                for &n in xs {
                    write_leb128(&mut out, zigzag_enc(n as i64));
                }
            }
            (DataType::I16, ColumnData::I16(xs)) => {
                for &n in xs {
                    write_leb128(&mut out, zigzag_enc(n as i64));
                }
            }
            (DataType::I32, ColumnData::I32(xs)) => {
                for &n in xs {
                    write_leb128(&mut out, zigzag_enc(n as i64));
                }
            }
            (DataType::I64, ColumnData::I64(xs)) => {
                for &n in xs {
                    write_leb128(&mut out, zigzag_enc(n));
                }
            }
            (DataType::U8, ColumnData::U8(xs)) => {
                for &n in xs {
                    write_leb128(&mut out, n as u64);
                }
            }
            (DataType::U16, ColumnData::U16(xs)) => {
                for &n in xs {
                    write_leb128(&mut out, n as u64);
                }
            }
            (DataType::U32, ColumnData::U32(xs)) => {
                for &n in xs {
                    write_leb128(&mut out, n as u64);
                }
            }
            (DataType::U64, ColumnData::U64(xs)) => {
                for &n in xs {
                    write_leb128(&mut out, n);
                }
            }
            _ => {
                return Err(HeliumError::RuntimeType {
                    coder: self.id().into(),
                    expected: self.data_type,
                });
            }
        }
        Ok(ColumnData::Bytes(out))
    }

    fn decode(&self, input: &ColumnData) -> Result<ColumnData> {
        let ColumnData::Bytes(bytes) = input else {
            return Err(HeliumError::RuntimeType {
                coder: self.id().into(),
                expected: DataType::Bytes,
            });
        };
        let raw = read_all_leb128(bytes, self.id())?;
        match self.data_type {
            DataType::I8 => to_signed_vec::<i8>(&raw, self.id()).map(ColumnData::I8),
            DataType::I16 => to_signed_vec::<i16>(&raw, self.id()).map(ColumnData::I16),
            DataType::I32 => to_signed_vec::<i32>(&raw, self.id()).map(ColumnData::I32),
            DataType::I64 => to_signed_vec::<i64>(&raw, self.id()).map(ColumnData::I64),
            DataType::U8 => to_unsigned_vec::<u8>(&raw, self.id()).map(ColumnData::U8),
            DataType::U16 => to_unsigned_vec::<u16>(&raw, self.id()).map(ColumnData::U16),
            DataType::U32 => to_unsigned_vec::<u32>(&raw, self.id()).map(ColumnData::U32),
            DataType::U64 => Ok(ColumnData::U64(raw)),
            _ => Err(HeliumError::RuntimeType {
                coder: self.id().into(),
                expected: self.data_type,
            }),
        }
    }
}

#[inline]
fn zigzag_enc(n: i64) -> u64 {
    ((n << 1) ^ (n >> 63)) as u64
}

#[inline]
fn zigzag_dec(u: u64) -> i64 {
    ((u >> 1) as i64) ^ -((u & 1) as i64)
}

fn write_leb128(out: &mut Vec<u8>, mut u: u64) {
    loop {
        let byte = (u & 0x7f) as u8;
        u >>= 7;
        if u == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

fn read_all_leb128(bytes: &[u8], coder_id: &'static str) -> Result<Vec<u64>> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < bytes.len() {
        let mut result: u64 = 0;
        let mut shift: u32 = 0;
        loop {
            if i >= bytes.len() {
                return Err(HeliumError::Corrupted {
                    coder: coder_id.into(),
                    reason: "unterminated leb128 sequence".into(),
                });
            }
            let b = bytes[i];
            i += 1;
            result |= ((b & 0x7f) as u64) << shift;
            if b & 0x80 == 0 {
                break;
            }
            shift += 7;
            if shift >= 64 {
                return Err(HeliumError::Corrupted {
                    coder: coder_id.into(),
                    reason: "leb128 value exceeds 64 bits".into(),
                });
            }
        }
        out.push(result);
    }
    Ok(out)
}

trait FromI64Checked: Sized {
    fn try_from_i64(v: i64) -> Option<Self>;
}

trait FromU64Checked: Sized {
    fn try_from_u64(v: u64) -> Option<Self>;
}

macro_rules! impl_from_i64 {
    ($($t:ty),*) => {
        $(impl FromI64Checked for $t {
            fn try_from_i64(v: i64) -> Option<Self> { <$t>::try_from(v).ok() }
        })*
    };
}
macro_rules! impl_from_u64 {
    ($($t:ty),*) => {
        $(impl FromU64Checked for $t {
            fn try_from_u64(v: u64) -> Option<Self> { <$t>::try_from(v).ok() }
        })*
    };
}
impl_from_i64!(i8, i16, i32, i64);
impl_from_u64!(u8, u16, u32);

fn to_signed_vec<T: FromI64Checked>(raw: &[u64], coder_id: &'static str) -> Result<Vec<T>> {
    let mut out = Vec::with_capacity(raw.len());
    for &u in raw {
        let v = zigzag_dec(u);
        out.push(T::try_from_i64(v).ok_or_else(|| HeliumError::Corrupted {
            coder: coder_id.into(),
            reason: format!("value {v} out of range for target type"),
        })?);
    }
    Ok(out)
}

fn to_unsigned_vec<T: FromU64Checked>(raw: &[u64], coder_id: &'static str) -> Result<Vec<T>> {
    let mut out = Vec::with_capacity(raw.len());
    for &u in raw {
        out.push(T::try_from_u64(u).ok_or_else(|| HeliumError::Corrupted {
            coder: coder_id.into(),
            reason: format!("value {u} out of range for target type"),
        })?);
    }
    Ok(out)
}
