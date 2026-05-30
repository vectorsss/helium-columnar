use crate::coders::numeric::WrappingArith;
use crate::core::coder::{Coder, CoderKind, ColumnData, DataType, NonBlockCoder};
use crate::core::error::{HeliumError, Result};

/// Run-length encoding with interleaved `[value, count, value, count, …]`
/// layout. Value and count share the column's physical type, so a run whose
/// length overflows that type is rejected (pick a wider type or downstream
/// dictionary if that happens).
pub struct Rle {
    data_type: DataType,
}

impl Rle {
    /// Create a new RLE coder for `data_type` (must be an integer type).
    pub fn new(data_type: DataType) -> Result<Self> {
        if !data_type.is_integer() {
            return Err(HeliumError::InvalidParam {
                coder: "rle".into(),
                param: "<input_type>".into(),
                reason: format!("rle only supports integer types, got {data_type:?}"),
            });
        }
        Ok(Self { data_type })
    }
}

impl Coder for Rle {
    fn id(&self) -> &'static str {
        "rle"
    }
    fn kind(&self) -> CoderKind {
        CoderKind::NonBlock
    }
    fn accepted_input_type(&self) -> DataType {
        self.data_type
    }
    fn produced_output_type(&self) -> DataType {
        self.data_type
    }
}

trait CountLike: WrappingArith {
    fn from_usize_checked(n: usize) -> Option<Self>;
    fn to_usize_checked(self) -> Option<usize>;
    fn is_positive(self) -> bool;
    /// Largest positive run length this type can represent in a single pair.
    /// Clamped to `usize::MAX` on narrow-usize targets.
    fn max_positive_run() -> usize;
}

macro_rules! impl_count_like {
    ($($t:ty => $max:expr),* $(,)?) => {
        $(impl CountLike for $t {
            fn from_usize_checked(n: usize) -> Option<Self> { <$t>::try_from(n).ok() }
            fn to_usize_checked(self) -> Option<usize> { usize::try_from(self).ok() }
            fn is_positive(self) -> bool { self > 0 }
            fn max_positive_run() -> usize { $max }
        })*
    };
}
impl_count_like!(
    i8 => i8::MAX as usize,
    i16 => i16::MAX as usize,
    i32 => i32::MAX as usize,
    i64 => usize::try_from(i64::MAX).unwrap_or(usize::MAX),
    u8 => u8::MAX as usize,
    u16 => u16::MAX as usize,
    u32 => u32::MAX as usize,
    u64 => usize::try_from(u64::MAX).unwrap_or(usize::MAX),
);

fn encode<T: CountLike>(xs: &[T], _coder_id: &'static str) -> Result<Vec<T>> {
    let max_run = T::max_positive_run().max(1);
    let mut out = Vec::with_capacity(xs.len());
    let mut i = 0usize;
    while i < xs.len() {
        let value = xs[i];
        let mut count: usize = 1;
        while i + 1 < xs.len() && xs[i + 1] == value {
            count += 1;
            i += 1;
        }
        // Split runs longer than max_run into multiple (value, max_run) pairs
        // plus a final (value, remainder). Decode doesn't care that they were
        // split — consecutive pairs with the same value simply concatenate.
        let mut remaining = count;
        while remaining > max_run {
            // max_run == T::max_positive_run() which is defined as the largest
            // value T can hold, so from_usize_checked cannot return None here.
            let chunk = T::from_usize_checked(max_run).ok_or_else(|| HeliumError::CoderFailed {
                coder: _coder_id.into(),
                reason: "run length overflows type (internal error)".into(),
            })?;
            out.push(value);
            out.push(chunk);
            remaining -= max_run;
        }
        // remaining <= max_run, so from_usize_checked cannot return None here.
        let last = T::from_usize_checked(remaining).ok_or_else(|| HeliumError::CoderFailed {
            coder: _coder_id.into(),
            reason: "run length overflows type (internal error)".into(),
        })?;
        out.push(value);
        out.push(last);
        i += 1;
    }
    Ok(out)
}

fn decode<T: CountLike>(xs: &[T], coder_id: &'static str) -> Result<Vec<T>> {
    if !xs.len().is_multiple_of(2) {
        return Err(HeliumError::Corrupted {
            coder: coder_id.into(),
            reason: format!("odd number of values: {}", xs.len()),
        });
    }
    let mut out = Vec::new();
    for chunk in xs.chunks_exact(2) {
        let value = chunk[0];
        let count = chunk[1];
        if !count.is_positive() {
            return Err(HeliumError::Corrupted {
                coder: coder_id.into(),
                reason: "non-positive run length".into(),
            });
        }
        let run = count
            .to_usize_checked()
            .ok_or_else(|| HeliumError::Corrupted {
                coder: coder_id.into(),
                reason: "run length does not fit in usize".into(),
            })?;
        for _ in 0..run {
            out.push(value);
        }
    }
    Ok(out)
}

impl NonBlockCoder for Rle {
    fn encode(&self, input: &ColumnData) -> Result<ColumnData> {
        Ok(match (self.data_type, input) {
            (DataType::I8, ColumnData::I8(xs)) => ColumnData::I8(encode(xs, self.id())?),
            (DataType::I16, ColumnData::I16(xs)) => ColumnData::I16(encode(xs, self.id())?),
            (DataType::I32, ColumnData::I32(xs)) => ColumnData::I32(encode(xs, self.id())?),
            (DataType::I64, ColumnData::I64(xs)) => ColumnData::I64(encode(xs, self.id())?),
            (DataType::U8, ColumnData::U8(xs)) => ColumnData::U8(encode(xs, self.id())?),
            (DataType::U16, ColumnData::U16(xs)) => ColumnData::U16(encode(xs, self.id())?),
            (DataType::U32, ColumnData::U32(xs)) => ColumnData::U32(encode(xs, self.id())?),
            (DataType::U64, ColumnData::U64(xs)) => ColumnData::U64(encode(xs, self.id())?),
            _ => {
                return Err(HeliumError::RuntimeType {
                    coder: self.id().into(),
                    expected: self.data_type,
                });
            }
        })
    }

    fn decode(&self, input: &ColumnData) -> Result<ColumnData> {
        Ok(match (self.data_type, input) {
            (DataType::I8, ColumnData::I8(xs)) => ColumnData::I8(decode(xs, self.id())?),
            (DataType::I16, ColumnData::I16(xs)) => ColumnData::I16(decode(xs, self.id())?),
            (DataType::I32, ColumnData::I32(xs)) => ColumnData::I32(decode(xs, self.id())?),
            (DataType::I64, ColumnData::I64(xs)) => ColumnData::I64(decode(xs, self.id())?),
            (DataType::U8, ColumnData::U8(xs)) => ColumnData::U8(decode(xs, self.id())?),
            (DataType::U16, ColumnData::U16(xs)) => ColumnData::U16(decode(xs, self.id())?),
            (DataType::U32, ColumnData::U32(xs)) => ColumnData::U32(decode(xs, self.id())?),
            (DataType::U64, ColumnData::U64(xs)) => ColumnData::U64(decode(xs, self.id())?),
            _ => {
                return Err(HeliumError::RuntimeType {
                    coder: self.id().into(),
                    expected: self.data_type,
                });
            }
        })
    }
}
