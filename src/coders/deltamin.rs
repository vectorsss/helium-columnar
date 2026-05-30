use crate::coders::numeric::WrappingArith;
use crate::core::coder::{BlockCoder, Coder, CoderKind, ColumnData, DataType};
use crate::core::error::{HeliumError, Result};

/// Frame-of-reference: `out[i] = in[i] - min(in)`. The min is prepended so
/// decode is self-contained.
pub struct DeltaMin {
    data_type: DataType,
}

impl DeltaMin {
    /// Create a new frame-of-reference coder for `data_type` (must be an integer type).
    pub fn new(data_type: DataType) -> Result<Self> {
        if !data_type.is_integer() {
            return Err(HeliumError::InvalidParam {
                coder: "deltamin".into(),
                param: "<input_type>".into(),
                reason: format!("deltamin only supports integer types, got {data_type:?}"),
            });
        }
        Ok(Self { data_type })
    }
}

impl Coder for DeltaMin {
    fn id(&self) -> &'static str {
        "deltamin"
    }
    fn kind(&self) -> CoderKind {
        CoderKind::Block
    }
    fn accepted_input_type(&self) -> DataType {
        self.data_type
    }
    fn produced_output_type(&self) -> DataType {
        self.data_type
    }
}

fn encode<T: WrappingArith + Ord>(xs: &[T]) -> Vec<T> {
    if xs.is_empty() {
        return Vec::new();
    }
    // SAFETY: xs is non-empty (checked at top of function).
    let Some(&min) = xs.iter().min() else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(xs.len() + 1);
    out.push(min);
    for &x in xs {
        out.push(x.wrap_sub(min));
    }
    out
}

fn decode<T: WrappingArith>(xs: &[T]) -> Vec<T> {
    if xs.is_empty() {
        return Vec::new();
    }
    let min = xs[0];
    xs[1..].iter().map(|&d| min.wrap_add(d)).collect()
}

impl BlockCoder for DeltaMin {
    fn encode_block(&self, input: &ColumnData) -> Result<ColumnData> {
        Ok(match (self.data_type, input) {
            (DataType::I8, ColumnData::I8(xs)) => ColumnData::I8(encode(xs)),
            (DataType::I16, ColumnData::I16(xs)) => ColumnData::I16(encode(xs)),
            (DataType::I32, ColumnData::I32(xs)) => ColumnData::I32(encode(xs)),
            (DataType::I64, ColumnData::I64(xs)) => ColumnData::I64(encode(xs)),
            (DataType::U8, ColumnData::U8(xs)) => ColumnData::U8(encode(xs)),
            (DataType::U16, ColumnData::U16(xs)) => ColumnData::U16(encode(xs)),
            (DataType::U32, ColumnData::U32(xs)) => ColumnData::U32(encode(xs)),
            (DataType::U64, ColumnData::U64(xs)) => ColumnData::U64(encode(xs)),
            _ => {
                return Err(HeliumError::RuntimeType {
                    coder: self.id().into(),
                    expected: self.data_type,
                });
            }
        })
    }

    fn decode_block(&self, input: &ColumnData) -> Result<ColumnData> {
        Ok(match (self.data_type, input) {
            (DataType::I8, ColumnData::I8(xs)) => ColumnData::I8(decode(xs)),
            (DataType::I16, ColumnData::I16(xs)) => ColumnData::I16(decode(xs)),
            (DataType::I32, ColumnData::I32(xs)) => ColumnData::I32(decode(xs)),
            (DataType::I64, ColumnData::I64(xs)) => ColumnData::I64(decode(xs)),
            (DataType::U8, ColumnData::U8(xs)) => ColumnData::U8(decode(xs)),
            (DataType::U16, ColumnData::U16(xs)) => ColumnData::U16(decode(xs)),
            (DataType::U32, ColumnData::U32(xs)) => ColumnData::U32(decode(xs)),
            (DataType::U64, ColumnData::U64(xs)) => ColumnData::U64(decode(xs)),
            _ => {
                return Err(HeliumError::RuntimeType {
                    coder: self.id().into(),
                    expected: self.data_type,
                });
            }
        })
    }
}
