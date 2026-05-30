use crate::coders::numeric::{delta_decode, delta_encode, int_dispatch};
use crate::core::coder::{Coder, CoderKind, ColumnData, DataType, NonBlockCoder};
use crate::core::error::{HeliumError, Result};

/// `out[i] = in[i] - in[i-1]` with `in[-1] := 0`.  Works on any integer
/// primitive (signed or unsigned); arithmetic wraps on overflow so the
/// round-trip is always exact.
pub struct Delta {
    data_type: DataType,
}

impl Delta {
    /// Create a new delta coder for `data_type` (must be an integer type).
    pub fn new(data_type: DataType) -> Result<Self> {
        if !data_type.is_integer() {
            return Err(HeliumError::InvalidParam {
                coder: "delta".into(),
                param: "<input_type>".into(),
                reason: format!("delta only supports integer types, got {data_type:?}"),
            });
        }
        Ok(Self { data_type })
    }
}

impl Coder for Delta {
    fn id(&self) -> &'static str {
        "delta"
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

impl NonBlockCoder for Delta {
    fn encode(&self, input: &ColumnData) -> Result<ColumnData> {
        int_dispatch!(self.id(), self.data_type, input, delta_encode)
    }

    fn decode(&self, input: &ColumnData) -> Result<ColumnData> {
        int_dispatch!(self.id(), self.data_type, input, delta_decode)
    }
}
