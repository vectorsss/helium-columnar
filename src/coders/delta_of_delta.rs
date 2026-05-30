//! Second-difference encoding: `out[i] = x[i] - 2*x[i-1] + x[i-2]`.
//!
//! For uniformly-sampled sequences the output is zero past the second element
//! — ideal as a shaping pass before a varint + compression tail.

use crate::coders::numeric::{WrappingArith, int_dispatch};
use crate::core::coder::{Coder, CoderKind, ColumnData, DataType, NonBlockCoder};
use crate::core::error::{HeliumError, Result};

/// Second-difference coder: `out[i] = x[i] - 2*x[i-1] + x[i-2]`.
///
/// For uniformly-spaced sequences the output collapses to near-zero after the
/// first two elements — pair with `leb128 → zstd` for best results.
pub struct DeltaOfDelta {
    data_type: DataType,
}

impl DeltaOfDelta {
    /// Create a new delta-of-delta coder for `data_type` (must be an integer type).
    pub fn new(data_type: DataType) -> Result<Self> {
        if !data_type.is_integer() {
            return Err(HeliumError::InvalidParam {
                coder: "delta_of_delta".into(),
                param: "<input_type>".into(),
                reason: format!("delta_of_delta only supports integer types, got {data_type:?}"),
            });
        }
        Ok(Self { data_type })
    }
}

impl Coder for DeltaOfDelta {
    fn id(&self) -> &'static str {
        "delta_of_delta"
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

fn encode<T: WrappingArith>(xs: &[T]) -> Vec<T> {
    let mut out = Vec::with_capacity(xs.len());
    let mut prev_x = T::zero();
    let mut prev_dx = T::zero();
    for &x in xs {
        let dx = x.wrap_sub(prev_x);
        let dod = dx.wrap_sub(prev_dx);
        out.push(dod);
        prev_x = x;
        prev_dx = dx;
    }
    out
}

fn decode<T: WrappingArith>(xs: &[T]) -> Vec<T> {
    let mut out = Vec::with_capacity(xs.len());
    let mut prev_x = T::zero();
    let mut prev_dx = T::zero();
    for &dod in xs {
        let dx = dod.wrap_add(prev_dx);
        let x = prev_x.wrap_add(dx);
        out.push(x);
        prev_x = x;
        prev_dx = dx;
    }
    out
}

impl NonBlockCoder for DeltaOfDelta {
    fn encode(&self, input: &ColumnData) -> Result<ColumnData> {
        int_dispatch!(self.id(), self.data_type, input, encode)
    }

    fn decode(&self, input: &ColumnData) -> Result<ColumnData> {
        int_dispatch!(self.id(), self.data_type, input, decode)
    }
}
