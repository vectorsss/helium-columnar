//! Default pipelines for the self-describing codec API.
//!
//! [`default_pipeline_for`] returns the same defaults the Schema-level encoder
//! uses (from `src/schema/encodings.rs`), keeping both code paths in sync.
//!
//! | DataType          | Pipeline              |
//! |-------------------|-----------------------|
//! | I8 / I16          | leb128 → zstd         |
//! | I32 / I64         | delta → leb128 → zstd |
//! | U8 / U16          | leb128 → zstd         |
//! | U32 / U64         | delta → leb128 → zstd |
//! | F32 / F64         | gorilla → zstd        |
//! | Bytes             | zstd                  |

use crate::core::coder::DataType;
use crate::core::error::Result;
use crate::core::pipeline::Pipeline;
use crate::core::registry::{CoderRegistry, CoderSpec};

/// Build the default compression pipeline for `data_type`.
///
/// Uses the same policy as `helium::schema::encodings::default_encodings` so
/// that the simple API and the schema-driven path compress identically.
pub fn default_pipeline_for(data_type: DataType) -> Result<Pipeline> {
    let specs = default_specs_for(data_type);
    build_pipeline(data_type, &specs)
}

/// Return the `CoderSpec` list for the default pipeline for `data_type`.
/// Exposed so callers that need the spec list can reuse it without rebuilding.
pub(crate) fn default_specs_for(data_type: DataType) -> Vec<CoderSpec> {
    match data_type {
        // Signed integers with differencing to amplify LEB128 gains.
        DataType::I32 | DataType::I64 => vec![
            CoderSpec::new("delta"),
            CoderSpec::new("leb128"),
            CoderSpec::new("zstd"),
        ],
        // Small signed types — delta of i8/i16 wraps poorly; leb128 + zstd is safer.
        DataType::I8 | DataType::I16 => {
            vec![CoderSpec::new("leb128"), CoderSpec::new("zstd")]
        }
        // Unsigned — U32/U64 benefit from delta (monotone counters, timestamps).
        DataType::U32 | DataType::U64 => vec![
            CoderSpec::new("delta"),
            CoderSpec::new("leb128"),
            CoderSpec::new("zstd"),
        ],
        // U8/U16 — small, no delta (same reasoning as U8 in encodings.rs).
        DataType::U8 | DataType::U16 => {
            vec![CoderSpec::new("leb128"), CoderSpec::new("zstd")]
        }
        // Floats: Gorilla XOR-encodes small drifts, then zstd.
        DataType::F32 | DataType::F64 => {
            vec![CoderSpec::new("gorilla"), CoderSpec::new("zstd")]
        }
        // Raw bytes: just zstd.
        DataType::Bytes => vec![CoderSpec::new("zstd")],
    }
}

/// Construct a `Pipeline` from a list of `CoderSpec` values.
pub(crate) fn build_pipeline(input_type: DataType, specs: &[CoderSpec]) -> Result<Pipeline> {
    let reg = CoderRegistry::with_builtins();
    let mut stages = Vec::with_capacity(specs.len());
    let mut current = input_type;
    for spec in specs {
        let stage = reg.build(spec, current)?;
        current = stage.produced_output_type();
        stages.push(stage);
    }
    Pipeline::new(input_type, stages)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::coder::{ColumnData, DataType};

    #[test]
    fn default_pipeline_for_all_types() {
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
            let pipeline = default_pipeline_for(dt);
            assert!(
                pipeline.is_ok(),
                "default_pipeline_for({dt:?}) failed: {pipeline:?}"
            );
        }
    }

    #[test]
    fn i64_pipeline_round_trips() {
        let pipeline = default_pipeline_for(DataType::I64).unwrap();
        let values: Vec<i64> = (0..100).map(|i| i * 1000).collect();
        let data = ColumnData::I64(values.clone());
        let encoded = pipeline.encode(data).unwrap();
        let decoded = pipeline.decode(encoded).unwrap();
        assert_eq!(decoded, ColumnData::I64(values));
    }

    #[test]
    fn bytes_pipeline_round_trips() {
        let pipeline = default_pipeline_for(DataType::Bytes).unwrap();
        let payload = b"hello world this is a test payload".to_vec();
        let data = ColumnData::Bytes(payload.clone());
        let encoded = pipeline.encode(data).unwrap();
        let decoded = pipeline.decode(encoded).unwrap();
        assert_eq!(decoded, ColumnData::Bytes(payload));
    }

    #[test]
    fn f64_pipeline_terminates_in_bytes() {
        let pipeline = default_pipeline_for(DataType::F64).unwrap();
        assert_eq!(pipeline.output_type(), DataType::Bytes);
    }
}
