use crate::core::coder::{BlockCoder, Coder, CoderKind, ColumnData, DataType};
use crate::core::error::{HeliumError, Result};

/// Snappy (raw framing, length-prefixed). Block coder: compresses the
/// whole input buffer at once. Widely used as a fast default block
/// compressor — Parquet's historical default and what ClickBench ships.
pub struct Snappy;

impl Coder for Snappy {
    fn id(&self) -> &'static str {
        "snappy"
    }
    fn kind(&self) -> CoderKind {
        CoderKind::Block
    }
    fn accepted_input_type(&self) -> DataType {
        DataType::Bytes
    }
    fn produced_output_type(&self) -> DataType {
        DataType::Bytes
    }
}

impl BlockCoder for Snappy {
    fn encode_block(&self, input: &ColumnData) -> Result<ColumnData> {
        let ColumnData::Bytes(src) = input else {
            return Err(HeliumError::RuntimeType {
                coder: self.id().into(),
                expected: DataType::Bytes,
            });
        };
        let out =
            snap::raw::Encoder::new()
                .compress_vec(src)
                .map_err(|e| HeliumError::CoderFailed {
                    coder: self.id().into(),
                    reason: e.to_string(),
                })?;
        Ok(ColumnData::Bytes(out))
    }

    fn decode_block(&self, input: &ColumnData) -> Result<ColumnData> {
        let ColumnData::Bytes(src) = input else {
            return Err(HeliumError::RuntimeType {
                coder: self.id().into(),
                expected: DataType::Bytes,
            });
        };
        let out =
            snap::raw::Decoder::new()
                .decompress_vec(src)
                .map_err(|e| HeliumError::Corrupted {
                    coder: self.id().into(),
                    reason: e.to_string(),
                })?;
        Ok(ColumnData::Bytes(out))
    }
}
