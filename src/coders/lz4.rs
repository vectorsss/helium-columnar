use crate::core::coder::{BlockCoder, Coder, CoderKind, ColumnData, DataType};
use crate::core::error::{HeliumError, Result};

/// LZ4 via `lz4_flex`. Uses the "size-prepended" frame so the decoded size is
/// carried inline — no external row/byte count needed.
///
/// Block because LZ4, like any LZ77-family compressor, needs the whole buffer
/// to find repeated substrings.
pub struct Lz4;

impl Coder for Lz4 {
    fn id(&self) -> &'static str {
        "lz4"
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

impl BlockCoder for Lz4 {
    fn encode_block(&self, input: &ColumnData) -> Result<ColumnData> {
        let ColumnData::Bytes(src) = input else {
            return Err(HeliumError::RuntimeType {
                coder: self.id().into(),
                expected: DataType::Bytes,
            });
        };
        let out = lz4_flex::compress_prepend_size(src);
        Ok(ColumnData::Bytes(out))
    }

    fn decode_block(&self, input: &ColumnData) -> Result<ColumnData> {
        let ColumnData::Bytes(src) = input else {
            return Err(HeliumError::RuntimeType {
                coder: self.id().into(),
                expected: DataType::Bytes,
            });
        };
        let out = lz4_flex::decompress_size_prepended(src).map_err(|e| HeliumError::Corrupted {
            coder: self.id().into(),
            reason: e.to_string(),
        })?;
        Ok(ColumnData::Bytes(out))
    }
}
