use crate::core::coder::{BlockCoder, Coder, CoderKind, ColumnData, DataType};
use crate::core::error::{HeliumError, Result};

/// Zstandard. Block coder: compresses the whole input buffer at once.
pub struct Zstd {
    level: i32,
}

impl Zstd {
    /// Create a new Zstandard block compressor at the given `level` (1–22;
    /// level 3 is the zstd default).
    pub fn new(level: i32) -> Self {
        Self { level }
    }
}

impl Default for Zstd {
    fn default() -> Self {
        Self::new(3)
    }
}

impl Coder for Zstd {
    fn id(&self) -> &'static str {
        "zstd"
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

impl BlockCoder for Zstd {
    fn encode_block(&self, input: &ColumnData) -> Result<ColumnData> {
        let ColumnData::Bytes(src) = input else {
            return Err(HeliumError::RuntimeType {
                coder: self.id().into(),
                expected: DataType::Bytes,
            });
        };
        let compressed = ::zstd::stream::encode_all(src.as_slice(), self.level)?;
        Ok(ColumnData::Bytes(compressed))
    }

    fn decode_block(&self, input: &ColumnData) -> Result<ColumnData> {
        let ColumnData::Bytes(src) = input else {
            return Err(HeliumError::RuntimeType {
                coder: self.id().into(),
                expected: DataType::Bytes,
            });
        };
        let decompressed = ::zstd::stream::decode_all(src.as_slice())?;
        Ok(ColumnData::Bytes(decompressed))
    }
}
