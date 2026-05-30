use pco::ChunkConfig;
use pco::standalone::{simple_compress, simple_decompress};

use crate::core::coder::{BlockCoder, Coder, CoderKind, ColumnData, DataType};
use crate::core::error::{HeliumError, Result};

/// pcodec / pco. Typed-numeric block compressor for the types pco supports:
/// `i32`, `i64`, `u32`, `u64`, `f32`, `f64`.
pub struct Pcodec {
    data_type: DataType,
    config: ChunkConfig,
}

impl Pcodec {
    /// Create a new pcodec block compressor.
    ///
    /// `data_type` must be one of `I32`, `I64`, `U32`, `U64`, `F32`, `F64`.
    /// `level` is the pco compression level (0–12); `None` uses the pco default.
    pub fn new(data_type: DataType, level: Option<usize>) -> Result<Self> {
        if !matches!(
            data_type,
            DataType::I32
                | DataType::I64
                | DataType::U32
                | DataType::U64
                | DataType::F32
                | DataType::F64
        ) {
            return Err(HeliumError::InvalidParam {
                coder: "pcodec".into(),
                param: "<input_type>".into(),
                reason: format!("pcodec supports i32/i64/u32/u64/f32/f64, got {data_type:?}"),
            });
        }
        let config = match level {
            Some(l) => ChunkConfig::default().with_compression_level(l),
            None => ChunkConfig::default(),
        };
        Ok(Self { data_type, config })
    }
}

impl Coder for Pcodec {
    fn id(&self) -> &'static str {
        "pcodec"
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
}

fn fail(e: pco::errors::PcoError) -> HeliumError {
    HeliumError::CoderFailed {
        coder: "pcodec".into(),
        reason: e.to_string(),
    }
}

fn corrupt(e: pco::errors::PcoError) -> HeliumError {
    HeliumError::Corrupted {
        coder: "pcodec".into(),
        reason: e.to_string(),
    }
}

impl BlockCoder for Pcodec {
    fn encode_block(&self, input: &ColumnData) -> Result<ColumnData> {
        let bytes = match (self.data_type, input) {
            (DataType::I32, ColumnData::I32(xs)) => {
                simple_compress::<i32>(xs, &self.config).map_err(fail)?
            }
            (DataType::I64, ColumnData::I64(xs)) => {
                simple_compress::<i64>(xs, &self.config).map_err(fail)?
            }
            (DataType::U32, ColumnData::U32(xs)) => {
                simple_compress::<u32>(xs, &self.config).map_err(fail)?
            }
            (DataType::U64, ColumnData::U64(xs)) => {
                simple_compress::<u64>(xs, &self.config).map_err(fail)?
            }
            (DataType::F32, ColumnData::F32(xs)) => {
                simple_compress::<f32>(xs, &self.config).map_err(fail)?
            }
            (DataType::F64, ColumnData::F64(xs)) => {
                simple_compress::<f64>(xs, &self.config).map_err(fail)?
            }
            _ => {
                return Err(HeliumError::RuntimeType {
                    coder: self.id().into(),
                    expected: self.data_type,
                });
            }
        };
        Ok(ColumnData::Bytes(bytes))
    }

    fn decode_block(&self, input: &ColumnData) -> Result<ColumnData> {
        let ColumnData::Bytes(src) = input else {
            return Err(HeliumError::RuntimeType {
                coder: self.id().into(),
                expected: DataType::Bytes,
            });
        };
        Ok(match self.data_type {
            DataType::I32 => ColumnData::I32(simple_decompress::<i32>(src).map_err(corrupt)?),
            DataType::I64 => ColumnData::I64(simple_decompress::<i64>(src).map_err(corrupt)?),
            DataType::U32 => ColumnData::U32(simple_decompress::<u32>(src).map_err(corrupt)?),
            DataType::U64 => ColumnData::U64(simple_decompress::<u64>(src).map_err(corrupt)?),
            DataType::F32 => ColumnData::F32(simple_decompress::<f32>(src).map_err(corrupt)?),
            DataType::F64 => ColumnData::F64(simple_decompress::<f64>(src).map_err(corrupt)?),
            _ => unreachable!("validated in new()"),
        })
    }
}
