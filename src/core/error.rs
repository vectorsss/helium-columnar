use thiserror::Error;

use super::coder::DataType;

/// All errors produced by the helium crate.
///
/// Variants carry enough context (column name, coder ID, data types) to
/// pinpoint which column or pipeline stage failed without re-reading the file.
#[derive(Debug, Error)]
pub enum HeliumError {
    /// A non-block coder appears after a block coder in the same pipeline.
    ///
    /// Block coders consume the whole buffer; no per-element stage can follow.
    #[error("pipeline validation: non-block coder '{0}' appears after a block coder")]
    PipelineOrder(String),

    /// The input type flowing into a coder does not match what it expects.
    #[error(
        "pipeline validation: coder '{coder}' expects input {expected:?} but previous stage produces {got:?}"
    )]
    TypeMismatch {
        /// Coder that rejected the input.
        coder: String,
        /// Type the coder requires.
        expected: DataType,
        /// Type actually flowing in from the previous stage.
        got: DataType,
    },

    /// A coder received data with the wrong runtime type.
    #[error("coder '{coder}': input has wrong runtime type (expected {expected:?})")]
    RuntimeType {
        /// Coder that detected the mismatch.
        coder: String,
        /// Type the coder expected.
        expected: DataType,
    },

    /// Encoded bytes appear corrupted (e.g. CRC mismatch, invalid bit pattern).
    #[error("coder '{coder}': data appears corrupted ({reason})")]
    Corrupted {
        /// Coder that detected the corruption.
        coder: String,
        /// Human-readable reason, suitable for operator logs.
        reason: String,
    },

    /// A coder's encode or decode operation failed.
    #[error("coder '{coder}' failed: {reason}")]
    CoderFailed {
        /// ID of the failing coder.
        coder: String,
        /// Human-readable failure description.
        reason: String,
    },

    /// A schema references a coder ID that has not been registered.
    #[error("unknown coder id '{0}' — register it before resolving this schema")]
    UnknownCoder(String),

    /// A coder parameter is missing, has the wrong type, or is out of range.
    #[error("coder '{coder}': invalid or missing parameter '{param}' ({reason})")]
    InvalidParam {
        /// Coder whose parameter is invalid.
        coder: String,
        /// Name of the invalid parameter.
        param: String,
        /// Description of why the value is invalid.
        reason: String,
    },

    /// A schema-level constraint was violated (wrong column count, row-count mismatch, etc.).
    #[error("schema column '{column}': {reason}")]
    Schema {
        /// Name of the column that triggered the violation.
        column: String,
        /// Description of the violation.
        reason: String,
    },

    /// A file-format-level error (bad magic, truncated file, CRC mismatch, etc.).
    #[error("file format: {0}")]
    Format(String),

    /// Underlying I/O error.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialisation or deserialisation error (schema / footer parsing).
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Alias for `std::result::Result<T, HeliumError>`.
pub type Result<T> = std::result::Result<T, HeliumError>;
