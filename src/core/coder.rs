use serde::{Deserialize, Serialize};

use super::error::Result;

/// Physical data types the framework operates on. Names are lowercase in JSON
/// (`"i64"`, `"f32"`, …) for schema interop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DataType {
    /// Signed 8-bit integer.
    I8,
    /// Signed 16-bit integer.
    I16,
    /// Signed 32-bit integer.
    I32,
    /// Signed 64-bit integer.
    I64,
    /// Unsigned 8-bit integer.
    U8,
    /// Unsigned 16-bit integer.
    U16,
    /// Unsigned 32-bit integer.
    U32,
    /// Unsigned 64-bit integer.
    U64,
    /// 32-bit float (IEEE 754 single precision).
    F32,
    /// 64-bit float (IEEE 754 double precision).
    F64,
    /// Opaque byte buffer — post-encoding intermediate, not a user-level
    /// column type. Pipelines end here when they feed the file format.
    Bytes,
}

impl DataType {
    /// Returns `true` if this type is a signed integer (`I8`, `I16`, `I32`, `I64`).
    pub fn is_signed_integer(self) -> bool {
        matches!(self, Self::I8 | Self::I16 | Self::I32 | Self::I64)
    }

    /// Returns `true` if this type is an unsigned integer (`U8`, `U16`, `U32`, `U64`).
    pub fn is_unsigned_integer(self) -> bool {
        matches!(self, Self::U8 | Self::U16 | Self::U32 | Self::U64)
    }

    /// Returns `true` if this type is any integer (signed or unsigned).
    pub fn is_integer(self) -> bool {
        self.is_signed_integer() || self.is_unsigned_integer()
    }

    /// Returns `true` if this type is a floating-point number (`F32` or `F64`).
    pub fn is_float(self) -> bool {
        matches!(self, Self::F32 | Self::F64)
    }

    /// Returns the fixed byte width of this type, or `None` for `Bytes`.
    pub fn byte_width(self) -> Option<usize> {
        Some(match self {
            Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::I64 | Self::U64 | Self::F64 => 8,
            Self::Bytes => return None,
        })
    }
}

/// Whether a coder operates per-element (streaming) or on the whole buffer at once.
///
/// Used by [`Pipeline::new`](crate::Pipeline::new) to enforce the design §2.2 ordering rule:
/// non-block stages must precede all block stages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoderKind {
    /// Per-element coder — `encode`/`decode` process one element at a time.
    NonBlock,
    /// Whole-buffer coder — `encode_block`/`decode_block` consume the full input.
    Block,
}

/// Whether an algorithm supports independent per-element decoding.
///
/// This is a property of the **algorithm**, not of the current decode API.
/// The framework today always decodes the whole column; this enum is exposed
/// so:
/// - schemas can declare "this column must support point queries"
/// - pipeline composition can statically check the property is preserved
///   (weakest-link: any `SequentialOnly` stage downgrades the whole chain)
/// - a future `decode_at` / `read_range` reader API can drop in without
///   touching coder trait signatures
///
/// Rule for combining across a pipeline: every stage must be `RandomAccess`
/// for the pipeline to be `RandomAccess`. A single `SequentialOnly` stage
/// (delta, leb128, zstd, …) forces sequential decode of the entire chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AccessPattern {
    /// Must decode from the start to reach element `i`.
    SequentialOnly,
    /// Any element can be decoded independently (O(1) or O(log n)).
    RandomAccess,
}

impl AccessPattern {
    /// Pipeline combination: weakest link wins.
    pub fn combine(self, other: Self) -> Self {
        match (self, other) {
            (Self::RandomAccess, Self::RandomAccess) => Self::RandomAccess,
            _ => Self::SequentialOnly,
        }
    }
}

/// A typed column payload. Variant tag carries the physical type; inner `Vec`
/// holds the values. For `Bytes`, the buffer has no inherent row structure —
/// it is the opaque output of a block coder or the input of another.
#[derive(Debug, Clone, PartialEq)]
pub enum ColumnData {
    /// Signed 8-bit integer column.
    I8(Vec<i8>),
    /// Signed 16-bit integer column.
    I16(Vec<i16>),
    /// Signed 32-bit integer column.
    I32(Vec<i32>),
    /// Signed 64-bit integer column.
    I64(Vec<i64>),
    /// Unsigned 8-bit integer column (also used for booleans and present bitmaps).
    U8(Vec<u8>),
    /// Unsigned 16-bit integer column.
    U16(Vec<u16>),
    /// Unsigned 32-bit integer column (also used for offsets and indices).
    U32(Vec<u32>),
    /// Unsigned 64-bit integer column.
    U64(Vec<u64>),
    /// 32-bit float column.
    F32(Vec<f32>),
    /// 64-bit float column.
    F64(Vec<f64>),
    /// Opaque byte buffer — the output of block coders and the final output
    /// written to the `.he` file.
    Bytes(Vec<u8>),
}

impl ColumnData {
    /// Returns the [`DataType`] tag of this payload.
    pub fn data_type(&self) -> DataType {
        match self {
            Self::I8(_) => DataType::I8,
            Self::I16(_) => DataType::I16,
            Self::I32(_) => DataType::I32,
            Self::I64(_) => DataType::I64,
            Self::U8(_) => DataType::U8,
            Self::U16(_) => DataType::U16,
            Self::U32(_) => DataType::U32,
            Self::U64(_) => DataType::U64,
            Self::F32(_) => DataType::F32,
            Self::F64(_) => DataType::F64,
            Self::Bytes(_) => DataType::Bytes,
        }
    }

    /// Returns the number of elements (rows for typed variants, bytes for `Bytes`).
    pub fn len(&self) -> usize {
        match self {
            Self::I8(v) => v.len(),
            Self::I16(v) => v.len(),
            Self::I32(v) => v.len(),
            Self::I64(v) => v.len(),
            Self::U8(v) => v.len(),
            Self::U16(v) => v.len(),
            Self::U32(v) => v.len(),
            Self::U64(v) => v.len(),
            Self::F32(v) => v.len(),
            Self::F64(v) => v.len(),
            Self::Bytes(v) => v.len(),
        }
    }

    /// Returns `true` if the column is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Core coder trait — identity and type signature.
///
/// All coders implement this trait.  The sub-traits [`NonBlockCoder`] and
/// [`BlockCoder`] add the actual encode/decode methods.
pub trait Coder: Send + Sync {
    /// Stable string identifier for this coder (e.g. `"delta"`, `"zstd"`).
    ///
    /// IDs are **frozen** — once any `.he` file is written with a given ID the
    /// ID must never be repurposed for a different algorithm.
    fn id(&self) -> &'static str;

    /// Whether this is a [`NonBlockCoder`] or a [`BlockCoder`].
    fn kind(&self) -> CoderKind;

    /// The [`DataType`] this coder accepts as input.
    fn accepted_input_type(&self) -> DataType;

    /// The [`DataType`] this coder produces as output.
    fn produced_output_type(&self) -> DataType;

    /// Does the algorithm admit independent per-element decoding?
    /// Conservative default: `SequentialOnly`. Coders that support O(1) or
    /// O(log n) access to an arbitrary element (e.g. bit-packing,
    /// Elias-Fano) override this to return `RandomAccess`.
    fn access_pattern(&self) -> AccessPattern {
        AccessPattern::SequentialOnly
    }
}

/// Per-element (streaming) coder.
///
/// `encode` and `decode` operate element-by-element — the output length
/// equals the input length for most transformations (delta, leb128, rle).
/// Must come before all [`BlockCoder`]s in a [`Pipeline`](crate::Pipeline).
pub trait NonBlockCoder: Coder {
    /// Encode `input`, producing a new column of possibly different type.
    fn encode(&self, input: &ColumnData) -> Result<ColumnData>;
    /// Decode `input`, reversing the encode transformation.
    fn decode(&self, input: &ColumnData) -> Result<ColumnData>;
}

/// Whole-buffer coder (compressor / decompressor).
///
/// `encode_block` and `decode_block` consume the entire input buffer at once.
/// Block coders always produce and consume [`DataType::Bytes`].
/// Must come after all [`NonBlockCoder`]s in a [`Pipeline`](crate::Pipeline).
pub trait BlockCoder: Coder {
    /// Compress the entire `input` buffer into a `Bytes` output.
    fn encode_block(&self, input: &ColumnData) -> Result<ColumnData>;
    /// Decompress the `input` `Bytes` buffer.
    fn decode_block(&self, input: &ColumnData) -> Result<ColumnData>;
}

/// A type-erased coder that is either a [`NonBlockCoder`] or a [`BlockCoder`].
///
/// Produced by [`CoderRegistry::build`](crate::CoderRegistry::build) and
/// held in a [`Pipeline`](crate::Pipeline) stage list.
pub enum StageCoder {
    /// A per-element coder stage.
    NonBlock(Box<dyn NonBlockCoder>),
    /// A whole-buffer coder stage.
    Block(Box<dyn BlockCoder>),
}

impl StageCoder {
    /// Stable ID of the underlying coder.
    pub fn id(&self) -> &'static str {
        match self {
            Self::NonBlock(c) => c.id(),
            Self::Block(c) => c.id(),
        }
    }

    /// Kind of the underlying coder (`NonBlock` or `Block`).
    pub fn kind(&self) -> CoderKind {
        match self {
            Self::NonBlock(_) => CoderKind::NonBlock,
            Self::Block(_) => CoderKind::Block,
        }
    }

    /// The [`DataType`] this stage accepts as input.
    pub fn accepted_input_type(&self) -> DataType {
        match self {
            Self::NonBlock(c) => c.accepted_input_type(),
            Self::Block(c) => c.accepted_input_type(),
        }
    }

    /// The [`DataType`] this stage produces as output.
    pub fn produced_output_type(&self) -> DataType {
        match self {
            Self::NonBlock(c) => c.produced_output_type(),
            Self::Block(c) => c.produced_output_type(),
        }
    }

    /// Access pattern of the underlying coder.
    pub fn access_pattern(&self) -> AccessPattern {
        match self {
            Self::NonBlock(c) => c.access_pattern(),
            Self::Block(c) => c.access_pattern(),
        }
    }
}
