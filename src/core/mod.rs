//! Core compression framework: coders, pipelines, schema, file format.
//!
//! These modules were previously the top-level `helium-core` library.
//! All public items are re-exported at the crate root for convenience.

/// Schema canonicalization and BLAKE3 hashing for v4/v6 catalog mode.
pub mod canonicalize;
/// Core coder traits ([`Coder`], [`NonBlockCoder`], [`BlockCoder`]) and
/// the physical [`DataType`] / [`ColumnData`] enums.
pub mod coder;
/// [`HeliumError`] and the [`Result`] type alias.
pub mod error;
/// [`HeliumWriter`] and [`HeliumReader`] — the `.he` file format.
pub mod file;
/// Per-stripe column statistics (`MinMaxValue`, `PhysicalColumnStats`) and
/// containment filters (`ContainmentFilter`) stored in the `.he` footer.
pub mod footer_stats;
/// [`Pipeline`] — validated, ordered sequence of [`StageCoder`]s.
pub mod pipeline;
/// [`CoderRegistry`] and [`CoderSpec`] — coder factory registry.
pub mod registry;
/// [`Schema`], [`LogicalType`], [`LogicalColumn`] and physical-field decomposition.
pub mod schema;

pub use canonicalize::{canonicalize_json, schema_hash};
pub use coder::{
    AccessPattern, BlockCoder, Coder, CoderKind, ColumnData, DataType, NonBlockCoder, StageCoder,
};
pub use error::{HeliumError, Result};
pub use file::{CATALOG_SCHEMA_SLOT_LEN, FORMAT_VERSION, HeliumReader, HeliumWriter, MAGIC};
pub use footer_stats::{
    ContainmentFilter, MinMaxValue, PhysicalColumnStats, bloom_might_contain,
    filter_might_contain_mmv, min_max_value_to_hash_bytes,
};
pub use pipeline::Pipeline;
pub use registry::{CoderRegistry, CoderSpec};
pub use schema::{
    CURRENT_SCHEMA_VERSION, ColumnSpec, DateUnit, FieldSpec, LogicalColumn, LogicalType,
    MAX_NESTED_DEPTH, PhysicalField, Schema, TimeUnit,
};
