#![warn(missing_docs)]
#![warn(rustdoc::broken_intra_doc_links)]
#![warn(rustdoc::private_intra_doc_links)]
// Safety-first: no panicking unwrap/expect/panic in library code. Scoped to the
// library crate (and the binary, in main.rs) on purpose — integration tests,
// examples, and benches are separate crates that don't inherit this, so test
// code may unwrap freely. clippy.toml's allow-*-in-tests covers the inline
// `#[cfg(test)]` modules within src/. Restructure to fallible/Option handling
// rather than silencing; narrow #[allow] + a // SAFETY: note only where a guard
// makes the call provably infallible and rewriting would hurt clarity.
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! helium — columnar compression framework.
//!
//! Implements the core mechanism from `helium-design.md`:
//!
//! - [`Coder`] + [`NonBlockCoder`] / [`BlockCoder`] traits with the two-way split.
//! - [`Pipeline`] executor that statically validates non-block-then-block ordering.
//! - Reference coders spanning the 2×2 of block/non-block and integer/byte coders.
//! - [`Schema`] + [`CoderRegistry`] — a declarative, JSON-serializable schema.
//! - [`HeliumWriter`] / [`HeliumReader`] — a versioned `.he` file format.
//!
//! # Feature flags
//!
//! | Feature | Module | Description |
//! |---|---|---|
//! | `schema-avro` *(default)* | `schema::avro` | Avro `.avsc` → Schema |
//! | `schema-csv` | `schema::csv` | CSV type inferrer |
//! | `schema-json` | `schema::json` | JSON / NDJSON type inferrer |
//! | `schema-parquet` | `schema::parquet` | Parquet schema translator |
//! | `cli` | — | Enables the `helium` binary |
//! | `arrow` | `arrow` | Arrow `ArrayRef` ↔ `LogicalColumn` bridge |
//! | `datafusion` | `sql` | DataFusion `TableProvider` for `.he` files |

/// Arrow `ArrayRef` ↔ [`crate::LogicalColumn`] bridge.
///
/// Feature gate: `arrow`.
#[cfg(feature = "arrow")]
pub mod arrow;
/// Opt-in shared-schema catalog for catalog-mode (hash-reference) `.he` files.
///
/// See [`crate::catalog::Catalog`] for the filesystem-backed entry point.
pub mod catalog;
/// Typed, self-describing compression API (the "zstd-style" surface).
///
/// See [`compress`] / [`decompress`] for the simplest usage, or
/// [`TypedCodec`] for named pre-fabricated pipelines.
pub mod codec;
/// Concrete coder implementations (`delta`, `leb128`, `zstd`, `pcodec`, …).
///
/// Callers normally reach these through [`CoderRegistry::default()`] and
/// [`CoderSpec`] rather than instantiating them directly.
pub mod coders;
/// Core framework: traits, schema, file format.
///
/// All public items are re-exported at the crate root.
pub mod core;
/// Measured-optimal encoding picker.
///
/// See [`crate::optimizer::Optimizer`] for the primary entry point.
pub mod optimizer;
/// Source-format → [`crate::Schema`] conversion adapters.
///
/// Each adapter is behind its own feature flag (`schema-avro`, `schema-csv`, etc.).
pub mod schema;
/// DataFusion SQL query support for `.he` files.
///
/// Feature gate: `datafusion`. See [`crate::sql::HeliumTableProvider`].
#[cfg(feature = "datafusion")]
pub mod sql;

// Convenience re-exports — keep these matching the old helium-core public API
// so external code that did `use helium_core::Schema` only has to swap the
// crate name to `helium::Schema`.
pub use codec::{
    TypedCodec, compress, compress_with, compress_with_pipeline, decompress,
    decompress_with_pipeline,
};
pub use coders::{
    BitpackAuto, BitpackFixed, Delta, DeltaMin, DeltaOfDelta, EliasFano, GorillaXor, Leb128, Lz4,
    Pcodec, Rle, Snappy, Zstd,
};
pub use core::{
    AccessPattern, BlockCoder, CATALOG_SCHEMA_SLOT_LEN, CURRENT_SCHEMA_VERSION, Coder, CoderKind,
    CoderRegistry, CoderSpec, ColumnData, ColumnSpec, ContainmentFilter, DataType, DateUnit,
    FORMAT_VERSION, FieldSpec, HeliumError, HeliumReader, HeliumWriter, LogicalColumn, LogicalType,
    MAGIC, MAX_NESTED_DEPTH, MinMaxValue, NonBlockCoder, PhysicalColumnStats,
    PhysicalField, Pipeline, Result, Schema, StageCoder, TimeUnit, bloom_might_contain,
    canonicalize_json, filter_might_contain_mmv, min_max_value_to_hash_bytes, schema_hash,
};
