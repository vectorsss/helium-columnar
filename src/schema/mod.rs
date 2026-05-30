//! Source-format → Helium schema conversion tools.
//!
//! Converts external schema and data formats into Helium [`crate::Schema`]
//! objects suitable for use with [`crate::HeliumWriter`] / [`crate::HeliumReader`].
//!
//! # Adapters (feature flags)
//!
//! | Feature | Module | Entry point |
//! |---|---|---|
//! | `schema-avro` *(default)* | `avro` | `avsc_to_schema` |
//! | `schema-csv` | `csv` | `csv::schema_from_csv` |
//! | `schema-json` | `json` | `json::schema_from_json` |
//! | `schema-parquet` | `parquet` | `parquet::schema_from_parquet` |
//!
//! Cargo.toml examples:
//! ```toml
//! # All adapters:
//! helium = { features = ["schema-avro", "schema-csv", "schema-json", "schema-parquet"] }
//! # CSV only (no Avro):
//! helium = { default-features = false, features = ["schema-csv"] }
//! ```
//!
//! # Architecture
//!
//! This module is **write-side only** — it converts formats and produces
//! [`crate::Schema`] values.  It is never on the `.he` read/write hot
//! path.  Format-specific dependencies (e.g. `csv`, `parquet`) are optional
//! and stay out of the default dependency closure.
//!
//! # Default encoding policy
//!
//! All adapters call `encodings::default_encodings` — a single shared
//! module — to assign sensible per-type compression pipelines.  See that
//! module for the full policy table.

/// Shared default-encoding policy used by every format adapter.
///
/// See [`encodings::default_encodings`] for the full policy table.
pub mod encodings;

/// Avro `.avsc` → Helium [`crate::Schema`] parser.
///
/// Feature flag: `schema-avro` (default).
#[cfg(feature = "schema-avro")]
pub mod avro;

/// CSV type-inferrer.
///
/// Feature flag: `schema-csv`.
#[cfg(feature = "schema-csv")]
pub mod csv;

/// JSON / NDJSON type-inferrer.
///
/// Feature flag: `schema-json`.
#[cfg(feature = "schema-json")]
pub mod json;

/// Parquet schema → Helium [`crate::Schema`] translator.
///
/// Feature flag: `schema-parquet`.
#[cfg(feature = "schema-parquet")]
pub mod parquet;

// Convenience re-exports from the Avro adapter.
#[cfg(feature = "schema-avro")]
pub use avro::{
    avsc_to_logical_type, avsc_to_schema, read_avro_data, read_avro_data_chunked, write_avro_data,
};
