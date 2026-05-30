//! CLI subcommand implementations.
//!
//! Each module implements one `helium` subcommand.

pub mod catalog;
pub mod compare;
pub mod convert;
pub mod infer_schema;
pub mod loader;
pub mod optimize_schema;
pub mod slice;
#[cfg(feature = "datafusion")]
pub mod sql;
pub mod stats;
pub mod verify;
