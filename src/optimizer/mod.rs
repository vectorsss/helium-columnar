//! `helium-optimizer` — encoding picker library for helium-core.
//!
//! Automatically selects optimal compression pipelines per column by measuring
//! compressed size on sample data.  The main entry point is [`crate::optimizer::Optimizer`], which
//! accepts (name, [`LogicalType`], [`LogicalColumn`]) triples and returns a
//! [`Schema`] with best-fit encodings filled in.
//!
//! # Quick start
//!
//! ```rust,ignore
//! use helium::{ColumnData, LogicalColumn, LogicalType, DataType};
//! use helium::optimizer::Optimizer;
//!
//! // Build sample data
//! let values: Vec<i64> = (0..1000).map(|i| i * 1000).collect();
//! let lc = LogicalColumn::Primitive(ColumnData::I64(values));
//! let lt = LogicalType::Primitive { data_type: DataType::I64 };
//!
//! // Optimize
//! let schema = Optimizer::new()
//!     .optimize(vec![("timestamp".to_string(), lt, lc)])
//!     .unwrap();
//! assert_eq!(schema.columns.len(), 1);
//! ```
//!
//! # Extended API
//!
//! - [`crate::optimizer::measure_encoding`] — measure the compressed byte count of a given
//!   `ColumnSpec` + `LogicalColumn` pair.
//! - [`crate::optimizer::candidates::LeafCandidate`] — a candidate pipeline for a single physical leaf.
//! - [`crate::optimizer::candidates::structural_candidates`] / [`crate::optimizer::candidates::data_candidates`] — generate
//!   heuristic candidate sets for structural and data leaves.
//! - [`crate::optimizer::picker::measure_pipeline`] — measure one leaf pipeline.
//! - [`crate::optimizer::picker::pick_best_leaf`] — pick best encoding for one physical leaf.

pub mod candidates;
pub mod picker;
pub mod recursive;

use crate::{CoderRegistry, CoderSpec, ColumnSpec, LogicalColumn, LogicalType, Result, Schema};

pub use candidates::{LeafCandidate, data_candidates, structural_candidates};
pub use picker::{measure_pipeline, pick_best_leaf};

// ---------------------------------------------------------------------------
// Public measurement primitive (promoted from examples/compare_codecs.rs)
// ---------------------------------------------------------------------------

/// Measure the total compressed byte count when encoding `lc` using the
/// pipeline declared in `spec`.
///
/// This is the promoted form of `encode_one_column` from
/// `examples/compare_codecs.rs`.  It builds physical pipelines from the
/// `ColumnSpec`, decomposes `lc` into physical leaf columns, runs each through
/// its pipeline, and sums the output byte counts.
///
/// Returns an error if any pipeline stage fails (type mismatch, unknown coder,
/// invalid encoding, etc.).
pub fn measure_encoding(
    spec: &ColumnSpec,
    lc: LogicalColumn,
    registry: &CoderRegistry,
) -> Result<usize> {
    let schema = Schema::new(vec![spec.clone()]);
    let all_pipelines = schema.resolve_all(registry)?;
    let pipes = &all_pipelines[0];
    let parts = lc.decompose(&spec.logical_type)?;
    let mut total = 0usize;
    for (part, pipe) in parts.into_iter().zip(pipes.iter()) {
        let encoded = pipe.encode(part)?;
        if let crate::ColumnData::Bytes(b) = encoded {
            total += b.len();
        }
    }
    Ok(total)
}

// ---------------------------------------------------------------------------
// Optimizer
// ---------------------------------------------------------------------------

/// Configuration for the optimizer.
///
/// Controls which terminal block compressor (`zstd`, `lz4`, `snappy`) is placed
/// at the end of every pipeline.  Other structural coders (delta, gorilla,
/// leb128, rle, bitpack_auto, pcodec, etc.) are selected per-leaf by the
/// heuristic candidate generator in [`candidates`].
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    /// Terminal block compressor (`"zstd"`, `"lz4"`, or `"snappy"`).
    /// Default: `"zstd"`.
    pub terminal: String,
    /// Global zstd compression level, applied to every `zstd` terminal the
    /// optimizer emits. The level is a single global setting — it is **not**
    /// searched per column. `None` leaves the terminal parameter-free, which
    /// the coder reads as the zstd default (3). Ignored when `terminal` is not
    /// `"zstd"`.
    pub zstd_level: Option<i32>,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            terminal: "zstd".into(),
            zstd_level: None,
        }
    }
}

/// Encoding picker that automatically selects compression pipelines.
///
/// The optimizer measures compressed size on sample data and picks the
/// pipeline that produces the fewest bytes for each physical leaf column.
/// It handles all `LogicalType` variants including v3 recursive types
/// (Struct, List, Map, Nullable, Union).
///
/// # Usage pattern
///
/// 1. Build sample data as `LogicalColumn` values (typically a representative
///    stripe of ≥1 000 rows for reliable compression measurements).
/// 2. Call [`Optimizer::optimize`] with `(name, LogicalType, LogicalColumn)` triples.
/// 3. Use the returned `Schema` to construct a `HeliumWriter`.
///
/// # Note on dict types
///
/// The optimizer does not automatically promote `Primitive` columns to `Dictionary`.
/// If you want dictionary encoding, build a `LogicalColumn::Dictionary` first
/// (use `LogicalColumn::dict_encode_primitive` or `dict_encode_utf8` to prepare it).
pub struct Optimizer {
    registry: CoderRegistry,
    config: OptimizerConfig,
}

impl Default for Optimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Optimizer {
    /// Create a new optimizer with default settings (zstd terminal).
    pub fn new() -> Self {
        Self {
            registry: CoderRegistry::default(),
            config: OptimizerConfig::default(),
        }
    }

    /// Create a new optimizer with a custom terminal compressor.
    ///
    /// `terminal` must be one of `"zstd"`, `"lz4"`, `"snappy"`.
    pub fn with_terminal(terminal: impl Into<String>) -> Self {
        Self {
            registry: CoderRegistry::default(),
            config: OptimizerConfig {
                terminal: terminal.into(),
                ..OptimizerConfig::default()
            },
        }
    }

    /// Create a new optimizer with a custom configuration.
    pub fn with_config(config: OptimizerConfig) -> Self {
        Self {
            registry: CoderRegistry::default(),
            config,
        }
    }

    /// Set the global zstd compression level (1–22; the zstd default is 3).
    ///
    /// The level is applied to every `zstd` terminal the optimizer emits; it is
    /// a single global knob, not searched per column. No effect unless the
    /// terminal compressor is `zstd`.
    pub fn with_zstd_level(mut self, level: i32) -> Self {
        self.config.zstd_level = Some(level);
        self
    }

    /// Return the current terminal coder ID.
    pub fn terminal(&self) -> &str {
        &self.config.terminal
    }

    /// Build the fully-specified terminal coder spec (id + global level).
    fn terminal_spec(&self) -> CoderSpec {
        let spec = CoderSpec::new(&self.config.terminal);
        match self.config.zstd_level {
            Some(level) if self.config.terminal == "zstd" => spec.with_param("level", level),
            _ => spec,
        }
    }

    /// Optimize encodings for a list of columns.
    ///
    /// Each entry is `(column_name, logical_type_skeleton, sample_data)`.
    /// The skeleton specifies the structural type (which physical leaves exist and
    /// their data types); the optimizer fills in the encoding pipelines.
    ///
    /// Returns a [`Schema`] ready for use with [`crate::HeliumWriter`].
    pub fn optimize(&self, columns: Vec<(String, LogicalType, LogicalColumn)>) -> Result<Schema> {
        let terminal = self.terminal_spec();
        let mut specs = Vec::with_capacity(columns.len());
        for (name, lt, lc) in columns {
            let spec = recursive::optimize_column(&name, lt, lc, &terminal, &self.registry)?;
            specs.push(spec);
        }
        Ok(Schema::new(specs))
    }
}
