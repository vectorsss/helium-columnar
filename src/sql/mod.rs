//! DataFusion SQL query support for Helium `.he` files.
//!
//! This module exposes [`crate::sql::HeliumTableProvider`], which implements
//! DataFusion's `TableProvider` trait so that `.he` files can be
//! registered with a `SessionContext` and queried via SQL.
//!
//! # Feature gate
//!
//! This module is only compiled when the `datafusion` Cargo feature is
//! enabled. The feature transitively enables the `arrow` feature.
//!
//! # Example
//!
//! ```rust,no_run
//! # #[cfg(feature = "datafusion")]
//! # async fn example() -> datafusion::error::Result<()> {
//! use datafusion::prelude::*;
//! use helium::sql::HeliumTableProvider;
//!
//! let provider = HeliumTableProvider::try_new("data.he").unwrap();
//! let ctx = SessionContext::new();
//! ctx.register_table("my_table", std::sync::Arc::new(provider))?;
//! let df = ctx.sql("SELECT count(*) FROM my_table").await?;
//! df.show().await?;
//! # Ok(())
//! # }
//! ```
//!
//! # Projection pushdown
//!
//! When DataFusion requests a column subset, [`crate::sql::HeliumTableProvider`]
//! passes the projection down to the underlying `HeliumExec`, which
//! reads only the requested columns from disk — Helium's existing
//! column-pruning machinery handles the byte-level work.
//!
//! # Filter pushdown (stripe-level)
//!
//! All filters are reported as `TableProviderFilterPushDown::Inexact`.
//! DataFusion applies every filter in memory after the scan to guarantee
//! correctness.
//!
//! Additionally, the min/max statistics embedded in the `.he` footer (per
//! stripe per physical column) are used to **prune entire stripes** at scan
//! time via DataFusion's `PruningPredicate` machinery.  Stripes that are
//! proven to contain no matching rows are skipped without reading their body
//! bytes, yielding a near-instant result when all stripes are prunable.
//!
//! This is stripe-level (coarse) pruning — within a kept stripe, rows are
//! still filtered in memory by DataFusion.  This is why `Inexact` is returned
//! rather than `Exact`.
//!
//! # Async I/O
//!
//! DataFusion's `execute` method is called from an async context but
//! Helium I/O is synchronous. We use `tokio::task::block_in_place` inside
//! `HeliumExec::execute` to avoid blocking the Tokio thread pool. This is
//! correct for the current state of the implementation; a future task can
//! migrate to `tokio::fs` for fully async I/O.
//!
//! [`TableProvider`]: ::datafusion::catalog::TableProvider
//! [`SessionContext`]: ::datafusion::prelude::SessionContext
//! [`PruningPredicate`]: ::datafusion::physical_optimizer::pruning::PruningPredicate

pub mod exec;
pub mod pruning;

pub use exec::HeliumExec;
pub use pruning::HeliumPruningStatistics;

use std::any::Any;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use ::async_trait::async_trait;
use ::datafusion::catalog::{Session, TableProvider};
use ::datafusion::common::stats::Precision;
use ::datafusion::common::{ColumnStatistics, DFSchema, Statistics};
use ::datafusion::datasource::TableType;
use ::datafusion::error::{DataFusionError, Result as DfResult};
use ::datafusion::logical_expr::{Expr, TableProviderFilterPushDown};
use ::datafusion::physical_expr::create_physical_expr;
use ::datafusion::physical_optimizer::pruning::PruningPredicate;
use ::datafusion::physical_plan::ExecutionPlan;
use arrow::datatypes::SchemaRef;

use crate::arrow::schema_to_arrow;
use crate::core::registry::CoderRegistry;
use crate::core::schema::Schema as HeliumSchema;
use crate::sql::pruning::min_max_to_scalar;

/// DataFusion [`TableProvider`] for a Helium `.he` file.
///
/// Created with [`HeliumTableProvider::try_new`]. The file is opened once at
/// construction time to read the schema, stripe count, total row count, and
/// per-stripe per-column min/max statistics (used for stripe-level predicate
/// pruning at scan time).
///
/// Body bytes are not loaded until a query is executed.
///
/// Register with [`SessionContext::register_table`] and query via SQL.
///
/// [`TableProvider`]: ::datafusion::catalog::TableProvider
/// [`SessionContext::register_table`]: ::datafusion::prelude::SessionContext::register_table
#[derive(Debug)]
pub struct HeliumTableProvider {
    /// Path to the `.he` file.
    path: PathBuf,
    /// Arrow schema derived from the Helium schema (all columns).
    arrow_schema: SchemaRef,
    /// Cached Helium schema for metadata access and as in-memory resolver for catalog-mode files.
    helium_schema: Arc<HeliumSchema>,
    /// Number of stripes in the file.
    stripe_count: usize,
    /// Total row count across all stripes.
    total_rows: u64,
    /// Per-stripe row counts (needed by `HeliumExec` for zero-column projections
    /// and for `PruningStatistics::row_counts`).
    stripe_row_counts: Vec<u64>,
    /// Pre-built pruning statistics (one entry per logical column, per stripe).
    /// Used in `scan()` to evaluate `PruningPredicate`s before constructing
    /// `HeliumExec`, allowing us to skip stripes with no matching rows.
    pruning_stats: HeliumPruningStatistics,
    /// File-wide DataFusion [`Statistics`] pre-computed at open time.
    ///
    /// `num_rows` is `Exact` (from footer). `total_byte_size` is `Exact`
    /// (on-disk encoded size — acceptable for cost-model purposes; it is the
    /// compressed size, not the in-memory decoded size, but DF uses it only
    /// for cost estimates, not for correctness). Per-column min/max and
    /// null_count are `Exact` when every stripe has stats for that column,
    /// otherwise `Absent` (we never emit `Inexact` — partial stats are
    /// misleading for the planner).
    helium_statistics: Statistics,
}

impl HeliumTableProvider {
    /// Open a `.he` file and extract its schema + stripe metadata.
    ///
    /// The file is closed after reading the header/footer — body data is not
    /// loaded until [`scan`] is called.
    ///
    /// # Errors
    ///
    /// Returns a [`crate::HeliumError`] if the file cannot be opened or
    /// if its schema is not representable as an Arrow schema.
    ///
    /// [`scan`]: HeliumTableProvider::scan
    pub fn try_new(path: impl Into<PathBuf>) -> crate::Result<Self> {
        use std::fs::File;

        let path: PathBuf = path.into();
        let file = File::open(&path).map_err(|e| {
            crate::HeliumError::Format(format!(
                "HeliumTableProvider: cannot open {}: {e}",
                path.display()
            ))
        })?;

        let registry = CoderRegistry::default();
        let reader = crate::HeliumReader::new(file, &registry)?;
        Self::from_reader(path, reader)
    }

    /// Open a catalog-mode `.he` file with a schema resolver.
    ///
    /// The resolver is called with the BLAKE3 hash stored in the file header to
    /// look up the schema from the caller's catalog directory.  For self-contained
    /// files the resolver is ignored.
    ///
    /// # Errors
    ///
    /// Returns a [`crate::HeliumError`] if the file cannot be opened, the hash
    /// is not found by the resolver, or the schema cannot be represented as an
    /// Arrow schema.
    pub fn try_new_with_catalog(
        path: impl Into<PathBuf>,
        resolver: impl Fn(&blake3::Hash) -> crate::Result<crate::core::schema::Schema> + 'static,
    ) -> crate::Result<Self> {
        use std::fs::File;

        let path: PathBuf = path.into();
        let file = File::open(&path).map_err(|e| {
            crate::HeliumError::Format(format!(
                "HeliumTableProvider: cannot open {}: {e}",
                path.display()
            ))
        })?;

        let registry = CoderRegistry::default();
        let reader = crate::HeliumReader::new_with_resolver(file, &registry, resolver)?;
        Self::from_reader(path, reader)
    }

    /// Shared initialization from an already-opened `HeliumReader`.
    fn from_reader(
        path: PathBuf,
        reader: crate::HeliumReader<std::fs::File>,
    ) -> crate::Result<Self> {
        let helium_schema = Arc::new(reader.schema().clone());
        let stripe_count = reader.stripe_count();
        let total_rows = reader.row_count();
        let stripe_row_counts = reader.stripe_row_counts();
        let arrow_schema = Arc::new(schema_to_arrow(&helium_schema));

        let mut column_stats: HashMap<String, Vec<Vec<crate::PhysicalColumnStats>>> =
            HashMap::new();
        let mut column_types: HashMap<String, crate::LogicalType> = HashMap::new();
        let mut column_filters: HashMap<String, Vec<Vec<Option<crate::ContainmentFilter>>>> =
            HashMap::new();

        for col_spec in &helium_schema.columns {
            let mut per_stripe_stats: Vec<Vec<crate::PhysicalColumnStats>> =
                Vec::with_capacity(stripe_count);
            let mut per_stripe_filters: Vec<Vec<Option<crate::ContainmentFilter>>> =
                Vec::with_capacity(stripe_count);
            for stripe_idx in 0..stripe_count {
                let leaves = reader
                    .stripe_column_stats(stripe_idx, &col_spec.name)
                    .unwrap_or_default();
                per_stripe_stats.push(leaves);
                let filters = reader
                    .stripe_column_filter(stripe_idx, &col_spec.name)
                    .unwrap_or_default();
                per_stripe_filters.push(filters);
            }
            column_stats.insert(col_spec.name.clone(), per_stripe_stats);
            column_types.insert(col_spec.name.clone(), col_spec.logical_type.clone());
            column_filters.insert(col_spec.name.clone(), per_stripe_filters);
        }

        let pruning_stats = HeliumPruningStatistics {
            stripe_count,
            stripe_row_counts: stripe_row_counts.clone(),
            column_stats,
            column_types,
            column_filters,
        };

        // ----------------------------------------------------------------
        // Pre-compute file-wide DataFusion Statistics from footer metadata.
        // ----------------------------------------------------------------
        let total_byte_size: u64 = reader.column_byte_sizes().iter().map(|(_, sz)| sz).sum();

        let column_statistics: Vec<ColumnStatistics> = helium_schema
            .columns
            .iter()
            .map(|col_spec| {
                build_column_statistics(
                    &col_spec.name,
                    &col_spec.logical_type,
                    pruning_stats.column_stats.get(&col_spec.name),
                    stripe_count,
                )
            })
            .collect();

        let helium_statistics = Statistics {
            num_rows: Precision::Exact(total_rows as usize),
            // total_byte_size is the on-disk encoded (compressed) size.
            // DataFusion uses this for cost estimation, not correctness,
            // so the compressed size is an acceptable approximation.
            total_byte_size: Precision::Exact(total_byte_size as usize),
            column_statistics,
        };

        Ok(Self {
            path,
            arrow_schema,
            helium_schema,
            stripe_count,
            total_rows,
            stripe_row_counts,
            pruning_stats,
            helium_statistics,
        })
    }

    /// The Helium schema as read from the `.he` file header.
    pub fn helium_schema(&self) -> &HeliumSchema {
        &self.helium_schema
    }

    /// The Arrow schema (all columns, pre-computed at open time).
    pub fn arrow_schema(&self) -> &SchemaRef {
        &self.arrow_schema
    }

    /// The pre-built stripe-level pruning statistics.
    ///
    /// Exposed for testing and tooling that wants to inspect stats directly.
    pub fn pruning_stats(&self) -> &HeliumPruningStatistics {
        &self.pruning_stats
    }

    /// Total row count across all stripes.
    pub fn total_rows(&self) -> u64 {
        self.total_rows
    }

    /// Number of stripes (partitions) in the file.
    pub fn stripe_count(&self) -> usize {
        self.stripe_count
    }

    /// File-wide DataFusion [`Statistics`] pre-computed at open time.
    ///
    /// Exposed for testing and tooling that wants to inspect statistics directly.
    pub fn file_statistics(&self) -> &Statistics {
        &self.helium_statistics
    }
}

#[async_trait]
impl TableProvider for HeliumTableProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.arrow_schema)
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    /// Create a scan plan for this table.
    ///
    /// - `projection`: column indices to include (None = all columns).
    /// - `filters`: filter expressions. We report `Inexact` for all of them
    ///   so DataFusion re-applies them after the scan. **Additionally**, we use
    ///   DataFusion's `PruningPredicate` machinery to skip entire stripes whose
    ///   min/max statistics prove no rows can match the predicate.
    /// - `limit`: optional row limit to push down.
    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        // Validate projection indices.
        if let Some(proj) = projection {
            let ncols = self.arrow_schema.fields().len();
            for &idx in proj {
                if idx >= ncols {
                    return Err(DataFusionError::Plan(format!(
                        "HeliumTableProvider: projection index {idx} out of range \
                         (schema has {ncols} columns)"
                    )));
                }
            }
        }

        // ----------------------------------------------------------------
        // Stripe-level predicate pruning
        // ----------------------------------------------------------------
        // Build a keep-mask over stripes using min/max statistics. We start
        // with all stripes kept, then AND in each filter's pruning result.
        // If PruningPredicate construction or evaluation fails for any filter
        // we conservatively keep all stripes (never drops rows incorrectly).
        let mut keep: Vec<bool> = vec![true; self.stripe_count.max(1)];

        if self.stripe_count > 0 && !filters.is_empty() {
            // Build a DFSchema from the Arrow schema (unqualified, no table alias).
            if let Ok(df_schema) = DFSchema::try_from(self.arrow_schema.as_ref().clone()) {
                let execution_props = state.execution_props();
                for filter in filters {
                    // Convert the logical Expr to a PhysicalExpr.
                    let phys_expr = match create_physical_expr(filter, &df_schema, execution_props)
                    {
                        Ok(e) => e,
                        Err(_) => continue, // unsupported expr — keep all
                    };
                    // Build and evaluate the PruningPredicate.
                    let pp = match PruningPredicate::try_new(
                        phys_expr,
                        Arc::clone(&self.arrow_schema),
                    ) {
                        Ok(pp) => pp,
                        Err(_) => continue, // can't build — keep all
                    };
                    match pp.prune(&self.pruning_stats) {
                        Ok(stripe_keep) => {
                            for (i, k) in stripe_keep.iter().enumerate() {
                                if !k {
                                    keep[i] = false;
                                }
                            }
                        }
                        Err(_) => {
                            // Evaluation failed — conservatively keep all stripes.
                        }
                    }
                }
            }
        }

        let keep_stripes: Vec<usize> = keep
            .iter()
            .enumerate()
            .filter(|(_, k)| **k)
            .map(|(i, _)| i)
            .collect();

        let exec = HeliumExec::new(
            self.path.clone(),
            Arc::clone(&self.arrow_schema),
            projection.cloned(),
            limit,
            self.stripe_count,
            self.stripe_row_counts.clone(),
            Some(keep_stripes),
            Arc::clone(&self.helium_schema),
            self.helium_statistics.clone(),
        );

        Ok(Arc::new(exec))
    }

    /// Report filter pushdown support.
    ///
    /// We return [`Inexact`] for every filter: DataFusion will pass them to
    /// `scan()` and then re-evaluate them in memory. This is always safe —
    /// no rows are ever silently dropped. Upgrading to `Exact` requires
    /// per-column min/max statistics which are not yet in the file format.
    ///
    /// [`Inexact`]: TableProviderFilterPushDown::Inexact
    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> DfResult<Vec<TableProviderFilterPushDown>> {
        Ok(filters
            .iter()
            .map(|_| TableProviderFilterPushDown::Inexact)
            .collect())
    }

    /// Return file-wide statistics pre-computed at open time.
    ///
    /// - `num_rows`: `Exact` — sum of per-stripe row counts from the footer.
    /// - `total_byte_size`: `Exact` — total on-disk encoded (compressed) size
    ///   across all columns and stripes.  This is the compressed size; DataFusion
    ///   uses it for cost estimation only, so the approximation is acceptable.
    /// - Per-column `min_value` / `max_value`: `Exact` when all stripes have
    ///   stats for that column; `Absent` if any stripe is missing stats.
    /// - Per-column `null_count`: same rule as min/max.
    /// - `distinct_count` and `sum_value`: always `Absent` (not tracked).
    fn statistics(&self) -> Option<Statistics> {
        Some(self.helium_statistics.clone())
    }
}

// ---------------------------------------------------------------------------
// Statistics helper
// ---------------------------------------------------------------------------

/// Aggregate per-stripe physical-leaf stats for one logical column into a
/// single DataFusion [`ColumnStatistics`].
///
/// Rules (all-or-nothing):
/// - If **every** stripe has stats for the relevant leaf → `Exact`.
/// - If **any** stripe is missing stats (or the column type is unsupported
///   for min/max, e.g. Struct/List/Map) → `Absent`.
///
/// This is conservative: partial stats are worse than no stats for the
/// DataFusion planner (it might make wrong cost-model assumptions).
fn build_column_statistics(
    col_name: &str,
    logical_type: &crate::LogicalType,
    per_stripe: Option<&Vec<Vec<crate::PhysicalColumnStats>>>,
    stripe_count: usize,
) -> ColumnStatistics {
    use crate::core::schema::LogicalType;
    use datafusion::common::ScalarValue;

    // Determine which physical leaf holds meaningful min/max data.
    // This mirrors `LeafSelection::leaf_indices` in `pruning.rs`.
    let (data_leaf_idx, null_count_leaf_idx): (usize, Option<usize>) = match logical_type {
        LogicalType::Primitive { .. } => (0, None),
        LogicalType::Utf8 | LogicalType::Binary => (1, None),
        LogicalType::Nullable { inner } => match inner.as_ref() {
            LogicalType::Primitive { .. } => (1, Some(0)),
            LogicalType::Utf8 | LogicalType::Binary => (2, Some(0)),
            LogicalType::Date { .. } | LogicalType::Datetime { .. } => (1, Some(0)),
            LogicalType::Decimal128 { .. } => (1, Some(0)),
            _ => return ColumnStatistics::new_unknown(),
        },
        LogicalType::Date { .. } | LogicalType::Datetime { .. } => (0, None),
        LogicalType::Decimal128 { .. } => (0, None),
        _ => return ColumnStatistics::new_unknown(),
    };

    let per_stripe = match per_stripe {
        Some(s) => s,
        None => return ColumnStatistics::new_unknown(),
    };

    // Degenerate: no stripes (empty file).
    if stripe_count == 0 {
        return ColumnStatistics::new_unknown();
    }

    // Collect per-stripe scalars — bail out on any missing value.
    let mut min_scalar: Option<ScalarValue> = None;
    let mut max_scalar: Option<ScalarValue> = None;
    let mut total_null_count: u64 = 0;
    let mut null_count_valid = null_count_leaf_idx.is_some();

    for (stripe_idx, stripe_leaves) in per_stripe.iter().enumerate() {
        let data_leaf = match stripe_leaves.get(data_leaf_idx) {
            Some(l) => l,
            None => {
                // This stripe is missing the leaf entirely — stats are absent.
                let _ = col_name; // suppress unused-var hint in release
                let _ = stripe_idx;
                return ColumnStatistics::new_unknown();
            }
        };

        // Min value for this stripe.
        match &data_leaf.min {
            None => return ColumnStatistics::new_unknown(),
            Some(mmv) => {
                let sv = match min_max_to_scalar_typed(mmv, logical_type, data_leaf_idx) {
                    Some(sv) => sv,
                    None => return ColumnStatistics::new_unknown(),
                };
                min_scalar = Some(match min_scalar {
                    None => sv.clone(),
                    Some(existing) => scalar_min(existing, sv),
                });
            }
        }

        // Max value for this stripe.
        match &data_leaf.max {
            None => return ColumnStatistics::new_unknown(),
            Some(mmv) => {
                let sv = match min_max_to_scalar_typed(mmv, logical_type, data_leaf_idx) {
                    Some(sv) => sv,
                    None => return ColumnStatistics::new_unknown(),
                };
                max_scalar = Some(match max_scalar {
                    None => sv.clone(),
                    Some(existing) => scalar_max(existing, sv),
                });
            }
        }

        // Null count.
        if null_count_valid && let Some(nc_idx) = null_count_leaf_idx {
            match stripe_leaves.get(nc_idx).and_then(|l| l.null_count) {
                Some(nc) => total_null_count += nc,
                None => null_count_valid = false,
            }
        }
    }

    let min_value = min_scalar
        .map(Precision::Exact)
        .unwrap_or(Precision::Absent);
    let max_value = max_scalar
        .map(Precision::Exact)
        .unwrap_or(Precision::Absent);
    let null_count = if null_count_valid && null_count_leaf_idx.is_some() {
        Precision::Exact(total_null_count as usize)
    } else {
        Precision::Absent
    };

    ColumnStatistics {
        null_count,
        max_value,
        min_value,
        sum_value: Precision::Absent,
        distinct_count: Precision::Absent,
    }
}

/// Convert a [`crate::MinMaxValue`] to a [`datafusion::common::ScalarValue`]
/// taking the column's `LogicalType` into account to produce correctly-typed
/// scalars (e.g. `Date32` for `Date { Days }`).
fn min_max_to_scalar_typed(
    value: &crate::MinMaxValue,
    logical_type: &crate::LogicalType,
    _data_leaf_idx: usize,
) -> Option<datafusion::common::ScalarValue> {
    use crate::MinMaxValue;
    use crate::core::schema::{DateUnit, LogicalType, TimeUnit};
    use datafusion::common::ScalarValue;

    // Unwrap one layer of Nullable so the match below works uniformly.
    let effective_type = match logical_type {
        LogicalType::Nullable { inner } => inner.as_ref(),
        other => other,
    };

    match effective_type {
        LogicalType::Date {
            unit: DateUnit::Days,
        } => match value {
            // New typed variant.
            MinMaxValue::Date { value: v, .. } => Some(ScalarValue::Date32(Some(*v as i32))),
            // Legacy raw-integer variant (older files).
            MinMaxValue::I32(v) => Some(ScalarValue::Date32(Some(*v))),
            _ => None,
        },
        LogicalType::Date {
            unit: DateUnit::Millis,
        } => match value {
            MinMaxValue::Date { value: v, .. } => Some(ScalarValue::Date64(Some(*v))),
            MinMaxValue::I64(v) => Some(ScalarValue::Date64(Some(*v))),
            _ => None,
        },
        LogicalType::Datetime { unit, timezone } => {
            let tz: Option<Arc<str>> = timezone.as_ref().map(|s| Arc::from(s.as_str()));
            let raw = match value {
                MinMaxValue::Datetime { value: v, .. } => Some(*v),
                MinMaxValue::I64(v) => Some(*v),
                _ => None,
            };
            raw.map(|v| match unit {
                TimeUnit::Seconds => ScalarValue::TimestampSecond(Some(v), tz),
                TimeUnit::Millis => ScalarValue::TimestampMillisecond(Some(v), tz),
                TimeUnit::Micros => ScalarValue::TimestampMicrosecond(Some(v), tz),
                TimeUnit::Nanos => ScalarValue::TimestampNanosecond(Some(v), tz),
            })
        }
        _ => min_max_to_scalar(value),
    }
}

/// Return the lex-min of two [`datafusion::common::ScalarValue`]s.
///
/// If the comparison fails (type mismatch), returns `a` conservatively.
fn scalar_min(
    a: datafusion::common::ScalarValue,
    b: datafusion::common::ScalarValue,
) -> datafusion::common::ScalarValue {
    match a.partial_cmp(&b) {
        Some(std::cmp::Ordering::Greater) => b,
        _ => a,
    }
}

/// Return the lex-max of two [`datafusion::common::ScalarValue`]s.
///
/// If the comparison fails (type mismatch), returns `a` conservatively.
fn scalar_max(
    a: datafusion::common::ScalarValue,
    b: datafusion::common::ScalarValue,
) -> datafusion::common::ScalarValue {
    match a.partial_cmp(&b) {
        Some(std::cmp::Ordering::Less) => b,
        _ => a,
    }
}
