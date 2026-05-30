//! Custom [`ExecutionPlan`] that scans a Helium `.he` file stripe by stripe.
//!
//! Each stripe is exposed as a separate partition so DataFusion can (in
//! principle) schedule them in parallel. Because Helium I/O is currently
//! synchronous, we wrap the blocking read inside
//! `tokio::task::block_in_place` when executing.
//!
//! ## Projection pushdown
//!
//! When DataFusion supplies a `projection` (a list of column indices from the
//! full schema), only those columns are read from disk — Helium's existing
//! column-pruning machinery handles the rest.
//!
//! ## Zero-column projection
//!
//! DataFusion sometimes requests a zero-column projection (e.g., for
//! `COUNT(*)`). In this case we build a zero-column `RecordBatch` with an
//! explicit row count taken from the pre-computed per-stripe row counts.

use std::any::Any;
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

use ::datafusion::common::Statistics;
use ::datafusion::error::Result as DfResult;
use ::datafusion::execution::TaskContext;
use ::datafusion::physical_expr::{EquivalenceProperties, Partitioning};
use ::datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use ::datafusion::physical_plan::memory::MemoryStream;
use ::datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
};
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatchOptions;

use crate::core::error::HeliumError;
use crate::core::registry::CoderRegistry;
use crate::core::schema::Schema as HeliumSchema;

/// Custom [`ExecutionPlan`] that reads one Helium stripe per partition.
///
/// Each call to [`execute`](HeliumExec::execute) (with `partition = stripe_idx`)
/// opens the `.he` file, reads only the requested columns for that stripe, and
/// returns a single-batch [`SendableRecordBatchStream`].
///
/// File I/O is synchronous; we use [`tokio::task::block_in_place`] to avoid
/// blocking the Tokio executor.
///
/// # Stripe pruning
///
/// When `keep_stripes` is `Some(indices)`, partitions whose stripe index is
/// not in `keep_stripes` return an **empty** `RecordBatch` immediately,
/// skipping all disk I/O for that stripe.  Partitioning is unchanged (still
/// `stripe_count` partitions) so DataFusion's scheduler remains consistent;
/// pruned partitions just emit zero rows very quickly.
#[derive(Debug, Clone)]
pub struct HeliumExec {
    /// Path to the `.he` file.
    pub(crate) path: PathBuf,
    /// Arrow schema for the *projected* output (subset of full schema).
    pub(crate) projected_schema: SchemaRef,
    /// `None` = all columns; `Some(indices)` = only these, in order.
    pub(crate) projection: Option<Vec<usize>>,
    /// Optional row limit pushed down from the query planner.
    pub(crate) limit: Option<usize>,
    /// Total number of stripes in the file (= number of partitions).
    pub(crate) stripe_count: usize,
    /// Per-stripe row counts — used to build zero-column batches for COUNT(*).
    pub(crate) stripe_row_counts: Vec<u64>,
    /// Stripe indices that survived predicate pruning.
    ///
    /// `None` means "keep all stripes" (no pruning applied or no filters
    /// provided).  `Some(indices)` lists stripe indices to actually read;
    /// any partition not in this set returns an empty batch immediately.
    pub(crate) keep_stripes: Option<Vec<usize>>,
    /// Pre-loaded Helium schema.  Passed as an in-memory resolver so that v6
    /// (catalog-mode) files can be opened without a catalog directory at
    /// read time — the schema was already resolved at `HeliumTableProvider`
    /// construction time.
    pub(crate) helium_schema: Arc<HeliumSchema>,
    /// Cached plan properties (partitioning, emission type, boundedness).
    pub(crate) properties: PlanProperties,
    /// File-wide statistics propagated from [`crate::sql::HeliumTableProvider`].
    ///
    /// Projected via `Statistics::project` when a projection is active so
    /// DataFusion receives per-column stats in the projected order.
    pub(crate) helium_statistics: Statistics,
}

impl HeliumExec {
    /// Build a new `HeliumExec`.
    ///
    /// `full_schema` is the Arrow schema for *all* columns in the file.
    /// `projection` narrows which columns are emitted (indices into `full_schema`).
    /// `stripe_row_counts` holds the row count for each stripe; must have length
    /// equal to `stripe_count`.
    /// `keep_stripes` is the set of stripe indices that survived predicate
    /// pruning; `None` means keep all stripes.
    /// `helium_schema` is the pre-loaded Helium schema; used as an in-memory
    /// resolver so v6 files can be read without a catalog directory.
    /// `helium_statistics` is the file-wide statistics pre-computed by
    /// `HeliumTableProvider`; projected as needed.
    // Each argument is a distinct piece of state needed by the plan node;
    // introducing a builder struct would add indirection without simplifying
    // call sites — the caller (HeliumTableProvider::scan) has all pieces at hand.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        path: PathBuf,
        full_schema: SchemaRef,
        projection: Option<Vec<usize>>,
        limit: Option<usize>,
        stripe_count: usize,
        stripe_row_counts: Vec<u64>,
        keep_stripes: Option<Vec<usize>>,
        helium_schema: Arc<HeliumSchema>,
        helium_statistics: Statistics,
    ) -> Self {
        let projected_schema: SchemaRef = match &projection {
            None => Arc::clone(&full_schema),
            Some(indices) => {
                let fields: Vec<_> = indices
                    .iter()
                    .map(|&i| full_schema.field(i).clone())
                    .collect();
                Arc::new(arrow::datatypes::Schema::new(fields))
            }
        };

        // Project the file-wide statistics to match the output schema.
        let projected_statistics = helium_statistics.clone().project(projection.as_ref());

        let eq_props = EquivalenceProperties::new(Arc::clone(&projected_schema));
        // Use at least 1 partition even if the file has 0 stripes (degenerate case).
        let partitioning = Partitioning::UnknownPartitioning(stripe_count.max(1));
        let plan_props = PlanProperties::new(
            eq_props,
            partitioning,
            EmissionType::Incremental,
            Boundedness::Bounded,
        );

        Self {
            path,
            projected_schema,
            projection,
            limit,
            stripe_count,
            stripe_row_counts,
            keep_stripes,
            helium_schema,
            properties: plan_props,
            helium_statistics: projected_statistics,
        }
    }

    /// The stripe-level pruning mask set at scan time.
    ///
    /// `None` = all stripes are kept.  `Some(indices)` = only these stripe
    /// indices will be actually read; others return empty batches immediately.
    ///
    /// Exposed for testing and observability (e.g., asserting pruning
    /// effectiveness in integration tests).
    pub fn keep_stripes(&self) -> Option<&[usize]> {
        self.keep_stripes.as_deref()
    }
}

impl DisplayAs for HeliumExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "HeliumExec: path={}, stripes={}, projection={:?}, limit={:?}",
            self.path.display(),
            self.stripe_count,
            self.projection,
            self.limit,
        )
    }
}

impl ExecutionPlan for HeliumExec {
    fn name(&self) -> &str {
        "HeliumExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        // Leaf node — no children.
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        if children.is_empty() {
            Ok(self)
        } else {
            Err(::datafusion::error::DataFusionError::Internal(
                "HeliumExec has no children".into(),
            ))
        }
    }

    /// Return the file-wide statistics projected to match the output schema.
    ///
    /// DataFusion uses these to optimize aggregate queries: an `Exact` row
    /// count lets the planner constant-fold `COUNT(*)` without scanning data;
    /// exact per-column `min`/`max` values enable metadata-only `MIN(col)` /
    /// `MAX(col)` results.
    fn statistics(&self) -> DfResult<Statistics> {
        Ok(self.helium_statistics.clone())
    }

    /// Execute a single partition (= one Helium stripe).
    ///
    /// Helium I/O is synchronous. We use `tokio::task::block_in_place` to
    /// avoid blocking the Tokio executor thread. The resulting `RecordBatch`
    /// is wrapped into a `MemoryStream` (one batch per partition).
    ///
    /// If `keep_stripes` was set and `partition` is not in the keep list,
    /// returns an **empty** `RecordBatch` immediately (no disk I/O).
    fn execute(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> DfResult<SendableRecordBatchStream> {
        if partition >= self.stripe_count.max(1) {
            return Err(::datafusion::error::DataFusionError::Internal(format!(
                "HeliumExec: partition {partition} out of range (stripe_count={})",
                self.stripe_count
            )));
        }

        let projected_schema = Arc::clone(&self.projected_schema);

        // If stripe-level pruning is active, check whether this partition was
        // kept. If not, emit an empty batch immediately — no disk I/O needed.
        if let Some(ref keep) = self.keep_stripes
            && !keep.contains(&partition)
        {
            let empty_batch = arrow::record_batch::RecordBatch::try_new_with_options(
                Arc::clone(&projected_schema),
                // One null-array per projected column, length 0.
                (0..projected_schema.fields().len())
                    .map(|i| arrow::array::new_empty_array(projected_schema.field(i).data_type()))
                    .collect(),
                &arrow::record_batch::RecordBatchOptions::new().with_row_count(Some(0)),
            )
            .map_err(|e| {
                ::datafusion::error::DataFusionError::Internal(format!(
                    "HeliumExec: empty batch for pruned partition {partition}: {e}"
                ))
            })?;
            let stream =
                MemoryStream::try_new(vec![empty_batch], Arc::clone(&projected_schema), None)?;
            return Ok(Box::pin(stream));
        }

        // Clone state for the blocking closure.
        let path = self.path.clone();
        let projection = self.projection.clone();
        let limit = self.limit;
        let stripe_idx = partition;
        // Row count for this stripe (used for zero-column projections).
        let stripe_rows = self.stripe_row_counts.get(stripe_idx).copied().unwrap_or(0);
        let helium_schema = Arc::clone(&self.helium_schema);

        // Run the synchronous Helium I/O on a blocking thread to avoid
        // starving the async executor.
        let batch = tokio::task::block_in_place(|| {
            read_stripe_batch(
                &path,
                stripe_idx,
                &projection,
                &projected_schema,
                limit,
                stripe_rows,
                &helium_schema,
            )
            .map_err(|e| ::datafusion::error::DataFusionError::External(Box::new(e)))
        })?;

        let stream = MemoryStream::try_new(vec![batch], Arc::clone(&projected_schema), None)?;
        Ok(Box::pin(stream))
    }
}

// ---------------------------------------------------------------------------
// Synchronous read helpers
// ---------------------------------------------------------------------------

/// Read one stripe of a `.he` file, applying optional column projection and
/// row limit, and return an Arrow `RecordBatch`.
///
/// `helium_schema` is pre-loaded from the `HeliumTableProvider`; it is used
/// as an in-memory resolver so v6 (catalog-mode) files can be read without
/// a catalog directory on disk.
///
/// When `projection` is `Some([])` (zero columns requested by DataFusion for
/// e.g. `COUNT(*)`), builds a zero-column batch with `stripe_rows` rows using
/// [`RecordBatchOptions`].
fn read_stripe_batch(
    path: &PathBuf,
    stripe_idx: usize,
    projection: &Option<Vec<usize>>,
    projected_schema: &SchemaRef,
    limit: Option<usize>,
    stripe_rows: u64,
    helium_schema: &HeliumSchema,
) -> Result<arrow::record_batch::RecordBatch, HeliumError> {
    use crate::arrow::to_arrow_array;
    use std::fs::File;

    // Determine which Helium column indices to read.
    let col_indices: Vec<usize> = match projection {
        None => (0..helium_schema.columns.len()).collect(),
        Some(proj) => proj.clone(),
    };

    // Zero-column projection (e.g., COUNT(*)) — return a batch with explicit
    // row count and no columns.
    if col_indices.is_empty() {
        let row_count = match limit {
            Some(lim) => (stripe_rows as usize).min(lim),
            None => stripe_rows as usize,
        };
        return arrow::record_batch::RecordBatch::try_new_with_options(
            Arc::clone(projected_schema),
            vec![],
            &RecordBatchOptions::new().with_row_count(Some(row_count)),
        )
        .map_err(|e| HeliumError::Format(format!("RecordBatch::try_new (zero-col): {e}")));
    }

    let file = File::open(path).map_err(|e| {
        HeliumError::Format(format!("HeliumExec: cannot open {}: {e}", path.display()))
    })?;
    let registry = CoderRegistry::default();
    // Use an in-memory resolver that returns the pre-loaded schema.
    // This makes v6 (catalog-mode) files work without a catalog directory
    // on the reader side — the schema was already resolved at
    // `HeliumTableProvider` construction time.
    let schema_clone = helium_schema.clone();
    let mut reader = crate::HeliumReader::new_with_resolver(file, &registry, move |_hash| {
        Ok(schema_clone.clone())
    })?;

    let mut arrays: Vec<arrow::array::ArrayRef> = Vec::with_capacity(col_indices.len());

    for &col_idx in &col_indices {
        let spec = &helium_schema.columns[col_idx];
        let col = reader.read_column_at_stripe(&spec.name, stripe_idx)?;
        let arr = to_arrow_array(&col, &spec.logical_type).map_err(|e| HeliumError::Schema {
            column: spec.name.clone(),
            reason: format!("Arrow conversion: {e}"),
        })?;

        // Apply limit: slice the array if needed.
        let arr = if let Some(lim) = limit {
            if arr.len() > lim {
                arr.slice(0, lim)
            } else {
                arr
            }
        } else {
            arr
        };

        arrays.push(arr);
    }

    arrow::record_batch::RecordBatch::try_new(Arc::clone(projected_schema), arrays)
        .map_err(|e| HeliumError::Format(format!("RecordBatch::try_new: {e}")))
}
