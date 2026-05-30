//! Stripe-level predicate pushdown tests for Helium + DataFusion.
//!
//! Two categories of tests:
//!
//! **A. Correctness** — verifies that pruning does not drop matching rows.
//!    Each test creates a multi-stripe `.he` file with a controlled value
//!    distribution, runs a SQL query with a WHERE clause, and asserts the
//!    exact result set.
//!
//! **B. Pruning effectiveness** — verifies that stripes are actually skipped.
//!    We inspect `HeliumExec::keep_stripes` after `scan()` to confirm the
//!    pruning mask excludes the expected stripes, rather than relying purely
//!    on result correctness.
//!
//! The approach for pruning observation is direct inspection of the
//! `keep_stripes` field on `HeliumExec` (cast via `ExecutionPlan::as_any`).
//! This avoids adding any runtime counters to production code.

#![cfg(feature = "datafusion")]

use std::path::Path;
use std::sync::Arc;

use arrow::array::{Array, Int32Array, Int64Array, StringArray};
use datafusion::catalog::TableProvider;
use datafusion::common::DFSchema;
use datafusion::logical_expr::Expr;
use datafusion::physical_expr::create_physical_expr;
use datafusion::physical_optimizer::pruning::PruningPredicate;
use datafusion::prelude::{SessionContext, col, lit};
use helium::sql::{HeliumExec, HeliumTableProvider};
use helium::{
    CoderRegistry, CoderSpec, ColumnData, ColumnSpec, DataType, HeliumWriter, LogicalColumn,
    LogicalType, Schema,
};
use tempfile::NamedTempFile;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

macro_rules! tokio_run {
    ($body:expr) => {{
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("tokio runtime")
            .block_on($body)
    }};
}

fn int64_pipeline() -> Vec<CoderSpec> {
    vec![
        CoderSpec::new("delta"),
        CoderSpec::new("leb128"),
        CoderSpec::new("zstd"),
    ]
}

fn int32_pipeline() -> Vec<CoderSpec> {
    vec![
        CoderSpec::new("delta"),
        CoderSpec::new("leb128"),
        CoderSpec::new("zstd"),
    ]
}

fn utf8_pipeline() -> Vec<Vec<CoderSpec>> {
    vec![
        vec![
            CoderSpec::new("delta"),
            CoderSpec::new("leb128"),
            CoderSpec::new("zstd"),
        ],
        vec![CoderSpec::new("zstd")],
    ]
}

/// Write a `.he` file with multiple stripes, one per `stripe_values` entry.
/// Each stripe holds a single I64 column named `"val"`.
fn write_i64_stripes(path: &Path, stripe_values: &[Vec<i64>]) {
    let schema = Schema::new(vec![ColumnSpec::new(
        "val",
        LogicalType::Primitive {
            data_type: DataType::I64,
        },
        vec![int64_pipeline()],
    )]);
    let registry = CoderRegistry::default();
    let file = std::fs::File::create(path).expect("create file");
    let mut w = HeliumWriter::new(file, schema, &registry).expect("writer");
    for values in stripe_values {
        w.write_column(
            "val",
            LogicalColumn::Primitive(ColumnData::I64(values.clone())),
        )
        .expect("write_column");
        w.finish_stripe().expect("finish_stripe");
    }
    w.finish().expect("finish");
}

/// Write a `.he` file with multiple stripes, one per `stripe_values` entry.
/// Each stripe holds a single I32 column named `"val"`.
fn write_i32_stripes(path: &Path, stripe_values: &[Vec<i32>]) {
    let schema = Schema::new(vec![ColumnSpec::new(
        "val",
        LogicalType::Primitive {
            data_type: DataType::I32,
        },
        vec![int32_pipeline()],
    )]);
    let registry = CoderRegistry::default();
    let file = std::fs::File::create(path).expect("create file");
    let mut w = HeliumWriter::new(file, schema, &registry).expect("writer");
    for values in stripe_values {
        w.write_column(
            "val",
            LogicalColumn::Primitive(ColumnData::I32(values.clone())),
        )
        .expect("write_column");
        w.finish_stripe().expect("finish_stripe");
    }
    w.finish().expect("finish");
}

/// Collect all `i64` values from column 0 across all batches, in order.
fn collect_i64(batches: &[arrow::record_batch::RecordBatch]) -> Vec<i64> {
    let mut out = Vec::new();
    for b in batches {
        if b.num_rows() == 0 {
            continue;
        }
        let arr = b
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("Int64Array");
        for i in 0..arr.len() {
            out.push(arr.value(i));
        }
    }
    out
}

/// Collect all `i32` values from column 0 across all batches, in order.
fn collect_i32(batches: &[arrow::record_batch::RecordBatch]) -> Vec<i32> {
    let mut out = Vec::new();
    for b in batches {
        if b.num_rows() == 0 {
            continue;
        }
        let arr = b
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("Int32Array");
        for i in 0..arr.len() {
            out.push(arr.value(i));
        }
    }
    out
}

/// Collect all `String` values from column 0 across all batches.
fn collect_str(batches: &[arrow::record_batch::RecordBatch]) -> Vec<String> {
    let mut out = Vec::new();
    for b in batches {
        if b.num_rows() == 0 {
            continue;
        }
        let arr = b
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("StringArray");
        for i in 0..arr.len() {
            out.push(arr.value(i).to_string());
        }
    }
    out
}

/// Total row count across all batches.
fn total_rows(batches: &[arrow::record_batch::RecordBatch]) -> usize {
    batches.iter().map(|b| b.num_rows()).sum()
}

/// Run a SQL query on `path` and collect all result batches.
async fn run_sql(path: &Path, query: &str) -> Vec<arrow::record_batch::RecordBatch> {
    let provider = HeliumTableProvider::try_new(path).expect("HeliumTableProvider::try_new");
    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::new(provider))
        .expect("register_table");
    let df = ctx.sql(query).await.expect("sql parse/plan");
    df.collect().await.expect("collect")
}

/// Extract the `keep_stripes` field from the first `HeliumExec` node in the
/// logical plan built by `HeliumTableProvider::scan`.  This is the primary
/// mechanism for asserting *pruning effectiveness* without a runtime counter.
///
/// Returns `Some(keep_stripes)` when pruning is active, `None` when no pruning
/// mask was set (all stripes kept).
async fn get_keep_stripes(path: &Path, filter: Expr) -> Option<Vec<usize>> {
    let provider = HeliumTableProvider::try_new(path).expect("try_new");
    let ctx = SessionContext::new();

    // Use the session's state to call scan() directly with the filter.
    let state = ctx.state();
    let plan = TableProvider::scan(&provider, &state, None, &[filter], None)
        .await
        .expect("scan");

    // Downcast to HeliumExec.
    let exec = plan
        .as_any()
        .downcast_ref::<HeliumExec>()
        .expect("HeliumExec");
    exec.keep_stripes().map(|s| s.to_vec())
}

// ---------------------------------------------------------------------------
// A. Correctness tests
// ---------------------------------------------------------------------------

/// Test A1: WHERE val > 1000 returns only rows from the high-value stripe.
///
/// Stripe 0: 0..100, stripe 1: 500..600, stripe 2: 5000..6000.
/// The predicate should keep only stripe 2.
#[test]
fn correctness_a1_greater_than() {
    let tmp = NamedTempFile::new().unwrap();
    let stripe0: Vec<i64> = (0..100).collect();
    let stripe1: Vec<i64> = (500..600).collect();
    let stripe2: Vec<i64> = (5000..6000).collect();
    write_i64_stripes(tmp.path(), &[stripe0, stripe1, stripe2]);

    let batches = tokio_run!(run_sql(tmp.path(), "SELECT val FROM t WHERE val > 1000"));
    let mut vals = collect_i64(&batches);
    vals.sort_unstable();
    assert_eq!(
        vals.len(),
        1000,
        "should return exactly stripe 2's 1000 rows"
    );
    assert_eq!(vals[0], 5000);
    assert_eq!(vals[999], 5999);
}

/// Test A2: WHERE val BETWEEN 50 AND 70 returns only matching rows from stripe 0.
#[test]
fn correctness_a2_between() {
    let tmp = NamedTempFile::new().unwrap();
    let stripe0: Vec<i64> = (0..100).collect();
    let stripe1: Vec<i64> = (500..600).collect();
    let stripe2: Vec<i64> = (5000..6000).collect();
    write_i64_stripes(tmp.path(), &[stripe0, stripe1, stripe2]);

    let batches = tokio_run!(run_sql(
        tmp.path(),
        "SELECT val FROM t WHERE val BETWEEN 50 AND 70"
    ));
    let mut vals = collect_i64(&batches);
    vals.sort_unstable();
    // 50..=70 inclusive = 21 values
    assert_eq!(vals.len(), 21);
    assert_eq!(vals[0], 50);
    assert_eq!(vals[20], 70);
}

/// Test A3: WHERE str_col = 'banana' with stripes whose lexicographic min/max
/// exclude 'banana' for two of three stripes.
#[test]
fn correctness_a3_string_equality() {
    let tmp = NamedTempFile::new().unwrap();
    // Stripe 0: words starting with 'a' (lexicographically below 'banana')
    // Stripe 1: words that include 'banana' (lex range spans 'banana')
    // Stripe 2: words starting with 'c' (lexicographically above 'banana')
    let schema = Schema::new(vec![ColumnSpec::new(
        "val",
        LogicalType::Utf8,
        utf8_pipeline(),
    )]);
    let registry = CoderRegistry::default();
    let file = std::fs::File::create(tmp.path()).expect("create file");
    let mut w = HeliumWriter::new(file, schema, &registry).expect("writer");

    // Stripe 0: apple, avocado — max is 'avocado' < 'banana'
    w.write_column(
        "val",
        LogicalColumn::Utf8(vec!["apple".to_string(), "avocado".to_string()]),
    )
    .expect("write stripe 0");
    w.finish_stripe().expect("finish stripe 0");

    // Stripe 1: banana, blueberry — spans 'banana'
    w.write_column(
        "val",
        LogicalColumn::Utf8(vec!["banana".to_string(), "blueberry".to_string()]),
    )
    .expect("write stripe 1");
    w.finish_stripe().expect("finish stripe 1");

    // Stripe 2: cherry, coconut — min is 'cherry' > 'banana'
    w.write_column(
        "val",
        LogicalColumn::Utf8(vec!["cherry".to_string(), "coconut".to_string()]),
    )
    .expect("write stripe 2");
    w.finish().expect("finish");

    let batches = tokio_run!(run_sql(
        tmp.path(),
        "SELECT val FROM t WHERE val = 'banana'"
    ));
    let vals = collect_str(&batches);
    assert_eq!(vals, vec!["banana"]);
}

/// Test A4: WHERE val IS NULL returns 0 rows for a non-nullable column.
#[test]
fn correctness_a4_is_null_non_nullable() {
    let tmp = NamedTempFile::new().unwrap();
    let stripe0: Vec<i64> = (0..100).collect();
    write_i64_stripes(tmp.path(), &[stripe0]);

    let batches = tokio_run!(run_sql(tmp.path(), "SELECT val FROM t WHERE val IS NULL"));
    assert_eq!(total_rows(&batches), 0);
}

/// Test A5: Compound predicate WHERE val > 5 AND val < 10 works correctly.
#[test]
fn correctness_a5_compound_predicate() {
    let tmp = NamedTempFile::new().unwrap();
    let stripe0: Vec<i32> = (0..20).collect();
    let stripe1: Vec<i32> = (100..120).collect();
    write_i32_stripes(tmp.path(), &[stripe0, stripe1]);

    let batches = tokio_run!(run_sql(
        tmp.path(),
        "SELECT val FROM t WHERE val > 5 AND val < 10"
    ));
    let mut vals = collect_i32(&batches);
    vals.sort_unstable();
    // 6, 7, 8, 9 from stripe 0 only (stripe 1's range 100..120 is entirely above 10)
    assert_eq!(vals, vec![6, 7, 8, 9]);
}

/// Test A6: Predicate that excludes ALL stripes returns 0 rows.
#[test]
fn correctness_a6_all_stripes_excluded() {
    let tmp = NamedTempFile::new().unwrap();
    let stripe0: Vec<i64> = (0..100).collect();
    let stripe1: Vec<i64> = (500..600).collect();
    write_i64_stripes(tmp.path(), &[stripe0, stripe1]);

    let batches = tokio_run!(run_sql(
        tmp.path(),
        "SELECT val FROM t WHERE val > 999999999"
    ));
    assert_eq!(total_rows(&batches), 0, "all stripes should be pruned");
}

// ---------------------------------------------------------------------------
// B. Pruning effectiveness tests
// ---------------------------------------------------------------------------

/// Test B1: Verify stripe 0 and 1 are pruned when only stripe 2 matches.
///
/// Uses `get_keep_stripes` to directly inspect the pruning mask on `HeliumExec`.
#[test]
fn pruning_b1_keep_stripes_inspection() {
    let tmp = NamedTempFile::new().unwrap();
    let stripe0: Vec<i64> = (0..100).collect();
    let stripe1: Vec<i64> = (500..600).collect();
    let stripe2: Vec<i64> = (5000..6000).collect();
    write_i64_stripes(tmp.path(), &[stripe0, stripe1, stripe2]);

    // WHERE val > 1000 → only stripe 2 can match (stripe 0 max=99, stripe 1 max=599)
    let filter = col("val").gt(lit(1000i64));
    let keep = tokio_run!(get_keep_stripes(tmp.path(), filter));

    // The keep list should only contain stripe 2 (index 2).
    let keep = keep.expect("pruning_stats should produce Some(keep_stripes)");
    assert_eq!(keep, vec![2], "only stripe 2 should survive pruning");
}

/// Test B2: No pruning when filter can't rule out any stripe.
#[test]
fn pruning_b2_no_stripes_pruned() {
    let tmp = NamedTempFile::new().unwrap();
    let stripe0: Vec<i64> = (0..100).collect();
    let stripe1: Vec<i64> = (500..600).collect();
    write_i64_stripes(tmp.path(), &[stripe0, stripe1]);

    // WHERE val > 0 → both stripes potentially match (stripe 0 max=99, stripe 1 max=599)
    // Stripe 0 has values 1..99 > 0, so it can't be pruned.
    let filter = col("val").gt(lit(0i64));
    let keep = tokio_run!(get_keep_stripes(tmp.path(), filter));
    // Both stripes should be kept.
    let keep = keep.expect("should have keep list");
    assert_eq!(keep.len(), 2, "both stripes should survive");
    assert!(keep.contains(&0));
    assert!(keep.contains(&1));
}

/// Test B3: All stripes pruned when predicate is impossible.
#[test]
fn pruning_b3_all_pruned() {
    let tmp = NamedTempFile::new().unwrap();
    let stripe0: Vec<i64> = (0..100).collect();
    let stripe1: Vec<i64> = (500..600).collect();
    write_i64_stripes(tmp.path(), &[stripe0, stripe1]);

    // WHERE val > 1000000 — neither stripe has values that high
    let filter = col("val").gt(lit(1_000_000i64));
    let keep = tokio_run!(get_keep_stripes(tmp.path(), filter));
    // Should be empty (all pruned).
    let keep = keep.expect("should have keep list");
    assert!(keep.is_empty(), "all stripes should be pruned: {keep:?}");
}

/// Test B4: Direct PruningStatistics API — verify min/max arrays are correct.
///
/// This tests `HeliumPruningStatistics` in isolation from DataFusion's planner.
#[test]
fn pruning_b4_statistics_api() {
    use datafusion::common::Column;
    use datafusion::physical_optimizer::pruning::PruningStatistics;

    let tmp = NamedTempFile::new().unwrap();
    let stripe0: Vec<i64> = (0..100).collect(); // min=0,  max=99
    let stripe1: Vec<i64> = (500..600).collect(); // min=500, max=599
    let stripe2: Vec<i64> = (5000..6000).collect(); // min=5000, max=5999
    write_i64_stripes(tmp.path(), &[stripe0, stripe1, stripe2]);

    let provider = HeliumTableProvider::try_new(tmp.path()).expect("try_new");

    // Access the pruning_stats field via the public accessor.
    let stats = provider.pruning_stats();
    let col = Column::from_name("val");

    // min_values should be an Int64Array with [0, 500, 5000]
    let min_arr = stats.min_values(&col).expect("min_values should be Some");
    assert_eq!(min_arr.len(), 3);
    let min_arr = min_arr
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("Int64Array");
    assert_eq!(min_arr.value(0), 0);
    assert_eq!(min_arr.value(1), 500);
    assert_eq!(min_arr.value(2), 5000);

    // max_values should be [99, 599, 5999]
    let max_arr = stats.max_values(&col).expect("max_values should be Some");
    let max_arr = max_arr
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("Int64Array");
    assert_eq!(max_arr.value(0), 99);
    assert_eq!(max_arr.value(1), 599);
    assert_eq!(max_arr.value(2), 5999);
}

/// Test B5: PruningPredicate evaluated against HeliumPruningStatistics directly.
///
/// Creates a `PruningPredicate` for `val > 1000` and runs it against the stats.
/// This verifies the full pipeline without a DataFusion query context.
#[test]
fn pruning_b5_predicate_eval() {
    let tmp = NamedTempFile::new().unwrap();
    let stripe0: Vec<i64> = (0..100).collect(); // max=99  < 1000 → prune
    let stripe1: Vec<i64> = (500..600).collect(); // max=599 < 1000 → prune
    let stripe2: Vec<i64> = (5000..6000).collect(); // min=5000 > 1000 → keep
    write_i64_stripes(tmp.path(), &[stripe0, stripe1, stripe2]);

    let provider = HeliumTableProvider::try_new(tmp.path()).expect("try_new");
    let arrow_schema = provider.arrow_schema();
    let stats = provider.pruning_stats();

    // Build the DFSchema and physical expr for `val > 1000`.
    let df_schema = DFSchema::try_from(arrow_schema.as_ref().clone()).expect("DFSchema");
    let filter = col("val").gt(lit(1000i64));
    let exec_props = datafusion::execution::context::ExecutionProps::new();
    let phys_expr = create_physical_expr(&filter, &df_schema, &exec_props).expect("phys_expr");

    let pp = PruningPredicate::try_new(phys_expr, Arc::clone(arrow_schema)).expect("PP");
    let keep = pp.prune(stats).expect("prune");

    assert_eq!(keep, vec![false, false, true]);
}

// ---------------------------------------------------------------------------
// C. Files without stats fall back gracefully
// ---------------------------------------------------------------------------

/// Test C1: A `.he` file written with `with_stats_disabled()` still returns
/// correct query results (no crashes; pruning conservatively keeps all stripes).
#[test]
fn fallback_c1_stats_disabled() {
    let tmp = NamedTempFile::new().unwrap();
    let schema = Schema::new(vec![ColumnSpec::new(
        "val",
        LogicalType::Primitive {
            data_type: DataType::I64,
        },
        vec![int64_pipeline()],
    )]);
    let registry = CoderRegistry::default();
    let file = std::fs::File::create(tmp.path()).expect("create");
    let mut w = HeliumWriter::new(file, schema, &registry)
        .expect("writer")
        .with_stats_disabled();

    for start in [0i64, 500, 5000] {
        w.write_column(
            "val",
            LogicalColumn::Primitive(ColumnData::I64((start..start + 100).collect())),
        )
        .expect("write");
        w.finish_stripe().expect("finish_stripe");
    }
    w.finish().expect("finish");

    // Should return the right rows — no crash, just conservative (no pruning).
    let batches = tokio_run!(run_sql(tmp.path(), "SELECT val FROM t WHERE val > 1000"));
    let mut vals = collect_i64(&batches);
    vals.sort_unstable();
    assert_eq!(vals.len(), 100);
    assert_eq!(vals[0], 5000);
    assert_eq!(vals[99], 5099);
}

// ---------------------------------------------------------------------------
// D. Catalog mode (v6) — same correctness as standard mode
// ---------------------------------------------------------------------------

/// Test D1: Catalog-mode (v6) file produces correct query results with pruning.
#[test]
fn catalog_d1_v6_correctness() {
    use helium::catalog::Catalog;

    let dir = tempfile::tempdir().unwrap();
    let catalog = Catalog::open(dir.path()).expect("open catalog");

    let tmp = NamedTempFile::new().unwrap();
    let schema = Schema::new(vec![ColumnSpec::new(
        "val",
        LogicalType::Primitive {
            data_type: DataType::I64,
        },
        vec![int64_pipeline()],
    )]);
    let registry = CoderRegistry::default();
    let file = std::fs::File::create(tmp.path()).expect("create");

    let mut w = catalog
        .open_writer(file, schema, &registry)
        .expect("open_writer");
    // Two stripes: one below threshold, one above.
    w.write_column(
        "val",
        LogicalColumn::Primitive(ColumnData::I64((0i64..100).collect())),
    )
    .expect("write stripe 0");
    w.finish_stripe().expect("finish_stripe 0");
    w.write_column(
        "val",
        LogicalColumn::Primitive(ColumnData::I64((5000i64..6000).collect())),
    )
    .expect("write stripe 1");
    w.finish().expect("finish");

    // HeliumTableProvider::try_new_with_catalog reads v6 files using the
    // catalog as the resolver for the schema hash.
    let resolver = catalog.resolver();
    let provider = HeliumTableProvider::try_new_with_catalog(tmp.path(), move |h| resolver(h))
        .expect("try_new_with_catalog");

    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::new(provider))
        .expect("register");
    let batches = tokio_run!(async move {
        let df = ctx
            .sql("SELECT val FROM t WHERE val > 1000")
            .await
            .expect("sql");
        df.collect().await.expect("collect")
    });

    let mut vals = collect_i64(&batches);
    vals.sort_unstable();
    assert_eq!(vals.len(), 1000);
    assert_eq!(vals[0], 5000);
}
