//! Tests for DataFusion [`Statistics`] exposed by [`HeliumTableProvider`] and
//! [`HeliumExec`].
//!
//! These tests verify that:
//! 1. `TableProvider::statistics()` returns correct file-wide statistics.
//! 2. `COUNT(*)` is correct (we verify the value; metadata-only optimization
//!    is a bonus but not assertable in a version-agnostic way).
//! 3. `SELECT min(col), max(col)` returns correct values.
//! 4. Files written with `with_stats_disabled()` report `Absent` column stats
//!    but still report an exact `num_rows`.
//! 5. Multi-stripe aggregation computes file-wide min/max correctly.
//! 6. Any stripe missing stats causes `Absent` for that column's stats.

#![cfg(feature = "datafusion")]

use std::path::Path;
use std::sync::Arc;

use arrow::array::Int64Array;
use datafusion::catalog::TableProvider;
use datafusion::common::stats::Precision;
use datafusion::prelude::SessionContext;
use helium::sql::HeliumTableProvider;
use helium::{
    CoderRegistry, CoderSpec, ColumnData, ColumnSpec, DataType, HeliumWriter, LogicalColumn,
    LogicalType, Schema,
};
use tempfile::NamedTempFile;

// ---------------------------------------------------------------------------
// Tokio runtime helper
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

// ---------------------------------------------------------------------------
// Helpers: write test files
// ---------------------------------------------------------------------------

fn simple_pipeline() -> Vec<CoderSpec> {
    vec![
        CoderSpec::new("delta"),
        CoderSpec::new("leb128"),
        CoderSpec::new("zstd"),
    ]
}

fn bytes_pipeline() -> Vec<CoderSpec> {
    vec![CoderSpec::new("zstd")]
}

/// Write a 1-stripe file with 3 columns: i32, i64, utf8.
fn write_3col_file(path: &Path) {
    let schema = Schema::new(vec![
        ColumnSpec::new(
            "id",
            LogicalType::Primitive {
                data_type: DataType::I32,
            },
            vec![simple_pipeline()],
        ),
        ColumnSpec::new(
            "score",
            LogicalType::Primitive {
                data_type: DataType::I64,
            },
            vec![simple_pipeline()],
        ),
        ColumnSpec::new(
            "label",
            LogicalType::Utf8,
            vec![
                vec![
                    CoderSpec::new("delta"),
                    CoderSpec::new("leb128"),
                    CoderSpec::new("zstd"),
                ],
                bytes_pipeline(),
            ],
        ),
    ]);

    let registry = CoderRegistry::default();
    let file = std::fs::File::create(path).expect("create");
    let mut w = HeliumWriter::new(file, schema, &registry).expect("writer");
    w.write_column(
        "id",
        LogicalColumn::Primitive(ColumnData::I32(vec![1, 2, 3, 4, 5])),
    )
    .expect("write id");
    w.write_column(
        "score",
        LogicalColumn::Primitive(ColumnData::I64(vec![10, 20, 30, 40, 50])),
    )
    .expect("write score");
    w.write_column(
        "label",
        LogicalColumn::Utf8(vec![
            "apple".to_string(),
            "banana".to_string(),
            "cherry".to_string(),
            "date".to_string(),
            "elderberry".to_string(),
        ]),
    )
    .expect("write label");
    w.finish().expect("finish");
}

/// Write a multi-stripe i64 file with given per-stripe values.
fn write_i64_stripes(path: &Path, stripes: &[Vec<i64>]) {
    let schema = Schema::new(vec![ColumnSpec::new(
        "val",
        LogicalType::Primitive {
            data_type: DataType::I64,
        },
        vec![simple_pipeline()],
    )]);
    let registry = CoderRegistry::default();
    let file = std::fs::File::create(path).expect("create");
    let mut w = HeliumWriter::new(file, schema, &registry).expect("writer");
    for vals in stripes {
        w.write_column(
            "val",
            LogicalColumn::Primitive(ColumnData::I64(vals.clone())),
        )
        .expect("write val");
        w.finish_stripe().expect("finish_stripe");
    }
    w.finish().expect("finish");
}

/// Write a 1-stripe file with stats disabled.
fn write_stats_disabled(path: &Path) {
    let schema = Schema::new(vec![ColumnSpec::new(
        "x",
        LogicalType::Primitive {
            data_type: DataType::I32,
        },
        vec![simple_pipeline()],
    )]);
    let registry = CoderRegistry::default();
    let file = std::fs::File::create(path).expect("create");
    let mut w = HeliumWriter::new(file, schema, &registry)
        .expect("writer")
        .with_stats_disabled();
    w.write_column(
        "x",
        LogicalColumn::Primitive(ColumnData::I32(vec![1, 2, 3])),
    )
    .expect("write x");
    w.finish().expect("finish");
}

/// Run a SQL query against `path` and collect all batches.
async fn run_sql(path: &Path, query: &str) -> Vec<arrow::record_batch::RecordBatch> {
    let provider = HeliumTableProvider::try_new(path).expect("try_new");
    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::new(provider))
        .expect("register");
    let df = ctx.sql(query).await.expect("sql");
    df.collect().await.expect("collect")
}

/// Total row count across batches.
fn total_rows(batches: &[arrow::record_batch::RecordBatch]) -> usize {
    batches.iter().map(|b| b.num_rows()).sum()
}

// ---------------------------------------------------------------------------
// Test 1: statistics() on a known 5-row × 3-col fixture
// ---------------------------------------------------------------------------

#[test]
fn statistics_on_known_fixture() {
    let tmp = NamedTempFile::new().expect("tmpfile");
    write_3col_file(tmp.path());

    let provider = HeliumTableProvider::try_new(tmp.path()).expect("try_new");

    // The trait method returns Option<Statistics>.
    let stats = provider.statistics().expect("statistics should be Some");

    // num_rows must be Exact(5).
    assert_eq!(
        stats.num_rows,
        Precision::Exact(5),
        "num_rows should be Exact(5)"
    );

    // total_byte_size must be Exact and > 0.
    assert!(
        matches!(stats.total_byte_size, Precision::Exact(n) if n > 0),
        "total_byte_size should be Exact > 0, got {:?}",
        stats.total_byte_size,
    );

    // Must have one ColumnStatistics entry per top-level column.
    assert_eq!(
        stats.column_statistics.len(),
        3,
        "3 columns → 3 ColumnStatistics"
    );

    // Column 0 (id: I32) — min = 1, max = 5.
    let id_stats = &stats.column_statistics[0];
    assert_eq!(
        id_stats.min_value,
        Precision::Exact(datafusion::common::ScalarValue::Int32(Some(1))),
        "id min"
    );
    assert_eq!(
        id_stats.max_value,
        Precision::Exact(datafusion::common::ScalarValue::Int32(Some(5))),
        "id max"
    );

    // Column 1 (score: I64) — min = 10, max = 50.
    let score_stats = &stats.column_statistics[1];
    assert_eq!(
        score_stats.min_value,
        Precision::Exact(datafusion::common::ScalarValue::Int64(Some(10))),
        "score min"
    );
    assert_eq!(
        score_stats.max_value,
        Precision::Exact(datafusion::common::ScalarValue::Int64(Some(50))),
        "score max"
    );

    // Column 2 (label: Utf8) — min = "apple", max = "elderberry" (lex order).
    let label_stats = &stats.column_statistics[2];
    assert_eq!(
        label_stats.min_value,
        Precision::Exact(datafusion::common::ScalarValue::Utf8(Some(
            "apple".to_string()
        ))),
        "label min"
    );
    assert_eq!(
        label_stats.max_value,
        Precision::Exact(datafusion::common::ScalarValue::Utf8(Some(
            "elderberry".to_string()
        ))),
        "label max"
    );
}

// ---------------------------------------------------------------------------
// Test 2: SELECT count(*) — correctness
// ---------------------------------------------------------------------------

#[test]
fn count_star_correctness() {
    let tmp = NamedTempFile::new().expect("tmpfile");
    write_i64_stripes(tmp.path(), &[vec![1, 2, 3], vec![4, 5], vec![6, 7, 8, 9]]);

    let batches = tokio_run!(run_sql(tmp.path(), "SELECT count(*) FROM t"));
    // 3 + 2 + 4 = 9 rows total; COUNT(*) should return 9.
    assert_eq!(total_rows(&batches), 1, "count(*) returns exactly 1 row");
    let arr = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("Int64Array");
    assert_eq!(arr.value(0), 9, "count(*) should equal 9");
}

// ---------------------------------------------------------------------------
// Test 3: SELECT min(val), max(val) — correctness
// ---------------------------------------------------------------------------

#[test]
fn min_max_correctness() {
    let tmp = NamedTempFile::new().expect("tmpfile");
    // Three stripes: [10,20], [5,30], [15,25]
    write_i64_stripes(tmp.path(), &[vec![10, 20], vec![5, 30], vec![15, 25]]);

    let batches = tokio_run!(run_sql(tmp.path(), "SELECT min(val), max(val) FROM t"));
    assert_eq!(total_rows(&batches), 1, "aggregate returns 1 row");

    let min_arr = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("Int64Array");
    let max_arr = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("Int64Array");
    assert_eq!(min_arr.value(0), 5, "min(val) = 5");
    assert_eq!(max_arr.value(0), 30, "max(val) = 30");
}

// ---------------------------------------------------------------------------
// Test 4: Stats-disabled file
// ---------------------------------------------------------------------------

#[test]
fn stats_disabled_file() {
    let tmp = NamedTempFile::new().expect("tmpfile");
    write_stats_disabled(tmp.path());

    let provider = HeliumTableProvider::try_new(tmp.path()).expect("try_new");
    let stats = provider.statistics().expect("statistics should be Some");

    // num_rows must still be Exact(3) — row count is in the footer regardless
    // of the stats flag.
    assert_eq!(
        stats.num_rows,
        Precision::Exact(3),
        "num_rows Exact even without per-column stats"
    );

    // Column stats must be Absent when stats were disabled.
    let col = &stats.column_statistics[0];
    assert_eq!(
        col.min_value,
        Precision::Absent,
        "min_value Absent when stats disabled"
    );
    assert_eq!(
        col.max_value,
        Precision::Absent,
        "max_value Absent when stats disabled"
    );
}

// ---------------------------------------------------------------------------
// Test 5: Multi-stripe aggregation
// ---------------------------------------------------------------------------

#[test]
fn multi_stripe_aggregation() {
    let tmp = NamedTempFile::new().expect("tmpfile");
    // Stripe 0: [100, 200]   min=100, max=200
    // Stripe 1: [50, 150]    min=50,  max=150
    // Stripe 2: [300, 400]   min=300, max=400
    // File-wide: min=50, max=400
    write_i64_stripes(tmp.path(), &[vec![100, 200], vec![50, 150], vec![300, 400]]);

    let provider = HeliumTableProvider::try_new(tmp.path()).expect("try_new");
    let stats = provider.statistics().expect("statistics");

    assert_eq!(
        stats.num_rows,
        Precision::Exact(6),
        "6 rows across 3 stripes"
    );
    assert_eq!(
        stats.column_statistics[0].min_value,
        Precision::Exact(datafusion::common::ScalarValue::Int64(Some(50))),
        "file-wide min = 50 (minimum of stripe mins)"
    );
    assert_eq!(
        stats.column_statistics[0].max_value,
        Precision::Exact(datafusion::common::ScalarValue::Int64(Some(400))),
        "file-wide max = 400 (maximum of stripe maxes)"
    );
}

// ---------------------------------------------------------------------------
// Test 6: Missing-stats stripe makes column stats Absent
// ---------------------------------------------------------------------------

#[test]
fn missing_stats_stripe_causes_absent() {
    use helium::{
        CoderRegistry, CoderSpec, ColumnData, ColumnSpec, DataType, HeliumWriter, LogicalColumn,
        LogicalType, Schema,
    };

    // Write a two-stripe file with stats disabled on all stripes.
    // This tests the "any absent stripe → Absent file-wide" invariant.
    // With `with_stats_disabled()` every stripe lacks per-column stats,
    // so the file-wide column stats should be Absent even though num_rows
    // is always available (it comes from a different footer field).

    let tmp = NamedTempFile::new().expect("tmpfile");
    let schema = Schema::new(vec![ColumnSpec::new(
        "v",
        LogicalType::Primitive {
            data_type: DataType::I32,
        },
        // I32 → delta+leb128 → zstd terminates in Bytes.
        vec![vec![
            CoderSpec::new("delta"),
            CoderSpec::new("leb128"),
            CoderSpec::new("zstd"),
        ]],
    )]);
    let registry = CoderRegistry::default();
    let file = std::fs::File::create(tmp.path()).expect("create");
    let mut w = HeliumWriter::new(file, schema, &registry)
        .expect("writer")
        .with_stats_disabled();
    w.write_column("v", LogicalColumn::Primitive(ColumnData::I32(vec![1, 2])))
        .expect("write");
    w.finish_stripe().expect("stripe 0");
    w.write_column("v", LogicalColumn::Primitive(ColumnData::I32(vec![3, 4])))
        .expect("write");
    w.finish().expect("finish");

    let provider = HeliumTableProvider::try_new(tmp.path()).expect("try_new");
    let stats = provider.statistics().expect("statistics");

    // 4 rows total, exact.
    assert_eq!(stats.num_rows, Precision::Exact(4));

    // Column stats must be Absent since no stripe has stats.
    assert_eq!(
        stats.column_statistics[0].min_value,
        Precision::Absent,
        "column stats Absent when all stripes lack stats"
    );
    assert_eq!(
        stats.column_statistics[0].max_value,
        Precision::Absent,
        "column stats Absent when all stripes lack stats"
    );
}

// ---------------------------------------------------------------------------
// Test 7: HeliumExec::statistics() returns projected stats
// ---------------------------------------------------------------------------

#[test]
fn helium_exec_statistics() {
    tokio_run!(async {
        let tmp = NamedTempFile::new().expect("tmpfile");
        write_3col_file(tmp.path());

        let provider = Arc::new(HeliumTableProvider::try_new(tmp.path()).expect("try_new"));

        // Use SessionContext to scan with a projection of just column 1 (score).
        let ctx = SessionContext::new();
        ctx.register_table("t", Arc::clone(&provider) as Arc<dyn TableProvider>)
            .expect("register");

        // Get the scan plan with projection [1] (score column only).
        let state = ctx.state();
        let plan = provider
            .scan(&state, Some(&vec![1usize]), &[], None)
            .await
            .expect("scan");

        // The ExecutionPlan::statistics() should report projected statistics.
        let exec_stats = plan.statistics().expect("exec statistics");
        // num_rows is still the full file count (same regardless of projection).
        assert_eq!(
            exec_stats.num_rows,
            Precision::Exact(5),
            "projected exec num_rows"
        );
        // One column_statistics entry for the projected column.
        assert_eq!(exec_stats.column_statistics.len(), 1, "1 projected column");
        // score min=10, max=50.
        assert_eq!(
            exec_stats.column_statistics[0].min_value,
            Precision::Exact(datafusion::common::ScalarValue::Int64(Some(10))),
            "projected exec score min"
        );
        assert_eq!(
            exec_stats.column_statistics[0].max_value,
            Precision::Exact(datafusion::common::ScalarValue::Int64(Some(50))),
            "projected exec score max"
        );
    });
}

// ---------------------------------------------------------------------------
// Test 8: file_statistics() accessor
// ---------------------------------------------------------------------------

#[test]
fn file_statistics_accessor() {
    let tmp = NamedTempFile::new().expect("tmpfile");
    write_i64_stripes(tmp.path(), &[vec![1, 2, 3]]);

    let provider = HeliumTableProvider::try_new(tmp.path()).expect("try_new");
    let stats = provider.file_statistics();
    assert_eq!(
        stats.num_rows,
        Precision::Exact(3),
        "file_statistics() num_rows"
    );
}
