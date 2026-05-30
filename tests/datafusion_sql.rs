//! Integration tests for `helium::sql::HeliumTableProvider` — DataFusion SQL
//! queries over `.he` files.
//!
//! These tests use DataFusion's `SessionContext` to register a
//! `HeliumTableProvider`, run SQL queries, and assert correctness.

#![cfg(feature = "datafusion")]

use std::sync::Arc;

use arrow::array::{Array, Int64Array, StringArray};
use datafusion::catalog::TableProvider;
use datafusion::prelude::*;
use helium::sql::HeliumTableProvider;
use helium::{
    CoderRegistry, CoderSpec, ColumnData, ColumnSpec, HeliumWriter, LogicalColumn, LogicalType,
    Schema,
};
use tempfile::NamedTempFile;

// ---------------------------------------------------------------------------
// Helper: write a simple 3-column × N-row file
// ---------------------------------------------------------------------------

/// Build a small `.he` file with three columns:
///   - `id` : I64
///   - `val` : I32
///   - `label` : Utf8
///
/// Returns a `NamedTempFile` whose path stays valid for the test's lifetime.
fn write_simple_file(rows: u64) -> NamedTempFile {
    let schema = Schema::new(vec![
        ColumnSpec::primitive(
            "id",
            helium::DataType::I64,
            vec![
                CoderSpec::new("delta"),
                CoderSpec::new("leb128"),
                CoderSpec::new("zstd"),
            ],
        ),
        ColumnSpec::primitive(
            "val",
            helium::DataType::I32,
            vec![
                CoderSpec::new("delta"),
                CoderSpec::new("leb128"),
                CoderSpec::new("zstd"),
            ],
        ),
        ColumnSpec::utf8(
            "label",
            vec![
                CoderSpec::new("delta"),
                CoderSpec::new("leb128"),
                CoderSpec::new("zstd"),
            ],
            vec![CoderSpec::new("zstd")],
        ),
    ]);

    let tmp = NamedTempFile::new().expect("tempfile");
    let registry = CoderRegistry::default();
    let mut writer =
        HeliumWriter::new(tmp.as_file().try_clone().expect("clone"), schema, &registry)
            .expect("writer");

    let ids: Vec<i64> = (0..rows as i64).collect();
    let vals: Vec<i32> = (0..rows as i32).collect();
    let labels: Vec<String> = (0..rows).map(|i| format!("row_{i}")).collect();

    writer
        .write_column("id", LogicalColumn::Primitive(ColumnData::I64(ids)))
        .expect("write id");
    writer
        .write_column("val", LogicalColumn::Primitive(ColumnData::I32(vals)))
        .expect("write val");
    writer
        .write_column("label", LogicalColumn::Utf8(labels))
        .expect("write label");

    writer.finish().expect("finish");
    tmp
}

/// Build a simple file with an explicit stripe boundary at `split_at` rows.
fn write_multistripe_file(rows_per_stripe: u64, stripe_count: usize) -> NamedTempFile {
    let schema = Schema::new(vec![ColumnSpec::primitive(
        "id",
        helium::DataType::I64,
        vec![
            CoderSpec::new("delta"),
            CoderSpec::new("leb128"),
            CoderSpec::new("zstd"),
        ],
    )]);

    let tmp = NamedTempFile::new().expect("tempfile");
    let registry = CoderRegistry::default();
    let mut writer =
        HeliumWriter::new(tmp.as_file().try_clone().expect("clone"), schema, &registry)
            .expect("writer");

    let mut global_id: i64 = 0;
    for _s in 0..stripe_count {
        let ids: Vec<i64> = (global_id..global_id + rows_per_stripe as i64).collect();
        global_id += rows_per_stripe as i64;
        writer
            .write_column("id", LogicalColumn::Primitive(ColumnData::I64(ids)))
            .expect("write id");
        writer.finish_stripe().expect("finish_stripe");
    }

    writer.finish().expect("finish");
    tmp
}

// ---------------------------------------------------------------------------
// Helper: async test harness
// ---------------------------------------------------------------------------

/// Run an async block using the tokio multi-thread runtime (required by
/// `block_in_place` inside `HeliumExec::execute`).
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
// Test 1 — SELECT * returns all rows and columns
// ---------------------------------------------------------------------------

#[test]
fn select_star_returns_all_rows() {
    tokio_run!(async {
        let tmp = write_simple_file(5);
        let provider = HeliumTableProvider::try_new(tmp.path()).expect("provider");
        let ctx = SessionContext::new();
        ctx.register_table("t", Arc::new(provider))
            .expect("register");

        let df = ctx.sql("SELECT * FROM t").await.expect("sql");
        let batches = df.collect().await.expect("collect");

        // Count total rows across all batches.
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 5, "expected 5 rows");

        // All 3 columns should be present.
        let num_cols = batches[0].num_columns();
        assert_eq!(num_cols, 3, "expected 3 columns, got {num_cols}");
    });
}

// ---------------------------------------------------------------------------
// Test 2 — SELECT subset (projection pushdown)
// ---------------------------------------------------------------------------

#[test]
fn select_subset_projects_correctly() {
    tokio_run!(async {
        let tmp = write_simple_file(5);
        let provider = HeliumTableProvider::try_new(tmp.path()).expect("provider");
        let ctx = SessionContext::new();
        ctx.register_table("t", Arc::new(provider))
            .expect("register");

        let df = ctx.sql("SELECT id, label FROM t").await.expect("sql");
        let batches = df.collect().await.expect("collect");

        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 5);
        // Only 2 columns (id + label), not all 3.
        assert_eq!(batches[0].num_columns(), 2);
        assert_eq!(batches[0].schema().field(0).name(), "id");
        assert_eq!(batches[0].schema().field(1).name(), "label");
    });
}

// ---------------------------------------------------------------------------
// Test 3 — COUNT(*) aggregate
// ---------------------------------------------------------------------------

#[test]
fn count_star_returns_row_count() {
    tokio_run!(async {
        let tmp = write_simple_file(5);
        let provider = HeliumTableProvider::try_new(tmp.path()).expect("provider");
        let ctx = SessionContext::new();
        ctx.register_table("t", Arc::new(provider))
            .expect("register");

        let df = ctx.sql("SELECT count(*) FROM t").await.expect("sql");
        let batches = df.collect().await.expect("collect");

        assert_eq!(batches.len(), 1);
        let arr = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("Int64Array");
        assert_eq!(arr.value(0), 5, "expected count(*) = 5");
    });
}

// ---------------------------------------------------------------------------
// Test 4 — WHERE filter (Inexact pushdown — DataFusion re-evaluates post-scan)
// ---------------------------------------------------------------------------

#[test]
fn where_filter_returns_correct_rows() {
    tokio_run!(async {
        let tmp = write_simple_file(5);
        let provider = HeliumTableProvider::try_new(tmp.path()).expect("provider");
        let ctx = SessionContext::new();
        ctx.register_table("t", Arc::new(provider))
            .expect("register");

        // id values are 0,1,2,3,4 — ids > 2 → rows with id=3, id=4
        let df = ctx.sql("SELECT id FROM t WHERE id > 2").await.expect("sql");
        let batches = df.collect().await.expect("collect");

        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 2, "expected 2 rows where id > 2");

        // Verify exact values.
        let arr = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("Int64Array");
        let mut vals: Vec<i64> = (0..arr.len()).map(|i| arr.value(i)).collect();
        vals.sort_unstable();
        assert_eq!(vals, vec![3, 4]);
    });
}

// ---------------------------------------------------------------------------
// Test 5 — LIMIT
// ---------------------------------------------------------------------------

#[test]
fn limit_returns_at_most_n_rows() {
    tokio_run!(async {
        let tmp = write_simple_file(5);
        let provider = HeliumTableProvider::try_new(tmp.path()).expect("provider");
        let ctx = SessionContext::new();
        ctx.register_table("t", Arc::new(provider))
            .expect("register");

        let df = ctx.sql("SELECT * FROM t LIMIT 3").await.expect("sql");
        let batches = df.collect().await.expect("collect");

        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 3, "expected at most 3 rows");
    });
}

// ---------------------------------------------------------------------------
// Test 6 — Multi-stripe: COUNT(*) sums across stripes
// ---------------------------------------------------------------------------

#[test]
fn multi_stripe_count_star() {
    tokio_run!(async {
        // 3 stripes × 4 rows = 12 rows total
        let tmp = write_multistripe_file(4, 3);
        let provider = HeliumTableProvider::try_new(tmp.path()).expect("provider");
        assert_eq!(provider.stripe_count(), 3);
        assert_eq!(provider.total_rows(), 12);

        let ctx = SessionContext::new();
        ctx.register_table("t", Arc::new(provider))
            .expect("register");

        let df = ctx.sql("SELECT count(*) FROM t").await.expect("sql");
        let batches = df.collect().await.expect("collect");
        let arr = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("Int64Array");
        assert_eq!(arr.value(0), 12, "expected count(*) = 12");
    });
}

// ---------------------------------------------------------------------------
// Test 7 — Nullable column: WHERE col IS NULL
// ---------------------------------------------------------------------------

#[test]
fn nullable_column_null_filter() {
    // Write a file with a Nullable<I64> column.
    // present = [true, false, true, false, true] → nulls at positions 1 and 3.
    let schema = Schema::new(vec![ColumnSpec::nullable(
        "score",
        LogicalType::Primitive {
            data_type: helium::DataType::I64,
        },
        vec![
            // present bitmap: U8 → leb128 → zstd
            vec![CoderSpec::new("leb128"), CoderSpec::new("zstd")],
            // values: I64 → delta → leb128 → zstd
            vec![
                CoderSpec::new("delta"),
                CoderSpec::new("leb128"),
                CoderSpec::new("zstd"),
            ],
        ],
    )]);

    let tmp = NamedTempFile::new().expect("tempfile");
    let registry = CoderRegistry::default();
    let mut writer =
        HeliumWriter::new(tmp.as_file().try_clone().expect("clone"), schema, &registry)
            .expect("writer");

    writer
        .write_column(
            "score",
            LogicalColumn::Nullable {
                present: vec![true, false, true, false, true],
                // Only the 3 non-null values (compact form).
                value: Box::new(LogicalColumn::Primitive(ColumnData::I64(vec![10, 30, 50]))),
            },
        )
        .expect("write score");
    writer.finish().expect("finish");

    tokio_run!(async {
        let provider = HeliumTableProvider::try_new(tmp.path()).expect("provider");
        let ctx = SessionContext::new();
        ctx.register_table("t", Arc::new(provider))
            .expect("register");

        // WHERE score IS NULL → should return 2 rows.
        let df = ctx
            .sql("SELECT score FROM t WHERE score IS NULL")
            .await
            .expect("sql");
        let batches = df.collect().await.expect("collect");
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 2, "expected 2 null rows");

        // WHERE score IS NOT NULL → should return 3 rows.
        let df2 = ctx
            .sql("SELECT score FROM t WHERE score IS NOT NULL")
            .await
            .expect("sql");
        let batches2 = df2.collect().await.expect("collect");
        let total2: usize = batches2.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total2, 3, "expected 3 non-null rows");
    });
}

// ---------------------------------------------------------------------------
// Test 8 — String column round-trip
// ---------------------------------------------------------------------------

#[test]
fn string_column_round_trip() {
    tokio_run!(async {
        let tmp = write_simple_file(3);
        let provider = HeliumTableProvider::try_new(tmp.path()).expect("provider");
        let ctx = SessionContext::new();
        ctx.register_table("t", Arc::new(provider))
            .expect("register");

        let df = ctx.sql("SELECT label FROM t").await.expect("sql");
        let batches = df.collect().await.expect("collect");

        let mut labels: Vec<String> = Vec::new();
        for batch in &batches {
            let arr = batch
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("StringArray");
            for i in 0..arr.len() {
                labels.push(arr.value(i).to_string());
            }
        }
        labels.sort();
        assert_eq!(labels, vec!["row_0", "row_1", "row_2"]);
    });
}

// ---------------------------------------------------------------------------
// Test 9 — Table metadata (stripe_count, total_rows, schema)
// ---------------------------------------------------------------------------

#[test]
fn table_provider_metadata() {
    let tmp = write_simple_file(5);
    let provider = HeliumTableProvider::try_new(tmp.path()).expect("provider");
    assert_eq!(provider.total_rows(), 5);
    assert_eq!(provider.stripe_count(), 1);
    assert_eq!(provider.helium_schema().columns.len(), 3);

    let arrow_schema = provider.schema();
    assert_eq!(arrow_schema.fields().len(), 3);
    assert_eq!(arrow_schema.field(0).name(), "id");
    assert_eq!(arrow_schema.field(1).name(), "val");
    assert_eq!(arrow_schema.field(2).name(), "label");
}

// ---------------------------------------------------------------------------
// Test 10 — WHERE with string predicate
// ---------------------------------------------------------------------------

#[test]
fn where_string_filter() {
    tokio_run!(async {
        let tmp = write_simple_file(5);
        let provider = HeliumTableProvider::try_new(tmp.path()).expect("provider");
        let ctx = SessionContext::new();
        ctx.register_table("t", Arc::new(provider))
            .expect("register");

        // label = 'row_2' → exactly 1 row
        let df = ctx
            .sql("SELECT id FROM t WHERE label = 'row_2'")
            .await
            .expect("sql");
        let batches = df.collect().await.expect("collect");
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 1, "expected 1 row matching label='row_2'");

        let arr = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("Int64Array");
        assert_eq!(arr.value(0), 2);
    });
}
