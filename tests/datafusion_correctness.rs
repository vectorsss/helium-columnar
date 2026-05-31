//! Data-correctness tests for the Helium → Arrow → DataFusion pipeline.
//!
//! Each test:
//!   1. Writes a `.he` fixture with hand-built values.
//!   2. Registers the file as a DataFusion table via `HeliumTableProvider`.
//!   3. Runs a SQL query.
//!   4. Asserts specific cell values — not just row counts.
//!
//! These tests catch bugs where plumbing is correct but values are wrong
//! (type mis-mapping, byte swapping, null position errors, etc.).

#![cfg(feature = "datafusion")]

use std::path::Path;
use std::sync::Arc;

use arrow::array::{
    Array, BinaryArray, Float32Array, Float64Array, Int8Array, Int16Array, Int32Array, Int64Array,
    ListArray, StringArray, StructArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
};
use datafusion::prelude::SessionContext;
use helium::sql::HeliumTableProvider;
use helium::{
    CoderRegistry, CoderSpec, ColumnData, ColumnSpec, FieldSpec, HeliumWriter, LogicalColumn,
    LogicalType, Schema,
};
use tempfile::NamedTempFile;

// ---------------------------------------------------------------------------
// Helper: async multi-thread runner
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
// Helper: build a .he file and run a SQL query
// ---------------------------------------------------------------------------

/// Collect all rows from `query` run against the `.he` file at `path`.
async fn run_sql(path: &Path, query: &str) -> Vec<arrow::record_batch::RecordBatch> {
    let provider = HeliumTableProvider::try_new(path).expect("HeliumTableProvider::try_new");
    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::new(provider))
        .expect("register_table");
    let df = ctx.sql(query).await.expect("sql parse/plan");
    df.collect().await.expect("collect")
}

/// Total row count across all batches.
fn total_rows(batches: &[arrow::record_batch::RecordBatch]) -> usize {
    batches.iter().map(|b| b.num_rows()).sum()
}

/// Collect all `i64` values from column 0 across all batches, in order.
fn collect_i64(batches: &[arrow::record_batch::RecordBatch]) -> Vec<i64> {
    let mut out = Vec::new();
    for b in batches {
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

/// Collect all `i64` values from column 0, preserving `None` for nulls.
fn collect_i64_nullable(batches: &[arrow::record_batch::RecordBatch]) -> Vec<Option<i64>> {
    let mut out = Vec::new();
    for b in batches {
        let arr = b
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("Int64Array");
        for i in 0..arr.len() {
            if arr.is_null(i) {
                out.push(None);
            } else {
                out.push(Some(arr.value(i)));
            }
        }
    }
    out
}

/// Collect all `String` values from column 0 across all batches.
fn collect_str(batches: &[arrow::record_batch::RecordBatch]) -> Vec<String> {
    let mut out = Vec::new();
    for b in batches {
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

// ---------------------------------------------------------------------------
// Pipeline helpers
//
// Rules:
//   - `zstd` requires Bytes input (it's a block coder that compresses byte blobs).
//   - `leb128` converts any integer type (I8..U64) → Bytes; can be followed by zstd.
//   - `gorilla` converts F32/F64 → Bytes; can be followed by zstd.
//   - `delta` → `leb128` → `zstd` works for monotone/small-delta integer sequences.
//   - For boundary value tests (where delta would overflow), skip delta and use leb128 directly.
// ---------------------------------------------------------------------------

/// leb128 → zstd: integer types (I8 / I16 / I32 / I64 / U8 / U16 / U32 / U64).
fn int_pipe() -> Vec<CoderSpec> {
    vec![CoderSpec::new("leb128"), CoderSpec::new("zstd")]
}

/// gorilla → zstd: float types (F32 / F64).
fn float_pipe() -> Vec<CoderSpec> {
    vec![CoderSpec::new("gorilla"), CoderSpec::new("zstd")]
}

/// delta → leb128 → zstd: safe for monotone/small-delta sequences.
fn delta_leb_zstd() -> Vec<CoderSpec> {
    vec![
        CoderSpec::new("delta"),
        CoderSpec::new("leb128"),
        CoderSpec::new("zstd"),
    ]
}

/// leb128 → zstd for offsets (U32 sequences).
fn offset_pipe() -> Vec<CoderSpec> {
    int_pipe()
}

/// zstd directly on Bytes input (for data fields, offsets after leb128, etc.).
/// This is only valid as the **final** stage after a coder that emits Bytes.
fn bytes_zstd() -> Vec<CoderSpec> {
    vec![CoderSpec::new("zstd")]
}

// ===========================================================================
// Numeric primitive round-trips
// ===========================================================================

// ── test 1a: i8 boundary values ────────────────────────────────────────────

#[test]
fn prim_i8_boundary_values() {
    let schema = Schema::new(vec![ColumnSpec::primitive(
        "x",
        helium::DataType::I8,
        int_pipe(), // leb128 → zstd: handles all i8 values
    )]);
    let tmp = NamedTempFile::new().unwrap();
    let reg = CoderRegistry::default();
    let mut w = HeliumWriter::new(tmp.as_file().try_clone().unwrap(), schema, &reg).unwrap();
    w.write_column(
        "x",
        LogicalColumn::Primitive(ColumnData::I8(vec![-128, -1, 0, 1, 127])),
    )
    .unwrap();
    w.finish().unwrap();

    tokio_run!(async {
        let batches = run_sql(tmp.path(), "SELECT x FROM t").await;
        assert_eq!(total_rows(&batches), 5);
        let arr = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int8Array>()
            .expect("Int8Array");
        assert_eq!(arr.value(0), -128, "i8 min");
        assert_eq!(arr.value(1), -1);
        assert_eq!(arr.value(2), 0);
        assert_eq!(arr.value(3), 1);
        assert_eq!(arr.value(4), 127, "i8 max");
    });
}

// ── test 1b: i16 boundary values ───────────────────────────────────────────

#[test]
fn prim_i16_boundary_values() {
    let schema = Schema::new(vec![ColumnSpec::primitive(
        "x",
        helium::DataType::I16,
        int_pipe(),
    )]);
    let tmp = NamedTempFile::new().unwrap();
    let reg = CoderRegistry::default();
    let mut w = HeliumWriter::new(tmp.as_file().try_clone().unwrap(), schema, &reg).unwrap();
    w.write_column(
        "x",
        LogicalColumn::Primitive(ColumnData::I16(vec![i16::MIN, -1, 0, 1, i16::MAX])),
    )
    .unwrap();
    w.finish().unwrap();

    tokio_run!(async {
        let batches = run_sql(tmp.path(), "SELECT x FROM t").await;
        let arr = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int16Array>()
            .unwrap();
        assert_eq!(arr.value(0), i16::MIN, "i16 min");
        assert_eq!(arr.value(2), 0);
        assert_eq!(arr.value(4), i16::MAX, "i16 max");
    });
}

// ── test 1c: i32 boundary values ───────────────────────────────────────────

#[test]
fn prim_i32_boundary_values() {
    let schema = Schema::new(vec![ColumnSpec::primitive(
        "x",
        helium::DataType::I32,
        int_pipe(),
    )]);
    let tmp = NamedTempFile::new().unwrap();
    let reg = CoderRegistry::default();
    let mut w = HeliumWriter::new(tmp.as_file().try_clone().unwrap(), schema, &reg).unwrap();
    w.write_column(
        "x",
        LogicalColumn::Primitive(ColumnData::I32(vec![i32::MIN, -1, 0, 1, i32::MAX])),
    )
    .unwrap();
    w.finish().unwrap();

    tokio_run!(async {
        let batches = run_sql(tmp.path(), "SELECT x FROM t").await;
        let arr = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(arr.value(0), i32::MIN, "i32 min");
        assert_eq!(arr.value(4), i32::MAX, "i32 max");
    });
}

// ── test 1d: i64 boundary values ───────────────────────────────────────────

#[test]
fn prim_i64_boundary_values() {
    let schema = Schema::new(vec![ColumnSpec::primitive(
        "x",
        helium::DataType::I64,
        int_pipe(),
    )]);
    let tmp = NamedTempFile::new().unwrap();
    let reg = CoderRegistry::default();
    let mut w = HeliumWriter::new(tmp.as_file().try_clone().unwrap(), schema, &reg).unwrap();
    w.write_column(
        "x",
        LogicalColumn::Primitive(ColumnData::I64(vec![i64::MIN, -1, 0, 1, i64::MAX])),
    )
    .unwrap();
    w.finish().unwrap();

    tokio_run!(async {
        let batches = run_sql(tmp.path(), "SELECT x FROM t").await;
        let arr = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(arr.value(0), i64::MIN, "i64 min");
        assert_eq!(arr.value(4), i64::MAX, "i64 max");
    });
}

// ── test 1e: u8 values ─────────────────────────────────────────────────────

#[test]
fn prim_u8_values() {
    let schema = Schema::new(vec![ColumnSpec::primitive(
        "x",
        helium::DataType::U8,
        int_pipe(),
    )]);
    let tmp = NamedTempFile::new().unwrap();
    let reg = CoderRegistry::default();
    let mut w = HeliumWriter::new(tmp.as_file().try_clone().unwrap(), schema, &reg).unwrap();
    w.write_column(
        "x",
        LogicalColumn::Primitive(ColumnData::U8(vec![0, 1, 127, 255])),
    )
    .unwrap();
    w.finish().unwrap();

    tokio_run!(async {
        let batches = run_sql(tmp.path(), "SELECT x FROM t").await;
        let arr = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<UInt8Array>()
            .unwrap();
        assert_eq!(arr.value(0), 0u8);
        assert_eq!(arr.value(3), 255u8, "u8 max");
    });
}

// ── test 1f: u16 boundary values ───────────────────────────────────────────

#[test]
fn prim_u16_values() {
    let schema = Schema::new(vec![ColumnSpec::primitive(
        "x",
        helium::DataType::U16,
        int_pipe(),
    )]);
    let tmp = NamedTempFile::new().unwrap();
    let reg = CoderRegistry::default();
    let mut w = HeliumWriter::new(tmp.as_file().try_clone().unwrap(), schema, &reg).unwrap();
    w.write_column(
        "x",
        LogicalColumn::Primitive(ColumnData::U16(vec![0, 1, u16::MAX])),
    )
    .unwrap();
    w.finish().unwrap();

    tokio_run!(async {
        let batches = run_sql(tmp.path(), "SELECT x FROM t").await;
        let arr = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<UInt16Array>()
            .unwrap();
        assert_eq!(arr.value(0), 0u16);
        assert_eq!(arr.value(2), u16::MAX, "u16 max");
    });
}

// ── test 1g: u32 boundary values ───────────────────────────────────────────

#[test]
fn prim_u32_values() {
    let schema = Schema::new(vec![ColumnSpec::primitive(
        "x",
        helium::DataType::U32,
        int_pipe(),
    )]);
    let tmp = NamedTempFile::new().unwrap();
    let reg = CoderRegistry::default();
    let mut w = HeliumWriter::new(tmp.as_file().try_clone().unwrap(), schema, &reg).unwrap();
    w.write_column(
        "x",
        LogicalColumn::Primitive(ColumnData::U32(vec![0, 1, u32::MAX])),
    )
    .unwrap();
    w.finish().unwrap();

    tokio_run!(async {
        let batches = run_sql(tmp.path(), "SELECT x FROM t").await;
        let arr = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        assert_eq!(arr.value(0), 0u32);
        assert_eq!(arr.value(2), u32::MAX, "u32 max");
    });
}

// ── test 1h: u64 boundary values ───────────────────────────────────────────

#[test]
fn prim_u64_values() {
    let schema = Schema::new(vec![ColumnSpec::primitive(
        "x",
        helium::DataType::U64,
        int_pipe(),
    )]);
    let tmp = NamedTempFile::new().unwrap();
    let reg = CoderRegistry::default();
    let mut w = HeliumWriter::new(tmp.as_file().try_clone().unwrap(), schema, &reg).unwrap();
    w.write_column(
        "x",
        LogicalColumn::Primitive(ColumnData::U64(vec![0, 1, u64::MAX])),
    )
    .unwrap();
    w.finish().unwrap();

    tokio_run!(async {
        let batches = run_sql(tmp.path(), "SELECT x FROM t").await;
        let arr = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(arr.value(0), 0u64);
        assert_eq!(arr.value(2), u64::MAX, "u64 max");
    });
}

// ── test 2a: f32 special values ────────────────────────────────────────────

#[test]
fn prim_f32_special_values() {
    // gorilla → zstd: correctly handles F32 including NaN and infinities.
    let schema = Schema::new(vec![ColumnSpec::primitive(
        "x",
        helium::DataType::F32,
        float_pipe(),
    )]);
    let tmp = NamedTempFile::new().unwrap();
    let reg = CoderRegistry::default();
    let mut w = HeliumWriter::new(tmp.as_file().try_clone().unwrap(), schema, &reg).unwrap();
    w.write_column(
        "x",
        LogicalColumn::Primitive(ColumnData::F32(vec![
            0.0f32,
            -0.0f32,
            1.5f32,
            -1.5f32,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ])),
    )
    .unwrap();
    w.finish().unwrap();

    tokio_run!(async {
        let batches = run_sql(tmp.path(), "SELECT x FROM t").await;
        let arr = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        assert_eq!(arr.value(2), 1.5f32);
        assert_eq!(arr.value(3), -1.5f32);
        assert!(
            arr.value(4).is_infinite() && arr.value(4) > 0.0,
            "+Inf round-trip"
        );
        assert!(
            arr.value(5).is_infinite() && arr.value(5) < 0.0,
            "-Inf round-trip"
        );
        // NaN: DataFusion normalises NaN; we verify it survives as NaN.
        assert!(arr.value(6).is_nan(), "NaN must survive round-trip as NaN");
    });
}

// ── test 2b: f64 special values ────────────────────────────────────────────

#[test]
fn prim_f64_special_values() {
    let schema = Schema::new(vec![ColumnSpec::primitive(
        "x",
        helium::DataType::F64,
        float_pipe(),
    )]);
    let tmp = NamedTempFile::new().unwrap();
    let reg = CoderRegistry::default();
    let mut w = HeliumWriter::new(tmp.as_file().try_clone().unwrap(), schema, &reg).unwrap();
    w.write_column(
        "x",
        LogicalColumn::Primitive(ColumnData::F64(vec![
            0.0f64,
            -0.0f64,
            1.5f64,
            -1.5f64,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            f64::MIN,
            f64::MAX,
        ])),
    )
    .unwrap();
    w.finish().unwrap();

    tokio_run!(async {
        let batches = run_sql(tmp.path(), "SELECT x FROM t").await;
        let arr = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(arr.value(2), 1.5f64);
        assert_eq!(arr.value(3), -1.5f64);
        assert!(arr.value(4).is_infinite() && arr.value(4) > 0.0, "+Inf");
        assert!(arr.value(5).is_infinite() && arr.value(5) < 0.0, "-Inf");
        assert!(arr.value(6).is_nan(), "NaN must survive round-trip");
        assert_eq!(arr.value(7), f64::MIN, "f64::MIN");
        assert_eq!(arr.value(8), f64::MAX, "f64::MAX");
    });
}

// ── test 3: SUM(i64) ───────────────────────────────────────────────────────

#[test]
fn aggregate_sum_i64() {
    let schema = Schema::new(vec![ColumnSpec::primitive(
        "x",
        helium::DataType::I64,
        delta_leb_zstd(),
    )]);
    let tmp = NamedTempFile::new().unwrap();
    let reg = CoderRegistry::default();
    let mut w = HeliumWriter::new(tmp.as_file().try_clone().unwrap(), schema, &reg).unwrap();
    w.write_column(
        "x",
        LogicalColumn::Primitive(ColumnData::I64(vec![1, 2, 3, 4, 5])),
    )
    .unwrap();
    w.finish().unwrap();

    tokio_run!(async {
        let batches = run_sql(tmp.path(), "SELECT sum(x) FROM t").await;
        let arr = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(arr.value(0), 15, "1+2+3+4+5=15");
    });
}

// ── test 4: AVG(f64) ───────────────────────────────────────────────────────

#[test]
fn aggregate_avg_f64() {
    let schema = Schema::new(vec![ColumnSpec::primitive(
        "x",
        helium::DataType::F64,
        float_pipe(),
    )]);
    let tmp = NamedTempFile::new().unwrap();
    let reg = CoderRegistry::default();
    let mut w = HeliumWriter::new(tmp.as_file().try_clone().unwrap(), schema, &reg).unwrap();
    w.write_column(
        "x",
        LogicalColumn::Primitive(ColumnData::F64(vec![1.0, 2.0, 3.0])),
    )
    .unwrap();
    w.finish().unwrap();

    tokio_run!(async {
        let batches = run_sql(tmp.path(), "SELECT avg(x) FROM t").await;
        let arr = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        let v = arr.value(0);
        assert!((v - 2.0).abs() < 1e-9, "avg should be 2.0, got {v}");
    });
}

// ── test 5: MIN / MAX(i64) with negatives ──────────────────────────────────

#[test]
fn aggregate_min_max_i64_negatives() {
    let schema = Schema::new(vec![ColumnSpec::primitive(
        "x",
        helium::DataType::I64,
        delta_leb_zstd(),
    )]);
    let tmp = NamedTempFile::new().unwrap();
    let reg = CoderRegistry::default();
    let mut w = HeliumWriter::new(tmp.as_file().try_clone().unwrap(), schema, &reg).unwrap();
    // Sort for delta to work without overflow: [-5, -1, 3, 10]
    w.write_column(
        "x",
        LogicalColumn::Primitive(ColumnData::I64(vec![-5, 3, -1, 10])),
    )
    .unwrap();
    w.finish().unwrap();

    tokio_run!(async {
        let batches = run_sql(tmp.path(), "SELECT min(x), max(x) FROM t").await;
        let min_arr = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let max_arr = batches[0]
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(min_arr.value(0), -5, "min = -5");
        assert_eq!(max_arr.value(0), 10, "max = 10");
    });
}

// ── test 6: COUNT with WHERE and NULLs ────────────────────────────────────

#[test]
fn count_where_with_nulls() {
    // Column: [1, 2, NULL, 4, 5]
    // WHERE x > 2 should match 4 and 5 only (NULL excluded by comparison).
    let schema = Schema::new(vec![ColumnSpec::nullable(
        "x",
        LogicalType::Primitive {
            data_type: helium::DataType::I64,
        },
        vec![int_pipe(), delta_leb_zstd()],
    )]);
    let tmp = NamedTempFile::new().unwrap();
    let reg = CoderRegistry::default();
    let mut w = HeliumWriter::new(tmp.as_file().try_clone().unwrap(), schema, &reg).unwrap();
    w.write_column(
        "x",
        LogicalColumn::Nullable {
            present: vec![true, true, false, true, true],
            value: Box::new(LogicalColumn::Primitive(ColumnData::I64(vec![1, 2, 4, 5]))),
        },
    )
    .unwrap();
    w.finish().unwrap();

    tokio_run!(async {
        let batches = run_sql(tmp.path(), "SELECT count(*) FROM t WHERE x > 2").await;
        let arr = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(
            arr.value(0),
            2,
            "count(*) WHERE x > 2 should be 2 (rows 4, 5)"
        );
    });
}

// ===========================================================================
// Strings
// ===========================================================================

// ── test 7: Utf8 edge-case round-trip ──────────────────────────────────────

#[test]
fn utf8_edge_cases_roundtrip() {
    let values = [
        "",
        "ascii",
        "中文",
        "emoji 🦀",
        "with\nnewline",
        "\"quoted\"",
        "tab\there",
    ];
    let schema = Schema::new(vec![ColumnSpec::utf8("s", delta_leb_zstd(), bytes_zstd())]);
    let tmp = NamedTempFile::new().unwrap();
    let reg = CoderRegistry::default();
    let mut w = HeliumWriter::new(tmp.as_file().try_clone().unwrap(), schema, &reg).unwrap();
    w.write_column(
        "s",
        LogicalColumn::Utf8(values.iter().map(|&s| s.to_string()).collect()),
    )
    .unwrap();
    w.finish().unwrap();

    tokio_run!(async {
        let batches = run_sql(tmp.path(), "SELECT s FROM t").await;
        assert_eq!(total_rows(&batches), values.len());
        let mut got: Vec<String> = Vec::new();
        for b in &batches {
            let arr = b.column(0).as_any().downcast_ref::<StringArray>().unwrap();
            for i in 0..arr.len() {
                got.push(arr.value(i).to_string());
            }
        }
        for (i, (expected, actual)) in values.iter().zip(got.iter()).enumerate() {
            assert_eq!(
                *expected,
                actual.as_str(),
                "row {i}: expected {:?}, got {:?}",
                expected,
                actual
            );
        }
    });
}

// ── test 8: WHERE LIKE ─────────────────────────────────────────────────────

#[test]
fn utf8_where_like() {
    // Case-sensitive LIKE; '%cat%' should match "cat", "category", "scatter"
    // but NOT "Cat" or "dog".
    let rows = ["cat", "category", "dog", "Cat", "scatter"];
    let schema = Schema::new(vec![ColumnSpec::utf8(
        "name",
        delta_leb_zstd(),
        bytes_zstd(),
    )]);
    let tmp = NamedTempFile::new().unwrap();
    let reg = CoderRegistry::default();
    let mut w = HeliumWriter::new(tmp.as_file().try_clone().unwrap(), schema, &reg).unwrap();
    w.write_column(
        "name",
        LogicalColumn::Utf8(rows.iter().map(|&s| s.to_string()).collect()),
    )
    .unwrap();
    w.finish().unwrap();

    tokio_run!(async {
        let batches = run_sql(tmp.path(), "SELECT name FROM t WHERE name LIKE '%cat%'").await;
        let mut got = collect_str(&batches);
        got.sort();
        assert_eq!(
            got,
            vec!["cat", "category", "scatter"],
            "LIKE '%cat%' (case-sensitive) should match cat, category, scatter"
        );
    });
}

// ── test 9: case-sensitive equality ────────────────────────────────────────

#[test]
fn utf8_case_sensitive_equality() {
    let rows = ["foo", "Foo", "FOO"];
    let schema = Schema::new(vec![ColumnSpec::utf8(
        "name",
        delta_leb_zstd(),
        bytes_zstd(),
    )]);
    let tmp = NamedTempFile::new().unwrap();
    let reg = CoderRegistry::default();
    let mut w = HeliumWriter::new(tmp.as_file().try_clone().unwrap(), schema, &reg).unwrap();
    w.write_column(
        "name",
        LogicalColumn::Utf8(rows.iter().map(|&s| s.to_string()).collect()),
    )
    .unwrap();
    w.finish().unwrap();

    tokio_run!(async {
        let batches = run_sql(tmp.path(), "SELECT name FROM t WHERE name = 'Foo'").await;
        let got = collect_str(&batches);
        assert_eq!(got, vec!["Foo"], "Only 'Foo' should match (case-sensitive)");
    });
}

// ===========================================================================
// Binary
// ===========================================================================

// ── test 10: Binary round-trip ─────────────────────────────────────────────

#[test]
fn binary_roundtrip() {
    let blobs: Vec<Vec<u8>> = vec![
        vec![],
        vec![0xff],
        vec![0x00, 0x01, 0x02],
        (1u8..=255).collect(),
    ];
    let schema = Schema::new(vec![ColumnSpec::binary(
        "b",
        delta_leb_zstd(),
        bytes_zstd(),
    )]);
    let tmp = NamedTempFile::new().unwrap();
    let reg = CoderRegistry::default();
    let mut w = HeliumWriter::new(tmp.as_file().try_clone().unwrap(), schema, &reg).unwrap();
    w.write_column("b", LogicalColumn::Binary(blobs.clone()))
        .unwrap();
    w.finish().unwrap();

    tokio_run!(async {
        let batches = run_sql(tmp.path(), "SELECT b FROM t").await;
        assert_eq!(total_rows(&batches), blobs.len());
        let mut got: Vec<Vec<u8>> = Vec::new();
        for batch in &batches {
            let arr = batch
                .column(0)
                .as_any()
                .downcast_ref::<BinaryArray>()
                .unwrap();
            for i in 0..arr.len() {
                got.push(arr.value(i).to_vec());
            }
        }
        for (i, (expected, actual)) in blobs.iter().zip(got.iter()).enumerate() {
            assert_eq!(expected, actual, "row {i} binary data mismatch");
        }
    });
}

// ===========================================================================
// Nullable
// ===========================================================================

// ── test 11: Nullable<I64> mixed nulls, value correctness ─────────────────

#[test]
fn nullable_i64_value_correctness() {
    // [Some(1), None, Some(3), None, Some(5)]
    let schema = Schema::new(vec![ColumnSpec::nullable(
        "x",
        LogicalType::Primitive {
            data_type: helium::DataType::I64,
        },
        vec![int_pipe(), delta_leb_zstd()],
    )]);
    let tmp = NamedTempFile::new().unwrap();
    let reg = CoderRegistry::default();
    let mut w = HeliumWriter::new(tmp.as_file().try_clone().unwrap(), schema, &reg).unwrap();
    w.write_column(
        "x",
        LogicalColumn::Nullable {
            present: vec![true, false, true, false, true],
            value: Box::new(LogicalColumn::Primitive(ColumnData::I64(vec![1, 3, 5]))),
        },
    )
    .unwrap();
    w.finish().unwrap();

    tokio_run!(async {
        let batches = run_sql(tmp.path(), "SELECT x FROM t").await;
        let got = collect_i64_nullable(&batches);
        assert_eq!(
            got,
            vec![Some(1), None, Some(3), None, Some(5)],
            "Nullable<I64> values must match in order"
        );
    });
}

// ── test 12: IS NULL filter ────────────────────────────────────────────────

#[test]
fn nullable_is_null_filter() {
    let schema = Schema::new(vec![ColumnSpec::nullable(
        "x",
        LogicalType::Primitive {
            data_type: helium::DataType::I64,
        },
        vec![int_pipe(), delta_leb_zstd()],
    )]);
    let tmp = NamedTempFile::new().unwrap();
    let reg = CoderRegistry::default();
    let mut w = HeliumWriter::new(tmp.as_file().try_clone().unwrap(), schema, &reg).unwrap();
    w.write_column(
        "x",
        LogicalColumn::Nullable {
            present: vec![true, false, true, false, true],
            value: Box::new(LogicalColumn::Primitive(ColumnData::I64(vec![1, 3, 5]))),
        },
    )
    .unwrap();
    w.finish().unwrap();

    tokio_run!(async {
        // IS NULL → 2 rows
        let batches = run_sql(tmp.path(), "SELECT count(*) FROM t WHERE x IS NULL").await;
        let arr = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(arr.value(0), 2, "IS NULL count should be 2");

        // IS NOT NULL → 3 rows
        let batches2 = run_sql(tmp.path(), "SELECT count(*) FROM t WHERE x IS NOT NULL").await;
        let arr2 = batches2[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(arr2.value(0), 3, "IS NOT NULL count should be 3");
    });
}

// ── test 13: Nullable<Utf8> COUNT DISTINCT ─────────────────────────────────

#[test]
fn nullable_utf8_count_distinct() {
    // [Some("a"), None, Some("c"), Some("a"), None]
    // distinct non-null values: {"a", "c"} → 2
    let schema = Schema::new(vec![ColumnSpec::nullable(
        "s",
        LogicalType::Utf8,
        vec![int_pipe(), delta_leb_zstd(), bytes_zstd()],
    )]);
    let tmp = NamedTempFile::new().unwrap();
    let reg = CoderRegistry::default();
    let mut w = HeliumWriter::new(tmp.as_file().try_clone().unwrap(), schema, &reg).unwrap();
    w.write_column(
        "s",
        LogicalColumn::Nullable {
            present: vec![true, false, true, true, false],
            value: Box::new(LogicalColumn::Utf8(vec![
                "a".into(),
                "c".into(),
                "a".into(),
            ])),
        },
    )
    .unwrap();
    w.finish().unwrap();

    tokio_run!(async {
        let batches = run_sql(tmp.path(), "SELECT count(distinct s) FROM t").await;
        let arr = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(
            arr.value(0),
            2,
            "count(distinct) over nullable Utf8 should be 2"
        );
    });
}

// ===========================================================================
// Nested — Struct
// ===========================================================================

// ── test 14: Struct round-trip, field access via Arrow ─────────────────────
//
// DataFusion 47 struct-field access syntax:
//   person['name']  — subscript notation (works in DF 47)
//   person.name     — dot notation (may or may not work depending on quoting)
// We verify the struct array directly after SELECT *, then use subscript in test 15.

#[test]
fn struct_roundtrip_field_access() {
    let schema = Schema::new(vec![ColumnSpec::struct_col(
        "person",
        vec![
            FieldSpec::utf8("name", delta_leb_zstd(), bytes_zstd()),
            FieldSpec::primitive("age", helium::DataType::I32, int_pipe()),
        ],
    )]);
    let tmp = NamedTempFile::new().unwrap();
    let reg = CoderRegistry::default();
    let mut w = HeliumWriter::new(tmp.as_file().try_clone().unwrap(), schema, &reg).unwrap();
    w.write_column(
        "person",
        LogicalColumn::Struct {
            fields: vec![
                (
                    "name".into(),
                    LogicalColumn::Utf8(vec!["Alice".into(), "Bob".into()]),
                ),
                (
                    "age".into(),
                    LogicalColumn::Primitive(ColumnData::I32(vec![30, 25])),
                ),
            ],
        },
    )
    .unwrap();
    w.finish().unwrap();

    tokio_run!(async {
        // SELECT * returns person as a struct column; verify row count.
        let batches = run_sql(tmp.path(), "SELECT * FROM t").await;
        assert_eq!(total_rows(&batches), 2, "struct table should have 2 rows");
        assert_eq!(
            batches[0].num_columns(),
            1,
            "should be 1 column (the struct)"
        );

        // Verify the struct array contains correct name and age values.
        let struct_arr = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        assert_eq!(struct_arr.len(), 2);

        let name_col = struct_arr
            .column_by_name("name")
            .expect("name field in struct");
        let name_arr = name_col.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(name_arr.value(0), "Alice", "first row name");
        assert_eq!(name_arr.value(1), "Bob", "second row name");

        let age_col = struct_arr
            .column_by_name("age")
            .expect("age field in struct");
        let age_arr = age_col.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(age_arr.value(0), 30, "Alice age");
        assert_eq!(age_arr.value(1), 25, "Bob age");
    });
}

// ── test 15: Struct — SQL subscript-notation field access ──────────────────
//
// DataFusion 47 field access confirmed: `column_name['field_name']` syntax works.
// Dot notation `column_name.field_name` also works but the result column is named
// differently. We use subscript here as it's unambiguous.

#[test]
fn struct_sql_field_projection() {
    let schema = Schema::new(vec![ColumnSpec::struct_col(
        "person",
        vec![
            FieldSpec::utf8("name", delta_leb_zstd(), bytes_zstd()),
            FieldSpec::primitive("age", helium::DataType::I32, int_pipe()),
        ],
    )]);
    let tmp = NamedTempFile::new().unwrap();
    let reg = CoderRegistry::default();
    let mut w = HeliumWriter::new(tmp.as_file().try_clone().unwrap(), schema, &reg).unwrap();
    w.write_column(
        "person",
        LogicalColumn::Struct {
            fields: vec![
                (
                    "name".into(),
                    LogicalColumn::Utf8(vec!["Alice".into(), "Bob".into()]),
                ),
                (
                    "age".into(),
                    LogicalColumn::Primitive(ColumnData::I32(vec![30, 25])),
                ),
            ],
        },
    )
    .unwrap();
    w.finish().unwrap();

    tokio_run!(async {
        // DataFusion 47: subscript notation extracts struct field as a scalar column.
        let batches = run_sql(tmp.path(), "SELECT person['name'] FROM t").await;
        let got = collect_str(&batches);
        assert_eq!(
            got,
            vec!["Alice", "Bob"],
            "struct field access via subscript notation"
        );
    });
}

// ===========================================================================
// Nested — List
// ===========================================================================

// ── test 16: List<I32> round-trip ─────────────────────────────────────────

#[test]
fn list_i32_roundtrip() {
    // Rows: [[1,2,3], [], [4,5]]
    // Offsets: [0, 3, 3, 5]
    // Values: [1, 2, 3, 4, 5]
    let schema = Schema::new(vec![ColumnSpec::list(
        "vals",
        LogicalType::Primitive {
            data_type: helium::DataType::I32,
        },
        vec![offset_pipe(), int_pipe()], // [offsets pipeline, values pipeline]
    )]);
    let tmp = NamedTempFile::new().unwrap();
    let reg = CoderRegistry::default();
    let mut w = HeliumWriter::new(tmp.as_file().try_clone().unwrap(), schema, &reg).unwrap();
    w.write_column(
        "vals",
        LogicalColumn::List {
            offsets: vec![0, 3, 3, 5],
            values: Box::new(LogicalColumn::Primitive(ColumnData::I32(vec![
                1, 2, 3, 4, 5,
            ]))),
        },
    )
    .unwrap();
    w.finish().unwrap();

    tokio_run!(async {
        let batches = run_sql(tmp.path(), "SELECT vals FROM t").await;
        assert_eq!(total_rows(&batches), 3, "3 list rows");
        let list_arr = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();

        // Row 0: [1,2,3]
        let row0 = list_arr.value(0);
        let row0_int = row0.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(row0_int.len(), 3);
        assert_eq!(row0_int.value(0), 1);
        assert_eq!(row0_int.value(1), 2);
        assert_eq!(row0_int.value(2), 3);

        // Row 1: []
        let row1 = list_arr.value(1);
        assert_eq!(row1.len(), 0, "row 1 should be empty list");

        // Row 2: [4,5]
        let row2 = list_arr.value(2);
        let row2_int = row2.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(row2_int.len(), 2);
        assert_eq!(row2_int.value(0), 4);
        assert_eq!(row2_int.value(1), 5);
    });
}

// ── test 17: List with WHERE on sibling flat column ────────────────────────

#[test]
fn list_with_where_on_flat_column() {
    // Schema: (id: I64, tags: List<Utf8>)
    // Rows: (1, ["a","b"]), (2, []), (3, ["a"])
    // WHERE id = 2 → 1 row, id=2
    let schema = Schema::new(vec![
        ColumnSpec::primitive("id", helium::DataType::I64, delta_leb_zstd()),
        ColumnSpec::list(
            "tags",
            LogicalType::Utf8,
            // List<Utf8>: offsets + (Utf8 offsets + Utf8 data) = 3 pipelines
            vec![offset_pipe(), offset_pipe(), bytes_zstd()],
        ),
    ]);
    let tmp = NamedTempFile::new().unwrap();
    let reg = CoderRegistry::default();
    let mut w = HeliumWriter::new(tmp.as_file().try_clone().unwrap(), schema, &reg).unwrap();
    w.write_column(
        "id",
        LogicalColumn::Primitive(ColumnData::I64(vec![1, 2, 3])),
    )
    .unwrap();
    w.write_column(
        "tags",
        LogicalColumn::List {
            offsets: vec![0, 2, 2, 3],
            values: Box::new(LogicalColumn::Utf8(vec![
                "a".into(),
                "b".into(),
                "a".into(),
            ])),
        },
    )
    .unwrap();
    w.finish().unwrap();

    tokio_run!(async {
        let batches = run_sql(tmp.path(), "SELECT id FROM t WHERE id = 2").await;
        assert_eq!(total_rows(&batches), 1);
        let ids = collect_i64(&batches);
        assert_eq!(ids, vec![2i64]);
    });
}

// ===========================================================================
// Multi-stripe
// ===========================================================================

// ── test 18: Cross-stripe SELECT with ORDER BY ─────────────────────────────

#[test]
fn multistripe_select_ordered() {
    // 3 stripes: rows 0-9, 10-19, 20-29 in column "x: I64"
    let schema = Schema::new(vec![ColumnSpec::primitive(
        "x",
        helium::DataType::I64,
        delta_leb_zstd(),
    )]);
    let tmp = NamedTempFile::new().unwrap();
    let reg = CoderRegistry::default();
    let mut w = HeliumWriter::new(tmp.as_file().try_clone().unwrap(), schema, &reg).unwrap();
    for stripe in 0..3usize {
        let base = (stripe * 10) as i64;
        let vals: Vec<i64> = (base..base + 10).collect();
        w.write_column("x", LogicalColumn::Primitive(ColumnData::I64(vals)))
            .unwrap();
        w.finish_stripe().unwrap();
    }
    w.finish().unwrap();

    tokio_run!(async {
        let batches = run_sql(tmp.path(), "SELECT x FROM t ORDER BY x").await;
        assert_eq!(total_rows(&batches), 30, "30 total rows across 3 stripes");

        let mut all_vals: Vec<i64> = Vec::new();
        for b in &batches {
            let arr = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            for i in 0..arr.len() {
                all_vals.push(arr.value(i));
            }
        }
        let expected: Vec<i64> = (0..30).collect();
        assert_eq!(
            all_vals, expected,
            "cross-stripe values must be 0..29 in order"
        );
    });
}

// ── test 19: Cross-stripe SUM ─────────────────────────────────────────────

#[test]
fn multistripe_sum() {
    // 0+1+...+29 = 435
    let schema = Schema::new(vec![ColumnSpec::primitive(
        "x",
        helium::DataType::I64,
        delta_leb_zstd(),
    )]);
    let tmp = NamedTempFile::new().unwrap();
    let reg = CoderRegistry::default();
    let mut w = HeliumWriter::new(tmp.as_file().try_clone().unwrap(), schema, &reg).unwrap();
    for stripe in 0..3usize {
        let base = (stripe * 10) as i64;
        let vals: Vec<i64> = (base..base + 10).collect();
        w.write_column("x", LogicalColumn::Primitive(ColumnData::I64(vals)))
            .unwrap();
        w.finish_stripe().unwrap();
    }
    w.finish().unwrap();

    tokio_run!(async {
        let batches = run_sql(tmp.path(), "SELECT sum(x) FROM t").await;
        let arr = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(arr.value(0), 435, "0+1+...+29 = 435");
    });
}

// ===========================================================================
// Edge cases
// ===========================================================================

// ── test 20: Single-row stripe ─────────────────────────────────────────────

#[test]
fn single_row_stripe() {
    let schema = Schema::new(vec![ColumnSpec::primitive(
        "x",
        helium::DataType::I64,
        int_pipe(),
    )]);
    let tmp = NamedTempFile::new().unwrap();
    let reg = CoderRegistry::default();
    let mut w = HeliumWriter::new(tmp.as_file().try_clone().unwrap(), schema, &reg).unwrap();
    w.write_column("x", LogicalColumn::Primitive(ColumnData::I64(vec![42])))
        .unwrap();
    w.finish().unwrap();

    tokio_run!(async {
        let batches = run_sql(tmp.path(), "SELECT count(*) FROM t").await;
        let arr = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(arr.value(0), 1, "single-row stripe should have 1 row");

        let batches2 = run_sql(tmp.path(), "SELECT x FROM t").await;
        let vals = collect_i64(&batches2);
        assert_eq!(vals, vec![42i64], "single-row value must be 42");
    });
}

// ── test 21: Wide column count (50 columns × 10 rows) ─────────────────────

#[test]
fn wide_column_count() {
    let n_cols = 50usize;
    let n_rows = 10usize;

    let cols: Vec<ColumnSpec> = (0..n_cols)
        .map(|i| ColumnSpec::primitive(format!("c{i}"), helium::DataType::I32, int_pipe()))
        .collect();
    let schema = Schema::new(cols);
    let tmp = NamedTempFile::new().unwrap();
    let reg = CoderRegistry::default();
    let mut w = HeliumWriter::new(tmp.as_file().try_clone().unwrap(), schema, &reg).unwrap();
    for i in 0..n_cols {
        let vals: Vec<i32> = (0..n_rows as i32).map(|r| r * 100 + i as i32).collect();
        w.write_column(
            format!("c{i}").as_str(),
            LogicalColumn::Primitive(ColumnData::I32(vals)),
        )
        .unwrap();
    }
    w.finish().unwrap();

    tokio_run!(async {
        // COUNT(*) → 10
        let batches = run_sql(tmp.path(), "SELECT count(*) FROM t").await;
        let arr = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(arr.value(0), 10, "wide table should have 10 rows");

        // SELECT c7 → 10 rows with values [7, 107, 207, ..., 907]
        let batches2 = run_sql(tmp.path(), "SELECT c7 FROM t").await;
        assert_eq!(total_rows(&batches2), 10);
        let col = batches2[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        for r in 0..n_rows {
            let expected = r as i32 * 100 + 7;
            assert_eq!(col.value(r), expected, "c7 row {r} mismatch");
        }
    });
}

// ── test 22: Strings with CSV-unsafe chars (comma, quote, newline) ─────────

#[test]
fn utf8_csv_unsafe_chars() {
    let values = [
        "a,b,c",
        "he said \"hello\"",
        "line1\nline2",
        "col1\tcol2",
        "normal",
    ];
    let schema = Schema::new(vec![ColumnSpec::utf8("s", delta_leb_zstd(), bytes_zstd())]);
    let tmp = NamedTempFile::new().unwrap();
    let reg = CoderRegistry::default();
    let mut w = HeliumWriter::new(tmp.as_file().try_clone().unwrap(), schema, &reg).unwrap();
    w.write_column(
        "s",
        LogicalColumn::Utf8(values.iter().map(|&s| s.to_string()).collect()),
    )
    .unwrap();
    w.finish().unwrap();

    tokio_run!(async {
        let batches = run_sql(tmp.path(), "SELECT s FROM t").await;
        assert_eq!(total_rows(&batches), values.len());
        let mut got = Vec::new();
        for b in &batches {
            let arr = b.column(0).as_any().downcast_ref::<StringArray>().unwrap();
            for i in 0..arr.len() {
                got.push(arr.value(i).to_string());
            }
        }
        for (i, (exp, act)) in values.iter().zip(got.iter()).enumerate() {
            assert_eq!(*exp, act.as_str(), "row {i} CSV-unsafe string mismatch");
        }
    });
}

// ── test 23: Nullable<Utf8> round-trip with mixed nulls ───────────────────

#[test]
fn nullable_utf8_roundtrip() {
    // [Some("hello"), None, Some("world"), None]
    let schema = Schema::new(vec![ColumnSpec::nullable(
        "s",
        LogicalType::Utf8,
        vec![int_pipe(), delta_leb_zstd(), bytes_zstd()],
    )]);
    let tmp = NamedTempFile::new().unwrap();
    let reg = CoderRegistry::default();
    let mut w = HeliumWriter::new(tmp.as_file().try_clone().unwrap(), schema, &reg).unwrap();
    w.write_column(
        "s",
        LogicalColumn::Nullable {
            present: vec![true, false, true, false],
            value: Box::new(LogicalColumn::Utf8(vec!["hello".into(), "world".into()])),
        },
    )
    .unwrap();
    w.finish().unwrap();

    tokio_run!(async {
        let batches = run_sql(tmp.path(), "SELECT s FROM t").await;
        assert_eq!(total_rows(&batches), 4);
        let arr = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert!(!arr.is_null(0));
        assert_eq!(arr.value(0), "hello");
        assert!(arr.is_null(1), "row 1 should be null");
        assert!(!arr.is_null(2));
        assert_eq!(arr.value(2), "world");
        assert!(arr.is_null(3), "row 3 should be null");
    });
}

// ── test 24: Nullable<Binary> round-trip ──────────────────────────────────

#[test]
fn nullable_binary_roundtrip() {
    // [Some([0x01]), None, Some([]), None, Some([0xAB, 0xCD])]
    let schema = Schema::new(vec![ColumnSpec::nullable(
        "b",
        LogicalType::Binary,
        vec![int_pipe(), delta_leb_zstd(), bytes_zstd()],
    )]);
    let tmp = NamedTempFile::new().unwrap();
    let reg = CoderRegistry::default();
    let mut w = HeliumWriter::new(tmp.as_file().try_clone().unwrap(), schema, &reg).unwrap();
    w.write_column(
        "b",
        LogicalColumn::Nullable {
            present: vec![true, false, true, false, true],
            value: Box::new(LogicalColumn::Binary(vec![
                vec![0x01],
                vec![],
                vec![0xAB, 0xCD],
            ])),
        },
    )
    .unwrap();
    w.finish().unwrap();

    tokio_run!(async {
        let batches = run_sql(tmp.path(), "SELECT b FROM t").await;
        assert_eq!(total_rows(&batches), 5);
        let arr = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<BinaryArray>()
            .unwrap();
        assert!(!arr.is_null(0));
        assert_eq!(arr.value(0), &[0x01u8]);
        assert!(arr.is_null(1), "row 1 should be null");
        assert!(!arr.is_null(2));
        assert_eq!(arr.value(2), &[] as &[u8], "row 2 should be empty blob");
        assert!(arr.is_null(3), "row 3 should be null");
        assert_eq!(arr.value(4), &[0xABu8, 0xCD]);
    });
}

// ── test 25: Dictionary{Utf8} round-trip via SQL ─────────────────────────

#[test]
fn dict_utf8_roundtrip_sql() {
    // dictionary: ["cat", "dog", "fish"]
    // indices:    [0, 1, 0, 2, 1]  → ["cat", "dog", "cat", "fish", "dog"]
    let schema = Schema::new(vec![ColumnSpec::new(
        "animal",
        LogicalType::Dictionary {
            inner: Box::new(LogicalType::Utf8),
        },
        vec![
            delta_leb_zstd(), // dict Utf8 offsets
            bytes_zstd(),     // dict Utf8 data
            int_pipe(),       // indices
        ],
    )]);
    let tmp = NamedTempFile::new().unwrap();
    let reg = CoderRegistry::default();
    let mut w = HeliumWriter::new(tmp.as_file().try_clone().unwrap(), schema, &reg).unwrap();
    w.write_column(
        "animal",
        LogicalColumn::Dictionary {
            dictionary: Box::new(LogicalColumn::Utf8(vec![
                "cat".into(),
                "dog".into(),
                "fish".into(),
            ])),
            indices: vec![0, 1, 0, 2, 1],
        },
    )
    .unwrap();
    w.finish().unwrap();

    tokio_run!(async {
        // DataFusion resolves dictionary arrays; result should have string semantics.
        let batches = run_sql(tmp.path(), "SELECT count(*) FROM t WHERE animal = 'cat'").await;
        let arr = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(arr.value(0), 2, "two rows should have animal='cat'");
    });
}

// ── test 26: Two-column mixed query (i64 + Utf8) ──────────────────────────

#[test]
fn two_column_mixed_query() {
    // id: I64, name: Utf8
    // WHERE id > 1 → rows (2,"Bob") and (3,"Carol")
    let schema = Schema::new(vec![
        ColumnSpec::primitive("id", helium::DataType::I64, delta_leb_zstd()),
        ColumnSpec::utf8("name", delta_leb_zstd(), bytes_zstd()),
    ]);
    let tmp = NamedTempFile::new().unwrap();
    let reg = CoderRegistry::default();
    let mut w = HeliumWriter::new(tmp.as_file().try_clone().unwrap(), schema, &reg).unwrap();
    w.write_column(
        "id",
        LogicalColumn::Primitive(ColumnData::I64(vec![1, 2, 3])),
    )
    .unwrap();
    w.write_column(
        "name",
        LogicalColumn::Utf8(vec!["Alice".into(), "Bob".into(), "Carol".into()]),
    )
    .unwrap();
    w.finish().unwrap();

    tokio_run!(async {
        let batches = run_sql(
            tmp.path(),
            "SELECT id, name FROM t WHERE id > 1 ORDER BY id",
        )
        .await;
        assert_eq!(total_rows(&batches), 2);
        let ids = collect_i64(&batches);
        assert_eq!(ids, vec![2i64, 3]);

        let mut names = Vec::new();
        for b in &batches {
            let arr = b.column(1).as_any().downcast_ref::<StringArray>().unwrap();
            for i in 0..arr.len() {
                names.push(arr.value(i).to_string());
            }
        }
        assert_eq!(names, vec!["Bob", "Carol"]);
    });
}

// ── test 27: Nullable<I32> value-equality filter ──────────────────────────

#[test]
fn nullable_i32_equality_filter() {
    // [Some(10), None, Some(20), Some(10), None]
    // WHERE x = 10 → 2 rows
    let schema = Schema::new(vec![ColumnSpec::nullable(
        "x",
        LogicalType::Primitive {
            data_type: helium::DataType::I32,
        },
        vec![int_pipe(), int_pipe()],
    )]);
    let tmp = NamedTempFile::new().unwrap();
    let reg = CoderRegistry::default();
    let mut w = HeliumWriter::new(tmp.as_file().try_clone().unwrap(), schema, &reg).unwrap();
    w.write_column(
        "x",
        LogicalColumn::Nullable {
            present: vec![true, false, true, true, false],
            value: Box::new(LogicalColumn::Primitive(ColumnData::I32(vec![10, 20, 10]))),
        },
    )
    .unwrap();
    w.finish().unwrap();

    tokio_run!(async {
        let batches = run_sql(tmp.path(), "SELECT count(*) FROM t WHERE x = 10").await;
        let arr = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(arr.value(0), 2, "WHERE x = 10 should match 2 rows");
    });
}

// ── test 28: List<Utf8> round-trip ─────────────────────────────────────────

#[test]
fn list_utf8_roundtrip() {
    // Rows: [["hello", "world"], [], ["foo"]]
    let schema = Schema::new(vec![ColumnSpec::list(
        "tags",
        LogicalType::Utf8,
        // List<Utf8>: 1 (outer offsets) + 2 (Utf8 = inner offsets + data) = 3 pipelines
        vec![offset_pipe(), offset_pipe(), bytes_zstd()],
    )]);
    let tmp = NamedTempFile::new().unwrap();
    let reg = CoderRegistry::default();
    let mut w = HeliumWriter::new(tmp.as_file().try_clone().unwrap(), schema, &reg).unwrap();
    w.write_column(
        "tags",
        LogicalColumn::List {
            offsets: vec![0, 2, 2, 3],
            values: Box::new(LogicalColumn::Utf8(vec![
                "hello".into(),
                "world".into(),
                "foo".into(),
            ])),
        },
    )
    .unwrap();
    w.finish().unwrap();

    tokio_run!(async {
        let batches = run_sql(tmp.path(), "SELECT tags FROM t").await;
        assert_eq!(total_rows(&batches), 3);
        let list_arr = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();

        // Row 0: ["hello", "world"]
        let row0 = list_arr.value(0);
        let row0_str = row0.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(row0_str.len(), 2);
        assert_eq!(row0_str.value(0), "hello");
        assert_eq!(row0_str.value(1), "world");

        // Row 1: []
        assert_eq!(list_arr.value(1).len(), 0, "row 1 should be empty list");

        // Row 2: ["foo"]
        let row2 = list_arr.value(2);
        let row2_str = row2.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(row2_str.value(0), "foo");
    });
}

// ── test 29: GROUP BY aggregate correctness ────────────────────────────────

#[test]
fn group_by_sum() {
    // category: Utf8, amount: I64
    // (A,10), (B,20), (A,30), (B,40), (A,50)
    // GROUP BY category → A: 90, B: 60
    let schema = Schema::new(vec![
        ColumnSpec::utf8("cat", delta_leb_zstd(), bytes_zstd()),
        ColumnSpec::primitive("amount", helium::DataType::I64, delta_leb_zstd()),
    ]);
    let tmp = NamedTempFile::new().unwrap();
    let reg = CoderRegistry::default();
    let mut w = HeliumWriter::new(tmp.as_file().try_clone().unwrap(), schema, &reg).unwrap();
    w.write_column(
        "cat",
        LogicalColumn::Utf8(vec![
            "A".into(),
            "B".into(),
            "A".into(),
            "B".into(),
            "A".into(),
        ]),
    )
    .unwrap();
    w.write_column(
        "amount",
        LogicalColumn::Primitive(ColumnData::I64(vec![10, 20, 30, 40, 50])),
    )
    .unwrap();
    w.finish().unwrap();

    tokio_run!(async {
        let batches = run_sql(
            tmp.path(),
            "SELECT cat, sum(amount) FROM t GROUP BY cat ORDER BY cat",
        )
        .await;
        assert_eq!(total_rows(&batches), 2, "2 groups");

        let cat_arr = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let sum_arr = batches[0]
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(cat_arr.value(0), "A");
        assert_eq!(sum_arr.value(0), 90, "A: 10+30+50=90");
        assert_eq!(cat_arr.value(1), "B");
        assert_eq!(sum_arr.value(1), 60, "B: 20+40=60");
    });
}

// ── test 30: Projection column order preserved ─────────────────────────────

#[test]
fn projection_column_order() {
    // Write a, b, c columns. SELECT c, a FROM t → columns in that order.
    let schema = Schema::new(vec![
        ColumnSpec::primitive("a", helium::DataType::I32, int_pipe()),
        ColumnSpec::primitive("b", helium::DataType::I32, int_pipe()),
        ColumnSpec::primitive("c", helium::DataType::I32, int_pipe()),
    ]);
    let tmp = NamedTempFile::new().unwrap();
    let reg = CoderRegistry::default();
    let mut w = HeliumWriter::new(tmp.as_file().try_clone().unwrap(), schema, &reg).unwrap();
    w.write_column("a", LogicalColumn::Primitive(ColumnData::I32(vec![1])))
        .unwrap();
    w.write_column("b", LogicalColumn::Primitive(ColumnData::I32(vec![2])))
        .unwrap();
    w.write_column("c", LogicalColumn::Primitive(ColumnData::I32(vec![3])))
        .unwrap();
    w.finish().unwrap();

    tokio_run!(async {
        let batches = run_sql(tmp.path(), "SELECT c, a FROM t").await;
        assert_eq!(batches[0].num_columns(), 2);
        assert_eq!(batches[0].schema().field(0).name(), "c");
        assert_eq!(batches[0].schema().field(1).name(), "a");
        let c_arr = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let a_arr = batches[0]
            .column(1)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(c_arr.value(0), 3, "c column value");
        assert_eq!(a_arr.value(0), 1, "a column value");
    });
}

// ── test 31: NULL-excluding aggregate (AVG over nullable F64) ─────────────

#[test]
fn avg_over_nullable_excludes_nulls() {
    // [Some(10.0), None, Some(30.0)] → avg of non-null values = (10+30)/2 = 20.0
    let schema = Schema::new(vec![ColumnSpec::nullable(
        "x",
        LogicalType::Primitive {
            data_type: helium::DataType::F64,
        },
        vec![int_pipe(), float_pipe()],
    )]);
    let tmp = NamedTempFile::new().unwrap();
    let reg = CoderRegistry::default();
    let mut w = HeliumWriter::new(tmp.as_file().try_clone().unwrap(), schema, &reg).unwrap();
    w.write_column(
        "x",
        LogicalColumn::Nullable {
            present: vec![true, false, true],
            value: Box::new(LogicalColumn::Primitive(ColumnData::F64(vec![10.0, 30.0]))),
        },
    )
    .unwrap();
    w.finish().unwrap();

    tokio_run!(async {
        let batches = run_sql(tmp.path(), "SELECT avg(x) FROM t").await;
        let arr = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        let v = arr.value(0);
        assert!(
            (v - 20.0).abs() < 1e-9,
            "avg(nullable F64) should be 20.0, got {v}"
        );
    });
}

// ── test 32: Multistripe — correct per-stripe values ──────────────────────

#[test]
fn multistripe_per_stripe_values() {
    // Stripe 0: [100, 200], Stripe 1: [300, 400]
    let schema = Schema::new(vec![ColumnSpec::primitive(
        "x",
        helium::DataType::I64,
        int_pipe(),
    )]);
    let tmp = NamedTempFile::new().unwrap();
    let reg = CoderRegistry::default();
    let mut w = HeliumWriter::new(tmp.as_file().try_clone().unwrap(), schema, &reg).unwrap();
    w.write_column(
        "x",
        LogicalColumn::Primitive(ColumnData::I64(vec![100, 200])),
    )
    .unwrap();
    w.finish_stripe().unwrap();
    w.write_column(
        "x",
        LogicalColumn::Primitive(ColumnData::I64(vec![300, 400])),
    )
    .unwrap();
    w.finish().unwrap();

    tokio_run!(async {
        let batches = run_sql(tmp.path(), "SELECT x FROM t ORDER BY x").await;
        let vals: Vec<i64> = batches
            .iter()
            .flat_map(|b| {
                let arr = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
                (0..arr.len()).map(|i| arr.value(i)).collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(
            vals,
            vec![100, 200, 300, 400],
            "multistripe values in order"
        );
    });
}
