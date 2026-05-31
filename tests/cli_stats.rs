//! Integration tests for `helium stats`.
//!
//! Uses [`assert_cmd`] to invoke the binary and check stdout / stderr / exit
//! codes.  All tests write deterministic `.he` files to temporary directories.
#![cfg(feature = "cli")]

use std::fs::File;
use std::path::PathBuf;

use assert_cmd::Command;
use helium::{
    CoderRegistry, CoderSpec, ColumnData, ColumnSpec, DataType, HeliumWriter, LogicalColumn, Schema,
};
use predicates::prelude::*;
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn helium() -> Command {
    Command::cargo_bin("helium").expect("helium binary not found")
}

/// Write a 3-column `.he` file: id (I64), name (Utf8), score (F64).
fn write_3col_he(dir: &TempDir) -> PathBuf {
    let schema = Schema {
        version: 1,
        columns: vec![
            // I64 → leb128 (NonBlock, I64→I64) → zstd (Block, Bytes→Bytes):
            // leb128 accepts integer types and outputs Bytes, zstd accepts Bytes.
            ColumnSpec::primitive(
                "id",
                DataType::I64,
                vec![CoderSpec::new("leb128"), CoderSpec::new("zstd")],
            ),
            ColumnSpec::utf8(
                "name",
                vec![
                    CoderSpec::new("delta"),
                    CoderSpec::new("leb128"),
                    CoderSpec::new("zstd"),
                ],
                vec![CoderSpec::new("zstd")],
            ),
            // F64 → gorilla (NonBlock, F64→Bytes) → zstd (Block, Bytes→Bytes).
            ColumnSpec::primitive(
                "score",
                DataType::F64,
                vec![CoderSpec::new("gorilla"), CoderSpec::new("zstd")],
            ),
        ],
    };
    let reg = CoderRegistry::default();
    let path = dir.path().join("three.he");
    let file = File::create(&path).unwrap();
    let mut w = HeliumWriter::new(file, schema, &reg).unwrap();
    w.write_column(
        "id",
        LogicalColumn::Primitive(ColumnData::I64(vec![1, 2, 3, 4, 5])),
    )
    .unwrap();
    w.write_column(
        "name",
        LogicalColumn::Utf8(vec![
            "a".into(),
            "b".into(),
            "c".into(),
            "d".into(),
            "e".into(),
        ]),
    )
    .unwrap();
    w.write_column(
        "score",
        LogicalColumn::Primitive(ColumnData::F64(vec![1.1, 2.2, 3.3, 4.4, 5.5])),
    )
    .unwrap();
    w.finish().unwrap();
    path
}

/// Write a single-column `.he` file with known I64 values.
fn write_known_i64(dir: &TempDir, name: &str, values: Vec<i64>) -> PathBuf {
    let schema = Schema {
        version: 1,
        columns: vec![ColumnSpec::primitive(
            "data",
            DataType::I64,
            vec![CoderSpec::new("leb128"), CoderSpec::new("zstd")],
        )],
    };
    let reg = CoderRegistry::default();
    let path = dir.path().join(name);
    let file = File::create(&path).unwrap();
    let mut w = HeliumWriter::new(file, schema, &reg).unwrap();
    w.write_column("data", LogicalColumn::Primitive(ColumnData::I64(values)))
        .unwrap();
    w.finish().unwrap();
    path
}

/// Write a single nullable-prim column file.
fn write_nullable_i64(dir: &TempDir, values: Vec<Option<i64>>) -> PathBuf {
    let schema = Schema {
        version: 1,
        columns: vec![ColumnSpec::nullable_prim(
            "val",
            DataType::I64,
            // present bitmap: U8 → rle (NonBlock, U8→U8) → leb128 (NonBlock, U8→Bytes) → zstd.
            vec![
                CoderSpec::new("rle"),
                CoderSpec::new("leb128"),
                CoderSpec::new("zstd"),
            ],
            // values: I64 → leb128 → zstd.
            vec![CoderSpec::new("leb128"), CoderSpec::new("zstd")],
        )],
    };
    let reg = CoderRegistry::default();
    let path = dir.path().join("nullable.he");
    let file = File::create(&path).unwrap();
    let mut w = HeliumWriter::new(file, schema, &reg).unwrap();
    let present: Vec<bool> = values.iter().map(|v| v.is_some()).collect();
    let vals: Vec<i64> = values.iter().filter_map(|v| *v).collect();
    w.write_column(
        "val",
        LogicalColumn::NullablePrim {
            present,
            values: ColumnData::I64(vals),
        },
    )
    .unwrap();
    w.finish().unwrap();
    path
}

/// Write a 2-stripe `.he` file.
fn write_two_stripe_he(dir: &TempDir) -> PathBuf {
    let schema = Schema {
        version: 1,
        columns: vec![ColumnSpec::primitive(
            "x",
            DataType::I64,
            vec![CoderSpec::new("leb128"), CoderSpec::new("zstd")],
        )],
    };
    let reg = CoderRegistry::default();
    let path = dir.path().join("stripes.he");
    let file = File::create(&path).unwrap();
    let mut w = HeliumWriter::new(file, schema, &reg).unwrap();

    // Stripe 1.
    w.write_column(
        "x",
        LogicalColumn::Primitive(ColumnData::I64(vec![1, 2, 3])),
    )
    .unwrap();
    w.finish_stripe().unwrap();

    // Stripe 2.
    w.write_column("x", LogicalColumn::Primitive(ColumnData::I64(vec![4, 5])))
        .unwrap();
    w.finish().unwrap();
    path
}

// ---------------------------------------------------------------------------
// Test 1 — basic markdown output
// ---------------------------------------------------------------------------

#[test]
fn stats_basic_markdown() {
    let dir = TempDir::new().unwrap();
    let he = write_3col_he(&dir);

    helium()
        .arg("stats")
        .arg(&he)
        .assert()
        .success()
        .stdout(predicate::str::contains("id"))
        .stdout(predicate::str::contains("name"))
        .stdout(predicate::str::contains("score"))
        .stdout(predicate::str::contains("Bytes"))
        .stdout(predicate::str::contains("File:"))
        .stdout(predicate::str::contains("Format:"));
}

// ---------------------------------------------------------------------------
// Test 2 — JSON output
// ---------------------------------------------------------------------------

#[test]
fn stats_json_output() {
    let dir = TempDir::new().unwrap();
    let he = write_3col_he(&dir);

    let output = helium()
        .arg("stats")
        .arg(&he)
        .arg("--json")
        .output()
        .unwrap();

    assert!(output.status.success(), "expected success, got: {output:?}");
    let stdout = String::from_utf8(output.stdout).unwrap();
    let parsed: serde_json::Value =
        serde_json::from_str(&stdout).expect("stdout should be valid JSON");
    let cols = parsed["columns"]
        .as_array()
        .expect("columns should be an array");
    assert_eq!(cols.len(), 3, "expected 3 columns");
    // All column names should be present.
    let names: Vec<&str> = cols.iter().map(|c| c["name"].as_str().unwrap()).collect();
    assert!(names.contains(&"id"));
    assert!(names.contains(&"name"));
    assert!(names.contains(&"score"));
}

// ---------------------------------------------------------------------------
// Test 3 — --no-values
// ---------------------------------------------------------------------------

#[test]
fn stats_no_values() {
    let dir = TempDir::new().unwrap();
    let he = write_3col_he(&dir);

    let output = helium()
        .arg("stats")
        .arg(&he)
        .arg("--no-values")
        .arg("--json")
        .output()
        .unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    for col in parsed["columns"].as_array().unwrap() {
        // With --no-values, min/max should be null.
        assert!(
            col["min"].is_null(),
            "expected null min for {}, got {}",
            col["name"],
            col["min"]
        );
        assert!(
            col["max"].is_null(),
            "expected null max for {}, got {}",
            col["name"],
            col["max"]
        );
    }
}

// ---------------------------------------------------------------------------
// Test 3b — --no-values in markdown shows "—" for min/max
// ---------------------------------------------------------------------------

#[test]
fn stats_no_values_markdown_dash() {
    let dir = TempDir::new().unwrap();
    let he = write_3col_he(&dir);

    helium()
        .arg("stats")
        .arg(&he)
        .arg("--no-values")
        .assert()
        .success()
        // The em-dash character "—" should appear in the min/max columns.
        .stdout(predicate::str::contains('\u{2014}'));
}

// ---------------------------------------------------------------------------
// Test 4 — min/max correctness on known values
// ---------------------------------------------------------------------------

#[test]
fn stats_min_max_correctness() {
    let dir = TempDir::new().unwrap();
    let values = vec![3i64, 1, 4, 1, 5, 9, 2, 6];
    let he = write_known_i64(&dir, "known.he", values);

    let output = helium()
        .arg("stats")
        .arg(&he)
        .arg("--json")
        .output()
        .unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    let col = &parsed["columns"][0];
    assert_eq!(col["min"], serde_json::json!(1), "min should be 1");
    assert_eq!(col["max"], serde_json::json!(9), "max should be 9");
}

// ---------------------------------------------------------------------------
// Test 5 — nullable column: nulls excluded from min/max
// ---------------------------------------------------------------------------

#[test]
fn stats_nullable_excludes_nulls() {
    let dir = TempDir::new().unwrap();
    // Values: [1, null, 3, null, 5] → min=1, max=5, rows_non_null=3
    let he = write_nullable_i64(&dir, vec![Some(1), None, Some(3), None, Some(5)]);

    let output = helium()
        .arg("stats")
        .arg(&he)
        .arg("--json")
        .output()
        .unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    let col = &parsed["columns"][0];
    assert_eq!(col["min"], serde_json::json!(1), "min should be 1");
    assert_eq!(col["max"], serde_json::json!(5), "max should be 5");
    assert_eq!(
        col["rows_non_null"],
        serde_json::json!(3),
        "rows_non_null should be 3"
    );
}

// ---------------------------------------------------------------------------
// Test 6 — all-null column: min/max are null
// ---------------------------------------------------------------------------

#[test]
fn stats_all_null_column() {
    let dir = TempDir::new().unwrap();
    let he = write_nullable_i64(&dir, vec![None, None, None]);

    let output = helium()
        .arg("stats")
        .arg(&he)
        .arg("--json")
        .output()
        .unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    let col = &parsed["columns"][0];
    assert!(col["min"].is_null(), "all-null: min should be null");
    assert!(col["max"].is_null(), "all-null: max should be null");
}

// ---------------------------------------------------------------------------
// Test 7 — multi-stripe: rows is the sum, bytes span both stripes
// ---------------------------------------------------------------------------

#[test]
fn stats_multi_stripe_rows_and_bytes() {
    let dir = TempDir::new().unwrap();
    let he = write_two_stripe_he(&dir);

    let output = helium()
        .arg("stats")
        .arg(&he)
        .arg("--json")
        .output()
        .unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&stdout).unwrap();

    // Total rows should be 3 + 2 = 5.
    assert_eq!(
        parsed["rows"],
        serde_json::json!(5),
        "total rows should be 5"
    );
    assert_eq!(
        parsed["stripes"],
        serde_json::json!(2),
        "stripe count should be 2"
    );

    // The column bytes should be > 0.
    let col_bytes = parsed["columns"][0]["bytes"].as_u64().unwrap();
    assert!(col_bytes > 0, "column bytes should be positive");
}

// ---------------------------------------------------------------------------
// Test 8 — --catalog: smoke test (catalog-mode file readable via catalog)
// ---------------------------------------------------------------------------

#[test]
fn stats_with_catalog() {
    use helium::catalog::Catalog;

    let dir = TempDir::new().unwrap();
    let catalog_dir = dir.path().join("catalog");
    std::fs::create_dir_all(&catalog_dir).unwrap();

    let schema = Schema {
        version: 1,
        columns: vec![ColumnSpec::primitive(
            "x",
            DataType::I64,
            vec![CoderSpec::new("leb128"), CoderSpec::new("zstd")],
        )],
    };
    let reg = CoderRegistry::default();

    let catalog = Catalog::open(&catalog_dir).unwrap();
    let he_path = dir.path().join("catalog_mode.he");
    let file = File::create(&he_path).unwrap();
    let mut w = catalog.open_writer(file, schema, &reg).unwrap();
    w.write_column(
        "x",
        LogicalColumn::Primitive(ColumnData::I64(vec![10, 20, 30])),
    )
    .unwrap();
    w.finish().unwrap();

    helium()
        .arg("stats")
        .arg(&he_path)
        .arg("--catalog")
        .arg(&catalog_dir)
        .assert()
        .success()
        .stdout(predicate::str::contains("x"));
}

// ---------------------------------------------------------------------------
// Test 9 — error: nonexistent file
// ---------------------------------------------------------------------------

#[test]
fn stats_nonexistent_file() {
    helium()
        .arg("stats")
        .arg("/tmp/helium_stats_does_not_exist_abc123.he")
        .assert()
        .failure()
        .stderr(predicate::str::contains("error"));
}

// ---------------------------------------------------------------------------
// Test: region_sizes are positive and sum to file size
// ---------------------------------------------------------------------------

#[test]
fn region_sizes_sum_to_file_size() {
    use helium::HeliumReader;

    let dir = TempDir::new().unwrap();
    let he = write_3col_he(&dir);
    let file_meta = std::fs::metadata(&he).unwrap();
    let file_size = file_meta.len();

    let reg = CoderRegistry::default();
    let f = File::open(&he).unwrap();
    let reader = HeliumReader::new(f, &reg).unwrap();
    let (h, b, f_size) = reader.region_sizes();
    assert_eq!(
        h + b + f_size,
        file_size,
        "region sizes should sum to file size"
    );
    assert!(h > 0, "header bytes should be positive");
    assert!(b > 0, "body bytes should be positive");
    assert!(f_size > 0, "footer bytes should be positive");
}

// ---------------------------------------------------------------------------
// Test: column_byte_sizes returns correct column count
// ---------------------------------------------------------------------------

#[test]
fn column_byte_sizes_count() {
    use helium::HeliumReader;

    let dir = TempDir::new().unwrap();
    let he = write_3col_he(&dir);

    let reg = CoderRegistry::default();
    let f = File::open(&he).unwrap();
    let reader = HeliumReader::new(f, &reg).unwrap();
    let sizes = reader.column_byte_sizes();
    assert_eq!(sizes.len(), 3, "should return 3 entries");
    assert_eq!(sizes[0].0, "id");
    assert_eq!(sizes[1].0, "name");
    assert_eq!(sizes[2].0, "score");
    for (name, bytes) in &sizes {
        assert!(
            *bytes > 0,
            "column '{name}' should have positive byte count"
        );
    }
}
