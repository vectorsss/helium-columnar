//! Integration tests for `helium sql` CLI subcommand.
//!
//! Uses `assert_cmd` to invoke the `helium` binary and verify stdout / stderr
//! / exit codes for SQL queries over synthetic `.he` files.
//!
//! All tests require both the `cli` and `datafusion` features.

#![cfg(all(feature = "cli", feature = "datafusion"))]

use std::fs;
use std::path::PathBuf;

use assert_cmd::Command;
use predicates::prelude::*;
use tempfile::{NamedTempFile, TempDir};

use helium::{
    CoderRegistry, CoderSpec, ColumnData, ColumnSpec, DataType, HeliumWriter, LogicalColumn,
    LogicalType, Schema,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Get a `Command` for the `helium` binary.
fn helium_cmd() -> Command {
    Command::cargo_bin("helium").expect("helium binary not found")
}

/// Write a simple `.he` file with one I64 `id` column and N rows.
/// Returns a `NamedTempFile` whose path is valid for the test's lifetime.
fn write_id_file(rows: u64) -> NamedTempFile {
    let schema = Schema::new(vec![ColumnSpec::primitive(
        "id",
        DataType::I64,
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

    let ids: Vec<i64> = (0..rows as i64).collect();
    writer
        .write_column("id", LogicalColumn::Primitive(ColumnData::I64(ids)))
        .expect("write id");
    writer.finish().expect("finish");
    tmp
}

/// Write a two-column `.he` file with `id: I64` and `label: Utf8`.
fn write_labeled_file(rows: u64) -> NamedTempFile {
    let schema = Schema::new(vec![
        ColumnSpec::primitive(
            "id",
            DataType::I64,
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
    let labels: Vec<String> = (0..rows).map(|i| format!("item_{i}")).collect();

    writer
        .write_column("id", LogicalColumn::Primitive(ColumnData::I64(ids)))
        .expect("write id");
    writer
        .write_column("label", LogicalColumn::Utf8(labels))
        .expect("write label");
    writer.finish().expect("finish");
    tmp
}

/// Write a nullable I32 `.he` file with `score: Nullable<I32>`.
fn write_nullable_file(rows: u64) -> NamedTempFile {
    let schema = Schema::new(vec![ColumnSpec::nullable(
        "score",
        LogicalType::Primitive {
            data_type: DataType::I32,
        },
        vec![
            vec![CoderSpec::new("leb128"), CoderSpec::new("zstd")],
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

    // Alternate present/absent: row 0 non-null, row 1 null, etc.
    let present: Vec<bool> = (0..rows).map(|i| i % 2 == 0).collect();
    let non_null_count = present.iter().filter(|&&b| b).count();
    let values: Vec<i32> = (0..non_null_count as i32).map(|i| i * 10).collect();

    writer
        .write_column(
            "score",
            LogicalColumn::Nullable {
                present,
                value: Box::new(LogicalColumn::Primitive(ColumnData::I32(values))),
            },
        )
        .expect("write score");
    writer.finish().expect("finish");
    tmp
}

/// Copy a `NamedTempFile` to a named path inside `dir` (with the given filename).
/// This is needed so the file stem becomes the table name.
fn copy_to_named(src: &NamedTempFile, dir: &TempDir, filename: &str) -> PathBuf {
    let dest = dir.path().join(filename);
    fs::copy(src.path(), &dest).expect("copy");
    dest
}

// ---------------------------------------------------------------------------
// Test 1 — Basic SELECT count(*): stdout contains the row count
// ---------------------------------------------------------------------------

#[test]
fn sql_count_star_five_rows() {
    let tmp = write_id_file(5);
    let dir = TempDir::new().unwrap();
    let he = copy_to_named(&tmp, &dir, "data.he");

    helium_cmd()
        .arg("sql")
        .arg("SELECT count(*) FROM data")
        .arg(he.to_str().unwrap())
        .assert()
        .success()
        .stdout(predicate::str::contains("5"));
}

// ---------------------------------------------------------------------------
// Test 2 — Default table naming: stem of filename (without .he)
// ---------------------------------------------------------------------------

#[test]
fn sql_default_table_name_from_stem() {
    let tmp = write_id_file(3);
    let dir = TempDir::new().unwrap();
    let he = copy_to_named(&tmp, &dir, "events.he");

    // Table name should be "events" by default.
    helium_cmd()
        .arg("sql")
        .arg("SELECT count(*) FROM events")
        .arg(he.to_str().unwrap())
        .assert()
        .success()
        .stdout(predicate::str::contains("3"));
}

// ---------------------------------------------------------------------------
// Test 3 — Explicit name= override
// ---------------------------------------------------------------------------

#[test]
fn sql_explicit_name_override() {
    let tmp = write_id_file(4);
    let dir = TempDir::new().unwrap();
    let he = copy_to_named(&tmp, &dir, "data.he");

    // Override table name to "x".
    let name_arg = format!("x={}", he.to_str().unwrap());

    helium_cmd()
        .arg("sql")
        .arg("SELECT count(*) FROM x")
        .arg(&name_arg)
        .assert()
        .success()
        .stdout(predicate::str::contains("4"));
}

// ---------------------------------------------------------------------------
// Test 4 — Multi-file JOIN
// ---------------------------------------------------------------------------

#[test]
fn sql_multi_file_join() {
    // Write file a: id I64 (0..5)
    let tmp_a = write_id_file(5);
    // Write file b: id I64 + label Utf8
    let tmp_b = write_labeled_file(5);

    let dir = TempDir::new().unwrap();
    let he_a = copy_to_named(&tmp_a, &dir, "a.he");
    let he_b = copy_to_named(&tmp_b, &dir, "b.he");

    // JOIN on id — should return 5 matched rows.
    helium_cmd()
        .arg("sql")
        .arg("SELECT count(*) FROM a JOIN b ON a.id = b.id")
        .arg(he_a.to_str().unwrap())
        .arg(he_b.to_str().unwrap())
        .assert()
        .success()
        .stdout(predicate::str::contains("5"));
}

// ---------------------------------------------------------------------------
// Test 5 — Error: nonexistent file
// ---------------------------------------------------------------------------

#[test]
fn sql_error_nonexistent_file() {
    helium_cmd()
        .arg("sql")
        .arg("SELECT count(*) FROM missing")
        .arg("/nonexistent/absolutely/missing.he")
        .assert()
        .failure()
        .stderr(predicate::str::contains("error"));
}

// ---------------------------------------------------------------------------
// Test 6 — Error: invalid SQL
// ---------------------------------------------------------------------------

#[test]
fn sql_error_invalid_sql() {
    let tmp = write_id_file(3);
    let dir = TempDir::new().unwrap();
    let he = copy_to_named(&tmp, &dir, "t.he");

    helium_cmd()
        .arg("sql")
        .arg("THIS IS NOT VALID SQL !!!!")
        .arg(he.to_str().unwrap())
        .assert()
        .failure()
        .stderr(predicate::str::contains("error"));
}

// ---------------------------------------------------------------------------
// Test 7 — Error: duplicate table names
// ---------------------------------------------------------------------------

#[test]
fn sql_error_duplicate_table_names() {
    let tmp = write_id_file(3);
    let dir = TempDir::new().unwrap();
    let he = copy_to_named(&tmp, &dir, "a.he");
    let he_str = he.to_str().unwrap();

    // Pass the same file twice — both derive table name "a".
    helium_cmd()
        .arg("sql")
        .arg("SELECT count(*) FROM a")
        .arg(he_str)
        .arg(he_str)
        .assert()
        .failure()
        .stderr(predicate::str::contains("duplicate table name 'a'"));
}

// ---------------------------------------------------------------------------
// Test 8 — SELECT * returns rows and columns
// ---------------------------------------------------------------------------

#[test]
fn sql_select_star() {
    let tmp = write_labeled_file(3);
    let dir = TempDir::new().unwrap();
    let he = copy_to_named(&tmp, &dir, "rows.he");

    // Should succeed and print something containing column names or values.
    helium_cmd()
        .arg("sql")
        .arg("SELECT * FROM rows")
        .arg(he.to_str().unwrap())
        .assert()
        .success()
        // Pretty-printed output has a header line with column names.
        .stdout(predicate::str::contains("id"))
        .stdout(predicate::str::contains("label"));
}

// ---------------------------------------------------------------------------
// Test 9 — WHERE filter works
// ---------------------------------------------------------------------------

#[test]
fn sql_where_filter() {
    let tmp = write_id_file(10);
    let dir = TempDir::new().unwrap();
    let he = copy_to_named(&tmp, &dir, "nums.he");

    // id values 0..10 — rows where id > 7 → ids 8 and 9 → count = 2
    helium_cmd()
        .arg("sql")
        .arg("SELECT count(*) FROM nums WHERE id > 7")
        .arg(he.to_str().unwrap())
        .assert()
        .success()
        .stdout(predicate::str::contains("2"));
}

// ---------------------------------------------------------------------------
// Test 10 — Nullable column query
// ---------------------------------------------------------------------------

#[test]
fn sql_nullable_column() {
    // 6 rows: present pattern [true,false,true,false,true,false]
    // → 3 non-null, 3 null.
    let tmp = write_nullable_file(6);
    let dir = TempDir::new().unwrap();
    let he = copy_to_named(&tmp, &dir, "nullable.he");

    helium_cmd()
        .arg("sql")
        .arg("SELECT count(*) FROM nullable WHERE score IS NOT NULL")
        .arg(he.to_str().unwrap())
        .assert()
        .success()
        .stdout(predicate::str::contains("3"));
}

// ---------------------------------------------------------------------------
// Test 11 — Smoke: hits_1.he if available (skipped when file is absent)
// ---------------------------------------------------------------------------

#[test]
fn sql_smoke_hits_1() {
    // Look for hits_1.he relative to the manifest directory or current dir.
    let candidates = [
        PathBuf::from("hits_1.he"),
        PathBuf::from("../hits_1.he"),
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("hits_1.he"),
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../hits_1.he"),
    ];

    let he_path = candidates.iter().find(|p| p.exists());
    let Some(he_path) = he_path else {
        eprintln!("sql_smoke_hits_1: hits_1.he not found — skipping");
        return;
    };

    use std::time::Instant;
    let t0 = Instant::now();

    helium_cmd()
        .arg("sql")
        .arg("SELECT count(*) FROM hits_1")
        .arg(he_path.to_str().unwrap())
        .assert()
        .success()
        .stdout(predicate::str::contains("1000000"));

    eprintln!(
        "sql_smoke_hits_1: SELECT count(*) FROM hits_1 took {:.3}s",
        t0.elapsed().as_secs_f64()
    );
}
