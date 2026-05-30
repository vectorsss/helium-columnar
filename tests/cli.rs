//! Integration tests for the `helium` binary.
//!
//! Uses [`assert_cmd`] to invoke the binary and check stdout / stderr / exit codes.
//! All tests use deterministic synthetic data written to temporary files.
#![cfg(feature = "cli")]

use std::fs;
use std::io::Write;
use std::path::PathBuf;

use assert_cmd::Command;
use predicates::prelude::*;
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convenience: get an `assert_cmd::Command` for the `helium` binary.
fn helium() -> Command {
    Command::cargo_bin("helium").expect("helium binary not found")
}

/// Write a synthetic CSV file with three columns: id (i64), value (f64), label (utf8).
fn write_test_csv(dir: &TempDir) -> PathBuf {
    let path = dir.path().join("test.csv");
    let mut f = fs::File::create(&path).unwrap();
    writeln!(f, "id,value,label").unwrap();
    for i in 0..100u64 {
        writeln!(f, "{},{:.2},item_{}", i, i as f64 * 1.5, i % 10).unwrap();
    }
    path
}

/// Write a synthetic JSON (array of objects) file.
fn write_test_json(dir: &TempDir) -> PathBuf {
    let path = dir.path().join("test.json");
    let mut records = Vec::new();
    for i in 0..50u64 {
        records.push(format!(r#"{{"id":{},"score":{:.1}}}"#, i, i as f64 * 2.0));
    }
    let content = format!("[{}]", records.join(","));
    fs::write(&path, content).unwrap();
    path
}

// ---------------------------------------------------------------------------
// infer-schema: stdout
// ---------------------------------------------------------------------------

#[test]
fn infer_schema_csv_stdout() {
    let dir = TempDir::new().unwrap();
    let csv = write_test_csv(&dir);

    let mut cmd = helium();
    cmd.arg("infer-schema").arg(&csv);
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("\"columns\""))
        .stdout(predicate::str::contains("\"id\""))
        .stdout(predicate::str::contains("\"value\""))
        .stdout(predicate::str::contains("\"label\""));
}

// ---------------------------------------------------------------------------
// infer-schema: write to file
// ---------------------------------------------------------------------------

#[test]
fn infer_schema_csv_to_file() {
    let dir = TempDir::new().unwrap();
    let csv = write_test_csv(&dir);
    let out = dir.path().join("schema.json");

    let mut cmd = helium();
    cmd.arg("infer-schema").arg(&csv).arg("--out").arg(&out);
    cmd.assert().success();

    assert!(out.exists(), "schema file was not created");
    let content = fs::read_to_string(&out).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();
    assert!(
        parsed.get("columns").is_some(),
        "schema JSON missing 'columns' key"
    );
}

// ---------------------------------------------------------------------------
// convert: infer schema + round-trip read back
// ---------------------------------------------------------------------------

#[test]
fn convert_csv_inferred_schema_roundtrip() {
    let dir = TempDir::new().unwrap();
    let csv = write_test_csv(&dir);
    let he = dir.path().join("out.he");

    // Convert without explicit schema (infer).
    let mut cmd = helium();
    cmd.arg("convert").arg(&csv).arg("-o").arg(&he);
    cmd.assert().success();

    assert!(he.exists(), ".he file was not created");
    assert!(he.metadata().unwrap().len() > 0, ".he file is empty");

    // Verify the file can be read back cleanly.
    let mut vcmd = helium();
    vcmd.arg("verify").arg(&he);
    vcmd.assert()
        .success()
        .stdout(predicate::str::contains("OK"));
}

// ---------------------------------------------------------------------------
// convert: explicit schema file
// ---------------------------------------------------------------------------

#[test]
fn convert_csv_explicit_schema_roundtrip() {
    let dir = TempDir::new().unwrap();
    let csv = write_test_csv(&dir);
    let schema_path = dir.path().join("schema.json");
    let he = dir.path().join("explicit.he");

    // First, infer a schema and save it.
    let mut cmd1 = helium();
    cmd1.arg("infer-schema")
        .arg(&csv)
        .arg("--out")
        .arg(&schema_path);
    cmd1.assert().success();

    // Then convert using the explicit schema.
    let mut cmd2 = helium();
    cmd2.arg("convert")
        .arg(&csv)
        .arg("--schema")
        .arg(&schema_path)
        .arg("-o")
        .arg(&he);
    cmd2.assert().success();

    assert!(he.exists(), ".he file was not created");

    let mut vcmd = helium();
    vcmd.arg("verify").arg(&he);
    vcmd.assert()
        .success()
        .stdout(predicate::str::contains("OK"));
}

// ---------------------------------------------------------------------------
// optimize-schema: produces parseable JSON
// ---------------------------------------------------------------------------

#[test]
fn optimize_schema_csv_stdout() {
    let dir = TempDir::new().unwrap();
    let csv = write_test_csv(&dir);

    let mut cmd = helium();
    cmd.arg("optimize-schema").arg(&csv);
    let output = cmd.assert().success();

    let stdout = String::from_utf8(output.get_output().stdout.clone()).unwrap();
    let parsed: serde_json::Value =
        serde_json::from_str(&stdout).expect("optimize-schema output is not valid JSON");
    assert!(parsed.get("columns").is_some());
}

// ---------------------------------------------------------------------------
// optimize-schema: write to file
// ---------------------------------------------------------------------------

#[test]
fn optimize_schema_csv_to_file() {
    let dir = TempDir::new().unwrap();
    let csv = write_test_csv(&dir);
    let out = dir.path().join("opt_schema.json");

    let mut cmd = helium();
    cmd.arg("optimize-schema").arg(&csv).arg("--out").arg(&out);
    cmd.assert().success();

    assert!(out.exists());
    let content = fs::read_to_string(&out).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();
    assert!(parsed.get("columns").is_some());
}

// ---------------------------------------------------------------------------
// compare: markdown table with at least 3 rows
// ---------------------------------------------------------------------------

#[test]
fn compare_csv_default_codecs() {
    let dir = TempDir::new().unwrap();
    let csv = write_test_csv(&dir);

    let mut cmd = helium();
    cmd.arg("compare").arg(&csv);
    let output = cmd.assert().success();

    let stdout = String::from_utf8(output.get_output().stdout.clone()).unwrap();
    // Should have a markdown table header
    assert!(stdout.contains("Codec"), "missing header row");
    // Should have at least 3 data rows (zstd, lz4, snappy)
    let data_rows: Vec<_> = stdout
        .lines()
        .filter(|l| l.starts_with('|') && !l.contains("Codec") && !l.contains("---"))
        .collect();
    assert!(
        data_rows.len() >= 3,
        "expected ≥3 data rows, got {}: {}",
        data_rows.len(),
        stdout
    );
}

// ---------------------------------------------------------------------------
// compare: custom codec list
// ---------------------------------------------------------------------------

#[test]
fn compare_csv_custom_codecs() {
    let dir = TempDir::new().unwrap();
    let csv = write_test_csv(&dir);

    let mut cmd = helium();
    cmd.arg("compare")
        .arg(&csv)
        .arg("--codecs")
        .arg("zstd,snappy");
    let output = cmd.assert().success();

    let stdout = String::from_utf8(output.get_output().stdout.clone()).unwrap();
    let data_rows: Vec<_> = stdout
        .lines()
        .filter(|l| l.starts_with('|') && !l.contains("Codec") && !l.contains("---"))
        .collect();
    assert_eq!(data_rows.len(), 2, "expected exactly 2 data rows");
    assert!(stdout.contains("zstd"), "missing zstd row");
    assert!(stdout.contains("snappy"), "missing snappy row");
}

// ---------------------------------------------------------------------------
// verify: success on a valid file
// ---------------------------------------------------------------------------

#[test]
fn verify_valid_he_file() {
    let dir = TempDir::new().unwrap();
    let csv = write_test_csv(&dir);
    let he = dir.path().join("verify_test.he");

    // Create a valid .he file.
    let mut cmd = helium();
    cmd.arg("convert").arg(&csv).arg("-o").arg(&he);
    cmd.assert().success();

    let mut vcmd = helium();
    vcmd.arg("verify").arg(&he);
    vcmd.assert()
        .success()
        .stdout(predicate::str::contains("OK"));
}

// ---------------------------------------------------------------------------
// Error cases
// ---------------------------------------------------------------------------

/// Missing input file → non-zero exit, error message mentions the path.
#[test]
fn error_missing_input_file() {
    let mut cmd = helium();
    cmd.arg("infer-schema").arg("/nonexistent/path/missing.csv");
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("error"));
}

/// Invalid schema JSON in --schema flag → non-zero exit with error.
#[test]
fn error_invalid_schema_json() {
    let dir = TempDir::new().unwrap();
    let csv = write_test_csv(&dir);
    let bad_schema = dir.path().join("bad.json");
    fs::write(&bad_schema, b"{ not valid json !!!").unwrap();
    let he = dir.path().join("out.he");

    let mut cmd = helium();
    cmd.arg("convert")
        .arg(&csv)
        .arg("--schema")
        .arg(&bad_schema)
        .arg("-o")
        .arg(&he);
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("error"));
}

/// Unknown subcommand → non-zero exit.
#[test]
fn error_unknown_subcommand() {
    let mut cmd = helium();
    cmd.arg("frobnicate");
    cmd.assert().failure();
}

/// Missing required -o flag for convert → non-zero exit.
#[test]
fn error_convert_missing_output() {
    let dir = TempDir::new().unwrap();
    let csv = write_test_csv(&dir);

    let mut cmd = helium();
    cmd.arg("convert").arg(&csv);
    // No -o → clap should reject it
    cmd.assert().failure();
}

// ---------------------------------------------------------------------------
// JSON input
// ---------------------------------------------------------------------------

#[test]
fn infer_schema_json_stdout() {
    let dir = TempDir::new().unwrap();
    let json = write_test_json(&dir);

    let mut cmd = helium();
    cmd.arg("infer-schema").arg(&json);
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("\"columns\""));
}

#[test]
fn convert_json_roundtrip() {
    let dir = TempDir::new().unwrap();
    let json = write_test_json(&dir);
    let he = dir.path().join("json_out.he");

    let mut cmd = helium();
    cmd.arg("convert").arg(&json).arg("-o").arg(&he);
    cmd.assert().success();

    assert!(he.exists());

    let mut vcmd = helium();
    vcmd.arg("verify").arg(&he);
    vcmd.assert()
        .success()
        .stdout(predicate::str::contains("OK"));
}
