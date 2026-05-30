//! Integration tests for `helium optimize-schema`, focused on `--sample-rows`.
#![cfg(feature = "cli")]

use assert_cmd::Command;
use predicates::prelude::*;
use tempfile::TempDir;

fn helium() -> Command {
    Command::cargo_bin("helium").expect("helium binary not found")
}

/// Write a small CSV with more rows than the sample cap.
fn write_csv(dir: &TempDir) -> std::path::PathBuf {
    let path = dir.path().join("data.csv");
    let mut s = String::from("id,label\n");
    for i in 0..50 {
        s.push_str(&format!("{i},row{}\n", i % 3));
    }
    std::fs::write(&path, s).unwrap();
    path
}

#[test]
fn optimize_schema_sampled_emits_valid_schema() {
    let dir = TempDir::new().unwrap();
    let csv = write_csv(&dir);
    let out = dir.path().join("schema.json");

    // Sample only the first 5 rows (exercises the early-stop sampling path).
    helium()
        .args([
            "optimize-schema",
            csv.to_str().unwrap(),
            "--sample-rows",
            "5",
            "--out",
            out.to_str().unwrap(),
        ])
        .assert()
        .success();

    let json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&out).unwrap()).expect("valid schema JSON");
    let cols = json["columns"].as_array().expect("columns array");
    assert_eq!(cols.len(), 2, "id + label");

    // The emitted schema applies to the FULL file (all 50 rows convert + verify).
    let he = dir.path().join("data.he");
    helium()
        .args([
            "convert",
            csv.to_str().unwrap(),
            "-o",
            he.to_str().unwrap(),
            "--schema",
            out.to_str().unwrap(),
        ])
        .assert()
        .success();
    helium()
        .args(["verify", he.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("50 rows"));
}

#[test]
fn optimize_schema_whole_file_with_zero() {
    let dir = TempDir::new().unwrap();
    let csv = write_csv(&dir);
    // --sample-rows 0 = whole file; should still succeed.
    helium()
        .args([
            "optimize-schema",
            csv.to_str().unwrap(),
            "--sample-rows",
            "0",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"columns\""));
}
