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

/// Regression: when the optimizer promotes a low-cardinality column to a
/// `Dictionary`, `convert --schema` must materialize it from the flat CSV
/// (load the inner column, then dict-encode) — it used to bail with
/// "dict types are not directly loadable from flat files".
#[test]
fn optimize_then_convert_with_dict_promotion() {
    let dir = TempDir::new().unwrap();
    let csv = dir.path().join("low_card.csv");
    // 3000 rows, a low-cardinality column of 6 distinct longish strings —
    // strongly favours dictionary, so the optimizer promotes it.
    let mut s = String::from("id;label\n");
    let cats = [
        "neighbor_cell_region_alpha",
        "neighbor_cell_region_bravo",
        "neighbor_cell_region_charlie",
        "neighbor_cell_region_delta",
        "neighbor_cell_region_echo",
        "neighbor_cell_region_foxtrot",
    ];
    // At 20k rows the measured dict pipeline beats the plain one, so the
    // optimizer promotes the low-cardinality column to a Dictionary.
    for i in 0..20000 {
        s.push_str(&format!("{i};{}\n", cats[i % cats.len()]));
    }
    std::fs::write(&csv, s).unwrap();

    let schema = dir.path().join("schema.json");
    helium()
        .args([
            "optimize-schema",
            csv.to_str().unwrap(),
            "--delimiter",
            ";",
            "--out",
            schema.to_str().unwrap(),
        ])
        .assert()
        .success();

    // The optimizer should have promoted the label column to a dictionary.
    let schema_json = std::fs::read_to_string(&schema).unwrap();
    assert!(
        schema_json.contains("\"dictionary\""),
        "expected a dictionary-promoted column in the optimized schema"
    );

    // The step that used to crash: convert the CSV using that schema.
    let he = dir.path().join("out.he");
    helium()
        .args([
            "convert",
            csv.to_str().unwrap(),
            "-o",
            he.to_str().unwrap(),
            "--schema",
            schema.to_str().unwrap(),
            "--delimiter",
            ";",
        ])
        .assert()
        .success();
    assert!(he.exists(), ".he was not written");

    // And it reads back (round-trip through the dict pipeline).
    let back = dir.path().join("back.csv");
    helium()
        .args(["convert", he.to_str().unwrap(), "-o", back.to_str().unwrap()])
        .assert()
        .success();
    let back_csv = std::fs::read_to_string(&back).unwrap();
    assert_eq!(back_csv.lines().count(), 20001, "header + 20000 rows");
    assert!(back_csv.contains("neighbor_cell_region_charlie"));
}
