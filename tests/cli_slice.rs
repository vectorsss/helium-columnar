//! Integration tests for `helium slice` (column projection into a new file).
//!
//! Uses [`assert_cmd`] to invoke the binary; writes a deterministic `.he`
//! source file, slices a column subset, and checks the output.
#![cfg(feature = "cli")]

use std::path::PathBuf;

use assert_cmd::Command;
use helium::{
    CoderRegistry, CoderSpec, ColumnData, ColumnSpec, DataType, HeliumWriter, LogicalColumn, Schema,
};
use predicates::prelude::*;
use tempfile::TempDir;

fn helium() -> Command {
    Command::cargo_bin("helium").expect("helium binary not found")
}

/// Write a 3-column `.he` file: id (I64), name (Utf8), score (I64).
fn write_3col(dir: &TempDir) -> PathBuf {
    let schema = Schema::new(vec![
        ColumnSpec::primitive(
            "id",
            DataType::I64,
            vec![CoderSpec::new("leb128"), CoderSpec::new("zstd")],
        ),
        ColumnSpec::utf8(
            "name",
            vec![CoderSpec::new("leb128"), CoderSpec::new("zstd")],
            vec![CoderSpec::new("zstd")],
        ),
        ColumnSpec::primitive(
            "score",
            DataType::I64,
            vec![CoderSpec::new("leb128"), CoderSpec::new("zstd")],
        ),
    ]);
    let path = dir.path().join("full.he");
    let file = std::fs::File::create(&path).unwrap();
    let mut w = HeliumWriter::new(file, schema, &CoderRegistry::default()).unwrap();
    w.write_column(
        "id",
        LogicalColumn::Primitive(ColumnData::I64(vec![1, 2, 3])),
    )
    .unwrap();
    w.write_column(
        "name",
        LogicalColumn::Utf8(vec!["a".into(), "b".into(), "c".into()]),
    )
    .unwrap();
    w.write_column(
        "score",
        LogicalColumn::Primitive(ColumnData::I64(vec![10, 20, 30])),
    )
    .unwrap();
    w.finish().unwrap();
    path
}

#[test]
fn slice_keeps_only_requested_columns() {
    let dir = TempDir::new().unwrap();
    let src = write_3col(&dir);
    let out = dir.path().join("slice.he");

    helium()
        .args([
            "slice",
            src.to_str().unwrap(),
            "-o",
            out.to_str().unwrap(),
            "--columns",
            "id,score",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("sliced 2 column(s)"));

    // stats on the output: id and score present, name absent.
    helium()
        .args(["stats", out.to_str().unwrap(), "--no-values"])
        .assert()
        .success()
        .stdout(predicate::str::contains("| id "))
        .stdout(predicate::str::contains("| score "))
        .stdout(predicate::str::contains("| name ").not());

    // The sliced file verifies clean.
    helium()
        .args(["verify", out.to_str().unwrap()])
        .assert()
        .success();
}

#[test]
fn slice_missing_column_errors() {
    let dir = TempDir::new().unwrap();
    let src = write_3col(&dir);
    let out = dir.path().join("bad.he");
    helium()
        .args([
            "slice",
            src.to_str().unwrap(),
            "-o",
            out.to_str().unwrap(),
            "--columns",
            "id,nope",
        ])
        .assert()
        .failure();
}
