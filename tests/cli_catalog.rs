//! Integration tests for the `helium catalog` subcommand group and
//! `--catalog` flag on `helium convert` / `helium verify`.
//!
//! # Coverage
//!
//! 1. `convert --catalog` + `verify --catalog`: CSV → v6 .he round-trip
//! 2. `verify` without `--catalog` on a v6 file fails with "requires schema resolver"
//! 3. `.he` → CSV with `--catalog` (export path with resolver)
//! 4. `helium catalog list <DIR>` lists registered hashes (one per line, 64 hex chars)
//! 5. `helium catalog verify <DIR>` clean → "OK: N schema(s)"
//! 6. `helium catalog verify <DIR>` corrupted → non-zero exit with "catalog inconsistency"
//! 7. `convert --catalog` with neither side `.he` → stderr "no effect" warning (command errors as expected)
#![cfg(feature = "cli")]

use std::fs;
use std::io::Write;
use std::path::PathBuf;

use assert_cmd::Command;
use helium::catalog::Catalog;
use helium::{
    CoderRegistry, CoderSpec, ColumnData, ColumnSpec, DataType, LogicalColumn, MAGIC_V6, Schema,
};
use predicates::prelude::*;
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn helium() -> Command {
    Command::cargo_bin("helium").expect("helium binary not found")
}

/// Write a minimal CSV fixture with 3 columns.
fn write_csv_fixture(dir: &TempDir) -> PathBuf {
    let path = dir.path().join("data.csv");
    let mut f = fs::File::create(&path).unwrap();
    writeln!(f, "id,value,label").unwrap();
    writeln!(f, "1,1.5,alpha").unwrap();
    writeln!(f, "2,2.5,beta").unwrap();
    writeln!(f, "3,3.5,gamma").unwrap();
    path
}

// ---------------------------------------------------------------------------
// 1. convert --catalog (import) + verify --catalog round-trip
// ---------------------------------------------------------------------------

#[test]
fn convert_and_verify_with_catalog() {
    let dir = TempDir::new().unwrap();
    let cat_dir = dir.path().join("catalog");
    fs::create_dir_all(&cat_dir).unwrap();

    let csv = write_csv_fixture(&dir);
    let he = dir.path().join("out.he");

    // Convert CSV → .he with catalog mode.
    helium()
        .arg("convert")
        .arg(&csv)
        .arg("-o")
        .arg(&he)
        .arg("--catalog")
        .arg(&cat_dir)
        .assert()
        .success();

    assert!(he.exists(), ".he file was not created");

    // The file must start with MAGIC_V6 (catalog mode).
    let bytes = fs::read(&he).unwrap();
    assert_eq!(
        &bytes[..8],
        MAGIC_V6,
        "expected MAGIC_V6 for catalog-mode output, got {:?}",
        &bytes[..8]
    );

    // Verify the file using the catalog resolver.
    helium()
        .arg("verify")
        .arg(&he)
        .arg("--catalog")
        .arg(&cat_dir)
        .assert()
        .success()
        .stdout(predicate::str::contains("OK"));
}

// ---------------------------------------------------------------------------
// 2. verify without --catalog on a v6 file fails
// ---------------------------------------------------------------------------

#[test]
fn verify_v6_without_catalog_fails_with_resolver_error() {
    let dir = TempDir::new().unwrap();
    let cat_dir = dir.path().join("catalog");
    fs::create_dir_all(&cat_dir).unwrap();

    let csv = write_csv_fixture(&dir);
    let he = dir.path().join("out.he");

    // Create a v6 file.
    helium()
        .arg("convert")
        .arg(&csv)
        .arg("-o")
        .arg(&he)
        .arg("--catalog")
        .arg(&cat_dir)
        .assert()
        .success();

    // Verify without --catalog should fail with the greppable reason.
    helium()
        .arg("verify")
        .arg(&he)
        .assert()
        .failure()
        .stderr(predicate::str::contains("requires schema resolver"));
}

// ---------------------------------------------------------------------------
// 3. he → CSV with --catalog (export with resolver)
// ---------------------------------------------------------------------------

#[test]
fn export_v6_he_to_csv_with_catalog() {
    let dir = TempDir::new().unwrap();
    let cat_dir = dir.path().join("catalog");

    // Write a v6 file via the API directly (no CLI for write, use library).
    let schema = Schema::new(vec![
        ColumnSpec::primitive(
            "x",
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

    let he_path = dir.path().join("api_written.he");
    {
        let catalog = Catalog::open(&cat_dir).unwrap();
        let registry = CoderRegistry::default();
        let he_file = fs::File::create(&he_path).unwrap();
        let mut w = catalog.open_writer(he_file, schema, &registry).unwrap();
        w.write_column(
            "x",
            LogicalColumn::Primitive(ColumnData::I64(vec![10, 20, 30])),
        )
        .unwrap();
        w.write_column(
            "label",
            LogicalColumn::Utf8(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
        )
        .unwrap();
        w.finish().unwrap();
    }

    // Now export via CLI with --catalog.
    let csv_out = dir.path().join("roundtrip.csv");
    helium()
        .arg("convert")
        .arg(&he_path)
        .arg("-o")
        .arg(&csv_out)
        .arg("--catalog")
        .arg(&cat_dir)
        .assert()
        .success();

    assert!(csv_out.exists(), "output CSV not created");
    let content = fs::read_to_string(&csv_out).unwrap();
    assert!(
        content.contains("10"),
        "expected value '10' in CSV output: {content}"
    );
    assert!(
        content.contains("label"),
        "expected header 'label' in CSV output: {content}"
    );
}

// ---------------------------------------------------------------------------
// 4. helium catalog list <DIR>
// ---------------------------------------------------------------------------

#[test]
fn catalog_list_shows_hashes() {
    let dir = TempDir::new().unwrap();
    let cat_dir = dir.path().join("catalog");

    // Register two distinct schemas via the API.
    let catalog = Catalog::open(&cat_dir).unwrap();
    let schema_a = Schema::new(vec![ColumnSpec::primitive(
        "a",
        DataType::I64,
        vec![CoderSpec::new("zstd")],
    )]);
    let schema_b = Schema::new(vec![ColumnSpec::primitive(
        "b",
        DataType::I32,
        vec![CoderSpec::new("zstd")],
    )]);
    let hash_a = catalog.add_schema(&schema_a).unwrap();
    let hash_b = catalog.add_schema(&schema_b).unwrap();

    // Run `helium catalog list`.
    let output = helium()
        .arg("catalog")
        .arg("list")
        .arg(&cat_dir)
        .assert()
        .success();

    let stdout = String::from_utf8(output.get_output().stdout.clone()).unwrap();
    let lines: Vec<&str> = stdout.lines().filter(|l| !l.trim().is_empty()).collect();

    assert_eq!(
        lines.len(),
        2,
        "expected 2 hash lines, got {}: {stdout}",
        lines.len()
    );

    // Each line must be exactly 64 lowercase hex chars.
    for line in &lines {
        assert_eq!(line.len(), 64, "expected 64-char hash, got: '{line}'");
        assert!(
            line.chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_uppercase()),
            "expected lowercase hex, got: '{line}'"
        );
    }

    // Both registered hashes must appear.
    let expected_a = hash_a.to_hex().to_string();
    let expected_b = hash_b.to_hex().to_string();
    assert!(
        lines.contains(&expected_a.as_str()),
        "hash_a not found in output: {stdout}"
    );
    assert!(
        lines.contains(&expected_b.as_str()),
        "hash_b not found in output: {stdout}"
    );

    // Lines must be sorted lexicographically.
    let mut sorted = lines.clone();
    sorted.sort();
    assert_eq!(
        lines, sorted,
        "output lines must be sorted lexicographically"
    );
}

// ---------------------------------------------------------------------------
// 5. helium catalog verify <DIR> — clean catalog
// ---------------------------------------------------------------------------

#[test]
fn catalog_verify_clean_succeeds() {
    let dir = TempDir::new().unwrap();
    let cat_dir = dir.path().join("catalog");

    // Register two schemas.
    let catalog = Catalog::open(&cat_dir).unwrap();
    catalog
        .add_schema(&Schema::new(vec![ColumnSpec::primitive(
            "a",
            DataType::I64,
            vec![CoderSpec::new("zstd")],
        )]))
        .unwrap();
    catalog
        .add_schema(&Schema::new(vec![ColumnSpec::primitive(
            "b",
            DataType::I32,
            vec![CoderSpec::new("zstd")],
        )]))
        .unwrap();

    helium()
        .arg("catalog")
        .arg("verify")
        .arg(&cat_dir)
        .assert()
        .success()
        .stdout(
            predicate::str::contains("OK").and(predicate::str::contains("2 schema(s) registered")),
        );
}

// ---------------------------------------------------------------------------
// 6. helium catalog verify <DIR> — corrupted catalog
// ---------------------------------------------------------------------------

#[test]
fn catalog_verify_corrupted_fails() {
    let dir = TempDir::new().unwrap();
    let cat_dir = dir.path().join("catalog");

    // Register a schema.
    let catalog = Catalog::open(&cat_dir).unwrap();
    let schema = Schema::new(vec![ColumnSpec::primitive(
        "x",
        DataType::I64,
        vec![CoderSpec::new("zstd")],
    )]);
    let hash = catalog.add_schema(&schema).unwrap();

    // Overwrite the catalog file with different bytes (keeps filename).
    let path = catalog.path_for(&hash);
    fs::write(&path, b"{\"columns\":[]}").unwrap();

    helium()
        .arg("catalog")
        .arg("verify")
        .arg(&cat_dir)
        .assert()
        .failure()
        .stderr(predicate::str::contains("catalog inconsistency"));
}

// ---------------------------------------------------------------------------
// 7. convert --catalog with neither side .he → warning on stderr
// ---------------------------------------------------------------------------

#[test]
fn convert_catalog_neither_side_he_warns() {
    let dir = TempDir::new().unwrap();
    let cat_dir = dir.path().join("catalog");
    fs::create_dir_all(&cat_dir).unwrap();

    // Write a CSV file.
    let csv_in = write_csv_fixture(&dir);
    let csv_out = dir.path().join("out.csv");

    // helium convert csv → csv with --catalog and --to csv.
    // This will fail with "neither .he" error (expected), but the warning
    // should appear on stderr before the error.
    helium()
        .arg("convert")
        .arg(&csv_in)
        .arg("-o")
        .arg(&csv_out)
        .arg("--to")
        .arg("csv")
        .arg("--catalog")
        .arg(&cat_dir)
        .assert()
        .failure()
        .stderr(predicate::str::contains("no effect"));
}
