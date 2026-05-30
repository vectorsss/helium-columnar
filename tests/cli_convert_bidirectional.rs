//! Bidirectional `helium convert` integration tests.
//!
//! Covers:
//! 1. CSV → he → CSV round-trip
//! 2. JSON (NDJSON) → he → JSON round-trip
//! 3. Parquet → he → Parquet round-trip (flat schema)
//! 4. he → csv with explicit `--from he --to csv` flags
//! 5. Error: both extensions `.he` (he → he)
//! 6. Error: neither extension `.he` (csv → csv)
//! 7. Error: `--to avsc` (Avro export not supported)
//! 8. `--from` / `--to` override extension (data.txt with `--from csv`)
//! 9. Parquet with OPTIONAL columns → he (regression: production loader
//!    must produce v3-shaped LogicalColumn::Nullable, not v2 NullablePrim)
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

fn helium() -> Command {
    Command::cargo_bin("helium").expect("helium binary not found")
}

/// Write a small CSV fixture (3 columns: id, value, label).
fn write_csv_fixture(dir: &TempDir) -> PathBuf {
    let path = dir.path().join("data.csv");
    let mut f = fs::File::create(&path).unwrap();
    writeln!(f, "id,value,label").unwrap();
    writeln!(f, "1,1.5,alpha").unwrap();
    writeln!(f, "2,2.5,beta").unwrap();
    writeln!(f, "3,3.5,gamma").unwrap();
    path
}

/// Write a small NDJSON fixture (2 columns: id, score).
fn write_ndjson_fixture(dir: &TempDir) -> PathBuf {
    let path = dir.path().join("data.json");
    let mut f = fs::File::create(&path).unwrap();
    writeln!(f, r#"{{"id":1,"score":10}}"#).unwrap();
    writeln!(f, r#"{{"id":2,"score":20}}"#).unwrap();
    writeln!(f, r#"{{"id":3,"score":30}}"#).unwrap();
    path
}

/// Write a small Parquet fixture using the parquet crate directly.
fn write_parquet_fixture(dir: &TempDir) -> PathBuf {
    use parquet::basic::{ConvertedType, Repetition, Type as PqPhysical};
    use parquet::data_type::{ByteArray, ByteArrayType, Int64Type};
    use parquet::file::properties::WriterProperties;
    use parquet::file::writer::SerializedFileWriter;
    use parquet::schema::types::Type as PqType;
    use std::sync::Arc;

    let path = dir.path().join("data.parquet");
    let file = fs::File::create(&path).unwrap();

    let id_field = PqType::primitive_type_builder("id", PqPhysical::INT64)
        .with_repetition(Repetition::REQUIRED)
        .with_converted_type(ConvertedType::INT_64)
        .build()
        .unwrap();
    let label_field = PqType::primitive_type_builder("label", PqPhysical::BYTE_ARRAY)
        .with_repetition(Repetition::REQUIRED)
        .with_converted_type(ConvertedType::UTF8)
        .build()
        .unwrap();
    let schema = Arc::new(
        PqType::group_type_builder("schema")
            .with_fields(vec![Arc::new(id_field), Arc::new(label_field)])
            .build()
            .unwrap(),
    );

    let props = Arc::new(WriterProperties::builder().build());
    let mut writer = SerializedFileWriter::new(file, schema, props).unwrap();
    let mut rg = writer.next_row_group().unwrap();

    // Write id column.
    let mut col = rg.next_column().unwrap().unwrap();
    col.typed::<Int64Type>()
        .write_batch(&[1, 2, 3], None, None)
        .unwrap();
    col.close().unwrap();

    // Write label column.
    let labels: Vec<ByteArray> = vec![
        ByteArray::from("alice".as_bytes()),
        ByteArray::from("bob".as_bytes()),
        ByteArray::from("carol".as_bytes()),
    ];
    let mut col = rg.next_column().unwrap().unwrap();
    col.typed::<ByteArrayType>()
        .write_batch(&labels, None, None)
        .unwrap();
    col.close().unwrap();

    rg.close().unwrap();
    writer.close().unwrap();
    path
}

// ---------------------------------------------------------------------------
// 1. CSV → he → CSV round-trip
// ---------------------------------------------------------------------------

#[test]
fn csv_to_he_to_csv_roundtrip() {
    let dir = TempDir::new().unwrap();
    let csv_in = write_csv_fixture(&dir);
    let he_path = dir.path().join("data.he");
    let csv_out = dir.path().join("out.csv");

    // Convert CSV → he.
    helium()
        .arg("convert")
        .arg(&csv_in)
        .arg("-o")
        .arg(&he_path)
        .assert()
        .success();
    assert!(he_path.exists(), ".he file not created");

    // Convert he → CSV.
    helium()
        .arg("convert")
        .arg(&he_path)
        .arg("-o")
        .arg(&csv_out)
        .assert()
        .success();
    assert!(csv_out.exists(), "output CSV not created");

    let out_content = fs::read_to_string(&csv_out).unwrap();
    // Header must be present.
    assert!(
        out_content.contains("id"),
        "output CSV missing 'id' column: {out_content}"
    );
    // Values must round-trip.
    assert!(
        out_content.contains("alpha"),
        "output CSV missing 'alpha': {out_content}"
    );
    assert!(
        out_content.contains("beta"),
        "output CSV missing 'beta': {out_content}"
    );
}

// ---------------------------------------------------------------------------
// 2. JSON → he → JSON round-trip
// ---------------------------------------------------------------------------

#[test]
fn json_to_he_to_json_roundtrip() {
    let dir = TempDir::new().unwrap();
    let json_in = write_ndjson_fixture(&dir);
    let he_path = dir.path().join("data.he");
    let json_out = dir.path().join("out.json");

    // Convert JSON → he.
    helium()
        .arg("convert")
        .arg(&json_in)
        .arg("-o")
        .arg(&he_path)
        .assert()
        .success();

    // Convert he → JSON.
    helium()
        .arg("convert")
        .arg(&he_path)
        .arg("-o")
        .arg(&json_out)
        .assert()
        .success();
    assert!(json_out.exists(), "output JSON not created");

    let out_content = fs::read_to_string(&json_out).unwrap();
    // Should be NDJSON with 3 rows.
    let rows: Vec<serde_json::Value> = out_content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(rows.len(), 3, "expected 3 NDJSON rows: {out_content}");
    assert_eq!(rows[0]["id"], serde_json::json!(1));
    assert_eq!(rows[1]["id"], serde_json::json!(2));
}

// ---------------------------------------------------------------------------
// 3. Parquet → he → Parquet round-trip
// ---------------------------------------------------------------------------

#[test]
fn parquet_to_he_to_parquet_roundtrip() {
    let dir = TempDir::new().unwrap();
    let pq_in = write_parquet_fixture(&dir);
    let he_path = dir.path().join("data.he");
    let pq_out = dir.path().join("out.parquet");

    // Convert Parquet → he.
    helium()
        .arg("convert")
        .arg(&pq_in)
        .arg("-o")
        .arg(&he_path)
        .assert()
        .success();

    // Convert he → Parquet.
    helium()
        .arg("convert")
        .arg(&he_path)
        .arg("-o")
        .arg(&pq_out)
        .assert()
        .success();
    assert!(pq_out.exists(), "output Parquet not created");

    // Verify output Parquet is readable.
    use parquet::file::reader::{FileReader, SerializedFileReader};
    let pq_file = fs::File::open(&pq_out).unwrap();
    let reader = SerializedFileReader::new(pq_file).unwrap();
    assert_eq!(
        reader.metadata().file_metadata().num_rows(),
        3,
        "expected 3 rows in output Parquet"
    );
}

// ---------------------------------------------------------------------------
// 4. he → csv with explicit --from he --to csv flags
// ---------------------------------------------------------------------------

#[test]
fn he_to_csv_explicit_flags() {
    let dir = TempDir::new().unwrap();
    let csv_in = write_csv_fixture(&dir);
    let he_path = dir.path().join("data.he");
    let csv_out = dir.path().join("out.csv");

    // First convert to .he.
    helium()
        .arg("convert")
        .arg(&csv_in)
        .arg("-o")
        .arg(&he_path)
        .assert()
        .success();

    // Export with explicit --from he --to csv.
    helium()
        .arg("convert")
        .arg(&he_path)
        .arg("-o")
        .arg(&csv_out)
        .arg("--from")
        .arg("he")
        .arg("--to")
        .arg("csv")
        .assert()
        .success();

    let out = fs::read_to_string(&csv_out).unwrap();
    assert!(out.contains("label"), "output CSV missing 'label': {out}");
}

// ---------------------------------------------------------------------------
// 5. Error: both extensions .he
// ---------------------------------------------------------------------------

#[test]
fn error_both_he_extensions() {
    let dir = TempDir::new().unwrap();
    // Create a dummy .he file.
    let he_in = dir.path().join("in.he");
    fs::write(&he_in, b"dummy").unwrap();
    let he_out = dir.path().join("out.he");

    helium()
        .arg("convert")
        .arg(&he_in)
        .arg("-o")
        .arg(&he_out)
        .assert()
        .failure()
        .stderr(predicate::str::contains("both").and(predicate::str::contains("nothing to do")));
}

// ---------------------------------------------------------------------------
// 6. Error: neither extension .he
// ---------------------------------------------------------------------------

#[test]
fn error_neither_he_extension() {
    let dir = TempDir::new().unwrap();
    let csv_in = write_csv_fixture(&dir);
    let csv_out = dir.path().join("out.csv");

    helium()
        .arg("convert")
        .arg(&csv_in)
        .arg("-o")
        .arg(&csv_out)
        .assert()
        .failure()
        .stderr(predicate::str::contains("neither"));
}

// ---------------------------------------------------------------------------
// 7. Error: --to avsc (Avro export not supported)
// ---------------------------------------------------------------------------

#[test]
fn error_to_avsc_not_supported() {
    let dir = TempDir::new().unwrap();
    let csv_in = write_csv_fixture(&dir);
    let he_path = dir.path().join("data.he");

    // First create a .he file.
    helium()
        .arg("convert")
        .arg(&csv_in)
        .arg("-o")
        .arg(&he_path)
        .assert()
        .success();

    let avsc_out = dir.path().join("out.avsc");
    helium()
        .arg("convert")
        .arg(&he_path)
        .arg("-o")
        .arg(&avsc_out)
        .arg("--to")
        .arg("avsc")
        .assert()
        .failure()
        .stderr(predicate::str::contains("avsc").and(predicate::str::contains("avro")));
}

// ---------------------------------------------------------------------------
// 8. --from / --to override extension
// ---------------------------------------------------------------------------

#[test]
fn flag_overrides_extension() {
    let dir = TempDir::new().unwrap();
    // Write CSV content to a .txt file.
    let txt_path = dir.path().join("data.txt");
    let mut f = fs::File::create(&txt_path).unwrap();
    writeln!(f, "id,value").unwrap();
    writeln!(f, "42,99.9").unwrap();
    let he_path = dir.path().join("data.he");

    // Use --from csv to tell helium that data.txt is CSV.
    helium()
        .arg("convert")
        .arg(&txt_path)
        .arg("-o")
        .arg(&he_path)
        .arg("--from")
        .arg("csv")
        .arg("--to")
        .arg("he")
        .assert()
        .success();
    assert!(he_path.exists(), ".he file not created from .txt input");

    // Verify with verify subcommand.
    helium()
        .arg("verify")
        .arg(&he_path)
        .assert()
        .success()
        .stdout(predicate::str::contains("OK"));
}

// ---------------------------------------------------------------------------
// 9. Parquet with OPTIONAL columns — regression for v3 Nullable shape mismatch
// ---------------------------------------------------------------------------
//
// The production loader (`src/cli/loader.rs::strings_to_logical_column`) must
// produce v3-shaped `LogicalColumn::Nullable { present, value: Box::new(...) }`
// when the inferred schema is v3 `Nullable { inner }`. Previously it emitted
// v2-shaped `NullablePrim` / `NullableUtf8` / `NullableBinary`, which the
// writer rejects with "logical column shape does not match schema". Real-world
// Parquet inputs (e.g. ClickBench's hits_1.parquet) use OPTIONAL columns
// extensively, so this path needs explicit coverage.

#[test]
fn parquet_optional_columns_to_he() {
    use parquet::basic::{ConvertedType, Repetition, Type as PqPhysical};
    use parquet::data_type::{ByteArray, ByteArrayType, Int64Type};
    use parquet::file::properties::WriterProperties;
    use parquet::file::writer::SerializedFileWriter;
    use parquet::schema::types::Type as PqType;
    use std::sync::Arc;

    let dir = TempDir::new().unwrap();
    let parquet_path = dir.path().join("nullable.parquet");

    // Schema: id INT64 OPTIONAL, label UTF8 OPTIONAL.
    let id_field = PqType::primitive_type_builder("id", PqPhysical::INT64)
        .with_repetition(Repetition::OPTIONAL)
        .with_converted_type(ConvertedType::INT_64)
        .build()
        .unwrap();
    let label_field = PqType::primitive_type_builder("label", PqPhysical::BYTE_ARRAY)
        .with_repetition(Repetition::OPTIONAL)
        .with_converted_type(ConvertedType::UTF8)
        .build()
        .unwrap();
    let schema = Arc::new(
        PqType::group_type_builder("schema")
            .with_fields(vec![Arc::new(id_field), Arc::new(label_field)])
            .build()
            .unwrap(),
    );

    let file = fs::File::create(&parquet_path).unwrap();
    let props = Arc::new(WriterProperties::builder().build());
    let mut writer = SerializedFileWriter::new(file, schema, props).unwrap();
    let mut rg = writer.next_row_group().unwrap();

    // id: rows [10, NULL, 30]  →  values [10, 30], def_levels [1, 0, 1]
    let mut col = rg.next_column().unwrap().unwrap();
    col.typed::<Int64Type>()
        .write_batch(&[10, 30], Some(&[1, 0, 1]), None)
        .unwrap();
    col.close().unwrap();

    // label: rows ["alpha", "beta", NULL]
    let labels: Vec<ByteArray> = vec![
        ByteArray::from("alpha".as_bytes()),
        ByteArray::from("beta".as_bytes()),
    ];
    let mut col = rg.next_column().unwrap().unwrap();
    col.typed::<ByteArrayType>()
        .write_batch(&labels, Some(&[1, 1, 0]), None)
        .unwrap();
    col.close().unwrap();

    rg.close().unwrap();
    writer.close().unwrap();

    // Convert parquet → he.
    let he_path = dir.path().join("nullable.he");
    helium()
        .arg("convert")
        .arg(&parquet_path)
        .arg("-o")
        .arg(&he_path)
        .assert()
        .success();
    assert!(
        he_path.exists(),
        ".he file not created from nullable parquet"
    );

    // Sanity: verify reads it back end-to-end (CRC + decode).
    helium()
        .arg("verify")
        .arg(&he_path)
        .assert()
        .success()
        .stdout(predicate::str::contains("OK"));
}

// ---------------------------------------------------------------------------
// 10. Semicolon-delimited CSV → he (3 columns, not 1)
// ---------------------------------------------------------------------------

/// Write a small semicolon-separated CSV fixture (3 columns: id, price, name).
fn write_semicolon_csv(dir: &TempDir) -> std::path::PathBuf {
    let path = dir.path().join("euro.csv");
    let mut f = fs::File::create(&path).unwrap();
    writeln!(f, "id;price;name").unwrap();
    writeln!(f, "1;9.99;Widget").unwrap();
    writeln!(f, "2;19.50;Gadget").unwrap();
    writeln!(f, "3;4.00;Donut").unwrap();
    path
}

#[test]
fn semicolon_csv_to_he_three_columns() {
    let dir = TempDir::new().unwrap();
    let csv_in = write_semicolon_csv(&dir);
    let he_path = dir.path().join("euro.he");

    // Without --delimiter, whole-row lands in a single column.
    helium()
        .arg("convert")
        .arg(&csv_in)
        .arg("-o")
        .arg(&he_path)
        .assert()
        .success();
    // The .he file exists; use stats --json to count columns.
    let no_delim_stats = helium()
        .arg("stats")
        .arg(&he_path)
        .arg("--json")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let no_delim_json: serde_json::Value =
        serde_json::from_slice(&no_delim_stats).expect("stats --json produced invalid JSON");
    let no_delim_cols = no_delim_json["columns"]
        .as_array()
        .expect("expected 'columns' array")
        .len();
    // Without the delimiter flag, every row is a single field: 1 column.
    assert_eq!(
        no_delim_cols, 1,
        "expected 1 column without --delimiter, got {no_delim_cols}"
    );

    // With --delimiter ';', 3 columns should be produced.
    let he_path2 = dir.path().join("euro2.he");
    helium()
        .arg("convert")
        .arg(&csv_in)
        .arg("-o")
        .arg(&he_path2)
        .arg("--delimiter")
        .arg(";")
        .assert()
        .success();

    helium()
        .arg("verify")
        .arg(&he_path2)
        .assert()
        .success()
        .stdout(predicate::str::contains("OK"));

    let stats_out = helium()
        .arg("stats")
        .arg(&he_path2)
        .arg("--json")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let stats_json: serde_json::Value =
        serde_json::from_slice(&stats_out).expect("stats --json produced invalid JSON");
    let col_count = stats_json["columns"]
        .as_array()
        .expect("expected 'columns' array")
        .len();
    assert_eq!(
        col_count, 3,
        "expected 3 columns with --delimiter ';', got {col_count}"
    );
}

// ---------------------------------------------------------------------------
// 11. Round-trip: semicolon CSV → he → semicolon CSV
// ---------------------------------------------------------------------------

#[test]
fn semicolon_csv_roundtrip() {
    let dir = TempDir::new().unwrap();
    let csv_in = write_semicolon_csv(&dir);
    let he_path = dir.path().join("euro.he");
    let csv_out = dir.path().join("euro_out.csv");

    // Import with semicolon delimiter.
    helium()
        .arg("convert")
        .arg(&csv_in)
        .arg("-o")
        .arg(&he_path)
        .arg("--delimiter")
        .arg(";")
        .assert()
        .success();

    // Export back to CSV with semicolon delimiter.
    helium()
        .arg("convert")
        .arg(&he_path)
        .arg("-o")
        .arg(&csv_out)
        .arg("--delimiter")
        .arg(";")
        .assert()
        .success();

    let content = fs::read_to_string(&csv_out).unwrap();
    // Header and values must survive the round-trip.
    assert!(
        content.contains("id"),
        "round-trip CSV missing 'id' column: {content}"
    );
    assert!(
        content.contains("price"),
        "round-trip CSV missing 'price' column: {content}"
    );
    assert!(
        content.contains("name"),
        "round-trip CSV missing 'name' column: {content}"
    );
    assert!(
        content.contains("Widget"),
        "round-trip CSV missing 'Widget': {content}"
    );
    assert!(
        content.contains("Gadget"),
        "round-trip CSV missing 'Gadget': {content}"
    );
    // Semicolons should appear in the output (delimiter in use).
    assert!(
        content.contains(';'),
        "output CSV should use ';' as delimiter: {content}"
    );
}

// ---------------------------------------------------------------------------
// 12. Default delimiter unchanged (comma CSV still works without flag)
// ---------------------------------------------------------------------------

#[test]
fn default_comma_delimiter_unchanged() {
    let dir = TempDir::new().unwrap();
    let csv_in = write_csv_fixture(&dir);
    let he_path = dir.path().join("data.he");

    // No --delimiter flag → default comma should work as before.
    helium()
        .arg("convert")
        .arg(&csv_in)
        .arg("-o")
        .arg(&he_path)
        .assert()
        .success();

    helium()
        .arg("verify")
        .arg(&he_path)
        .assert()
        .success()
        .stdout(predicate::str::contains("OK"));

    let stats_out = helium()
        .arg("stats")
        .arg(&he_path)
        .arg("--json")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let stats_json: serde_json::Value =
        serde_json::from_slice(&stats_out).expect("stats --json produced invalid JSON");
    let col_count = stats_json["columns"]
        .as_array()
        .expect("expected 'columns' array")
        .len();
    // The fixture has 3 columns: id, value, label.
    assert_eq!(
        col_count, 3,
        "expected 3 columns with default comma delimiter, got {col_count}"
    );
}

// ---------------------------------------------------------------------------
// 13. Bad delimiter: multi-char or empty → non-zero exit with clear error
// ---------------------------------------------------------------------------

#[test]
fn bad_delimiter_multichar_fails() {
    let dir = TempDir::new().unwrap();
    let csv_in = write_csv_fixture(&dir);
    let he_path = dir.path().join("data.he");

    helium()
        .arg("convert")
        .arg(&csv_in)
        .arg("-o")
        .arg(&he_path)
        .arg("--delimiter")
        .arg(";;")
        .assert()
        .failure()
        .stderr(predicate::str::contains("single ASCII character"));
}

#[test]
fn bad_delimiter_empty_fails() {
    let dir = TempDir::new().unwrap();
    let csv_in = write_csv_fixture(&dir);
    let he_path = dir.path().join("data.he");

    helium()
        .arg("convert")
        .arg(&csv_in)
        .arg("-o")
        .arg(&he_path)
        .arg("--delimiter")
        .arg("")
        .assert()
        .failure()
        // clap will reject the empty string at the arg level (default_value applies only
        // when the flag is absent; when present with "" clap still parses it)
        .stderr(
            predicate::str::contains("single ASCII character")
                .or(predicate::str::contains("error")),
        );
}

// ---------------------------------------------------------------------------
// 14. Regression: late-null beyond type-sample window must not fail convert
// ---------------------------------------------------------------------------
//
// Before the fix, `schema_from_csv` only checked nullability in the first
// `max_rows` (default 1000) rows.  A column with all non-empty values in the
// first 1000 rows but an empty value at row 1050 would be inferred as
// non-nullable, causing the data loader to fail with a parse error.
//
// This test generates a CSV with 1050 rows where column `x` is numeric for
// the first 1000 rows and empty at row 1050, then verifies that `helium convert`
// succeeds (it would have panicked / errored before the fix).

fn write_late_null_csv(dir: &TempDir) -> std::path::PathBuf {
    use std::io::BufWriter;
    let path = dir.path().join("late_null.csv");
    let f = fs::File::create(&path).unwrap();
    let mut w = BufWriter::new(f);
    writeln!(w, "x,y").unwrap();
    // 1049 rows of valid integers.
    for i in 0..1049_u32 {
        writeln!(w, "{},{}", i, i * 2).unwrap();
    }
    // Row 1050: x is empty (null), y is a normal integer.
    writeln!(w, ",2098").unwrap();
    path
}

#[test]
fn late_null_beyond_sample_window_convert_succeeds() {
    let dir = TempDir::new().unwrap();
    let csv_in = write_late_null_csv(&dir);
    let he_path = dir.path().join("late_null.he");

    // This must succeed: the inferred schema must have Nullable<I64> for column
    // `x` so the loader can handle the empty value at row 1050.
    helium()
        .arg("convert")
        .arg(&csv_in)
        .arg("-o")
        .arg(&he_path)
        .assert()
        .success();

    assert!(he_path.exists(), ".he file not created");

    // Verify the .he file is readable end-to-end.
    helium()
        .arg("verify")
        .arg(&he_path)
        .assert()
        .success()
        .stdout(predicate::str::contains("OK"));
}
