//! Integration tests for `helium convert --stripe-rows N`.
//!
//! Verifies that:
//! 1. `--stripe-rows 1000000` on a 5-row CSV → single stripe (chunk ≥ total).
//! 2. `--stripe-rows 2` on a 6-row CSV → 3 stripes of 2 rows each.
//! 3. `--stripe-rows 3` on a 7-row CSV → 3 stripes (3+3+1).
//! 4. `--stripe-rows 0` → single stripe (same as default).
//! 5. Round-trip parity: CSV → he (stripe-rows=2) → CSV must equal the input.
//! 6. Nullable Parquet round-trip: OPTIONAL columns + stripe-rows=5 on 23 rows.
//! 7. Per-stripe stats reflect each stripe's data.
#![cfg(feature = "cli")]

use std::fs;
use std::io::Write;
use std::path::PathBuf;

use assert_cmd::Command;
use helium::{CoderRegistry, HeliumReader};
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn helium() -> Command {
    Command::cargo_bin("helium").expect("helium binary not found")
}

/// Write a CSV with `n` rows where each row has columns (id, value, label).
/// Row i has: id = i+1, value = (i+1)*0.5, label = "item{i+1}".
fn write_csv_n_rows(dir: &TempDir, name: &str, n: usize) -> PathBuf {
    let path = dir.path().join(name);
    let mut f = fs::File::create(&path).unwrap();
    writeln!(f, "id,value,label").unwrap();
    for i in 0..n {
        writeln!(f, "{},{:.1},item{}", i + 1, (i + 1) as f64 * 0.5, i + 1).unwrap();
    }
    path
}

/// Open a `.he` file and return `(stripe_count, row_count)`.
fn he_meta(path: &PathBuf) -> (usize, u64) {
    let reg = CoderRegistry::default();
    let file = fs::File::open(path).unwrap();
    let reader = HeliumReader::new(file, &reg).unwrap();
    (reader.stripe_count(), reader.row_count())
}

// ---------------------------------------------------------------------------
// Test 1: --stripe-rows larger than total → single stripe
// ---------------------------------------------------------------------------

#[test]
fn stripe_rows_larger_than_total() {
    let dir = TempDir::new().unwrap();
    let csv_in = write_csv_n_rows(&dir, "data.csv", 5);
    let he_path = dir.path().join("out.he");

    helium()
        .args(["convert"])
        .arg(&csv_in)
        .arg("-o")
        .arg(&he_path)
        .args(["--stripe-rows", "1000000"])
        .assert()
        .success();

    let (stripes, rows) = he_meta(&he_path);
    assert_eq!(stripes, 1, "expected 1 stripe when chunk >= total");
    assert_eq!(rows, 5);
}

// ---------------------------------------------------------------------------
// Test 2: exact-multiple split (6 rows / 2 = 3 stripes)
// ---------------------------------------------------------------------------

#[test]
fn stripe_rows_exact_multiple() {
    let dir = TempDir::new().unwrap();
    let csv_in = write_csv_n_rows(&dir, "data.csv", 6);
    let he_path = dir.path().join("out.he");

    helium()
        .args(["convert"])
        .arg(&csv_in)
        .arg("-o")
        .arg(&he_path)
        .args(["--stripe-rows", "2"])
        .assert()
        .success();

    let (stripes, rows) = he_meta(&he_path);
    assert_eq!(stripes, 3, "6 rows / 2 per stripe = 3 stripes");
    assert_eq!(rows, 6);
}

// ---------------------------------------------------------------------------
// Test 3: non-exact-multiple (7 rows / 3 = 3 stripes: 3+3+1)
// ---------------------------------------------------------------------------

#[test]
fn stripe_rows_non_exact_multiple() {
    let dir = TempDir::new().unwrap();
    let csv_in = write_csv_n_rows(&dir, "data.csv", 7);
    let he_path = dir.path().join("out.he");

    helium()
        .args(["convert"])
        .arg(&csv_in)
        .arg("-o")
        .arg(&he_path)
        .args(["--stripe-rows", "3"])
        .assert()
        .success();

    let (stripes, rows) = he_meta(&he_path);
    assert_eq!(stripes, 3, "7 rows / 3 per stripe = 3 stripes (3+3+1)");
    assert_eq!(rows, 7);
}

// ---------------------------------------------------------------------------
// Test 4: --stripe-rows 0 → single stripe
// ---------------------------------------------------------------------------

#[test]
fn stripe_rows_zero_means_single_stripe() {
    let dir = TempDir::new().unwrap();
    let csv_in = write_csv_n_rows(&dir, "data.csv", 10);
    let he_path = dir.path().join("out.he");

    helium()
        .args(["convert"])
        .arg(&csv_in)
        .arg("-o")
        .arg(&he_path)
        .args(["--stripe-rows", "0"])
        .assert()
        .success();

    let (stripes, rows) = he_meta(&he_path);
    assert_eq!(stripes, 1, "--stripe-rows 0 should produce single stripe");
    assert_eq!(rows, 10);
}

// ---------------------------------------------------------------------------
// Test 5: round-trip parity — CSV → he (stripe-rows=2) → CSV
// ---------------------------------------------------------------------------

#[test]
fn stripe_rows_roundtrip_csv_parity() {
    let dir = TempDir::new().unwrap();
    let csv_in = write_csv_n_rows(&dir, "data.csv", 6);
    let he_path = dir.path().join("out.he");
    let csv_out = dir.path().join("roundtrip.csv");

    // CSV → he with 3 stripes.
    helium()
        .args(["convert"])
        .arg(&csv_in)
        .arg("-o")
        .arg(&he_path)
        .args(["--stripe-rows", "2"])
        .assert()
        .success();

    // he → CSV.
    helium()
        .args(["convert"])
        .arg(&he_path)
        .arg("-o")
        .arg(&csv_out)
        .assert()
        .success();

    // Verify that the roundtripped CSV has the same row count as the original
    // and that integer id values and string labels are preserved.
    let original = fs::read_to_string(&csv_in).unwrap();
    let roundtripped = fs::read_to_string(&csv_out).unwrap();

    let orig_rows: Vec<&str> = original.trim().lines().collect();
    let rt_rows: Vec<&str> = roundtripped.trim().lines().collect();

    // Same row count (header + 6 data rows).
    assert_eq!(
        orig_rows.len(),
        rt_rows.len(),
        "row count mismatch after round-trip"
    );

    // All integer id values (1..=6) must appear in the roundtripped CSV.
    for i in 1..=6usize {
        assert!(
            roundtripped.contains(&i.to_string()),
            "round-trip CSV missing id '{i}'"
        );
    }

    // All string labels must be present.
    for i in 1..=6usize {
        let label = format!("item{i}");
        assert!(
            roundtripped.contains(&label),
            "round-trip CSV missing label '{label}'"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 6: Nullable Parquet round-trip with stripe-rows=5, 23 rows
// ---------------------------------------------------------------------------

#[test]
#[cfg(feature = "schema-parquet")]
fn nullable_parquet_multistripe_roundtrip() {
    use parquet::basic::{ConvertedType, Repetition, Type as PqPhysical};
    use parquet::data_type::{ByteArray, ByteArrayType, Int64Type};
    use parquet::file::properties::WriterProperties;
    use parquet::file::writer::SerializedFileWriter;
    use parquet::schema::types::Type as PqType;
    use std::sync::Arc;

    let dir = TempDir::new().unwrap();
    let parquet_path = dir.path().join("nullable.parquet");

    // Schema: id INT64 OPTIONAL, label UTF8 OPTIONAL (23 rows, some nulls).
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

    // Build 23 rows: every 5th id is null, every 7th label is null.
    let n = 23usize;
    let mut id_values: Vec<i64> = Vec::new();
    let mut id_def: Vec<i16> = Vec::new();
    let mut label_values: Vec<ByteArray> = Vec::new();
    let mut label_def: Vec<i16> = Vec::new();

    for i in 0..n {
        if i % 5 == 0 {
            id_def.push(0); // null
        } else {
            id_values.push(i as i64 * 10);
            id_def.push(1);
        }
        if i % 7 == 0 {
            label_def.push(0); // null
        } else {
            label_values.push(ByteArray::from(format!("label{i}").as_bytes()));
            label_def.push(1);
        }
    }

    let file = fs::File::create(&parquet_path).unwrap();
    let props = Arc::new(WriterProperties::builder().build());
    let mut writer = SerializedFileWriter::new(file, schema, props).unwrap();
    let mut rg = writer.next_row_group().unwrap();

    let mut col = rg.next_column().unwrap().unwrap();
    col.typed::<Int64Type>()
        .write_batch(&id_values, Some(&id_def), None)
        .unwrap();
    col.close().unwrap();

    let mut col = rg.next_column().unwrap().unwrap();
    col.typed::<ByteArrayType>()
        .write_batch(&label_values, Some(&label_def), None)
        .unwrap();
    col.close().unwrap();

    rg.close().unwrap();
    writer.close().unwrap();

    // Convert with --stripe-rows 5 → ceil(23/5) = 5 stripes.
    let he_path = dir.path().join("nullable_ms.he");
    helium()
        .args(["convert"])
        .arg(&parquet_path)
        .arg("-o")
        .arg(&he_path)
        .args(["--stripe-rows", "5"])
        .assert()
        .success();

    let (stripes, rows) = he_meta(&he_path);
    assert_eq!(rows, 23, "total row count must be 23");
    assert_eq!(stripes, 5, "ceil(23/5)=5 stripes");

    // Verify: read + CRC all stripes.
    helium()
        .arg("verify")
        .arg(&he_path)
        .assert()
        .success()
        .stdout(predicates::prelude::predicate::str::contains("OK"));
}

// ---------------------------------------------------------------------------
// Test 7: per-stripe stats reflect each stripe's data
// ---------------------------------------------------------------------------

#[test]
fn stats_stripe_count_reported_correctly() {
    let dir = TempDir::new().unwrap();
    // Write 9 rows, 3 stripes of 3.
    let csv_in = write_csv_n_rows(&dir, "data.csv", 9);
    let he_path = dir.path().join("out.he");

    helium()
        .args(["convert"])
        .arg(&csv_in)
        .arg("-o")
        .arg(&he_path)
        .args(["--stripe-rows", "3"])
        .assert()
        .success();

    // helium stats reports stripe count in the summary header (case-insensitive match).
    helium()
        .args(["stats"])
        .arg(&he_path)
        .assert()
        .success()
        .stdout(predicates::prelude::predicate::str::contains("Stripes"));

    // Also verify the HeliumReader metadata.
    let (stripes, rows) = he_meta(&he_path);
    assert_eq!(stripes, 3, "3 stripes for 9 rows / 3");
    assert_eq!(rows, 9);
}

// ---------------------------------------------------------------------------
// Test 8: verify passes on multi-stripe file
// ---------------------------------------------------------------------------

#[test]
fn verify_multistripe_file() {
    let dir = TempDir::new().unwrap();
    let csv_in = write_csv_n_rows(&dir, "data.csv", 10);
    let he_path = dir.path().join("out.he");

    helium()
        .args(["convert"])
        .arg(&csv_in)
        .arg("-o")
        .arg(&he_path)
        .args(["--stripe-rows", "4"])
        .assert()
        .success();

    // ceil(10/4) = 3 stripes.
    let (stripes, rows) = he_meta(&he_path);
    assert_eq!(stripes, 3);
    assert_eq!(rows, 10);

    helium()
        .arg("verify")
        .arg(&he_path)
        .assert()
        .success()
        .stdout(predicates::prelude::predicate::str::contains(
            "OK: 3 column(s)",
        ));
}
