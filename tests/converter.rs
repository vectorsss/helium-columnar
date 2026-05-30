//! Converter tests — synthetic Parquet round-tripped through the production
//! `helium` CLI binary (`helium convert`, `helium verify`, `helium stats`).
//!
//! Real ClickBench verification runs when `HELIUM_PARQUET_PATH` is set.
#![cfg(feature = "cli")]

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use assert_cmd::Command;
use parquet::basic::{LogicalType as PqLogical, Repetition, Type as PqPhysical};
use parquet::data_type::{ByteArray, ByteArrayType, DoubleType, Int32Type, Int64Type};
use parquet::file::properties::WriterProperties;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::file::writer::SerializedFileWriter;
use parquet::record::Field;
use parquet::schema::types::Type as SchemaType;

// ============================================================================
// Build a small Parquet file with every supported column shape
// ============================================================================

fn build_synthetic_parquet(path: &Path, n: usize) {
    let fields = vec![
        // Required primitives
        Arc::new(
            SchemaType::primitive_type_builder("id_i32", PqPhysical::INT32)
                .with_repetition(Repetition::REQUIRED)
                .build()
                .unwrap(),
        ),
        Arc::new(
            SchemaType::primitive_type_builder("ts_i64", PqPhysical::INT64)
                .with_repetition(Repetition::REQUIRED)
                .build()
                .unwrap(),
        ),
        Arc::new(
            SchemaType::primitive_type_builder("score_f64", PqPhysical::DOUBLE)
                .with_repetition(Repetition::REQUIRED)
                .build()
                .unwrap(),
        ),
        // Required strings (UTF8 logical)
        Arc::new(
            SchemaType::primitive_type_builder("name_utf8", PqPhysical::BYTE_ARRAY)
                .with_repetition(Repetition::REQUIRED)
                .with_logical_type(Some(PqLogical::String))
                .build()
                .unwrap(),
        ),
        // Required binary (no logical)
        Arc::new(
            SchemaType::primitive_type_builder("blob_bin", PqPhysical::BYTE_ARRAY)
                .with_repetition(Repetition::REQUIRED)
                .build()
                .unwrap(),
        ),
        // Optional primitives
        Arc::new(
            SchemaType::primitive_type_builder("maybe_i32", PqPhysical::INT32)
                .with_repetition(Repetition::OPTIONAL)
                .build()
                .unwrap(),
        ),
        // Optional string
        Arc::new(
            SchemaType::primitive_type_builder("maybe_utf8", PqPhysical::BYTE_ARRAY)
                .with_repetition(Repetition::OPTIONAL)
                .with_logical_type(Some(PqLogical::String))
                .build()
                .unwrap(),
        ),
        // Optional binary
        Arc::new(
            SchemaType::primitive_type_builder("maybe_bin", PqPhysical::BYTE_ARRAY)
                .with_repetition(Repetition::OPTIONAL)
                .build()
                .unwrap(),
        ),
    ];
    let root = Arc::new(
        SchemaType::group_type_builder("root")
            .with_fields(fields)
            .build()
            .unwrap(),
    );

    let file = File::create(path).unwrap();
    let props = Arc::new(WriterProperties::builder().build());
    let mut w = SerializedFileWriter::new(file, root, props).unwrap();
    let mut rg = w.next_row_group().unwrap();

    // id_i32 (required)
    let id: Vec<i32> = (0..n as i32).collect();
    let mut cw = rg.next_column().unwrap().unwrap();
    cw.typed::<Int32Type>()
        .write_batch(&id, None, None)
        .unwrap();
    cw.close().unwrap();

    // ts_i64 (required, monotone)
    let ts: Vec<i64> = (0..n).map(|i| 1_700_000_000 + i as i64 * 30).collect();
    let mut cw = rg.next_column().unwrap().unwrap();
    cw.typed::<Int64Type>()
        .write_batch(&ts, None, None)
        .unwrap();
    cw.close().unwrap();

    // score_f64 (required)
    let score: Vec<f64> = (0..n).map(|i| 10.0 + (i as f64).sin()).collect();
    let mut cw = rg.next_column().unwrap().unwrap();
    cw.typed::<DoubleType>()
        .write_batch(&score, None, None)
        .unwrap();
    cw.close().unwrap();

    // name_utf8 (required)
    let names: Vec<ByteArray> = (0..n)
        .map(|i| ByteArray::from(format!("user_{i:04}").into_bytes()))
        .collect();
    let mut cw = rg.next_column().unwrap().unwrap();
    cw.typed::<ByteArrayType>()
        .write_batch(&names, None, None)
        .unwrap();
    cw.close().unwrap();

    // blob_bin (required, includes non-UTF8 bytes)
    let blobs: Vec<ByteArray> = (0..n)
        .map(|i| ByteArray::from(vec![(i % 256) as u8, 0xff, 0xfe, 0xfd, (i >> 8) as u8]))
        .collect();
    let mut cw = rg.next_column().unwrap().unwrap();
    cw.typed::<ByteArrayType>()
        .write_batch(&blobs, None, None)
        .unwrap();
    cw.close().unwrap();

    // maybe_i32 (optional: every 3rd row is null)
    let mi_def: Vec<i16> = (0..n as i16)
        .map(|i| if i % 3 == 0 { 0 } else { 1 })
        .collect();
    let mi_vals: Vec<i32> = (0..n as i32).filter(|i| i % 3 != 0).collect();
    let mut cw = rg.next_column().unwrap().unwrap();
    cw.typed::<Int32Type>()
        .write_batch(&mi_vals, Some(&mi_def), None)
        .unwrap();
    cw.close().unwrap();

    // maybe_utf8 (optional: every 5th null)
    let mu_def: Vec<i16> = (0..n as i16)
        .map(|i| if i % 5 == 0 { 0 } else { 1 })
        .collect();
    let mu_vals: Vec<ByteArray> = (0..n)
        .filter(|i| i % 5 != 0)
        .map(|i| ByteArray::from(format!("v_{i:03}").into_bytes()))
        .collect();
    let mut cw = rg.next_column().unwrap().unwrap();
    cw.typed::<ByteArrayType>()
        .write_batch(&mu_vals, Some(&mu_def), None)
        .unwrap();
    cw.close().unwrap();

    // maybe_bin (optional: every 4th null, contains 0xff bytes)
    let mb_def: Vec<i16> = (0..n as i16)
        .map(|i| if i % 4 == 0 { 0 } else { 1 })
        .collect();
    let mb_vals: Vec<ByteArray> = (0..n)
        .filter(|i| i % 4 != 0)
        .map(|i| ByteArray::from(vec![0xff, (i % 256) as u8, 0x00, 0xfe]))
        .collect();
    let mut cw = rg.next_column().unwrap().unwrap();
    cw.typed::<ByteArrayType>()
        .write_batch(&mb_vals, Some(&mb_def), None)
        .unwrap();
    cw.close().unwrap();

    rg.close().unwrap();
    w.close().unwrap();
}

// ============================================================================
// Parquet round-trip comparison helper
//
// Reads both parquet files via the row iterator and compares every cell.
// Columns are matched by name (not position) to tolerate any reordering.
// Values are compared as a canonical string representation so that type
// widening (e.g. INT32 → INT64) does not produce false failures.
// ============================================================================

/// A single cell value, normalised for comparison.
#[derive(Debug, PartialEq)]
enum Cell {
    Null,
    Int(i64),
    Float(u64), // f64 bits, bit-exact comparison
    Bytes(Vec<u8>),
    Str(String),
}

fn field_to_cell(f: &Field) -> Cell {
    match f {
        Field::Null => Cell::Null,
        Field::Bool(b) => Cell::Int(*b as i64),
        Field::Byte(x) => Cell::Int(*x as i64),
        Field::Short(x) => Cell::Int(*x as i64),
        Field::Int(x) => Cell::Int(*x as i64),
        Field::Long(x) => Cell::Int(*x),
        Field::UByte(x) => Cell::Int(*x as i64),
        Field::UShort(x) => Cell::Int(*x as i64),
        Field::UInt(x) => Cell::Int(*x as i64),
        Field::ULong(x) => Cell::Int(*x as i64),
        Field::Float(x) => Cell::Float((*x as f64).to_bits()),
        Field::Double(x) => Cell::Float(x.to_bits()),
        Field::Str(s) => Cell::Str(s.clone()),
        // Raw bytes — the loader now preserves these byte-exact, so the
        // comparison is lossless for Binary columns containing non-UTF-8 data.
        Field::Bytes(b) => Cell::Bytes(b.data().to_vec()),
        Field::TimestampMillis(x) | Field::TimestampMicros(x) => Cell::Int(*x),
        Field::Date(x) => Cell::Int(*x as i64),
        // Decimal: compare as bytes for exact match.
        Field::Decimal(d) => Cell::Bytes(format!("{d:?}").into_bytes()),
        _ => Cell::Str(format!("{f:?}")),
    }
}

/// Load a parquet file as a map from column name to column-of-cells (ordered
/// by row). Supports both REQUIRED and OPTIONAL columns.
fn load_parquet_as_cells(path: &Path) -> HashMap<String, Vec<Cell>> {
    let file = File::open(path).expect("open parquet for comparison");
    let reader = SerializedFileReader::new(file).expect("parse parquet for comparison");
    let meta = reader.metadata();
    let schema = meta.file_metadata().schema();

    // Gather column names in order.
    let col_names: Vec<String> = schema
        .get_fields()
        .iter()
        .map(|f| f.name().to_owned())
        .collect();

    let mut columns: HashMap<String, Vec<Cell>> =
        col_names.iter().map(|n| (n.clone(), Vec::new())).collect();

    for row in reader.get_row_iter(None).expect("row iter for comparison") {
        let row = row.expect("read row for comparison");
        let row_map: HashMap<&str, &Field> = row
            .get_column_iter()
            .map(|(n, f)| (n.as_str(), f))
            .collect();
        for name in &col_names {
            let cell = row_map
                .get(name.as_str())
                .map(|f| field_to_cell(f))
                .unwrap_or(Cell::Null);
            columns.get_mut(name).unwrap().push(cell);
        }
    }
    columns
}

/// Assert that two parquet files contain identical data.
/// Matches columns by name; panics with a detailed message on first mismatch.
fn assert_parquet_equal(orig: &Path, roundtrip: &Path) {
    let orig_cols = load_parquet_as_cells(orig);
    let rt_cols = load_parquet_as_cells(roundtrip);

    assert_eq!(
        orig_cols.len(),
        rt_cols.len(),
        "column count mismatch: orig {} vs roundtrip {}",
        orig_cols.len(),
        rt_cols.len()
    );

    for (name, orig_vals) in &orig_cols {
        let rt_vals = rt_cols.get(name).unwrap_or_else(|| {
            panic!("column '{name}' present in orig but missing from round-tripped parquet")
        });
        assert_eq!(
            orig_vals.len(),
            rt_vals.len(),
            "column '{name}' row count mismatch: orig {} vs roundtrip {}",
            orig_vals.len(),
            rt_vals.len()
        );
        for (row_idx, (a, b)) in orig_vals.iter().zip(rt_vals).enumerate() {
            assert_eq!(
                a, b,
                "column '{name}' row {row_idx}: orig {a:?} vs roundtrip {b:?}"
            );
        }
    }
}

// ============================================================================
// Production CLI helper
// ============================================================================

fn helium() -> Command {
    Command::cargo_bin("helium").expect("helium binary not found")
}

// ============================================================================
// Tests
// ============================================================================

/// Build a 10k-row synthetic Parquet, convert → .he → Parquet via the
/// production CLI, then compare both Parquets cell-by-cell.
#[test]
fn synthetic_parquet_roundtrip() {
    let tmp = tempfile::tempdir().unwrap();
    let pq_in = tmp.path().join("in.parquet");
    let he_mid = tmp.path().join("mid.he");
    let pq_out = tmp.path().join("out.parquet");

    build_synthetic_parquet(&pq_in, 10_000);

    // Step 1: Parquet → .he
    helium()
        .arg("convert")
        .arg(&pq_in)
        .arg("-o")
        .arg(&he_mid)
        .assert()
        .success();
    assert!(he_mid.exists(), ".he file not created");

    // Step 2: .he → Parquet
    helium()
        .arg("convert")
        .arg(&he_mid)
        .arg("-o")
        .arg(&pq_out)
        .assert()
        .success();
    assert!(pq_out.exists(), "round-tripped parquet not created");

    // Step 3: verify CRC integrity of the intermediate .he file
    helium().arg("verify").arg(&he_mid).assert().success();

    // Step 4: compare both parquets value-by-value
    assert_parquet_equal(&pq_in, &pq_out);
}

/// Run convert (parquet→he) and convert (he→parquet) as two separate
/// subprocess invocations, plus `helium stats` and `helium verify` to confirm
/// those subcommands also work correctly.
#[test]
fn synthetic_parquet_separate_steps() {
    let tmp = tempfile::tempdir().unwrap();
    let pq_in = tmp.path().join("in.parquet");
    let he_mid = tmp.path().join("mid.he");
    let pq_out = tmp.path().join("out.parquet");

    build_synthetic_parquet(&pq_in, 2_000);

    // Convert parquet → .he
    helium()
        .arg("convert")
        .arg(&pq_in)
        .arg("-o")
        .arg(&he_mid)
        .assert()
        .success();

    // Convert .he → parquet
    helium()
        .arg("convert")
        .arg(&he_mid)
        .arg("-o")
        .arg(&pq_out)
        .assert()
        .success();

    // `helium stats --no-values` should succeed and emit per-column lines.
    helium()
        .arg("stats")
        .arg(&he_mid)
        .arg("--no-values")
        .assert()
        .success();

    // `helium verify` should report OK.
    helium().arg("verify").arg(&he_mid).assert().success();

    // Compare round-tripped parquet cell-by-cell.
    assert_parquet_equal(&pq_in, &pq_out);
}

// ============================================================================
// Focused binary round-trip tests (byte-exact, non-UTF-8 payload)
// ============================================================================

/// Build a tiny Parquet with a single required Binary column containing
/// non-UTF-8 bytes, convert to .he and back, and assert byte-exact equality.
fn build_binary_only_parquet(path: &Path, rows: &[Vec<u8>]) {
    let fields = vec![Arc::new(
        SchemaType::primitive_type_builder("bin_col", PqPhysical::BYTE_ARRAY)
            .with_repetition(Repetition::REQUIRED)
            .build()
            .unwrap(),
    )];
    let root = Arc::new(
        SchemaType::group_type_builder("root")
            .with_fields(fields)
            .build()
            .unwrap(),
    );
    let file = File::create(path).unwrap();
    let props = Arc::new(WriterProperties::builder().build());
    let mut w = SerializedFileWriter::new(file, root, props).unwrap();
    let mut rg = w.next_row_group().unwrap();

    let blobs: Vec<ByteArray> = rows.iter().map(|b| ByteArray::from(b.clone())).collect();
    let mut cw = rg.next_column().unwrap().unwrap();
    cw.typed::<ByteArrayType>()
        .write_batch(&blobs, None, None)
        .unwrap();
    cw.close().unwrap();
    rg.close().unwrap();
    w.close().unwrap();
}

/// Build a tiny Parquet with a single optional Binary column, with some null
/// rows, convert to .he and back, and assert byte-exact equality.
fn build_nullable_binary_parquet(path: &Path, rows: &[Option<Vec<u8>>]) {
    let fields = vec![Arc::new(
        SchemaType::primitive_type_builder("bin_col", PqPhysical::BYTE_ARRAY)
            .with_repetition(Repetition::OPTIONAL)
            .build()
            .unwrap(),
    )];
    let root = Arc::new(
        SchemaType::group_type_builder("root")
            .with_fields(fields)
            .build()
            .unwrap(),
    );
    let file = File::create(path).unwrap();
    let props = Arc::new(WriterProperties::builder().build());
    let mut w = SerializedFileWriter::new(file, root, props).unwrap();
    let mut rg = w.next_row_group().unwrap();

    let def_levels: Vec<i16> = rows
        .iter()
        .map(|r| if r.is_some() { 1 } else { 0 })
        .collect();
    let non_null: Vec<ByteArray> = rows
        .iter()
        .filter_map(|r| r.as_ref())
        .map(|b| ByteArray::from(b.clone()))
        .collect();
    let mut cw = rg.next_column().unwrap().unwrap();
    cw.typed::<ByteArrayType>()
        .write_batch(&non_null, Some(&def_levels), None)
        .unwrap();
    cw.close().unwrap();
    rg.close().unwrap();
    w.close().unwrap();
}

/// Non-UTF-8 required Binary column round-trips byte-exact through
/// `helium convert parquet → he → parquet`.
#[test]
fn binary_column_non_utf8_roundtrip_exact() {
    let tmp = tempfile::tempdir().unwrap();
    let pq_in = tmp.path().join("binary_in.parquet");
    let he_mid = tmp.path().join("binary_mid.he");
    let pq_out = tmp.path().join("binary_out.parquet");

    // These bytes cannot be decoded as UTF-8 — 0xff, 0xfe, 0xfd are all
    // invalid UTF-8 lead bytes.
    let rows: Vec<Vec<u8>> = vec![
        vec![0xff, 0xfe],
        vec![],
        vec![0x00, 0x01, 0xff],
        vec![0xfd, 0xfe, 0xff, 0x00],
    ];
    build_binary_only_parquet(&pq_in, &rows);

    helium()
        .arg("convert")
        .arg(&pq_in)
        .arg("-o")
        .arg(&he_mid)
        .assert()
        .success();

    helium()
        .arg("convert")
        .arg(&he_mid)
        .arg("-o")
        .arg(&pq_out)
        .assert()
        .success();

    // Compare byte-for-byte (no lossy normalization on either side).
    let orig = load_parquet_as_cells(&pq_in);
    let rt = load_parquet_as_cells(&pq_out);
    assert_eq!(orig, rt, "binary column did not round-trip byte-exact");
}

/// Nullable Binary column with null rows round-trips byte-exact.
#[test]
fn nullable_binary_column_non_utf8_roundtrip_exact() {
    let tmp = tempfile::tempdir().unwrap();
    let pq_in = tmp.path().join("nullable_binary_in.parquet");
    let he_mid = tmp.path().join("nullable_binary_mid.he");
    let pq_out = tmp.path().join("nullable_binary_out.parquet");

    let rows: Vec<Option<Vec<u8>>> = vec![
        Some(vec![0xff, 0x00]),
        None,
        Some(vec![0xfe, 0xfd, 0x80]),
        None,
        Some(vec![0x01, 0x02, 0xff]),
    ];
    build_nullable_binary_parquet(&pq_in, &rows);

    helium()
        .arg("convert")
        .arg(&pq_in)
        .arg("-o")
        .arg(&he_mid)
        .assert()
        .success();

    helium()
        .arg("convert")
        .arg(&he_mid)
        .arg("-o")
        .arg(&pq_out)
        .assert()
        .success();

    let orig = load_parquet_as_cells(&pq_in);
    let rt = load_parquet_as_cells(&pq_out);
    assert_eq!(
        orig, rt,
        "nullable binary column did not round-trip byte-exact"
    );
}

/// When `HELIUM_PARQUET_PATH` is set, convert → .he and verify CRC integrity.
/// Does NOT attempt he→parquet for ClickBench (1M × 105 cols is memory-heavy).
/// Uses `--stripe-rows 10000` so the writer streams instead of buffering all rows.
#[test]
fn real_clickbench_roundtrip_if_available() {
    let Ok(path) = std::env::var("HELIUM_PARQUET_PATH") else {
        eprintln!("HELIUM_PARQUET_PATH not set — skipping real-ClickBench converter test");
        return;
    };
    let tmp = tempfile::tempdir().unwrap();
    let he_out = tmp.path().join("clickbench.he");

    helium()
        .arg("convert")
        .arg(&path)
        .arg("-o")
        .arg(&he_out)
        .arg("--stripe-rows")
        .arg("10000")
        .assert()
        .success();
    assert!(he_out.exists(), "clickbench .he file not created");

    helium().arg("verify").arg(&he_out).assert().success();
}
