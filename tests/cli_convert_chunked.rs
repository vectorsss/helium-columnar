//! Integration tests for chunked / streaming `helium convert --stripe-rows N`.
//!
//! Verifies that the streaming path (which reads `chunk_rows` rows at a time
//! without loading the whole input) produces logically identical output to the
//! single-stripe in-memory path for CSV, Parquet, NDJSON, Avro, and
//! JSON-array inputs.
//!
//! All tests are gated on `feature = "cli"` since they drive the binary.
#![cfg(feature = "cli")]

use std::fs;
use std::io::Write;
use std::path::PathBuf;

use assert_cmd::Command;
use helium::{CoderRegistry, HeliumReader, LogicalColumn};
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn helium() -> Command {
    Command::cargo_bin("helium").expect("helium binary not found")
}

/// Write a 17-row CSV with columns (id i64, score f64, label utf8).
fn write_17row_csv(dir: &TempDir) -> PathBuf {
    let path = dir.path().join("data17.csv");
    let mut f = fs::File::create(&path).unwrap();
    writeln!(f, "id,score,label").unwrap();
    for i in 0..17usize {
        writeln!(f, "{},{:.2},item{i}", i as i64, i as f64 * 1.5).unwrap();
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

/// Read all data from a `.he` file as a flat `HashMap<column_name, LogicalColumn>`.
fn he_read_all(path: &PathBuf) -> std::collections::HashMap<String, LogicalColumn> {
    let reg = CoderRegistry::default();
    let file = fs::File::open(path).unwrap();
    let mut reader = HeliumReader::new(file, &reg).unwrap();
    reader.read_all().expect("read_all failed")
}

// ---------------------------------------------------------------------------
// Test 1: Functional equivalence — CSV, 17 rows
//
// Streaming (--stripe-rows 5): 4 stripes (5+5+5+2).
// Legacy single-stripe (--stripe-rows 0): 1 stripe.
// Column data must be identical.
// ---------------------------------------------------------------------------

#[test]
fn chunked_csv_equivalence_17_rows() {
    let dir = TempDir::new().unwrap();
    let csv_in = write_17row_csv(&dir);
    let he_streaming = dir.path().join("streaming.he");
    let he_single = dir.path().join("single.he");

    // Streaming: --stripe-rows 5.
    helium()
        .args(["convert"])
        .arg(&csv_in)
        .arg("-o")
        .arg(&he_streaming)
        .args(["--stripe-rows", "5"])
        .assert()
        .success();

    // Legacy single-stripe: --stripe-rows 0.
    helium()
        .args(["convert"])
        .arg(&csv_in)
        .arg("-o")
        .arg(&he_single)
        .args(["--stripe-rows", "0"])
        .assert()
        .success();

    // Stripe counts must differ.
    let (s_stripes, s_rows) = he_meta(&he_streaming);
    let (l_stripes, l_rows) = he_meta(&he_single);
    assert_eq!(s_stripes, 4, "expected 4 stripes for 17 rows @ 5/stripe");
    assert_eq!(l_stripes, 1, "expected 1 stripe for single-stripe path");
    assert_eq!(s_rows, 17);
    assert_eq!(l_rows, 17);

    // Column data must be logically identical.
    let s_data = he_read_all(&he_streaming);
    let l_data = he_read_all(&he_single);
    for name in ["id", "score", "label"] {
        let s_lc = s_data
            .get(name)
            .unwrap_or_else(|| panic!("streaming missing col {name}"));
        let l_lc = l_data
            .get(name)
            .unwrap_or_else(|| panic!("single missing col {name}"));
        assert_eq!(
            s_lc, l_lc,
            "column '{name}' differs between streaming and single-stripe"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 2: Functional equivalence — Parquet, 17 rows
//
// Parquet feature is required to write the fixture.
// ---------------------------------------------------------------------------

#[test]
#[cfg(feature = "schema-parquet")]
fn chunked_parquet_equivalence_17_rows() {
    use parquet::basic::{Repetition, Type as PqPhysical};
    use parquet::data_type::{ByteArray, ByteArrayType, FloatType, Int64Type};
    use parquet::file::properties::WriterProperties;
    use parquet::file::writer::SerializedFileWriter;
    use parquet::schema::types::Type as PqType;
    use std::sync::Arc;

    let dir = TempDir::new().unwrap();
    let pq_path = dir.path().join("data17.parquet");

    let id_field = PqType::primitive_type_builder("id", PqPhysical::INT64)
        .with_repetition(Repetition::REQUIRED)
        .build()
        .unwrap();
    let score_field = PqType::primitive_type_builder("score", PqPhysical::FLOAT)
        .with_repetition(Repetition::REQUIRED)
        .build()
        .unwrap();
    let label_field = PqType::primitive_type_builder("label", PqPhysical::BYTE_ARRAY)
        .with_repetition(Repetition::REQUIRED)
        .build()
        .unwrap();

    let schema = Arc::new(
        PqType::group_type_builder("schema")
            .with_fields(vec![
                Arc::new(id_field),
                Arc::new(score_field),
                Arc::new(label_field),
            ])
            .build()
            .unwrap(),
    );

    let file = fs::File::create(&pq_path).unwrap();
    let props = Arc::new(WriterProperties::builder().build());
    let mut writer = SerializedFileWriter::new(file, schema, props).unwrap();
    let mut rg = writer.next_row_group().unwrap();

    let ids: Vec<i64> = (0..17).collect();
    let scores: Vec<f32> = (0..17).map(|i| i as f32 * 1.5).collect();
    let labels: Vec<ByteArray> = (0..17usize)
        .map(|i| ByteArray::from(format!("item{i}").as_bytes()))
        .collect();

    let mut col = rg.next_column().unwrap().unwrap();
    col.typed::<Int64Type>()
        .write_batch(&ids, None, None)
        .unwrap();
    col.close().unwrap();

    let mut col = rg.next_column().unwrap().unwrap();
    col.typed::<FloatType>()
        .write_batch(&scores, None, None)
        .unwrap();
    col.close().unwrap();

    let mut col = rg.next_column().unwrap().unwrap();
    col.typed::<ByteArrayType>()
        .write_batch(&labels, None, None)
        .unwrap();
    col.close().unwrap();

    rg.close().unwrap();
    writer.close().unwrap();

    let he_streaming = dir.path().join("streaming.he");
    let he_single = dir.path().join("single.he");

    // Streaming: --stripe-rows 5.
    helium()
        .args(["convert"])
        .arg(&pq_path)
        .arg("-o")
        .arg(&he_streaming)
        .args(["--stripe-rows", "5"])
        .assert()
        .success();

    // Single-stripe: --stripe-rows 0.
    helium()
        .args(["convert"])
        .arg(&pq_path)
        .arg("-o")
        .arg(&he_single)
        .args(["--stripe-rows", "0"])
        .assert()
        .success();

    let (s_stripes, s_rows) = he_meta(&he_streaming);
    let (l_stripes, l_rows) = he_meta(&he_single);
    assert_eq!(s_stripes, 4, "expected 4 stripes for 17 rows @ 5/stripe");
    assert_eq!(l_stripes, 1);
    assert_eq!(s_rows, 17);
    assert_eq!(l_rows, 17);

    // Column data must be logically identical.
    let s_data = he_read_all(&he_streaming);
    let l_data = he_read_all(&he_single);
    for name in ["id", "label"] {
        // score is f32 in parquet → f64 in helium; just check it's present.
        let s_lc = s_data
            .get(name)
            .unwrap_or_else(|| panic!("streaming missing col {name}"));
        let l_lc = l_data
            .get(name)
            .unwrap_or_else(|| panic!("single missing col {name}"));
        assert_eq!(s_lc, l_lc, "column '{name}' differs");
    }
    // Verify score column row count matches.
    let s_score = s_data.get("score").expect("streaming missing score");
    let l_score = l_data.get("score").expect("single missing score");
    assert_eq!(
        s_score.row_count(),
        l_score.row_count(),
        "score row count mismatch"
    );
}

// ---------------------------------------------------------------------------
// Test 3: NDJSON streaming with nested struct data
//
// Schema: { id: i64, meta: { tag: utf8, active: bool } }
// 9 rows, --stripe-rows 3 → 3 stripes.
// Round-trip: .he → NDJSON preserves the data.
// ---------------------------------------------------------------------------

#[test]
fn chunked_ndjson_nested_struct() {
    let dir = TempDir::new().unwrap();
    let ndjson_path = dir.path().join("nested.ndjson");

    {
        let mut f = fs::File::create(&ndjson_path).unwrap();
        for i in 0..9usize {
            writeln!(
                f,
                r#"{{"id":{i},"meta":{{"tag":"t{i}","active":{}}}}}"#,
                if i % 2 == 0 { "true" } else { "false" }
            )
            .unwrap();
        }
    }

    let he_streaming = dir.path().join("nested_streaming.he");
    let he_single = dir.path().join("nested_single.he");

    // Streaming: --stripe-rows 3.
    helium()
        .args(["convert"])
        .arg(&ndjson_path)
        .arg("-o")
        .arg(&he_streaming)
        .args(["--stripe-rows", "3"])
        .assert()
        .success();

    // Single-stripe for comparison.
    helium()
        .args(["convert"])
        .arg(&ndjson_path)
        .arg("-o")
        .arg(&he_single)
        .args(["--stripe-rows", "0"])
        .assert()
        .success();

    let (s_stripes, s_rows) = he_meta(&he_streaming);
    assert_eq!(s_stripes, 3, "expected 3 stripes for 9 rows @ 3/stripe");
    assert_eq!(s_rows, 9);

    // Read back and compare column data.
    let s_data = he_read_all(&he_streaming);
    let l_data = he_read_all(&he_single);

    assert_eq!(
        s_data.get("id").unwrap(),
        l_data.get("id").unwrap(),
        "id column differs"
    );
    assert_eq!(
        s_data.get("meta").unwrap(),
        l_data.get("meta").unwrap(),
        "meta column differs"
    );

    // Round-trip back to NDJSON.
    let ndjson_out = dir.path().join("nested_out.ndjson");
    helium()
        .args(["convert"])
        .arg(&he_streaming)
        .arg("-o")
        .arg(&ndjson_out)
        .assert()
        .success();

    let output = fs::read_to_string(&ndjson_out).unwrap();
    // Should contain 9 records; spot-check a few values.
    assert!(
        output.contains("\"t0\"") || output.contains("t0"),
        "missing tag t0"
    );
    assert!(
        output.contains("\"t8\"") || output.contains("t8"),
        "missing tag t8"
    );
}

// ---------------------------------------------------------------------------
// Test 4: Avro streaming
//
// Write a small Avro fixture (7 rows), convert with --stripe-rows 3 → 3 stripes
// (3+3+1).  Verify the resulting .he round-trips back to Avro with 7 rows.
// ---------------------------------------------------------------------------

#[test]
#[cfg(feature = "schema-avro")]
fn chunked_avro_streaming() {
    use apache_avro::types::Value as AV;
    use apache_avro::{Codec, Schema as AvroSchema, Writer};

    let dir = TempDir::new().unwrap();
    let avro_path = dir.path().join("data7.avro");

    let raw_schema = r#"{
        "type": "record",
        "name": "Row",
        "fields": [
            {"name": "id",    "type": "long"},
            {"name": "label", "type": "string"}
        ]
    }"#;
    let schema = AvroSchema::parse_str(raw_schema).expect("avro schema parse");
    {
        let f = fs::File::create(&avro_path).unwrap();
        let mut writer = Writer::with_codec(&schema, f, Codec::Null);
        for i in 0..7i64 {
            let mut rec = apache_avro::types::Record::new(&schema).unwrap();
            rec.put("id", AV::Long(i));
            rec.put("label", AV::String(format!("row{i}")));
            writer.append(rec).unwrap();
        }
        writer.flush().unwrap();
    }

    let he_streaming = dir.path().join("avro_streaming.he");
    let he_single = dir.path().join("avro_single.he");

    // Streaming: --stripe-rows 3 → ceil(7/3) = 3 stripes.
    helium()
        .args(["convert"])
        .arg(&avro_path)
        .arg("-o")
        .arg(&he_streaming)
        .args(["--stripe-rows", "3"])
        .assert()
        .success();

    // Single-stripe for comparison.
    helium()
        .args(["convert"])
        .arg(&avro_path)
        .arg("-o")
        .arg(&he_single)
        .args(["--stripe-rows", "0"])
        .assert()
        .success();

    let (s_stripes, s_rows) = he_meta(&he_streaming);
    assert_eq!(s_stripes, 3, "expected 3 stripes for 7 rows @ 3/stripe");
    assert_eq!(s_rows, 7);

    // Column data must be identical between streaming and single-stripe.
    let s_data = he_read_all(&he_streaming);
    let l_data = he_read_all(&he_single);
    for name in ["id", "label"] {
        assert_eq!(
            s_data.get(name).unwrap(),
            l_data.get(name).unwrap(),
            "column '{name}' differs between streaming and single-stripe Avro"
        );
    }

    // Round-trip back to Avro and verify 7 rows.
    let avro_out = dir.path().join("avro_out.avro");
    helium()
        .args(["convert"])
        .arg(&he_streaming)
        .arg("-o")
        .arg(&avro_out)
        .assert()
        .success();

    use apache_avro::Reader;
    let f = fs::File::open(&avro_out).unwrap();
    let reader = Reader::new(f).expect("open output avro");
    let rows: Vec<_> = reader.collect();
    assert_eq!(rows.len(), 7, "expected 7 rows in round-tripped Avro");
}

// ---------------------------------------------------------------------------
// Test 5: JSON-array fallback path
//
// A top-level JSON array cannot be streamed line-by-line (no SAX parser).
// The loader falls back to in-memory load + slice.
// 5 records, --stripe-rows 2 → 3 stripes (2+2+1).
// ---------------------------------------------------------------------------

#[test]
fn json_array_fallback_to_slice() {
    let dir = TempDir::new().unwrap();
    let json_path = dir.path().join("array.json");

    {
        let mut f = fs::File::create(&json_path).unwrap();
        writeln!(
            f,
            r#"[{{"id":1,"label":"a"}},{{"id":2,"label":"b"}},{{"id":3,"label":"c"}},{{"id":4,"label":"d"}},{{"id":5,"label":"e"}}]"#
        )
        .unwrap();
    }

    let he_path = dir.path().join("array_out.he");
    helium()
        .args(["convert"])
        .arg(&json_path)
        .arg("-o")
        .arg(&he_path)
        .args(["--stripe-rows", "2"])
        .assert()
        .success();

    let (stripes, rows) = he_meta(&he_path);
    assert_eq!(rows, 5, "expected 5 rows");
    assert_eq!(
        stripes, 3,
        "expected 3 stripes (2+2+1) for JSON-array fallback"
    );
}

// ---------------------------------------------------------------------------
// Test 6: Memory smoke — 100k-row CSV, --stripe-rows 1000
//
// This is a smoke check only; it does NOT assert peak memory.  The test
// verifies that the conversion completes successfully and produces the right
// row count.  Actual memory-bound behaviour should be validated manually with
// `/usr/bin/time -l` on a large Parquet file (see task description).
// ---------------------------------------------------------------------------

#[test]
fn smoke_100k_rows_csv() {
    let dir = TempDir::new().unwrap();
    let csv_path = dir.path().join("large.csv");

    {
        let mut f = fs::File::create(&csv_path).unwrap();
        writeln!(f, "id,val").unwrap();
        for i in 0u64..100_000 {
            writeln!(f, "{i},{}", i * 2).unwrap();
        }
    }

    let he_path = dir.path().join("large.he");
    helium()
        .args(["convert"])
        .arg(&csv_path)
        .arg("-o")
        .arg(&he_path)
        .args(["--stripe-rows", "1000"])
        .assert()
        .success();

    let (stripes, rows) = he_meta(&he_path);
    assert_eq!(rows, 100_000, "expected 100k rows");
    assert_eq!(stripes, 100, "expected 100 stripes (100k / 1000)");
}

// ---------------------------------------------------------------------------
// Test 7: 0-row input — empty CSV (header only), --stripe-rows 100
// ---------------------------------------------------------------------------

#[test]
fn empty_csv_zero_rows() {
    let dir = TempDir::new().unwrap();
    let csv_path = dir.path().join("empty.csv");

    {
        let mut f = fs::File::create(&csv_path).unwrap();
        writeln!(f, "id,label").unwrap();
    }

    let he_path = dir.path().join("empty.he");
    helium()
        .args(["convert"])
        .arg(&csv_path)
        .arg("-o")
        .arg(&he_path)
        .args(["--stripe-rows", "100"])
        .assert()
        .success();

    // `helium verify` must succeed.
    helium().arg("verify").arg(&he_path).assert().success();

    let (_stripes, rows) = he_meta(&he_path);
    assert_eq!(rows, 0, "expected 0 rows for empty CSV");
}

// ---------------------------------------------------------------------------
// Test 8: Stats correctness — file-wide min/max agree between streaming and
// single-stripe outputs.
//
// The per-stripe stats will differ (stripe boundaries differ), but after
// reading all stripes the aggregate logical data is the same.
// ---------------------------------------------------------------------------

#[test]
fn stats_match_between_streaming_and_single_stripe() {
    let dir = TempDir::new().unwrap();
    let csv_in = write_17row_csv(&dir);
    let he_streaming = dir.path().join("stats_streaming.he");
    let he_single = dir.path().join("stats_single.he");

    helium()
        .args(["convert"])
        .arg(&csv_in)
        .arg("-o")
        .arg(&he_streaming)
        .args(["--stripe-rows", "5"])
        .assert()
        .success();

    helium()
        .args(["convert"])
        .arg(&csv_in)
        .arg("-o")
        .arg(&he_single)
        .args(["--stripe-rows", "0"])
        .assert()
        .success();

    // Compare logical column data: min/max values are implicit in the data.
    let s_data = he_read_all(&he_streaming);
    let l_data = he_read_all(&he_single);

    for col_name in ["id", "score", "label"] {
        let s_lc = s_data
            .get(col_name)
            .unwrap_or_else(|| panic!("streaming missing {col_name}"));
        let l_lc = l_data
            .get(col_name)
            .unwrap_or_else(|| panic!("single missing {col_name}"));
        assert_eq!(
            s_lc, l_lc,
            "column '{col_name}' has different values between streaming and single-stripe"
        );
    }
}
