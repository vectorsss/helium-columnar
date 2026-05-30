//! CLI integration tests for Avro Object Container Format (`.avro`) round-trips.
//!
//! Tests cover:
//! 1. Flat `.avro` → `.he` → `.avro` round-trip (primitives)
//! 2. `.avro` with nullable fields → `.he` → `.avro` round-trip
//! 3. `.avro` with nested struct → `.he` → `.avro` round-trip
//! 4. `.avro` with list field → `.he` → `.avro` round-trip
//!
//! All tests are gated on `feature = "cli"` (which implies `schema-avro`).
#![cfg(feature = "cli")]

use std::fs;
use std::path::PathBuf;

use assert_cmd::Command;
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn helium() -> Command {
    Command::cargo_bin("helium").expect("helium binary not found")
}

/// Write an Avro OCF file with a flat schema: id (long) + label (string).
fn write_flat_avro(dir: &TempDir) -> PathBuf {
    use apache_avro::types::Value as AV;
    use apache_avro::{Codec, Schema as AvroSchema, Writer};

    let path = dir.path().join("flat.avro");
    let raw_schema = r#"{
        "type": "record",
        "name": "Row",
        "fields": [
            {"name": "id",    "type": "long"},
            {"name": "label", "type": "string"}
        ]
    }"#;
    let schema = AvroSchema::parse_str(raw_schema).expect("schema parse");
    let out = fs::File::create(&path).expect("create avro");
    let mut writer = Writer::with_codec(&schema, out, Codec::Null);
    for (id, label) in [(1i64, "alpha"), (2, "beta"), (3, "gamma")] {
        let mut record = apache_avro::types::Record::new(&schema).expect("record");
        record.put("id", AV::Long(id));
        record.put("label", AV::String(label.to_string()));
        writer.append(record).expect("append");
    }
    writer.flush().expect("flush");
    path
}

/// Write an Avro OCF file with a nullable field: id (long) + score (["null","double"]).
fn write_nullable_avro(dir: &TempDir) -> PathBuf {
    use apache_avro::types::Value as AV;
    use apache_avro::{Codec, Schema as AvroSchema, Writer};

    let path = dir.path().join("nullable.avro");
    let raw_schema = r#"{
        "type": "record",
        "name": "NullableRow",
        "fields": [
            {"name": "id",    "type": "long"},
            {"name": "score", "type": ["null", "double"], "default": null}
        ]
    }"#;
    let schema = AvroSchema::parse_str(raw_schema).expect("schema parse");
    let out = fs::File::create(&path).expect("create avro");
    let mut writer = Writer::with_codec(&schema, out, Codec::Null);

    let rows: &[(i64, Option<f64>)] = &[(10, Some(1.5)), (20, None), (30, Some(3.5))];
    for (id, score) in rows {
        let mut record = apache_avro::types::Record::new(&schema).expect("record");
        record.put("id", AV::Long(*id));
        let score_val = match score {
            None => AV::Union(0, Box::new(AV::Null)),
            Some(v) => AV::Union(1, Box::new(AV::Double(*v))),
        };
        record.put("score", score_val);
        writer.append(record).expect("append");
    }
    writer.flush().expect("flush");
    path
}

/// Write an Avro OCF file with a nested struct field.
fn write_nested_avro(dir: &TempDir) -> PathBuf {
    use apache_avro::types::Value as AV;
    use apache_avro::{Codec, Schema as AvroSchema, Writer};

    let path = dir.path().join("nested.avro");
    let raw_schema = r#"{
        "type": "record",
        "name": "Event",
        "fields": [
            {"name": "name", "type": "string"},
            {"name": "loc", "type": {
                "type": "record",
                "name": "Location",
                "fields": [
                    {"name": "lat", "type": "double"},
                    {"name": "lon", "type": "double"}
                ]
            }}
        ]
    }"#;
    let schema = AvroSchema::parse_str(raw_schema).expect("schema parse");
    let out = fs::File::create(&path).expect("create avro");
    let mut writer = Writer::with_codec(&schema, out, Codec::Null);

    let rows: &[(&str, f64, f64)] = &[
        ("p1", 37.7, -122.4),
        ("p2", 40.7, -74.0),
        ("p3", 51.5, -0.1),
    ];
    for (name, lat, lon) in rows {
        let mut record = apache_avro::types::Record::new(&schema).expect("record");
        record.put("name", AV::String(name.to_string()));
        let loc = AV::Record(vec![
            ("lat".to_string(), AV::Double(*lat)),
            ("lon".to_string(), AV::Double(*lon)),
        ]);
        record.put("loc", loc);
        writer.append(record).expect("append");
    }
    writer.flush().expect("flush");
    path
}

/// Write an Avro OCF file with a list-of-long field.
fn write_list_avro(dir: &TempDir) -> PathBuf {
    use apache_avro::types::Value as AV;
    use apache_avro::{Codec, Schema as AvroSchema, Writer};

    let path = dir.path().join("list.avro");
    let raw_schema = r#"{
        "type": "record",
        "name": "ListRow",
        "fields": [
            {"name": "key",  "type": "string"},
            {"name": "nums", "type": {"type": "array", "items": "long"}}
        ]
    }"#;
    let schema = AvroSchema::parse_str(raw_schema).expect("schema parse");
    let out = fs::File::create(&path).expect("create avro");
    let mut writer = Writer::with_codec(&schema, out, Codec::Null);

    let rows: &[(&str, &[i64])] = &[("a", &[1, 2, 3]), ("b", &[]), ("c", &[10, 20])];
    for (key, nums) in rows {
        let mut record = apache_avro::types::Record::new(&schema).expect("record");
        record.put("key", AV::String(key.to_string()));
        let arr = AV::Array(nums.iter().map(|n| AV::Long(*n)).collect());
        record.put("nums", arr);
        writer.append(record).expect("append");
    }
    writer.flush().expect("flush");
    path
}

// ---------------------------------------------------------------------------
// 1. Flat record round-trip: .avro → .he → .avro
// ---------------------------------------------------------------------------

#[test]
fn avro_flat_to_he_to_avro_roundtrip() {
    let dir = TempDir::new().unwrap();
    let avro_in = write_flat_avro(&dir);
    let he_path = dir.path().join("flat.he");
    let avro_out = dir.path().join("flat_out.avro");

    // .avro → .he
    helium()
        .arg("convert")
        .arg(&avro_in)
        .arg("-o")
        .arg(&he_path)
        .assert()
        .success();
    assert!(he_path.exists(), ".he file not created from flat .avro");

    // .he → .avro
    helium()
        .arg("convert")
        .arg(&he_path)
        .arg("-o")
        .arg(&avro_out)
        .assert()
        .success();
    assert!(avro_out.exists(), "output .avro not created");

    // Verify the round-tripped Avro file is readable and has 3 rows.
    use apache_avro::Reader;
    let f = fs::File::open(&avro_out).unwrap();
    let reader = Reader::new(f).expect("open roundtrip avro");
    let values: Vec<_> = reader.collect();
    assert_eq!(values.len(), 3, "expected 3 rows in round-tripped .avro");
}

// ---------------------------------------------------------------------------
// 2. Nullable field round-trip: .avro → .he → .avro
// ---------------------------------------------------------------------------

#[test]
fn avro_nullable_to_he_to_avro_roundtrip() {
    let dir = TempDir::new().unwrap();
    let avro_in = write_nullable_avro(&dir);
    let he_path = dir.path().join("nullable.he");
    let avro_out = dir.path().join("nullable_out.avro");

    // .avro → .he
    helium()
        .arg("convert")
        .arg(&avro_in)
        .arg("-o")
        .arg(&he_path)
        .assert()
        .success();
    assert!(he_path.exists(), ".he file not created from nullable .avro");

    // .he → .avro
    helium()
        .arg("convert")
        .arg(&he_path)
        .arg("-o")
        .arg(&avro_out)
        .assert()
        .success();
    assert!(avro_out.exists(), "output nullable .avro not created");

    // Verify readable.
    use apache_avro::Reader;
    let f = fs::File::open(&avro_out).unwrap();
    let reader = Reader::new(f).expect("open roundtrip nullable avro");
    let values: Vec<_> = reader.collect();
    assert_eq!(values.len(), 3, "expected 3 rows");
}

// ---------------------------------------------------------------------------
// 3. Nested struct round-trip: .avro → .he → .avro
// ---------------------------------------------------------------------------

#[test]
fn avro_nested_struct_to_he_to_avro_roundtrip() {
    let dir = TempDir::new().unwrap();
    let avro_in = write_nested_avro(&dir);
    let he_path = dir.path().join("nested.he");
    let avro_out = dir.path().join("nested_out.avro");

    // .avro → .he
    helium()
        .arg("convert")
        .arg(&avro_in)
        .arg("-o")
        .arg(&he_path)
        .assert()
        .success();
    assert!(he_path.exists(), ".he file not created from nested .avro");

    // .he → .avro
    helium()
        .arg("convert")
        .arg(&he_path)
        .arg("-o")
        .arg(&avro_out)
        .assert()
        .success();
    assert!(avro_out.exists(), "output nested .avro not created");

    // Verify 3 readable rows.
    use apache_avro::Reader;
    let f = fs::File::open(&avro_out).unwrap();
    let reader = Reader::new(f).expect("open roundtrip nested avro");
    let values: Vec<_> = reader.collect();
    assert_eq!(values.len(), 3, "expected 3 rows in nested round-trip");
}

// ---------------------------------------------------------------------------
// 4. List field round-trip: .avro → .he → .avro
// ---------------------------------------------------------------------------

#[test]
fn avro_list_to_he_to_avro_roundtrip() {
    let dir = TempDir::new().unwrap();
    let avro_in = write_list_avro(&dir);
    let he_path = dir.path().join("list.he");
    let avro_out = dir.path().join("list_out.avro");

    // .avro → .he
    helium()
        .arg("convert")
        .arg(&avro_in)
        .arg("-o")
        .arg(&he_path)
        .assert()
        .success();
    assert!(he_path.exists(), ".he file not created from list .avro");

    // .he → .avro
    helium()
        .arg("convert")
        .arg(&he_path)
        .arg("-o")
        .arg(&avro_out)
        .assert()
        .success();
    assert!(avro_out.exists(), "output list .avro not created");

    // Verify 3 readable rows.
    use apache_avro::Reader;
    let f = fs::File::open(&avro_out).unwrap();
    let reader = Reader::new(f).expect("open roundtrip list avro");
    let values: Vec<_> = reader.collect();
    assert_eq!(values.len(), 3, "expected 3 rows in list round-trip");
}
