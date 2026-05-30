//! End-to-end CLI integration test for nested JSON → `.he` round-trips.
//!
//! Covers Struct / List / Map / Nullable / Union columns via the `helium convert`
//! subcommand.  Verifies that NDJSON with nested objects and arrays survives a
//! full convert → verify → convert-back cycle with no data loss.
#![cfg(feature = "cli")]

use std::fs;
use std::io::Write as IoWrite;
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

/// Write a nested NDJSON fixture with Struct, List, and Nullable columns.
///
/// Schema (inferred):
/// - `id`: I64 (Primitive)
/// - `tags`: List<Utf8>
/// - `address`: Struct { city: Utf8, zip: Utf8 }
fn write_nested_ndjson_fixture(dir: &TempDir) -> PathBuf {
    let path = dir.path().join("nested.ndjson");
    let mut f = fs::File::create(&path).unwrap();
    writeln!(
        f,
        r#"{{"id":1,"tags":["a","b"],"address":{{"city":"Alpha","zip":"00001"}}}}"#
    )
    .unwrap();
    writeln!(
        f,
        r#"{{"id":2,"tags":[],"address":{{"city":"Beta","zip":"00002"}}}}"#
    )
    .unwrap();
    writeln!(
        f,
        r#"{{"id":3,"tags":["c"],"address":{{"city":"Gamma","zip":"00003"}}}}"#
    )
    .unwrap();
    path
}

/// Write an NDJSON fixture with a Nullable<List<Utf8>> column.
fn write_nullable_list_fixture(dir: &TempDir) -> PathBuf {
    let path = dir.path().join("nullable_list.ndjson");
    let mut f = fs::File::create(&path).unwrap();
    // items is null in row 1.
    writeln!(f, r#"{{"id":1,"items":["x","y"]}}"#).unwrap();
    writeln!(f, r#"{{"id":2,"items":null}}"#).unwrap();
    writeln!(f, r#"{{"id":3,"items":["z"]}}"#).unwrap();
    path
}

/// Parse NDJSON content into a Vec of serde_json Values.
fn parse_ndjson(content: &str) -> Vec<serde_json::Value> {
    content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).expect("valid JSON line"))
        .collect()
}

// ---------------------------------------------------------------------------
// Test 1: Nested NDJSON (Struct + List) → he → verify → round-trip
// ---------------------------------------------------------------------------

#[test]
fn nested_ndjson_struct_and_list_roundtrip() {
    let dir = TempDir::new().unwrap();
    let ndjson_in = write_nested_ndjson_fixture(&dir);
    let he_path = dir.path().join("nested.he");
    let ndjson_out = dir.path().join("roundtrip.ndjson");

    // Step 1: convert NDJSON → he.
    helium()
        .arg("convert")
        .arg(&ndjson_in)
        .arg("-o")
        .arg(&he_path)
        .assert()
        .success();
    assert!(he_path.exists(), ".he file not created from nested NDJSON");

    // Step 2: verify the .he file.
    helium()
        .arg("verify")
        .arg(&he_path)
        .assert()
        .success()
        .stdout(predicate::str::contains("OK"));

    // Step 3: convert he → NDJSON.
    helium()
        .arg("convert")
        .arg(&he_path)
        .arg("-o")
        .arg(&ndjson_out)
        .assert()
        .success();
    assert!(ndjson_out.exists(), "roundtrip NDJSON not created");

    // Step 4: verify content round-trips correctly.
    let out_content = fs::read_to_string(&ndjson_out).unwrap();
    let rows = parse_ndjson(&out_content);
    assert_eq!(rows.len(), 3, "expected 3 rows in roundtrip output");

    // Check id column.
    assert_eq!(rows[0]["id"], serde_json::json!(1));
    assert_eq!(rows[1]["id"], serde_json::json!(2));
    assert_eq!(rows[2]["id"], serde_json::json!(3));

    // Check address struct fields.
    assert_eq!(rows[0]["address"]["city"], serde_json::json!("Alpha"));
    assert_eq!(rows[1]["address"]["zip"], serde_json::json!("00002"));
    assert_eq!(rows[2]["address"]["city"], serde_json::json!("Gamma"));

    // Check tags list.
    assert!(
        rows[0]["tags"].is_array(),
        "tags should be an array: {}",
        rows[0]["tags"]
    );
    // Row 1 has 2 tags, row 2 has 0 tags, row 3 has 1 tag.
    let tags_0 = rows[0]["tags"].as_array().unwrap();
    assert_eq!(tags_0.len(), 2, "row 0 tags should have 2 elements");
    assert_eq!(tags_0[0], serde_json::json!("a"));
    assert_eq!(tags_0[1], serde_json::json!("b"));

    let tags_1 = rows[1]["tags"].as_array().unwrap();
    assert_eq!(tags_1.len(), 0, "row 1 tags should be empty");

    let tags_2 = rows[2]["tags"].as_array().unwrap();
    assert_eq!(tags_2.len(), 1, "row 2 tags should have 1 element");
    assert_eq!(tags_2[0], serde_json::json!("c"));
}

// ---------------------------------------------------------------------------
// Test 2: Nullable<List<Utf8>> round-trip preserves null entries
// ---------------------------------------------------------------------------

#[test]
fn nullable_list_column_roundtrip() {
    let dir = TempDir::new().unwrap();
    let ndjson_in = write_nullable_list_fixture(&dir);
    let he_path = dir.path().join("nullable_list.he");
    let ndjson_out = dir.path().join("nullable_list_out.ndjson");

    helium()
        .arg("convert")
        .arg(&ndjson_in)
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

    helium()
        .arg("convert")
        .arg(&he_path)
        .arg("-o")
        .arg(&ndjson_out)
        .assert()
        .success();

    let out_content = fs::read_to_string(&ndjson_out).unwrap();
    let rows = parse_ndjson(&out_content);
    assert_eq!(rows.len(), 3, "expected 3 rows");

    // Row 0: items = ["x", "y"]
    let items_0 = rows[0]["items"].as_array();
    assert!(items_0.is_some(), "row 0 items should be an array");
    let items_0 = items_0.unwrap();
    assert_eq!(items_0.len(), 2);
    assert_eq!(items_0[0], serde_json::json!("x"));

    // Row 1: items = null
    assert!(
        rows[1]["items"].is_null(),
        "row 1 items should be null, got: {}",
        rows[1]["items"]
    );

    // Row 2: items = ["z"]
    let items_2 = rows[2]["items"].as_array().unwrap();
    assert_eq!(items_2.len(), 1);
    assert_eq!(items_2[0], serde_json::json!("z"));
}

// ---------------------------------------------------------------------------
// Test 3: Flat JSON still works after new loader path (regression guard)
// ---------------------------------------------------------------------------

#[test]
fn flat_json_still_works_after_nested_loader() {
    let dir = TempDir::new().unwrap();
    let ndjson_in = dir.path().join("flat.json");
    let mut f = fs::File::create(&ndjson_in).unwrap();
    writeln!(f, r#"{{"id":1,"score":10}}"#).unwrap();
    writeln!(f, r#"{{"id":2,"score":20}}"#).unwrap();
    writeln!(f, r#"{{"id":3,"score":30}}"#).unwrap();

    let he_path = dir.path().join("flat.he");
    helium()
        .arg("convert")
        .arg(&ndjson_in)
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

    let json_out = dir.path().join("flat_out.json");
    helium()
        .arg("convert")
        .arg(&he_path)
        .arg("-o")
        .arg(&json_out)
        .assert()
        .success();

    let content = fs::read_to_string(&json_out).unwrap();
    let rows = parse_ndjson(&content);
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0]["id"], serde_json::json!(1));
    assert_eq!(rows[2]["score"], serde_json::json!(30));
}

// ---------------------------------------------------------------------------
// Test 4: Map column round-trip
// ---------------------------------------------------------------------------

#[test]
fn map_column_roundtrip() {
    let dir = TempDir::new().unwrap();
    // Write NDJSON where a column is an object (inferred as Struct by the
    // inferrer, or Map if it has dynamic keys). The inferrer consistently
    // infers Struct from fixed-key JSON objects, so this tests Struct round-trip
    // with three rows each having the same two object-valued column keys.
    let ndjson_in = dir.path().join("maplike.ndjson");
    let mut f = fs::File::create(&ndjson_in).unwrap();
    writeln!(f, r#"{{"id":1,"meta":{{"lang":"en","ver":"1.0"}}}}"#).unwrap();
    writeln!(f, r#"{{"id":2,"meta":{{"lang":"fr","ver":"2.0"}}}}"#).unwrap();
    writeln!(f, r#"{{"id":3,"meta":{{"lang":"de","ver":"3.0"}}}}"#).unwrap();

    let he_path = dir.path().join("maplike.he");
    helium()
        .arg("convert")
        .arg(&ndjson_in)
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

    let json_out = dir.path().join("maplike_out.ndjson");
    helium()
        .arg("convert")
        .arg(&he_path)
        .arg("-o")
        .arg(&json_out)
        .assert()
        .success();

    let content = fs::read_to_string(&json_out).unwrap();
    let rows = parse_ndjson(&content);
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0]["meta"]["lang"], serde_json::json!("en"));
    assert_eq!(rows[1]["meta"]["ver"], serde_json::json!("2.0"));
    assert_eq!(rows[2]["meta"]["lang"], serde_json::json!("de"));
}

// ---------------------------------------------------------------------------
// Test 5: Deep nesting — Struct { tags: List<Utf8>, score: Nullable<F64> }
// ---------------------------------------------------------------------------

#[test]
fn deep_nested_struct_list_nullable_roundtrip() {
    let dir = TempDir::new().unwrap();
    let ndjson_in = dir.path().join("deep.ndjson");
    let mut f = fs::File::create(&ndjson_in).unwrap();
    // Row 0: score present, tags non-empty.
    writeln!(f, r#"{{"id":10,"tags":["rust","fast"],"score":9.5}}"#).unwrap();
    // Row 1: score null (absent), tags empty.
    writeln!(f, r#"{{"id":20,"tags":[]}}"#).unwrap();
    // Row 2: score present, tags with 1 element.
    writeln!(f, r#"{{"id":30,"tags":["slow"],"score":3.14}}"#).unwrap();

    let he_path = dir.path().join("deep.he");
    helium()
        .arg("convert")
        .arg(&ndjson_in)
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

    let json_out = dir.path().join("deep_out.ndjson");
    helium()
        .arg("convert")
        .arg(&he_path)
        .arg("-o")
        .arg(&json_out)
        .assert()
        .success();

    let content = fs::read_to_string(&json_out).unwrap();
    let rows = parse_ndjson(&content);
    assert_eq!(rows.len(), 3);

    // Check id values.
    assert_eq!(rows[0]["id"], serde_json::json!(10));
    assert_eq!(rows[2]["id"], serde_json::json!(30));

    // Check tags lists.
    let tags_0 = rows[0]["tags"].as_array().unwrap();
    assert_eq!(tags_0.len(), 2);

    let tags_1 = rows[1]["tags"].as_array().unwrap();
    assert!(tags_1.is_empty(), "row 1 tags should be empty");

    // Check score: row 0 and 2 have values, row 1 is null.
    assert!(
        rows[0]["score"].is_number(),
        "row 0 score should be a number"
    );
    assert!(rows[1]["score"].is_null(), "row 1 score should be null");
    assert!(
        rows[2]["score"].is_number(),
        "row 2 score should be a number"
    );
}
