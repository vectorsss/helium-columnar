//! Integration tests for `helium convert` — Struct flattening in CSV output.
//!
//! Covers:
//! 1. Struct flattens to dotted columns in the CSV header.
//! 2. Nested Struct flattens recursively (outer.middle.inner).
//! 3. Nullable<Struct> emits empty cells for all sub-columns when null.
//! 4. Mixed Struct + List: Struct flattens, List JSON-stringifies.
//! 5. Round-trip regression: flat schema produces the same CSV output as before.
//! 6. --csv-strict errors on List columns.
//! 7. --csv-strict accepts pure-Struct schemas (no error).
//!
//! Plus unit tests for `write_csv_with_options` directly.
#![cfg(feature = "cli")]

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use assert_cmd::Command;
use predicates::prelude::*;
use tempfile::TempDir;

// Re-use from the library to build test fixtures.
use helium::schema::csv::{CsvWriteOptions, write_csv, write_csv_with_options};
use helium::{
    CoderRegistry, ColumnData, ColumnSpec, FieldSpec, HeliumWriter, LogicalColumn, LogicalType,
    Schema,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn helium() -> Command {
    Command::cargo_bin("helium").expect("helium binary not found")
}

/// Write a `.he` file from the given schema and columns, returning the path.
fn write_he(dir: &TempDir, schema: Schema, columns: HashMap<String, LogicalColumn>) -> PathBuf {
    let path = dir.path().join("data.he");
    let file = fs::File::create(&path).unwrap();
    let registry = CoderRegistry::default();
    let mut writer = HeliumWriter::new(file, schema, &registry).unwrap();
    for (name, lc) in columns {
        writer.write_column(&name, lc).unwrap();
    }
    writer.finish().unwrap();
    path
}

/// Default encodings for a Utf8 field.
fn field_enc_utf8() -> Vec<Vec<helium::CoderSpec>> {
    helium::schema::encodings::default_encodings(&LogicalType::Utf8)
}

/// Build a ColumnSpec for a Primitive column with default encodings.
fn col_prim(name: &str, dt: helium::DataType) -> ColumnSpec {
    let enc =
        helium::schema::encodings::default_encodings(&LogicalType::Primitive { data_type: dt });
    ColumnSpec::new(name, LogicalType::Primitive { data_type: dt }, enc)
}

// ---------------------------------------------------------------------------
// Test 1: Struct flattens to dotted columns
// ---------------------------------------------------------------------------

#[test]
fn struct_flattens_to_dotted_columns() {
    let dir = TempDir::new().unwrap();

    // Schema: id: I64, addr: Struct { street: Utf8, zip: Utf8 }
    let schema = Schema::new(vec![
        col_prim("id", helium::DataType::I64),
        ColumnSpec::struct_col(
            "addr",
            vec![
                FieldSpec::utf8(
                    "street",
                    field_enc_utf8()[0].clone(),
                    field_enc_utf8()[1].clone(),
                ),
                FieldSpec::utf8(
                    "zip",
                    field_enc_utf8()[0].clone(),
                    field_enc_utf8()[1].clone(),
                ),
            ],
        ),
    ]);

    let mut cols: HashMap<String, LogicalColumn> = HashMap::new();
    cols.insert(
        "id".to_string(),
        LogicalColumn::Primitive(ColumnData::I64(vec![1, 2, 3])),
    );
    cols.insert(
        "addr".to_string(),
        LogicalColumn::Struct {
            fields: vec![
                (
                    "street".to_string(),
                    LogicalColumn::Utf8(vec![
                        "Main St".to_string(),
                        "Oak Ave".to_string(),
                        "Pine Rd".to_string(),
                    ]),
                ),
                (
                    "zip".to_string(),
                    LogicalColumn::Utf8(vec![
                        "10001".to_string(),
                        "20002".to_string(),
                        "30003".to_string(),
                    ]),
                ),
            ],
        },
    );

    let he_path = write_he(&dir, schema, cols);
    let csv_out = dir.path().join("out.csv");

    helium()
        .arg("convert")
        .arg(&he_path)
        .arg("-o")
        .arg(&csv_out)
        .assert()
        .success();

    let content = fs::read_to_string(&csv_out).unwrap();
    let lines: Vec<&str> = content.lines().collect();

    // Header should have dotted sub-columns, NOT a single "addr" column.
    assert!(
        lines[0].contains("addr.street"),
        "header missing 'addr.street': {content}"
    );
    assert!(
        lines[0].contains("addr.zip"),
        "header missing 'addr.zip': {content}"
    );
    assert!(
        !lines[0].contains(",addr,"),
        "header should NOT have bare 'addr' column: {content}"
    );

    // Data rows should have the actual values.
    assert!(
        content.contains("Main St"),
        "missing street value: {content}"
    );
    assert!(content.contains("10001"), "missing zip value: {content}");
}

// ---------------------------------------------------------------------------
// Test 2: Nested Struct flattens recursively
// ---------------------------------------------------------------------------

#[test]
fn nested_struct_flattens_recursively() {
    let dir = TempDir::new().unwrap();

    // Schema: user: Struct { name: Utf8, addr: Struct { street: Utf8, zip: Utf8 } }
    let schema = Schema::new(vec![ColumnSpec::struct_col(
        "user",
        vec![
            FieldSpec::utf8(
                "name",
                field_enc_utf8()[0].clone(),
                field_enc_utf8()[1].clone(),
            ),
            FieldSpec::struct_field(
                "addr",
                vec![
                    FieldSpec::utf8(
                        "street",
                        field_enc_utf8()[0].clone(),
                        field_enc_utf8()[1].clone(),
                    ),
                    FieldSpec::utf8(
                        "zip",
                        field_enc_utf8()[0].clone(),
                        field_enc_utf8()[1].clone(),
                    ),
                ],
            ),
        ],
    )]);

    let mut cols: HashMap<String, LogicalColumn> = HashMap::new();
    cols.insert(
        "user".to_string(),
        LogicalColumn::Struct {
            fields: vec![
                (
                    "name".to_string(),
                    LogicalColumn::Utf8(vec!["Alice".to_string(), "Bob".to_string()]),
                ),
                (
                    "addr".to_string(),
                    LogicalColumn::Struct {
                        fields: vec![
                            (
                                "street".to_string(),
                                LogicalColumn::Utf8(vec![
                                    "Main St".to_string(),
                                    "Oak Ave".to_string(),
                                ]),
                            ),
                            (
                                "zip".to_string(),
                                LogicalColumn::Utf8(vec!["10001".to_string(), "20002".to_string()]),
                            ),
                        ],
                    },
                ),
            ],
        },
    );

    let he_path = write_he(&dir, schema, cols);
    let csv_out = dir.path().join("out.csv");

    helium()
        .arg("convert")
        .arg(&he_path)
        .arg("-o")
        .arg(&csv_out)
        .assert()
        .success();

    let content = fs::read_to_string(&csv_out).unwrap();
    let header_line = content.lines().next().unwrap();

    assert!(
        header_line.contains("user.name"),
        "header missing 'user.name': {header_line}"
    );
    assert!(
        header_line.contains("user.addr.street"),
        "header missing 'user.addr.street': {header_line}"
    );
    assert!(
        header_line.contains("user.addr.zip"),
        "header missing 'user.addr.zip': {header_line}"
    );

    // Values should appear in the data rows.
    assert!(content.contains("Alice"), "missing 'Alice': {content}");
    assert!(content.contains("Main St"), "missing 'Main St': {content}");
    assert!(content.contains("20002"), "missing '20002': {content}");
}

// ---------------------------------------------------------------------------
// Test 3: Nullable<Struct> emits empty cells when null
// ---------------------------------------------------------------------------

#[test]
fn nullable_struct_emits_empty_cells_when_null() {
    let dir = TempDir::new().unwrap();

    // Schema: addr: Nullable<Struct { street: Utf8, zip: Utf8 }>
    let inner_struct = LogicalType::Struct {
        fields: vec![
            FieldSpec::utf8(
                "street",
                field_enc_utf8()[0].clone(),
                field_enc_utf8()[1].clone(),
            ),
            FieldSpec::utf8(
                "zip",
                field_enc_utf8()[0].clone(),
                field_enc_utf8()[1].clone(),
            ),
        ],
    };
    let nullable_enc = helium::schema::encodings::default_encodings(&LogicalType::Nullable {
        inner: Box::new(inner_struct.clone()),
    });
    let schema = Schema::new(vec![ColumnSpec::nullable(
        "addr",
        inner_struct,
        nullable_enc,
    )]);

    let mut cols: HashMap<String, LogicalColumn> = HashMap::new();
    // Row 0: present, Row 1: null, Row 2: present
    cols.insert(
        "addr".to_string(),
        LogicalColumn::Nullable {
            present: vec![true, false, true],
            value: Box::new(LogicalColumn::Struct {
                fields: vec![
                    (
                        "street".to_string(),
                        LogicalColumn::Utf8(vec!["Main St".to_string(), "Pine Rd".to_string()]),
                    ),
                    (
                        "zip".to_string(),
                        LogicalColumn::Utf8(vec!["10001".to_string(), "30003".to_string()]),
                    ),
                ],
            }),
        },
    );

    let he_path = write_he(&dir, schema, cols);
    let csv_out = dir.path().join("out.csv");

    helium()
        .arg("convert")
        .arg(&he_path)
        .arg("-o")
        .arg(&csv_out)
        .assert()
        .success();

    let content = fs::read_to_string(&csv_out).unwrap();
    // Parse the CSV to check exact cell values.
    let records: Vec<Vec<String>> = content
        .lines()
        .skip(1) // skip header
        .map(|line| {
            csv::ReaderBuilder::new()
                .has_headers(false)
                .from_reader(line.as_bytes())
                .records()
                .next()
                .unwrap()
                .unwrap()
                .iter()
                .map(|f| f.to_string())
                .collect()
        })
        .collect();

    assert_eq!(records.len(), 3, "expected 3 data rows: {content}");
    // Row 0: both sub-columns filled.
    assert_eq!(records[0][0], "Main St", "row 0 street mismatch: {content}");
    assert_eq!(records[0][1], "10001", "row 0 zip mismatch: {content}");
    // Row 1: null → both sub-columns empty.
    assert_eq!(
        records[1][0], "",
        "row 1 (null) street should be empty: {content}"
    );
    assert_eq!(
        records[1][1], "",
        "row 1 (null) zip should be empty: {content}"
    );
    // Row 2: filled again.
    assert_eq!(records[2][0], "Pine Rd", "row 2 street mismatch: {content}");
    assert_eq!(records[2][1], "30003", "row 2 zip mismatch: {content}");

    // Header should use dotted names.
    let header = content.lines().next().unwrap();
    assert!(
        header.contains("addr.street"),
        "header missing addr.street: {header}"
    );
    assert!(
        header.contains("addr.zip"),
        "header missing addr.zip: {header}"
    );
}

// ---------------------------------------------------------------------------
// Test 4: Mixed Struct + List
// ---------------------------------------------------------------------------

#[test]
fn mixed_struct_and_list() {
    let dir = TempDir::new().unwrap();

    // Schema: name: Struct { first: Utf8, last: Utf8 }, tags: List<Utf8>
    let list_enc = helium::schema::encodings::default_encodings(&LogicalType::List {
        inner: Box::new(LogicalType::Utf8),
    });
    let schema = Schema::new(vec![
        ColumnSpec::struct_col(
            "name",
            vec![
                FieldSpec::utf8(
                    "first",
                    field_enc_utf8()[0].clone(),
                    field_enc_utf8()[1].clone(),
                ),
                FieldSpec::utf8(
                    "last",
                    field_enc_utf8()[0].clone(),
                    field_enc_utf8()[1].clone(),
                ),
            ],
        ),
        ColumnSpec::list("tags", LogicalType::Utf8, list_enc),
    ]);

    let mut cols: HashMap<String, LogicalColumn> = HashMap::new();
    cols.insert(
        "name".to_string(),
        LogicalColumn::Struct {
            fields: vec![
                (
                    "first".to_string(),
                    LogicalColumn::Utf8(vec!["Alice".to_string(), "Bob".to_string()]),
                ),
                (
                    "last".to_string(),
                    LogicalColumn::Utf8(vec!["Smith".to_string(), "Jones".to_string()]),
                ),
            ],
        },
    );
    cols.insert(
        "tags".to_string(),
        LogicalColumn::List {
            offsets: vec![0, 2, 3],
            values: Box::new(LogicalColumn::Utf8(vec![
                "rust".to_string(),
                "dev".to_string(),
                "ops".to_string(),
            ])),
        },
    );

    let he_path = write_he(&dir, schema, cols);
    let csv_out = dir.path().join("out.csv");

    helium()
        .arg("convert")
        .arg(&he_path)
        .arg("-o")
        .arg(&csv_out)
        .assert()
        .success();

    let content = fs::read_to_string(&csv_out).unwrap();
    let header = content.lines().next().unwrap();

    // Struct should be flattened.
    assert!(
        header.contains("name.first"),
        "header missing name.first: {header}"
    );
    assert!(
        header.contains("name.last"),
        "header missing name.last: {header}"
    );
    // List should appear as a single column.
    assert!(header.contains("tags"), "header missing tags: {header}");
    // List column should NOT be split into dotted sub-columns.
    assert!(
        !header.contains("tags."),
        "tags should not be dotted: {header}"
    );

    // Values.
    assert!(content.contains("Alice"), "missing Alice: {content}");
    // The list cell should be JSON-stringified.
    assert!(
        content.contains("rust"),
        "list content should appear: {content}"
    );
}

// ---------------------------------------------------------------------------
// Test 5: Flat schema round-trip regression
// ---------------------------------------------------------------------------

#[test]
fn flat_schema_produces_same_csv_as_before() {
    // A schema with only Primitive + Utf8 columns should produce the same CSV as
    // the pre-flattening implementation (no regressions).
    let schema = Schema::new(vec![
        col_prim("id", helium::DataType::I64),
        ColumnSpec::new("label", LogicalType::Utf8, field_enc_utf8()),
    ]);
    let mut cols: HashMap<String, LogicalColumn> = HashMap::new();
    cols.insert(
        "id".to_string(),
        LogicalColumn::Primitive(ColumnData::I64(vec![10, 20, 30])),
    );
    cols.insert(
        "label".to_string(),
        LogicalColumn::Utf8(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
    );

    let mut out = Vec::new();
    write_csv(&schema, &cols, &mut out).unwrap();
    let csv = String::from_utf8(out).unwrap();
    let lines: Vec<&str> = csv.lines().collect();

    assert_eq!(lines[0], "id,label", "header mismatch: {}", lines[0]);
    assert_eq!(lines[1], "10,a", "row 0 mismatch: {}", lines[1]);
    assert_eq!(lines[2], "20,b", "row 1 mismatch: {}", lines[2]);
    assert_eq!(lines[3], "30,c", "row 2 mismatch: {}", lines[3]);
}

// ---------------------------------------------------------------------------
// Test 6: --csv-strict errors on List
// ---------------------------------------------------------------------------

#[test]
fn csv_strict_errors_on_list() {
    let dir = TempDir::new().unwrap();

    // Schema with a List column.
    let list_enc = helium::schema::encodings::default_encodings(&LogicalType::List {
        inner: Box::new(LogicalType::Utf8),
    });
    let schema = Schema::new(vec![
        col_prim("id", helium::DataType::I64),
        ColumnSpec::list("tags", LogicalType::Utf8, list_enc),
    ]);
    let mut cols: HashMap<String, LogicalColumn> = HashMap::new();
    cols.insert(
        "id".to_string(),
        LogicalColumn::Primitive(ColumnData::I64(vec![1])),
    );
    cols.insert(
        "tags".to_string(),
        LogicalColumn::List {
            offsets: vec![0, 1],
            values: Box::new(LogicalColumn::Utf8(vec!["rust".to_string()])),
        },
    );

    let he_path = write_he(&dir, schema, cols);
    let csv_out = dir.path().join("out.csv");

    helium()
        .arg("convert")
        .arg(&he_path)
        .arg("-o")
        .arg(&csv_out)
        .arg("--csv-strict")
        .assert()
        .failure()
        .stderr(
            predicate::str::contains("strict mode")
                .and(predicate::str::contains("tags"))
                .and(predicate::str::contains("List")),
        );
}

// ---------------------------------------------------------------------------
// Test 7: --csv-strict accepts pure-Struct schema
// ---------------------------------------------------------------------------

#[test]
fn csv_strict_accepts_pure_struct() {
    let dir = TempDir::new().unwrap();

    // Schema: addr: Struct { street: Utf8, zip: Utf8 } — no List/Map/Union.
    let schema = Schema::new(vec![ColumnSpec::struct_col(
        "addr",
        vec![
            FieldSpec::utf8(
                "street",
                field_enc_utf8()[0].clone(),
                field_enc_utf8()[1].clone(),
            ),
            FieldSpec::utf8(
                "zip",
                field_enc_utf8()[0].clone(),
                field_enc_utf8()[1].clone(),
            ),
        ],
    )]);
    let mut cols: HashMap<String, LogicalColumn> = HashMap::new();
    cols.insert(
        "addr".to_string(),
        LogicalColumn::Struct {
            fields: vec![
                (
                    "street".to_string(),
                    LogicalColumn::Utf8(vec!["Main St".to_string()]),
                ),
                (
                    "zip".to_string(),
                    LogicalColumn::Utf8(vec!["10001".to_string()]),
                ),
            ],
        },
    );

    let he_path = write_he(&dir, schema, cols);
    let csv_out = dir.path().join("out.csv");

    helium()
        .arg("convert")
        .arg(&he_path)
        .arg("-o")
        .arg(&csv_out)
        .arg("--csv-strict")
        .assert()
        .success();

    let content = fs::read_to_string(&csv_out).unwrap();
    assert!(
        content.contains("addr.street"),
        "missing addr.street: {content}"
    );
    assert!(content.contains("Main St"), "missing value: {content}");
}

// ---------------------------------------------------------------------------
// Unit tests for write_csv_with_options (library API, no CLI shell)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod unit {
    use super::*;

    /// Build a minimal Schema for a Struct column with two Utf8 fields.
    fn addr_struct_schema() -> (Schema, HashMap<String, LogicalColumn>) {
        let schema = Schema::new(vec![ColumnSpec::struct_col(
            "addr",
            vec![
                FieldSpec::utf8(
                    "street",
                    field_enc_utf8()[0].clone(),
                    field_enc_utf8()[1].clone(),
                ),
                FieldSpec::utf8(
                    "zip",
                    field_enc_utf8()[0].clone(),
                    field_enc_utf8()[1].clone(),
                ),
            ],
        )]);
        let mut cols: HashMap<String, LogicalColumn> = HashMap::new();
        cols.insert(
            "addr".to_string(),
            LogicalColumn::Struct {
                fields: vec![
                    (
                        "street".to_string(),
                        LogicalColumn::Utf8(vec!["Main St".to_string(), "Oak Ave".to_string()]),
                    ),
                    (
                        "zip".to_string(),
                        LogicalColumn::Utf8(vec!["10001".to_string(), "20002".to_string()]),
                    ),
                ],
            },
        );
        (schema, cols)
    }

    #[test]
    fn unit_struct_flatten_header() {
        let (schema, cols) = addr_struct_schema();
        let mut out = Vec::new();
        write_csv(&schema, &cols, &mut out).unwrap();
        let csv = String::from_utf8(out).unwrap();
        let header = csv.lines().next().unwrap();
        assert_eq!(header, "addr.street,addr.zip", "header: {header}");
    }

    #[test]
    fn unit_struct_flatten_values() {
        let (schema, cols) = addr_struct_schema();
        let mut out = Vec::new();
        write_csv(&schema, &cols, &mut out).unwrap();
        let csv = String::from_utf8(out).unwrap();
        let rows: Vec<&str> = csv.lines().collect();
        assert_eq!(rows.len(), 3, "expected header + 2 data rows: {csv}");
        assert_eq!(rows[1], "Main St,10001", "row 0: {}", rows[1]);
        assert_eq!(rows[2], "Oak Ave,20002", "row 1: {}", rows[2]);
    }

    #[test]
    fn unit_strict_errors_on_list() {
        let list_enc = helium::schema::encodings::default_encodings(&LogicalType::List {
            inner: Box::new(LogicalType::Utf8),
        });
        let schema = Schema::new(vec![ColumnSpec::list("tags", LogicalType::Utf8, list_enc)]);
        let mut cols: HashMap<String, LogicalColumn> = HashMap::new();
        cols.insert(
            "tags".to_string(),
            LogicalColumn::List {
                offsets: vec![0, 1],
                values: Box::new(LogicalColumn::Utf8(vec!["rust".to_string()])),
            },
        );
        let opts = CsvWriteOptions {
            strict: true,
            ..CsvWriteOptions::default()
        };
        let mut out = Vec::new();
        let result = write_csv_with_options(&schema, &cols, &mut out, &opts);
        assert!(result.is_err(), "expected error in strict mode");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("strict mode"),
            "error should mention strict mode: {msg}"
        );
        assert!(msg.contains("tags"), "error should name the column: {msg}");
        assert!(msg.contains("List"), "error should name the type: {msg}");
    }

    #[test]
    fn unit_strict_errors_on_map() {
        let map_enc = helium::schema::encodings::default_encodings(&LogicalType::Map {
            key: Box::new(LogicalType::Utf8),
            value: Box::new(LogicalType::Primitive {
                data_type: helium::DataType::I64,
            }),
        });
        let schema = Schema::new(vec![ColumnSpec::map(
            "counts",
            LogicalType::Utf8,
            LogicalType::Primitive {
                data_type: helium::DataType::I64,
            },
            map_enc,
        )]);
        let mut cols: HashMap<String, LogicalColumn> = HashMap::new();
        cols.insert(
            "counts".to_string(),
            LogicalColumn::Map {
                offsets: vec![0, 1],
                keys: Box::new(LogicalColumn::Utf8(vec!["a".to_string()])),
                values: Box::new(LogicalColumn::Primitive(ColumnData::I64(vec![1]))),
            },
        );
        let opts = CsvWriteOptions {
            strict: true,
            ..CsvWriteOptions::default()
        };
        let mut out = Vec::new();
        let result = write_csv_with_options(&schema, &cols, &mut out, &opts);
        assert!(result.is_err(), "expected error in strict mode");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("Map"), "error should name the type: {msg}");
    }

    #[test]
    fn unit_strict_struct_succeeds() {
        let (schema, cols) = addr_struct_schema();
        let opts = CsvWriteOptions {
            strict: true,
            ..CsvWriteOptions::default()
        };
        let mut out = Vec::new();
        write_csv_with_options(&schema, &cols, &mut out, &opts).unwrap();
        let csv = String::from_utf8(out).unwrap();
        assert!(csv.contains("addr.street"), "missing addr.street: {csv}");
    }

    #[test]
    fn unit_nullable_struct_empty_cells_when_null() {
        // Nullable<Struct { street, zip }> — row 1 is null.
        let inner_struct = LogicalType::Struct {
            fields: vec![
                FieldSpec::utf8(
                    "street",
                    field_enc_utf8()[0].clone(),
                    field_enc_utf8()[1].clone(),
                ),
                FieldSpec::utf8(
                    "zip",
                    field_enc_utf8()[0].clone(),
                    field_enc_utf8()[1].clone(),
                ),
            ],
        };
        let nullable_enc = helium::schema::encodings::default_encodings(&LogicalType::Nullable {
            inner: Box::new(inner_struct.clone()),
        });
        let schema = Schema::new(vec![ColumnSpec::nullable(
            "addr",
            inner_struct,
            nullable_enc,
        )]);
        let mut cols: HashMap<String, LogicalColumn> = HashMap::new();
        cols.insert(
            "addr".to_string(),
            LogicalColumn::Nullable {
                present: vec![true, false, true],
                value: Box::new(LogicalColumn::Struct {
                    fields: vec![
                        (
                            "street".to_string(),
                            LogicalColumn::Utf8(vec!["Main St".to_string(), "Pine Rd".to_string()]),
                        ),
                        (
                            "zip".to_string(),
                            LogicalColumn::Utf8(vec!["10001".to_string(), "30003".to_string()]),
                        ),
                    ],
                }),
            },
        );

        let mut out = Vec::new();
        write_csv(&schema, &cols, &mut out).unwrap();
        let csv = String::from_utf8(out).unwrap();
        let rows: Vec<Vec<String>> = csv
            .lines()
            .skip(1)
            .map(|l| {
                csv::ReaderBuilder::new()
                    .has_headers(false)
                    .from_reader(l.as_bytes())
                    .records()
                    .next()
                    .unwrap()
                    .unwrap()
                    .iter()
                    .map(|f| f.to_string())
                    .collect()
            })
            .collect();

        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0], vec!["Main St", "10001"]);
        assert_eq!(rows[1], vec!["", ""]); // null row → both empty
        assert_eq!(rows[2], vec!["Pine Rd", "30003"]);
    }
}
