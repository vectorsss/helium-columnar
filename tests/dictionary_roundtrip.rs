//! Round-trip tests for the v3 `Dictionary { inner }` logical type.
//!
//! Covers:
//! - `Dictionary { inner: Utf8 }` — the common string-dict case
//! - `Dictionary { inner: Primitive(I64) }` — primitive value dict
//! - `Nullable { inner: Dictionary { inner: Utf8 } }` — nullable wrapper around dict
//! - Nested dict inside a struct
//! - Schema JSON round-trip and wire format (`"kind":"dictionary"`)

use helium::{
    CoderRegistry, CoderSpec, ColumnSpec, LogicalColumn, LogicalType, Schema,
    core::coder::{ColumnData, DataType},
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn zstd_pipe() -> Vec<CoderSpec> {
    vec![CoderSpec::new("zstd")]
}

fn delta_leb_zstd() -> Vec<CoderSpec> {
    vec![
        CoderSpec::new("delta"),
        CoderSpec::new("leb128"),
        CoderSpec::new("zstd"),
    ]
}

fn leb_zstd() -> Vec<CoderSpec> {
    vec![CoderSpec::new("leb128"), CoderSpec::new("zstd")]
}

/// Write + read a single-stripe file and return the read-back column.
fn roundtrip_column(spec: ColumnSpec, col: LogicalColumn) -> LogicalColumn {
    use helium::{HeliumReader, HeliumWriter};
    use std::io::Cursor;

    let registry = CoderRegistry::default();
    let schema = Schema::new(vec![spec]);

    let mut buf = Vec::new();
    {
        let mut writer =
            HeliumWriter::new(Cursor::new(&mut buf), schema.clone(), &registry).expect("writer");
        writer.write_column("col", col).expect("write");
        writer.finish().expect("finish");
    }
    let mut reader = HeliumReader::new(Cursor::new(&buf), &registry).expect("reader");
    reader.read_column("col").expect("read back")
}

// ---------------------------------------------------------------------------
// Dictionary { inner: Utf8 }
// ---------------------------------------------------------------------------

#[test]
fn dictionary_utf8_roundtrip() {
    let lt = LogicalType::Dictionary {
        inner: Box::new(LogicalType::Utf8),
    };
    // Encodings: inner Utf8 needs [offsets, data], then 1 for indices.
    let encodings = vec![
        delta_leb_zstd(), // dict.offsets
        zstd_pipe(),      // dict.data
        delta_leb_zstd(), // indices
    ];
    let spec = ColumnSpec::new("col", lt, encodings);

    // 5 rows, 3 distinct values.
    let col = LogicalColumn::Dictionary {
        dictionary: Box::new(LogicalColumn::Utf8(vec![
            "apple".to_string(),
            "banana".to_string(),
            "cherry".to_string(),
        ])),
        indices: vec![0, 1, 2, 1, 0],
    };
    let result = roundtrip_column(spec, col.clone());
    assert_eq!(result, col, "Dictionary(Utf8) round-trip mismatch");
}

// ---------------------------------------------------------------------------
// Dictionary { inner: Primitive(I64) }
// ---------------------------------------------------------------------------

#[test]
fn dictionary_primitive_i64_roundtrip() {
    let lt = LogicalType::Dictionary {
        inner: Box::new(LogicalType::Primitive {
            data_type: DataType::I64,
        }),
    };
    // Encodings: inner Primitive(I64) = 1 leaf, then 1 for indices.
    let encodings = vec![
        delta_leb_zstd(), // dict.values
        delta_leb_zstd(), // indices
    ];
    let spec = ColumnSpec::new("col", lt, encodings);

    let col = LogicalColumn::Dictionary {
        dictionary: Box::new(LogicalColumn::Primitive(ColumnData::I64(vec![
            100, 200, 300,
        ]))),
        indices: vec![2, 0, 1, 0, 2],
    };
    let result = roundtrip_column(spec, col.clone());
    assert_eq!(
        result, col,
        "Dictionary(Primitive(I64)) round-trip mismatch"
    );
}

// ---------------------------------------------------------------------------
// Nullable { inner: Dictionary { inner: Utf8 } }
// ---------------------------------------------------------------------------

#[test]
fn nullable_dictionary_utf8_roundtrip() {
    let inner_dict_lt = LogicalType::Dictionary {
        inner: Box::new(LogicalType::Utf8),
    };
    let lt = LogicalType::Nullable {
        inner: Box::new(inner_dict_lt),
    };
    // Encodings:
    //   present (U8)   = 1
    //   dict.offsets   = 1
    //   dict.data      = 1
    //   indices        = 1
    let encodings = vec![
        leb_zstd(),       // present
        delta_leb_zstd(), // dict.offsets
        zstd_pipe(),      // dict.data
        delta_leb_zstd(), // indices
    ];
    let spec = ColumnSpec::new("col", lt, encodings);

    // 4 rows, 2 non-null.
    let col = LogicalColumn::Nullable {
        present: vec![true, false, true, false],
        value: Box::new(LogicalColumn::Dictionary {
            dictionary: Box::new(LogicalColumn::Utf8(vec!["x".to_string(), "y".to_string()])),
            indices: vec![1, 0],
        }),
    };
    let result = roundtrip_column(spec, col.clone());
    assert_eq!(
        result, col,
        "Nullable(Dictionary(Utf8)) round-trip mismatch"
    );
}

// ---------------------------------------------------------------------------
// Nested: Struct containing a Dictionary field
// ---------------------------------------------------------------------------

#[test]
fn struct_with_dictionary_field_roundtrip() {
    use helium::FieldSpec;

    let dict_lt = LogicalType::Dictionary {
        inner: Box::new(LogicalType::Utf8),
    };
    let dict_encodings = vec![delta_leb_zstd(), zstd_pipe(), delta_leb_zstd()];

    let struct_lt = LogicalType::Struct {
        fields: vec![
            FieldSpec::primitive("id", DataType::I32, delta_leb_zstd()),
            FieldSpec::new("label", dict_lt, dict_encodings),
        ],
    };
    let spec = ColumnSpec::new("col", struct_lt, vec![]);

    let col = LogicalColumn::Struct {
        fields: vec![
            (
                "id".to_string(),
                LogicalColumn::Primitive(ColumnData::I32(vec![1, 2, 3])),
            ),
            (
                "label".to_string(),
                LogicalColumn::Dictionary {
                    dictionary: Box::new(LogicalColumn::Utf8(vec![
                        "cat".to_string(),
                        "dog".to_string(),
                    ])),
                    indices: vec![0, 1, 0],
                },
            ),
        ],
    };
    let result = roundtrip_column(spec, col.clone());
    assert_eq!(
        result, col,
        "Struct with Dictionary field round-trip mismatch"
    );
}

// ---------------------------------------------------------------------------
// Schema JSON round-trip: "kind":"dictionary"
// ---------------------------------------------------------------------------

#[test]
fn schema_json_roundtrip_dictionary_kind() {
    let lt = LogicalType::Dictionary {
        inner: Box::new(LogicalType::Utf8),
    };
    let encodings = vec![delta_leb_zstd(), zstd_pipe(), delta_leb_zstd()];
    let schema = Schema::new(vec![ColumnSpec::new("col", lt, encodings)]);

    let json_bytes = schema.to_json().expect("serialize");
    let json_str = std::str::from_utf8(&json_bytes).expect("utf8");

    // Wire format must contain "kind":"dictionary"
    assert!(
        json_str.contains("\"kind\":\"dictionary\""),
        "schema JSON must contain '\"kind\":\"dictionary\"', got: {json_str}"
    );

    // Round-trip: deserialize and compare.
    let schema2 = Schema::from_json(&json_bytes).expect("deserialize");
    assert_eq!(schema, schema2, "schema JSON round-trip mismatch");
}

// ---------------------------------------------------------------------------
// Verify row_count and basic accessors
// ---------------------------------------------------------------------------

#[test]
fn dictionary_row_count() {
    let col = LogicalColumn::Dictionary {
        dictionary: Box::new(LogicalColumn::Utf8(vec!["a".to_string(), "b".to_string()])),
        indices: vec![0, 1, 0, 0, 1],
    };
    assert_eq!(col.row_count(), 5);
}

#[test]
fn dictionary_empty_roundtrip() {
    let lt = LogicalType::Dictionary {
        inner: Box::new(LogicalType::Primitive {
            data_type: DataType::I32,
        }),
    };
    let encodings = vec![delta_leb_zstd(), delta_leb_zstd()];
    let spec = ColumnSpec::new("col", lt, encodings);

    let col = LogicalColumn::Dictionary {
        dictionary: Box::new(LogicalColumn::Primitive(ColumnData::I32(vec![]))),
        indices: vec![],
    };
    let result = roundtrip_column(spec, col.clone());
    assert_eq!(result, col, "Empty Dictionary round-trip mismatch");
}
