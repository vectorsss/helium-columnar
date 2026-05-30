//! Round-trip tests for `LogicalType::Map` (§5.3) plus carry-over regression
//! tests from §5.2 review (recursive validation through containers).
//!
//! Tests covered:
//! - `Map<Utf8, Primitive>` basic round-trip
//! - `Map<Utf8, Struct>` (value is struct — inner struct uses FieldSpec encodings)
//! - empty maps mixed with non-empty rows
//! - duplicate keys preserved (Avro semantics)
//! - multi-stripe write/read concatenation
//! - `physical_fields()` dotted-path leaf names for Map
//! - `expected_encodings_len()` for various Map shapes
//! - `row_count()` for Map
//! - schema JSON `"kind": "map"` tag round-trip
//! - key type restriction: Struct/List/Map as key is rejected at validation
//! - validation error cases
//! - **Carry-over #1**: SAFETY-comment audit (static check via grep in comment)
//! - **Carry-over #2 regression**: `List<Struct>` with wrong inner FieldSpec
//!   encoding count must fail `Schema::validate()`, and same for `Map<K, Struct>`

use std::io::Cursor;

use helium::{
    CoderRegistry, CoderSpec, ColumnData, ColumnSpec, DataType, FieldSpec, HeliumReader,
    HeliumWriter, LogicalColumn, LogicalType, Schema,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn zstd() -> CoderSpec {
    CoderSpec::new("zstd")
}

fn delta_leb_zstd() -> Vec<CoderSpec> {
    vec![CoderSpec::new("delta"), CoderSpec::new("leb128"), zstd()]
}

fn zstd_only() -> Vec<CoderSpec> {
    vec![zstd()]
}

fn registry() -> CoderRegistry {
    CoderRegistry::default()
}

fn roundtrip(spec: ColumnSpec, data: LogicalColumn) -> LogicalColumn {
    let name = spec.name.clone();
    let schema = Schema::new(vec![spec]);
    let reg = registry();

    let mut buf = Cursor::new(Vec::<u8>::new());
    let mut writer = HeliumWriter::new(&mut buf, schema, &reg).expect("writer");
    writer.write_column(&name, data).expect("write_column");
    writer.finish().expect("finish");

    buf.set_position(0);
    let mut reader = HeliumReader::new(&mut buf, &reg).expect("reader");
    reader.read_column(&name).expect("read_column")
}

// ---------------------------------------------------------------------------
// §5.3: Map<Utf8, Primitive>
// ---------------------------------------------------------------------------

#[test]
fn map_utf8_to_primitive_roundtrip() {
    // 3 rows:
    //   row 0: {"a"→1, "b"→2}
    //   row 1: {"c"→3}
    //   row 2: {} (empty map)
    let offsets: Vec<u32> = vec![0, 2, 3, 3];
    let keys = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let values = ColumnData::I32(vec![1, 2, 3]);

    let spec = ColumnSpec::map(
        "attrs",
        LogicalType::Utf8,
        LogicalType::Primitive {
            data_type: DataType::I32,
        },
        // encodings: [offsets, key.offsets, key.data, value.values]
        vec![
            delta_leb_zstd(), // offsets
            delta_leb_zstd(), // key string offsets
            zstd_only(),      // key string data
            delta_leb_zstd(), // value primitives
        ],
    );

    let data = LogicalColumn::Map {
        offsets: offsets.clone(),
        keys: Box::new(LogicalColumn::Utf8(keys.clone())),
        values: Box::new(LogicalColumn::Primitive(values.clone())),
    };

    let result = roundtrip(spec, data);
    let LogicalColumn::Map {
        offsets: ro,
        keys: rk,
        values: rv,
    } = result
    else {
        panic!("expected Map");
    };
    assert_eq!(ro, offsets);
    assert_eq!(*rk, LogicalColumn::Utf8(keys));
    assert_eq!(*rv, LogicalColumn::Primitive(values));
}

#[test]
fn map_primitive_to_primitive_roundtrip() {
    // 2 rows: {10→100, 20→200} and {30→300}
    let offsets: Vec<u32> = vec![0, 2, 3];
    let keys = ColumnData::I32(vec![10, 20, 30]);
    let values = ColumnData::I64(vec![100, 200, 300]);

    let spec = ColumnSpec::map(
        "m",
        LogicalType::Primitive {
            data_type: DataType::I32,
        },
        LogicalType::Primitive {
            data_type: DataType::I64,
        },
        vec![
            delta_leb_zstd(), // offsets
            delta_leb_zstd(), // key.values
            delta_leb_zstd(), // value.values
        ],
    );

    let data = LogicalColumn::Map {
        offsets: offsets.clone(),
        keys: Box::new(LogicalColumn::Primitive(keys.clone())),
        values: Box::new(LogicalColumn::Primitive(values.clone())),
    };

    let result = roundtrip(spec, data);
    let LogicalColumn::Map {
        offsets: ro,
        keys: rk,
        values: rv,
    } = result
    else {
        panic!("expected Map");
    };
    assert_eq!(ro, offsets);
    assert_eq!(*rk, LogicalColumn::Primitive(keys));
    assert_eq!(*rv, LogicalColumn::Primitive(values));
}

// ---------------------------------------------------------------------------
// §5.3: Map<Utf8, Struct>
// ---------------------------------------------------------------------------

#[test]
fn map_utf8_to_struct_roundtrip() {
    // Schema: events: Map<Utf8, Struct { ts: I64, score: I32 }>
    // 2 rows: {"click"→{ts:1000,score:5}, "hover"→{ts:2000,score:3}} and {"submit"→{ts:3000,score:9}}
    let inner_struct = LogicalType::Struct {
        fields: vec![
            FieldSpec::primitive("ts", DataType::I64, delta_leb_zstd()),
            FieldSpec::primitive("score", DataType::I32, delta_leb_zstd()),
        ],
    };

    // Map<Utf8, Struct>:
    // expected_encodings_len = 1 (offsets) + 2 (Utf8 key: inner_offsets+data) + 0 (Struct value: FieldSpec) = 3
    let spec = ColumnSpec::map(
        "events",
        LogicalType::Utf8,
        inner_struct,
        vec![
            delta_leb_zstd(), // offsets
            delta_leb_zstd(), // key string offsets
            zstd_only(),      // key string data
                              // no value entries — struct value encodings live in FieldSpec
        ],
    );

    let offsets: Vec<u32> = vec![0, 2, 3];
    let key_strings = vec![
        "click".to_string(),
        "hover".to_string(),
        "submit".to_string(),
    ];
    let ts_vals = ColumnData::I64(vec![1000, 2000, 3000]);
    let score_vals = ColumnData::I32(vec![5, 3, 9]);

    let value_struct = LogicalColumn::Struct {
        fields: vec![
            ("ts".to_string(), LogicalColumn::Primitive(ts_vals.clone())),
            (
                "score".to_string(),
                LogicalColumn::Primitive(score_vals.clone()),
            ),
        ],
    };

    let data = LogicalColumn::Map {
        offsets: offsets.clone(),
        keys: Box::new(LogicalColumn::Utf8(key_strings.clone())),
        values: Box::new(value_struct),
    };

    let result = roundtrip(spec, data);
    let LogicalColumn::Map {
        offsets: ro,
        keys: rk,
        values: rv,
    } = result
    else {
        panic!("expected Map");
    };
    assert_eq!(ro, offsets);
    assert_eq!(*rk, LogicalColumn::Utf8(key_strings));

    let LogicalColumn::Struct { fields } = *rv else {
        panic!("expected Struct value");
    };
    assert_eq!(fields[0].1, LogicalColumn::Primitive(ts_vals));
    assert_eq!(fields[1].1, LogicalColumn::Primitive(score_vals));
}

// ---------------------------------------------------------------------------
// §5.3: empty maps mixed with non-empty rows
// ---------------------------------------------------------------------------

#[test]
fn map_empty_rows_mixed_with_populated() {
    // 4 rows: {} {"x"→1} {} {"y"→2,"z"→3}
    let offsets: Vec<u32> = vec![0, 0, 1, 1, 3];
    let keys = vec!["x".to_string(), "y".to_string(), "z".to_string()];
    let values = ColumnData::I32(vec![1, 2, 3]);

    let spec = ColumnSpec::map(
        "m",
        LogicalType::Utf8,
        LogicalType::Primitive {
            data_type: DataType::I32,
        },
        vec![
            delta_leb_zstd(),
            delta_leb_zstd(),
            zstd_only(),
            delta_leb_zstd(),
        ],
    );

    let data = LogicalColumn::Map {
        offsets: offsets.clone(),
        keys: Box::new(LogicalColumn::Utf8(keys.clone())),
        values: Box::new(LogicalColumn::Primitive(values.clone())),
    };

    let result = roundtrip(spec, data);
    let LogicalColumn::Map {
        offsets: ro,
        keys: rk,
        values: rv,
    } = result
    else {
        panic!("expected Map");
    };
    assert_eq!(ro, offsets);
    assert_eq!(*rk, LogicalColumn::Utf8(keys));
    assert_eq!(*rv, LogicalColumn::Primitive(values));
}

// ---------------------------------------------------------------------------
// §5.3: duplicate keys preserved (Avro semantics)
// ---------------------------------------------------------------------------

#[test]
fn map_duplicate_keys_preserved() {
    // 1 row with 3 entries, "key" repeated twice
    let offsets: Vec<u32> = vec![0, 3];
    let keys = vec!["key".to_string(), "key".to_string(), "other".to_string()];
    let values = ColumnData::I32(vec![1, 2, 3]);

    let spec = ColumnSpec::map(
        "dup",
        LogicalType::Utf8,
        LogicalType::Primitive {
            data_type: DataType::I32,
        },
        vec![
            delta_leb_zstd(),
            delta_leb_zstd(),
            zstd_only(),
            delta_leb_zstd(),
        ],
    );

    let data = LogicalColumn::Map {
        offsets: offsets.clone(),
        keys: Box::new(LogicalColumn::Utf8(keys.clone())),
        values: Box::new(LogicalColumn::Primitive(values.clone())),
    };

    let result = roundtrip(spec, data);
    let LogicalColumn::Map {
        offsets: ro,
        keys: rk,
        values: rv,
    } = result
    else {
        panic!("expected Map");
    };
    // Duplicate "key" must be preserved exactly as written
    assert_eq!(ro, offsets);
    assert_eq!(*rk, LogicalColumn::Utf8(keys));
    assert_eq!(*rv, LogicalColumn::Primitive(values));
}

// ---------------------------------------------------------------------------
// §5.3: row_count for Map
// ---------------------------------------------------------------------------

#[test]
fn map_row_count() {
    let data = LogicalColumn::Map {
        offsets: vec![0, 2, 3, 3],
        keys: Box::new(LogicalColumn::Utf8(vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
        ])),
        values: Box::new(LogicalColumn::Primitive(ColumnData::I32(vec![1, 2, 3]))),
    };
    assert_eq!(data.row_count(), 3);

    let empty = LogicalColumn::Map {
        offsets: vec![0],
        keys: Box::new(LogicalColumn::Utf8(vec![])),
        values: Box::new(LogicalColumn::Primitive(ColumnData::I32(vec![]))),
    };
    assert_eq!(empty.row_count(), 0);
}

// ---------------------------------------------------------------------------
// §5.3: physical_fields dotted-path names
// ---------------------------------------------------------------------------

#[test]
fn map_physical_fields_utf8_to_primitive() {
    let lt = LogicalType::Map {
        key: Box::new(LogicalType::Utf8),
        value: Box::new(LogicalType::Primitive {
            data_type: DataType::I32,
        }),
    };
    let pf = lt.physical_fields();
    assert_eq!(pf.len(), 4);
    assert_eq!(pf[0].role, "offsets");
    assert_eq!(pf[0].data_type, DataType::U32);
    assert_eq!(pf[1].role, "key.offsets");
    assert_eq!(pf[1].data_type, DataType::U32);
    assert_eq!(pf[2].role, "key.data");
    assert_eq!(pf[2].data_type, DataType::Bytes);
    assert_eq!(pf[3].role, "value.values");
    assert_eq!(pf[3].data_type, DataType::I32);
}

#[test]
fn map_physical_fields_primitive_to_struct() {
    // Map<I32, Struct { x: I64 }>
    let lt = LogicalType::Map {
        key: Box::new(LogicalType::Primitive {
            data_type: DataType::I32,
        }),
        value: Box::new(LogicalType::Struct {
            fields: vec![FieldSpec::primitive("x", DataType::I64, vec![])],
        }),
    };
    let pf = lt.physical_fields();
    // offsets + key.values + value.x.values
    assert_eq!(pf.len(), 3);
    assert_eq!(pf[0].role, "offsets");
    assert_eq!(pf[1].role, "key.values");
    assert_eq!(pf[2].role, "value.x.values");
}

// ---------------------------------------------------------------------------
// §5.3: expected_encodings_len for Map
// ---------------------------------------------------------------------------

#[test]
fn map_expected_encodings_len() {
    // Map<Primitive, Primitive>: 1 + 1 + 1 = 3
    let m_pp = LogicalType::Map {
        key: Box::new(LogicalType::Primitive {
            data_type: DataType::I32,
        }),
        value: Box::new(LogicalType::Primitive {
            data_type: DataType::I64,
        }),
    };
    assert_eq!(m_pp.expected_encodings_len(), 3);

    // Map<Utf8, Primitive>: 1 + 2 + 1 = 4
    let m_up = LogicalType::Map {
        key: Box::new(LogicalType::Utf8),
        value: Box::new(LogicalType::Primitive {
            data_type: DataType::I32,
        }),
    };
    assert_eq!(m_up.expected_encodings_len(), 4);

    // Map<Utf8, Struct>: 1 + 2 + 0 = 3
    let m_us = LogicalType::Map {
        key: Box::new(LogicalType::Utf8),
        value: Box::new(LogicalType::Struct { fields: vec![] }),
    };
    assert_eq!(m_us.expected_encodings_len(), 3);

    // Map<Binary, List<Primitive>>: 1 + 2 + 2 = 5
    let m_bl = LogicalType::Map {
        key: Box::new(LogicalType::Binary),
        value: Box::new(LogicalType::List {
            inner: Box::new(LogicalType::Primitive {
                data_type: DataType::I32,
            }),
        }),
    };
    assert_eq!(m_bl.expected_encodings_len(), 5);
}

// ---------------------------------------------------------------------------
// §5.3: schema JSON "kind": "map" tag round-trip
// ---------------------------------------------------------------------------

#[test]
fn map_schema_json_kind_tag() {
    let spec = ColumnSpec::map(
        "m",
        LogicalType::Utf8,
        LogicalType::Primitive {
            data_type: DataType::I32,
        },
        vec![
            delta_leb_zstd(),
            delta_leb_zstd(),
            zstd_only(),
            delta_leb_zstd(),
        ],
    );
    let schema = Schema::new(vec![spec]);
    let json = schema.to_json().expect("to_json");
    let json_str = String::from_utf8(json.clone()).expect("utf8");

    assert!(
        json_str.contains("\"kind\":\"map\""),
        "missing map kind tag: {json_str}"
    );
    assert!(
        json_str.contains("\"key\""),
        "missing key field: {json_str}"
    );
    assert!(
        json_str.contains("\"value\""),
        "missing value field: {json_str}"
    );

    let schema2 = Schema::from_json(&json).expect("from_json");
    assert_eq!(schema, schema2);
}

// ---------------------------------------------------------------------------
// §5.3: multi-stripe concat for Map
// ---------------------------------------------------------------------------

#[test]
fn map_multi_stripe_concat() {
    let spec = ColumnSpec::map(
        "m",
        LogicalType::Utf8,
        LogicalType::Primitive {
            data_type: DataType::I32,
        },
        vec![
            delta_leb_zstd(),
            delta_leb_zstd(),
            zstd_only(),
            delta_leb_zstd(),
        ],
    );
    let schema = Schema::new(vec![spec.clone()]);
    let reg = registry();

    // Stripe 1: 2 rows: {"a"→1} and {}
    let s1 = LogicalColumn::Map {
        offsets: vec![0, 1, 1],
        keys: Box::new(LogicalColumn::Utf8(vec!["a".to_string()])),
        values: Box::new(LogicalColumn::Primitive(ColumnData::I32(vec![1]))),
    };
    // Stripe 2: 1 row: {"b"→2,"c"→3}
    let s2 = LogicalColumn::Map {
        offsets: vec![0, 2],
        keys: Box::new(LogicalColumn::Utf8(vec!["b".to_string(), "c".to_string()])),
        values: Box::new(LogicalColumn::Primitive(ColumnData::I32(vec![2, 3]))),
    };

    let mut buf = Cursor::new(Vec::<u8>::new());
    let mut writer = HeliumWriter::new(&mut buf, schema, &reg).expect("writer");
    writer.write_column("m", s1).expect("write s1");
    writer.finish_stripe().expect("finish_stripe");
    writer.write_column("m", s2).expect("write s2");
    writer.finish().expect("finish");

    buf.set_position(0);
    let mut reader = HeliumReader::new(&mut buf, &reg).expect("reader");
    let result = reader.read_column("m").expect("read");

    let LogicalColumn::Map {
        offsets: ro,
        keys: rk,
        values: rv,
    } = result
    else {
        panic!("expected Map");
    };
    // 3 rows total, entries [0,1,1,3]
    assert_eq!(ro, vec![0u32, 1, 1, 3]);
    assert_eq!(
        *rk,
        LogicalColumn::Utf8(vec!["a".to_string(), "b".to_string(), "c".to_string()])
    );
    assert_eq!(
        *rv,
        LogicalColumn::Primitive(ColumnData::I32(vec![1, 2, 3]))
    );
}

// ---------------------------------------------------------------------------
// §5.3: validation — key type restriction
// ---------------------------------------------------------------------------

#[test]
fn map_rejects_struct_key() {
    let spec = ColumnSpec::map(
        "bad",
        LogicalType::Struct { fields: vec![] }, // Struct key not allowed
        LogicalType::Primitive {
            data_type: DataType::I32,
        },
        vec![zstd_only()],
    );
    let err = Schema::new(vec![spec]).validate().expect_err("should fail");
    let msg = err.to_string();
    assert!(
        msg.contains("key") || msg.contains("Primitive") || msg.contains("Utf8"),
        "unexpected error: {msg}"
    );
}

#[test]
fn map_rejects_list_key() {
    let spec = ColumnSpec::map(
        "bad",
        LogicalType::List {
            inner: Box::new(LogicalType::Primitive {
                data_type: DataType::I32,
            }),
        },
        LogicalType::Primitive {
            data_type: DataType::I32,
        },
        vec![zstd_only(), zstd_only(), zstd_only()],
    );
    let err = Schema::new(vec![spec]).validate().expect_err("should fail");
    assert!(err.to_string().contains("key") || err.to_string().contains("Primitive"));
}

#[test]
fn map_rejects_map_key() {
    let spec = ColumnSpec::map(
        "bad",
        LogicalType::Map {
            key: Box::new(LogicalType::Utf8),
            value: Box::new(LogicalType::Primitive {
                data_type: DataType::I32,
            }),
        },
        LogicalType::Primitive {
            data_type: DataType::I32,
        },
        vec![zstd_only()],
    );
    let err = Schema::new(vec![spec]).validate().expect_err("should fail");
    assert!(err.to_string().contains("key") || err.to_string().contains("Primitive"));
}

#[test]
fn map_wrong_encoding_count_fails() {
    // Map<Utf8, I32> needs 4 encoding vectors; giving 2 fails
    let spec = ColumnSpec::map(
        "bad",
        LogicalType::Utf8,
        LogicalType::Primitive {
            data_type: DataType::I32,
        },
        vec![zstd_only(), zstd_only()], // 2 instead of 4
    );
    let err = Schema::new(vec![spec]).validate().expect_err("should fail");
    assert!(
        err.to_string().contains("encoding") || err.to_string().contains("4"),
        "unexpected: {}",
        err
    );
}

// ---------------------------------------------------------------------------
// Carry-over #2 regression: List<Struct> with wrong inner FieldSpec encodings
// fails Schema::validate()
// ---------------------------------------------------------------------------

#[test]
fn list_of_struct_validate_rejects_inner_field_encoding_mismatch() {
    // List<Struct { x: I32 }> where the FieldSpec for "x" has 2 encoding
    // vectors instead of the required 1.
    let bad_field = FieldSpec {
        name: "x".to_string(),
        logical_type: LogicalType::Primitive {
            data_type: DataType::I32,
        },
        encodings: vec![zstd_only(), zstd_only()], // 2 instead of 1 — wrong
    };
    let inner_struct = LogicalType::Struct {
        fields: vec![bad_field],
    };
    // List<Struct>: expected_encodings_len = 1 (only offsets)
    let spec = ColumnSpec::list("bad", inner_struct, vec![delta_leb_zstd()]);
    let err = Schema::new(vec![spec])
        .validate()
        .expect_err("List<Struct> with malformed FieldSpec encodings should fail validation");
    let msg = err.to_string();
    assert!(
        msg.contains("encoding") || msg.contains("field") || msg.contains("1"),
        "unexpected error: {msg}"
    );
}

#[test]
fn map_to_struct_validate_rejects_inner_field_encoding_mismatch() {
    // Map<Utf8, Struct { y: Utf8 }> where the FieldSpec for "y" (Utf8)
    // has 1 encoding vector instead of the required 2.
    let bad_field = FieldSpec {
        name: "y".to_string(),
        logical_type: LogicalType::Utf8,
        encodings: vec![zstd_only()], // 1 instead of 2 — wrong
    };
    let value_struct = LogicalType::Struct {
        fields: vec![bad_field],
    };
    // Map<Utf8, Struct>: expected_encodings_len = 1 + 2 + 0 = 3
    let spec = ColumnSpec::map(
        "bad",
        LogicalType::Utf8,
        value_struct,
        vec![delta_leb_zstd(), delta_leb_zstd(), zstd_only()],
    );
    let err = Schema::new(vec![spec])
        .validate()
        .expect_err("Map<K, Struct> with malformed FieldSpec encodings should fail validation");
    let msg = err.to_string();
    assert!(
        msg.contains("encoding") || msg.contains("field") || msg.contains("2"),
        "unexpected error: {msg}"
    );
}

// ---------------------------------------------------------------------------
// Carry-over #2 regression: List<List<Struct>> deep nesting also validated
// ---------------------------------------------------------------------------

#[test]
fn list_of_list_of_struct_validates_deeply() {
    // List<List<Struct { v: I32 }>> where inner Struct field has wrong encoding count
    let bad_field = FieldSpec {
        name: "v".to_string(),
        logical_type: LogicalType::Primitive {
            data_type: DataType::I32,
        },
        encodings: vec![], // 0 instead of 1
    };
    let inner_struct = LogicalType::Struct {
        fields: vec![bad_field],
    };
    let inner_list = LogicalType::List {
        inner: Box::new(inner_struct),
    };
    // List<List<Struct>>: expected_encodings_len = 1 + (1 + 0) = 2
    let spec = ColumnSpec::list(
        "bad",
        inner_list,
        vec![delta_leb_zstd(), delta_leb_zstd()], // 2 offsets (correct count)
    );
    let err = Schema::new(vec![spec])
        .validate()
        .expect_err("List<List<Struct>> with bad inner FieldSpec should fail validation");
    let msg = err.to_string();
    assert!(
        msg.contains("encoding") || msg.contains("field") || msg.contains("v"),
        "unexpected error: {msg}"
    );
}

// ---------------------------------------------------------------------------
// Map decompose error cases
// ---------------------------------------------------------------------------

#[test]
fn map_keys_values_count_mismatch_in_decompose() {
    let lt = LogicalType::Map {
        key: Box::new(LogicalType::Utf8),
        value: Box::new(LogicalType::Primitive {
            data_type: DataType::I32,
        }),
    };
    // keys has 2 entries, values has 3 — mismatch
    let data = LogicalColumn::Map {
        offsets: vec![0, 2, 3],
        keys: Box::new(LogicalColumn::Utf8(vec!["a".to_string(), "b".to_string()])),
        values: Box::new(LogicalColumn::Primitive(ColumnData::I32(vec![1, 2, 3]))),
    };
    let err = data.decompose(&lt).expect_err("should fail");
    let msg = err.to_string();
    assert!(
        msg.contains("entries") || msg.contains("2") || msg.contains("3"),
        "unexpected error: {msg}"
    );
}

#[test]
fn map_offsets_last_mismatch_in_decompose() {
    let lt = LogicalType::Map {
        key: Box::new(LogicalType::Utf8),
        value: Box::new(LogicalType::Primitive {
            data_type: DataType::I32,
        }),
    };
    // offsets last (5) != entry_count (2)
    let data = LogicalColumn::Map {
        offsets: vec![0, 5],
        keys: Box::new(LogicalColumn::Utf8(vec!["a".to_string(), "b".to_string()])),
        values: Box::new(LogicalColumn::Primitive(ColumnData::I32(vec![1, 2]))),
    };
    let err = data.decompose(&lt).expect_err("should fail");
    let msg = err.to_string();
    assert!(
        msg.contains("offset") || msg.contains("5") || msg.contains("2"),
        "unexpected error: {msg}"
    );
}
