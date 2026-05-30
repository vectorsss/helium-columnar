//! Round-trip tests for `LogicalType::List` (§5.2).
//!
//! Tests covered:
//! - `List<Primitive>` (basic integer list)
//! - `List<Utf8>` (list of strings)
//! - `List<Struct>` (list of structs — inner struct uses FieldSpec encodings)
//! - `List<List<T>>` (doubly-nested list)
//! - empty inner (rows where every list is empty)
//! - `physical_fields()` dotted-path leaf names for List
//! - `expected_encodings_len()` for various List shapes
//! - `row_count()` for List
//! - multi-stripe write/read concatenation for List
//! - schema JSON `"kind": "list"` tag round-trip
//! - v2 back-compat: `ArrayOf` and `ArrayOfUtf8` schemas remain readable
//! - validation error cases

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
// §5.2 test: List<Primitive>
// ---------------------------------------------------------------------------

#[test]
fn list_of_primitive_roundtrip() {
    // 4 rows: [[10,20], [30], [], [40,50,60]]
    let offsets: Vec<u32> = vec![0, 2, 3, 3, 6];
    let values = ColumnData::I32(vec![10, 20, 30, 40, 50, 60]);

    let spec = ColumnSpec::list(
        "nums",
        LogicalType::Primitive {
            data_type: DataType::I32,
        },
        vec![delta_leb_zstd(), delta_leb_zstd()],
    );
    let data = LogicalColumn::List {
        offsets: offsets.clone(),
        values: Box::new(LogicalColumn::Primitive(values.clone())),
    };

    let result = roundtrip(spec, data);
    let LogicalColumn::List {
        offsets: r_offsets,
        values: r_values,
    } = result
    else {
        panic!("expected List");
    };
    assert_eq!(r_offsets, offsets);
    assert_eq!(*r_values, LogicalColumn::Primitive(values));
}

#[test]
fn list_of_primitive_all_empty_lists() {
    // 5 rows, all empty: [[], [], [], [], []]
    let offsets: Vec<u32> = vec![0, 0, 0, 0, 0, 0];
    let values = ColumnData::I64(vec![]);

    let spec = ColumnSpec::list(
        "empty_lists",
        LogicalType::Primitive {
            data_type: DataType::I64,
        },
        vec![delta_leb_zstd(), delta_leb_zstd()],
    );
    let data = LogicalColumn::List {
        offsets: offsets.clone(),
        values: Box::new(LogicalColumn::Primitive(values.clone())),
    };

    let result = roundtrip(spec, data);
    let LogicalColumn::List {
        offsets: ro,
        values: rv,
    } = result
    else {
        panic!("expected List");
    };
    assert_eq!(ro, offsets);
    assert_eq!(*rv, LogicalColumn::Primitive(values));
}

#[test]
fn list_of_primitive_zero_rows() {
    // 0 outer rows
    let offsets: Vec<u32> = vec![0];
    let values = ColumnData::I32(vec![]);

    let spec = ColumnSpec::list(
        "empty",
        LogicalType::Primitive {
            data_type: DataType::I32,
        },
        vec![delta_leb_zstd(), delta_leb_zstd()],
    );
    let data = LogicalColumn::List {
        offsets: offsets.clone(),
        values: Box::new(LogicalColumn::Primitive(values.clone())),
    };

    let result = roundtrip(spec, data);
    let LogicalColumn::List {
        offsets: ro,
        values: rv,
    } = result
    else {
        panic!("expected List");
    };
    assert_eq!(ro, offsets);
    assert_eq!(*rv, LogicalColumn::Primitive(values));
}

// ---------------------------------------------------------------------------
// §5.2 test: List<Utf8>
// ---------------------------------------------------------------------------

#[test]
fn list_of_utf8_roundtrip() {
    // 3 rows: [["hello","world"], ["foo"], ["bar","baz","qux"]]
    let outer_offsets: Vec<u32> = vec![0, 2, 3, 6];
    let strings = vec![
        "hello".to_string(),
        "world".to_string(),
        "foo".to_string(),
        "bar".to_string(),
        "baz".to_string(),
        "qux".to_string(),
    ];

    let spec = ColumnSpec::list(
        "words",
        LogicalType::Utf8,
        vec![delta_leb_zstd(), delta_leb_zstd(), zstd_only()],
    );
    let data = LogicalColumn::List {
        offsets: outer_offsets.clone(),
        values: Box::new(LogicalColumn::Utf8(strings.clone())),
    };

    let result = roundtrip(spec, data);
    let LogicalColumn::List {
        offsets: ro,
        values: rv,
    } = result
    else {
        panic!("expected List");
    };
    assert_eq!(ro, outer_offsets);
    assert_eq!(*rv, LogicalColumn::Utf8(strings));
}

#[test]
fn list_of_utf8_with_empty_strings() {
    // 2 rows: [["", "a"], ["b", ""]]
    let outer_offsets: Vec<u32> = vec![0, 2, 4];
    let strings = vec![
        "".to_string(),
        "a".to_string(),
        "b".to_string(),
        "".to_string(),
    ];

    let spec = ColumnSpec::list(
        "ws",
        LogicalType::Utf8,
        vec![delta_leb_zstd(), delta_leb_zstd(), zstd_only()],
    );
    let data = LogicalColumn::List {
        offsets: outer_offsets.clone(),
        values: Box::new(LogicalColumn::Utf8(strings.clone())),
    };

    let result = roundtrip(spec, data);
    let LogicalColumn::List {
        offsets: ro,
        values: rv,
    } = result
    else {
        panic!("expected List");
    };
    assert_eq!(ro, outer_offsets);
    assert_eq!(*rv, LogicalColumn::Utf8(strings));
}

// ---------------------------------------------------------------------------
// §5.2 test: List<Struct>
// ---------------------------------------------------------------------------

#[test]
fn list_of_struct_roundtrip() {
    // Schema: items: List<{id: I32, label: Utf8}>
    // 2 rows: [{id:1,label:"a"},{id:2,label:"b"}] and [{id:3,label:"c"}]
    let inner_struct = LogicalType::Struct {
        fields: vec![
            FieldSpec::primitive("id", DataType::I32, delta_leb_zstd()),
            FieldSpec::utf8("label", delta_leb_zstd(), zstd_only()),
        ],
    };

    // List<Struct>: encodings = [offsets_coders]  (only 1 entry; struct inner's coders in FieldSpec)
    let spec = ColumnSpec::list("items", inner_struct, vec![delta_leb_zstd()]);

    let outer_offsets: Vec<u32> = vec![0, 2, 3];
    let ids = ColumnData::I32(vec![1, 2, 3]);
    let labels = vec!["a".to_string(), "b".to_string(), "c".to_string()];

    let inner_data = LogicalColumn::Struct {
        fields: vec![
            ("id".to_string(), LogicalColumn::Primitive(ids.clone())),
            ("label".to_string(), LogicalColumn::Utf8(labels.clone())),
        ],
    };

    let data = LogicalColumn::List {
        offsets: outer_offsets.clone(),
        values: Box::new(inner_data),
    };

    let result = roundtrip(spec, data);
    let LogicalColumn::List {
        offsets: ro,
        values: rv,
    } = result
    else {
        panic!("expected List");
    };
    assert_eq!(ro, outer_offsets);

    let LogicalColumn::Struct { fields } = *rv else {
        panic!("expected inner Struct");
    };
    assert_eq!(fields.len(), 2);
    assert_eq!(fields[0].1, LogicalColumn::Primitive(ids));
    assert_eq!(fields[1].1, LogicalColumn::Utf8(labels));
}

// ---------------------------------------------------------------------------
// §5.2 test: List<List<T>>
// ---------------------------------------------------------------------------

#[test]
fn list_of_list_of_primitive_roundtrip() {
    // Outer: 2 rows
    // Row 0: [[1,2],[3]]  → inner_offsets=[0,2,3], inner_values=[1,2,3]
    // Row 1: [[4,5,6]]    → inner_offsets appended: [0,2,3,6], inner_values=[1,2,3,4,5,6]
    // outer_offsets = [0, 2, 3]  (row 0 has 2 inner lists, row 1 has 1)

    // Outer List offsets (indexes into inner List)
    let outer_offsets: Vec<u32> = vec![0, 2, 3];
    // Inner List offsets (indexes into the flat values)
    let inner_offsets: Vec<u32> = vec![0, 2, 3, 6];
    let flat_values = ColumnData::I32(vec![1, 2, 3, 4, 5, 6]);

    // Schema: List<List<Primitive(I32)>>
    // Outer: encodings = [outer_offsets_coders, inner_offsets_coders, values_coders]
    let spec = ColumnSpec::list(
        "nested",
        LogicalType::List {
            inner: Box::new(LogicalType::Primitive {
                data_type: DataType::I32,
            }),
        },
        vec![
            delta_leb_zstd(), // outer offsets
            delta_leb_zstd(), // inner offsets
            delta_leb_zstd(), // flat values
        ],
    );

    let inner_data = LogicalColumn::List {
        offsets: inner_offsets.clone(),
        values: Box::new(LogicalColumn::Primitive(flat_values.clone())),
    };
    let data = LogicalColumn::List {
        offsets: outer_offsets.clone(),
        values: Box::new(inner_data),
    };

    let result = roundtrip(spec, data);
    let LogicalColumn::List {
        offsets: ro,
        values: rv,
    } = result
    else {
        panic!("expected outer List");
    };
    assert_eq!(ro, outer_offsets);

    let LogicalColumn::List {
        offsets: ri,
        values: rvv,
    } = *rv
    else {
        panic!("expected inner List");
    };
    assert_eq!(ri, inner_offsets);
    assert_eq!(*rvv, LogicalColumn::Primitive(flat_values));
}

// ---------------------------------------------------------------------------
// §5.2 test: physical_fields() for List
// ---------------------------------------------------------------------------

#[test]
fn list_physical_fields_primitive_inner() {
    let lt = LogicalType::List {
        inner: Box::new(LogicalType::Primitive {
            data_type: DataType::I32,
        }),
    };
    let pf = lt.physical_fields();
    assert_eq!(pf.len(), 2);
    assert_eq!(pf[0].role, "offsets");
    assert_eq!(pf[0].data_type, DataType::U32);
    assert_eq!(pf[1].role, "item.values");
    assert_eq!(pf[1].data_type, DataType::I32);
}

#[test]
fn list_physical_fields_utf8_inner() {
    let lt = LogicalType::List {
        inner: Box::new(LogicalType::Utf8),
    };
    let pf = lt.physical_fields();
    assert_eq!(pf.len(), 3);
    assert_eq!(pf[0].role, "offsets");
    assert_eq!(pf[1].role, "item.offsets");
    assert_eq!(pf[2].role, "item.data");
}

#[test]
fn list_physical_fields_struct_inner() {
    // List<Struct { a: I32, b: Utf8 }>
    let inner = LogicalType::Struct {
        fields: vec![
            FieldSpec::primitive("a", DataType::I32, vec![]),
            FieldSpec::utf8("b", vec![], vec![]),
        ],
    };
    let lt = LogicalType::List {
        inner: Box::new(inner),
    };
    let pf = lt.physical_fields();
    // offsets + inner struct leaves (a.values, b.offsets, b.data)
    assert_eq!(pf.len(), 4);
    assert_eq!(pf[0].role, "offsets");
    assert_eq!(pf[1].role, "item.a.values");
    assert_eq!(pf[2].role, "item.b.offsets");
    assert_eq!(pf[3].role, "item.b.data");
}

#[test]
fn list_physical_fields_list_inner() {
    // List<List<Primitive(I32)>>
    let inner = LogicalType::List {
        inner: Box::new(LogicalType::Primitive {
            data_type: DataType::I32,
        }),
    };
    let lt = LogicalType::List {
        inner: Box::new(inner),
    };
    let pf = lt.physical_fields();
    // outer offsets, item.offsets, item.item.values
    assert_eq!(pf.len(), 3);
    assert_eq!(pf[0].role, "offsets");
    assert_eq!(pf[1].role, "item.offsets");
    assert_eq!(pf[2].role, "item.item.values");
}

// ---------------------------------------------------------------------------
// §5.2 test: expected_encodings_len()
// ---------------------------------------------------------------------------

#[test]
fn expected_encodings_len_list_variants() {
    // List<Primitive>: 1 (offsets) + 1 (values) = 2
    let l_prim = LogicalType::List {
        inner: Box::new(LogicalType::Primitive {
            data_type: DataType::I32,
        }),
    };
    assert_eq!(l_prim.expected_encodings_len(), 2);

    // List<Utf8>: 1 (offsets) + 2 (inner_offsets + data) = 3
    let l_utf8 = LogicalType::List {
        inner: Box::new(LogicalType::Utf8),
    };
    assert_eq!(l_utf8.expected_encodings_len(), 3);

    // List<Struct>: 1 (offsets) + 0 (struct uses FieldSpec) = 1
    let l_struct = LogicalType::List {
        inner: Box::new(LogicalType::Struct { fields: vec![] }),
    };
    assert_eq!(l_struct.expected_encodings_len(), 1);

    // List<List<Primitive>>: 1 + 2 = 3
    let l_l_prim = LogicalType::List {
        inner: Box::new(LogicalType::List {
            inner: Box::new(LogicalType::Primitive {
                data_type: DataType::I32,
            }),
        }),
    };
    assert_eq!(l_l_prim.expected_encodings_len(), 3);
}

// ---------------------------------------------------------------------------
// §5.2 test: row_count() for List
// ---------------------------------------------------------------------------

#[test]
fn list_row_count() {
    let data = LogicalColumn::List {
        offsets: vec![0, 2, 3, 6],
        values: Box::new(LogicalColumn::Primitive(ColumnData::I32(vec![
            1, 2, 3, 4, 5, 6,
        ]))),
    };
    assert_eq!(data.row_count(), 3);

    let empty = LogicalColumn::List {
        offsets: vec![0],
        values: Box::new(LogicalColumn::Primitive(ColumnData::I32(vec![]))),
    };
    assert_eq!(empty.row_count(), 0);
}

// ---------------------------------------------------------------------------
// §5.2 test: schema JSON "kind": "list" tag round-trip
// ---------------------------------------------------------------------------

#[test]
fn list_schema_json_kind_tag() {
    let spec = ColumnSpec::list(
        "items",
        LogicalType::Primitive {
            data_type: DataType::I32,
        },
        vec![delta_leb_zstd(), delta_leb_zstd()],
    );
    let schema = Schema::new(vec![spec]);
    let json = schema.to_json().expect("to_json");
    let json_str = String::from_utf8(json.clone()).expect("utf8");

    assert!(
        json_str.contains("\"kind\":\"list\""),
        "missing list kind tag: {json_str}"
    );

    let schema2 = Schema::from_json(&json).expect("from_json");
    assert_eq!(schema, schema2);
}

#[test]
fn nested_list_schema_json_round_trip() {
    // List<List<Primitive>>
    let spec = ColumnSpec::list(
        "nested",
        LogicalType::List {
            inner: Box::new(LogicalType::Primitive {
                data_type: DataType::I64,
            }),
        },
        vec![delta_leb_zstd(), delta_leb_zstd(), delta_leb_zstd()],
    );
    let schema = Schema::new(vec![spec]);
    let json = schema.to_json().expect("to_json");
    let json_str = String::from_utf8(json.clone()).expect("utf8");

    // Both outer and inner list should have "kind": "list"
    let count = json_str.matches("\"kind\":\"list\"").count();
    assert_eq!(
        count, 2,
        "expected 2 list kind tags, got {count}: {json_str}"
    );

    let schema2 = Schema::from_json(&json).expect("from_json");
    assert_eq!(schema, schema2);
}

// ---------------------------------------------------------------------------
// §5.2 test: multi-stripe write/read concatenation for List
// ---------------------------------------------------------------------------

#[test]
fn list_multi_stripe_concat() {
    let spec = ColumnSpec::list(
        "nums",
        LogicalType::Primitive {
            data_type: DataType::I32,
        },
        vec![delta_leb_zstd(), delta_leb_zstd()],
    );
    let schema = Schema::new(vec![spec.clone()]);
    let reg = registry();

    // Stripe 1: 2 rows [[1,2],[3]]
    let s1 = LogicalColumn::List {
        offsets: vec![0, 2, 3],
        values: Box::new(LogicalColumn::Primitive(ColumnData::I32(vec![1, 2, 3]))),
    };
    // Stripe 2: 1 row [[4,5,6]]
    let s2 = LogicalColumn::List {
        offsets: vec![0, 3],
        values: Box::new(LogicalColumn::Primitive(ColumnData::I32(vec![4, 5, 6]))),
    };

    let mut buf = Cursor::new(Vec::<u8>::new());
    let mut writer = HeliumWriter::new(&mut buf, schema, &reg).expect("writer");
    writer.write_column("nums", s1).expect("write s1");
    writer.finish_stripe().expect("finish_stripe");
    writer.write_column("nums", s2).expect("write s2");
    writer.finish().expect("finish");

    buf.set_position(0);
    let mut reader = HeliumReader::new(&mut buf, &reg).expect("reader");
    assert_eq!(reader.stripe_count(), 2);

    let result = reader.read_column("nums").expect("read_column");
    let LogicalColumn::List {
        offsets: ro,
        values: rv,
    } = result
    else {
        panic!("expected List");
    };
    // Concatenated: 3 rows total [[1,2],[3],[4,5,6]]
    assert_eq!(ro, vec![0u32, 2, 3, 6]);
    assert_eq!(
        *rv,
        LogicalColumn::Primitive(ColumnData::I32(vec![1, 2, 3, 4, 5, 6]))
    );
}

// ---------------------------------------------------------------------------
// §5.2 back-compat test: v2 ArrayOf remains readable
// ---------------------------------------------------------------------------

#[test]
fn v2_array_of_remains_readable() {
    // Write using v2 ArrayOf schema
    let spec = ColumnSpec::array_of("tags", DataType::I32, delta_leb_zstd(), delta_leb_zstd());
    let schema = Schema::new(vec![spec.clone()]);
    let reg = registry();

    // 3 rows: [[10,20],[30],[40,50]]
    let offsets: Vec<u32> = vec![0, 2, 3, 5];
    let values = ColumnData::I32(vec![10, 20, 30, 40, 50]);

    let data = LogicalColumn::ArrayOf {
        offsets: offsets.clone(),
        values: values.clone(),
    };

    let mut buf = Cursor::new(Vec::<u8>::new());
    let mut writer = HeliumWriter::new(&mut buf, schema, &reg).expect("writer");
    writer.write_column("tags", data).expect("write");
    writer.finish().expect("finish");

    // Read back — must still work with v3 reader
    buf.set_position(0);
    let mut reader = HeliumReader::new(&mut buf, &reg).expect("reader");
    let result = reader.read_column("tags").expect("read");
    let LogicalColumn::ArrayOf {
        offsets: ro,
        values: rv,
    } = result
    else {
        panic!("expected ArrayOf, got something else");
    };
    assert_eq!(ro, offsets);
    assert_eq!(rv, values);
}

#[test]
fn v2_array_of_utf8_remains_readable() {
    // Write using v2 ArrayOfUtf8 schema
    let spec = ColumnSpec::array_of_utf8("names", delta_leb_zstd(), delta_leb_zstd(), zstd_only());
    let schema = Schema::new(vec![spec]);
    let reg = registry();

    // 2 rows: [["hello","world"],["foo"]]
    let offsets: Vec<u32> = vec![0, 2, 3];
    let strings = vec!["hello".to_string(), "world".to_string(), "foo".to_string()];

    let data = LogicalColumn::ArrayOfUtf8 {
        offsets: offsets.clone(),
        strings: strings.clone(),
    };

    let mut buf = Cursor::new(Vec::<u8>::new());
    let mut writer = HeliumWriter::new(&mut buf, schema, &reg).expect("writer");
    writer.write_column("names", data).expect("write");
    writer.finish().expect("finish");

    buf.set_position(0);
    let mut reader = HeliumReader::new(&mut buf, &reg).expect("reader");
    let result = reader.read_column("names").expect("read");
    let LogicalColumn::ArrayOfUtf8 {
        offsets: ro,
        strings: rs,
    } = result
    else {
        panic!("expected ArrayOfUtf8");
    };
    assert_eq!(ro, offsets);
    assert_eq!(rs, strings);
}

#[test]
fn v2_array_of_schema_json_still_deserializes() {
    // Confirm that a v2 ArrayOf schema JSON is still parseable by the v3 reader.
    let json = r#"{"version":1,"columns":[{"name":"x","logical_type":{"kind":"array_of","data_type":"i32"},"encodings":[[{"id":"zstd"}],[{"id":"zstd"}]]}]}"#;
    let schema = Schema::from_json(json.as_bytes()).expect("should parse v2 ArrayOf schema");
    assert!(matches!(
        schema.columns[0].logical_type,
        LogicalType::ArrayOf {
            data_type: DataType::I32
        }
    ));
}

// ---------------------------------------------------------------------------
// Validation error tests
// ---------------------------------------------------------------------------

#[test]
fn list_validate_rejects_wrong_encoding_count() {
    // List<Primitive(I32)> needs 2 encoding vectors; providing 3 fails.
    let spec = ColumnSpec::list(
        "bad",
        LogicalType::Primitive {
            data_type: DataType::I32,
        },
        vec![zstd_only(), zstd_only(), zstd_only()], // 3 instead of 2
    );
    let schema = Schema::new(vec![spec]);
    let err = schema.validate().expect_err("should fail");
    let msg = err.to_string();
    assert!(
        msg.contains("encoding") || msg.contains("2") || msg.contains("3"),
        "unexpected error: {msg}"
    );
}

#[test]
fn list_validate_rejects_zero_encodings() {
    // List requires at least 1 (offsets). 0 must fail.
    let spec = ColumnSpec::list(
        "bad",
        LogicalType::Primitive {
            data_type: DataType::I32,
        },
        vec![],
    );
    let schema = Schema::new(vec![spec]);
    schema.validate().expect_err("should fail with 0 encodings");
}

#[test]
fn list_of_struct_expects_only_offsets_encoding() {
    // List<Struct>: expected_encodings_len == 1. Providing 2 should fail.
    let inner_struct = LogicalType::Struct {
        fields: vec![FieldSpec::primitive("x", DataType::I32, delta_leb_zstd())],
    };
    let spec = ColumnSpec::list(
        "bad",
        inner_struct,
        vec![zstd_only(), zstd_only()], // 2 instead of 1
    );
    let schema = Schema::new(vec![spec]);
    let err = schema.validate().expect_err("should fail");
    let msg = err.to_string();
    assert!(
        msg.contains("encoding") || msg.contains("1") || msg.contains("2"),
        "unexpected error: {msg}"
    );
}

#[test]
fn list_offsets_non_monotonic_rejected_in_decompose() {
    // offsets must be non-decreasing
    let lt = LogicalType::List {
        inner: Box::new(LogicalType::Primitive {
            data_type: DataType::I32,
        }),
    };
    let data = LogicalColumn::List {
        offsets: vec![0, 3, 1], // non-monotonic
        values: Box::new(LogicalColumn::Primitive(ColumnData::I32(vec![1, 2, 3]))),
    };
    let err = data.decompose(&lt).expect_err("should fail");
    let msg = err.to_string();
    assert!(
        msg.contains("non-decreasing") || msg.contains("offset"),
        "unexpected error: {msg}"
    );
}

#[test]
fn list_offsets_last_mismatch_rejected_in_decompose() {
    // last offset (5) != values.row_count() (3)
    let lt = LogicalType::List {
        inner: Box::new(LogicalType::Primitive {
            data_type: DataType::I32,
        }),
    };
    let data = LogicalColumn::List {
        offsets: vec![0, 2, 5], // last offset=5 but only 3 values
        values: Box::new(LogicalColumn::Primitive(ColumnData::I32(vec![1, 2, 3]))),
    };
    let err = data.decompose(&lt).expect_err("should fail");
    let msg = err.to_string();
    assert!(
        msg.contains("last offset") || msg.contains("5") || msg.contains("3"),
        "unexpected error: {msg}"
    );
}
