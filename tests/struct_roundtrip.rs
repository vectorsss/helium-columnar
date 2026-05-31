//! Round-trip tests for `LogicalType::Struct`.
//!
//! Tests covered:
//! - flat struct (all primitive fields)
//! - 2-level nested struct
//! - struct with mixed primitive + utf8 fields
//! - empty field list (schema JSON + single-stripe file with 0 rows)
//! - multi-stripe write/read concatenation
//! - schema JSON round-trip preservation with `serde(tag = "kind")`
//! - `physical_fields()` dotted-path leaf names
//! - validation: duplicate field names, wrong encodings count, name mismatch

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

/// Write a schema with a single Struct column and a single stripe, then read
/// back and return the `LogicalColumn::Struct`.
fn roundtrip_struct(spec: ColumnSpec, data: LogicalColumn) -> LogicalColumn {
    let schema = Schema::new(vec![spec.clone()]);
    let reg = registry();

    let mut buf = Cursor::new(Vec::<u8>::new());
    let mut writer =
        HeliumWriter::new(&mut buf, schema.clone(), &reg).expect("writer construction");
    writer.write_column(&spec.name, data).expect("write_column");
    writer.finish().expect("finish");

    buf.set_position(0);
    let mut reader = HeliumReader::new(&mut buf, &reg).expect("reader construction");
    reader.read_column(&spec.name).expect("read_column")
}

// ---------------------------------------------------------------------------
// flat struct (all primitive fields)
// ---------------------------------------------------------------------------

#[test]
fn flat_struct_roundtrip() {
    let spec = ColumnSpec::struct_col(
        "point",
        vec![
            FieldSpec::primitive("x", DataType::I32, delta_leb_zstd()),
            FieldSpec::primitive("y", DataType::I32, delta_leb_zstd()),
            FieldSpec::primitive("z", DataType::I32, delta_leb_zstd()),
        ],
    );

    let n = 200usize;
    let x: Vec<i32> = (0..n as i32).collect();
    let y: Vec<i32> = (0..n as i32).map(|i| i * 2).collect();
    let z: Vec<i32> = (0..n as i32).map(|i| -i).collect();

    let data = LogicalColumn::Struct {
        fields: vec![
            (
                "x".to_string(),
                LogicalColumn::Primitive(ColumnData::I32(x.clone())),
            ),
            (
                "y".to_string(),
                LogicalColumn::Primitive(ColumnData::I32(y.clone())),
            ),
            (
                "z".to_string(),
                LogicalColumn::Primitive(ColumnData::I32(z.clone())),
            ),
        ],
    };

    let result = roundtrip_struct(spec, data);
    let LogicalColumn::Struct { fields } = result else {
        panic!("expected Struct, got something else");
    };
    assert_eq!(fields.len(), 3);
    assert_eq!(fields[0].0, "x");
    assert_eq!(fields[1].0, "y");
    assert_eq!(fields[2].0, "z");
    assert_eq!(fields[0].1, LogicalColumn::Primitive(ColumnData::I32(x)));
    assert_eq!(fields[1].1, LogicalColumn::Primitive(ColumnData::I32(y)));
    assert_eq!(fields[2].1, LogicalColumn::Primitive(ColumnData::I32(z)));
}

// ---------------------------------------------------------------------------
// 2-level nested struct
// ---------------------------------------------------------------------------

#[test]
fn nested_struct_two_levels_roundtrip() {
    // Schema: event { ts: I64, location { lat: I32, lon: I32 } }
    let spec = ColumnSpec::struct_col(
        "event",
        vec![
            FieldSpec::primitive("ts", DataType::I64, delta_leb_zstd()),
            FieldSpec::struct_field(
                "location",
                vec![
                    FieldSpec::primitive("lat", DataType::I32, delta_leb_zstd()),
                    FieldSpec::primitive("lon", DataType::I32, delta_leb_zstd()),
                ],
            ),
        ],
    );

    let n = 100usize;
    let ts: Vec<i64> = (1_700_000_000i64..).take(n).collect();
    let lat: Vec<i32> = (0..n as i32).collect();
    let lon: Vec<i32> = (0..n as i32).map(|i| i + 1000).collect();

    let data = LogicalColumn::Struct {
        fields: vec![
            (
                "ts".to_string(),
                LogicalColumn::Primitive(ColumnData::I64(ts.clone())),
            ),
            (
                "location".to_string(),
                LogicalColumn::Struct {
                    fields: vec![
                        (
                            "lat".to_string(),
                            LogicalColumn::Primitive(ColumnData::I32(lat.clone())),
                        ),
                        (
                            "lon".to_string(),
                            LogicalColumn::Primitive(ColumnData::I32(lon.clone())),
                        ),
                    ],
                },
            ),
        ],
    };

    let result = roundtrip_struct(spec, data);
    let LogicalColumn::Struct { fields: top } = result else {
        panic!("expected Struct");
    };
    assert_eq!(top.len(), 2);
    assert_eq!(top[0].0, "ts");
    assert_eq!(top[0].1, LogicalColumn::Primitive(ColumnData::I64(ts)));

    let LogicalColumn::Struct { fields: loc } = &top[1].1 else {
        panic!("expected nested Struct for location");
    };
    assert_eq!(top[1].0, "location");
    assert_eq!(loc.len(), 2);
    assert_eq!(loc[0].0, "lat");
    assert_eq!(loc[0].1, LogicalColumn::Primitive(ColumnData::I32(lat)));
    assert_eq!(loc[1].0, "lon");
    assert_eq!(loc[1].1, LogicalColumn::Primitive(ColumnData::I32(lon)));
}

// ---------------------------------------------------------------------------
// struct of mixed primitive / utf8
// ---------------------------------------------------------------------------

#[test]
fn struct_mixed_primitive_utf8_roundtrip() {
    let spec = ColumnSpec::struct_col(
        "user",
        vec![
            FieldSpec::primitive("id", DataType::I64, delta_leb_zstd()),
            FieldSpec::utf8("name", delta_leb_zstd(), zstd_only()),
        ],
    );

    let n = 150usize;
    let ids: Vec<i64> = (1..=n as i64).collect();
    let names: Vec<String> = (0..n).map(|i| format!("user_{i}")).collect();

    let data = LogicalColumn::Struct {
        fields: vec![
            (
                "id".to_string(),
                LogicalColumn::Primitive(ColumnData::I64(ids.clone())),
            ),
            ("name".to_string(), LogicalColumn::Utf8(names.clone())),
        ],
    };

    let result = roundtrip_struct(spec, data);
    let LogicalColumn::Struct { fields } = result else {
        panic!("expected Struct");
    };
    assert_eq!(fields.len(), 2);
    assert_eq!(fields[0].0, "id");
    assert_eq!(fields[0].1, LogicalColumn::Primitive(ColumnData::I64(ids)));
    assert_eq!(fields[1].0, "name");
    assert_eq!(fields[1].1, LogicalColumn::Utf8(names));
}

// ---------------------------------------------------------------------------
// empty field list
// ---------------------------------------------------------------------------

#[test]
fn empty_struct_schema_json_roundtrip() {
    // An empty struct is a valid schema type.
    let spec = ColumnSpec::struct_col("empty", vec![]);
    let schema = Schema::new(vec![spec]);

    // JSON round-trip
    let json = schema.to_json().expect("to_json");
    let schema2 = Schema::from_json(&json).expect("from_json");
    assert_eq!(schema, schema2);

    // Verify it serializes the kind tag correctly
    let json_str = String::from_utf8(json).expect("utf8");
    assert!(
        json_str.contains("\"kind\":\"struct\""),
        "missing struct kind tag"
    );
    assert!(json_str.contains("\"fields\":[]"), "missing empty fields");
}

#[test]
fn empty_struct_file_roundtrip_zero_rows() {
    let spec = ColumnSpec::struct_col("empty", vec![]);
    let schema = Schema::new(vec![spec.clone()]);
    let reg = registry();

    let mut buf = Cursor::new(Vec::<u8>::new());
    let mut writer = HeliumWriter::new(&mut buf, schema.clone(), &reg).expect("writer");
    writer
        .write_column("empty", LogicalColumn::Struct { fields: vec![] })
        .expect("write");
    writer.finish().expect("finish");

    buf.set_position(0);
    let mut reader = HeliumReader::new(&mut buf, &reg).expect("reader");
    let result = reader.read_column("empty").expect("read");
    let LogicalColumn::Struct { fields } = result else {
        panic!("expected Struct");
    };
    assert!(fields.is_empty(), "empty struct should have no fields");
}

// ---------------------------------------------------------------------------
// schema JSON round-trip with serde(tag = "kind")
// ---------------------------------------------------------------------------

#[test]
fn struct_schema_json_tag_preserved() {
    let spec = ColumnSpec::struct_col(
        "rec",
        vec![
            FieldSpec::primitive("a", DataType::I32, vec![zstd()]),
            FieldSpec::utf8("b", vec![zstd()], vec![zstd()]),
        ],
    );
    let schema = Schema::new(vec![spec]);

    let json = schema.to_json().expect("to_json");
    let json_str = String::from_utf8(json.clone()).expect("utf8");

    // The serde(tag = "kind") serializes as "kind": "struct"
    assert!(
        json_str.contains("\"kind\":\"struct\""),
        "struct kind tag not found in JSON: {json_str}"
    );

    // Round-trip must produce identical schema
    let schema2 = Schema::from_json(&json).expect("from_json");
    assert_eq!(schema, schema2);
}

#[test]
fn nested_struct_schema_json_tag_preserved() {
    let spec = ColumnSpec::struct_col(
        "outer",
        vec![FieldSpec::struct_field(
            "inner",
            vec![FieldSpec::primitive("v", DataType::I64, vec![zstd()])],
        )],
    );
    let schema = Schema::new(vec![spec]);
    let json = schema.to_json().expect("to_json");

    // Both outer and inner struct must have "kind": "struct"
    let json_str = String::from_utf8(json.clone()).expect("utf8");
    let struct_count = json_str.matches("\"kind\":\"struct\"").count();
    assert_eq!(
        struct_count, 2,
        "expected 2 struct kind tags, got {struct_count}"
    );

    let schema2 = Schema::from_json(&json).expect("from_json");
    assert_eq!(schema, schema2);
}

// ---------------------------------------------------------------------------
// physical_fields() dotted-path leaf names
// ---------------------------------------------------------------------------

#[test]
fn physical_fields_dotted_paths_flat_struct() {
    let lt = LogicalType::Struct {
        fields: vec![
            FieldSpec::primitive("id", DataType::I32, vec![]),
            FieldSpec::utf8("name", vec![], vec![]),
        ],
    };
    let pf = lt.physical_fields();
    assert_eq!(pf.len(), 3);
    assert_eq!(pf[0].role, "id.values");
    assert_eq!(pf[0].data_type, DataType::I32);
    assert_eq!(pf[1].role, "name.offsets");
    assert_eq!(pf[1].data_type, DataType::U32);
    assert_eq!(pf[2].role, "name.data");
    assert_eq!(pf[2].data_type, DataType::Bytes);
}

#[test]
fn physical_fields_dotted_paths_nested_struct() {
    // outer { a: I64, inner { b: I32, c: Utf8 } }
    let lt = LogicalType::Struct {
        fields: vec![
            FieldSpec::primitive("a", DataType::I64, vec![]),
            FieldSpec::struct_field(
                "inner",
                vec![
                    FieldSpec::primitive("b", DataType::I32, vec![]),
                    FieldSpec::utf8("c", vec![], vec![]),
                ],
            ),
        ],
    };
    let pf = lt.physical_fields();
    assert_eq!(pf.len(), 4, "expected 4 leaves, got {}", pf.len());
    assert_eq!(pf[0].role, "a.values");
    assert_eq!(pf[1].role, "inner.b.values");
    assert_eq!(pf[2].role, "inner.c.offsets");
    assert_eq!(pf[3].role, "inner.c.data");
}

#[test]
fn physical_fields_empty_struct() {
    let lt = LogicalType::Struct { fields: vec![] };
    assert!(lt.physical_fields().is_empty());
}

// ---------------------------------------------------------------------------
// multi-stripe write/read concatenation
// ---------------------------------------------------------------------------

#[test]
fn struct_multi_stripe_roundtrip() {
    let spec = ColumnSpec::struct_col(
        "rec",
        vec![
            FieldSpec::primitive("id", DataType::I32, delta_leb_zstd()),
            FieldSpec::utf8("label", delta_leb_zstd(), zstd_only()),
        ],
    );
    let schema = Schema::new(vec![spec.clone()]);
    let reg = registry();

    let stripe1_ids: Vec<i32> = (0..50).collect();
    let stripe1_labels: Vec<String> = (0..50usize).map(|i| format!("a_{i}")).collect();
    let stripe2_ids: Vec<i32> = (50..100).collect();
    let stripe2_labels: Vec<String> = (50..100usize).map(|i| format!("b_{i}")).collect();

    let make_data = |ids: Vec<i32>, labels: Vec<String>| LogicalColumn::Struct {
        fields: vec![
            (
                "id".to_string(),
                LogicalColumn::Primitive(ColumnData::I32(ids)),
            ),
            ("label".to_string(), LogicalColumn::Utf8(labels)),
        ],
    };

    let mut buf = Cursor::new(Vec::<u8>::new());
    let mut writer = HeliumWriter::new(&mut buf, schema, &reg).expect("writer");
    writer
        .write_column(
            "rec",
            make_data(stripe1_ids.clone(), stripe1_labels.clone()),
        )
        .expect("write stripe1");
    writer.finish_stripe().expect("finish_stripe");
    writer
        .write_column(
            "rec",
            make_data(stripe2_ids.clone(), stripe2_labels.clone()),
        )
        .expect("write stripe2");
    writer.finish().expect("finish");

    buf.set_position(0);
    let mut reader = HeliumReader::new(&mut buf, &reg).expect("reader");
    assert_eq!(reader.stripe_count(), 2);

    // read_column concatenates stripes
    let result = reader.read_column("rec").expect("read_column");
    let LogicalColumn::Struct { fields } = result else {
        panic!("expected Struct");
    };
    assert_eq!(fields.len(), 2);

    let expected_ids: Vec<i32> = (0..100).collect();
    let expected_labels: Vec<String> = (0..50usize)
        .map(|i| format!("a_{i}"))
        .chain((50..100usize).map(|i| format!("b_{i}")))
        .collect();
    assert_eq!(
        fields[0].1,
        LogicalColumn::Primitive(ColumnData::I32(expected_ids))
    );
    assert_eq!(fields[1].1, LogicalColumn::Utf8(expected_labels));
}

// ---------------------------------------------------------------------------
// Validation error tests
// ---------------------------------------------------------------------------

#[test]
fn struct_validate_rejects_nonempty_top_level_encodings() {
    let spec = ColumnSpec {
        name: "bad".to_string(),
        logical_type: LogicalType::Struct {
            fields: vec![FieldSpec::primitive("x", DataType::I32, vec![zstd()])],
        },
        // Top-level encodings must be empty for Struct
        encodings: vec![vec![zstd()]],
    };
    let schema = Schema::new(vec![spec]);
    let err = schema.validate().expect_err("should fail validation");
    let msg = err.to_string();
    assert!(
        msg.contains("empty top-level encodings") || msg.contains("Struct"),
        "unexpected error: {msg}"
    );
}

#[test]
fn struct_validate_rejects_duplicate_field_names() {
    let spec = ColumnSpec::struct_col(
        "bad",
        vec![
            FieldSpec::primitive("x", DataType::I32, vec![zstd()]),
            FieldSpec::primitive("x", DataType::I64, delta_leb_zstd()),
        ],
    );
    let schema = Schema::new(vec![spec]);
    let err = schema.validate().expect_err("should fail validation");
    let msg = err.to_string();
    assert!(
        msg.contains("duplicate") || msg.contains("field name"),
        "unexpected error: {msg}"
    );
}

#[test]
fn struct_validate_rejects_wrong_encodings_count_for_field() {
    // A primitive field needs 1 encoding vector; providing 2 should fail.
    let bad_field = FieldSpec {
        name: "x".to_string(),
        logical_type: LogicalType::Primitive {
            data_type: DataType::I32,
        },
        encodings: vec![vec![zstd()], vec![zstd()]], // 2 instead of 1
    };
    let spec = ColumnSpec::struct_col("rec", vec![bad_field]);
    let schema = Schema::new(vec![spec]);
    let err = schema.validate().expect_err("should fail validation");
    let msg = err.to_string();
    assert!(
        msg.contains("pipeline") || msg.contains("expects"),
        "unexpected error: {msg}"
    );
}

#[test]
fn struct_field_name_mismatch_in_decompose() {
    // Data has field named "wrong" but schema expects "x".
    let lt = LogicalType::Struct {
        fields: vec![FieldSpec::primitive("x", DataType::I32, vec![])],
    };
    let data = LogicalColumn::Struct {
        fields: vec![(
            "wrong".to_string(),
            LogicalColumn::Primitive(ColumnData::I32(vec![1, 2, 3])),
        )],
    };
    let err = data.decompose(&lt).expect_err("should fail decompose");
    let msg = err.to_string();
    assert!(
        msg.contains("mismatch") || msg.contains("wrong") || msg.contains("x"),
        "unexpected error: {msg}"
    );
}

#[test]
fn struct_field_count_mismatch_in_decompose() {
    let lt = LogicalType::Struct {
        fields: vec![
            FieldSpec::primitive("a", DataType::I32, vec![]),
            FieldSpec::primitive("b", DataType::I32, vec![]),
        ],
    };
    // Only 1 field in data, schema expects 2.
    let data = LogicalColumn::Struct {
        fields: vec![(
            "a".to_string(),
            LogicalColumn::Primitive(ColumnData::I32(vec![1])),
        )],
    };
    let err = data.decompose(&lt).expect_err("should fail");
    let msg = err.to_string();
    assert!(
        msg.contains("field") || msg.contains("2") || msg.contains("1"),
        "unexpected error: {msg}"
    );
}

// ---------------------------------------------------------------------------
// row_count() for Struct
// ---------------------------------------------------------------------------

#[test]
fn struct_row_count() {
    let data = LogicalColumn::Struct {
        fields: vec![
            (
                "a".to_string(),
                LogicalColumn::Primitive(ColumnData::I32(vec![1, 2, 3])),
            ),
            (
                "b".to_string(),
                LogicalColumn::Primitive(ColumnData::I32(vec![4, 5, 6])),
            ),
        ],
    };
    assert_eq!(data.row_count(), 3);

    let empty_struct = LogicalColumn::Struct { fields: vec![] };
    assert_eq!(empty_struct.row_count(), 0);
}
