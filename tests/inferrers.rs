//! Integration tests for the CSV, JSON, and Parquet schema inferrers.
//!
//! Requires all three features:
//! ```
//! cargo test --test inferrers --features schema-csv,schema-json,schema-parquet
//! ```
//! (Or `--all-features` which also enables `schema-avro`.)

#![cfg(all(
    feature = "schema-csv",
    feature = "schema-json",
    feature = "schema-parquet"
))]

use std::io::Cursor;

use helium::schema::csv::{CsvInferOptions, schema_from_csv_str, schema_from_csv_str_with_options};
use helium::schema::json::{
    JsonInferOptions, schema_from_json_str, schema_from_json_str_with_options,
};
use helium::schema::parquet::schema_from_parquet_schema;
use helium::{
    CoderRegistry, ColumnData, DataType, DateUnit, HeliumReader, HeliumWriter, LogicalColumn,
    LogicalType, Schema, TimeUnit,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn write_read_roundtrip(schema: Schema, data: Vec<(String, LogicalColumn)>) -> Vec<LogicalColumn> {
    let reg = CoderRegistry::default();
    let mut buf = Cursor::new(Vec::<u8>::new());
    let mut writer =
        HeliumWriter::new(&mut buf, schema.clone(), &reg).expect("writer construction");
    for (name, col) in &data {
        writer
            .write_column(name, col.clone())
            .expect("write_column");
    }
    writer.finish().expect("finish");
    buf.set_position(0);
    let mut reader = HeliumReader::new(&mut buf, &reg).expect("reader construction");
    data.iter()
        .map(|(name, _)| reader.read_column(name).expect("read_column"))
        .collect()
}

// ===========================================================================
// CSV inferrer tests
// ===========================================================================

#[test]
fn csv_single_int_column() {
    let csv = "count\n1\n2\n3\n";
    let schema = schema_from_csv_str(csv).expect("infer");
    assert_eq!(schema.columns.len(), 1);
    assert_eq!(schema.columns[0].name, "count");
    assert_eq!(
        schema.columns[0].logical_type,
        LogicalType::Primitive {
            data_type: DataType::I64
        }
    );
}

#[test]
fn csv_single_float_column() {
    let csv = "score\n1.5\n2.7\n3.14\n";
    let schema = schema_from_csv_str(csv).expect("infer");
    assert_eq!(
        schema.columns[0].logical_type,
        LogicalType::Primitive {
            data_type: DataType::F64
        }
    );
}

#[test]
fn csv_string_column() {
    let csv = "name\nAlice\nBob\nCharlotte\n";
    let schema = schema_from_csv_str(csv).expect("infer");
    assert_eq!(schema.columns[0].logical_type, LogicalType::Utf8);
}

#[test]
fn csv_mixed_int_float_promotes_to_float() {
    let csv = "val\n1\n2\n3.14\n4\n";
    let schema = schema_from_csv_str(csv).expect("infer");
    assert_eq!(
        schema.columns[0].logical_type,
        LogicalType::Primitive {
            data_type: DataType::F64
        }
    );
}

#[test]
fn csv_mixed_int_string_falls_back_to_utf8() {
    let csv = "x\n1\nhello\n3\n";
    let schema = schema_from_csv_str(csv).expect("infer");
    assert_eq!(schema.columns[0].logical_type, LogicalType::Utf8);
}

#[test]
fn csv_nullable_empty_sentinel() {
    let csv = "v\n1\n\n3\n";
    let schema = schema_from_csv_str(csv).expect("infer");
    let LogicalType::Nullable { inner } = &schema.columns[0].logical_type else {
        panic!(
            "expected Nullable, got {:?}",
            schema.columns[0].logical_type
        )
    };
    assert_eq!(
        **inner,
        LogicalType::Primitive {
            data_type: DataType::I64
        }
    );
}

#[test]
fn csv_null_sentinel_null_keyword() {
    let csv = "v\n1\nNULL\n3\n";
    let schema = schema_from_csv_str(csv).expect("infer");
    assert!(
        matches!(
            &schema.columns[0].logical_type,
            LogicalType::Nullable { .. }
        ),
        "expected Nullable"
    );
}

#[test]
fn csv_multi_column_schema() {
    let csv = "id,name,score\n1,Alice,9.5\n2,Bob,8.0\n";
    let schema = schema_from_csv_str(csv).expect("infer");
    assert_eq!(schema.columns.len(), 3);
    assert_eq!(schema.columns[0].name, "id");
    assert_eq!(
        schema.columns[0].logical_type,
        LogicalType::Primitive {
            data_type: DataType::I64
        }
    );
    assert_eq!(schema.columns[1].name, "name");
    assert_eq!(schema.columns[1].logical_type, LogicalType::Utf8);
    assert_eq!(schema.columns[2].name, "score");
    assert_eq!(
        schema.columns[2].logical_type,
        LogicalType::Primitive {
            data_type: DataType::F64
        }
    );
}

#[test]
fn csv_all_null_column_defaults_to_nullable_utf8() {
    let csv = "v\n\n\n\n";
    let schema = schema_from_csv_str(csv).expect("infer");
    let lt = &schema.columns[0].logical_type;
    match lt {
        LogicalType::Utf8 => {}
        LogicalType::Nullable { inner } => assert_eq!(**inner, LogicalType::Utf8),
        other => panic!("expected Utf8 or Nullable(Utf8), got {other:?}"),
    }
}

#[test]
fn csv_semicolon_delimiter() {
    let opts = CsvInferOptions {
        delimiter: b';',
        ..CsvInferOptions::default()
    };
    let csv = "a;b\n1;hello\n2;world\n";
    let schema = schema_from_csv_str_with_options(csv, &opts).expect("infer");
    assert_eq!(schema.columns.len(), 2);
    assert_eq!(
        schema.columns[0].logical_type,
        LogicalType::Primitive {
            data_type: DataType::I64
        }
    );
    assert_eq!(schema.columns[1].logical_type, LogicalType::Utf8);
}

#[test]
fn csv_no_header_generates_col_names() {
    let opts = CsvInferOptions {
        has_header: false,
        ..CsvInferOptions::default()
    };
    let csv = "1,Alice\n2,Bob\n";
    let schema = schema_from_csv_str_with_options(csv, &opts).expect("infer");
    assert_eq!(schema.columns.len(), 2);
    assert_eq!(schema.columns[0].name, "col_0");
    assert_eq!(schema.columns[1].name, "col_1");
}

#[test]
fn csv_max_rows_option_limits_scan() {
    let opts = CsvInferOptions {
        max_rows: 2,
        ..CsvInferOptions::default()
    };
    let csv = "v\n1\n2\nhello\n"; // row 3 would widen to Utf8
    let schema = schema_from_csv_str_with_options(csv, &opts).expect("infer");
    assert_eq!(
        schema.columns[0].logical_type,
        LogicalType::Primitive {
            data_type: DataType::I64
        }
    );
}

#[test]
fn csv_schema_validates_cleanly() {
    let csv = "id,name,score\n1,Alice,9.5\n2,Bob,8.0\n";
    let schema = schema_from_csv_str(csv).expect("infer");
    schema.validate().expect("schema must be valid");
}

// ── Nullability full-scan regression tests ─────────────────────────────────

/// A column whose first null appears beyond the type-sample window must be
/// inferred as `Nullable<T>`.  Here `max_rows=10` but the null is at row 15.
#[test]
fn csv_late_null_beyond_sample_window_promotes_to_nullable() {
    let opts = CsvInferOptions {
        max_rows: 10,
        ..CsvInferOptions::default()
    };
    // 14 rows of pure integers, then one empty (null) cell.
    let mut csv = "x\n".to_string();
    for i in 0..14 {
        csv.push_str(&format!("{}\n", i));
    }
    csv.push('\n'); // row 15: empty = null sentinel
    let schema = schema_from_csv_str_with_options(&csv, &opts).expect("infer");
    assert_eq!(schema.columns.len(), 1);
    let LogicalType::Nullable { inner } = &schema.columns[0].logical_type else {
        panic!(
            "expected Nullable, got {:?} (late null must promote to Nullable)",
            schema.columns[0].logical_type
        )
    };
    assert_eq!(
        **inner,
        LogicalType::Primitive {
            data_type: DataType::I64
        },
        "inner type must be I64"
    );
}

/// A column with no null values anywhere in the file stays non-nullable.
#[test]
fn csv_no_null_stays_non_nullable() {
    let opts = CsvInferOptions {
        max_rows: 5,
        ..CsvInferOptions::default()
    };
    // 20 rows, all non-empty integers — no null at any position.
    let mut csv = "count\n".to_string();
    for i in 1..=20 {
        csv.push_str(&format!("{}\n", i));
    }
    let schema = schema_from_csv_str_with_options(&csv, &opts).expect("infer");
    assert_eq!(
        schema.columns[0].logical_type,
        LogicalType::Primitive {
            data_type: DataType::I64
        },
        "no-null column must remain non-nullable"
    );
}

/// Two-column test: one column has a late null (beyond sample), the other does
/// not.  Both are correctly classified independently.
#[test]
fn csv_late_null_only_on_affected_column() {
    let opts = CsvInferOptions {
        max_rows: 5,
        ..CsvInferOptions::default()
    };
    // 8 rows: `a` gets a null at row 7; `b` is clean throughout.
    let mut csv = "a,b\n".to_string();
    for i in 0..6 {
        csv.push_str(&format!("{},{}\n", i, i * 2));
    }
    csv.push_str(&format!(",{}\n", 12)); // row 7: a is null, b is fine
    let schema = schema_from_csv_str_with_options(&csv, &opts).expect("infer");
    assert_eq!(schema.columns.len(), 2);
    // `a` should be Nullable<I64>
    let LogicalType::Nullable { inner } = &schema.columns[0].logical_type else {
        panic!(
            "expected Nullable for column `a`, got {:?}",
            schema.columns[0].logical_type
        )
    };
    assert_eq!(
        **inner,
        LogicalType::Primitive {
            data_type: DataType::I64
        }
    );
    // `b` should be non-nullable I64
    assert_eq!(
        schema.columns[1].logical_type,
        LogicalType::Primitive {
            data_type: DataType::I64
        },
        "column `b` has no nulls and must remain non-nullable"
    );
}

// ── CSV round-trip ──────────────────────────────────────────────────────────

#[test]
fn csv_round_trip_int_utf8() {
    let csv = "id,label\n10,hello\n20,world\n";
    let schema = schema_from_csv_str(csv).expect("infer");

    let ids = LogicalColumn::Primitive(ColumnData::I64(vec![10, 20]));
    let labels = LogicalColumn::Utf8(vec!["hello".into(), "world".into()]);

    let results = write_read_roundtrip(
        schema,
        vec![("id".into(), ids.clone()), ("label".into(), labels.clone())],
    );
    assert_eq!(results[0], ids);
    assert_eq!(results[1], labels);
}

#[test]
fn csv_round_trip_nullable_float() {
    let csv = "score\n1.5\n\n3.0\n";
    let schema = schema_from_csv_str(csv).expect("infer");

    let col = LogicalColumn::Nullable {
        present: vec![true, false, true],
        value: Box::new(LogicalColumn::Primitive(ColumnData::F64(vec![1.5, 3.0]))),
    };
    let results = write_read_roundtrip(schema, vec![("score".into(), col.clone())]);
    assert_eq!(results[0], col);
}

// ===========================================================================
// JSON inferrer tests
// ===========================================================================

#[test]
fn json_flat_int_string() {
    let json = r#"[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]"#;
    let schema = schema_from_json_str(json).expect("infer");
    assert_eq!(schema.columns.len(), 2);
    let id = schema.column("id").expect("id col");
    assert_eq!(
        id.logical_type,
        LogicalType::Primitive {
            data_type: DataType::I64
        }
    );
    let name = schema.column("name").expect("name col");
    assert_eq!(name.logical_type, LogicalType::Utf8);
}

#[test]
fn json_boolean_becomes_u8() {
    let json = r#"[{"flag": true}, {"flag": false}]"#;
    let schema = schema_from_json_str(json).expect("infer");
    assert_eq!(
        schema.columns[0].logical_type,
        LogicalType::Primitive {
            data_type: DataType::U8
        }
    );
}

#[test]
fn json_float_column() {
    let json = r#"[{"x": 1.5}, {"x": 2.7}]"#;
    let schema = schema_from_json_str(json).expect("infer");
    assert_eq!(
        schema.columns[0].logical_type,
        LogicalType::Primitive {
            data_type: DataType::F64
        }
    );
}

#[test]
fn json_int_float_promotes_to_float() {
    let json = r#"[{"x": 1}, {"x": 2.5}]"#;
    let schema = schema_from_json_str(json).expect("infer");
    assert_eq!(
        schema.columns[0].logical_type,
        LogicalType::Primitive {
            data_type: DataType::F64
        }
    );
}

#[test]
fn json_nullable_field_with_null() {
    let json = r#"[{"v": 1}, {"v": null}, {"v": 3}]"#;
    let schema = schema_from_json_str(json).expect("infer");
    let LogicalType::Nullable { inner } = &schema.columns[0].logical_type else {
        panic!(
            "expected Nullable, got {:?}",
            schema.columns[0].logical_type
        )
    };
    assert_eq!(
        **inner,
        LogicalType::Primitive {
            data_type: DataType::I64
        }
    );
}

#[test]
fn json_absent_field_becomes_nullable() {
    let json = r#"[{"id": 1, "extra": "x"}, {"id": 2}]"#;
    let schema = schema_from_json_str(json).expect("infer");
    let extra = schema.column("extra").expect("extra col");
    assert!(
        matches!(&extra.logical_type, LogicalType::Nullable { .. }),
        "absent field must be Nullable, got {:?}",
        extra.logical_type
    );
}

#[test]
fn json_nested_object_becomes_struct() {
    let json = r#"[
        {"user": {"name": "Alice", "age": 30}},
        {"user": {"name": "Bob",   "age": 25}}
    ]"#;
    let schema = schema_from_json_str(json).expect("infer");
    let user = schema.column("user").expect("user col");
    let LogicalType::Struct { fields } = &user.logical_type else {
        panic!("expected Struct, got {:?}", user.logical_type)
    };
    assert_eq!(fields.len(), 2);
    // serde_json's default Map type (BTreeMap) iterates keys in alphabetical order,
    // so field order in the inferred Struct matches ASCII-sorted JSON key order.
    let name_f = fields
        .iter()
        .find(|f| f.name == "name")
        .expect("name field");
    let age_f = fields.iter().find(|f| f.name == "age").expect("age field");
    assert_eq!(name_f.logical_type, LogicalType::Utf8);
    assert_eq!(
        age_f.logical_type,
        LogicalType::Primitive {
            data_type: DataType::I64
        }
    );
}

#[test]
fn json_array_of_strings_becomes_list_utf8() {
    let json = r#"[{"tags": ["a", "b"]}, {"tags": ["c"]}]"#;
    let schema = schema_from_json_str(json).expect("infer");
    let tags = schema.column("tags").expect("tags col");
    let LogicalType::List { inner } = &tags.logical_type else {
        panic!("expected List, got {:?}", tags.logical_type)
    };
    assert_eq!(**inner, LogicalType::Utf8);
}

#[test]
fn json_array_of_ints_becomes_list_i64() {
    let json = r#"[{"nums": [1, 2, 3]}]"#;
    let schema = schema_from_json_str(json).expect("infer");
    let nums = schema.column("nums").expect("nums col");
    let LogicalType::List { inner } = &nums.logical_type else {
        panic!("expected List")
    };
    assert_eq!(
        **inner,
        LogicalType::Primitive {
            data_type: DataType::I64
        }
    );
}

#[test]
fn json_ndjson_format() {
    let ndjson = "{\"id\": 1, \"v\": \"a\"}\n{\"id\": 2, \"v\": \"b\"}\n";
    let schema = schema_from_json_str(ndjson).expect("infer NDJSON");
    assert_eq!(schema.columns.len(), 2);
}

#[test]
fn json_too_many_variants_falls_back_to_utf8() {
    // 4 distinct kinds with default max 3 → Utf8.
    let json = r#"[{"x": 1}, {"x": "hello"}, {"x": true}, {"x": 3.14}]"#;
    let opts = JsonInferOptions {
        max_union_variants: 3,
        ..JsonInferOptions::default()
    };
    let schema = schema_from_json_str_with_options(json, &opts).expect("infer");
    assert_eq!(
        schema.columns[0].logical_type,
        LogicalType::Utf8,
        "should fall back to Utf8 for >3 incompatible types"
    );
}

#[test]
fn json_two_incompatible_types_produce_union() {
    let json = r#"[{"x": 1}, {"x": "hello"}]"#;
    let opts = JsonInferOptions {
        max_union_variants: 3,
        ..JsonInferOptions::default()
    };
    let schema = schema_from_json_str_with_options(json, &opts).expect("infer");
    assert!(
        matches!(&schema.columns[0].logical_type, LogicalType::Union { .. }),
        "expected Union for 2 incompatible types, got {:?}",
        schema.columns[0].logical_type
    );
}

#[test]
fn json_schema_validates_cleanly() {
    let json = r#"[{"id": 1, "name": "Alice", "scores": [9, 10]}]"#;
    let schema = schema_from_json_str(json).expect("infer");
    schema.validate().expect("valid schema");
}

// ── JSON round-trip ─────────────────────────────────────────────────────────

#[test]
fn json_round_trip_flat_record() {
    let json = r#"[{"id": 1, "label": "hello"}, {"id": 2, "label": "world"}]"#;
    let schema = schema_from_json_str(json).expect("infer");

    let ids = LogicalColumn::Primitive(ColumnData::I64(vec![1, 2]));
    let labels = LogicalColumn::Utf8(vec!["hello".into(), "world".into()]);

    let results = write_read_roundtrip(
        schema,
        vec![("id".into(), ids.clone()), ("label".into(), labels.clone())],
    );
    assert_eq!(results[0], ids);
    assert_eq!(results[1], labels);
}

// ===========================================================================
// Parquet inferrer tests
// ===========================================================================

fn helium_from_parquet_str(msg: &str) -> Schema {
    let pq_schema = parquet::schema::parser::parse_message_type(msg).expect("parse parquet schema");
    schema_from_parquet_schema(&pq_schema).expect("convert parquet schema")
}

#[test]
fn parquet_required_int32_column() {
    let schema = helium_from_parquet_str("message m { required int32 id; }");
    assert_eq!(schema.columns.len(), 1);
    assert_eq!(schema.columns[0].name, "id");
    assert_eq!(
        schema.columns[0].logical_type,
        LogicalType::Primitive {
            data_type: DataType::I32
        }
    );
}

#[test]
fn parquet_optional_becomes_nullable() {
    let schema = helium_from_parquet_str("message m { optional int64 ts; }");
    let LogicalType::Nullable { inner } = &schema.columns[0].logical_type else {
        panic!("expected Nullable")
    };
    assert_eq!(
        **inner,
        LogicalType::Primitive {
            data_type: DataType::I64
        }
    );
}

#[test]
fn parquet_byte_array_utf8_becomes_utf8() {
    let schema = helium_from_parquet_str("message m { required binary name (UTF8); }");
    assert_eq!(schema.columns[0].logical_type, LogicalType::Utf8);
}

#[test]
fn parquet_byte_array_no_annotation_is_binary() {
    let schema = helium_from_parquet_str("message m { required binary blob; }");
    assert_eq!(schema.columns[0].logical_type, LogicalType::Binary);
}

#[test]
fn parquet_float_double_columns() {
    let schema =
        helium_from_parquet_str("message m { required float f32_col; required double f64_col; }");
    assert_eq!(
        schema.columns[0].logical_type,
        LogicalType::Primitive {
            data_type: DataType::F32
        }
    );
    assert_eq!(
        schema.columns[1].logical_type,
        LogicalType::Primitive {
            data_type: DataType::F64
        }
    );
}

#[test]
fn parquet_boolean_becomes_u8() {
    let schema = helium_from_parquet_str("message m { required boolean flag; }");
    assert_eq!(
        schema.columns[0].logical_type,
        LogicalType::Primitive {
            data_type: DataType::U8
        }
    );
}

#[test]
fn parquet_plain_group_becomes_struct() {
    let schema = helium_from_parquet_str(
        r#"message m {
            required group loc {
                required double lat;
                required double lon;
            }
        }"#,
    );
    let LogicalType::Struct { fields } = &schema.columns[0].logical_type else {
        panic!("expected Struct")
    };
    assert_eq!(fields.len(), 2);
    assert_eq!(fields[0].name, "lat");
    assert_eq!(fields[1].name, "lon");
}

#[test]
fn parquet_list_becomes_list() {
    let schema = helium_from_parquet_str(
        r#"message m {
            required group tags (LIST) {
                repeated group list {
                    required binary element (UTF8);
                }
            }
        }"#,
    );
    let LogicalType::List { inner } = &schema.columns[0].logical_type else {
        panic!("expected List, got {:?}", schema.columns[0].logical_type)
    };
    assert_eq!(**inner, LogicalType::Utf8);
}

#[test]
fn parquet_optional_list_is_nullable_list() {
    let schema = helium_from_parquet_str(
        r#"message m {
            optional group scores (LIST) {
                repeated group list {
                    required double element;
                }
            }
        }"#,
    );
    let LogicalType::Nullable { inner } = &schema.columns[0].logical_type else {
        panic!("expected Nullable(List)")
    };
    assert!(
        matches!(**inner, LogicalType::List { .. }),
        "inner must be List"
    );
}

#[test]
fn parquet_map_becomes_map() {
    let schema = helium_from_parquet_str(
        r#"message m {
            required group attrs (MAP) {
                repeated group key_value {
                    required binary key (UTF8);
                    optional int32 value;
                }
            }
        }"#,
    );
    let LogicalType::Map { key, value } = &schema.columns[0].logical_type else {
        panic!("expected Map, got {:?}", schema.columns[0].logical_type)
    };
    assert_eq!(**key, LogicalType::Utf8, "map key must be Utf8");
    assert!(
        matches!(value.as_ref(), LogicalType::Nullable { .. }),
        "optional map value must be Nullable, got {value:?}"
    );
}

#[test]
fn parquet_date_is_date_days() {
    let schema = helium_from_parquet_str("message m { required int32 d (DATE); }");
    assert_eq!(
        schema.columns[0].logical_type,
        LogicalType::Date {
            unit: DateUnit::Days
        }
    );
}

#[test]
fn parquet_timestamp_millis_is_datetime() {
    let schema = helium_from_parquet_str("message m { required int64 ts (TIMESTAMP_MILLIS); }");
    // TIMESTAMP_MILLIS in the legacy annotation is treated as UTC-adjusted.
    assert!(
        matches!(
            schema.columns[0].logical_type,
            LogicalType::Datetime {
                unit: TimeUnit::Millis,
                ..
            }
        ),
        "got: {:?}",
        schema.columns[0].logical_type
    );
}

#[test]
fn parquet_schema_validates_cleanly() {
    let schema = helium_from_parquet_str(
        r#"message event {
            required int64 id;
            optional binary name (UTF8);
            required double score;
        }"#,
    );
    schema.validate().expect("valid schema");
}

// ── Parquet round-trip ──────────────────────────────────────────────────────

#[test]
fn parquet_round_trip_primitives() {
    let schema =
        helium_from_parquet_str(r#"message row { required int64 id; required double value; }"#);
    let ids = LogicalColumn::Primitive(ColumnData::I64(vec![1, 2, 3]));
    let values = LogicalColumn::Primitive(ColumnData::F64(vec![1.0, 2.5, 3.7]));
    let results = write_read_roundtrip(
        schema,
        vec![("id".into(), ids.clone()), ("value".into(), values.clone())],
    );
    assert_eq!(results[0], ids);
    assert_eq!(results[1], values);
}

// ── Parquet real file (env-gated) ────────────────────────────────────────────

#[test]
fn parquet_real_file_if_env_set() {
    let path = match std::env::var("HELIUM_PARQUET_PATH") {
        Ok(p) => std::path::PathBuf::from(p),
        Err(_) => return,
    };
    use helium::schema::parquet::schema_from_parquet;
    let schema = schema_from_parquet(&path).expect("schema_from_parquet on real file");
    assert!(!schema.columns.is_empty(), "real parquet must have columns");
    schema
        .validate()
        .expect("inferred real parquet schema must be valid");
}
