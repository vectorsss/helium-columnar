//! Round-trip / conversion tests for the Avro `.avsc` → Helium `Schema` parser.
#![cfg(feature = "schema-avro")]
//!
//! Tests cover:
//! * All Avro primitive types
//! * Records (→ Struct), arrays (→ List), maps (→ Map)
//! * `["null", T]` / `[T, "null"]` unions → Nullable canonicalization
//! * Single-element unions → unwrapped type
//! * Multi-variant non-null unions → Union
//! * `["null", A, B]` unions → Nullable(Union)
//! * Avro logical types (timestamp-millis/micros, date, decimal, uuid)
//! * Enum → Dictionary { inner: Utf8 }
//! * Fixed → Binary
//! * Named type references
//! * 3 representative MR-shape schemas
//! * Negative tests (bad Avro shapes that must produce HeliumError::Schema)

use helium::schema::avro::{avsc_to_logical_type, avsc_to_schema};
use helium::{DataType, DateUnit, LogicalType, TimeUnit};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn lt(avsc: &str) -> LogicalType {
    avsc_to_logical_type(avsc).unwrap_or_else(|e| panic!("parse failed: {e}"))
}

fn schema_err(avsc: &str) -> String {
    avsc_to_schema(avsc)
        .expect_err("expected error")
        .to_string()
}

fn lt_err(avsc: &str) -> String {
    avsc_to_logical_type(avsc)
        .expect_err("expected error")
        .to_string()
}

// ---------------------------------------------------------------------------
// §5.7: Avro primitives
// ---------------------------------------------------------------------------

#[test]
fn boolean_to_u8() {
    assert_eq!(
        lt(r#""boolean""#),
        LogicalType::Primitive {
            data_type: DataType::U8
        }
    );
}

#[test]
fn int_to_i32() {
    assert_eq!(
        lt(r#""int""#),
        LogicalType::Primitive {
            data_type: DataType::I32
        }
    );
}

#[test]
fn long_to_i64() {
    assert_eq!(
        lt(r#""long""#),
        LogicalType::Primitive {
            data_type: DataType::I64
        }
    );
}

#[test]
fn float_to_f32() {
    assert_eq!(
        lt(r#""float""#),
        LogicalType::Primitive {
            data_type: DataType::F32
        }
    );
}

#[test]
fn double_to_f64() {
    assert_eq!(
        lt(r#""double""#),
        LogicalType::Primitive {
            data_type: DataType::F64
        }
    );
}

#[test]
fn bytes_to_binary() {
    assert_eq!(lt(r#""bytes""#), LogicalType::Binary);
}

#[test]
fn string_to_utf8() {
    assert_eq!(lt(r#""string""#), LogicalType::Utf8);
}

// ---------------------------------------------------------------------------
// §5.7: Avro logical types
// ---------------------------------------------------------------------------

#[test]
fn date_int_to_date_days() {
    let avsc = r#"{"type": "int", "logicalType": "date"}"#;
    assert_eq!(
        lt(avsc),
        LogicalType::Date {
            unit: DateUnit::Days
        }
    );
}

#[test]
fn time_millis_to_i32() {
    let avsc = r#"{"type": "int", "logicalType": "time-millis"}"#;
    assert_eq!(
        lt(avsc),
        LogicalType::Primitive {
            data_type: DataType::I32
        }
    );
}

#[test]
fn timestamp_millis_to_datetime() {
    let avsc = r#"{"type": "long", "logicalType": "timestamp-millis"}"#;
    assert_eq!(
        lt(avsc),
        LogicalType::Datetime {
            unit: TimeUnit::Millis,
            timezone: None
        }
    );
}

#[test]
fn timestamp_micros_to_datetime() {
    let avsc = r#"{"type": "long", "logicalType": "timestamp-micros"}"#;
    assert_eq!(
        lt(avsc),
        LogicalType::Datetime {
            unit: TimeUnit::Micros,
            timezone: None
        }
    );
}

#[test]
fn local_timestamp_millis_to_datetime() {
    let avsc = r#"{"type": "long", "logicalType": "local-timestamp-millis"}"#;
    assert_eq!(
        lt(avsc),
        LogicalType::Datetime {
            unit: TimeUnit::Millis,
            timezone: None
        }
    );
}

#[test]
fn local_timestamp_micros_to_datetime() {
    let avsc = r#"{"type": "long", "logicalType": "local-timestamp-micros"}"#;
    assert_eq!(
        lt(avsc),
        LogicalType::Datetime {
            unit: TimeUnit::Micros,
            timezone: None
        }
    );
}

#[test]
fn time_micros_to_i64() {
    let avsc = r#"{"type": "long", "logicalType": "time-micros"}"#;
    assert_eq!(
        lt(avsc),
        LogicalType::Primitive {
            data_type: DataType::I64
        }
    );
}

#[test]
fn decimal_bytes_to_decimal128() {
    let avsc = r#"{"type": "bytes", "logicalType": "decimal", "precision": 12, "scale": 2}"#;
    assert_eq!(
        lt(avsc),
        LogicalType::Decimal128 {
            precision: 12,
            scale: 2
        }
    );
}

#[test]
fn uuid_string_to_binary() {
    let avsc = r#"{"type": "string", "logicalType": "uuid"}"#;
    assert_eq!(lt(avsc), LogicalType::Binary);
}

// ---------------------------------------------------------------------------
// §5.7: Avro array → List
// ---------------------------------------------------------------------------

#[test]
fn array_of_string_to_list_utf8() {
    let avsc = r#"{"type": "array", "items": "string"}"#;
    let expected = LogicalType::List {
        inner: Box::new(LogicalType::Utf8),
    };
    assert_eq!(lt(avsc), expected);
}

#[test]
fn array_of_int_to_list_primitive() {
    let avsc = r#"{"type": "array", "items": "int"}"#;
    let expected = LogicalType::List {
        inner: Box::new(LogicalType::Primitive {
            data_type: DataType::I32,
        }),
    };
    assert_eq!(lt(avsc), expected);
}

#[test]
fn nested_array_to_list_of_list() {
    let avsc = r#"{"type": "array", "items": {"type": "array", "items": "double"}}"#;
    let expected = LogicalType::List {
        inner: Box::new(LogicalType::List {
            inner: Box::new(LogicalType::Primitive {
                data_type: DataType::F64,
            }),
        }),
    };
    assert_eq!(lt(avsc), expected);
}

// ---------------------------------------------------------------------------
// §5.7: Avro map → Map<Utf8, V>
// ---------------------------------------------------------------------------

#[test]
fn map_string_values_to_map_utf8_utf8() {
    let avsc = r#"{"type": "map", "values": "string"}"#;
    let expected = LogicalType::Map {
        key: Box::new(LogicalType::Utf8),
        value: Box::new(LogicalType::Utf8),
    };
    assert_eq!(lt(avsc), expected);
}

#[test]
fn map_long_values_to_map_utf8_i64() {
    let avsc = r#"{"type": "map", "values": "long"}"#;
    let expected = LogicalType::Map {
        key: Box::new(LogicalType::Utf8),
        value: Box::new(LogicalType::Primitive {
            data_type: DataType::I64,
        }),
    };
    assert_eq!(lt(avsc), expected);
}

#[test]
fn map_key_is_always_utf8() {
    // Avro maps are always string-keyed regardless of input type.
    let avsc = r#"{"type": "map", "values": "int"}"#;
    let result = lt(avsc);
    let LogicalType::Map { key, .. } = result else {
        panic!("expected Map")
    };
    assert_eq!(*key, LogicalType::Utf8, "Map key must always be Utf8");
}

// ---------------------------------------------------------------------------
// §5.7: Avro enum → Dictionary{inner:Utf8}
// ---------------------------------------------------------------------------

#[test]
fn enum_to_dict_utf8() {
    let avsc = r#"{"type": "enum", "name": "Status", "symbols": ["PENDING", "DONE", "FAILED"]}"#;
    // Avro enums now produce the v3 Dictionary{inner:Utf8} type.
    assert_eq!(
        lt(avsc),
        LogicalType::Dictionary {
            inner: Box::new(LogicalType::Utf8),
        }
    );
}

#[test]
fn enum_via_named_reference() {
    // An enum defined inline can be referenced by name from a later field.
    let avsc = r#"{
        "type": "record",
        "name": "Event",
        "fields": [
            {"name": "status", "type": {"type": "enum", "name": "Status", "symbols": ["A", "B"]}},
            {"name": "prev_status", "type": "Status"}
        ]
    }"#;
    let schema = avsc_to_schema(avsc).expect("parse");
    let status_col = schema.column("status").expect("status col");
    let prev_col = schema.column("prev_status").expect("prev_status col");
    let dict_utf8 = LogicalType::Dictionary {
        inner: Box::new(LogicalType::Utf8),
    };
    assert_eq!(status_col.logical_type, dict_utf8);
    assert_eq!(prev_col.logical_type, dict_utf8);
}

// ---------------------------------------------------------------------------
// §5.7: Avro fixed → Binary
// ---------------------------------------------------------------------------

#[test]
fn fixed_to_binary() {
    let avsc = r#"{"type": "fixed", "name": "Uuid", "size": 16}"#;
    assert_eq!(lt(avsc), LogicalType::Binary);
}

#[test]
fn fixed_decimal_to_decimal128() {
    let avsc = r#"{"type": "fixed", "name": "Amount", "size": 8, "logicalType": "decimal", "precision": 10, "scale": 2}"#;
    // fixed + decimal logicalType with precision → Decimal128(10, 2)
    assert_eq!(
        lt(avsc),
        LogicalType::Decimal128 {
            precision: 10,
            scale: 2
        }
    );
}

#[test]
fn fixed_no_precision_to_binary() {
    // fixed without decimal logicalType (or without precision) → Binary
    let avsc = r#"{"type": "fixed", "name": "HashVal", "size": 16}"#;
    assert_eq!(lt(avsc), LogicalType::Binary);
}

// ---------------------------------------------------------------------------
// §5.7: Union → Nullable / Union
// ---------------------------------------------------------------------------

#[test]
fn null_then_string_to_nullable_utf8() {
    // ["null", T] → Nullable(T), regardless of order.
    let avsc = r#"["null", "string"]"#;
    let expected = LogicalType::Nullable {
        inner: Box::new(LogicalType::Utf8),
    };
    assert_eq!(lt(avsc), expected);
}

#[test]
fn string_then_null_to_nullable_utf8() {
    // [T, "null"] → same canonicalization as ["null", T].
    let avsc = r#"["string", "null"]"#;
    let expected = LogicalType::Nullable {
        inner: Box::new(LogicalType::Utf8),
    };
    assert_eq!(lt(avsc), expected);
}

#[test]
fn null_then_int_to_nullable_i32() {
    let avsc = r#"["null", "int"]"#;
    let expected = LogicalType::Nullable {
        inner: Box::new(LogicalType::Primitive {
            data_type: DataType::I32,
        }),
    };
    assert_eq!(lt(avsc), expected);
}

#[test]
fn null_then_double_to_nullable_f64() {
    let avsc = r#"["null", "double"]"#;
    let expected = LogicalType::Nullable {
        inner: Box::new(LogicalType::Primitive {
            data_type: DataType::F64,
        }),
    };
    assert_eq!(lt(avsc), expected);
}

#[test]
fn single_element_union_unwrapped() {
    // [T] → T (union wrapper stripped).
    let avsc = r#"["int"]"#;
    assert_eq!(
        lt(avsc),
        LogicalType::Primitive {
            data_type: DataType::I32
        }
    );
}

#[test]
fn multi_variant_non_null_union() {
    // [A, B] (no null) → Union { variants: [("string", Utf8), ("int", I32)] }
    let avsc = r#"["string", "int"]"#;
    let expected = LogicalType::Union {
        variants: vec![
            ("string".to_string(), LogicalType::Utf8),
            (
                "int".to_string(),
                LogicalType::Primitive {
                    data_type: DataType::I32,
                },
            ),
        ],
    };
    assert_eq!(lt(avsc), expected);
}

#[test]
fn null_plus_multi_variants_to_nullable_union() {
    // ["null", A, B] → Nullable(Union { variants: [(A, ...), (B, ...)] })
    let avsc = r#"["null", "string", "int"]"#;
    let result = lt(avsc);
    let LogicalType::Nullable { inner } = result else {
        panic!("expected Nullable, got {result:?}")
    };
    let LogicalType::Union { variants } = *inner else {
        panic!("expected Union inside Nullable")
    };
    assert_eq!(variants.len(), 2);
    assert_eq!(variants[0].0, "string");
    assert_eq!(variants[0].1, LogicalType::Utf8);
    assert_eq!(variants[1].0, "int");
    assert_eq!(
        variants[1].1,
        LogicalType::Primitive {
            data_type: DataType::I32
        }
    );
}

#[test]
fn union_variant_name_from_record() {
    // Record variant in a union uses the record's name field.
    let avsc = r#"["string", {"type": "record", "name": "Point", "fields": [{"name": "x", "type": "int"}]}]"#;
    let result = lt(avsc);
    let LogicalType::Union { variants } = result else {
        panic!("expected Union")
    };
    assert_eq!(variants.len(), 2);
    assert_eq!(variants[0].0, "string");
    assert_eq!(variants[1].0, "Point");
}

#[test]
fn union_variant_name_from_array_type() {
    // An array type in a union gets name "array".
    let avsc = r#"["string", {"type": "array", "items": "int"}]"#;
    let result = lt(avsc);
    let LogicalType::Union { variants } = result else {
        panic!("expected Union")
    };
    assert_eq!(variants[1].0, "array");
}

// ---------------------------------------------------------------------------
// §5.7: Avro record → Struct
// ---------------------------------------------------------------------------

#[test]
fn record_to_struct_flat() {
    let avsc = r#"{
        "type": "record",
        "name": "Point",
        "fields": [
            {"name": "x", "type": "int"},
            {"name": "y", "type": "int"}
        ]
    }"#;
    let result = lt(avsc);
    let LogicalType::Struct { fields } = result else {
        panic!("expected Struct, got {result:?}")
    };
    assert_eq!(fields.len(), 2);
    assert_eq!(fields[0].name, "x");
    assert_eq!(
        fields[0].logical_type,
        LogicalType::Primitive {
            data_type: DataType::I32
        }
    );
    assert_eq!(fields[1].name, "y");
}

#[test]
fn record_fields_become_schema_columns() {
    let avsc = r#"{
        "type": "record",
        "name": "Row",
        "fields": [
            {"name": "id",   "type": "long"},
            {"name": "name", "type": "string"},
            {"name": "score","type": "double"}
        ]
    }"#;
    let schema = avsc_to_schema(avsc).expect("parse");
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
fn nested_records_become_nested_struct() {
    let avsc = r#"{
        "type": "record",
        "name": "Event",
        "fields": [
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
    let schema = avsc_to_schema(avsc).expect("parse");
    let loc_col = schema.column("loc").expect("loc column");
    let LogicalType::Struct { fields } = &loc_col.logical_type else {
        panic!("expected Struct for loc")
    };
    assert_eq!(fields.len(), 2);
    assert_eq!(fields[0].name, "lat");
    assert_eq!(fields[1].name, "lon");
}

// ---------------------------------------------------------------------------
// §5.7: Schema validates cleanly (encoding count check)
// ---------------------------------------------------------------------------

#[test]
fn produced_schema_passes_validate() {
    // A representative schema with most types should pass Schema::validate().
    let avsc = r#"{
        "type": "record",
        "name": "Sample",
        "fields": [
            {"name": "id",      "type": "long"},
            {"name": "label",   "type": "string"},
            {"name": "score",   "type": ["null", "double"]},
            {"name": "tags",    "type": {"type": "array", "items": "string"}},
            {"name": "meta",    "type": {"type": "map", "values": "string"}},
            {"name": "kind",    "type": {"type": "enum", "name": "Kind", "symbols": ["A", "B"]}},
            {"name": "payload", "type": "bytes"}
        ]
    }"#;
    let schema = avsc_to_schema(avsc).expect("avsc_to_schema should succeed");
    // avsc_to_schema calls schema.validate() internally; if we reach here it passed.
    assert_eq!(schema.columns.len(), 7);
}

// ---------------------------------------------------------------------------
// §5.7: MR-shape schema #1 — Server log record
// ---------------------------------------------------------------------------

/// Representative MR (map-reduce) data shape: server access log.
///
/// Demonstrates: timestamp (logical type), enum (Dictionary{Utf8}), string, map, and
/// nullable nested record.
///
/// Note: in production, real `.avsc` schemas would replace this hand-crafted
/// example once available from the pipeline owners.
#[test]
fn mr_schema_1_server_log() {
    let avsc = r#"{
        "type": "record",
        "name": "ServerLog",
        "fields": [
            {
                "name": "timestamp",
                "type": {"type": "long", "logicalType": "timestamp-millis"}
            },
            {
                "name": "level",
                "type": {
                    "type": "enum",
                    "name": "LogLevel",
                    "symbols": ["DEBUG", "INFO", "WARN", "ERROR"]
                }
            },
            {"name": "message", "type": "string"},
            {
                "name": "attrs",
                "type": {"type": "map", "values": "string"}
            },
            {
                "name": "request",
                "type": ["null", {
                    "type": "record",
                    "name": "HttpRequest",
                    "fields": [
                        {"name": "method",      "type": "string"},
                        {"name": "path",        "type": "string"},
                        {"name": "status_code", "type": "int"}
                    ]
                }]
            }
        ]
    }"#;

    let schema = avsc_to_schema(avsc).expect("server log schema should parse cleanly");
    assert_eq!(schema.columns.len(), 5, "expected 5 top-level columns");

    // timestamp: long with logicalType → Datetime(Millis)
    let ts = schema.column("timestamp").expect("timestamp");
    assert_eq!(
        ts.logical_type,
        LogicalType::Datetime {
            unit: TimeUnit::Millis,
            timezone: None
        },
        "timestamp must be Datetime(Millis)"
    );

    // level: enum → Dictionary{inner:Utf8} (v3 type)
    let level = schema.column("level").expect("level");
    assert_eq!(
        level.logical_type,
        LogicalType::Dictionary {
            inner: Box::new(LogicalType::Utf8),
        },
        "level must be Dictionary{{inner:Utf8}}"
    );

    // message: string → Utf8
    let message = schema.column("message").expect("message");
    assert_eq!(message.logical_type, LogicalType::Utf8);

    // attrs: map<string, string> → Map { key: Utf8, value: Utf8 }
    let attrs = schema.column("attrs").expect("attrs");
    let LogicalType::Map { key, value } = &attrs.logical_type else {
        panic!("attrs must be Map, got {:?}", attrs.logical_type)
    };
    assert_eq!(**key, LogicalType::Utf8, "map key must be Utf8");
    assert_eq!(**value, LogicalType::Utf8, "map value must be Utf8");

    // request: ["null", record] → Nullable(Struct)
    let request = schema.column("request").expect("request");
    let LogicalType::Nullable { inner } = &request.logical_type else {
        panic!("request must be Nullable, got {:?}", request.logical_type)
    };
    let LogicalType::Struct { fields } = inner.as_ref() else {
        panic!("request inner must be Struct, got {inner:?}")
    };
    assert_eq!(fields.len(), 3, "HttpRequest must have 3 fields");
    assert_eq!(fields[0].name, "method");
    assert_eq!(fields[1].name, "path");
    assert_eq!(fields[2].name, "status_code");
    assert_eq!(
        fields[2].logical_type,
        LogicalType::Primitive {
            data_type: DataType::I32
        }
    );
}

// ---------------------------------------------------------------------------
// §5.7: MR-shape schema #2 — Transaction event
// ---------------------------------------------------------------------------

/// Representative MR data shape: financial transaction event.
///
/// Demonstrates: string, decimal bytes (Binary), array of string, nullable
/// string (parent_id).
///
/// Note: real schemas should replace this example once available.
#[test]
fn mr_schema_2_transaction_event() {
    let avsc = r#"{
        "type": "record",
        "name": "Transaction",
        "fields": [
            {"name": "id",       "type": "string"},
            {
                "name": "amount",
                "type": {
                    "type": "bytes",
                    "logicalType": "decimal",
                    "precision": 12,
                    "scale": 2
                }
            },
            {"name": "currency", "type": "string"},
            {
                "name": "metadata",
                "type": {"type": "array", "items": "string"}
            },
            {"name": "parent_id", "type": ["null", "string"]}
        ]
    }"#;

    let schema = avsc_to_schema(avsc).expect("transaction schema should parse cleanly");
    assert_eq!(schema.columns.len(), 5);

    // id: string → Utf8
    let id = schema.column("id").expect("id");
    assert_eq!(id.logical_type, LogicalType::Utf8);

    // amount: bytes with decimal logicalType → Decimal128(12, 2)
    let amount = schema.column("amount").expect("amount");
    assert_eq!(
        amount.logical_type,
        LogicalType::Decimal128 {
            precision: 12,
            scale: 2
        },
        "decimal bytes must be Decimal128"
    );

    // currency: string → Utf8
    let currency = schema.column("currency").expect("currency");
    assert_eq!(currency.logical_type, LogicalType::Utf8);

    // metadata: array<string> → List<Utf8>
    let metadata = schema.column("metadata").expect("metadata");
    let LogicalType::List { inner } = &metadata.logical_type else {
        panic!("metadata must be List, got {:?}", metadata.logical_type)
    };
    assert_eq!(**inner, LogicalType::Utf8, "list inner must be Utf8");

    // parent_id: ["null", "string"] → Nullable(Utf8)
    let parent_id = schema.column("parent_id").expect("parent_id");
    let LogicalType::Nullable { inner } = &parent_id.logical_type else {
        panic!(
            "parent_id must be Nullable, got {:?}",
            parent_id.logical_type
        )
    };
    assert_eq!(**inner, LogicalType::Utf8, "nullable inner must be Utf8");
}

// ---------------------------------------------------------------------------
// §5.7: MR-shape schema #3 — Sensor reading
// ---------------------------------------------------------------------------

/// Representative MR data shape: IoT sensor reading stream.
///
/// Demonstrates: string, array of nested record, nullable double inside
/// nested struct.
///
/// Note: real schemas should replace this example once available.
#[test]
fn mr_schema_3_sensor_reading() {
    let avsc = r#"{
        "type": "record",
        "name": "SensorReading",
        "fields": [
            {"name": "device_id", "type": "string"},
            {
                "name": "readings",
                "type": {
                    "type": "array",
                    "items": {
                        "type": "record",
                        "name": "Reading",
                        "fields": [
                            {"name": "key",     "type": "string"},
                            {"name": "value",   "type": "double"},
                            {"name": "quality", "type": ["null", "double"]}
                        ]
                    }
                }
            }
        ]
    }"#;

    let schema = avsc_to_schema(avsc).expect("sensor reading schema should parse cleanly");
    assert_eq!(schema.columns.len(), 2);

    // device_id: string → Utf8
    let device_id = schema.column("device_id").expect("device_id");
    assert_eq!(device_id.logical_type, LogicalType::Utf8);

    // readings: array<Reading> → List<Struct>
    let readings = schema.column("readings").expect("readings");
    let LogicalType::List { inner } = &readings.logical_type else {
        panic!("readings must be List, got {:?}", readings.logical_type)
    };
    let LogicalType::Struct { fields } = inner.as_ref() else {
        panic!("readings inner must be Struct (Reading record), got {inner:?}")
    };
    assert_eq!(fields.len(), 3, "Reading must have 3 fields");

    // key: string → Utf8
    assert_eq!(fields[0].name, "key");
    assert_eq!(fields[0].logical_type, LogicalType::Utf8);

    // value: double → F64
    assert_eq!(fields[1].name, "value");
    assert_eq!(
        fields[1].logical_type,
        LogicalType::Primitive {
            data_type: DataType::F64
        }
    );

    // quality: ["null", "double"] → Nullable(F64)
    assert_eq!(fields[2].name, "quality");
    let LogicalType::Nullable { inner: q_inner } = &fields[2].logical_type else {
        panic!("quality must be Nullable, got {:?}", fields[2].logical_type)
    };
    assert_eq!(
        **q_inner,
        LogicalType::Primitive {
            data_type: DataType::F64
        },
        "nullable quality inner must be F64"
    );
}

// ---------------------------------------------------------------------------
// Negative tests
// ---------------------------------------------------------------------------

#[test]
fn null_only_union_rejected() {
    let err = lt_err(r#"["null"]"#);
    assert!(
        err.contains("null") || err.contains("usable"),
        "unexpected: {err}"
    );
}

#[test]
fn nested_union_rejected() {
    // A union inside a union is not valid Avro.
    let err = lt_err(r#"["string", ["int", "long"]]"#);
    assert!(
        err.contains("nested") || err.contains("union"),
        "unexpected: {err}"
    );
}

#[test]
fn null_type_outside_union_rejected() {
    let err = lt_err(r#""null""#);
    assert!(
        err.contains("null") || err.contains("union"),
        "unexpected: {err}"
    );
}

#[test]
fn top_level_non_record_rejected_by_avsc_to_schema() {
    let err = schema_err(r#""string""#);
    assert!(
        err.contains("record") || err.contains("object"),
        "unexpected: {err}"
    );
}

#[test]
fn top_level_array_rejected_by_avsc_to_schema() {
    // An Avro union (JSON array) is not a valid top-level for avsc_to_schema.
    let err = schema_err(r#"["string", "int"]"#);
    assert!(
        err.contains("record") || err.contains("object"),
        "unexpected: {err}"
    );
}

#[test]
fn record_missing_fields_rejected() {
    let err = schema_err(r#"{"type": "record", "name": "Bad"}"#);
    assert!(err.contains("fields"), "unexpected: {err}");
}

#[test]
fn array_missing_items_rejected() {
    let err = lt_err(r#"{"type": "array"}"#);
    assert!(err.contains("items"), "unexpected: {err}");
}

#[test]
fn map_missing_values_rejected() {
    let err = lt_err(r#"{"type": "map"}"#);
    assert!(err.contains("values"), "unexpected: {err}");
}

#[test]
fn fixed_missing_size_rejected() {
    let err = lt_err(r#"{"type": "fixed", "name": "Bad"}"#);
    assert!(err.contains("size"), "unexpected: {err}");
}

#[test]
fn enum_missing_symbols_rejected() {
    let err = lt_err(r#"{"type": "enum", "name": "Bad"}"#);
    assert!(err.contains("symbols"), "unexpected: {err}");
}

#[test]
fn error_type_rejected() {
    let err = lt_err(r#"{"type": "error", "name": "Bad", "fields": []}"#);
    assert!(
        err.contains("error") || err.contains("record"),
        "unexpected: {err}"
    );
}

#[test]
fn unknown_type_rejected() {
    let err = lt_err(r#""quaternion""#);
    assert!(
        err.contains("unknown") || err.contains("quaternion"),
        "unexpected: {err}"
    );
}

#[test]
fn unknown_complex_type_rejected() {
    let err = lt_err(r#"{"type": "protocol"}"#);
    assert!(
        err.contains("unknown") || err.contains("protocol"),
        "unexpected: {err}"
    );
}

#[test]
fn recursive_schema_rejected() {
    // A record whose field type references the record's own name should fail.
    let err = lt_err(
        r#"{
        "type": "record",
        "name": "Node",
        "fields": [
            {"name": "value", "type": "int"},
            {"name": "next",  "type": "Node"}
        ]
    }"#,
    );
    assert!(
        err.contains("recursive") || err.contains("Node"),
        "expected recursion error, got: {err}"
    );
}

#[test]
fn depth_limit_exceeded_rejected() {
    // Build a deeply nested array of array of ... exceeding MAX_NESTED_DEPTH (64).
    // We build 66 levels of nesting so it reliably exceeds the cap.
    let mut avsc = String::from(r#"{"type": "array", "items": "#);
    for _ in 0..65 {
        avsc.push_str(r#"{"type": "array", "items": "#);
    }
    avsc.push_str(r#""int""#);
    for _ in 0..=65 {
        avsc.push('}');
    }
    let err = lt_err(&avsc);
    assert!(
        err.contains("depth") || err.contains("64") || err.contains("nesting"),
        "expected depth error, got: {err}"
    );
}

#[test]
fn forward_reference_rejected() {
    // Referencing a named type before it's been defined is not supported.
    let err = schema_err(
        r#"{
        "type": "record",
        "name": "Outer",
        "fields": [
            {"name": "inner", "type": "Inner"},
            {"name": "id",    "type": "int"}
        ]
    }"#,
    );
    // "Inner" is never defined, so it should fail as unknown type.
    assert!(
        err.contains("unknown") || err.contains("Inner"),
        "expected unknown-type error, got: {err}"
    );
}

#[test]
fn invalid_json_rejected() {
    let err = schema_err(r#"{not valid json"#);
    assert!(
        err.contains("invalid") || err.contains("JSON"),
        "unexpected: {err}"
    );
}
