//! Cross-format semantic-type mapping tests.
//!
//! Verifies that the Avro, Parquet, and Arrow format adapters all translate
//! `Decimal128`, `Date`, and `Datetime` representations into the correct
//! Helium `LogicalType` variants.  The Helium end-to-end round-trip for each
//! semantic type is covered in `decimal128_roundtrip.rs` and
//! `date_datetime_roundtrip.rs`; this file focuses on the adapter boundaries.
//!
//! Feature-gated sections:
//! - Avro tests: `schema-avro`
//! - Parquet tests: `schema-parquet`
//! - Arrow tests: `arrow`

use helium::{DateUnit, LogicalType, TimeUnit};

// ---------------------------------------------------------------------------
// § Avro adapter
// ---------------------------------------------------------------------------

#[cfg(feature = "schema-avro")]
mod avro_semantic {
    use super::*;
    use helium::schema::avro::avsc_to_logical_type;

    fn lt(avsc: &str) -> LogicalType {
        avsc_to_logical_type(avsc).unwrap_or_else(|e| panic!("parse failed: {e}"))
    }

    // Avro `date` over `int` → Date { Days }
    #[test]
    fn avro_date_over_int_maps_to_date_days() {
        let avsc = r#"{"type": "int", "logicalType": "date"}"#;
        assert_eq!(
            lt(avsc),
            LogicalType::Date {
                unit: DateUnit::Days
            }
        );
    }

    // Avro `timestamp-millis` over `long` → Datetime { Millis, None }
    #[test]
    fn avro_timestamp_millis_maps_to_datetime() {
        let avsc = r#"{"type": "long", "logicalType": "timestamp-millis"}"#;
        assert_eq!(
            lt(avsc),
            LogicalType::Datetime {
                unit: TimeUnit::Millis,
                timezone: None
            }
        );
    }

    // Avro `timestamp-micros` over `long` → Datetime { Micros, None }
    #[test]
    fn avro_timestamp_micros_maps_to_datetime() {
        let avsc = r#"{"type": "long", "logicalType": "timestamp-micros"}"#;
        assert_eq!(
            lt(avsc),
            LogicalType::Datetime {
                unit: TimeUnit::Micros,
                timezone: None
            }
        );
    }

    // Avro `local-timestamp-millis` → Datetime { Millis, None } (no timezone)
    #[test]
    fn avro_local_timestamp_millis_maps_to_datetime_no_tz() {
        let avsc = r#"{"type": "long", "logicalType": "local-timestamp-millis"}"#;
        let result = lt(avsc);
        assert!(
            matches!(
                result,
                LogicalType::Datetime {
                    unit: TimeUnit::Millis,
                    timezone: None
                }
            ),
            "expected Datetime(Millis, None), got {result:?}"
        );
    }

    // Avro `decimal` over `bytes` → Decimal128 { precision, scale }
    #[test]
    fn avro_decimal_bytes_maps_to_decimal128() {
        let avsc = r#"{"type": "bytes", "logicalType": "decimal", "precision": 18, "scale": 5}"#;
        assert_eq!(
            lt(avsc),
            LogicalType::Decimal128 {
                precision: 18,
                scale: 5
            }
        );
    }

    // Avro `decimal` with scale=0 → Decimal128 { precision, scale=0 }
    #[test]
    fn avro_decimal_scale_zero() {
        let avsc = r#"{"type": "bytes", "logicalType": "decimal", "precision": 10, "scale": 0}"#;
        assert_eq!(
            lt(avsc),
            LogicalType::Decimal128 {
                precision: 10,
                scale: 0
            }
        );
    }

    // Avro `decimal` with default scale (missing) → treated as scale=0
    #[test]
    fn avro_decimal_missing_scale_defaults_to_zero() {
        let avsc = r#"{"type": "bytes", "logicalType": "decimal", "precision": 10}"#;
        // scale defaults to 0 per Avro spec
        let result = lt(avsc);
        assert!(
            matches!(
                result,
                LogicalType::Decimal128 {
                    precision: 10,
                    scale: 0
                }
            ),
            "expected Decimal128(10, 0), got {result:?}"
        );
    }

    // Avro `decimal` over `string` (fixed) → Decimal128 when precision is present
    #[test]
    fn avro_decimal_fixed_maps_to_decimal128() {
        let avsc = r#"{
            "type": "fixed",
            "size": 8,
            "name": "amount",
            "logicalType": "decimal",
            "precision": 15,
            "scale": 3
        }"#;
        assert_eq!(
            lt(avsc),
            LogicalType::Decimal128 {
                precision: 15,
                scale: 3
            }
        );
    }

    // Record with mixed semantic types
    #[test]
    fn avro_record_with_date_and_decimal() {
        let avsc = r#"{
            "type": "record",
            "name": "Order",
            "fields": [
                {"name": "id", "type": "string"},
                {
                    "name": "amount",
                    "type": {
                        "type": "bytes",
                        "logicalType": "decimal",
                        "precision": 12,
                        "scale": 2
                    }
                },
                {
                    "name": "created_at",
                    "type": {"type": "int", "logicalType": "date"}
                },
                {
                    "name": "updated_at",
                    "type": {"type": "long", "logicalType": "timestamp-millis"}
                }
            ]
        }"#;
        use helium::schema::avro::avsc_to_schema;
        let schema = avsc_to_schema(avsc).unwrap();
        assert_eq!(
            schema.column("amount").unwrap().logical_type,
            LogicalType::Decimal128 {
                precision: 12,
                scale: 2
            }
        );
        assert_eq!(
            schema.column("created_at").unwrap().logical_type,
            LogicalType::Date {
                unit: DateUnit::Days
            }
        );
        assert_eq!(
            schema.column("updated_at").unwrap().logical_type,
            LogicalType::Datetime {
                unit: TimeUnit::Millis,
                timezone: None
            }
        );
    }

    // Avro write + read OCF round-trip for Date column (uses temp file)
    #[test]
    fn avro_ocf_date_roundtrip() {
        use helium::LogicalColumn;
        use helium::schema::avro::{read_avro_data, write_avro_data};
        use tempfile::NamedTempFile;

        let schema_str = r#"{
            "type": "record",
            "name": "Row",
            "fields": [
                {"name": "d", "type": {"type": "int", "logicalType": "date"}}
            ]
        }"#;

        let helium_schema = helium::schema::avro::avsc_to_schema(schema_str).unwrap();

        // Write as Avro OCF to a temp file
        let date_values = vec![0i32, 18993, -365, 19723];
        let cols = {
            let mut m = std::collections::HashMap::new();
            m.insert(
                "d".to_string(),
                LogicalColumn::Date32 {
                    values: date_values.clone(),
                },
            );
            m
        };
        let tmp = NamedTempFile::new().unwrap();
        write_avro_data(tmp.path(), &helium_schema, &cols).unwrap();

        // Read back as Avro OCF
        let (_, result) = read_avro_data(tmp.path()).unwrap();
        let d_col = result.get("d").unwrap();
        let read_back = match d_col {
            LogicalColumn::Date32 { values } => values.clone(),
            other => panic!("expected Date32, got {other:?}"),
        };
        assert_eq!(read_back, date_values);
    }
}

// ---------------------------------------------------------------------------
// § Parquet adapter
// ---------------------------------------------------------------------------

#[cfg(feature = "schema-parquet")]
mod parquet_semantic {
    use super::*;
    use helium::schema::parquet::schema_from_parquet_schema;
    use parquet::schema::parser::parse_message_type;

    fn helium_from_str(s: &str) -> helium::Schema {
        let pq = parse_message_type(s).unwrap();
        schema_from_parquet_schema(&pq).unwrap()
    }

    // Parquet `DATE` annotation on int32 → Date { Days }
    #[test]
    fn parquet_date_annotation_maps_to_date_days() {
        let schema = helium_from_str("message m { required int32 d (DATE); }");
        assert_eq!(
            schema.columns[0].logical_type,
            LogicalType::Date {
                unit: DateUnit::Days
            }
        );
    }

    // Parquet `TIMESTAMP_MILLIS` annotation → Datetime { Millis, .. }
    #[test]
    fn parquet_timestamp_millis_maps_to_datetime() {
        let schema = helium_from_str("message m { required int64 ts (TIMESTAMP_MILLIS); }");
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

    // Parquet logical `Timestamp(MICROS, adjusted=true)` → Datetime { Micros, Some("UTC") }
    #[test]
    fn parquet_logical_timestamp_micros_utc() {
        use parquet::basic::{LogicalType as PqLogical, Repetition, TimeUnit as PqTu};
        use parquet::schema::types::Type;
        use std::sync::Arc;

        // Build schema manually for logical-type annotation (parquet-rs API)
        let field = Type::primitive_type_builder("ts", parquet::basic::Type::INT64)
            .with_repetition(Repetition::REQUIRED)
            .with_logical_type(Some(PqLogical::Timestamp {
                // PqTu::MICROS is a unit struct constant, not a function call.
                unit: PqTu::MICROS,
                is_adjusted_to_u_t_c: true,
            }))
            .build()
            .unwrap();
        let message = Type::group_type_builder("m")
            .with_fields(vec![Arc::new(field)])
            .build()
            .unwrap();
        let schema = schema_from_parquet_schema(&message).unwrap();
        assert_eq!(
            schema.columns[0].logical_type,
            LogicalType::Datetime {
                unit: TimeUnit::Micros,
                timezone: Some("UTC".to_string()),
            }
        );
    }

    // Parquet logical `Timestamp(NANOS, adjusted=false)` → Datetime { Nanos, None }
    #[test]
    fn parquet_logical_timestamp_nanos_no_tz() {
        use parquet::basic::{LogicalType as PqLogical, Repetition, TimeUnit as PqTu};
        use parquet::schema::types::Type;
        use std::sync::Arc;

        let field = Type::primitive_type_builder("ts", parquet::basic::Type::INT64)
            .with_repetition(Repetition::REQUIRED)
            .with_logical_type(Some(PqLogical::Timestamp {
                unit: PqTu::NANOS,
                is_adjusted_to_u_t_c: false,
            }))
            .build()
            .unwrap();
        let message = Type::group_type_builder("m")
            .with_fields(vec![Arc::new(field)])
            .build()
            .unwrap();
        let schema = schema_from_parquet_schema(&message).unwrap();
        assert_eq!(
            schema.columns[0].logical_type,
            LogicalType::Datetime {
                unit: TimeUnit::Nanos,
                timezone: None
            }
        );
    }

    // Parquet `Decimal` over INT32 → Decimal128 { precision, scale }
    #[test]
    fn parquet_decimal_int32_maps_to_decimal128() {
        use parquet::basic::{LogicalType as PqLogical, Repetition};
        use parquet::schema::types::Type;
        use std::sync::Arc;

        // The parquet crate requires that `with_precision` / `with_scale` match
        // the logical type's precision and scale.
        let field = Type::primitive_type_builder("d", parquet::basic::Type::INT32)
            .with_repetition(Repetition::REQUIRED)
            .with_precision(9)
            .with_scale(2)
            .with_logical_type(Some(PqLogical::Decimal {
                precision: 9,
                scale: 2,
            }))
            .build()
            .unwrap();
        let message = Type::group_type_builder("m")
            .with_fields(vec![Arc::new(field)])
            .build()
            .unwrap();
        let schema = schema_from_parquet_schema(&message).unwrap();
        assert_eq!(
            schema.columns[0].logical_type,
            LogicalType::Decimal128 {
                precision: 9,
                scale: 2
            }
        );
    }

    // Parquet `Decimal` over INT64 → Decimal128
    #[test]
    fn parquet_decimal_int64_maps_to_decimal128() {
        use parquet::basic::{LogicalType as PqLogical, Repetition};
        use parquet::schema::types::Type;
        use std::sync::Arc;

        let field = Type::primitive_type_builder("d", parquet::basic::Type::INT64)
            .with_repetition(Repetition::REQUIRED)
            .with_precision(18)
            .with_scale(6)
            .with_logical_type(Some(PqLogical::Decimal {
                precision: 18,
                scale: 6,
            }))
            .build()
            .unwrap();
        let message = Type::group_type_builder("m")
            .with_fields(vec![Arc::new(field)])
            .build()
            .unwrap();
        let schema = schema_from_parquet_schema(&message).unwrap();
        assert_eq!(
            schema.columns[0].logical_type,
            LogicalType::Decimal128 {
                precision: 18,
                scale: 6
            }
        );
    }
}

// ---------------------------------------------------------------------------
// § Arrow adapter
// ---------------------------------------------------------------------------

#[cfg(feature = "arrow")]
mod arrow_semantic {
    use super::*;
    use arrow::datatypes::{
        DataType as ArrowDT, Field, Schema as ArrowSchema, TimeUnit as ArrowTU,
    };
    use helium::arrow::schema::schema_from_arrow;

    /// Build a one-field Arrow schema and convert to Helium, returning the
    /// resulting `LogicalType` of the single column.
    fn helium_lt_from_arrow(dt: ArrowDT) -> Result<LogicalType, helium::HeliumError> {
        let arrow_schema = ArrowSchema::new(vec![Field::new("x", dt, false)]);
        let helium_schema = schema_from_arrow(&arrow_schema)?;
        Ok(helium_schema.columns[0].logical_type.clone())
    }

    #[test]
    fn arrow_decimal128_maps_to_decimal128() {
        let result = helium_lt_from_arrow(ArrowDT::Decimal128(10, 3)).unwrap();
        assert_eq!(
            result,
            LogicalType::Decimal128 {
                precision: 10,
                scale: 3
            }
        );
    }

    #[test]
    fn arrow_date32_maps_to_date_days() {
        let result = helium_lt_from_arrow(ArrowDT::Date32).unwrap();
        assert_eq!(
            result,
            LogicalType::Date {
                unit: DateUnit::Days
            }
        );
    }

    #[test]
    fn arrow_date64_maps_to_date_millis() {
        let result = helium_lt_from_arrow(ArrowDT::Date64).unwrap();
        assert_eq!(
            result,
            LogicalType::Date {
                unit: DateUnit::Millis
            }
        );
    }

    #[test]
    fn arrow_timestamp_second_maps_to_datetime_seconds() {
        let result = helium_lt_from_arrow(ArrowDT::Timestamp(ArrowTU::Second, None)).unwrap();
        assert_eq!(
            result,
            LogicalType::Datetime {
                unit: TimeUnit::Seconds,
                timezone: None
            }
        );
    }

    #[test]
    fn arrow_timestamp_millis_utc_maps_to_datetime() {
        use std::sync::Arc;
        let tz = Some(Arc::from("UTC"));
        let result = helium_lt_from_arrow(ArrowDT::Timestamp(ArrowTU::Millisecond, tz)).unwrap();
        assert_eq!(
            result,
            LogicalType::Datetime {
                unit: TimeUnit::Millis,
                timezone: Some("UTC".to_string())
            }
        );
    }

    #[test]
    fn arrow_timestamp_micros_maps_to_datetime() {
        let result = helium_lt_from_arrow(ArrowDT::Timestamp(ArrowTU::Microsecond, None)).unwrap();
        assert_eq!(
            result,
            LogicalType::Datetime {
                unit: TimeUnit::Micros,
                timezone: None
            }
        );
    }

    #[test]
    fn arrow_timestamp_nanos_maps_to_datetime() {
        let result = helium_lt_from_arrow(ArrowDT::Timestamp(ArrowTU::Nanosecond, None)).unwrap();
        assert_eq!(
            result,
            LogicalType::Datetime {
                unit: TimeUnit::Nanos,
                timezone: None
            }
        );
    }

    // Arrow negative scale → error (Helium scale is u8, can't represent negative)
    #[test]
    fn arrow_decimal128_negative_scale_is_error() {
        // Arrow scale is i8; Helium scale is u8; negative scale is invalid.
        let result = helium_lt_from_arrow(ArrowDT::Decimal128(10, -1));
        assert!(
            result.is_err(),
            "negative Arrow Decimal128 scale must error"
        );
    }
}

// ---------------------------------------------------------------------------
// § Schema JSON round-trip for all three types in one schema
// ---------------------------------------------------------------------------

#[test]
fn combined_schema_json_round_trip() {
    use helium::{CoderSpec, ColumnSpec, Schema};

    fn i64_coders() -> Vec<CoderSpec> {
        vec![
            CoderSpec::new("delta"),
            CoderSpec::new("leb128"),
            CoderSpec::new("zstd"),
        ]
    }
    fn i32_coders() -> Vec<CoderSpec> {
        vec![
            CoderSpec::new("delta"),
            CoderSpec::new("leb128"),
            CoderSpec::new("zstd"),
        ]
    }

    let schema = Schema::new(vec![
        ColumnSpec::decimal128("amount", 18, 4, i64_coders(), i64_coders()),
        ColumnSpec::date32("created_on", i32_coders()),
        ColumnSpec::date64("due_on_ms", i64_coders()),
        ColumnSpec::datetime(
            "updated_at",
            TimeUnit::Millis,
            Some("UTC".to_string()),
            i64_coders(),
        ),
        ColumnSpec::datetime("local_ts", TimeUnit::Micros, None, i64_coders()),
    ]);

    let json = serde_json::to_string(&schema).unwrap();
    let parsed: Schema = serde_json::from_str(&json).unwrap();

    assert_eq!(parsed.columns.len(), 5);
    assert_eq!(
        parsed.columns[0].logical_type,
        LogicalType::Decimal128 {
            precision: 18,
            scale: 4
        }
    );
    assert_eq!(
        parsed.columns[1].logical_type,
        LogicalType::Date {
            unit: DateUnit::Days
        }
    );
    assert_eq!(
        parsed.columns[2].logical_type,
        LogicalType::Date {
            unit: DateUnit::Millis
        }
    );
    assert_eq!(
        parsed.columns[3].logical_type,
        LogicalType::Datetime {
            unit: TimeUnit::Millis,
            timezone: Some("UTC".to_string())
        }
    );
    assert_eq!(
        parsed.columns[4].logical_type,
        LogicalType::Datetime {
            unit: TimeUnit::Micros,
            timezone: None
        }
    );
}
