//! Round-trip tests for `Date { unit }` and `Datetime { unit, timezone }` semantic types.
//!
//! Verifies:
//! - Schema JSON serialization preserves the frozen `"kind":"date"` /
//!   `"kind":"datetime"` discriminants plus `unit` and `timezone` fields.
//! - Physical decomposition: Date(Days) → 1×I32 leaf, Date(Millis) → 1×I64
//!   leaf, Datetime → 1×I64 leaf.
//! - Write + read round-trip for all unit variants.
//! - Multi-stripe round-trip.
//! - Edge cases: epoch (0), negative (pre-epoch), large positive values.
//! - `DateUnit` and `TimeUnit` can be imported from the crate root.

use std::io::Cursor;

use helium::{
    CoderRegistry, CoderSpec, ColumnSpec, DateUnit, HeliumReader, HeliumWriter, LogicalColumn,
    LogicalType, Schema, TimeUnit,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn reg() -> CoderRegistry {
    CoderRegistry::default()
}

fn i32_coders() -> Vec<CoderSpec> {
    vec![
        CoderSpec::new("delta"),
        CoderSpec::new("leb128"),
        CoderSpec::new("zstd"),
    ]
}

fn i64_coders() -> Vec<CoderSpec> {
    vec![
        CoderSpec::new("delta"),
        CoderSpec::new("leb128"),
        CoderSpec::new("zstd"),
    ]
}

fn roundtrip_date32(values: Vec<i32>) -> Vec<i32> {
    let spec = ColumnSpec::date32("d", i32_coders());
    let schema = Schema::new(vec![spec]);
    let r = reg();
    let mut buf = Cursor::new(Vec::<u8>::new());
    {
        let mut writer = HeliumWriter::new(&mut buf, schema, &r).unwrap();
        writer
            .write_column(
                "d",
                LogicalColumn::Date32 {
                    values: values.clone(),
                },
            )
            .unwrap();
        writer.finish().unwrap();
    }
    buf.set_position(0);
    match HeliumReader::new(&mut buf, &r)
        .unwrap()
        .read_column("d")
        .unwrap()
    {
        LogicalColumn::Date32 { values } => values,
        other => panic!("expected Date32, got {other:?}"),
    }
}

fn roundtrip_date64(values: Vec<i64>) -> Vec<i64> {
    let spec = ColumnSpec::date64("d", i64_coders());
    let schema = Schema::new(vec![spec]);
    let r = reg();
    let mut buf = Cursor::new(Vec::<u8>::new());
    {
        let mut writer = HeliumWriter::new(&mut buf, schema, &r).unwrap();
        writer
            .write_column(
                "d",
                LogicalColumn::Date64 {
                    values: values.clone(),
                },
            )
            .unwrap();
        writer.finish().unwrap();
    }
    buf.set_position(0);
    match HeliumReader::new(&mut buf, &r)
        .unwrap()
        .read_column("d")
        .unwrap()
    {
        LogicalColumn::Date64 { values } => values,
        other => panic!("expected Date64, got {other:?}"),
    }
}

fn roundtrip_datetime(values: Vec<i64>, unit: TimeUnit, tz: Option<String>) -> Vec<i64> {
    let spec = ColumnSpec::datetime("ts", unit, tz.clone(), i64_coders());
    let schema = Schema::new(vec![spec]);
    let r = reg();
    let mut buf = Cursor::new(Vec::<u8>::new());
    {
        let mut writer = HeliumWriter::new(&mut buf, schema, &r).unwrap();
        writer
            .write_column(
                "ts",
                LogicalColumn::Datetime {
                    values: values.clone(),
                },
            )
            .unwrap();
        writer.finish().unwrap();
    }
    buf.set_position(0);
    match HeliumReader::new(&mut buf, &r)
        .unwrap()
        .read_column("ts")
        .unwrap()
    {
        LogicalColumn::Datetime { values } => values,
        other => panic!("expected Datetime, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Schema JSON / wire-format tests
// ---------------------------------------------------------------------------

#[test]
fn date_days_schema_json() {
    let spec = ColumnSpec::date32("created_at", i32_coders());
    let schema = Schema::new(vec![spec]);
    let json = serde_json::to_string(&schema).unwrap();
    // Wire-format: "kind":"date" (snake_case via rename_all on LogicalType);
    // DateUnit variant names are PascalCase (no rename_all on DateUnit).
    assert!(json.contains(r#""kind":"date""#), "JSON: {json}");
    assert!(json.contains(r#""unit":"Days""#), "JSON: {json}");
    let parsed: Schema = serde_json::from_str(&json).unwrap();
    assert_eq!(
        parsed.columns[0].logical_type,
        LogicalType::Date {
            unit: DateUnit::Days
        }
    );
}

#[test]
fn date_millis_schema_json() {
    let spec = ColumnSpec::date64("created_at", i64_coders());
    let schema = Schema::new(vec![spec]);
    let json = serde_json::to_string(&schema).unwrap();
    assert!(json.contains(r#""kind":"date""#), "JSON: {json}");
    assert!(json.contains(r#""unit":"Millis""#), "JSON: {json}");
    let parsed: Schema = serde_json::from_str(&json).unwrap();
    assert_eq!(
        parsed.columns[0].logical_type,
        LogicalType::Date {
            unit: DateUnit::Millis
        }
    );
}

#[test]
fn datetime_millis_utc_schema_json() {
    let spec = ColumnSpec::datetime(
        "ts",
        TimeUnit::Millis,
        Some("UTC".to_string()),
        i64_coders(),
    );
    let schema = Schema::new(vec![spec]);
    let json = serde_json::to_string(&schema).unwrap();
    assert!(json.contains(r#""kind":"datetime""#), "JSON: {json}");
    // TimeUnit variants are PascalCase (no rename_all on TimeUnit).
    assert!(json.contains(r#""unit":"Millis""#), "JSON: {json}");
    assert!(json.contains(r#""timezone":"UTC""#), "JSON: {json}");
    let parsed: Schema = serde_json::from_str(&json).unwrap();
    assert_eq!(
        parsed.columns[0].logical_type,
        LogicalType::Datetime {
            unit: TimeUnit::Millis,
            timezone: Some("UTC".to_string())
        }
    );
}

#[test]
fn datetime_no_timezone_schema_json() {
    let spec = ColumnSpec::datetime("ts", TimeUnit::Micros, None, i64_coders());
    let schema = Schema::new(vec![spec]);
    let json = serde_json::to_string(&schema).unwrap();
    assert!(json.contains(r#""kind":"datetime""#), "JSON: {json}");
    assert!(json.contains(r#""unit":"Micros""#), "JSON: {json}");
    // timezone is null — just verify round-trip.
    let parsed: Schema = serde_json::from_str(&json).unwrap();
    assert_eq!(
        parsed.columns[0].logical_type,
        LogicalType::Datetime {
            unit: TimeUnit::Micros,
            timezone: None
        }
    );
}

// ---------------------------------------------------------------------------
// Physical leaf tests
// ---------------------------------------------------------------------------

#[test]
fn date_days_has_one_i32_leaf() {
    use helium::DataType;
    let lt = LogicalType::Date {
        unit: DateUnit::Days,
    };
    let fields = lt.physical_fields();
    assert_eq!(
        fields.len(),
        1,
        "Date(Days) must have exactly 1 physical leaf"
    );
    assert_eq!(fields[0].data_type, DataType::I32);
}

#[test]
fn date_millis_has_one_i64_leaf() {
    use helium::DataType;
    let lt = LogicalType::Date {
        unit: DateUnit::Millis,
    };
    let fields = lt.physical_fields();
    assert_eq!(
        fields.len(),
        1,
        "Date(Millis) must have exactly 1 physical leaf"
    );
    assert_eq!(fields[0].data_type, DataType::I64);
}

#[test]
fn datetime_has_one_i64_leaf() {
    use helium::DataType;
    for unit in [
        TimeUnit::Seconds,
        TimeUnit::Millis,
        TimeUnit::Micros,
        TimeUnit::Nanos,
    ] {
        let lt = LogicalType::Datetime {
            unit,
            timezone: None,
        };
        let fields = lt.physical_fields();
        assert_eq!(
            fields.len(),
            1,
            "Datetime must have exactly 1 physical leaf"
        );
        assert_eq!(fields[0].data_type, DataType::I64);
    }
}

// ---------------------------------------------------------------------------
// Date32 round-trip tests
// ---------------------------------------------------------------------------

#[test]
fn date32_epoch() {
    assert_eq!(roundtrip_date32(vec![0]), vec![0i32]);
}

#[test]
fn date32_positive_dates() {
    // 2024-01-01 = 19723 days since epoch
    let values = vec![1i32, 365, 3650, 19723];
    assert_eq!(roundtrip_date32(values.clone()), values);
}

#[test]
fn date32_negative_dates() {
    // Dates before 1970-01-01
    let values = vec![-1i32, -365, -18000];
    assert_eq!(roundtrip_date32(values.clone()), values);
}

#[test]
fn date32_mixed() {
    let values = vec![
        -100i32,
        0,
        100,
        365,
        -365,
        18993,
        i32::MIN / 2,
        i32::MAX / 2,
    ];
    assert_eq!(roundtrip_date32(values.clone()), values);
}

#[test]
fn date32_large_batch() {
    let values: Vec<i32> = (0..5_000i32).map(|i| i * 10 - 25_000).collect();
    assert_eq!(roundtrip_date32(values.clone()), values);
}

// ---------------------------------------------------------------------------
// Date64 round-trip tests
// ---------------------------------------------------------------------------

#[test]
fn date64_epoch() {
    assert_eq!(roundtrip_date64(vec![0]), vec![0i64]);
}

#[test]
fn date64_millis_values() {
    // 2024-01-01 in milliseconds
    let values = vec![86_400_000i64, -86_400_000, 1_704_067_200_000];
    assert_eq!(roundtrip_date64(values.clone()), values);
}

#[test]
fn date64_large_batch() {
    let values: Vec<i64> = (0..5_000i64)
        .map(|i| i * 86_400_000 - 1_000_000_000_000)
        .collect();
    assert_eq!(roundtrip_date64(values.clone()), values);
}

// ---------------------------------------------------------------------------
// Datetime round-trip tests (all 4 units)
// ---------------------------------------------------------------------------

#[test]
fn datetime_millis_basic() {
    // Unix epoch + various timestamps
    let values = vec![0i64, 1_000, 1_704_067_200_000, -1_000_000];
    assert_eq!(
        roundtrip_datetime(values.clone(), TimeUnit::Millis, None),
        values
    );
}

#[test]
fn datetime_millis_utc() {
    let values = vec![0i64, 1_704_067_200_000, -86_400_000];
    assert_eq!(
        roundtrip_datetime(values.clone(), TimeUnit::Millis, Some("UTC".to_string())),
        values
    );
}

#[test]
fn datetime_micros() {
    let values = vec![0i64, 1_704_067_200_000_000, -1_000_000_000];
    assert_eq!(
        roundtrip_datetime(values.clone(), TimeUnit::Micros, None),
        values
    );
}

#[test]
fn datetime_nanos() {
    let values = vec![0i64, 1_704_067_200_000_000_000, -1_000_000_000_000];
    assert_eq!(
        roundtrip_datetime(values.clone(), TimeUnit::Nanos, None),
        values
    );
}

#[test]
fn datetime_seconds() {
    let values = vec![0i64, 1_704_067_200, -86_400, i32::MAX as i64];
    assert_eq!(
        roundtrip_datetime(values.clone(), TimeUnit::Seconds, None),
        values
    );
}

#[test]
fn datetime_timezone_preserved_in_schema() {
    // Verify the timezone string survives JSON round-trip.
    let spec = ColumnSpec::datetime(
        "ts",
        TimeUnit::Millis,
        Some("America/New_York".to_string()),
        i64_coders(),
    );
    let schema = Schema::new(vec![spec]);
    let json = serde_json::to_string(&schema).unwrap();
    let parsed: Schema = serde_json::from_str(&json).unwrap();
    let lt = &parsed.columns[0].logical_type;
    assert_eq!(
        lt,
        &LogicalType::Datetime {
            unit: TimeUnit::Millis,
            timezone: Some("America/New_York".to_string()),
        }
    );
}

#[test]
fn datetime_large_batch() {
    let values: Vec<i64> = (0..5_000i64).map(|i| i * 60_000 - 10_000_000).collect();
    assert_eq!(
        roundtrip_datetime(values.clone(), TimeUnit::Millis, None),
        values
    );
}

// ---------------------------------------------------------------------------
// Multi-stripe round-trip
// ---------------------------------------------------------------------------

#[test]
fn date32_multi_stripe() {
    let spec = ColumnSpec::date32("d", i32_coders());
    let schema = Schema::new(vec![spec]);
    let r = reg();
    let s1 = vec![100i32, 200, 300];
    let s2 = vec![-1i32, -2, 0];

    let mut buf = Cursor::new(Vec::<u8>::new());
    {
        let mut writer = HeliumWriter::new(&mut buf, schema.clone(), &r).unwrap();
        writer
            .write_column("d", LogicalColumn::Date32 { values: s1.clone() })
            .unwrap();
        writer.finish_stripe().unwrap();
        writer
            .write_column("d", LogicalColumn::Date32 { values: s2.clone() })
            .unwrap();
        writer.finish().unwrap();
    }

    buf.set_position(0);
    let mut reader = HeliumReader::new(&mut buf, &r).unwrap();
    assert_eq!(reader.stripe_count(), 2);

    let r1 = match reader.read_column_at_stripe("d", 0).unwrap() {
        LogicalColumn::Date32 { values } => values,
        other => panic!("{other:?}"),
    };
    assert_eq!(r1, s1);
    let r2 = match reader.read_column_at_stripe("d", 1).unwrap() {
        LogicalColumn::Date32 { values } => values,
        other => panic!("{other:?}"),
    };
    assert_eq!(r2, s2);
}

#[test]
fn datetime_multi_stripe() {
    let spec = ColumnSpec::datetime("ts", TimeUnit::Millis, None, i64_coders());
    let schema = Schema::new(vec![spec]);
    let r = reg();
    let s1 = vec![0i64, 1_000, 2_000];
    let s2 = vec![3_000i64, 4_000];

    let mut buf = Cursor::new(Vec::<u8>::new());
    {
        let mut writer = HeliumWriter::new(&mut buf, schema.clone(), &r).unwrap();
        writer
            .write_column("ts", LogicalColumn::Datetime { values: s1.clone() })
            .unwrap();
        writer.finish_stripe().unwrap();
        writer
            .write_column("ts", LogicalColumn::Datetime { values: s2.clone() })
            .unwrap();
        writer.finish().unwrap();
    }

    buf.set_position(0);
    let mut reader = HeliumReader::new(&mut buf, &r).unwrap();
    assert_eq!(reader.stripe_count(), 2);

    let r1 = match reader.read_column_at_stripe("ts", 0).unwrap() {
        LogicalColumn::Datetime { values } => values,
        other => panic!("{other:?}"),
    };
    assert_eq!(r1, s1);
    let r2 = match reader.read_column_at_stripe("ts", 1).unwrap() {
        LogicalColumn::Datetime { values } => values,
        other => panic!("{other:?}"),
    };
    assert_eq!(r2, s2);
}
