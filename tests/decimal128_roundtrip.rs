//! Round-trip tests for `Decimal128 { precision, scale }` semantic type.
//!
//! Verifies:
//! - Schema JSON serialization / deserialization preserves the `"kind":
//!   "Decimal128"` tag and `precision` / `scale` fields (wire-format frozen).
//! - Physical decomposition into two I64 leaves (high / low).
//! - Write + read round-trip via `HeliumWriter` / `HeliumReader`.
//! - Multi-stripe round-trip.
//! - Edge cases: zero, `i128::MIN`, `i128::MAX`, negative values.
//! - Scale=0 and non-zero scale are both preserved (scale is metadata only).

use std::io::Cursor;

use helium::{
    CoderRegistry, CoderSpec, ColumnSpec, HeliumReader, HeliumWriter, LogicalColumn, LogicalType,
    Schema,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn reg() -> CoderRegistry {
    CoderRegistry::default()
}

/// Default I64 pipeline: delta → leb128 → zstd.
fn i64_coders() -> Vec<CoderSpec> {
    vec![
        CoderSpec::new("delta"),
        CoderSpec::new("leb128"),
        CoderSpec::new("zstd"),
    ]
}

fn make_schema(precision: u8, scale: u8) -> Schema {
    // decimal128() takes (name, precision, scale, high_enc, low_enc)
    let spec = ColumnSpec::decimal128("amount", precision, scale, i64_coders(), i64_coders());
    Schema::new(vec![spec])
}

fn roundtrip(values: Vec<i128>, precision: u8, scale: u8) -> Vec<i128> {
    let schema = make_schema(precision, scale);
    let r = reg();
    let mut buf = Cursor::new(Vec::<u8>::new());
    {
        let mut writer = HeliumWriter::new(&mut buf, schema.clone(), &r).unwrap();
        writer
            .write_column(
                "amount",
                LogicalColumn::Decimal128 {
                    values: values.clone(),
                },
            )
            .unwrap();
        writer.finish().unwrap();
    }
    buf.set_position(0);
    let mut reader = HeliumReader::new(&mut buf, &r).unwrap();
    match reader.read_column("amount").unwrap() {
        LogicalColumn::Decimal128 { values } => values,
        other => panic!("expected Decimal128, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Wire-format / schema JSON tests
// ---------------------------------------------------------------------------

#[test]
fn schema_json_roundtrip_decimal128() {
    let schema = make_schema(18, 6);
    let json = serde_json::to_string(&schema).unwrap();
    // The `"kind":"decimal128"` tag (snake_case via rename_all) is wire-format frozen.
    assert!(
        json.contains(r#""kind":"decimal128""#),
        "schema JSON: {json}"
    );
    assert!(json.contains(r#""precision":18"#), "schema JSON: {json}");
    assert!(json.contains(r#""scale":6"#), "schema JSON: {json}");
    let parsed: Schema = serde_json::from_str(&json).unwrap();
    assert_eq!(
        parsed.columns[0].logical_type,
        schema.columns[0].logical_type
    );
}

#[test]
fn physical_fields_are_two_i64_leaves() {
    use helium::DataType;
    let lt = LogicalType::Decimal128 {
        precision: 10,
        scale: 3,
    };
    let fields = lt.physical_fields();
    assert_eq!(
        fields.len(),
        2,
        "Decimal128 must have exactly 2 physical leaves"
    );
    assert_eq!(fields[0].role, "high");
    assert_eq!(fields[0].data_type, DataType::I64);
    assert_eq!(fields[1].role, "low");
    assert_eq!(fields[1].data_type, DataType::I64);
}

// ---------------------------------------------------------------------------
// Value round-trip tests
// ---------------------------------------------------------------------------

#[test]
fn decimal128_zero() {
    let result = roundtrip(vec![0], 10, 0);
    assert_eq!(result, vec![0i128]);
}

#[test]
fn decimal128_positive_integers() {
    let values = vec![1i128, 42, 1_000_000, 999_999_999_999_999_999];
    assert_eq!(roundtrip(values.clone(), 18, 0), values);
}

#[test]
fn decimal128_negative_values() {
    let values = vec![-1i128, -42, -1_000_000, -999_999_999_999_999_999];
    assert_eq!(roundtrip(values.clone(), 18, 0), values);
}

#[test]
fn decimal128_mixed_sign() {
    let values = vec![-100i128, 0, 100, -1, 1, i64::MIN as i128, i64::MAX as i128];
    assert_eq!(roundtrip(values.clone(), 19, 0), values);
}

#[test]
fn decimal128_max_value() {
    let values = vec![i128::MAX];
    assert_eq!(roundtrip(values.clone(), 38, 0), values);
}

#[test]
fn decimal128_min_value() {
    let values = vec![i128::MIN];
    assert_eq!(roundtrip(values.clone(), 38, 0), values);
}

#[test]
fn decimal128_with_nonzero_scale() {
    // Scale is metadata — the stored unscaled integer is what's preserved.
    let values = vec![12345i128, -98765, 100000];
    // scale=2 means 123.45, -987.65, 1000.00 — but stored as integers
    assert_eq!(roundtrip(values.clone(), 10, 2), values);
}

#[test]
fn decimal128_high_bits_preserved() {
    // Values that set bits in both high and low words of the i128.
    // High word: (v >> 64), low word: v as i64 (sign-preserving).
    let a: i128 = (1i128 << 65) | 7; // high word = 2, low word = 7
    let b: i128 = -((1i128 << 65) | 7); // negative of same
    let c: i128 = (1i128 << 64) - 1; // all bits in low word
    let values = vec![a, b, c, i128::MAX, i128::MIN];
    assert_eq!(roundtrip(values.clone(), 38, 0), values);
}

// ---------------------------------------------------------------------------
// Multi-stripe round-trip
// ---------------------------------------------------------------------------

#[test]
fn decimal128_multi_stripe() {
    let schema = make_schema(18, 4);
    let r = reg();
    let stripe1 = vec![100i128, 200, 300];
    let stripe2 = vec![-1i128, -2, -3, 0];

    let mut buf = Cursor::new(Vec::<u8>::new());
    {
        let mut writer = HeliumWriter::new(&mut buf, schema.clone(), &r).unwrap();
        writer
            .write_column(
                "amount",
                LogicalColumn::Decimal128 {
                    values: stripe1.clone(),
                },
            )
            .unwrap();
        writer.finish_stripe().unwrap();
        writer
            .write_column(
                "amount",
                LogicalColumn::Decimal128 {
                    values: stripe2.clone(),
                },
            )
            .unwrap();
        writer.finish().unwrap();
    }

    buf.set_position(0);
    let mut reader = HeliumReader::new(&mut buf, &r).unwrap();
    assert_eq!(reader.stripe_count(), 2);

    let s1 = match reader.read_column_at_stripe("amount", 0).unwrap() {
        LogicalColumn::Decimal128 { values } => values,
        other => panic!("expected Decimal128, got {other:?}"),
    };
    assert_eq!(s1, stripe1);

    let s2 = match reader.read_column_at_stripe("amount", 1).unwrap() {
        LogicalColumn::Decimal128 { values } => values,
        other => panic!("expected Decimal128, got {other:?}"),
    };
    assert_eq!(s2, stripe2);
}

// ---------------------------------------------------------------------------
// Large batch (stress test for pipeline)
// ---------------------------------------------------------------------------

#[test]
fn decimal128_large_batch() {
    let n = 10_000usize;
    let values: Vec<i128> = (0..n as i128)
        .map(|i| if i % 3 == 0 { -i } else { i * 1000 })
        .collect();
    let result = roundtrip(values.clone(), 28, 6);
    assert_eq!(result, values);
}
