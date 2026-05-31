//! Bencher — schema-overhead corpus report.
//!
//! Measures `Schema::to_json()` byte count (raw) vs `zstd::encode_all(raw, 3)`
//! byte count (compressed, as stored on disk) across a representative corpus.
//!
//! No data round-trip needed — this is a schema-only measurement.
//!
//! Run:
//!   cargo test --test schema_overhead --release -- --nocapture

use helium::{CoderSpec, ColumnSpec, DataType, FieldSpec, LogicalType, Schema};

// ---------------------------------------------------------------------------
// Coder shorthands
// ---------------------------------------------------------------------------

fn zstd() -> CoderSpec {
    CoderSpec::new("zstd")
}
fn leb_zstd() -> Vec<CoderSpec> {
    vec![CoderSpec::new("leb128"), zstd()]
}
fn delta_leb_zstd() -> Vec<CoderSpec> {
    vec![CoderSpec::new("delta"), CoderSpec::new("leb128"), zstd()]
}
fn zstd_only() -> Vec<CoderSpec> {
    vec![zstd()]
}
fn gorilla_zstd() -> Vec<CoderSpec> {
    vec![CoderSpec::new("gorilla"), zstd()]
}

// ---------------------------------------------------------------------------
// Schema builders — one per shape
// ---------------------------------------------------------------------------

/// Shape A — Server Log (recursive form).
/// Struct { ts:I64, user_id:I64, level:Utf8, message:Utf8,
///          score:Nullable(F64), tags:List(Utf8) }
fn schema_server_log() -> Schema {
    Schema::new(vec![ColumnSpec::struct_col(
        "log",
        vec![
            FieldSpec::primitive("ts", DataType::I64, delta_leb_zstd()),
            FieldSpec::primitive("user_id", DataType::I64, delta_leb_zstd()),
            FieldSpec::utf8("level", delta_leb_zstd(), zstd_only()),
            FieldSpec::utf8("message", delta_leb_zstd(), zstd_only()),
            FieldSpec::nullable(
                "score",
                LogicalType::Primitive {
                    data_type: DataType::F64,
                },
                vec![leb_zstd(), gorilla_zstd()],
            ),
            FieldSpec::list(
                "tags",
                LogicalType::Utf8,
                vec![delta_leb_zstd(), delta_leb_zstd(), zstd_only()],
            ),
        ],
    )])
}

/// Shape B — Transaction Event (recursive).
/// Struct { id:Utf8, amount:Binary, currency:Utf8,
///          parent_id:Nullable(Utf8), metadata:List(Utf8) }
fn schema_transaction_event() -> Schema {
    Schema::new(vec![ColumnSpec::struct_col(
        "tx",
        vec![
            FieldSpec::utf8("id", delta_leb_zstd(), zstd_only()),
            FieldSpec::new(
                "amount",
                LogicalType::Binary,
                vec![delta_leb_zstd(), zstd_only()],
            ),
            FieldSpec::utf8("currency", delta_leb_zstd(), zstd_only()),
            // Nullable(Utf8): [present, item.offsets, item.data]
            FieldSpec::nullable(
                "parent_id",
                LogicalType::Utf8,
                vec![leb_zstd(), delta_leb_zstd(), zstd_only()],
            ),
            // List(Utf8): [offsets, item.offsets, item.data]
            FieldSpec::list(
                "metadata",
                LogicalType::Utf8,
                vec![delta_leb_zstd(), delta_leb_zstd(), zstd_only()],
            ),
        ],
    )])
}

/// Shape C — Sensor Reading (recursive).
/// Struct { device_id:Utf8, readings:List<Struct{key:Utf8, value:F64}> }
fn schema_sensor_reading() -> Schema {
    let inner = LogicalType::Struct {
        fields: vec![
            FieldSpec::utf8("key", delta_leb_zstd(), zstd_only()),
            FieldSpec::primitive("value", DataType::F64, gorilla_zstd()),
        ],
    };
    Schema::new(vec![ColumnSpec::struct_col(
        "sensors",
        vec![
            FieldSpec::utf8("device_id", delta_leb_zstd(), zstd_only()),
            // List<Struct>: 1 encoding (outer offsets); struct fields use FieldSpec
            FieldSpec::list("readings", inner, vec![delta_leb_zstd()]),
        ],
    )])
}

/// Shape D — String-Heavy MR Record (recursive).
/// Struct { request_id, method, path, status_text, tags:List(Utf8), notes }
fn schema_string_heavy_mr() -> Schema {
    Schema::new(vec![ColumnSpec::struct_col(
        "mr",
        vec![
            FieldSpec::utf8("request_id", delta_leb_zstd(), zstd_only()),
            FieldSpec::utf8("method", delta_leb_zstd(), zstd_only()),
            FieldSpec::utf8("path", delta_leb_zstd(), zstd_only()),
            FieldSpec::utf8("status_text", delta_leb_zstd(), zstd_only()),
            FieldSpec::list(
                "tags",
                LogicalType::Utf8,
                vec![delta_leb_zstd(), delta_leb_zstd(), zstd_only()],
            ),
            FieldSpec::utf8("notes", delta_leb_zstd(), zstd_only()),
        ],
    )])
}

/// Shape E — Builder sanity shape (8 flat columns, the shape that produced
/// the 80.3% headline). Reproduced exactly from the file-format mode tests so
/// readers can cross-check.
fn schema_builder_sanity_8col() -> Schema {
    Schema::new(vec![
        ColumnSpec::primitive("ts", DataType::I64, delta_leb_zstd()),
        ColumnSpec::primitive("user_id", DataType::I64, delta_leb_zstd()),
        ColumnSpec::utf8("level", delta_leb_zstd(), zstd_only()),
        ColumnSpec::utf8("message", delta_leb_zstd(), zstd_only()),
        ColumnSpec::nullable(
            "weight",
            LogicalType::Primitive {
                data_type: DataType::F64,
            },
            vec![leb_zstd(), gorilla_zstd()],
        ),
        ColumnSpec::list(
            "tags",
            LogicalType::Utf8,
            vec![delta_leb_zstd(), delta_leb_zstd(), zstd_only()],
        ),
        ColumnSpec::struct_col(
            "addr",
            vec![
                FieldSpec::utf8("street", delta_leb_zstd(), zstd_only()),
                FieldSpec::utf8("city", delta_leb_zstd(), zstd_only()),
                FieldSpec::primitive("zip", DataType::I32, delta_leb_zstd()),
            ],
        ),
        ColumnSpec::new(
            "status",
            LogicalType::Dictionary {
                inner: Box::new(LogicalType::Utf8),
            },
            vec![delta_leb_zstd(), zstd_only(), delta_leb_zstd()],
        ),
    ])
}

/// Shape F — Wide flat schema: 50 columns of mixed primitive/utf8 types.
/// Worst-case for schema verbosity; lots of per-column boilerplate JSON
/// that is highly repetitive → very favourable for zstd.
fn schema_wide_flat_50col() -> Schema {
    let mut cols = Vec::with_capacity(50);
    let types: &[(DataType, &str)] = &[
        (DataType::I32, "i32"),
        (DataType::I64, "i64"),
        (DataType::F32, "f32"),
        (DataType::F64, "f64"),
    ];
    // 10 columns each of I32, I64, F32, F64
    for (dt, suffix) in types {
        let coders = match dt {
            DataType::F32 | DataType::F64 => gorilla_zstd(),
            _ => delta_leb_zstd(),
        };
        for idx in 0..10usize {
            cols.push(ColumnSpec::primitive(
                format!("col_{suffix}_{idx:02}"),
                *dt,
                coders.clone(),
            ));
        }
    }
    // 10 Utf8 columns
    for idx in 0..10usize {
        cols.push(ColumnSpec::utf8(
            format!("col_utf8_{idx:02}"),
            delta_leb_zstd(),
            zstd_only(),
        ));
    }
    Schema::new(cols)
}

/// Shape G — Deep nesting: 5 container levels.
/// Struct { rows: List<Struct { metric: Map<Utf8, List<Nullable<F64>>> }> }
///
/// expected_encodings_len:
///   Nullable<F64>:                   2  [present, values]
///   List<Nullable<F64>>:         1+2=3  [offsets, present, values]
///   Map<Utf8, List<Nullable<F64>>: 1+2+3=6  [map_off, key.off, key.data, lst.off, pres, f64]
///   Struct { metric: Map<...> }:     0  (struct; FieldSpec "metric" holds 6)
///   List<inner_struct>:          1+0=1  [outer offsets only]
///   Struct "deep":                   0  (struct; FieldSpec "rows" holds 1)
fn schema_deep_nesting() -> Schema {
    // Map<Utf8, List<Nullable<F64>>>:
    // encodings[0] = map offsets
    // encodings[1] = key.offsets   (Utf8 leaf 1)
    // encodings[2] = key.data      (Utf8 leaf 2)
    // encodings[3] = list offsets  (List leaf 1)
    // encodings[4] = present       (Nullable leaf 1)
    // encodings[5] = f64 values    (F64 leaf 1)
    let map_type = LogicalType::Map {
        key: Box::new(LogicalType::Utf8),
        value: Box::new(LogicalType::List {
            inner: Box::new(LogicalType::Nullable {
                inner: Box::new(LogicalType::Primitive {
                    data_type: DataType::F64,
                }),
            }),
        }),
    };

    let inner_struct = LogicalType::Struct {
        fields: vec![FieldSpec::new(
            "metric",
            map_type,
            vec![
                delta_leb_zstd(), // map offsets
                delta_leb_zstd(), // key.offsets
                zstd_only(),      // key.data
                delta_leb_zstd(), // list offsets
                leb_zstd(),       // present
                gorilla_zstd(),   // f64 values
            ],
        )],
    };

    Schema::new(vec![ColumnSpec::struct_col(
        "deep",
        vec![
            // List<Struct>: 1 encoding (outer offsets); struct FieldSpec holds the rest
            FieldSpec::list("rows", inner_struct, vec![delta_leb_zstd()]),
        ],
    )])
}

/// Shape H — Real Avro-shape analytics event record.
/// A representative commerce/analytics event with nested line-items list,
/// nullable fields, and multiple string columns — the kind of schema
/// produced by Avro → Helium translation.
fn schema_avro_analytics_event() -> Schema {
    // Line item inner struct
    let line_item = LogicalType::Struct {
        fields: vec![
            FieldSpec::utf8("sku", delta_leb_zstd(), zstd_only()),
            FieldSpec::primitive("quantity", DataType::I32, delta_leb_zstd()),
            FieldSpec::primitive("unit_price", DataType::F64, gorilla_zstd()),
            FieldSpec::utf8("category", delta_leb_zstd(), zstd_only()),
        ],
    };

    Schema::new(vec![ColumnSpec::struct_col(
        "event",
        vec![
            FieldSpec::primitive("ts", DataType::I64, delta_leb_zstd()),
            FieldSpec::utf8("event_id", delta_leb_zstd(), zstd_only()),
            FieldSpec::utf8("event_type", delta_leb_zstd(), zstd_only()),
            // amount as Binary (Avro bytes / decimal)
            FieldSpec::new(
                "amount",
                LogicalType::Binary,
                vec![delta_leb_zstd(), zstd_only()],
            ),
            FieldSpec::utf8("currency", delta_leb_zstd(), zstd_only()),
            // List<line_item>
            FieldSpec::list("line_items", line_item, vec![delta_leb_zstd()]),
            // Nullable<Utf8> customer reference
            FieldSpec::nullable(
                "customer_ref",
                LogicalType::Utf8,
                vec![leb_zstd(), delta_leb_zstd(), zstd_only()],
            ),
            // List<Utf8> tags
            FieldSpec::list(
                "tags",
                LogicalType::Utf8,
                vec![delta_leb_zstd(), delta_leb_zstd(), zstd_only()],
            ),
            FieldSpec::utf8("channel", delta_leb_zstd(), zstd_only()),
            FieldSpec::utf8("region", delta_leb_zstd(), zstd_only()),
            // Nullable<I64> session duration
            FieldSpec::nullable(
                "session_ms",
                LogicalType::Primitive {
                    data_type: DataType::I64,
                },
                vec![leb_zstd(), delta_leb_zstd()],
            ),
        ],
    )])
}

// ---------------------------------------------------------------------------
// Measurement + report
// ---------------------------------------------------------------------------

struct Measurement {
    shape: &'static str,
    top_level_cols: usize,
    raw_bytes: usize,
    compressed_bytes: usize,
}

impl Measurement {
    fn savings_pct(&self) -> f64 {
        (self.raw_bytes as f64 - self.compressed_bytes as f64) / self.raw_bytes as f64 * 100.0
    }

    fn ratio(&self) -> f64 {
        self.raw_bytes as f64 / self.compressed_bytes as f64
    }
}

fn measure(shape: &'static str, schema: Schema) -> Measurement {
    let top_level_cols = schema.columns.len();
    let raw = schema.to_json().expect("to_json");
    let raw_bytes = raw.len();
    let compressed = zstd::encode_all(&raw[..], 3).expect("zstd compress");
    let compressed_bytes = compressed.len();
    Measurement {
        shape,
        top_level_cols,
        raw_bytes,
        compressed_bytes,
    }
}

#[test]
fn v4_schema_overhead_corpus() {
    let results = vec![
        measure("Server Log (recursive struct, 1-col)", schema_server_log()),
        measure(
            "Transaction Event (recursive struct, 1-col)",
            schema_transaction_event(),
        ),
        measure(
            "Sensor Reading (List<Struct>, 1-col)",
            schema_sensor_reading(),
        ),
        measure(
            "String-Heavy MR (recursive struct, 1-col)",
            schema_string_heavy_mr(),
        ),
        measure("Builder sanity 8-col flat", schema_builder_sanity_8col()),
        measure("Wide flat 50-col", schema_wide_flat_50col()),
        measure("Deep nesting (5 levels)", schema_deep_nesting()),
        measure(
            "Avro analytics event (recursive struct)",
            schema_avro_analytics_event(),
        ),
    ];

    // -- print markdown table --
    println!();
    println!(
        "| {:<40} | {:>5} | {:>18} | {:>18} | {:>7} | {:>17} |",
        "Shape", "Cols", "Raw schema (bytes)", "Compressed (bytes)", "Savings", "Compression ratio"
    );
    println!(
        "| {:-<40} | {:->5} | {:->18} | {:->18} | {:->7} | {:->17} |",
        "", "", "", "", "", ""
    );
    for r in &results {
        println!(
            "| {:<40} | {:>5} | {:>18} | {:>18} | {:>6.1}% | {:>15.1}× |",
            r.shape,
            r.top_level_cols,
            r.raw_bytes,
            r.compressed_bytes,
            r.savings_pct(),
            r.ratio(),
        );
    }

    let avg_savings: f64 =
        results.iter().map(|r| r.savings_pct()).sum::<f64>() / results.len() as f64;
    let worst_savings = results
        .iter()
        .map(|r| r.savings_pct())
        .fold(f64::INFINITY, f64::min);
    let best_savings = results
        .iter()
        .map(|r| r.savings_pct())
        .fold(f64::NEG_INFINITY, f64::max);

    println!();
    println!(
        "Average savings: {avg_savings:.1}%  |  Worst: {worst_savings:.1}%  |  Best: {best_savings:.1}%"
    );

    // -- assertions --

    // Every shape must compress smaller than raw (strict requirement)
    let mut any_fail = false;
    for r in &results {
        if r.compressed_bytes >= r.raw_bytes {
            eprintln!(
                "ANOMALY: '{}' — compressed ({}) >= raw ({})! zstd may not help on tiny schemas.",
                r.shape, r.compressed_bytes, r.raw_bytes
            );
            any_fail = true;
        }
    }
    assert!(
        !any_fail,
        "one or more shapes failed the compressed < raw requirement"
    );

    // Flag (but don't fail) anything below 50% savings
    for r in &results {
        if r.savings_pct() < 50.0 {
            println!(
                "FLAG: '{}' savings {:.1}% < 50% — may indicate a very small or already-compressed shape",
                r.shape,
                r.savings_pct()
            );
        }
    }
}
