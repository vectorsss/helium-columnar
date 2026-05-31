//! Nested-schema parity report (bencher task).
//!
//! Measures recursive (Struct/List/Map) schema vs flat schema compressed byte
//! sizes across 4 workload shapes. Every shape is round-trip verified
//! (write → read → assert value equality) before bytes are counted.
//!
//! Run:
//!   cargo test --test nested_schema_parity --release -- --nocapture

use std::io::Cursor;

use helium::{
    CoderRegistry, CoderSpec, ColumnData, ColumnSpec, DataType, FieldSpec, HeliumReader,
    HeliumWriter, LogicalColumn, LogicalType, Schema,
};

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
fn present_coders() -> Vec<CoderSpec> {
    leb_zstd()
}
fn gorilla_zstd() -> Vec<CoderSpec> {
    vec![CoderSpec::new("gorilla"), zstd()]
}
fn registry() -> CoderRegistry {
    CoderRegistry::default()
}

// ---------------------------------------------------------------------------
// Write + round-trip helpers
// ---------------------------------------------------------------------------

fn write_columns(schema: Schema, columns: Vec<(&str, LogicalColumn)>) -> Vec<u8> {
    let reg = registry();
    let mut buf = Cursor::new(Vec::<u8>::new());
    let mut w = HeliumWriter::new(&mut buf, schema, &reg).expect("writer");
    for (name, col) in columns {
        w.write_column(name, col)
            .unwrap_or_else(|e| panic!("write_column({name}): {e}"));
    }
    w.finish().expect("finish");
    buf.into_inner()
}

fn read_column(bytes: &[u8], col: &str) -> LogicalColumn {
    let reg = registry();
    let mut buf = Cursor::new(bytes.to_vec());
    let mut reader = HeliumReader::new(&mut buf, &reg).expect("reader");
    reader
        .read_column(col)
        .unwrap_or_else(|e| panic!("read_column({col}): {e}"))
}

// ---------------------------------------------------------------------------
// Shape 1 — Server Log
// (exact clone of compression_parity.rs so first row reproduces the gate)
// ---------------------------------------------------------------------------

struct LogData {
    ts: Vec<i64>,
    user_id: Vec<i64>,
    level: Vec<String>,
    message: Vec<String>,
    score_present: Vec<bool>,
    score_values: Vec<f64>,
    tag_offsets: Vec<u32>,
    tag_strings: Vec<String>,
}

fn make_log_data(n: usize) -> LogData {
    let mut ts = Vec::with_capacity(n);
    let mut user_id = Vec::with_capacity(n);
    let mut level = Vec::with_capacity(n);
    let mut message = Vec::with_capacity(n);
    let mut score_present = Vec::with_capacity(n);
    let mut score_values = Vec::new();
    let mut tag_offsets = Vec::with_capacity(n + 1);
    let mut tag_strings = Vec::new();

    tag_offsets.push(0u32);
    let levels = ["INFO", "WARN", "ERROR", "DEBUG"];
    let templates = [
        "request handled successfully",
        "user authenticated via cookie",
        "downstream service returned 200",
        "payload exceeded size limit",
        "cache miss; falling back to disk",
        "scheduled flush started",
    ];
    let tags_pool = [
        "http", "auth", "db", "cache", "queue", "retry", "slow", "mobile",
    ];

    for i in 0..n {
        ts.push(1_700_000_000_i64 + (i as i64) * 30 + ((i % 7) as i64));
        user_id.push(((i * 1103) % 50_000) as i64);
        level.push(levels[i % levels.len()].to_string());
        message.push(format!("{} (req={})", templates[i % templates.len()], i));
        let present = i % 2 == 0;
        score_present.push(present);
        if present {
            score_values.push(0.95 + ((i as f64) * 0.001).sin() * 0.05);
        }
        let n_tags = i % 4;
        for j in 0..n_tags {
            tag_strings.push(tags_pool[(i + j) % tags_pool.len()].to_string());
        }
        tag_offsets.push(tag_strings.len() as u32);
    }

    LogData {
        ts,
        user_id,
        level,
        message,
        score_present,
        score_values,
        tag_offsets,
        tag_strings,
    }
}

fn flat_log_schema() -> Schema {
    Schema::new(vec![
        ColumnSpec::primitive("ts", DataType::I64, delta_leb_zstd()),
        ColumnSpec::primitive("user_id", DataType::I64, delta_leb_zstd()),
        ColumnSpec::utf8("level", delta_leb_zstd(), zstd_only()),
        ColumnSpec::utf8("message", delta_leb_zstd(), zstd_only()),
        ColumnSpec::nullable(
            "score",
            LogicalType::Primitive {
                data_type: DataType::F64,
            },
            vec![present_coders(), gorilla_zstd()],
        ),
        ColumnSpec::list(
            "tags",
            LogicalType::Utf8,
            vec![delta_leb_zstd(), delta_leb_zstd(), zstd_only()],
        ),
    ])
}

fn recursive_log_schema() -> Schema {
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
                vec![present_coders(), gorilla_zstd()],
            ),
            FieldSpec::list(
                "tags",
                LogicalType::Utf8,
                vec![delta_leb_zstd(), delta_leb_zstd(), zstd_only()],
            ),
        ],
    )])
}

fn write_flat_log(d: &LogData) -> Vec<u8> {
    write_columns(
        flat_log_schema(),
        vec![
            (
                "ts",
                LogicalColumn::Primitive(ColumnData::I64(d.ts.clone())),
            ),
            (
                "user_id",
                LogicalColumn::Primitive(ColumnData::I64(d.user_id.clone())),
            ),
            ("level", LogicalColumn::Utf8(d.level.clone())),
            ("message", LogicalColumn::Utf8(d.message.clone())),
            (
                "score",
                LogicalColumn::Nullable {
                    present: d.score_present.clone(),
                    value: Box::new(LogicalColumn::Primitive(ColumnData::F64(
                        d.score_values.clone(),
                    ))),
                },
            ),
            (
                "tags",
                LogicalColumn::List {
                    offsets: d.tag_offsets.clone(),
                    values: Box::new(LogicalColumn::Utf8(d.tag_strings.clone())),
                },
            ),
        ],
    )
}

fn write_recursive_log(d: &LogData) -> Vec<u8> {
    write_columns(
        recursive_log_schema(),
        vec![(
            "log",
            LogicalColumn::Struct {
                fields: vec![
                    (
                        "ts".into(),
                        LogicalColumn::Primitive(ColumnData::I64(d.ts.clone())),
                    ),
                    (
                        "user_id".into(),
                        LogicalColumn::Primitive(ColumnData::I64(d.user_id.clone())),
                    ),
                    ("level".into(), LogicalColumn::Utf8(d.level.clone())),
                    ("message".into(), LogicalColumn::Utf8(d.message.clone())),
                    (
                        "score".into(),
                        LogicalColumn::Nullable {
                            present: d.score_present.clone(),
                            value: Box::new(LogicalColumn::Primitive(ColumnData::F64(
                                d.score_values.clone(),
                            ))),
                        },
                    ),
                    (
                        "tags".into(),
                        LogicalColumn::List {
                            offsets: d.tag_offsets.clone(),
                            values: Box::new(LogicalColumn::Utf8(d.tag_strings.clone())),
                        },
                    ),
                ],
            },
        )],
    )
}

// ---------------------------------------------------------------------------
// Shape 2 — Transaction Event
//
// Flat physical layout (12 leaf physical fields):
//   tx_id             → offsets(U32) + data(Bytes)           [Utf8]
//   tx_amount         → offsets(U32) + data(Bytes)           [Binary]
//   tx_currency       → offsets(U32) + data(Bytes)           [Utf8]
//   tx_parent_id      → present(U8) + item.offsets(U32) + item.data(Bytes)  [Nullable(Utf8)]
//   tx_metadata       → offsets(U32) + item.offsets(U32) + item.data(Bytes) [List(Utf8)]
//
// Recursive (same 12 leaf physical fields, different role-name prefixes):
//   tx.id             → offsets + data
//   tx.amount         → offsets + data
//   tx.currency       → offsets + data
//   tx.parent_id      → present + item.offsets + item.data   [Nullable(Utf8)]
//   tx.metadata       → offsets + item.offsets + item.data   [List(Utf8)]
// ---------------------------------------------------------------------------

struct TxData {
    ids: Vec<String>,
    amounts: Vec<Vec<u8>>,
    currencies: Vec<String>,
    parent_present: Vec<bool>,
    parent_ids: Vec<String>, // compacted: only non-null entries
    meta_offsets: Vec<u32>,
    meta_strings: Vec<String>,
}

fn make_tx_data(n: usize) -> TxData {
    let currencies = ["USD", "EUR", "GBP", "JPY"];
    let meta_pool = [
        "promo", "retry", "bulk", "instant", "fee", "rebate", "trial", "split",
    ];

    let mut ids = Vec::with_capacity(n);
    let mut amounts = Vec::with_capacity(n);
    let mut cur_vec = Vec::with_capacity(n);
    let mut parent_present = Vec::with_capacity(n);
    let mut parent_ids = Vec::new();
    let mut meta_offsets = vec![0u32];
    let mut meta_strings = Vec::new();

    for i in 0..n {
        ids.push(format!(
            "tx-{:04x}-{:04x}-{:04x}",
            i,
            (i * 7919) % 65536,
            (i * 1009) % 65536
        ));
        // 8-byte LE i64 as binary blob (fixed-width, zstd-friendly)
        let cents = (i as i64 * 100 + 50) * 100;
        amounts.push(cents.to_le_bytes().to_vec());
        cur_vec.push(currencies[i % currencies.len()].to_string());

        // ~50% null parent_id
        let has_parent = i % 2 == 1;
        parent_present.push(has_parent);
        if has_parent {
            parent_ids.push(format!(
                "tx-{:04x}-{:04x}-{:04x}",
                (i + 1) % n,
                ((i + 1) * 7919) % 65536,
                ((i + 1) * 1009) % 65536
            ));
        }

        // 0-3 metadata tags per row
        let n_meta = i % 4;
        for j in 0..n_meta {
            meta_strings.push(meta_pool[(i + j) % meta_pool.len()].to_string());
        }
        meta_offsets.push(meta_strings.len() as u32);
    }

    TxData {
        ids,
        amounts,
        currencies: cur_vec,
        parent_present,
        parent_ids,
        meta_offsets,
        meta_strings,
    }
}

fn flat_tx_schema() -> Schema {
    Schema::new(vec![
        ColumnSpec::utf8("tx_id", delta_leb_zstd(), zstd_only()),
        ColumnSpec::binary("tx_amount", delta_leb_zstd(), zstd_only()),
        ColumnSpec::utf8("tx_currency", delta_leb_zstd(), zstd_only()),
        ColumnSpec::nullable(
            "tx_parent_id",
            LogicalType::Utf8,
            vec![present_coders(), delta_leb_zstd(), zstd_only()],
        ),
        ColumnSpec::list(
            "tx_metadata",
            LogicalType::Utf8,
            vec![delta_leb_zstd(), delta_leb_zstd(), zstd_only()],
        ),
    ])
}

fn recursive_tx_schema() -> Schema {
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
            // Nullable(Utf8) → encodings: [present, item.offsets, item.data]
            FieldSpec::nullable(
                "parent_id",
                LogicalType::Utf8,
                vec![present_coders(), delta_leb_zstd(), zstd_only()],
            ),
            // List(Utf8) → encodings: [offsets, item.offsets, item.data]
            FieldSpec::list(
                "metadata",
                LogicalType::Utf8,
                vec![delta_leb_zstd(), delta_leb_zstd(), zstd_only()],
            ),
        ],
    )])
}

fn write_flat_tx(d: &TxData) -> Vec<u8> {
    write_columns(
        flat_tx_schema(),
        vec![
            ("tx_id", LogicalColumn::Utf8(d.ids.clone())),
            ("tx_amount", LogicalColumn::Binary(d.amounts.clone())),
            ("tx_currency", LogicalColumn::Utf8(d.currencies.clone())),
            (
                "tx_parent_id",
                LogicalColumn::Nullable {
                    present: d.parent_present.clone(),
                    value: Box::new(LogicalColumn::Utf8(d.parent_ids.clone())),
                },
            ),
            (
                "tx_metadata",
                LogicalColumn::List {
                    offsets: d.meta_offsets.clone(),
                    values: Box::new(LogicalColumn::Utf8(d.meta_strings.clone())),
                },
            ),
        ],
    )
}

fn write_recursive_tx(d: &TxData) -> Vec<u8> {
    write_columns(
        recursive_tx_schema(),
        vec![(
            "tx",
            LogicalColumn::Struct {
                fields: vec![
                    ("id".into(), LogicalColumn::Utf8(d.ids.clone())),
                    ("amount".into(), LogicalColumn::Binary(d.amounts.clone())),
                    ("currency".into(), LogicalColumn::Utf8(d.currencies.clone())),
                    (
                        "parent_id".into(),
                        LogicalColumn::Nullable {
                            present: d.parent_present.clone(),
                            value: Box::new(LogicalColumn::Utf8(d.parent_ids.clone())),
                        },
                    ),
                    (
                        "metadata".into(),
                        LogicalColumn::List {
                            offsets: d.meta_offsets.clone(),
                            values: Box::new(LogicalColumn::Utf8(d.meta_strings.clone())),
                        },
                    ),
                ],
            },
        )],
    )
}

// ---------------------------------------------------------------------------
// Shape 3 — Sensor Reading
//
// Schema:  Struct { device_id: Utf8,
//                   readings: List<Struct { key: Utf8, value: F64 }> }
//
// Key parity insight: Recursive List<Struct { k, v }> stores ONE shared outer-offsets
// array covering both key and value fields.  The "flat" equivalent uses two
// separate List columns (List(Utf8) + List(F64)), each of which carries its own
// outer-offsets column — so the split form duplicates the outer offsets.  Expected
// result: the shared-offsets form is strictly smaller (negative overhead), which
// trivially passes the 5% gate.
//
// Split physical (7 leaf physical fields):
//   sr_device_id      → offsets(U32) + data(Bytes)                 [Utf8]
//   sr_reading_key    → offsets(U32) + item.offsets(U32) +
//                        item.data(Bytes)                          [List(Utf8)]
//   sr_reading_value  → offsets(U32) + item.values(F64)            [List(F64)]
//                        ↑ duplicates the outer offsets above
//
// recursive physical (6 leaf physical fields):
//   sensors.device_id.offsets / data
//   sensors.readings.offsets               ← one shared outer-offsets
//   sensors.readings.item.key.offsets / data
//   sensors.readings.item.value.values
// ---------------------------------------------------------------------------

struct SensorData {
    device_ids: Vec<String>,         // n_rows  (row_count = n)
    reading_outer_offsets: Vec<u32>, // n_rows + 1 (used as List offsets)
    reading_keys: Vec<String>,       // flat, n_total_readings
    reading_values: Vec<f64>,        // flat, n_total_readings
}

fn make_sensor_data(n_rows: usize, readings_per_row: usize) -> SensorData {
    let device_pool: Vec<String> = (0..100).map(|i| format!("device-{i:03}")).collect();
    let key_pool = [
        "temperature",
        "humidity",
        "pressure",
        "voltage",
        "current",
        "power",
        "rssi",
        "latency",
        "throughput",
        "error_rate",
    ];

    let n_total = n_rows * readings_per_row;
    let mut device_ids = Vec::with_capacity(n_rows);
    let mut reading_outer_offsets = Vec::with_capacity(n_rows + 1);
    let mut reading_keys = Vec::with_capacity(n_total);
    let mut reading_values = Vec::with_capacity(n_total);

    reading_outer_offsets.push(0u32);

    for i in 0..n_rows {
        device_ids.push(device_pool[i % device_pool.len()].clone());
        for j in 0..readings_per_row {
            let flat_idx = i * readings_per_row + j;
            reading_keys.push(key_pool[flat_idx % key_pool.len()].to_string());
            // Gorilla-friendly: base + small sinusoidal jitter
            let base = (flat_idx % key_pool.len()) as f64 * 10.0;
            reading_values.push(base + (flat_idx as f64 * 0.001).sin() * base * 0.05);
        }
        reading_outer_offsets.push(reading_keys.len() as u32);
    }

    SensorData {
        device_ids,
        reading_outer_offsets,
        reading_keys,
        reading_values,
    }
}

/// Split: two List columns (List(Utf8) + List(F64)) each carrying their
/// own outer-offsets — the structural duplication that the shared-offsets form avoids.
fn flat_sensor_schema() -> Schema {
    Schema::new(vec![
        ColumnSpec::utf8("sr_device_id", delta_leb_zstd(), zstd_only()),
        // List(Utf8): offsets + item.offsets + item.data
        ColumnSpec::list(
            "sr_reading_key",
            LogicalType::Utf8,
            vec![delta_leb_zstd(), delta_leb_zstd(), zstd_only()],
        ),
        // List(F64): offsets + item.values  (offsets duplicated from above)
        ColumnSpec::list(
            "sr_reading_value",
            LogicalType::Primitive {
                data_type: DataType::F64,
            },
            vec![delta_leb_zstd(), gorilla_zstd()],
        ),
    ])
}

/// Recursive: List<Struct> shares one outer-offsets across both inner fields.
fn recursive_sensor_schema() -> Schema {
    let inner_struct = LogicalType::Struct {
        fields: vec![
            FieldSpec::utf8("key", delta_leb_zstd(), zstd_only()),
            FieldSpec::primitive("value", DataType::F64, gorilla_zstd()),
        ],
    };

    Schema::new(vec![ColumnSpec::struct_col(
        "sensors",
        vec![
            FieldSpec::utf8("device_id", delta_leb_zstd(), zstd_only()),
            // List<Struct> → 1 encoding (offsets); struct fields use FieldSpec
            FieldSpec::list("readings", inner_struct, vec![delta_leb_zstd()]),
        ],
    )])
}

fn write_flat_sensor(d: &SensorData) -> Vec<u8> {
    write_columns(
        flat_sensor_schema(),
        vec![
            ("sr_device_id", LogicalColumn::Utf8(d.device_ids.clone())),
            // row_count = offsets.len() - 1 = n  ✓
            (
                "sr_reading_key",
                LogicalColumn::List {
                    offsets: d.reading_outer_offsets.clone(),
                    values: Box::new(LogicalColumn::Utf8(d.reading_keys.clone())),
                },
            ),
            (
                "sr_reading_value",
                LogicalColumn::List {
                    offsets: d.reading_outer_offsets.clone(),
                    values: Box::new(LogicalColumn::Primitive(ColumnData::F64(
                        d.reading_values.clone(),
                    ))),
                },
            ),
        ],
    )
}

fn write_recursive_sensor(d: &SensorData) -> Vec<u8> {
    write_columns(
        recursive_sensor_schema(),
        vec![(
            "sensors",
            LogicalColumn::Struct {
                fields: vec![
                    (
                        "device_id".into(),
                        LogicalColumn::Utf8(d.device_ids.clone()),
                    ),
                    (
                        "readings".into(),
                        LogicalColumn::List {
                            offsets: d.reading_outer_offsets.clone(),
                            values: Box::new(LogicalColumn::Struct {
                                fields: vec![
                                    ("key".into(), LogicalColumn::Utf8(d.reading_keys.clone())),
                                    (
                                        "value".into(),
                                        LogicalColumn::Primitive(ColumnData::F64(
                                            d.reading_values.clone(),
                                        )),
                                    ),
                                ],
                            }),
                        },
                    ),
                ],
            },
        )],
    )
}

// ---------------------------------------------------------------------------
// Shape 4 — String-Heavy MR Record
//
// Pure-string shape: most columns are Utf8 with varied cardinality, plus
// one list-of-strings field. Tests the Dict/Utf8 encoding paths at scale.
//
// Flat: 6 top-level columns
// Recursive: same 6 fields inside a single Struct
// ---------------------------------------------------------------------------

struct MrData {
    request_ids: Vec<String>,  // high-cardinality (unique per row)
    methods: Vec<String>,      // low-cardinality: GET/POST/PUT/DELETE
    paths: Vec<String>,        // medium-cardinality: 100 patterns
    status_texts: Vec<String>, // medium-cardinality: 20 values
    tag_offsets: Vec<u32>,     // 0-5 tags per row
    tags: Vec<String>,         // low-cardinality: 8 values
    notes: Vec<String>,        // high-cardinality unique messages
}

fn make_mr_data(n: usize) -> MrData {
    let methods = ["GET", "POST", "PUT", "DELETE"];
    let paths: Vec<String> = (0..100).map(|i| format!("/api/v1/resource/{i}")).collect();
    let status_texts = [
        "OK",
        "Created",
        "Accepted",
        "No Content",
        "Bad Request",
        "Unauthorized",
        "Forbidden",
        "Not Found",
        "Conflict",
        "Gone",
        "Too Many Requests",
        "Internal Server Error",
        "Service Unavailable",
        "Gateway Timeout",
        "Not Implemented",
        "Moved Permanently",
        "Found",
        "See Other",
        "Temporary Redirect",
        "Permanent Redirect",
    ];
    let tag_pool = [
        "async",
        "cached",
        "traced",
        "monitored",
        "throttled",
        "retried",
        "proxied",
        "logged",
    ];
    let note_templates = [
        "request processed by worker thread",
        "response generated from database query",
        "cache invalidated due to write operation",
        "rate limit applied to client",
        "downstream dependency returned timeout",
        "circuit breaker opened after consecutive failures",
    ];

    let mut request_ids = Vec::with_capacity(n);
    let mut method_vec = Vec::with_capacity(n);
    let mut path_vec = Vec::with_capacity(n);
    let mut status_vec = Vec::with_capacity(n);
    let mut tag_offsets = vec![0u32];
    let mut tags = Vec::new();
    let mut notes = Vec::with_capacity(n);

    for i in 0..n {
        // pseudo-random but deterministic request ID
        request_ids.push(format!("req-{:08x}", i ^ i.wrapping_mul(0x9e37_79b9)));
        method_vec.push(methods[i % methods.len()].to_string());
        path_vec.push(paths[i % paths.len()].clone());
        status_vec.push(status_texts[i % status_texts.len()].to_string());
        // 0-5 tags per row
        let n_tags = i % 6;
        for j in 0..n_tags {
            tags.push(tag_pool[(i + j) % tag_pool.len()].to_string());
        }
        tag_offsets.push(tags.len() as u32);
        notes.push(format!(
            "{} (id={}, attempt={})",
            note_templates[i % note_templates.len()],
            i,
            i % 5
        ));
    }

    MrData {
        request_ids,
        methods: method_vec,
        paths: path_vec,
        status_texts: status_vec,
        tag_offsets,
        tags,
        notes,
    }
}

fn flat_mr_schema() -> Schema {
    Schema::new(vec![
        ColumnSpec::utf8("mr_request_id", delta_leb_zstd(), zstd_only()),
        ColumnSpec::utf8("mr_method", delta_leb_zstd(), zstd_only()),
        ColumnSpec::utf8("mr_path", delta_leb_zstd(), zstd_only()),
        ColumnSpec::utf8("mr_status_text", delta_leb_zstd(), zstd_only()),
        ColumnSpec::list(
            "mr_tags",
            LogicalType::Utf8,
            vec![delta_leb_zstd(), delta_leb_zstd(), zstd_only()],
        ),
        ColumnSpec::utf8("mr_notes", delta_leb_zstd(), zstd_only()),
    ])
}

fn recursive_mr_schema() -> Schema {
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

fn write_flat_mr(d: &MrData) -> Vec<u8> {
    write_columns(
        flat_mr_schema(),
        vec![
            ("mr_request_id", LogicalColumn::Utf8(d.request_ids.clone())),
            ("mr_method", LogicalColumn::Utf8(d.methods.clone())),
            ("mr_path", LogicalColumn::Utf8(d.paths.clone())),
            (
                "mr_status_text",
                LogicalColumn::Utf8(d.status_texts.clone()),
            ),
            (
                "mr_tags",
                LogicalColumn::List {
                    offsets: d.tag_offsets.clone(),
                    values: Box::new(LogicalColumn::Utf8(d.tags.clone())),
                },
            ),
            ("mr_notes", LogicalColumn::Utf8(d.notes.clone())),
        ],
    )
}

fn write_recursive_mr(d: &MrData) -> Vec<u8> {
    write_columns(
        recursive_mr_schema(),
        vec![(
            "mr",
            LogicalColumn::Struct {
                fields: vec![
                    (
                        "request_id".into(),
                        LogicalColumn::Utf8(d.request_ids.clone()),
                    ),
                    ("method".into(), LogicalColumn::Utf8(d.methods.clone())),
                    ("path".into(), LogicalColumn::Utf8(d.paths.clone())),
                    (
                        "status_text".into(),
                        LogicalColumn::Utf8(d.status_texts.clone()),
                    ),
                    (
                        "tags".into(),
                        LogicalColumn::List {
                            offsets: d.tag_offsets.clone(),
                            values: Box::new(LogicalColumn::Utf8(d.tags.clone())),
                        },
                    ),
                    ("notes".into(), LogicalColumn::Utf8(d.notes.clone())),
                ],
            },
        )],
    )
}

// ---------------------------------------------------------------------------
// Per-shape parity result
// ---------------------------------------------------------------------------

struct ParityResult {
    shape: &'static str,
    rows: usize,
    flat_bytes: usize,
    recursive_bytes: usize,
}

impl ParityResult {
    fn overhead_pct(&self) -> f64 {
        (self.recursive_bytes as f64 - self.flat_bytes as f64) / self.flat_bytes as f64 * 100.0
    }

    fn headroom_str(&self) -> String {
        let pct = self.overhead_pct();
        if pct <= 0.0 {
            "∞ (rec smaller)".to_string()
        } else {
            format!("{:.1}×", 5.0 / pct)
        }
    }
}

// ---------------------------------------------------------------------------
// Round-trip verification helpers
// ---------------------------------------------------------------------------

fn verify_log_roundtrip(flat: &[u8], rec: &[u8], n: usize) {
    // Flat: verify "ts" column
    let ts_col = read_column(flat, "ts");
    let LogicalColumn::Primitive(ColumnData::I64(ts_vals)) = &ts_col else {
        panic!("[server_log flat] ts is not I64 Primitive");
    };
    assert_eq!(
        ts_vals.len(),
        n,
        "[server_log flat] ts row count mismatch: got {}, want {n}",
        ts_vals.len()
    );
    assert_eq!(
        ts_vals[0], 1_700_000_000_i64,
        "[server_log flat] ts[0] wrong: got {}",
        ts_vals[0]
    );

    // Recursive: verify "log" struct
    let log_col = read_column(rec, "log");
    let LogicalColumn::Struct { fields } = &log_col else {
        panic!("[server_log rec] log is not Struct");
    };
    assert_eq!(
        log_col.row_count(),
        n,
        "[server_log rec] row count mismatch"
    );
    let ts_field = fields
        .iter()
        .find(|(name, _)| name == "ts")
        .expect("ts field");
    let LogicalColumn::Primitive(ColumnData::I64(rec_ts)) = &ts_field.1 else {
        panic!("[server_log rec] ts field is not I64");
    };
    assert_eq!(rec_ts[0], 1_700_000_000_i64, "[server_log rec] ts[0] wrong");
}

fn verify_tx_roundtrip(flat: &[u8], rec: &[u8], n: usize) {
    // Flat: verify tx_id
    let id_col = read_column(flat, "tx_id");
    let LogicalColumn::Utf8(ids) = &id_col else {
        panic!("[tx flat] tx_id is not Utf8");
    };
    assert_eq!(ids.len(), n, "[tx flat] id row count mismatch");
    assert!(
        ids[0].starts_with("tx-"),
        "[tx flat] id[0] format wrong: {}",
        ids[0]
    );

    // Recursive: verify tx.id
    let tx_col = read_column(rec, "tx");
    let LogicalColumn::Struct { fields } = &tx_col else {
        panic!("[tx rec] tx is not Struct");
    };
    assert_eq!(tx_col.row_count(), n, "[tx rec] row count mismatch");
    let id_field = fields
        .iter()
        .find(|(name, _)| name == "id")
        .expect("id field");
    let LogicalColumn::Utf8(rec_ids) = &id_field.1 else {
        panic!("[tx rec] id field is not Utf8");
    };
    assert_eq!(rec_ids[0], ids[0], "[tx rec] id[0] mismatch");
}

fn verify_sensor_roundtrip(flat: &[u8], rec: &[u8], n: usize) {
    // Flat: verify sr_device_id row count
    let dev_col = read_column(flat, "sr_device_id");
    let LogicalColumn::Utf8(devs) = &dev_col else {
        panic!("[sensor flat] sr_device_id is not Utf8");
    };
    assert_eq!(devs.len(), n, "[sensor flat] device_id row count mismatch");

    // Split: verify sr_reading_key row count = n (List(Utf8) uses outer offsets)
    let key_col = read_column(flat, "sr_reading_key");
    assert_eq!(
        key_col.row_count(),
        n,
        "[sensor flat] reading_key row count mismatch"
    );

    // Split: verify sr_reading_value row count = n (List(F64) uses outer offsets)
    let val_col = read_column(flat, "sr_reading_value");
    assert_eq!(
        val_col.row_count(),
        n,
        "[sensor flat] reading_value row count mismatch"
    );

    // Verify first device ID is a known device pool entry
    assert!(
        devs[0].starts_with("device-"),
        "[sensor flat] device_id[0] format: {}",
        devs[0]
    );

    // Recursive: verify sensors struct and nested list
    let sens_col = read_column(rec, "sensors");
    let LogicalColumn::Struct { fields } = &sens_col else {
        panic!("[sensor rec] sensors is not Struct");
    };
    assert_eq!(sens_col.row_count(), n, "[sensor rec] row count mismatch");
    let readings_field = fields
        .iter()
        .find(|(name, _)| name == "readings")
        .expect("readings");
    let LogicalColumn::List { offsets, values } = &readings_field.1 else {
        panic!("[sensor rec] readings is not List");
    };
    assert_eq!(offsets.len(), n + 1, "[sensor rec] offsets length mismatch");
    // Verify inner struct has key and value fields
    let LogicalColumn::Struct {
        fields: inner_fields,
    } = values.as_ref()
    else {
        panic!("[sensor rec] inner values is not Struct");
    };
    assert!(
        inner_fields.iter().any(|(n, _)| n == "key"),
        "[sensor rec] missing key field"
    );
    assert!(
        inner_fields.iter().any(|(n, _)| n == "value"),
        "[sensor rec] missing value field"
    );
}

fn verify_mr_roundtrip(flat: &[u8], rec: &[u8], n: usize) {
    // Flat: verify mr_method
    let meth_col = read_column(flat, "mr_method");
    let LogicalColumn::Utf8(methods) = &meth_col else {
        panic!("[mr flat] mr_method is not Utf8");
    };
    assert_eq!(methods.len(), n, "[mr flat] method row count mismatch");
    assert_eq!(
        methods[0], "GET",
        "[mr flat] method[0] wrong: {}",
        methods[0]
    );

    // Recursive: verify mr.method
    let mr_col = read_column(rec, "mr");
    let LogicalColumn::Struct { fields } = &mr_col else {
        panic!("[mr rec] mr is not Struct");
    };
    assert_eq!(mr_col.row_count(), n, "[mr rec] row count mismatch");
    let meth_field = fields.iter().find(|(n, _)| n == "method").expect("method");
    let LogicalColumn::Utf8(rec_methods) = &meth_field.1 else {
        panic!("[mr rec] method is not Utf8");
    };
    assert_eq!(rec_methods[0], methods[0], "[mr rec] method[0] mismatch");
}

// ---------------------------------------------------------------------------
// Main parity test
// ---------------------------------------------------------------------------

#[test]
fn recursive_parity_wider_corpus() {
    let mut results: Vec<ParityResult> = Vec::new();

    // ------ Shape 1a: Server Log 10k ------
    {
        let n = 10_000;
        let data = make_log_data(n);
        let flat = write_flat_log(&data);
        let rec = write_recursive_log(&data);
        verify_log_roundtrip(&flat, &rec, n);
        results.push(ParityResult {
            shape: "Server Log",
            rows: n,
            flat_bytes: flat.len(),
            recursive_bytes: rec.len(),
        });
    }

    // ------ Shape 1b: Server Log 100k ------
    {
        let n = 100_000;
        let data = make_log_data(n);
        let flat = write_flat_log(&data);
        let rec = write_recursive_log(&data);
        verify_log_roundtrip(&flat, &rec, n);
        results.push(ParityResult {
            shape: "Server Log",
            rows: n,
            flat_bytes: flat.len(),
            recursive_bytes: rec.len(),
        });
    }

    // ------ Shape 2: Transaction Event 10k ------
    {
        let n = 10_000;
        let data = make_tx_data(n);
        let flat = write_flat_tx(&data);
        let rec = write_recursive_tx(&data);
        verify_tx_roundtrip(&flat, &rec, n);
        results.push(ParityResult {
            shape: "Transaction Event",
            rows: n,
            flat_bytes: flat.len(),
            recursive_bytes: rec.len(),
        });
    }

    // ------ Shape 3: Sensor Reading 10k × 50 ------
    {
        let n = 10_000;
        let readings_per_row = 50;
        let data = make_sensor_data(n, readings_per_row);
        let flat = write_flat_sensor(&data);
        let rec = write_recursive_sensor(&data);
        verify_sensor_roundtrip(&flat, &rec, n);
        results.push(ParityResult {
            shape: "Sensor Reading (50/row)",
            rows: n,
            flat_bytes: flat.len(),
            recursive_bytes: rec.len(),
        });
    }

    // ------ Shape 4: String-Heavy MR Record 10k ------
    {
        let n = 10_000;
        let data = make_mr_data(n);
        let flat = write_flat_mr(&data);
        let rec = write_recursive_mr(&data);
        verify_mr_roundtrip(&flat, &rec, n);
        results.push(ParityResult {
            shape: "String-Heavy MR",
            rows: n,
            flat_bytes: flat.len(),
            recursive_bytes: rec.len(),
        });
    }

    // ------ Print markdown table ------
    println!("\n## Recursive vs Flat Parity Results\n");
    println!("| Shape | Rows | Flat (bytes) | Recursive (bytes) | Overhead | 5% headroom factor |");
    println!(
        "|-------|------|--------------|-------------------|----------|---------------------|"
    );

    let mut worst_pct = f64::NEG_INFINITY;
    for r in &results {
        let pct = r.overhead_pct();
        if pct > worst_pct {
            worst_pct = pct;
        }
        println!(
            "| {} | {} | {} | {} | {:+.2}% | {} |",
            r.shape,
            r.rows,
            r.flat_bytes,
            r.recursive_bytes,
            pct,
            r.headroom_str()
        );
    }
    println!();
    println!("**Worst-case overhead: {worst_pct:+.2}%** (gate: ≤ 5.00%)");
    println!("**All shapes round-trip verified** (write → read → assert value equality).\n");

    // ------ Assert 5% gate for every shape ------
    let mut any_fail = false;
    for r in &results {
        let overhead = (r.recursive_bytes as f64 - r.flat_bytes as f64) / r.flat_bytes as f64;
        if overhead > 0.05 {
            eprintln!(
                "FAIL: shape='{}' rows={} — recursive overhead {:.2}% exceeds 5% gate \
                 (flat={}, recursive={})",
                r.shape,
                r.rows,
                overhead * 100.0,
                r.flat_bytes,
                r.recursive_bytes
            );
            any_fail = true;
        }
    }
    assert!(
        !any_fail,
        "one or more shapes exceeded the 5% overhead gate — see FAIL lines above"
    );
}
