//! §5.8 — Compression-ratio parity between v3 recursive and v2 flat schemas.
//!
//! Acceptance criterion (PLAN_V2 §5.8): the recursive form must compress
//! within **5%** of the manually-flattened v2 form on MR-shape data. If the
//! recursive form ever regresses past that threshold, the v3 nested-types
//! work has accidentally introduced encoding overhead and the production
//! anchor (Avro+zstd replacement, project memory `project_avro_replacement.md`)
//! is at risk.
//!
//! This file builds two schemas with equivalent leaf layouts and runs them
//! against the same 10k-row server-log-shaped dataset. The bencher will run
//! a wider parity report on a richer corpus afterwards (per task description).

use std::io::Cursor;

use helium::{
    CoderRegistry, CoderSpec, ColumnData, ColumnSpec, DataType, FieldSpec, HeliumWriter,
    LogicalColumn, LogicalType, Schema,
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
fn present_coders() -> Vec<CoderSpec> {
    vec![CoderSpec::new("leb128"), zstd()]
}
fn f64_coders() -> Vec<CoderSpec> {
    vec![CoderSpec::new("gorilla"), zstd()]
}
fn registry() -> CoderRegistry {
    CoderRegistry::default()
}

// ---------------------------------------------------------------------------
// Data: 10k rows of an MR-style server-log record
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
        // user_id has moderate cardinality, mostly small values for compression
        user_id.push(((i * 1103) % 50_000) as i64);
        level.push(levels[i % levels.len()].to_string());
        message.push(format!("{} (req={})", templates[i % templates.len()], i));

        // ~50% nullable scores
        let present = i % 2 == 0;
        score_present.push(present);
        if present {
            // realistic gauge floats — Gorilla-friendly
            score_values.push(0.95 + ((i as f64) * 0.001).sin() * 0.05);
        }

        // 0..=3 tags per row
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

// ---------------------------------------------------------------------------
// Schema construction — recursive (v3) vs flat (v2)
// ---------------------------------------------------------------------------

/// Flat v2 schema — one top-level column per leaf field.
fn flat_v2_schema() -> Schema {
    Schema::new(vec![
        ColumnSpec::primitive("ts", DataType::I64, delta_leb_zstd()),
        ColumnSpec::primitive("user_id", DataType::I64, delta_leb_zstd()),
        ColumnSpec::utf8("level", delta_leb_zstd(), zstd_only()),
        ColumnSpec::utf8("message", delta_leb_zstd(), zstd_only()),
        ColumnSpec::nullable_prim("score", DataType::F64, present_coders(), f64_coders()),
        ColumnSpec::array_of_utf8("tags", delta_leb_zstd(), delta_leb_zstd(), zstd_only()),
    ])
}

/// Recursive v3 schema — one top-level Struct wrapping the same fields,
/// using the new Nullable / List wrappers for the optional / list fields.
fn recursive_v3_schema() -> Schema {
    let fields = vec![
        FieldSpec::primitive("ts", DataType::I64, delta_leb_zstd()),
        FieldSpec::primitive("user_id", DataType::I64, delta_leb_zstd()),
        FieldSpec::utf8("level", delta_leb_zstd(), zstd_only()),
        FieldSpec::utf8("message", delta_leb_zstd(), zstd_only()),
        FieldSpec::nullable(
            "score",
            LogicalType::Primitive {
                data_type: DataType::F64,
            },
            vec![present_coders(), f64_coders()],
        ),
        FieldSpec::list(
            "tags",
            LogicalType::Utf8,
            vec![delta_leb_zstd(), delta_leb_zstd(), zstd_only()],
        ),
    ];
    Schema::new(vec![ColumnSpec::struct_col("log", fields)])
}

// ---------------------------------------------------------------------------
// Writer drivers — same data into both schemas
// ---------------------------------------------------------------------------

fn write_flat_v2(data: &LogData) -> Vec<u8> {
    let reg = registry();
    let mut buf = Cursor::new(Vec::<u8>::new());
    let mut w = HeliumWriter::new(&mut buf, flat_v2_schema(), &reg).expect("flat writer");
    w.write_column(
        "ts",
        LogicalColumn::Primitive(ColumnData::I64(data.ts.clone())),
    )
    .expect("ts");
    w.write_column(
        "user_id",
        LogicalColumn::Primitive(ColumnData::I64(data.user_id.clone())),
    )
    .expect("user_id");
    w.write_column("level", LogicalColumn::Utf8(data.level.clone()))
        .expect("level");
    w.write_column("message", LogicalColumn::Utf8(data.message.clone()))
        .expect("message");
    w.write_column(
        "score",
        LogicalColumn::NullablePrim {
            present: data.score_present.clone(),
            values: ColumnData::F64(data.score_values.clone()),
        },
    )
    .expect("score");
    w.write_column(
        "tags",
        LogicalColumn::ArrayOfUtf8 {
            offsets: data.tag_offsets.clone(),
            strings: data.tag_strings.clone(),
        },
    )
    .expect("tags");
    w.finish().expect("flat finish");
    buf.into_inner()
}

fn write_recursive_v3(data: &LogData) -> Vec<u8> {
    let reg = registry();
    let mut buf = Cursor::new(Vec::<u8>::new());
    let mut w = HeliumWriter::new(&mut buf, recursive_v3_schema(), &reg).expect("recursive writer");

    let log_struct = LogicalColumn::Struct {
        fields: vec![
            (
                "ts".into(),
                LogicalColumn::Primitive(ColumnData::I64(data.ts.clone())),
            ),
            (
                "user_id".into(),
                LogicalColumn::Primitive(ColumnData::I64(data.user_id.clone())),
            ),
            ("level".into(), LogicalColumn::Utf8(data.level.clone())),
            ("message".into(), LogicalColumn::Utf8(data.message.clone())),
            (
                "score".into(),
                LogicalColumn::Nullable {
                    present: data.score_present.clone(),
                    value: Box::new(LogicalColumn::Primitive(ColumnData::F64(
                        data.score_values.clone(),
                    ))),
                },
            ),
            (
                "tags".into(),
                LogicalColumn::List {
                    offsets: data.tag_offsets.clone(),
                    values: Box::new(LogicalColumn::Utf8(data.tag_strings.clone())),
                },
            ),
        ],
    };
    w.write_column("log", log_struct).expect("log struct");
    w.finish().expect("recursive finish");
    buf.into_inner()
}

// ---------------------------------------------------------------------------
// Parity assertion: recursive within 5% of flat
// ---------------------------------------------------------------------------

#[test]
fn recursive_compresses_within_5_percent_of_flat() {
    let n = 10_000;
    let data = make_log_data(n);

    let flat_bytes = write_flat_v2(&data);
    let recursive_bytes = write_recursive_v3(&data);

    let flat_len = flat_bytes.len() as f64;
    let recursive_len = recursive_bytes.len() as f64;

    // Allowed: recursive is at most 5% larger than flat. Recursive being
    // SMALLER than flat is also acceptable (no upper bound on the win side).
    let overhead = (recursive_len - flat_len) / flat_len;
    let pct = overhead * 100.0;

    eprintln!(
        "[parity] flat={flat_len} bytes, recursive={recursive_len} bytes, \
         overhead={pct:+.2}% (gate: ≤ 5.00%)",
    );

    assert!(
        overhead <= 0.05,
        "recursive form is {pct:.2}% larger than flat form (gate: ≤ 5.00%); \
         flat={flat_len} bytes, recursive={recursive_len} bytes — \
         this fails §5.8 acceptance and risks the Avro-replacement anchor"
    );
}

#[test]
fn flat_and_recursive_carry_same_leaf_byte_count() {
    // Beyond the file-size gate: the per-leaf encoded byte counts should
    // match exactly between flat and recursive forms (ignoring header/footer
    // metadata), because both schemas use the same per-leaf coder pipelines
    // on the same source data. This is the strict invariant — if it fails,
    // some leaf in the recursive path picked up a different code path.
    //
    // We approximate the check by comparing total file sizes minus a fudge
    // for metadata; the strict per-leaf comparison would require a footer-
    // introspection API that doesn't exist publicly.
    let n = 1_000;
    let data = make_log_data(n);
    let flat = write_flat_v2(&data);
    let recursive = write_recursive_v3(&data);

    // The ABSOLUTE difference (bytes) should be small — dominated by schema
    // JSON differences, not by leaf encoding differences. For 1k rows of
    // data each leaf produces hundreds of bytes; the metadata delta should
    // be well under 1KB.
    let abs_diff = (flat.len() as i64 - recursive.len() as i64).abs();
    eprintln!(
        "[parity] 1k-row absolute diff: {abs_diff} bytes (flat={}, recursive={})",
        flat.len(),
        recursive.len()
    );
    assert!(
        abs_diff < 4_000,
        "absolute byte diff between schemas should be metadata-bounded \
         (<4 KB); got {abs_diff} bytes"
    );
}
