//! Cross-format file-size comparison report.
//!
//! Writes the same ~10 000-row dataset to every supported storage format and
//! every Helium configuration, records the file size on disk, and produces a
//! Markdown table comparing "bytes you would actually persist".
//!
//! **Formats measured**
//! - csv (plain)
//! - csv.zst (zstd level 3)
//! - ndjson
//! - ndjson.zst (zstd level 3)
//! - parquet (Snappy, default)
//! - parquet (zstd level 3)
//! - avro (deflate) — using apache-avro 0.21, Apache-2.0 licensed
//! - helium self-contained default schema
//! - helium self-contained optimized schema
//! - helium catalog-mode default schema
//! - helium catalog-mode optimized schema
//!
//! **Dataset**
//! - If `HELIUM_PARQUET_PATH` is set, the first 10 000 rows are read from that
//!   file. Only flat-compatible columns (Primitive, Utf8, Binary,
//!   Nullable of those) are kept; nested types are projected out.
//! - Otherwise a synthetic 10 000-row × 8-column flat dataset is generated.
//!
//! **Round-trip verification**
//! Every Helium variant is written and read back; the decoded LogicalColumn
//! values are asserted equal to the input before the size is recorded.
//! External formats are verified non-empty (the writer did not error).
//!
//! Run:
//!   cargo test --test format_comparison_report --release --all-features -- --nocapture
//!
//! Output: stdout + target/format-comparison.md

#![cfg(all(
    feature = "schema-csv",
    feature = "schema-json",
    feature = "schema-parquet"
))]

use std::collections::HashMap;
use std::fmt::Write as FmtWrite;

use apache_avro::{Codec, DeflateSettings, Schema as AvroSchema, Writer as AvroWriter};
use parquet::basic::Type as PqPhysical;
use parquet::basic::{Compression as PqCompression, ConvertedType, Repetition, ZstdLevel};
use parquet::data_type::{ByteArray, ByteArrayType, DoubleType, FloatType, Int32Type, Int64Type};
use parquet::file::properties::WriterProperties;
use parquet::file::writer::SerializedFileWriter;

use helium::catalog::Catalog;
use helium::optimizer::Optimizer;
use helium::schema::encodings::default_encodings;
use helium::{
    CoderRegistry, CoderSpec, ColumnData, ColumnSpec, DataType, HeliumReader, HeliumWriter,
    LogicalColumn, LogicalType, Schema,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DEFAULT_MAX_ROWS: usize = 10_000;

/// Row cap for the report. Default 10k keeps the test under ~5 minutes.
/// Override with `HELIUM_REPORT_MAX_ROWS=1000000` (or any value) to measure
/// the full dataset. Set to e.g. `usize::MAX` for "no cap" (use file's row count).
fn max_rows() -> usize {
    std::env::var("HELIUM_REPORT_MAX_ROWS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_MAX_ROWS)
}

// ---------------------------------------------------------------------------
// Synthetic dataset (deterministic, flat, 8 columns)
// ---------------------------------------------------------------------------

fn synth_ts_i64(n: usize) -> Vec<i64> {
    (0..n).map(|i| 1_700_000_000_i64 + i as i64 * 30).collect()
}

fn synth_status_i32(n: usize) -> Vec<i32> {
    let codes = [200i32, 200, 200, 301, 304, 404, 500];
    let mut rng = 0xDEAD_BEEF_u64;
    (0..n)
        .map(|_| {
            rng = rng
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            codes[((rng >> 32) as usize) % codes.len()]
        })
        .collect()
}

fn synth_ids_u32(n: usize) -> Vec<u32> {
    let mut rng = 0x1337_u64;
    let mut v: u32 = 1;
    (0..n)
        .map(|_| {
            rng = rng
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            v = v.saturating_add(1 + (rng >> 32) as u32 % 5);
            v
        })
        .collect()
}

fn synth_temp_f64(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let t = i as f64 * 0.01;
            ((20.0 + t.sin() * 2.0) * 10.0).round() / 10.0
        })
        .collect()
}

fn synth_level_utf8(n: usize) -> Vec<String> {
    let levels = ["INFO", "DEBUG", "WARN", "ERROR"];
    let weights = [60u64, 25, 10, 5];
    let total: u64 = weights.iter().sum();
    let mut rng = 0x1234_5678_u64;
    (0..n)
        .map(|_| {
            rng = rng
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let r = (rng >> 32) % total;
            let mut acc = 0u64;
            for (i, &w) in weights.iter().enumerate() {
                acc += w;
                if r < acc {
                    return levels[i].to_string();
                }
            }
            levels[0].to_string()
        })
        .collect()
}

fn synth_host_utf8(n: usize) -> Vec<String> {
    let mut rng = 0xCAFE_BABE_u64;
    (0..n)
        .map(|_| {
            rng = rng
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let idx = (rng >> 32) % 500;
            format!("host-{idx:04}.example.internal")
        })
        .collect()
}

fn synth_nullable_i32(n: usize) -> (Vec<bool>, Vec<i32>) {
    let present: Vec<bool> = (0..n).map(|i| i % 5 != 0).collect();
    let values: Vec<i32> = (0..n)
        .filter(|i| i % 5 != 0)
        .map(|i| i as i32 * 3)
        .collect();
    (present, values)
}

fn synth_binary(n: usize) -> Vec<Vec<u8>> {
    let mut rng = 0xFEED_FACE_u64;
    (0..n)
        .map(|_| {
            rng = rng
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            vec![
                ((rng >> 24) & 0xff) as u8,
                ((rng >> 16) & 0xff) as u8,
                ((rng >> 8) & 0xff) as u8,
                (rng & 0xff) as u8,
            ]
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Dataset struct
// ---------------------------------------------------------------------------

struct FlatDataset {
    schema: Schema,
    columns: HashMap<String, LogicalColumn>,
    row_count: usize,
    col_count: usize,
    source_label: String,
}

// ---------------------------------------------------------------------------
// Build synthetic dataset
// ---------------------------------------------------------------------------

fn build_synthetic(n: usize) -> FlatDataset {
    let ts = synth_ts_i64(n);
    let status = synth_status_i32(n);
    let id = synth_ids_u32(n);
    let temp = synth_temp_f64(n);
    let level = synth_level_utf8(n);
    let host = synth_host_utf8(n);
    let (np, nv) = synth_nullable_i32(n);
    let blobs = synth_binary(n);

    let lt_ts = LogicalType::Primitive {
        data_type: DataType::I64,
    };
    let lt_status = LogicalType::Primitive {
        data_type: DataType::I32,
    };
    let lt_id = LogicalType::Primitive {
        data_type: DataType::U32,
    };
    let lt_temp = LogicalType::Primitive {
        data_type: DataType::F64,
    };
    let lt_level = LogicalType::Utf8;
    let lt_host = LogicalType::Utf8;
    let lt_maybe = LogicalType::Nullable {
        inner: Box::new(LogicalType::Primitive {
            data_type: DataType::I32,
        }),
    };
    let lt_blob = LogicalType::Binary;

    let schema = Schema::new(vec![
        ColumnSpec::new("ts".to_string(), lt_ts.clone(), default_encodings(&lt_ts)),
        ColumnSpec::new(
            "status".to_string(),
            lt_status.clone(),
            default_encodings(&lt_status),
        ),
        ColumnSpec::new("id".to_string(), lt_id.clone(), default_encodings(&lt_id)),
        ColumnSpec::new(
            "temp".to_string(),
            lt_temp.clone(),
            default_encodings(&lt_temp),
        ),
        ColumnSpec::new(
            "level".to_string(),
            lt_level.clone(),
            default_encodings(&lt_level),
        ),
        ColumnSpec::new(
            "host".to_string(),
            lt_host.clone(),
            default_encodings(&lt_host),
        ),
        ColumnSpec::new(
            "maybe_val".to_string(),
            lt_maybe.clone(),
            default_encodings(&lt_maybe),
        ),
        ColumnSpec::new(
            "blob".to_string(),
            lt_blob.clone(),
            default_encodings(&lt_blob),
        ),
    ]);

    let mut cols: HashMap<String, LogicalColumn> = HashMap::new();
    cols.insert(
        "ts".to_string(),
        LogicalColumn::Primitive(ColumnData::I64(ts)),
    );
    cols.insert(
        "status".to_string(),
        LogicalColumn::Primitive(ColumnData::I32(status)),
    );
    cols.insert(
        "id".to_string(),
        LogicalColumn::Primitive(ColumnData::U32(id)),
    );
    cols.insert(
        "temp".to_string(),
        LogicalColumn::Primitive(ColumnData::F64(temp)),
    );
    cols.insert("level".to_string(), LogicalColumn::Utf8(level));
    cols.insert("host".to_string(), LogicalColumn::Utf8(host));
    cols.insert(
        "maybe_val".to_string(),
        LogicalColumn::Nullable {
            present: np,
            value: Box::new(LogicalColumn::Primitive(ColumnData::I32(nv))),
        },
    );
    cols.insert("blob".to_string(), LogicalColumn::Binary(blobs));

    FlatDataset {
        schema,
        columns: cols,
        row_count: n,
        col_count: 8,
        source_label: format!(
            "synthetic ({n} rows × 8 cols: ts_i64, status_i32, id_u32, temp_f64, \
             level_utf8, host_utf8, maybe_val_nullable_i32, blob_binary)"
        ),
    }
}

// ---------------------------------------------------------------------------
// Load from Parquet (project to flat-only columns, first max_rows() rows)
// ---------------------------------------------------------------------------

/// Returns true if `lt` is a flat type that the parquet writer and Avro writer support.
fn is_flat_compatible(lt: &LogicalType) -> bool {
    match lt {
        LogicalType::Primitive { .. } | LogicalType::Utf8 | LogicalType::Binary => true,
        LogicalType::Nullable { inner } => is_flat_leaf(inner),
        _ => false,
    }
}

fn is_flat_leaf(lt: &LogicalType) -> bool {
    matches!(
        lt,
        LogicalType::Primitive { .. } | LogicalType::Utf8 | LogicalType::Binary
    )
}

fn load_from_parquet(path: &str) -> FlatDataset {
    use parquet::file::reader::{FileReader, SerializedFileReader};
    use std::fs::File;

    let file = File::open(path).expect("open parquet");
    let reader = SerializedFileReader::new(file).expect("parquet reader");
    let total_rows = reader.metadata().file_metadata().num_rows() as usize;
    let n = total_rows.min(max_rows());
    let total_cols = reader
        .metadata()
        .file_metadata()
        .schema()
        .get_fields()
        .len();

    // Infer schema and project to flat types
    let full_schema = helium::schema::parquet::schema_from_parquet(std::path::Path::new(path))
        .expect("schema_from_parquet");
    let flat_specs: Vec<ColumnSpec> = full_schema
        .columns
        .into_iter()
        .filter(|spec| is_flat_compatible(&spec.logical_type))
        .collect();
    let dropped = total_cols - flat_specs.len();
    eprintln!(
        "  parquet: {} total cols, {} flat-compatible, {} dropped (nested), sampling {} rows",
        total_cols,
        flat_specs.len(),
        dropped,
        n
    );

    let col_names: Vec<String> = flat_specs.iter().map(|s| s.name.clone()).collect();
    let flat_schema = Schema::new(flat_specs);

    // Build per-column accumulators
    // Using the row-based parquet reader (simplest for mixed-type columns)
    let file2 = File::open(path).expect("open parquet 2");
    let reader2 = SerializedFileReader::new(file2).expect("parquet reader 2");

    // Per-column storage buffers
    let mut i64_buf: HashMap<String, Vec<i64>> = HashMap::new();
    let mut i32_buf: HashMap<String, Vec<i32>> = HashMap::new();
    let mut u32_buf: HashMap<String, Vec<u32>> = HashMap::new();
    let mut f64_buf: HashMap<String, Vec<f64>> = HashMap::new();
    let mut f32_buf: HashMap<String, Vec<f32>> = HashMap::new();
    let mut u8_buf: HashMap<String, Vec<u8>> = HashMap::new();
    let mut utf8_buf: HashMap<String, Vec<String>> = HashMap::new();
    let mut bin_buf: HashMap<String, Vec<Vec<u8>>> = HashMap::new();
    let mut present_buf: HashMap<String, Vec<bool>> = HashMap::new();

    for name in &col_names {
        i64_buf.insert(name.clone(), Vec::new());
        i32_buf.insert(name.clone(), Vec::new());
        u32_buf.insert(name.clone(), Vec::new());
        f64_buf.insert(name.clone(), Vec::new());
        f32_buf.insert(name.clone(), Vec::new());
        u8_buf.insert(name.clone(), Vec::new());
        utf8_buf.insert(name.clone(), Vec::new());
        bin_buf.insert(name.clone(), Vec::new());
        present_buf.insert(name.clone(), Vec::new());
    }

    let spec_map: HashMap<&str, &ColumnSpec> = flat_schema
        .columns
        .iter()
        .map(|s| (s.name.as_str(), s))
        .collect();

    let row_iter = reader2.get_row_iter(None).expect("row iter");
    for row_result in row_iter.take(n) {
        let row = row_result.expect("row read");
        for name in &col_names {
            let spec = spec_map[name.as_str()];
            let field = row
                .get_column_iter()
                .find(|(k, _)| *k == name.as_str())
                .map(|(_, v)| v);
            accumulate_field(
                name,
                &spec.logical_type,
                field,
                &mut i64_buf,
                &mut i32_buf,
                &mut u32_buf,
                &mut f64_buf,
                &mut f32_buf,
                &mut u8_buf,
                &mut utf8_buf,
                &mut bin_buf,
                &mut present_buf,
            );
        }
    }

    // Build LogicalColumns
    let mut columns: HashMap<String, LogicalColumn> = HashMap::new();
    for spec in &flat_schema.columns {
        let name = &spec.name;
        let lc = build_logical_column(
            &spec.logical_type,
            name,
            &mut i64_buf,
            &mut i32_buf,
            &mut u32_buf,
            &mut f64_buf,
            &mut f32_buf,
            &mut u8_buf,
            &mut utf8_buf,
            &mut bin_buf,
            &mut present_buf,
        );
        columns.insert(name.clone(), lc);
    }

    let source_label = format!(
        "{} (first {n} rows × {} flat cols, {dropped} nested cols dropped)",
        std::path::Path::new(path)
            .file_name()
            .unwrap_or_default()
            .to_string_lossy(),
        flat_schema.columns.len(),
    );
    let col_count = flat_schema.columns.len();

    FlatDataset {
        schema: flat_schema,
        columns,
        row_count: n,
        col_count,
        source_label,
    }
}

#[allow(clippy::too_many_arguments)]
fn accumulate_field(
    name: &str,
    lt: &LogicalType,
    field: Option<&parquet::record::Field>,
    i64_buf: &mut HashMap<String, Vec<i64>>,
    i32_buf: &mut HashMap<String, Vec<i32>>,
    u32_buf: &mut HashMap<String, Vec<u32>>,
    f64_buf: &mut HashMap<String, Vec<f64>>,
    f32_buf: &mut HashMap<String, Vec<f32>>,
    u8_buf: &mut HashMap<String, Vec<u8>>,
    utf8_buf: &mut HashMap<String, Vec<String>>,
    bin_buf: &mut HashMap<String, Vec<Vec<u8>>>,
    present_buf: &mut HashMap<String, Vec<bool>>,
) {
    use parquet::record::Field;

    let is_null = matches!(field, Some(Field::Null) | None);

    match lt {
        LogicalType::Primitive { data_type } => match data_type {
            DataType::I64 | DataType::U64 => {
                let v = match field {
                    Some(Field::Long(x)) => *x,
                    Some(Field::Int(x)) => *x as i64,
                    Some(Field::ULong(x)) => *x as i64,
                    _ => 0,
                };
                i64_buf.get_mut(name).unwrap().push(v);
            }
            DataType::I32 | DataType::I16 | DataType::I8 => {
                let v = match field {
                    Some(Field::Int(x)) => *x,
                    Some(Field::Short(x)) => *x as i32,
                    Some(Field::Byte(x)) => *x as i32,
                    _ => 0,
                };
                i32_buf.get_mut(name).unwrap().push(v);
            }
            DataType::U32 => {
                let v = match field {
                    Some(Field::UInt(x)) => *x,
                    Some(Field::Long(x)) => *x as u32,
                    Some(Field::Int(x)) => *x as u32,
                    _ => 0,
                };
                u32_buf.get_mut(name).unwrap().push(v);
            }
            DataType::U16 => {
                let v = match field {
                    Some(Field::UShort(x)) => *x as i32,
                    Some(Field::Int(x)) => *x,
                    _ => 0,
                };
                i32_buf.get_mut(name).unwrap().push(v);
            }
            DataType::U8 => {
                let v = match field {
                    Some(Field::Bool(x)) => *x as u8,
                    Some(Field::UByte(x)) => *x,
                    Some(Field::Byte(x)) => *x as u8,
                    Some(Field::Int(x)) => *x as u8,
                    _ => 0,
                };
                u8_buf.get_mut(name).unwrap().push(v);
            }
            DataType::F64 => {
                let v = match field {
                    Some(Field::Double(x)) => *x,
                    Some(Field::Float(x)) => *x as f64,
                    _ => 0.0,
                };
                f64_buf.get_mut(name).unwrap().push(v);
            }
            DataType::F32 => {
                let v = match field {
                    Some(Field::Float(x)) => *x,
                    Some(Field::Double(x)) => *x as f32,
                    _ => 0.0,
                };
                f32_buf.get_mut(name).unwrap().push(v);
            }
            DataType::Bytes => {
                let v = match field {
                    Some(Field::Bytes(b)) => b.data().to_vec(),
                    _ => vec![],
                };
                bin_buf.get_mut(name).unwrap().push(v);
            }
        },
        LogicalType::Utf8 => {
            let v = match field {
                Some(Field::Str(s)) => s.clone(),
                Some(Field::Bytes(b)) => String::from_utf8_lossy(b.data()).into_owned(),
                _ => String::new(),
            };
            utf8_buf.get_mut(name).unwrap().push(v);
        }
        LogicalType::Binary => {
            let v = match field {
                Some(Field::Bytes(b)) => b.data().to_vec(),
                Some(Field::Str(s)) => s.as_bytes().to_vec(),
                _ => vec![],
            };
            bin_buf.get_mut(name).unwrap().push(v);
        }
        LogicalType::Nullable { inner } => {
            present_buf.get_mut(name).unwrap().push(!is_null);
            if !is_null {
                let inner_lt = inner.as_ref().clone();
                accumulate_field(
                    name,
                    &inner_lt,
                    field,
                    i64_buf,
                    i32_buf,
                    u32_buf,
                    f64_buf,
                    f32_buf,
                    u8_buf,
                    utf8_buf,
                    bin_buf,
                    present_buf,
                );
            }
        }
        _ => {}
    }
}

#[allow(clippy::too_many_arguments)]
fn build_logical_column(
    lt: &LogicalType,
    name: &str,
    i64_buf: &mut HashMap<String, Vec<i64>>,
    i32_buf: &mut HashMap<String, Vec<i32>>,
    u32_buf: &mut HashMap<String, Vec<u32>>,
    f64_buf: &mut HashMap<String, Vec<f64>>,
    f32_buf: &mut HashMap<String, Vec<f32>>,
    u8_buf: &mut HashMap<String, Vec<u8>>,
    utf8_buf: &mut HashMap<String, Vec<String>>,
    bin_buf: &mut HashMap<String, Vec<Vec<u8>>>,
    present_buf: &mut HashMap<String, Vec<bool>>,
) -> LogicalColumn {
    match lt {
        LogicalType::Primitive { data_type } => {
            let cd = match data_type {
                DataType::I64 | DataType::U64 => {
                    ColumnData::I64(i64_buf.remove(name).unwrap_or_default())
                }
                DataType::I32 | DataType::I16 | DataType::I8 | DataType::U16 => {
                    ColumnData::I32(i32_buf.remove(name).unwrap_or_default())
                }
                DataType::U32 => ColumnData::U32(u32_buf.remove(name).unwrap_or_default()),
                DataType::F64 => ColumnData::F64(f64_buf.remove(name).unwrap_or_default()),
                DataType::F32 => ColumnData::F32(f32_buf.remove(name).unwrap_or_default()),
                DataType::U8 => ColumnData::U8(u8_buf.remove(name).unwrap_or_default()),
                DataType::Bytes => ColumnData::Bytes(
                    bin_buf
                        .remove(name)
                        .unwrap_or_default()
                        .into_iter()
                        .flatten()
                        .collect(),
                ),
            };
            LogicalColumn::Primitive(cd)
        }
        LogicalType::Utf8 => LogicalColumn::Utf8(utf8_buf.remove(name).unwrap_or_default()),
        LogicalType::Binary => LogicalColumn::Binary(bin_buf.remove(name).unwrap_or_default()),
        LogicalType::Nullable { inner } => {
            let present = present_buf.remove(name).unwrap_or_default();
            // Build only the non-null rows as the inner LogicalColumn.
            // The ColumnData variant must exactly match the logical type's
            // data_type — the writer validates this at write time.
            let inner_lc = match inner.as_ref() {
                LogicalType::Primitive { data_type } => {
                    let raw_i32 = i32_buf.remove(name).unwrap_or_default();
                    let cd = match data_type {
                        DataType::I64 | DataType::U64 => {
                            ColumnData::I64(i64_buf.remove(name).unwrap_or_default())
                        }
                        DataType::I32 => ColumnData::I32(raw_i32),
                        DataType::I16 => {
                            ColumnData::I16(raw_i32.into_iter().map(|x| x as i16).collect())
                        }
                        DataType::I8 => {
                            ColumnData::I8(raw_i32.into_iter().map(|x| x as i8).collect())
                        }
                        DataType::U16 => {
                            ColumnData::U16(raw_i32.into_iter().map(|x| x as u16).collect())
                        }
                        DataType::U32 => ColumnData::U32(u32_buf.remove(name).unwrap_or_default()),
                        DataType::F64 => ColumnData::F64(f64_buf.remove(name).unwrap_or_default()),
                        DataType::F32 => ColumnData::F32(f32_buf.remove(name).unwrap_or_default()),
                        DataType::U8 => ColumnData::U8(u8_buf.remove(name).unwrap_or_default()),
                        DataType::Bytes => ColumnData::Bytes(
                            bin_buf
                                .remove(name)
                                .unwrap_or_default()
                                .into_iter()
                                .flatten()
                                .collect(),
                        ),
                    };
                    LogicalColumn::Primitive(cd)
                }
                LogicalType::Utf8 => LogicalColumn::Utf8(utf8_buf.remove(name).unwrap_or_default()),
                LogicalType::Binary => {
                    LogicalColumn::Binary(bin_buf.remove(name).unwrap_or_default())
                }
                _ => LogicalColumn::Utf8(vec![]),
            };
            LogicalColumn::Nullable {
                present,
                value: Box::new(inner_lc),
            }
        }
        _ => LogicalColumn::Utf8(vec![]),
    }
}

// ---------------------------------------------------------------------------
// Dataset loader
// ---------------------------------------------------------------------------

fn load_dataset() -> FlatDataset {
    if let Ok(path) = std::env::var("HELIUM_PARQUET_PATH") {
        eprintln!("Loading from HELIUM_PARQUET_PATH={path}");
        return load_from_parquet(&path);
    }
    build_synthetic(max_rows())
}

// ---------------------------------------------------------------------------
// Format helpers
// ---------------------------------------------------------------------------

fn fmt_bytes(b: usize) -> String {
    if b >= 1024 * 1024 {
        format!("{:.2} MB", b as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.1} KB", b as f64 / 1024.0)
    }
}

fn fmt_ratio(bytes: usize, baseline: usize) -> String {
    if baseline == 0 {
        return "—".into();
    }
    format!("{:.2}x", baseline as f64 / bytes as f64)
}

// ---------------------------------------------------------------------------
// External format writers
// ---------------------------------------------------------------------------

/// LZ4 block-format compression (prepends original size, uses lz4_flex).
fn lz4_compress(data: &[u8]) -> Vec<u8> {
    lz4_flex::block::compress_prepend_size(data)
}

/// Serialise every flat column in the dataset to a single raw byte stream:
///
/// - Primitive: little-endian native bytes, no framing.
/// - Utf8/Binary: for each value, 4-byte LE length followed by content bytes.
/// - Nullable: 1 byte per row present mask
///   (0x00/0x01) followed by the compacted non-null values.
/// - Nested types (Struct/List/Map/Union/Dict): skipped with a note; the
///   function logs which columns were omitted.
///
/// No column boundary markers, no schema header. This is the lower-bound
/// target: "what if you had no schema and just ran a general-purpose
/// compressor on the raw numbers?"
fn write_raw_column_bytes(dataset: &FlatDataset) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::new();
    for spec in &dataset.schema.columns {
        let lc = &dataset.columns[&spec.name];
        append_lc_raw(&mut out, lc);
    }
    out
}

fn append_lc_raw(out: &mut Vec<u8>, lc: &LogicalColumn) {
    match lc {
        LogicalColumn::Primitive(cd) => append_cd_raw(out, cd),
        LogicalColumn::Utf8(strings) => {
            for s in strings {
                let b = s.as_bytes();
                out.extend_from_slice(&(b.len() as u32).to_le_bytes());
                out.extend_from_slice(b);
            }
        }
        LogicalColumn::Binary(blobs) => {
            for b in blobs {
                out.extend_from_slice(&(b.len() as u32).to_le_bytes());
                out.extend_from_slice(b);
            }
        }
        LogicalColumn::Nullable { present, value } => {
            for &p in present {
                out.push(p as u8);
            }
            append_lc_raw(out, value);
        }
        // Nested types (Struct/List/Map/Union/Dict) are skipped — the flat
        // dataset produced by this test should not contain them, but we
        // handle the arms gracefully rather than panicking.
        _ => {}
    }
}

fn append_cd_raw(out: &mut Vec<u8>, cd: &ColumnData) {
    match cd {
        ColumnData::I8(v) => out.extend_from_slice(bytemuck_i8_as_u8(v)),
        ColumnData::I16(v) => {
            for x in v {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        ColumnData::I32(v) => {
            for x in v {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        ColumnData::I64(v) => {
            for x in v {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        ColumnData::U8(v) => out.extend_from_slice(v),
        ColumnData::U16(v) => {
            for x in v {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        ColumnData::U32(v) => {
            for x in v {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        ColumnData::U64(v) => {
            for x in v {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        ColumnData::F32(v) => {
            for x in v {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        ColumnData::F64(v) => {
            for x in v {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        ColumnData::Bytes(v) => out.extend_from_slice(v),
    }
}

/// Cast `&[i8]` to `&[u8]` for raw byte output without a copy.
fn bytemuck_i8_as_u8(v: &[i8]) -> &[u8] {
    // SAFETY: i8 and u8 have identical layout; both are 1-byte aligned.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len()) }
}

/// Build a schema identical to `src` but with the terminal `"zstd"` coder in
/// every pipeline replaced by `"lz4"`.
///
/// Pipeline replacement rule:
/// - If the pipeline's last coder ID is `"zstd"`, replace it with `"lz4"`.
/// - Otherwise leave the pipeline unchanged (no terminal zstd to replace).
fn lz4_terminal_schema(src: &Schema) -> Schema {
    let columns = src
        .columns
        .iter()
        .map(|spec| {
            let encodings = spec
                .encodings
                .iter()
                .map(|pipeline| replace_terminal_zstd_with_lz4(pipeline))
                .collect();
            ColumnSpec {
                name: spec.name.clone(),
                logical_type: spec.logical_type.clone(),
                encodings,
            }
        })
        .collect();
    Schema::new(columns)
}

fn replace_terminal_zstd_with_lz4(pipeline: &[CoderSpec]) -> Vec<CoderSpec> {
    let mut out = pipeline.to_vec();
    if let Some(last) = out.last_mut()
        && last.id == "zstd"
    {
        *last = CoderSpec::new("lz4");
    }
    out
}

fn write_csv_bytes(dataset: &FlatDataset) -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();
    helium::schema::csv::write_csv(&dataset.schema, &dataset.columns, &mut buf).expect("write_csv");
    assert!(!buf.is_empty(), "csv output was empty");
    buf
}

fn write_ndjson_bytes(dataset: &FlatDataset) -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();
    helium::schema::json::write_json(&dataset.schema, &dataset.columns, &mut buf)
        .expect("write_json");
    assert!(!buf.is_empty(), "ndjson output was empty");
    buf
}

/// zstd level 3 (zstd's default — same level used elsewhere in the codebase
/// for schema/footer compression). Picked to be apples-to-apples with
/// `parquet (zstd)` row in this report.
fn zstd_compress(data: &[u8]) -> Vec<u8> {
    zstd::encode_all(data, 3).expect("zstd encode")
}

fn write_parquet_bytes(dataset: &FlatDataset, compression: PqCompression) -> Vec<u8> {
    // Write to a temp file and read back bytes (parquet writer needs Seek).
    let tmp = tempfile::NamedTempFile::new().expect("tempfile parquet");
    let props = std::sync::Arc::new(
        WriterProperties::builder()
            .set_compression(compression)
            .build(),
    );

    // Build Parquet schema fields
    use std::sync::Arc;
    let pq_fields: Vec<Arc<parquet::schema::types::Type>> = dataset
        .schema
        .columns
        .iter()
        .map(|s| helium_lt_to_pq_field(&s.name, &s.logical_type))
        .collect();
    let root = Arc::new(
        parquet::schema::types::Type::group_type_builder("schema")
            .with_fields(pq_fields)
            .build()
            .expect("pq schema"),
    );

    let file = tmp.as_file().try_clone().expect("clone file");
    let mut fw = SerializedFileWriter::new(file, root, props).expect("pq writer");
    let mut rg = fw.next_row_group().expect("row group");

    for spec in &dataset.schema.columns {
        let lc = &dataset.columns[&spec.name];
        let mut cw = rg.next_column().expect("next col").expect("col some");
        write_pq_col(&mut cw, &spec.logical_type, lc);
        cw.close().expect("col close");
    }
    rg.close().expect("rg close");
    fw.close().expect("fw close");

    let bytes = std::fs::read(tmp.path()).expect("read parquet");
    assert!(!bytes.is_empty(), "parquet output was empty");
    bytes
}

fn helium_lt_to_pq_field(
    name: &str,
    lt: &LogicalType,
) -> std::sync::Arc<parquet::schema::types::Type> {
    use std::sync::Arc;
    match lt {
        LogicalType::Primitive { data_type } => {
            let (phys, conv) = dt_to_pq(*data_type);
            Arc::new(
                parquet::schema::types::Type::primitive_type_builder(name, phys)
                    .with_repetition(Repetition::REQUIRED)
                    .with_converted_type(conv)
                    .build()
                    .expect("pq prim field"),
            )
        }
        LogicalType::Utf8 => Arc::new(
            parquet::schema::types::Type::primitive_type_builder(name, PqPhysical::BYTE_ARRAY)
                .with_repetition(Repetition::REQUIRED)
                .with_converted_type(ConvertedType::UTF8)
                .build()
                .expect("pq utf8 field"),
        ),
        LogicalType::Binary => Arc::new(
            parquet::schema::types::Type::primitive_type_builder(name, PqPhysical::BYTE_ARRAY)
                .with_repetition(Repetition::REQUIRED)
                .build()
                .expect("pq binary field"),
        ),
        LogicalType::Nullable { inner }
            if matches!(inner.as_ref(), LogicalType::Primitive { .. }) =>
        {
            let dt = match inner.as_ref() {
                LogicalType::Primitive { data_type } => *data_type,
                _ => unreachable!(),
            };
            let (phys, conv) = dt_to_pq(dt);
            Arc::new(
                parquet::schema::types::Type::primitive_type_builder(name, phys)
                    .with_repetition(Repetition::OPTIONAL)
                    .with_converted_type(conv)
                    .build()
                    .expect("pq nullable prim field"),
            )
        }
        LogicalType::Nullable { inner } if matches!(inner.as_ref(), LogicalType::Utf8) => Arc::new(
            parquet::schema::types::Type::primitive_type_builder(name, PqPhysical::BYTE_ARRAY)
                .with_repetition(Repetition::OPTIONAL)
                .with_converted_type(ConvertedType::UTF8)
                .build()
                .expect("pq nullable utf8 field"),
        ),
        LogicalType::Nullable { inner } if matches!(inner.as_ref(), LogicalType::Binary) => {
            Arc::new(
                parquet::schema::types::Type::primitive_type_builder(name, PqPhysical::BYTE_ARRAY)
                    .with_repetition(Repetition::OPTIONAL)
                    .build()
                    .expect("pq nullable binary field"),
            )
        }
        _ => {
            // Flat schema should not have nested types
            Arc::new(
                parquet::schema::types::Type::primitive_type_builder(name, PqPhysical::BYTE_ARRAY)
                    .with_repetition(Repetition::REQUIRED)
                    .build()
                    .expect("pq fallback field"),
            )
        }
    }
}

fn dt_to_pq(dt: DataType) -> (PqPhysical, ConvertedType) {
    match dt {
        DataType::I8 => (PqPhysical::INT32, ConvertedType::INT_8),
        DataType::I16 => (PqPhysical::INT32, ConvertedType::INT_16),
        DataType::I32 => (PqPhysical::INT32, ConvertedType::INT_32),
        DataType::I64 => (PqPhysical::INT64, ConvertedType::INT_64),
        DataType::U8 => (PqPhysical::INT32, ConvertedType::UINT_8),
        DataType::U16 => (PqPhysical::INT32, ConvertedType::UINT_16),
        // U32 and U64 must use INT64 physical type (no UINT_32 annotation on INT64 in parquet-rs)
        DataType::U32 => (PqPhysical::INT64, ConvertedType::NONE),
        DataType::U64 => (PqPhysical::INT64, ConvertedType::NONE),
        DataType::F32 => (PqPhysical::FLOAT, ConvertedType::NONE),
        DataType::F64 => (PqPhysical::DOUBLE, ConvertedType::NONE),
        DataType::Bytes => (PqPhysical::BYTE_ARRAY, ConvertedType::NONE),
    }
}

fn write_pq_col(
    cw: &mut parquet::file::writer::SerializedColumnWriter<'_>,
    lt: &LogicalType,
    lc: &LogicalColumn,
) {
    match (lt, lc) {
        // ---- non-nullable primitives ----
        (LogicalType::Primitive { .. }, LogicalColumn::Primitive(cd)) => {
            write_pq_cd(cw, cd, None);
        }
        (LogicalType::Utf8, LogicalColumn::Utf8(strings)) => {
            let data: Vec<ByteArray> = strings
                .iter()
                .map(|s| ByteArray::from(s.as_bytes()))
                .collect();
            cw.typed::<ByteArrayType>()
                .write_batch(&data, None, None)
                .expect("utf8 write");
        }
        (LogicalType::Binary, LogicalColumn::Binary(blobs)) => {
            let data: Vec<ByteArray> = blobs
                .iter()
                .map(|b| ByteArray::from(b.as_slice()))
                .collect();
            cw.typed::<ByteArrayType>()
                .write_batch(&data, None, None)
                .expect("binary write");
        }

        // ---- recursive Nullable (the form produced by load_from_parquet for ClickBench) ----
        // The inner LogicalColumn holds only the non-null rows; we must expand to
        // all rows, injecting a placeholder for nulls.
        (LogicalType::Nullable { inner }, LogicalColumn::Nullable { present, value }) => {
            let def: Vec<i16> = present.iter().map(|&p| if p { 1 } else { 0 }).collect();
            match (inner.as_ref(), value.as_ref()) {
                (LogicalType::Primitive { .. }, LogicalColumn::Primitive(cd)) => {
                    write_pq_cd_expanded(cw, cd, present, &def);
                }
                (LogicalType::Utf8, LogicalColumn::Utf8(strings)) => {
                    let expanded =
                        expand_compact_ba(present, strings, |s| ByteArray::from(s.as_bytes()));
                    cw.typed::<ByteArrayType>()
                        .write_batch(&expanded, Some(&def), None)
                        .expect("nullable utf8 (recursive) write");
                }
                (LogicalType::Binary, LogicalColumn::Binary(blobs)) => {
                    let expanded =
                        expand_compact_ba(present, blobs, |b| ByteArray::from(b.as_slice()));
                    cw.typed::<ByteArrayType>()
                        .write_batch(&expanded, Some(&def), None)
                        .expect("nullable binary (recursive) write");
                }
                (inner_lt, inner_lc) => {
                    panic!(
                        "write_pq_col: unhandled recursive Nullable inner: lt={inner_lt:?} lc={}",
                        lc_variant_name(inner_lc)
                    );
                }
            }
        }

        (lt, lc) => {
            panic!(
                "write_pq_col: unhandled combination lt={lt:?} lc_variant={}",
                lc_variant_name(lc)
            );
        }
    }
}

/// Expand a compact (non-null-only) ColumnData to full row count for parquet writing.
/// The `present` bitmap indicates which rows have values; nulls get 0 / 0.0 / empty.
fn write_pq_cd_expanded(
    cw: &mut parquet::file::writer::SerializedColumnWriter<'_>,
    cd: &ColumnData,
    present: &[bool],
    def: &[i16],
) {
    match cd {
        ColumnData::I8(v) => {
            let expanded = expand_compact_typed(present, v, 0i8, |x| *x as i32);
            cw.typed::<Int32Type>()
                .write_batch(&expanded, Some(def), None)
                .expect("i8 exp");
        }
        ColumnData::I16(v) => {
            let expanded = expand_compact_typed(present, v, 0i16, |x| *x as i32);
            cw.typed::<Int32Type>()
                .write_batch(&expanded, Some(def), None)
                .expect("i16 exp");
        }
        ColumnData::I32(v) => {
            let expanded = expand_compact_typed(present, v, 0i32, |x| *x);
            cw.typed::<Int32Type>()
                .write_batch(&expanded, Some(def), None)
                .expect("i32 exp");
        }
        ColumnData::I64(v) => {
            let expanded = expand_compact_typed(present, v, 0i64, |x| *x);
            cw.typed::<Int64Type>()
                .write_batch(&expanded, Some(def), None)
                .expect("i64 exp");
        }
        ColumnData::U8(v) => {
            let expanded = expand_compact_typed(present, v, 0u8, |x| *x as i32);
            cw.typed::<Int32Type>()
                .write_batch(&expanded, Some(def), None)
                .expect("u8 exp");
        }
        ColumnData::U16(v) => {
            let expanded = expand_compact_typed(present, v, 0u16, |x| *x as i32);
            cw.typed::<Int32Type>()
                .write_batch(&expanded, Some(def), None)
                .expect("u16 exp");
        }
        ColumnData::U32(v) => {
            let expanded = expand_compact_typed(present, v, 0u32, |x| *x as i64);
            cw.typed::<Int64Type>()
                .write_batch(&expanded, Some(def), None)
                .expect("u32 exp");
        }
        ColumnData::U64(v) => {
            let expanded = expand_compact_typed(present, v, 0u64, |x| *x as i64);
            cw.typed::<Int64Type>()
                .write_batch(&expanded, Some(def), None)
                .expect("u64 exp");
        }
        ColumnData::F32(v) => {
            let expanded = expand_compact_typed(present, v, 0f32, |x| *x);
            cw.typed::<FloatType>()
                .write_batch(&expanded, Some(def), None)
                .expect("f32 exp");
        }
        ColumnData::F64(v) => {
            let expanded = expand_compact_typed(present, v, 0f64, |x| *x);
            cw.typed::<DoubleType>()
                .write_batch(&expanded, Some(def), None)
                .expect("f64 exp");
        }
        ColumnData::Bytes(v) => {
            // Bytes is a single blob; treat as one BYTE_ARRAY with nullability
            let ba = ByteArray::from(v.as_slice());
            cw.typed::<ByteArrayType>()
                .write_batch(&[ba], Some(def), None)
                .expect("bytes exp");
        }
    }
}

/// Expand compact (non-null-only) typed slice to full-row-count Vec<O>,
/// inserting `default_val` at null positions.
fn expand_compact_typed<T: Copy, O, F>(
    present: &[bool],
    compact: &[T],
    default_val: T,
    map: F,
) -> Vec<O>
where
    F: Fn(&T) -> O,
{
    let mut out = Vec::with_capacity(present.len());
    let mut idx = 0usize;
    for &p in present {
        if p {
            out.push(map(&compact[idx]));
            idx += 1;
        } else {
            out.push(map(&default_val));
        }
    }
    out
}

/// Expand compact (non-null-only) slices to full row count for parquet writing,
/// inserting a default ByteArray at each null position.
fn expand_compact_ba<T, F>(present: &[bool], compact: &[T], to_ba: F) -> Vec<ByteArray>
where
    F: Fn(&T) -> ByteArray,
{
    let mut out = Vec::with_capacity(present.len());
    let mut idx = 0usize;
    for &p in present {
        if p {
            out.push(to_ba(&compact[idx]));
            idx += 1;
        } else {
            out.push(ByteArray::default());
        }
    }
    out
}

fn lc_variant_name(lc: &LogicalColumn) -> &'static str {
    match lc {
        LogicalColumn::Primitive(_) => "Primitive",
        LogicalColumn::Utf8(_) => "Utf8",
        LogicalColumn::Binary(_) => "Binary",
        LogicalColumn::Dictionary { .. } => "Dictionary",
        LogicalColumn::Struct { .. } => "Struct",
        LogicalColumn::List { .. } => "List",
        LogicalColumn::Map { .. } => "Map",
        LogicalColumn::Nullable { .. } => "Nullable",
        LogicalColumn::Union { .. } => "Union",
        LogicalColumn::Decimal128 { .. } => "Decimal128",
        LogicalColumn::Date32 { .. } => "Date32",
        LogicalColumn::Date64 { .. } => "Date64",
        LogicalColumn::Datetime { .. } => "Datetime",
    }
}

fn write_pq_cd(
    cw: &mut parquet::file::writer::SerializedColumnWriter<'_>,
    cd: &ColumnData,
    def: Option<&[i16]>,
) {
    match cd {
        ColumnData::I8(v) => {
            let d: Vec<i32> = v.iter().map(|x| *x as i32).collect();
            cw.typed::<Int32Type>()
                .write_batch(&d, def, None)
                .expect("i8 write");
        }
        ColumnData::I16(v) => {
            let d: Vec<i32> = v.iter().map(|x| *x as i32).collect();
            cw.typed::<Int32Type>()
                .write_batch(&d, def, None)
                .expect("i16 write");
        }
        ColumnData::I32(v) => {
            cw.typed::<Int32Type>()
                .write_batch(v, def, None)
                .expect("i32 write");
        }
        ColumnData::I64(v) => {
            cw.typed::<Int64Type>()
                .write_batch(v, def, None)
                .expect("i64 write");
        }
        ColumnData::U8(v) => {
            let d: Vec<i32> = v.iter().map(|x| *x as i32).collect();
            cw.typed::<Int32Type>()
                .write_batch(&d, def, None)
                .expect("u8 write");
        }
        ColumnData::U16(v) => {
            let d: Vec<i32> = v.iter().map(|x| *x as i32).collect();
            cw.typed::<Int32Type>()
                .write_batch(&d, def, None)
                .expect("u16 write");
        }
        ColumnData::U32(v) => {
            let d: Vec<i64> = v.iter().map(|x| *x as i64).collect();
            cw.typed::<Int64Type>()
                .write_batch(&d, def, None)
                .expect("u32 write");
        }
        ColumnData::U64(v) => {
            let d: Vec<i64> = v.iter().map(|x| *x as i64).collect();
            cw.typed::<Int64Type>()
                .write_batch(&d, def, None)
                .expect("u64 write");
        }
        ColumnData::F32(v) => {
            cw.typed::<FloatType>()
                .write_batch(v, def, None)
                .expect("f32 write");
        }
        ColumnData::F64(v) => {
            cw.typed::<DoubleType>()
                .write_batch(v, def, None)
                .expect("f64 write");
        }
        ColumnData::Bytes(v) => {
            let ba = ByteArray::from(v.as_slice());
            cw.typed::<ByteArrayType>()
                .write_batch(&[ba], def, None)
                .expect("bytes write");
        }
    }
}

// ---------------------------------------------------------------------------
// Avro writer
// ---------------------------------------------------------------------------

fn write_avro_deflate_bytes(dataset: &FlatDataset) -> Result<Vec<u8>, String> {
    let avro_schema_json = build_avro_schema_json(dataset)?;
    let avro_schema =
        AvroSchema::parse_str(&avro_schema_json).map_err(|e| format!("avro schema parse: {e}"))?;

    let mut writer = AvroWriter::builder()
        .schema(&avro_schema)
        .codec(Codec::Deflate(DeflateSettings::default()))
        .writer(Vec::new())
        .build();

    let row_count = dataset.row_count;
    for row in 0..row_count {
        let mut record =
            apache_avro::types::Record::new(&avro_schema).ok_or("avro record new failed")?;
        for spec in &dataset.schema.columns {
            let lc = &dataset.columns[&spec.name];
            let val = lc_row_to_avro_value(lc, &spec.logical_type, row);
            record.put(&spec.name, val);
        }
        writer
            .append(record)
            .map_err(|e| format!("avro append: {e}"))?;
    }
    let out = writer
        .into_inner()
        .map_err(|e| format!("avro into_inner: {e}"))?;
    if out.is_empty() {
        return Err("avro output was empty".into());
    }
    Ok(out)
}

fn build_avro_schema_json(dataset: &FlatDataset) -> Result<String, String> {
    let mut fields_json = Vec::new();
    for spec in &dataset.schema.columns {
        let avro_type = lt_to_avro_type_str(&spec.logical_type).ok_or_else(|| {
            format!(
                "no avro type for col {} lt={:?}",
                spec.name, spec.logical_type
            )
        })?;
        fields_json.push(format!(
            r#"{{"name":"{}","type":{}}}"#,
            spec.name, avro_type
        ));
    }
    Ok(format!(
        r#"{{"type":"record","name":"Row","fields":[{}]}}"#,
        fields_json.join(",")
    ))
}

fn lt_to_avro_type_str(lt: &LogicalType) -> Option<String> {
    match lt {
        LogicalType::Primitive { data_type } => match data_type {
            DataType::I8 | DataType::I16 | DataType::I32 | DataType::U8 | DataType::U16 => {
                Some("\"int\"".into())
            }
            DataType::I64 | DataType::U32 | DataType::U64 => Some("\"long\"".into()),
            DataType::F32 => Some("\"float\"".into()),
            DataType::F64 => Some("\"double\"".into()),
            DataType::Bytes => Some("\"bytes\"".into()),
        },
        LogicalType::Utf8 => Some("\"string\"".into()),
        LogicalType::Binary => Some("\"bytes\"".into()),
        LogicalType::Nullable { inner } => {
            let inner_str = lt_to_avro_type_str(inner)?;
            Some(format!(r#"["null",{}]"#, inner_str))
        }
        _ => None,
    }
}

fn lc_row_to_avro_value(
    lc: &LogicalColumn,
    lt: &LogicalType,
    row: usize,
) -> apache_avro::types::Value {
    use apache_avro::types::Value as AvroVal;
    match (lt, lc) {
        (LogicalType::Primitive { data_type }, LogicalColumn::Primitive(cd)) => {
            cd_row_to_avro(cd, *data_type, row)
        }
        (LogicalType::Utf8, LogicalColumn::Utf8(v)) => {
            AvroVal::String(v.get(row).cloned().unwrap_or_default())
        }
        (LogicalType::Binary, LogicalColumn::Binary(v)) => {
            AvroVal::Bytes(v.get(row).cloned().unwrap_or_default())
        }
        // recursive Nullable — inner lc holds only non-null rows (compact)
        (LogicalType::Nullable { inner }, LogicalColumn::Nullable { present, value }) => {
            if !present.get(row).copied().unwrap_or(false) {
                return AvroVal::Union(0, Box::new(AvroVal::Null));
            }
            // count non-null rows before `row` to get the compact index
            let idx = present[..row].iter().filter(|&&p| p).count();
            match (inner.as_ref(), value.as_ref()) {
                (LogicalType::Primitive { data_type }, LogicalColumn::Primitive(cd)) => {
                    AvroVal::Union(1, Box::new(cd_row_to_avro(cd, *data_type, idx)))
                }
                (LogicalType::Utf8, LogicalColumn::Utf8(strings)) => AvroVal::Union(
                    1,
                    Box::new(AvroVal::String(
                        strings.get(idx).cloned().unwrap_or_default(),
                    )),
                ),
                (LogicalType::Binary, LogicalColumn::Binary(blobs)) => AvroVal::Union(
                    1,
                    Box::new(AvroVal::Bytes(blobs.get(idx).cloned().unwrap_or_default())),
                ),
                _ => AvroVal::Null,
            }
        }
        _ => AvroVal::Null,
    }
}

fn cd_row_to_avro(cd: &ColumnData, dt: DataType, row: usize) -> apache_avro::types::Value {
    use apache_avro::types::Value as AvroVal;
    match cd {
        ColumnData::I8(v) => AvroVal::Int(*v.get(row).unwrap_or(&0) as i32),
        ColumnData::I16(v) => AvroVal::Int(*v.get(row).unwrap_or(&0) as i32),
        ColumnData::I32(v) => AvroVal::Int(*v.get(row).unwrap_or(&0)),
        ColumnData::I64(v) => match dt {
            DataType::I64 | DataType::U64 => AvroVal::Long(*v.get(row).unwrap_or(&0)),
            _ => AvroVal::Long(*v.get(row).unwrap_or(&0)),
        },
        ColumnData::U8(v) => AvroVal::Int(*v.get(row).unwrap_or(&0) as i32),
        ColumnData::U16(v) => AvroVal::Int(*v.get(row).unwrap_or(&0) as i32),
        ColumnData::U32(v) => AvroVal::Long(*v.get(row).unwrap_or(&0) as i64),
        ColumnData::U64(v) => AvroVal::Long(*v.get(row).unwrap_or(&0) as i64),
        ColumnData::F32(v) => AvroVal::Float(*v.get(row).unwrap_or(&0.0)),
        ColumnData::F64(v) => AvroVal::Double(*v.get(row).unwrap_or(&0.0)),
        ColumnData::Bytes(v) => AvroVal::Bytes(v.clone()),
    }
}

// ---------------------------------------------------------------------------
// Helium writers
// ---------------------------------------------------------------------------

struct HeliumMeasure {
    file_bytes: usize,
    catalog_side_bytes: Option<usize>,
}

fn build_optimized_schema(dataset: &FlatDataset) -> Schema {
    let triples: Vec<(String, LogicalType, LogicalColumn)> = dataset
        .schema
        .columns
        .iter()
        .map(|spec| {
            let lc = dataset.columns[&spec.name].clone();
            (spec.name.clone(), spec.logical_type.clone(), lc)
        })
        .collect();
    Optimizer::new().optimize(triples).expect("optimizer")
}

fn write_helium(schema: Schema, dataset: &FlatDataset, label: &str) -> HeliumMeasure {
    let registry = CoderRegistry::default();
    let tmp = tempfile::NamedTempFile::new().expect("tempfile self-contained");

    let col_names: Vec<String> = schema.columns.iter().map(|s| s.name.clone()).collect();
    let mut writer = HeliumWriter::new(
        tmp.as_file().try_clone().expect("clone file self-contained"),
        schema,
        &registry,
    )
    .expect("HeliumWriter::new");
    for name in &col_names {
        let lc = dataset.columns[name].clone();
        writer.write_column(name, lc).expect("write_column self-contained");
    }
    writer.finish().expect("finish self-contained");

    // Round-trip verification
    {
        let f = std::fs::File::open(tmp.path()).expect("open he self-contained");
        let registry2 = CoderRegistry::default();
        let mut reader = HeliumReader::new(f, &registry2).expect("HeliumReader self-contained");
        let schema_clone = reader
            .schema()
            .columns
            .iter()
            .map(|s| s.name.clone())
            .collect::<Vec<_>>();
        for name in &schema_clone {
            let decoded = reader.read_column(name).expect("read_column self-contained");
            let original = &dataset.columns[name];
            assert_eq!(
                &decoded, original,
                "round-trip mismatch for column '{name}' in {label}"
            );
        }
    }

    let file_bytes = tmp.path().metadata().expect("metadata self-contained").len() as usize;
    HeliumMeasure {
        file_bytes,
        catalog_side_bytes: None,
    }
}

/// Write a self-contained file in multiple stripes of `stripe_rows` rows each.
///
/// Uses `LogicalColumn::slice` to chop each column into chunks, calling
/// `HeliumWriter::finish_stripe` between chunks (same pattern as
/// `src/cli/convert.rs::write_multistripe`).  Round-trip is verified by
/// reading each stripe back with `read_column_at_stripe` and concatenating.
fn write_helium_multistripe(
    schema: Schema,
    dataset: &FlatDataset,
    stripe_rows: usize,
    label: &str,
) -> HeliumMeasure {
    let registry = CoderRegistry::default();
    let tmp = tempfile::NamedTempFile::new().expect("tempfile self-contained multi");

    let col_names: Vec<String> = schema.columns.iter().map(|s| s.name.clone()).collect();
    let mut writer = HeliumWriter::new(
        tmp.as_file().try_clone().expect("clone file self-contained multi"),
        schema,
        &registry,
    )
    .expect("HeliumWriter::new self-contained multi");

    let total = dataset.row_count;
    let mut offset = 0usize;
    while offset < total {
        let chunk = stripe_rows.min(total - offset);
        for name in &col_names {
            let lc = dataset.columns[name]
                .slice(offset, chunk)
                .unwrap_or_else(|e| panic!("slice column '{name}' at {offset}+{chunk}: {e}"));
            writer
                .write_column(name, lc)
                .expect("write_column self-contained multi");
        }
        offset += chunk;
        if offset < total {
            writer.finish_stripe().expect("finish_stripe self-contained multi");
        }
    }
    writer.finish().expect("finish self-contained multi");

    // Round-trip verification: read each stripe back and concatenate, then compare.
    {
        let f = std::fs::File::open(tmp.path()).expect("open he self-contained multi");
        let registry2 = CoderRegistry::default();
        let mut reader = HeliumReader::new(f, &registry2).expect("HeliumReader self-contained multi");
        let n_stripes = reader.stripe_count();
        let schema_cols = reader
            .schema()
            .columns
            .iter()
            .map(|s| s.name.clone())
            .collect::<Vec<_>>();
        for name in &schema_cols {
            // Collect per-stripe slices and verify they match the dataset slice.
            let mut global_offset = 0usize;
            for s_idx in 0..n_stripes {
                let stripe_col = reader
                    .read_column_at_stripe(name, s_idx)
                    .expect("read_column_at_stripe self-contained multi");
                let stripe_len = stripe_col.row_count();
                let expected = dataset.columns[name]
                    .slice(global_offset, stripe_len)
                    .expect("slice expected self-contained multi");
                assert_eq!(
                    stripe_col, expected,
                    "round-trip mismatch stripe {s_idx} col '{name}' in {label}"
                );
                global_offset += stripe_len;
            }
            assert_eq!(
                global_offset, total,
                "total rows mismatch for '{name}' in {label}"
            );
        }
    }

    let file_bytes = tmp.path().metadata().expect("metadata self-contained multi").len() as usize;
    HeliumMeasure {
        file_bytes,
        catalog_side_bytes: None,
    }
}

fn write_helium_catalog(schema: Schema, dataset: &FlatDataset, label: &str) -> HeliumMeasure {
    let registry = CoderRegistry::default();
    let catalog_dir = tempfile::tempdir().expect("catalog dir");
    let catalog = Catalog::open(catalog_dir.path()).expect("Catalog::open");

    let col_names: Vec<String> = schema.columns.iter().map(|s| s.name.clone()).collect();
    let hash = catalog.add_schema(&schema).expect("add_schema");
    let catalog_side_bytes = catalog
        .path_for(&hash)
        .metadata()
        .expect("catalog file metadata")
        .len() as usize;

    let tmp = tempfile::NamedTempFile::new().expect("tempfile catalog");
    let mut writer = catalog
        .open_writer(
            tmp.as_file().try_clone().expect("clone file catalog"),
            schema,
            &registry,
        )
        .expect("Catalog::open_writer");
    for name in &col_names {
        let lc = dataset.columns[name].clone();
        writer.write_column(name, lc).expect("write_column catalog");
    }
    writer.finish().expect("finish catalog");

    // Round-trip verification with resolver
    {
        let f = std::fs::File::open(tmp.path()).expect("open he catalog");
        let registry2 = CoderRegistry::default();
        let resolver = catalog.resolver();
        let mut reader =
            HeliumReader::new_with_resolver(f, &registry2, resolver).expect("HeliumReader catalog");
        let schema_clone = reader
            .schema()
            .columns
            .iter()
            .map(|s| s.name.clone())
            .collect::<Vec<_>>();
        for name in &schema_clone {
            let decoded = reader.read_column(name).expect("read_column catalog");
            let original = &dataset.columns[name];
            assert_eq!(
                &decoded, original,
                "round-trip mismatch for column '{name}' in {label}"
            );
        }
    }

    let file_bytes = tmp.path().metadata().expect("metadata catalog").len() as usize;
    HeliumMeasure {
        file_bytes,
        catalog_side_bytes: Some(catalog_side_bytes),
    }
}

/// Write a catalog-mode file in multiple stripes of `stripe_rows` rows each.
///
/// The catalog side-file is written once (schema is shared across all stripes);
/// only the `.he` file grows with each stripe's footer entries.
fn write_helium_catalog_multistripe(
    schema: Schema,
    dataset: &FlatDataset,
    stripe_rows: usize,
    label: &str,
) -> HeliumMeasure {
    let registry = CoderRegistry::default();
    let catalog_dir = tempfile::tempdir().expect("catalog dir catalog multi");
    let catalog = Catalog::open(catalog_dir.path()).expect("Catalog::open catalog multi");

    let col_names: Vec<String> = schema.columns.iter().map(|s| s.name.clone()).collect();
    let hash = catalog.add_schema(&schema).expect("add_schema catalog multi");
    let catalog_side_bytes = catalog
        .path_for(&hash)
        .metadata()
        .expect("catalog file metadata catalog multi")
        .len() as usize;

    let tmp = tempfile::NamedTempFile::new().expect("tempfile catalog multi");
    let mut writer = catalog
        .open_writer(
            tmp.as_file().try_clone().expect("clone file catalog multi"),
            schema,
            &registry,
        )
        .expect("Catalog::open_writer catalog multi");

    let total = dataset.row_count;
    let mut offset = 0usize;
    while offset < total {
        let chunk = stripe_rows.min(total - offset);
        for name in &col_names {
            let lc = dataset.columns[name]
                .slice(offset, chunk)
                .unwrap_or_else(|e| panic!("slice column '{name}' at {offset}+{chunk}: {e}"));
            writer
                .write_column(name, lc)
                .expect("write_column catalog multi");
        }
        offset += chunk;
        if offset < total {
            writer.finish_stripe().expect("finish_stripe catalog multi");
        }
    }
    writer.finish().expect("finish catalog multi");

    // Round-trip verification: read each stripe back and compare vs original slice.
    {
        let f = std::fs::File::open(tmp.path()).expect("open he catalog multi");
        let registry2 = CoderRegistry::default();
        let resolver = catalog.resolver();
        let mut reader = HeliumReader::new_with_resolver(f, &registry2, resolver)
            .expect("HeliumReader catalog multi");
        let n_stripes = reader.stripe_count();
        let schema_cols = reader
            .schema()
            .columns
            .iter()
            .map(|s| s.name.clone())
            .collect::<Vec<_>>();
        for name in &schema_cols {
            let mut global_offset = 0usize;
            for s_idx in 0..n_stripes {
                let stripe_col = reader
                    .read_column_at_stripe(name, s_idx)
                    .expect("read_column_at_stripe catalog multi");
                let stripe_len = stripe_col.row_count();
                let expected = dataset.columns[name]
                    .slice(global_offset, stripe_len)
                    .expect("slice expected catalog multi");
                assert_eq!(
                    stripe_col, expected,
                    "round-trip mismatch stripe {s_idx} col '{name}' in {label}"
                );
                global_offset += stripe_len;
            }
            assert_eq!(
                global_offset, total,
                "total rows mismatch for '{name}' in {label}"
            );
        }
    }

    let file_bytes = tmp.path().metadata().expect("metadata catalog multi").len() as usize;
    HeliumMeasure {
        file_bytes,
        catalog_side_bytes: Some(catalog_side_bytes),
    }
}

// ---------------------------------------------------------------------------
// Report row
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct ReportRow {
    label: String,
    bytes: usize,
    /// If this row represents ".he + catalog side" combined, this is the
    /// catalog-side portion (for annotation in the table).
    catalog_annotation: Option<(usize, usize)>, // (he_only, catalog_side)
}

// ---------------------------------------------------------------------------
// Metadata helpers
// ---------------------------------------------------------------------------

fn git_rev() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from(".")))
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

fn hardware_string() -> String {
    format!("{}/{}", std::env::consts::OS, std::env::consts::ARCH)
}

fn rust_version() -> String {
    std::process::Command::new("rustc")
        .args(["--version"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

// ---------------------------------------------------------------------------
// THE TEST
// ---------------------------------------------------------------------------

#[test]
fn format_comparison_report() {
    let dataset = load_dataset();
    eprintln!(
        "Dataset: {} (row_count={}, col_count={})",
        dataset.source_label, dataset.row_count, dataset.col_count
    );

    // Stripe size: at most 10 000 rows, but at least total/4 so we always get
    // >= 4 stripes even on the synthetic fallback (~10 000 rows → 2 500/stripe).
    let stripe_rows = 10_000_usize.min(dataset.row_count / 4).max(1);
    eprintln!(
        "  Multi-stripe row size: {stripe_rows} ({} stripes)",
        dataset.row_count.div_ceil(stripe_rows)
    );

    // ---- External formats ----

    eprintln!("  Writing csv...");
    let csv_bytes = write_csv_bytes(&dataset);

    eprintln!("  Writing csv.zst...");
    let csv_zst_bytes = zstd_compress(&csv_bytes);

    eprintln!("  Writing ndjson...");
    let ndjson_bytes = write_ndjson_bytes(&dataset);

    eprintln!("  Writing ndjson.zst...");
    let ndjson_zst_bytes = zstd_compress(&ndjson_bytes);

    eprintln!("  Writing parquet (snappy)...");
    let parquet_snappy_bytes = write_parquet_bytes(&dataset, PqCompression::SNAPPY);

    eprintln!("  Writing parquet (zstd)...");
    let parquet_zstd_bytes =
        write_parquet_bytes(&dataset, PqCompression::ZSTD(ZstdLevel::default()));

    // LZ4_RAW is the current Parquet standard LZ4 codec (non-deprecated).
    // LZ4 (legacy Hadoop framing) is deprecated per PARQUET-2032.
    eprintln!("  Writing parquet (lz4_raw)...");
    let parquet_lz4_bytes = write_parquet_bytes(&dataset, PqCompression::LZ4_RAW);

    // Pure raw bytes: no schema, no framing — just concatenated column bytes.
    eprintln!("  Writing raw column bytes (for pure-zstd / pure-lz4 baselines)...");
    let raw_bytes = write_raw_column_bytes(&dataset);
    let pure_zstd_bytes = zstd_compress(&raw_bytes);
    let pure_lz4_bytes = lz4_compress(&raw_bytes);

    eprintln!("  Writing avro (deflate)...");
    let avro_result = write_avro_deflate_bytes(&dataset);

    // ---- Helium variants ----

    eprintln!("  Building default schema...");
    let default_schema = dataset.schema.clone();

    eprintln!("  Building optimized schema (this may take a moment)...");
    let optimized_schema = build_optimized_schema(&dataset);

    // lz4-terminal variants: same pipelines as default/optimized but with
    // the trailing "zstd" coder replaced by "lz4" in every pipeline.
    let default_schema_lz4 = lz4_terminal_schema(&default_schema);
    let optimized_schema_lz4 = lz4_terminal_schema(&optimized_schema);

    eprintln!("  Writing helium default...");
    let he_sc_default = write_helium(default_schema.clone(), &dataset, "helium default");

    eprintln!("  Writing helium optimized...");
    let he_sc_opt = write_helium(optimized_schema.clone(), &dataset, "helium optimized");

    eprintln!("  Writing helium default (lz4 terminal)...");
    let he_sc_default_lz4 = write_helium(
        default_schema_lz4.clone(),
        &dataset,
        "helium default (lz4 terminal)",
    );

    eprintln!("  Writing helium optimized (lz4 terminal)...");
    let he_sc_opt_lz4 = write_helium(
        optimized_schema_lz4.clone(),
        &dataset,
        "helium optimized (lz4 terminal)",
    );

    eprintln!("  Writing helium-catalog default...");
    let he_cat_default = write_helium_catalog(default_schema.clone(), &dataset, "helium-catalog default");

    eprintln!("  Writing helium-catalog optimized...");
    let he_cat_opt = write_helium_catalog(optimized_schema.clone(), &dataset, "helium-catalog optimized");

    eprintln!("  Writing helium-catalog default (lz4 terminal)...");
    let he_cat_default_lz4 = write_helium_catalog(
        default_schema_lz4.clone(),
        &dataset,
        "helium-catalog default (lz4 terminal)",
    );

    eprintln!("  Writing helium-catalog optimized (lz4 terminal)...");
    let he_cat_opt_lz4 = write_helium_catalog(
        optimized_schema_lz4.clone(),
        &dataset,
        "helium-catalog optimized (lz4 terminal)",
    );

    eprintln!("  Writing helium default ({stripe_rows}-row stripes)...");
    let he_sc_default_ms = write_helium_multistripe(
        default_schema.clone(),
        &dataset,
        stripe_rows,
        "helium default (multi-stripe)",
    );

    eprintln!("  Writing helium optimized ({stripe_rows}-row stripes)...");
    let he_sc_opt_ms = write_helium_multistripe(
        optimized_schema.clone(),
        &dataset,
        stripe_rows,
        "helium optimized (multi-stripe)",
    );

    eprintln!("  Writing helium-catalog default ({stripe_rows}-row stripes)...");
    let he_cat_default_ms = write_helium_catalog_multistripe(
        default_schema,
        &dataset,
        stripe_rows,
        "helium-catalog default (multi-stripe)",
    );

    eprintln!("  Writing helium-catalog optimized ({stripe_rows}-row stripes)...");
    let he_cat_opt_ms = write_helium_catalog_multistripe(
        optimized_schema,
        &dataset,
        stripe_rows,
        "helium-catalog optimized (multi-stripe)",
    );

    // ---- Baseline sizes ----
    let csv_baseline = csv_bytes.len();
    let parquet_baseline = parquet_snappy_bytes.len();

    // ---- Assemble rows ----

    let mut external_rows: Vec<ReportRow> = vec![
        ReportRow {
            label: "csv".into(),
            bytes: csv_bytes.len(),
            catalog_annotation: None,
        },
        ReportRow {
            label: "csv.zst".into(),
            bytes: csv_zst_bytes.len(),
            catalog_annotation: None,
        },
        ReportRow {
            label: "ndjson".into(),
            bytes: ndjson_bytes.len(),
            catalog_annotation: None,
        },
        ReportRow {
            label: "ndjson.zst".into(),
            bytes: ndjson_zst_bytes.len(),
            catalog_annotation: None,
        },
        ReportRow {
            label: "parquet (snappy)".into(),
            bytes: parquet_snappy_bytes.len(),
            catalog_annotation: None,
        },
        ReportRow {
            label: "parquet (zstd)".into(),
            bytes: parquet_zstd_bytes.len(),
            catalog_annotation: None,
        },
        // LZ4_RAW is the non-deprecated Parquet LZ4 codec (PARQUET-2032).
        ReportRow {
            label: "parquet (lz4_raw)".into(),
            bytes: parquet_lz4_bytes.len(),
            catalog_annotation: None,
        },
        // Pure raw bytes: every column's native LE bytes concatenated with no
        // framing, then general-purpose compressed. These are the lower-bound
        // targets — the minimum size achievable if the compressor could see all
        // raw bytes at once with zero per-column overhead.
        ReportRow {
            label: "pure zstd (raw bytes)".into(),
            bytes: pure_zstd_bytes.len(),
            catalog_annotation: None,
        },
        ReportRow {
            label: "pure lz4 (raw bytes)".into(),
            bytes: pure_lz4_bytes.len(),
            catalog_annotation: None,
        },
    ];
    if let Ok(ref avro_b) = avro_result {
        external_rows.push(ReportRow {
            label: "avro (deflate)".into(),
            bytes: avro_b.len(),
            catalog_annotation: None,
        });
    }
    external_rows.sort_by_key(|r| r.bytes);

    let v6_def_he = he_cat_default.file_bytes;
    let v6_def_cat = he_cat_default.catalog_side_bytes.unwrap_or(0);
    let v6_opt_he = he_cat_opt.file_bytes;
    let v6_opt_cat = he_cat_opt.catalog_side_bytes.unwrap_or(0);

    let stripe_label = format!("{stripe_rows}-row stripes");

    let v6_def_lz4_he = he_cat_default_lz4.file_bytes;
    let v6_def_lz4_cat = he_cat_default_lz4.catalog_side_bytes.unwrap_or(0);
    let v6_opt_lz4_he = he_cat_opt_lz4.file_bytes;
    let v6_opt_lz4_cat = he_cat_opt_lz4.catalog_side_bytes.unwrap_or(0);

    let mut helium_rows: Vec<ReportRow> = vec![
        ReportRow {
            label: "helium default".into(),
            bytes: he_sc_default.file_bytes,
            catalog_annotation: None,
        },
        ReportRow {
            label: "helium optimized".into(),
            bytes: he_sc_opt.file_bytes,
            catalog_annotation: None,
        },
        ReportRow {
            label: "helium default (lz4 terminal)".into(),
            bytes: he_sc_default_lz4.file_bytes,
            catalog_annotation: None,
        },
        ReportRow {
            label: "helium optimized (lz4 terminal)".into(),
            bytes: he_sc_opt_lz4.file_bytes,
            catalog_annotation: None,
        },
        ReportRow {
            label: "helium-catalog default".into(),
            bytes: v6_def_he,
            catalog_annotation: None,
        },
        ReportRow {
            label: "helium-catalog default + catalog side".into(),
            bytes: v6_def_he + v6_def_cat,
            catalog_annotation: Some((v6_def_he, v6_def_cat)),
        },
        ReportRow {
            label: "helium-catalog optimized".into(),
            bytes: v6_opt_he,
            catalog_annotation: None,
        },
        ReportRow {
            label: "helium-catalog optimized + catalog side".into(),
            bytes: v6_opt_he + v6_opt_cat,
            catalog_annotation: Some((v6_opt_he, v6_opt_cat)),
        },
        ReportRow {
            label: "helium-catalog default (lz4 terminal)".into(),
            bytes: v6_def_lz4_he,
            catalog_annotation: None,
        },
        ReportRow {
            label: "helium-catalog default (lz4 terminal) + catalog side".into(),
            bytes: v6_def_lz4_he + v6_def_lz4_cat,
            catalog_annotation: Some((v6_def_lz4_he, v6_def_lz4_cat)),
        },
        ReportRow {
            label: "helium-catalog optimized (lz4 terminal)".into(),
            bytes: v6_opt_lz4_he,
            catalog_annotation: None,
        },
        ReportRow {
            label: "helium-catalog optimized (lz4 terminal) + catalog side".into(),
            bytes: v6_opt_lz4_he + v6_opt_lz4_cat,
            catalog_annotation: Some((v6_opt_lz4_he, v6_opt_lz4_cat)),
        },
        ReportRow {
            label: format!("helium default ({stripe_label})"),
            bytes: he_sc_default_ms.file_bytes,
            catalog_annotation: None,
        },
        ReportRow {
            label: format!("helium optimized ({stripe_label})"),
            bytes: he_sc_opt_ms.file_bytes,
            catalog_annotation: None,
        },
        ReportRow {
            label: format!("helium-catalog default ({stripe_label})"),
            bytes: he_cat_default_ms.file_bytes,
            catalog_annotation: None,
        },
        ReportRow {
            label: format!("helium-catalog optimized ({stripe_label})"),
            bytes: he_cat_opt_ms.file_bytes,
            catalog_annotation: None,
        },
    ];
    helium_rows.sort_by_key(|r| r.bytes);

    // ---- Build report ----

    let mut report = String::new();

    writeln!(&mut report, "# Cross-format compression comparison").unwrap();
    writeln!(&mut report).unwrap();
    writeln!(&mut report, "**Commit**: {}", git_rev()).unwrap();
    writeln!(&mut report, "**Hardware**: {}", hardware_string()).unwrap();
    writeln!(&mut report, "**Rust**: {}", rust_version()).unwrap();
    writeln!(&mut report, "**Build**: --release").unwrap();
    writeln!(&mut report).unwrap();
    writeln!(&mut report, "**Dataset**: {}", dataset.source_label).unwrap();
    writeln!(&mut report, "**Date**: 2026-05-05").unwrap();
    writeln!(
        &mut report,
        "**Multi-stripe size**: {} rows/stripe ({} stripes; computed as `min(10_000, total_rows/4)`)",
        stripe_rows,
        dataset.row_count.div_ceil(stripe_rows)
    )
    .unwrap();
    writeln!(&mut report).unwrap();

    if let Err(ref e) = avro_result {
        writeln!(
            &mut report,
            "> **Note**: Avro write failed: {}. Row is omitted from results.",
            e
        )
        .unwrap();
        writeln!(&mut report).unwrap();
    }

    // Table
    writeln!(
        &mut report,
        "| Format / Configuration | bytes | pretty | ratio vs csv | ratio vs parquet (snappy) |"
    )
    .unwrap();
    writeln!(&mut report, "|---|---:|---:|---:|---:|").unwrap();

    let print_row = |out: &mut String, row: &ReportRow| {
        let note = match row.catalog_annotation {
            Some((he, cat)) => format!(" (.he={} + catalog={})", fmt_bytes(he), fmt_bytes(cat)),
            None => String::new(),
        };
        writeln!(
            out,
            "| {}{} | {} | {} | {} | {} |",
            row.label,
            note,
            row.bytes,
            fmt_bytes(row.bytes),
            fmt_ratio(row.bytes, csv_baseline),
            fmt_ratio(row.bytes, parquet_baseline),
        )
        .unwrap();
    };

    writeln!(&mut report, "| **External formats** | | | | |").unwrap();
    for row in &external_rows {
        print_row(&mut report, row);
    }
    writeln!(&mut report).unwrap();
    writeln!(&mut report, "| **Helium configurations** | | | | |").unwrap();
    for row in &helium_rows {
        print_row(&mut report, row);
    }

    // ---- Takeaways ----

    writeln!(&mut report).unwrap();
    writeln!(&mut report, "## Takeaways").unwrap();
    writeln!(&mut report).unwrap();

    let best_he = helium_rows
        .iter()
        .filter(|r| r.catalog_annotation.is_none()) // exclude "combined" rows
        .min_by_key(|r| r.bytes)
        .expect("at least one helium row");

    let parquet_zstd_sz = parquet_zstd_bytes.len();

    let mut bullets: Vec<String> = Vec::new();

    // Best helium vs parquet-zstd
    if best_he.bytes < parquet_zstd_sz {
        let pct = 100.0 * (parquet_zstd_sz - best_he.bytes) as f64 / parquet_zstd_sz as f64;
        bullets.push(format!(
            "`{}` ({}) beats parquet/zstd ({}) by {:.1}% on this dataset.",
            best_he.label,
            fmt_bytes(best_he.bytes),
            fmt_bytes(parquet_zstd_sz),
            pct
        ));
    } else {
        let pct = 100.0 * (best_he.bytes - parquet_zstd_sz) as f64 / parquet_zstd_sz as f64;
        bullets.push(format!(
            "`{}` ({}) is {:.1}% larger than parquet/zstd ({}) on this dataset.",
            best_he.label,
            fmt_bytes(best_he.bytes),
            pct,
            fmt_bytes(parquet_zstd_sz)
        ));
    }

    // Catalog overhead per file
    let v5_def = he_sc_default.file_bytes;
    bullets.push(format!(
        "Catalog (catalog) per-file overhead vs self-contained: {} bytes (.he file: {} vs {}). \
         The catalog JSON side file is {} and is amortized across all files sharing the schema.",
        v6_def_he as i64 - v5_def as i64,
        fmt_bytes(v6_def_he),
        fmt_bytes(v5_def),
        fmt_bytes(v6_def_cat)
    ));

    // csv.zst vs parquet/snappy
    if csv_zst_bytes.len() < parquet_snappy_bytes.len() {
        let pct = 100.0 * (parquet_snappy_bytes.len() - csv_zst_bytes.len()) as f64
            / parquet_snappy_bytes.len() as f64;
        bullets.push(format!(
            "csv.zst ({}) beats parquet/snappy ({}) by {:.1}% on this flat dataset — \
             parquet per-column overhead is noticeable at small column counts.",
            fmt_bytes(csv_zst_bytes.len()),
            fmt_bytes(parquet_snappy_bytes.len()),
            pct
        ));
    } else {
        let pct = 100.0 * (csv_zst_bytes.len() - parquet_snappy_bytes.len()) as f64
            / parquet_snappy_bytes.len() as f64;
        bullets.push(format!(
            "parquet/snappy ({}) beats csv.zst ({}) by {:.1}% on this dataset.",
            fmt_bytes(parquet_snappy_bytes.len()),
            fmt_bytes(csv_zst_bytes.len()),
            pct
        ));
    }

    // csv vs ndjson raw
    if csv_bytes.len() < ndjson_bytes.len() {
        bullets.push(format!(
            "Raw csv ({}) is smaller than ndjson ({}) — NDJSON repeats field names per row, \
             adding {} bytes ({:.1}% overhead).",
            fmt_bytes(csv_bytes.len()),
            fmt_bytes(ndjson_bytes.len()),
            ndjson_bytes.len() - csv_bytes.len(),
            100.0 * (ndjson_bytes.len() - csv_bytes.len()) as f64 / csv_bytes.len() as f64
        ));
    } else {
        bullets.push(format!(
            "ndjson ({}) is smaller than csv ({}) on this dataset.",
            fmt_bytes(ndjson_bytes.len()),
            fmt_bytes(csv_bytes.len())
        ));
    }

    // avro if available
    if let Ok(ref avro_b) = avro_result {
        let avro_sz = avro_b.len();
        if avro_sz < parquet_snappy_bytes.len() {
            bullets.push(format!(
                "avro/deflate ({}) is {:.1}% smaller than parquet/snappy ({}).",
                fmt_bytes(avro_sz),
                100.0 * (parquet_snappy_bytes.len() - avro_sz) as f64
                    / parquet_snappy_bytes.len() as f64,
                fmt_bytes(parquet_snappy_bytes.len())
            ));
        } else {
            bullets.push(format!(
                "avro/deflate ({}) is {:.1}% larger than parquet/snappy ({}).",
                fmt_bytes(avro_sz),
                100.0 * (avro_sz - parquet_snappy_bytes.len()) as f64
                    / parquet_snappy_bytes.len() as f64,
                fmt_bytes(parquet_snappy_bytes.len())
            ));
        }
    }

    // Multi-stripe overhead bullet
    {
        let single = he_sc_default.file_bytes;
        let multi = he_sc_default_ms.file_bytes;
        let n_stripes = dataset.row_count.div_ceil(stripe_rows);
        let pct = 100.0 * (multi as f64 - single as f64) / single as f64;
        bullets.push(format!(
            "Multi-stripe overhead: helium default went from {} → {} ({:+.1}%) when \
             split into {stripe_rows}-row stripes ({n_stripes} stripes). The trade is \
             per-stripe footer entries × {n_stripes}, repaid by query-time stripe pruning.",
            fmt_bytes(single),
            fmt_bytes(multi),
            pct,
        ));
    }

    // ---- New isolation bullets ----

    // 1. What does helium's columnar shaping buy?
    // Compare pure-zstd (no schema) vs helium self-contained default (default pipelines + zstd terminal).
    {
        let pure = pure_zstd_bytes.len();
        let shaped = he_sc_default.file_bytes;
        if shaped < pure {
            let pct = 100.0 * (pure - shaped) as f64 / pure as f64;
            bullets.push(format!(
                "**Helium columnar shaping contribution**: helium default ({}) is {:.1}% \
                 smaller than pure-zstd on raw bytes ({}). That fraction of compression \
                 comes from Helium's pipelines (delta / gorilla / leb128 / dict), not from \
                 zstd itself.",
                fmt_bytes(shaped),
                pct,
                fmt_bytes(pure),
            ));
        } else {
            let pct = 100.0 * (shaped - pure) as f64 / pure as f64;
            bullets.push(format!(
                "**Helium columnar shaping contribution**: helium default ({}) is {:.1}% \
                 *larger* than pure-zstd on raw bytes ({}). The per-column framing overhead \
                 of Helium exceeds what the pipelines save on this dataset — zstd alone on \
                 raw bytes beats the default pipeline configuration.",
                fmt_bytes(shaped),
                pct,
                fmt_bytes(pure),
            ));
        }
    }

    // 2. What does the terminal compressor choice matter?
    // Compare helium self-contained default (zstd) vs helium self-contained default (lz4 terminal).
    {
        let zstd_sz = he_sc_default.file_bytes;
        let lz4_sz = he_sc_default_lz4.file_bytes;
        let pct = 100.0 * (lz4_sz as f64 - zstd_sz as f64) / zstd_sz as f64;
        bullets.push(format!(
            "**Terminal compressor choice**: helium default with zstd ({}) vs lz4 ({}) — \
             lz4 terminal is {:.1}% {} in file size. After delta/gorilla/leb128 shaping, \
             zstd vs lz4 difference {}.",
            fmt_bytes(zstd_sz),
            fmt_bytes(lz4_sz),
            pct.abs(),
            if lz4_sz > zstd_sz {
                "larger"
            } else {
                "smaller"
            },
            if pct.abs() < 5.0 {
                "is under 5% — shaping has already removed most of the entropy that \
                 differentiates these two compressors"
            } else {
                "is material — zstd extracts measurably more compression after shaping"
            },
        ));
    }

    // 3. Is parquet+zstd close to helium?
    // Reaffirm with new context row by row.
    {
        let pq_zstd = parquet_zstd_bytes.len();
        let he_best_no_cat = helium_rows
            .iter()
            .filter(|r| r.catalog_annotation.is_none())
            .filter(|r| !r.label.contains("stripe"))
            .min_by_key(|r| r.bytes)
            .map(|r| (r.label.as_str(), r.bytes));
        if let Some((label, he_sz)) = he_best_no_cat {
            if he_sz < pq_zstd {
                let pct = 100.0 * (pq_zstd - he_sz) as f64 / pq_zstd as f64;
                bullets.push(format!(
                    "**Helium vs parquet/zstd gap (single-stripe)**: best helium \
                     configuration `{}` ({}) beats parquet/zstd ({}) by {:.1}%.",
                    label,
                    fmt_bytes(he_sz),
                    fmt_bytes(pq_zstd),
                    pct,
                ));
            } else {
                let pct = 100.0 * (he_sz - pq_zstd) as f64 / pq_zstd as f64;
                bullets.push(format!(
                    "**Helium vs parquet/zstd gap (single-stripe)**: best helium \
                     configuration `{}` ({}) is {:.1}% larger than parquet/zstd ({}) on \
                     this dataset.",
                    label,
                    fmt_bytes(he_sz),
                    pct,
                    fmt_bytes(pq_zstd),
                ));
            }
        }
    }

    // 4. Is the file-format framing free?
    // Compare pure-zstd (no framing) vs helium self-contained default (zstd terminal; minimal shaping).
    {
        let pure = pure_zstd_bytes.len();
        let framed = he_sc_default.file_bytes;
        let diff = framed as i64 - pure as i64;
        let pct = 100.0 * diff as f64 / pure as f64;
        if diff > 0 {
            bullets.push(format!(
                "**File-format framing overhead**: pure-zstd on raw bytes ({}) vs \
                 helium default ({}) — the per-column/per-stripe framing costs {:+} bytes \
                 ({:+.1}%). This is the net difference between what zstd achieves on a flat \
                 byte stream vs what Helium's pipeline structure adds or removes.",
                fmt_bytes(pure),
                fmt_bytes(framed),
                diff,
                pct,
            ));
        } else {
            bullets.push(format!(
                "**File-format framing overhead**: pure-zstd on raw bytes ({}) vs \
                 helium default ({}) — Helium's columnar shaping more than pays for its \
                 framing: net saving {:+} bytes ({:+.1}%).",
                fmt_bytes(pure),
                fmt_bytes(framed),
                diff,
                pct,
            ));
        }
    }

    for b in &bullets {
        writeln!(&mut report, "- {b}").unwrap();
    }

    // ---- Print + write ----
    print!("{report}");

    std::fs::create_dir_all("target").ok();
    std::fs::write("target/format-comparison.md", &report).expect("write report");
    eprintln!("\nReport written to target/format-comparison.md");
}
