//! Compression ratio + speed comparison vs gzip / lz4 / zstd / pcodec.
//!
//! Runs every dataset × every compressor at 10k and 100k rows, measures
//! encoded size, encode time, and decode time, and writes a Markdown
//! report to `target/compression-report.md`. Also asserts a floor on the
//! helium ratio for datasets where we have a principled lower bound —
//! these catch regressions without depending on exact numbers.
//!
//! Run with `cargo test --test compression_report -- --nocapture` to see
//! the tables on stdout.

use std::fmt::Write as _;
use std::io::{Read, Write as _};
use std::time::Instant;

use flate2::Compression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use helium::{
    BlockCoder, CoderRegistry, CoderSpec, ColumnData, ColumnSpec, DataType, Delta, DeltaOfDelta,
    EliasFano, GorillaXor, Leb128, LogicalColumn, LogicalType, NonBlockCoder, Pcodec, Pipeline,
    Schema, StageCoder, Zstd,
};
use pco::ChunkConfig;

// ===========================================================================
// Dataset generators (deterministic)
// ===========================================================================

fn ts_uniform_i64(n: usize) -> Vec<i64> {
    (0..n).map(|i| 1_700_000_000_i64 + i as i64 * 30).collect()
}

fn ts_jittered_i64(n: usize) -> Vec<i64> {
    let mut rng = 0xCAFE_BABE_u32;
    let mut t = 1_700_000_000_000_i64;
    (0..n)
        .map(|_| {
            rng = rng.wrapping_mul(1_103_515_245).wrapping_add(12345);
            let jitter = ((rng >> 16) % 11) as i64 - 5;
            t += 30_000 + jitter;
            t
        })
        .collect()
}

fn rsrp_i32(n: usize) -> Vec<i32> {
    // Narrow-range signal measurements around -80 dBm.
    let mut rng = 0xDEAD_BEEF_u32;
    (0..n)
        .map(|_| {
            rng = rng.wrapping_mul(1_103_515_245).wrapping_add(12345);
            -120 + ((rng >> 16) % 81) as i32
        })
        .collect()
}

fn ids_sorted_u32(n: usize) -> Vec<u32> {
    let mut rng = 0x1337_4241_u32;
    let mut v: u32 = 0;
    (0..n)
        .map(|_| {
            rng = rng.wrapping_mul(1_103_515_245).wrapping_add(12345);
            v = v.saturating_add(1 + (rng >> 16) % 7);
            v
        })
        .collect()
}

fn random_u64(n: usize) -> Vec<u64> {
    let mut rng = 0xFEED_FACE_CAFE_BABE_u64;
    (0..n)
        .map(|_| {
            rng = rng
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            rng
        })
        .collect()
}

fn temp_gauge_f64(n: usize) -> Vec<f64> {
    // Slow sinusoidal drift, quantized to 0.1 — classic Gorilla sweet spot.
    (0..n)
        .map(|i| {
            let t = i as f64 * 0.01;
            ((20.0 + t.sin() * 2.0) * 10.0).round() / 10.0
        })
        .collect()
}

fn stock_prices_f64(n: usize) -> Vec<f64> {
    let mut rng = 0x01BC_DEF0_u32;
    let mut price: f64 = 100.0;
    (0..n)
        .map(|_| {
            rng = rng.wrapping_mul(1_103_515_245).wrapping_add(12345);
            let delta = ((rng >> 16) as f64 / 32768.0 - 0.5) * 0.1;
            price += delta;
            price
        })
        .collect()
}

fn random_f64(n: usize) -> Vec<f64> {
    // Floats in [0, 1000) — no structure adjacent values can exploit.
    let mut rng = 0x9E37_79B9_7F4A_7C15_u64;
    (0..n)
        .map(|_| {
            rng = rng
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (rng as f64 / u64::MAX as f64) * 1000.0
        })
        .collect()
}

fn log_levels_utf8(n: usize) -> Vec<String> {
    let levels = ["DEBUG", "INFO", "WARN", "ERROR", "FATAL"];
    let weights = [2u32, 80, 10, 7, 1];
    let total: u32 = weights.iter().sum();
    let mut rng = 0x1234_u32;
    (0..n)
        .map(|_| {
            rng = rng.wrapping_mul(1_103_515_245).wrapping_add(12345);
            let r = (rng >> 16) % total;
            let mut acc = 0u32;
            for (i, &w) in weights.iter().enumerate() {
                acc += w;
                if r < acc {
                    return levels[i].to_string();
                }
            }
            levels[1].to_string()
        })
        .collect()
}

fn user_agents_utf8(n: usize) -> Vec<String> {
    let uas: Vec<String> = (0..100)
        .map(|i| {
            format!(
                "Mozilla/5.0 (Client{i}) Gecko/20100101 Firefox/{}.0",
                100 + i
            )
        })
        .collect();
    let mut rng = 0xC0FF_EE00_u32;
    (0..n)
        .map(|_| {
            rng = rng.wrapping_mul(1_103_515_245).wrapping_add(12345);
            uas[((rng >> 16) as usize) % uas.len()].clone()
        })
        .collect()
}

fn log_messages_utf8(n: usize) -> Vec<String> {
    // Realistic log-line shape: template with variable IDs/values.
    let templates = [
        "request_id={} completed status=200 latency_ms={}",
        "user={} logged_in from_ip={} session={}",
        "query_id={} rows={} duration_ms={}",
        "cache miss key={} fallback=db duration_ms={}",
        "connection timeout peer={} elapsed_ms={}",
    ];
    let mut rng = 0xFACE_FEED_u32;
    (0..n)
        .map(|i| {
            rng = rng.wrapping_mul(1_103_515_245).wrapping_add(12345);
            let tmpl = templates[((rng >> 16) as usize) % templates.len()];
            let mut out = String::with_capacity(tmpl.len() + 32);
            let mut placeholders_seen = 0;
            for part in tmpl.split("{}") {
                out.push_str(part);
                if placeholders_seen < tmpl.matches("{}").count() {
                    rng = rng.wrapping_mul(1_103_515_245).wrapping_add(12345);
                    out.push_str(&((i as u32).wrapping_add(rng >> 20) % 100000).to_string());
                    placeholders_seen += 1;
                }
            }
            out
        })
        .collect()
}

fn uuids_utf8(n: usize) -> Vec<String> {
    let mut rng = 0x0123_4567_89AB_CDEF_u64;
    (0..n)
        .map(|_| {
            let mut lcg = || {
                rng = rng
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                rng
            };
            let a = lcg();
            let b = lcg();
            format!(
                "{:08x}-{:04x}-{:04x}-{:04x}-{:012x}",
                (a >> 32) & 0xffff_ffff,
                (a >> 16) & 0xffff,
                a & 0xffff,
                (b >> 48) & 0xffff,
                b & 0xffff_ffff_ffff
            )
        })
        .collect()
}

// ===========================================================================
// Raw-byte flattening (what the baselines compress)
// ===========================================================================

fn flatten_i32(xs: &[i32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(xs.len() * 4);
    for &x in xs {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}
fn flatten_i64(xs: &[i64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(xs.len() * 8);
    for &x in xs {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}
fn flatten_u32(xs: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(xs.len() * 4);
    for &x in xs {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}
fn flatten_u64(xs: &[u64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(xs.len() * 8);
    for &x in xs {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}
fn flatten_f64(xs: &[f64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(xs.len() * 8);
    for &x in xs {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn flatten_strings(xs: &[String]) -> Vec<u8> {
    // Newline-joined: baselines need some separator to recover boundaries.
    let mut out = Vec::new();
    for s in xs {
        out.extend_from_slice(s.as_bytes());
        out.push(b'\n');
    }
    out
}

// ===========================================================================
// Measurement helpers
// ===========================================================================

#[derive(Debug, Clone, Copy)]
struct Measure {
    encoded: usize,
    encode_ns: u128,
    decode_ns: u128,
}

fn time_median<F: FnMut() -> R, R>(iters: usize, mut f: F) -> (u128, R) {
    // Always keep the last result; measure each iteration and return the median ns.
    let mut times = Vec::with_capacity(iters);
    let mut last: Option<R> = None;
    for _ in 0..iters {
        let start = Instant::now();
        let r = f();
        times.push(start.elapsed().as_nanos());
        last = Some(r);
    }
    times.sort_unstable();
    (times[iters / 2], last.unwrap())
}

// ---- Baselines on raw bytes ----

fn gzip_encode(raw: &[u8]) -> Vec<u8> {
    let mut e = GzEncoder::new(Vec::new(), Compression::default());
    e.write_all(raw).unwrap();
    e.finish().unwrap()
}
fn gzip_decode(enc: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    GzDecoder::new(enc).read_to_end(&mut out).unwrap();
    out
}

fn bench_gzip(raw: &[u8], iters: usize) -> Measure {
    // Correctness: round-trip once and byte-compare.
    let encoded_once = gzip_encode(raw);
    let decoded_once = gzip_decode(&encoded_once);
    assert_eq!(decoded_once.as_slice(), raw, "gzip round-trip mismatch");

    let (encode_ns, encoded) = time_median(iters, || gzip_encode(raw));
    let (decode_ns, _) = time_median(iters, || gzip_decode(&encoded));
    Measure {
        encoded: encoded.len(),
        encode_ns,
        decode_ns,
    }
}

fn bench_lz4_raw(raw: &[u8], iters: usize) -> Measure {
    let encoded_once = lz4_flex::compress_prepend_size(raw);
    let decoded_once = lz4_flex::decompress_size_prepended(&encoded_once).unwrap();
    assert_eq!(decoded_once.as_slice(), raw, "lz4 round-trip mismatch");

    let (encode_ns, encoded) = time_median(iters, || lz4_flex::compress_prepend_size(raw));
    let (decode_ns, _) = time_median(iters, || {
        lz4_flex::decompress_size_prepended(&encoded).unwrap()
    });
    Measure {
        encoded: encoded.len(),
        encode_ns,
        decode_ns,
    }
}

fn bench_zstd_raw(raw: &[u8], iters: usize) -> Measure {
    let encoded_once = zstd::stream::encode_all(raw, 3).unwrap();
    let decoded_once = zstd::stream::decode_all(&encoded_once[..]).unwrap();
    assert_eq!(decoded_once.as_slice(), raw, "zstd round-trip mismatch");

    let (encode_ns, encoded) = time_median(iters, || zstd::stream::encode_all(raw, 3).unwrap());
    let (decode_ns, _) = time_median(iters, || zstd::stream::decode_all(&encoded[..]).unwrap());
    Measure {
        encoded: encoded.len(),
        encode_ns,
        decode_ns,
    }
}

// ---- pcodec baseline (typed) ----

fn bench_pcodec_typed<T>(values: &[T], iters: usize) -> Measure
where
    T: pco::data_types::Number + PartialEq + std::fmt::Debug,
{
    let cfg = ChunkConfig::default();
    let encoded_once = pco::standalone::simple_compress::<T>(values, &cfg).unwrap();
    let decoded_once = pco::standalone::simple_decompress::<T>(&encoded_once).unwrap();
    assert_eq!(
        decoded_once.as_slice(),
        values,
        "pcodec round-trip mismatch"
    );

    let (encode_ns, encoded) = time_median(iters, || {
        pco::standalone::simple_compress::<T>(values, &cfg).unwrap()
    });
    let (decode_ns, _) = time_median(iters, || {
        pco::standalone::simple_decompress::<T>(&encoded).unwrap()
    });
    Measure {
        encoded: encoded.len(),
        encode_ns,
        decode_ns,
    }
}

// ---- helium pipeline ----

fn bench_helium_pipeline(pipeline: &Pipeline, data: ColumnData, iters: usize) -> Measure {
    let encoded_once = pipeline.encode(data.clone()).unwrap();
    let decoded_once = pipeline.decode(encoded_once.clone()).unwrap();
    assert_eq!(decoded_once, data, "helium pipeline round-trip mismatch");

    let (encode_ns, encoded) = time_median(iters, || pipeline.encode(data.clone()).unwrap());
    let (decode_ns, _) = time_median(iters, || pipeline.decode(encoded.clone()).unwrap());
    Measure {
        encoded: encoded.len(),
        encode_ns,
        decode_ns,
    }
}

// ---- helium full file pipeline (for strings via LogicalColumn) ----

fn bench_helium_logical(
    schema: &Schema,
    column: LogicalColumn,
    column_name: &str,
    iters: usize,
) -> Measure {
    let registry = CoderRegistry::default();
    // Measure the column's encoded payload — not the whole file — so we
    // compare bytes-per-column with the baselines fairly.
    let pipelines = schema.resolve_all(&registry).unwrap();
    let idx = schema.column_index(column_name).unwrap();
    let col_pipelines = &pipelines[idx];
    let lt = schema.columns[idx].logical_type.clone();
    let row_count = column.row_count();

    // Correctness: decompose → encode every physical → decode every physical
    // → compose → compare to the original LogicalColumn.
    {
        let parts = column.clone().decompose(&lt).unwrap();
        let mut encoded = Vec::with_capacity(parts.len());
        for (part, pipe) in parts.into_iter().zip(col_pipelines.iter()) {
            encoded.push(pipe.encode(part).unwrap());
        }
        let mut decoded = Vec::with_capacity(encoded.len());
        for (e, pipe) in encoded.into_iter().zip(col_pipelines.iter()) {
            decoded.push(pipe.decode(e).unwrap());
        }
        let composed = LogicalColumn::compose(decoded, &lt, row_count).unwrap();
        assert_eq!(composed, column, "helium logical round-trip mismatch");
    }

    let (encode_ns, encoded_bytes) = time_median(iters, || {
        let parts = column.clone().decompose(&lt).unwrap();
        let mut total_size = 0usize;
        let mut physicals = Vec::with_capacity(col_pipelines.len());
        for (part, pipe) in parts.into_iter().zip(col_pipelines.iter()) {
            let encoded = pipe.encode(part).unwrap();
            if let ColumnData::Bytes(b) = &encoded {
                total_size += b.len();
            }
            physicals.push(encoded);
        }
        (total_size, physicals)
    });

    let encoded_size = encoded_bytes.0;
    let physicals = encoded_bytes.1;

    let (decode_ns, _) = time_median(iters, || {
        let mut decoded = Vec::with_capacity(physicals.len());
        for (p, pipe) in physicals.iter().zip(col_pipelines.iter()) {
            decoded.push(pipe.decode(p.clone()).unwrap());
        }
        let _ = LogicalColumn::compose(decoded, &lt, row_count).unwrap();
    });

    Measure {
        encoded: encoded_size,
        encode_ns,
        decode_ns,
    }
}

// ===========================================================================
// One row in the report
// ===========================================================================

#[derive(Debug)]
struct Row {
    dataset: String,
    raw_bytes: usize,
    gzip: Measure,
    lz4: Measure,
    zstd: Measure,
    pcodec: Option<Measure>,
    /// Helium pipeline ending in zstd (the "classic" tail).
    helium: Measure,
    helium_label: String,
    /// Optional alternative helium pipeline with the zstd block replaced by
    /// pcodec — measures whether helium's shaping stages give pcodec extra
    /// leverage beyond pcodec alone. `None` where the swap isn't
    /// semantically meaningful (elias_fano has no zstd; Utf8 data is bytes
    /// that pcodec can't consume; float pipelines collapse to "just pcodec"
    /// which is already its own column).
    helium_pco: Option<Measure>,
    helium_pco_label: Option<String>,
    floor_ratio: Option<f64>,
}

// ===========================================================================
// Pipeline + column helpers
// ===========================================================================

fn nb<T: 'static + NonBlockCoder>(c: T) -> StageCoder {
    StageCoder::NonBlock(Box::new(c))
}
fn blk<T: 'static + BlockCoder>(c: T) -> StageCoder {
    StageCoder::Block(Box::new(c))
}

// ===========================================================================
// Rows (one per dataset)
// ===========================================================================

fn row_ts_uniform(n: usize, iters: usize) -> Row {
    let xs = ts_uniform_i64(n);
    let raw = flatten_i64(&xs);
    let pipeline = Pipeline::new(
        DataType::I64,
        vec![
            nb(DeltaOfDelta::new(DataType::I64).unwrap()),
            nb(Leb128::new(DataType::I64).unwrap()),
            blk(Zstd::default()),
        ],
    )
    .unwrap();
    let pipeline_pco = Pipeline::new(
        DataType::I64,
        vec![
            nb(DeltaOfDelta::new(DataType::I64).unwrap()),
            blk(Pcodec::new(DataType::I64, None).unwrap()),
        ],
    )
    .unwrap();
    Row {
        dataset: format!("ts_uniform_i64 [{n}]"),
        raw_bytes: raw.len(),
        gzip: bench_gzip(&raw, iters),
        lz4: bench_lz4_raw(&raw, iters),
        zstd: bench_zstd_raw(&raw, iters),
        pcodec: Some(bench_pcodec_typed(&xs, iters)),
        helium: bench_helium_pipeline(&pipeline, ColumnData::I64(xs.clone()), iters),
        helium_label: "dod+leb128+zstd".into(),
        helium_pco: Some(bench_helium_pipeline(
            &pipeline_pco,
            ColumnData::I64(xs),
            iters,
        )),
        helium_pco_label: Some("dod+pcodec".into()),
        floor_ratio: Some(500.0),
    }
}

fn row_ts_jittered(n: usize, iters: usize) -> Row {
    let xs = ts_jittered_i64(n);
    let raw = flatten_i64(&xs);
    let pipeline = Pipeline::new(
        DataType::I64,
        vec![
            nb(Delta::new(DataType::I64).unwrap()),
            nb(Leb128::new(DataType::I64).unwrap()),
            blk(Zstd::default()),
        ],
    )
    .unwrap();
    let pipeline_pco = Pipeline::new(
        DataType::I64,
        vec![
            nb(Delta::new(DataType::I64).unwrap()),
            blk(Pcodec::new(DataType::I64, None).unwrap()),
        ],
    )
    .unwrap();
    Row {
        dataset: format!("ts_jittered_i64 [{n}]"),
        raw_bytes: raw.len(),
        gzip: bench_gzip(&raw, iters),
        lz4: bench_lz4_raw(&raw, iters),
        zstd: bench_zstd_raw(&raw, iters),
        pcodec: Some(bench_pcodec_typed(&xs, iters)),
        helium: bench_helium_pipeline(&pipeline, ColumnData::I64(xs.clone()), iters),
        helium_label: "delta+leb128+zstd".into(),
        helium_pco: Some(bench_helium_pipeline(
            &pipeline_pco,
            ColumnData::I64(xs),
            iters,
        )),
        helium_pco_label: Some("delta+pcodec".into()),
        floor_ratio: Some(8.0),
    }
}

fn row_rsrp(n: usize, iters: usize) -> Row {
    // RSRP is naturally negative (dBm), so the deltamin+bitpack combination
    // does not apply directly (deltamin prepends a negative min, which
    // bitpack rejects). delta+leb128+zstd handles narrow signed ranges
    // well via zigzag inside leb128.
    let xs = rsrp_i32(n);
    let raw = flatten_i32(&xs);
    let pipeline = Pipeline::new(
        DataType::I32,
        vec![
            nb(Delta::new(DataType::I32).unwrap()),
            nb(Leb128::new(DataType::I32).unwrap()),
            blk(Zstd::default()),
        ],
    )
    .unwrap();
    let pipeline_pco = Pipeline::new(
        DataType::I32,
        vec![
            nb(Delta::new(DataType::I32).unwrap()),
            blk(Pcodec::new(DataType::I32, None).unwrap()),
        ],
    )
    .unwrap();
    Row {
        dataset: format!("rsrp_i32 [{n}]"),
        raw_bytes: raw.len(),
        gzip: bench_gzip(&raw, iters),
        lz4: bench_lz4_raw(&raw, iters),
        zstd: bench_zstd_raw(&raw, iters),
        pcodec: Some(bench_pcodec_typed(&xs, iters)),
        helium: bench_helium_pipeline(&pipeline, ColumnData::I32(xs.clone()), iters),
        helium_label: "delta+leb128+zstd".into(),
        helium_pco: Some(bench_helium_pipeline(
            &pipeline_pco,
            ColumnData::I32(xs),
            iters,
        )),
        helium_pco_label: Some("delta+pcodec".into()),
        floor_ratio: Some(2.5),
    }
}

fn row_ids_sorted(n: usize, iters: usize) -> Row {
    let xs = ids_sorted_u32(n);
    let raw = flatten_u32(&xs);
    let pipeline = Pipeline::new(
        DataType::U32,
        vec![blk(EliasFano::new(DataType::U32).unwrap())],
    )
    .unwrap();
    Row {
        dataset: format!("ids_sorted_u32 [{n}]"),
        raw_bytes: raw.len(),
        gzip: bench_gzip(&raw, iters),
        lz4: bench_lz4_raw(&raw, iters),
        zstd: bench_zstd_raw(&raw, iters),
        pcodec: Some(bench_pcodec_typed(&xs, iters)),
        helium: bench_helium_pipeline(&pipeline, ColumnData::U32(xs), iters),
        helium_label: "elias_fano".into(),
        helium_pco: None,
        helium_pco_label: None,
        floor_ratio: Some(3.0),
    }
}

fn row_random_u64(n: usize, iters: usize) -> Row {
    let xs = random_u64(n);
    let raw = flatten_u64(&xs);
    // Random u64 is hard. Pipeline is leb128+zstd; expected to barely beat 1x.
    let pipeline = Pipeline::new(
        DataType::U64,
        vec![
            nb(Leb128::new(DataType::U64).unwrap()),
            blk(Zstd::default()),
        ],
    )
    .unwrap();
    Row {
        dataset: format!("random_u64 [{n}]"),
        raw_bytes: raw.len(),
        gzip: bench_gzip(&raw, iters),
        lz4: bench_lz4_raw(&raw, iters),
        zstd: bench_zstd_raw(&raw, iters),
        pcodec: Some(bench_pcodec_typed(&xs, iters)),
        helium: bench_helium_pipeline(&pipeline, ColumnData::U64(xs), iters),
        helium_label: "leb128+zstd".into(),
        helium_pco: None, // swap would collapse to bare pcodec (same as pcodec column)
        helium_pco_label: None,
        floor_ratio: Some(0.8),
    }
}

fn row_temp_gauge(n: usize, iters: usize) -> Row {
    let xs = temp_gauge_f64(n);
    let raw = flatten_f64(&xs);
    let pipeline = Pipeline::new(
        DataType::F64,
        vec![
            nb(GorillaXor::new(DataType::F64).unwrap()),
            blk(Zstd::default()),
        ],
    )
    .unwrap();
    Row {
        dataset: format!("temp_gauge_f64 [{n}]"),
        raw_bytes: raw.len(),
        gzip: bench_gzip(&raw, iters),
        lz4: bench_lz4_raw(&raw, iters),
        zstd: bench_zstd_raw(&raw, iters),
        pcodec: Some(bench_pcodec_typed(&xs, iters)),
        helium: bench_helium_pipeline(&pipeline, ColumnData::F64(xs), iters),
        helium_label: "gorilla+zstd".into(),
        helium_pco: None, // gorilla→pcodec isn't a type chain; bare pcodec == pcodec column
        helium_pco_label: None,
        floor_ratio: Some(10.0),
    }
}

fn row_stock_prices(n: usize, iters: usize) -> Row {
    let xs = stock_prices_f64(n);
    let raw = flatten_f64(&xs);
    let pipeline = Pipeline::new(
        DataType::F64,
        vec![
            nb(GorillaXor::new(DataType::F64).unwrap()),
            blk(Zstd::default()),
        ],
    )
    .unwrap();
    Row {
        dataset: format!("stock_prices_f64 [{n}]"),
        raw_bytes: raw.len(),
        gzip: bench_gzip(&raw, iters),
        lz4: bench_lz4_raw(&raw, iters),
        zstd: bench_zstd_raw(&raw, iters),
        pcodec: Some(bench_pcodec_typed(&xs, iters)),
        helium: bench_helium_pipeline(&pipeline, ColumnData::F64(xs), iters),
        helium_label: "gorilla+zstd".into(),
        helium_pco: None,
        helium_pco_label: None,
        floor_ratio: Some(1.2),
    }
}

fn row_random_f64(n: usize, iters: usize) -> Row {
    let xs = random_f64(n);
    let raw = flatten_f64(&xs);
    let pipeline = Pipeline::new(
        DataType::F64,
        vec![
            nb(GorillaXor::new(DataType::F64).unwrap()),
            blk(Zstd::default()),
        ],
    )
    .unwrap();
    Row {
        dataset: format!("random_f64 [{n}]"),
        raw_bytes: raw.len(),
        gzip: bench_gzip(&raw, iters),
        lz4: bench_lz4_raw(&raw, iters),
        zstd: bench_zstd_raw(&raw, iters),
        pcodec: Some(bench_pcodec_typed(&xs, iters)),
        helium: bench_helium_pipeline(&pipeline, ColumnData::F64(xs), iters),
        helium_label: "gorilla+zstd".into(),
        helium_pco: None,
        helium_pco_label: None,
        floor_ratio: Some(0.8),
    }
}

fn row_log_levels(n: usize, iters: usize) -> Row {
    let xs = log_levels_utf8(n);
    let raw = flatten_strings(&xs);
    let dict_lt = LogicalType::Dictionary {
        inner: Box::new(LogicalType::Utf8),
    };
    let schema = Schema::new(vec![ColumnSpec::new(
        "s",
        dict_lt.clone(),
        vec![
            vec![CoderSpec::new("leb128"), CoderSpec::new("zstd")],
            vec![CoderSpec::new("zstd")],
            vec![CoderSpec::new("bitpack_auto"), CoderSpec::new("zstd")],
        ],
    )]);
    let schema_pco = Schema::new(vec![ColumnSpec::new(
        "s",
        dict_lt,
        vec![
            vec![CoderSpec::new("leb128"), CoderSpec::new("zstd")],
            vec![CoderSpec::new("zstd")],
            vec![CoderSpec::new("pcodec")],
        ],
    )]);
    let col = LogicalColumn::dict_encode_utf8(xs);
    Row {
        dataset: format!("log_levels_utf8 [{n}]"),
        raw_bytes: raw.len(),
        gzip: bench_gzip(&raw, iters),
        lz4: bench_lz4_raw(&raw, iters),
        zstd: bench_zstd_raw(&raw, iters),
        pcodec: None,
        helium: bench_helium_logical(&schema, col.clone(), "s", iters),
        helium_label: "Dict<Utf8>(bitpack_auto+zstd)".into(),
        helium_pco: Some(bench_helium_logical(&schema_pco, col, "s", iters)),
        helium_pco_label: Some("Dict<Utf8>(pcodec)".into()),
        floor_ratio: Some(20.0),
    }
}

fn row_user_agents(n: usize, iters: usize) -> Row {
    let xs = user_agents_utf8(n);
    let raw = flatten_strings(&xs);
    let dict_lt = LogicalType::Dictionary {
        inner: Box::new(LogicalType::Utf8),
    };
    let schema = Schema::new(vec![ColumnSpec::new(
        "s",
        dict_lt.clone(),
        vec![
            vec![CoderSpec::new("leb128"), CoderSpec::new("zstd")],
            vec![CoderSpec::new("zstd")],
            vec![CoderSpec::new("bitpack_auto"), CoderSpec::new("zstd")],
        ],
    )]);
    let schema_pco = Schema::new(vec![ColumnSpec::new(
        "s",
        dict_lt,
        vec![
            vec![CoderSpec::new("leb128"), CoderSpec::new("zstd")],
            vec![CoderSpec::new("zstd")],
            vec![CoderSpec::new("pcodec")],
        ],
    )]);
    let col = LogicalColumn::dict_encode_utf8(xs);
    Row {
        dataset: format!("user_agents_utf8 [{n}]"),
        raw_bytes: raw.len(),
        gzip: bench_gzip(&raw, iters),
        lz4: bench_lz4_raw(&raw, iters),
        zstd: bench_zstd_raw(&raw, iters),
        pcodec: None,
        helium: bench_helium_logical(&schema, col.clone(), "s", iters),
        helium_label: "Dict<Utf8>(bitpack_auto+zstd)".into(),
        helium_pco: Some(bench_helium_logical(&schema_pco, col, "s", iters)),
        helium_pco_label: Some("Dict<Utf8>(pcodec)".into()),
        floor_ratio: Some(20.0),
    }
}

fn row_log_messages(n: usize, iters: usize) -> Row {
    let xs = log_messages_utf8(n);
    let raw = flatten_strings(&xs);
    let schema = Schema::new(vec![ColumnSpec::utf8(
        "s",
        vec![
            CoderSpec::new("delta"),
            CoderSpec::new("leb128"),
            CoderSpec::new("zstd"),
        ],
        vec![CoderSpec::new("zstd").with_param("level", 6)],
    )]);
    let col = LogicalColumn::Utf8(xs);
    Row {
        dataset: format!("log_messages_utf8 [{n}]"),
        raw_bytes: raw.len(),
        gzip: bench_gzip(&raw, iters),
        lz4: bench_lz4_raw(&raw, iters),
        zstd: bench_zstd_raw(&raw, iters),
        pcodec: None,
        helium: bench_helium_logical(&schema, col, "s", iters),
        helium_label: "Utf8(zstd L6 on data)".into(),
        helium_pco: None, // Utf8 data is Bytes; pcodec only consumes typed numerics
        helium_pco_label: None,
        floor_ratio: Some(1.5),
    }
}

fn row_uuids(n: usize, iters: usize) -> Row {
    let xs = uuids_utf8(n);
    let raw = flatten_strings(&xs);
    let schema = Schema::new(vec![ColumnSpec::utf8(
        "s",
        vec![
            CoderSpec::new("delta"),
            CoderSpec::new("leb128"),
            CoderSpec::new("zstd"),
        ],
        vec![CoderSpec::new("zstd")],
    )]);
    let col = LogicalColumn::Utf8(xs);
    Row {
        dataset: format!("uuids_utf8 [{n}]"),
        raw_bytes: raw.len(),
        gzip: bench_gzip(&raw, iters),
        lz4: bench_lz4_raw(&raw, iters),
        zstd: bench_zstd_raw(&raw, iters),
        pcodec: None,
        helium: bench_helium_logical(&schema, col, "s", iters),
        helium_label: "Utf8(zstd)".into(),
        helium_pco: None,
        helium_pco_label: None,
        floor_ratio: Some(0.9),
    }
}

fn collect_rows(n: usize, iters: usize) -> Vec<Row> {
    vec![
        row_ts_uniform(n, iters),
        row_ts_jittered(n, iters),
        row_rsrp(n, iters),
        row_ids_sorted(n, iters),
        row_random_u64(n, iters),
        row_temp_gauge(n, iters),
        row_stock_prices(n, iters),
        row_random_f64(n, iters),
        row_log_levels(n, iters),
        row_user_agents(n, iters),
        row_log_messages(n, iters),
        row_uuids(n, iters),
    ]
}

// ===========================================================================
// Table formatting
// ===========================================================================

// Kept compact — formats ratios as "Nx" and throughput as MB/s.
fn fmt_bytes(b: usize) -> String {
    if b >= 1024 * 1024 {
        format!("{:.1} MB", b as f64 / (1024.0 * 1024.0))
    } else if b >= 1024 {
        format!("{:.1} KB", b as f64 / 1024.0)
    } else {
        format!("{b} B")
    }
}

fn ratio(raw: usize, encoded: usize) -> f64 {
    raw as f64 / encoded.max(1) as f64
}

fn mbps(bytes: usize, ns: u128) -> f64 {
    if ns == 0 {
        return f64::INFINITY;
    }
    let sec = ns as f64 / 1e9;
    (bytes as f64 / 1_048_576.0) / sec
}

/// Find the index of the maximum among `values`, skipping `None`s. If all
/// are `None`, returns `None` (no winner). Ties go to the first entry.
fn best_index(values: &[Option<f64>]) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for (i, v) in values.iter().enumerate() {
        if let Some(x) = v {
            match best {
                None => best = Some((i, *x)),
                Some((_, b)) if *x > b => best = Some((i, *x)),
                _ => {}
            }
        }
    }
    best.map(|(i, _)| i)
}

fn fmt_cell_ratio(val: Option<f64>, is_best: bool) -> String {
    match val {
        None => "—".into(),
        Some(v) => {
            let s = format!("{v:.1}x");
            if is_best { format!("**{s}**") } else { s }
        }
    }
}

fn fmt_cell_mbps(val: Option<f64>, is_best: bool) -> String {
    match val {
        None => "—".into(),
        Some(v) => {
            let s = format!("{v:.0}");
            if is_best { format!("**{s}**") } else { s }
        }
    }
}

fn write_ratio_table(out: &mut String, rows: &[Row]) {
    writeln!(
        out,
        "\n### Compression ratio (raw / encoded, higher = better; **bold** = best in row)\n"
    )
    .unwrap();
    writeln!(
        out,
        "| Dataset | Raw | gzip | lz4 | zstd | pcodec | helium(zstd) | helium(pco) | helium pipeline | +pco pipeline |"
    )
    .unwrap();
    writeln!(out, "|---|---:|---:|---:|---:|---:|---:|---:|---|---|").unwrap();
    for r in rows {
        let vals = [
            Some(ratio(r.raw_bytes, r.gzip.encoded)),
            Some(ratio(r.raw_bytes, r.lz4.encoded)),
            Some(ratio(r.raw_bytes, r.zstd.encoded)),
            r.pcodec.map(|m| ratio(r.raw_bytes, m.encoded)),
            Some(ratio(r.raw_bytes, r.helium.encoded)),
            r.helium_pco.map(|m| ratio(r.raw_bytes, m.encoded)),
        ];
        let best = best_index(&vals);
        writeln!(
            out,
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            r.dataset,
            fmt_bytes(r.raw_bytes),
            fmt_cell_ratio(vals[0], best == Some(0)),
            fmt_cell_ratio(vals[1], best == Some(1)),
            fmt_cell_ratio(vals[2], best == Some(2)),
            fmt_cell_ratio(vals[3], best == Some(3)),
            fmt_cell_ratio(vals[4], best == Some(4)),
            fmt_cell_ratio(vals[5], best == Some(5)),
            r.helium_label,
            r.helium_pco_label.as_deref().unwrap_or("—"),
        )
        .unwrap();
    }
}

fn write_encode_throughput_table(out: &mut String, rows: &[Row]) {
    writeln!(
        out,
        "\n### Encode throughput (MB/s of raw input, higher = better; **bold** = best in row)\n"
    )
    .unwrap();
    writeln!(
        out,
        "| Dataset | gzip | lz4 | zstd | pcodec | helium(zstd) | helium(pco) |"
    )
    .unwrap();
    writeln!(out, "|---|---:|---:|---:|---:|---:|---:|").unwrap();
    for r in rows {
        let vals = [
            Some(mbps(r.raw_bytes, r.gzip.encode_ns)),
            Some(mbps(r.raw_bytes, r.lz4.encode_ns)),
            Some(mbps(r.raw_bytes, r.zstd.encode_ns)),
            r.pcodec.map(|m| mbps(r.raw_bytes, m.encode_ns)),
            Some(mbps(r.raw_bytes, r.helium.encode_ns)),
            r.helium_pco.map(|m| mbps(r.raw_bytes, m.encode_ns)),
        ];
        let best = best_index(&vals);
        writeln!(
            out,
            "| {} | {} | {} | {} | {} | {} | {} |",
            r.dataset,
            fmt_cell_mbps(vals[0], best == Some(0)),
            fmt_cell_mbps(vals[1], best == Some(1)),
            fmt_cell_mbps(vals[2], best == Some(2)),
            fmt_cell_mbps(vals[3], best == Some(3)),
            fmt_cell_mbps(vals[4], best == Some(4)),
            fmt_cell_mbps(vals[5], best == Some(5)),
        )
        .unwrap();
    }
}

fn write_decode_throughput_table(out: &mut String, rows: &[Row]) {
    writeln!(
        out,
        "\n### Decode throughput (MB/s of raw output, higher = better; **bold** = best in row)\n"
    )
    .unwrap();
    writeln!(
        out,
        "| Dataset | gzip | lz4 | zstd | pcodec | helium(zstd) | helium(pco) |"
    )
    .unwrap();
    writeln!(out, "|---|---:|---:|---:|---:|---:|---:|").unwrap();
    for r in rows {
        let vals = [
            Some(mbps(r.raw_bytes, r.gzip.decode_ns)),
            Some(mbps(r.raw_bytes, r.lz4.decode_ns)),
            Some(mbps(r.raw_bytes, r.zstd.decode_ns)),
            r.pcodec.map(|m| mbps(r.raw_bytes, m.decode_ns)),
            Some(mbps(r.raw_bytes, r.helium.decode_ns)),
            r.helium_pco.map(|m| mbps(r.raw_bytes, m.decode_ns)),
        ];
        let best = best_index(&vals);
        writeln!(
            out,
            "| {} | {} | {} | {} | {} | {} | {} |",
            r.dataset,
            fmt_cell_mbps(vals[0], best == Some(0)),
            fmt_cell_mbps(vals[1], best == Some(1)),
            fmt_cell_mbps(vals[2], best == Some(2)),
            fmt_cell_mbps(vals[3], best == Some(3)),
            fmt_cell_mbps(vals[4], best == Some(4)),
            fmt_cell_mbps(vals[5], best == Some(5)),
        )
        .unwrap();
    }
}

// ===========================================================================
// The test
// ===========================================================================

#[test]
fn compression_report_10k_and_100k() {
    let mut report = String::new();
    writeln!(&mut report, "# helium-core compression report").unwrap();
    writeln!(
        &mut report,
        "\nEvery dataset is compared against gzip / lz4 / zstd on raw bytes,\
        \npcodec on typed numerics, and two helium pipelines chosen for the\
        \ncolumn shape:\
        \n\n- **helium(zstd)** — helium shaping stages ending in zstd.\
        \n- **helium(pco)** — same shaping stages, but the final zstd block is\
        \n  swapped for pcodec. Measures whether helium's shaping buys pcodec\
        \n  any extra leverage versus pcodec alone. `—` where the swap is not\
        \n  semantically defined (elias_fano is a whole-stack coder; Utf8 data\
        \n  is bytes that pcodec can't consume; float pipelines collapse to\
        \n  bare pcodec, already in its own column).\
        \n\nHigher compression ratio is better; higher throughput is better.\
        \nNumbers are medians of multiple iterations.\
        \n\n**Every entry is round-trip verified** — encode then decode is\
        \nasserted byte- (or value-) identical to the input before the timing\
        \nloop runs. A broken coder would fail the test rather than silently\
        \nproduce pretty numbers.\n"
    )
    .unwrap();

    let mut all_regressions: Vec<String> = Vec::new();
    for (n, iters) in [(10_000usize, 11usize), (100_000usize, 5usize)] {
        writeln!(&mut report, "\n## {n} rows\n").unwrap();
        let rows = collect_rows(n, iters);
        write_ratio_table(&mut report, &rows);
        write_encode_throughput_table(&mut report, &rows);
        write_decode_throughput_table(&mut report, &rows);

        // Guardrail checks.
        for r in &rows {
            if let Some(floor) = r.floor_ratio {
                let actual = ratio(r.raw_bytes, r.helium.encoded);
                if actual < floor {
                    all_regressions.push(format!(
                        "{}: helium ratio {:.2}x fell below floor {:.2}x (pipeline: {})",
                        r.dataset, actual, floor, r.helium_label
                    ));
                }
            }
        }
    }

    print!("{report}");

    std::fs::create_dir_all("target").ok();
    std::fs::write("target/compression-report.md", &report).expect("write report");

    if !all_regressions.is_empty() {
        panic!(
            "compression regressions detected:\n  - {}",
            all_regressions.join("\n  - ")
        );
    }
}
