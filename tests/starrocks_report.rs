//! Offline compression comparison: Helium pipelines vs StarRocks' best
//! per-column config — the decision input for whether a Helium coder is worth
//! prototyping inside StarRocks' BE.
//!
//! - For each column shape, runs the StarRocks best-achievable baseline
//!   (BIT_SHUFFLE + ZSTD for numerics, DICT + ZSTD for strings, RLE + ZSTD for
//!   bool), the four generic baselines (gzip / lz4 / zstd / pcodec), and one or
//!   two Helium pipelines chosen per column.
//! - A decision summary scores Helium's best pipeline against the StarRocks
//!   baseline on two thresholds — compression-ratio gain ≥ 15% AND decode
//!   throughput ≥ 70% of baseline — to decide which columns justify a C++
//!   prototype. Float/timestamp columns are the strongest case (StarRocks'
//!   BIT_SHUFFLE is weakest there; pcodec/gorilla win).
//! - Every entry is round-trip verified before the timing loop.
//! - Emits a Markdown report to `target/starrocks-report.md`.
//!
//! Data sources:
//! - If `HELIUM_PARQUET_PATH` is set, the ClickBench-shape columns are loaded
//!   from that Parquet file. (ClickBench has no float columns, so the
//!   time-series float/timestamp columns are always synthetic.)
//! - Otherwise (the default, what CI runs), a deterministic synthetic dataset
//!   is used — realistic per-column distributions but ~100× smaller so the
//!   test stays fast.
//!
//! Run: `cargo test --test starrocks_report --release -- --nocapture`.

use std::fmt::Write as _;
use std::io::{Read, Write as _};
use std::time::Instant;

use flate2::Compression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use helium::{
    BitpackAuto, BlockCoder, CoderRegistry, CoderSpec, ColumnData, DataType, Delta, DeltaMin,
    DeltaOfDelta, GorillaXor, Leb128, LogicalColumn, NonBlockCoder, Pcodec, Pipeline, Rle, Schema,
    StageCoder, Zstd,
};
use pco::ChunkConfig;

// ============================================================================
// Synthetic ClickBench-shape dataset.
//
// Columns below model real ClickBench distributions based on the public
// schema of `hits.parquet` (see https://github.com/ClickHouse/ClickBench).
// Each generator is deterministic so reports are comparable run-to-run.
// ============================================================================

const N_ROWS: usize = 50_000;

// LCG state — used throughout for deterministic pseudo-random values.
#[derive(Copy, Clone)]
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }
    fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }
}

fn gen_watch_id(n: usize) -> Vec<i64> {
    // Monotone i64 with occasional small out-of-order jumps.
    let mut rng = Rng::new(0xc0ffee1);
    let mut t = 7_000_000_000_000_000_000i64;
    (0..n)
        .map(|_| {
            t = t.wrapping_add(1 + (rng.next_u32() % 100) as i64);
            t
        })
        .collect()
}

fn gen_user_id(n: usize) -> Vec<i64> {
    // ~50k distinct over ~n rows (many repeats).
    let mut rng = Rng::new(0xbadf00d);
    let pool: Vec<i64> = (0..(n / 10).max(500))
        .map(|i| 1_000_000_000 + i as i64 * 37)
        .collect();
    (0..n)
        .map(|_| pool[(rng.next_u32() as usize) % pool.len()])
        .collect()
}

fn gen_client_ip(n: usize) -> Vec<u32> {
    // High-cardinality uint32 — basically random.
    let mut rng = Rng::new(0xfacefeed);
    (0..n).map(|_| rng.next_u32()).collect()
}

fn gen_event_time(n: usize) -> Vec<i32> {
    // i32 unix seconds, monotone with small jitter (server logs).
    let mut rng = Rng::new(0x1234abcd);
    let mut t = 1_500_000_000i32;
    (0..n)
        .map(|_| {
            t = t.wrapping_add(((rng.next_u32() % 100) as i32).saturating_sub(5));
            t
        })
        .collect()
}

fn gen_event_date(n: usize) -> Vec<i16> {
    // Days since epoch — very low cardinality, monotone-ish across rows.
    (0..n).map(|i| 17_000i16 + (i / 5000) as i16).collect()
}

fn gen_os_code(n: usize) -> Vec<u8> {
    // ~12 distinct OS codes, very skewed distribution.
    let weights: [u8; 12] = [45, 25, 15, 5, 3, 2, 1, 1, 1, 1, 1, 1];
    let total: u16 = weights.iter().map(|&w| w as u16).sum();
    let mut rng = Rng::new(0xdeadbeef);
    (0..n)
        .map(|_| {
            let r = (rng.next_u32() as u16) % total;
            let mut acc = 0u16;
            for (i, &w) in weights.iter().enumerate() {
                acc += w as u16;
                if r < acc {
                    return i as u8;
                }
            }
            0
        })
        .collect()
}

fn gen_user_agent(n: usize) -> Vec<u8> {
    gen_os_code(n)
}

fn gen_country_id(n: usize) -> Vec<i16> {
    // ~200 countries, power-law skew.
    let mut rng = Rng::new(0xabadcafe);
    (0..n)
        .map(|_| {
            let r = rng.next_u32() as f64 / u32::MAX as f64;
            // Bias toward low indices (a few countries dominate).
            let id = (r.powf(3.0) * 200.0) as i16;
            id.clamp(0, 199)
        })
        .collect()
}

fn gen_is_refresh(n: usize) -> Vec<i8> {
    // ~5% refresh.
    let mut rng = Rng::new(0xfeedc0de);
    (0..n)
        .map(|_| {
            if rng.next_u32().is_multiple_of(20) {
                1i8
            } else {
                0
            }
        })
        .collect()
}

fn gen_title(n: usize) -> Vec<String> {
    // Medium cardinality, ~500 distinct.
    let pool: Vec<String> = (0..500)
        .map(|i| format!("Page Title #{i:04} — StarRocks / ClickHouse bench"))
        .collect();
    let mut rng = Rng::new(0x1b1b1b1b);
    (0..n)
        .map(|_| pool[(rng.next_u32() as usize) % pool.len()].clone())
        .collect()
}

fn gen_url(n: usize) -> Vec<String> {
    // Higher cardinality, some domain structure.
    let domains = [
        "example.com",
        "news.example.org",
        "shop.example.net",
        "cdn.static.example.com",
        "analytics.example.com",
    ];
    let mut rng = Rng::new(0xc000ffee);
    (0..n)
        .map(|_| {
            let d = domains[(rng.next_u32() as usize) % domains.len()];
            format!(
                "https://{d}/page/{}?ref={}",
                rng.next_u32() % 100_000,
                rng.next_u32() % 1000
            )
        })
        .collect()
}

fn gen_search_phrase(n: usize) -> Vec<String> {
    // Low cardinality (~50 distinct phrases, many empty).
    let phrases = [
        "",
        "",
        "",
        "",
        "",
        "", // lots of empty searches
        "free download",
        "weather today",
        "news latest",
        "starrocks vs clickhouse",
        "rust compression",
        "parquet vs orc",
        "how to install",
        "buy online discount",
    ];
    let mut rng = Rng::new(0xfeed1234);
    (0..n)
        .map(|_| phrases[(rng.next_u32() as usize) % phrases.len()].to_string())
        .collect()
}

// ----------------------------------------------------------------------------
// Time-series / sensor float + timestamp columns.
//
// ClickBench has essentially no float columns, but float/double and
// near-regular timestamps are exactly where pcodec and gorilla beat
// StarRocks's BIT_SHUFFLE — and where the StarRocks-integration case is
// strongest. These model DoNext-style sensor/IoT data so the report covers
// pco's main use case without pulling in an external CSV loader.
// ----------------------------------------------------------------------------

/// Slowly-varying f64 sensor reading (e.g. temperature): small random walk
/// around a drifting baseline. Gorilla XOR / pcodec exploit the tiny
/// value-to-value deltas; BIT_SHUFFLE cannot.
fn gen_sensor_f64(n: usize) -> Vec<f64> {
    let mut rng = Rng::new(0x5e_0501);
    let mut v = 21.5_f64;
    (0..n)
        .map(|_| {
            // step in [-0.05, 0.05), occasionally a small jump.
            let step = (rng.next_u32() as f64 / u32::MAX as f64 - 0.5) * 0.1;
            v += step;
            // round to 0.01 — real sensors report fixed precision, which makes
            // the XOR/quantized residuals especially compressible.
            (v * 100.0).round() / 100.0
        })
        .collect()
}

/// CPU-utilisation-style f32 in [0, 100], also slowly varying.
fn gen_cpu_f32(n: usize) -> Vec<f32> {
    let mut rng = Rng::new(0xc9_0001);
    let mut v = 30.0_f32;
    (0..n)
        .map(|_| {
            let step = (rng.next_u32() as f32 / u32::MAX as f32 - 0.5) * 4.0;
            v = (v + step).clamp(0.0, 100.0);
            (v * 10.0).round() / 10.0
        })
        .collect()
}

/// Microsecond event timestamps sampled at a near-constant interval (~1 ms)
/// with small jitter. delta-of-delta collapses to near-zero; this is the
/// canonical timestamp case.
fn gen_event_ts_micros(n: usize) -> Vec<i64> {
    let mut rng = Rng::new(0x715_77a3);
    let mut t = 1_700_000_000_000_000_i64; // ~2023 in micros
    (0..n)
        .map(|_| {
            let jitter = (rng.next_u32() % 200) as i64 - 100; // ±100 µs
            t += 1_000 + jitter; // ~1 ms cadence
            t
        })
        .collect()
}

// ============================================================================
// Parquet loader (only used when HELIUM_PARQUET_PATH is set)
// ============================================================================

mod parquet_loader {
    use parquet::file::reader::{FileReader, SerializedFileReader};
    use parquet::record::Field;
    use std::fs::File;
    use std::path::Path;

    /// Cached row-iterator so we only pay the Parquet parse once even when
    /// multiple columns are pulled.
    pub struct ParquetBatch {
        pub rows: Vec<parquet::record::Row>,
    }

    pub fn load_batch(path: &Path, max_rows: usize) -> Option<ParquetBatch> {
        let file = File::open(path).ok()?;
        let reader = SerializedFileReader::new(file).ok()?;
        let mut rows = Vec::with_capacity(max_rows);
        for row in reader.get_row_iter(None).ok()? {
            let row = row.ok()?;
            rows.push(row);
            if rows.len() >= max_rows {
                break;
            }
        }
        Some(ParquetBatch { rows })
    }

    pub fn strings(batch: &ParquetBatch, column: &str) -> Option<Vec<String>> {
        let mut out = Vec::with_capacity(batch.rows.len());
        for row in &batch.rows {
            let v = row
                .get_column_iter()
                .find(|(name, _)| *name == column)
                .map(|(_, field)| match field {
                    Field::Str(s) => s.clone(),
                    Field::Null => String::new(),
                    other => format!("{other}"),
                })
                .unwrap_or_default();
            out.push(v);
        }
        Some(out)
    }

    pub fn i64s(batch: &ParquetBatch, column: &str) -> Option<Vec<i64>> {
        let mut out = Vec::with_capacity(batch.rows.len());
        for row in &batch.rows {
            let v = row
                .get_column_iter()
                .find(|(name, _)| *name == column)
                .and_then(|(_, field)| match field {
                    Field::Long(x) => Some(*x),
                    Field::ULong(x) => Some(*x as i64),
                    Field::Int(x) => Some(*x as i64),
                    Field::UInt(x) => Some(*x as i64),
                    Field::TimestampMillis(x) => Some(*x),
                    _ => None,
                })
                .unwrap_or(0);
            out.push(v);
        }
        Some(out)
    }

    pub fn i32s(batch: &ParquetBatch, column: &str) -> Option<Vec<i32>> {
        let mut out = Vec::with_capacity(batch.rows.len());
        for row in &batch.rows {
            let v = row
                .get_column_iter()
                .find(|(name, _)| *name == column)
                .and_then(|(_, field)| match field {
                    Field::Int(x) => Some(*x),
                    Field::UInt(x) => Some(*x as i32),
                    Field::Long(x) => i32::try_from(*x).ok(),
                    _ => None,
                })
                .unwrap_or(0);
            out.push(v);
        }
        Some(out)
    }

    pub fn i16s(batch: &ParquetBatch, column: &str) -> Option<Vec<i16>> {
        Some(i32s(batch, column)?.into_iter().map(|x| x as i16).collect())
    }
    pub fn i8s(batch: &ParquetBatch, column: &str) -> Option<Vec<i8>> {
        Some(i32s(batch, column)?.into_iter().map(|x| x as i8).collect())
    }
    pub fn u8s(batch: &ParquetBatch, column: &str) -> Option<Vec<u8>> {
        Some(i32s(batch, column)?.into_iter().map(|x| x as u8).collect())
    }
    pub fn u32s(batch: &ParquetBatch, column: &str) -> Option<Vec<u32>> {
        Some(i32s(batch, column)?.into_iter().map(|x| x as u32).collect())
    }
}

// ============================================================================
// StarRocks default encoding: BIT_SHUFFLE + LZ4 for numerics,
// DICT + LZ4 for strings, RLE for bool. Byte-level shuffle approximates
// upstream bitshuffle at ~10% worse compression but no extra dep.
// ============================================================================

fn byte_shuffle<const W: usize>(values: &[[u8; W]]) -> Vec<u8> {
    let n = values.len();
    let mut out = vec![0u8; n * W];
    for (i, v) in values.iter().enumerate() {
        for b in 0..W {
            out[b * n + i] = v[b];
        }
    }
    out
}

fn byte_unshuffle<const W: usize>(bytes: &[u8], n: usize) -> Vec<[u8; W]> {
    let mut out = vec![[0u8; W]; n];
    for i in 0..n {
        for b in 0..W {
            out[i][b] = bytes[b * n + i];
        }
    }
    out
}

fn sr_shuffle_i64(v: &[i64]) -> Vec<u8> {
    let cells: Vec<[u8; 8]> = v.iter().map(|x| x.to_le_bytes()).collect();
    byte_shuffle::<8>(&cells)
}
fn sr_unshuffle_i64(bytes: &[u8], n: usize) -> Vec<i64> {
    byte_unshuffle::<8>(bytes, n)
        .into_iter()
        .map(i64::from_le_bytes)
        .collect()
}
fn sr_shuffle_i32(v: &[i32]) -> Vec<u8> {
    let cells: Vec<[u8; 4]> = v.iter().map(|x| x.to_le_bytes()).collect();
    byte_shuffle::<4>(&cells)
}
fn sr_unshuffle_i32(bytes: &[u8], n: usize) -> Vec<i32> {
    byte_unshuffle::<4>(bytes, n)
        .into_iter()
        .map(i32::from_le_bytes)
        .collect()
}
fn sr_shuffle_i16(v: &[i16]) -> Vec<u8> {
    let cells: Vec<[u8; 2]> = v.iter().map(|x| x.to_le_bytes()).collect();
    byte_shuffle::<2>(&cells)
}
fn sr_unshuffle_i16(bytes: &[u8], n: usize) -> Vec<i16> {
    byte_unshuffle::<2>(bytes, n)
        .into_iter()
        .map(i16::from_le_bytes)
        .collect()
}
fn sr_shuffle_u32(v: &[u32]) -> Vec<u8> {
    let cells: Vec<[u8; 4]> = v.iter().map(|x| x.to_le_bytes()).collect();
    byte_shuffle::<4>(&cells)
}
fn sr_unshuffle_u32(bytes: &[u8], n: usize) -> Vec<u32> {
    byte_unshuffle::<4>(bytes, n)
        .into_iter()
        .map(u32::from_le_bytes)
        .collect()
}
fn sr_shuffle_f64(v: &[f64]) -> Vec<u8> {
    let cells: Vec<[u8; 8]> = v.iter().map(|x| x.to_le_bytes()).collect();
    byte_shuffle::<8>(&cells)
}
fn sr_unshuffle_f64(bytes: &[u8], n: usize) -> Vec<f64> {
    byte_unshuffle::<8>(bytes, n)
        .into_iter()
        .map(f64::from_le_bytes)
        .collect()
}
fn sr_shuffle_f32(v: &[f32]) -> Vec<u8> {
    let cells: Vec<[u8; 4]> = v.iter().map(|x| x.to_le_bytes()).collect();
    byte_shuffle::<4>(&cells)
}
fn sr_unshuffle_f32(bytes: &[u8], n: usize) -> Vec<f32> {
    byte_unshuffle::<4>(bytes, n)
        .into_iter()
        .map(f32::from_le_bytes)
        .collect()
}

// ============================================================================
// Measurement + baseline runners (round-trip verified before timing)
// ============================================================================

#[derive(Debug, Clone, Copy)]
struct Measure {
    encoded: usize,
    encode_ns: u128,
    decode_ns: u128,
}

fn time_median<F: FnMut() -> R, R>(iters: usize, mut f: F) -> (u128, R) {
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

fn flatten_i64(v: &[i64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn flatten_i32(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn flatten_i16(v: &[i16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn flatten_u32(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn flatten_i8(v: &[i8]) -> Vec<u8> {
    v.iter().map(|&x| x as u8).collect()
}
fn flatten_u8(v: &[u8]) -> Vec<u8> {
    v.to_vec()
}
fn flatten_f64(v: &[f64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn flatten_f32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn flatten_strings(v: &[String]) -> Vec<u8> {
    let mut out = Vec::new();
    for s in v {
        out.extend_from_slice(s.as_bytes());
        out.push(b'\n');
    }
    out
}

fn bench_gzip(raw: &[u8], iters: usize) -> Measure {
    let enc_once = {
        let mut e = GzEncoder::new(Vec::new(), Compression::default());
        e.write_all(raw).unwrap();
        e.finish().unwrap()
    };
    let mut dec = Vec::new();
    GzDecoder::new(&enc_once[..]).read_to_end(&mut dec).unwrap();
    assert_eq!(dec.as_slice(), raw, "gzip round-trip mismatch");

    let (encode_ns, encoded) = time_median(iters, || {
        let mut e = GzEncoder::new(Vec::new(), Compression::default());
        e.write_all(raw).unwrap();
        e.finish().unwrap()
    });
    let (decode_ns, _) = time_median(iters, || {
        let mut out = Vec::new();
        GzDecoder::new(&encoded[..]).read_to_end(&mut out).unwrap();
        out
    });
    Measure {
        encoded: encoded.len(),
        encode_ns,
        decode_ns,
    }
}

fn bench_lz4(raw: &[u8], iters: usize) -> Measure {
    let enc_once = lz4_flex::compress_prepend_size(raw);
    let dec_once = lz4_flex::decompress_size_prepended(&enc_once).unwrap();
    assert_eq!(dec_once.as_slice(), raw, "lz4 round-trip mismatch");

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

fn bench_zstd(raw: &[u8], iters: usize) -> Measure {
    let enc_once = zstd::stream::encode_all(raw, 3).unwrap();
    let dec_once = zstd::stream::decode_all(&enc_once[..]).unwrap();
    assert_eq!(dec_once.as_slice(), raw, "zstd round-trip mismatch");

    let (encode_ns, encoded) = time_median(iters, || zstd::stream::encode_all(raw, 3).unwrap());
    let (decode_ns, _) = time_median(iters, || zstd::stream::decode_all(&encoded[..]).unwrap());
    Measure {
        encoded: encoded.len(),
        encode_ns,
        decode_ns,
    }
}

fn bench_pcodec_typed<T>(values: &[T], iters: usize) -> Measure
where
    T: pco::data_types::Number + PartialEq + std::fmt::Debug,
{
    let cfg = ChunkConfig::default();
    let enc_once = pco::standalone::simple_compress::<T>(values, &cfg).unwrap();
    let dec_once = pco::standalone::simple_decompress::<T>(&enc_once).unwrap();
    assert_eq!(dec_once.as_slice(), values, "pcodec round-trip mismatch");

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

// ---------------------------------------------------------------------------
// StarRocks BEST-ACHIEVABLE baseline: BIT_SHUFFLE (byte-wise) + ZSTD for
// numerics, RLE + ZSTD for bool, DICT + ZSTD for strings.
//
// StarRocks ships LZ4 as the default page compression, but supports ZSTD and
// it is the stronger config. We baseline Helium against ZSTD on purpose: a
// win here is a win against StarRocks' *best*, not its weakest default — so
// the numbers are conservative rather than flattering. (The standalone
// `lz4` / `zstd` columns still show the raw-compression floor for reference.)
// ---------------------------------------------------------------------------

/// Page-body compression backend for the StarRocks baseline (ZSTD level 3).
fn sr_block_compress(raw: &[u8]) -> Vec<u8> {
    zstd::stream::encode_all(raw, 3).unwrap()
}
fn sr_block_decompress(comp: &[u8]) -> Vec<u8> {
    zstd::stream::decode_all(comp).unwrap()
}

fn bench_starrocks_numeric_generic<T, S, U>(
    values: &[T],
    shuffle: S,
    unshuffle: U,
    iters: usize,
    name: &'static str,
) -> Measure
where
    T: Clone + PartialEq + std::fmt::Debug,
    S: Fn(&[T]) -> Vec<u8>,
    U: Fn(&[u8], usize) -> Vec<T>,
{
    let shuffled = shuffle(values);
    let compressed_once = sr_block_compress(&shuffled);
    let dec_bytes = sr_block_decompress(&compressed_once);
    let dec_vals = unshuffle(&dec_bytes, values.len());
    assert_eq!(dec_vals, values, "{name} round-trip mismatch");

    let (encode_ns, encoded) = time_median(iters, || {
        let shuf = shuffle(values);
        sr_block_compress(&shuf)
    });
    let n = values.len();
    let (decode_ns, _) = time_median(iters, || {
        let b = sr_block_decompress(&encoded);
        unshuffle(&b, n)
    });
    Measure {
        encoded: encoded.len(),
        encode_ns,
        decode_ns,
    }
}

/// StarRocks DICT_ENCODING for strings: hash unique values, write dict (plain
/// concat with offsets) + indices (bitshuffled u32). All wrapped with LZ4.
/// If dict exceeds dict_page_size, StarRocks falls back to PLAIN — not
/// modeled here; we compute the dict unconditionally for simplicity.
fn bench_starrocks_dict_utf8(values: &[String], iters: usize) -> Measure {
    use std::collections::HashMap;

    // Build dict (deterministic insertion order).
    let mut map: HashMap<String, u32> = HashMap::new();
    let mut dict: Vec<String> = Vec::new();
    let mut indices: Vec<u32> = Vec::with_capacity(values.len());
    for v in values {
        let idx = match map.get(v) {
            Some(&i) => i,
            None => {
                let i = dict.len() as u32;
                map.insert(v.clone(), i);
                dict.push(v.clone());
                i
            }
        };
        indices.push(idx);
    }

    // Encode: dict as (u32 count, u32 offsets[n+1], bytes), indices as bit-shuffled u32 + lz4.
    let encode = || -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(dict.len() as u32).to_le_bytes());
        let mut offsets = Vec::with_capacity(dict.len() + 1);
        offsets.push(0u32);
        let mut data = Vec::new();
        for s in &dict {
            data.extend_from_slice(s.as_bytes());
            offsets.push(data.len() as u32);
        }
        // offsets + data compressed as lz4 (simulating page body compression).
        let mut dict_page = Vec::new();
        for o in &offsets {
            dict_page.extend_from_slice(&o.to_le_bytes());
        }
        dict_page.extend_from_slice(&data);
        let compressed_dict = sr_block_compress(&dict_page);
        out.extend_from_slice(&(compressed_dict.len() as u32).to_le_bytes());
        out.extend_from_slice(&compressed_dict);

        let shuf = sr_shuffle_u32(&indices);
        let compressed_idx = sr_block_compress(&shuf);
        out.extend_from_slice(&(compressed_idx.len() as u32).to_le_bytes());
        out.extend_from_slice(&compressed_idx);
        out
    };
    let enc_once = encode();

    // Decode: verify round-trip.
    let decode = |bytes: &[u8]| -> Vec<String> {
        let dict_count = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
        let dict_len = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        let dict_page = sr_block_decompress(&bytes[8..8 + dict_len]);

        let off_bytes = 4 * (dict_count + 1);
        let mut offsets = Vec::with_capacity(dict_count + 1);
        for i in 0..=dict_count {
            offsets.push(u32::from_le_bytes(
                dict_page[i * 4..i * 4 + 4].try_into().unwrap(),
            ));
        }
        let data = &dict_page[off_bytes..];
        let dict: Vec<String> = (0..dict_count)
            .map(|i| {
                let s = offsets[i] as usize;
                let e = offsets[i + 1] as usize;
                String::from_utf8(data[s..e].to_vec()).unwrap()
            })
            .collect();

        let idx_off = 8 + dict_len;
        let idx_len = u32::from_le_bytes(bytes[idx_off..idx_off + 4].try_into().unwrap()) as usize;
        let idx_shuf = sr_block_decompress(&bytes[idx_off + 4..idx_off + 4 + idx_len]);
        let indices = sr_unshuffle_u32(&idx_shuf, values.len());

        indices
            .into_iter()
            .map(|i| dict[i as usize].clone())
            .collect()
    };
    let dec_once = decode(&enc_once);
    assert_eq!(dec_once, values, "sr_dict_utf8 round-trip mismatch");

    let (encode_ns, encoded) = time_median(iters, encode);
    let (decode_ns, _) = time_median(iters, || decode(&encoded));
    Measure {
        encoded: encoded.len(),
        encode_ns,
        decode_ns,
    }
}

/// StarRocks RLE for booleans (stored as i8 0/1).
fn bench_starrocks_rle_bool(values: &[i8], iters: usize) -> Measure {
    // Simple RLE of u8 (since bool is one byte in StarRocks).
    let encode = |vs: &[i8]| -> Vec<u8> {
        let mut out = Vec::new();
        if vs.is_empty() {
            return out;
        }
        let mut cur = vs[0];
        let mut run: u32 = 1;
        for &v in &vs[1..] {
            if v == cur && run < u32::MAX {
                run += 1;
            } else {
                out.push(cur as u8);
                out.extend_from_slice(&run.to_le_bytes());
                cur = v;
                run = 1;
            }
        }
        out.push(cur as u8);
        out.extend_from_slice(&run.to_le_bytes());
        out
    };
    let decode = |bytes: &[u8], n: usize| -> Vec<i8> {
        let mut out = Vec::with_capacity(n);
        let mut i = 0;
        while i < bytes.len() {
            let v = bytes[i] as i8;
            let run = u32::from_le_bytes(bytes[i + 1..i + 5].try_into().unwrap()) as usize;
            i += 5;
            for _ in 0..run {
                out.push(v);
            }
        }
        out
    };

    let raw = encode(values);
    let compressed_once = sr_block_compress(&raw);
    let dec = sr_block_decompress(&compressed_once);
    let dec_vals = decode(&dec, values.len());
    assert_eq!(dec_vals, values, "sr_rle_bool round-trip mismatch");

    let (encode_ns, encoded) = time_median(iters, || {
        let r = encode(values);
        sr_block_compress(&r)
    });
    let n = values.len();
    let (decode_ns, _) = time_median(iters, || {
        let r = sr_block_decompress(&encoded);
        decode(&r, n)
    });
    Measure {
        encoded: encoded.len(),
        encode_ns,
        decode_ns,
    }
}

// ---------------------------------------------------------------------------
// Helium pipeline bench
// ---------------------------------------------------------------------------

fn nb<T: 'static + NonBlockCoder>(c: T) -> StageCoder {
    StageCoder::NonBlock(Box::new(c))
}
fn blk<T: 'static + BlockCoder>(c: T) -> StageCoder {
    StageCoder::Block(Box::new(c))
}

fn bench_helium(pipeline: &Pipeline, data: ColumnData, iters: usize) -> Measure {
    let enc_once = pipeline.encode(data.clone()).unwrap();
    let dec_once = pipeline.decode(enc_once.clone()).unwrap();
    assert_eq!(dec_once, data, "helium pipeline round-trip mismatch");

    let (encode_ns, encoded) = time_median(iters, || pipeline.encode(data.clone()).unwrap());
    let (decode_ns, _) = time_median(iters, || pipeline.decode(encoded.clone()).unwrap());
    Measure {
        encoded: encoded.len(),
        encode_ns,
        decode_ns,
    }
}

fn bench_helium_logical(
    schema: &Schema,
    column: LogicalColumn,
    column_name: &str,
    iters: usize,
) -> Measure {
    let registry = CoderRegistry::default();
    let pipelines = schema.resolve_all(&registry).unwrap();
    let idx = schema.column_index(column_name).unwrap();
    let col_pipelines = &pipelines[idx];
    let lt = schema.columns[idx].logical_type.clone();
    let row_count = column.row_count();

    // Correctness.
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
        let mut total = 0usize;
        let mut physicals = Vec::with_capacity(col_pipelines.len());
        for (part, pipe) in parts.into_iter().zip(col_pipelines.iter()) {
            let e = pipe.encode(part).unwrap();
            if let ColumnData::Bytes(b) = &e {
                total += b.len();
            }
            physicals.push(e);
        }
        (total, physicals)
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

// ============================================================================
// Row + table formatting (winner per row)
// ============================================================================

struct Row {
    column: String,
    shape: String,
    raw_bytes: usize,
    starrocks: Measure,
    #[allow(dead_code)]
    starrocks_label: String,
    gzip: Measure,
    lz4: Measure,
    zstd: Measure,
    pcodec: Option<Measure>,
    helium: Measure,
    helium_label: String,
    helium_pco: Option<Measure>,
    helium_pco_label: Option<String>,
}

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

fn fmt_cell_ratio(v: Option<f64>, is_best: bool) -> String {
    match v {
        None => "—".into(),
        Some(x) => {
            let s = format!("{x:.1}x");
            if is_best { format!("**{s}**") } else { s }
        }
    }
}
fn fmt_cell_mbps(v: Option<f64>, is_best: bool) -> String {
    match v {
        None => "—".into(),
        Some(x) => {
            let s = format!("{x:.0}");
            if is_best { format!("**{s}**") } else { s }
        }
    }
}
fn fmt_bytes(b: usize) -> String {
    if b >= 1024 * 1024 {
        format!("{:.1} MB", b as f64 / 1_048_576.0)
    } else if b >= 1024 {
        format!("{:.1} KB", b as f64 / 1024.0)
    } else {
        format!("{b} B")
    }
}
fn ratio(raw: usize, enc: usize) -> f64 {
    raw as f64 / enc.max(1) as f64
}
fn mbps(bytes: usize, ns: u128) -> f64 {
    if ns == 0 {
        return f64::INFINITY;
    }
    let sec = ns as f64 / 1e9;
    (bytes as f64 / 1_048_576.0) / sec
}

fn write_ratio_table(out: &mut String, rows: &[Row]) {
    writeln!(
        out,
        "\n### Compression ratio — raw / encoded (higher = better; **bold** = best in row)\n"
    )
    .unwrap();
    writeln!(
        out,
        "| Column | Shape | Raw | sr(enc+zstd) | gzip | lz4 | zstd | pcodec | helium(zstd) | helium(pco) | helium pipeline | +pco pipeline |"
    )
    .unwrap();
    writeln!(
        out,
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"
    )
    .unwrap();
    for r in rows {
        let vals = [
            Some(ratio(r.raw_bytes, r.starrocks.encoded)),
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
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            r.column,
            r.shape,
            fmt_bytes(r.raw_bytes),
            fmt_cell_ratio(vals[0], best == Some(0)),
            fmt_cell_ratio(vals[1], best == Some(1)),
            fmt_cell_ratio(vals[2], best == Some(2)),
            fmt_cell_ratio(vals[3], best == Some(3)),
            fmt_cell_ratio(vals[4], best == Some(4)),
            fmt_cell_ratio(vals[5], best == Some(5)),
            fmt_cell_ratio(vals[6], best == Some(6)),
            r.helium_label,
            r.helium_pco_label.as_deref().unwrap_or("—"),
        )
        .unwrap();
    }
    writeln!(out).unwrap();
    writeln!(
        out,
        "_sr(enc+zstd) is StarRocks' **best-achievable** config per column type: BIT_SHUFFLE + ZSTD for numerics, DICT + ZSTD for strings, RLE + ZSTD for booleans. StarRocks ships LZ4 by default (weaker); we baseline against ZSTD so any Helium win is conservative. Byte-shuffle stands in for the upstream bit-shuffle library (no extra dep) — typically under 10% difference on ratio._"
    )
    .unwrap();
}

fn write_throughput_table(
    out: &mut String,
    rows: &[Row],
    which: &str,
    extract: impl Fn(&Measure) -> u128,
) {
    writeln!(
        out,
        "\n### {which} throughput — MB/s of raw (higher = better; **bold** = best in row)\n"
    )
    .unwrap();
    writeln!(
        out,
        "| Column | sr(enc+zstd) | gzip | lz4 | zstd | pcodec | helium(zstd) | helium(pco) |"
    )
    .unwrap();
    writeln!(out, "|---|---:|---:|---:|---:|---:|---:|---:|").unwrap();
    for r in rows {
        let vals = [
            Some(mbps(r.raw_bytes, extract(&r.starrocks))),
            Some(mbps(r.raw_bytes, extract(&r.gzip))),
            Some(mbps(r.raw_bytes, extract(&r.lz4))),
            Some(mbps(r.raw_bytes, extract(&r.zstd))),
            r.pcodec.map(|m| mbps(r.raw_bytes, extract(&m))),
            Some(mbps(r.raw_bytes, extract(&r.helium))),
            r.helium_pco.map(|m| mbps(r.raw_bytes, extract(&m))),
        ];
        let best = best_index(&vals);
        writeln!(
            out,
            "| {} | {} | {} | {} | {} | {} | {} | {} |",
            r.column,
            fmt_cell_mbps(vals[0], best == Some(0)),
            fmt_cell_mbps(vals[1], best == Some(1)),
            fmt_cell_mbps(vals[2], best == Some(2)),
            fmt_cell_mbps(vals[3], best == Some(3)),
            fmt_cell_mbps(vals[4], best == Some(4)),
            fmt_cell_mbps(vals[5], best == Some(5)),
            fmt_cell_mbps(vals[6], best == Some(6)),
        )
        .unwrap();
    }
}

// ============================================================================
// Per-column row builders
// ============================================================================

const ITERS: usize = 5;

fn row_watch_id(values: &[i64]) -> Row {
    let raw = flatten_i64(values);
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
        column: "WatchID".into(),
        shape: "i64 monotone".into(),
        raw_bytes: raw.len(),
        starrocks: bench_starrocks_numeric_generic(
            values,
            sr_shuffle_i64,
            sr_unshuffle_i64,
            ITERS,
            "sr_bitshuffle_i64",
        ),
        starrocks_label: "BIT_SHUFFLE+ZSTD".into(),
        gzip: bench_gzip(&raw, ITERS),
        lz4: bench_lz4(&raw, ITERS),
        zstd: bench_zstd(&raw, ITERS),
        pcodec: Some(bench_pcodec_typed(values, ITERS)),
        helium: bench_helium(&pipeline, ColumnData::I64(values.to_vec()), ITERS),
        helium_label: "delta+leb128+zstd".into(),
        helium_pco: Some(bench_helium(
            &pipeline_pco,
            ColumnData::I64(values.to_vec()),
            ITERS,
        )),
        helium_pco_label: Some("delta+pcodec".into()),
    }
}

fn row_user_id(values: &[i64]) -> Row {
    let raw = flatten_i64(values);
    // User IDs aren't sorted and have moderate cardinality — dictionary helps.
    let dict_col = LogicalColumn::dict_encode_primitive(ColumnData::I64(values.to_vec())).unwrap();
    let inner_lt = helium::LogicalType::Primitive {
        data_type: DataType::I64,
    };
    let dict_lt = helium::LogicalType::Dictionary {
        inner: Box::new(inner_lt),
    };
    let dict_schema = Schema::new(vec![helium::ColumnSpec::new(
        "c",
        dict_lt.clone(),
        vec![
            vec![CoderSpec::new("leb128"), CoderSpec::new("zstd")],
            vec![CoderSpec::new("bitpack_auto"), CoderSpec::new("zstd")],
        ],
    )]);
    let dict_pco_schema = Schema::new(vec![helium::ColumnSpec::new(
        "c",
        dict_lt,
        vec![
            vec![CoderSpec::new("leb128"), CoderSpec::new("zstd")],
            vec![CoderSpec::new("pcodec")],
        ],
    )]);
    Row {
        column: "UserID".into(),
        shape: "i64 repeating".into(),
        raw_bytes: raw.len(),
        starrocks: bench_starrocks_numeric_generic(
            values,
            sr_shuffle_i64,
            sr_unshuffle_i64,
            ITERS,
            "sr_bitshuffle_i64",
        ),
        starrocks_label: "BIT_SHUFFLE+ZSTD".into(),
        gzip: bench_gzip(&raw, ITERS),
        lz4: bench_lz4(&raw, ITERS),
        zstd: bench_zstd(&raw, ITERS),
        pcodec: Some(bench_pcodec_typed(values, ITERS)),
        helium: bench_helium_logical(&dict_schema, dict_col.clone(), "c", ITERS),
        helium_label: "Dict<I64>(bitpack+zstd)".into(),
        helium_pco: Some(bench_helium_logical(&dict_pco_schema, dict_col, "c", ITERS)),
        helium_pco_label: Some("Dict<I64>(pcodec)".into()),
    }
}

fn row_client_ip(values: &[u32]) -> Row {
    let raw = flatten_u32(values);
    let pipeline = Pipeline::new(
        DataType::U32,
        vec![
            nb(Leb128::new(DataType::U32).unwrap()),
            blk(Zstd::default()),
        ],
    )
    .unwrap();
    Row {
        column: "ClientIP".into(),
        shape: "u32 random".into(),
        raw_bytes: raw.len(),
        starrocks: bench_starrocks_numeric_generic(
            values,
            sr_shuffle_u32,
            sr_unshuffle_u32,
            ITERS,
            "sr_bitshuffle_u32",
        ),
        starrocks_label: "BIT_SHUFFLE+ZSTD".into(),
        gzip: bench_gzip(&raw, ITERS),
        lz4: bench_lz4(&raw, ITERS),
        zstd: bench_zstd(&raw, ITERS),
        pcodec: Some(bench_pcodec_typed(values, ITERS)),
        helium: bench_helium(&pipeline, ColumnData::U32(values.to_vec()), ITERS),
        helium_label: "leb128+zstd".into(),
        helium_pco: None,
        helium_pco_label: None,
    }
}

fn row_event_time(values: &[i32]) -> Row {
    let raw = flatten_i32(values);
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
        column: "EventTime".into(),
        shape: "i32 ~monotone".into(),
        raw_bytes: raw.len(),
        starrocks: bench_starrocks_numeric_generic(
            values,
            sr_shuffle_i32,
            sr_unshuffle_i32,
            ITERS,
            "sr_bitshuffle_i32",
        ),
        starrocks_label: "BIT_SHUFFLE+ZSTD".into(),
        gzip: bench_gzip(&raw, ITERS),
        lz4: bench_lz4(&raw, ITERS),
        zstd: bench_zstd(&raw, ITERS),
        pcodec: Some(bench_pcodec_typed(values, ITERS)),
        helium: bench_helium(&pipeline, ColumnData::I32(values.to_vec()), ITERS),
        helium_label: "delta+leb128+zstd".into(),
        helium_pco: Some(bench_helium(
            &pipeline_pco,
            ColumnData::I32(values.to_vec()),
            ITERS,
        )),
        helium_pco_label: Some("delta+pcodec".into()),
    }
}

fn row_event_date(values: &[i16]) -> Row {
    let raw = flatten_i16(values);
    let pipeline = Pipeline::new(
        DataType::I16,
        vec![
            blk(DeltaMin::new(DataType::I16).unwrap()),
            blk(BitpackAuto::new(DataType::I16).unwrap()),
            blk(Zstd::default()),
        ],
    )
    .unwrap();
    Row {
        column: "EventDate".into(),
        shape: "i16 low-card days".into(),
        raw_bytes: raw.len(),
        starrocks: bench_starrocks_numeric_generic(
            values,
            sr_shuffle_i16,
            sr_unshuffle_i16,
            ITERS,
            "sr_bitshuffle_i16",
        ),
        starrocks_label: "BIT_SHUFFLE+ZSTD".into(),
        gzip: bench_gzip(&raw, ITERS),
        lz4: bench_lz4(&raw, ITERS),
        zstd: bench_zstd(&raw, ITERS),
        pcodec: None,
        helium: bench_helium(&pipeline, ColumnData::I16(values.to_vec()), ITERS),
        helium_label: "deltamin+bitpack_auto+zstd".into(),
        helium_pco: None,
        helium_pco_label: None,
    }
}

fn row_os_code(values: &[u8]) -> Row {
    let raw = flatten_u8(values);
    let pipeline = Pipeline::new(
        DataType::U8,
        vec![
            nb(Rle::new(DataType::U8).unwrap()),
            nb(Leb128::new(DataType::U8).unwrap()),
            blk(Zstd::default()),
        ],
    )
    .unwrap();
    Row {
        column: "OS".into(),
        shape: "u8 low-card".into(),
        raw_bytes: raw.len(),
        starrocks: bench_starrocks_numeric_generic(
            values,
            |v| v.to_vec(),
            |b, n| b[..n].to_vec(),
            ITERS,
            "sr_u8_identity_lz4",
        ),
        starrocks_label: "BIT_SHUFFLE+ZSTD".into(),
        gzip: bench_gzip(&raw, ITERS),
        lz4: bench_lz4(&raw, ITERS),
        zstd: bench_zstd(&raw, ITERS),
        pcodec: None,
        helium: bench_helium(&pipeline, ColumnData::U8(values.to_vec()), ITERS),
        helium_label: "rle+leb128+zstd".into(),
        helium_pco: None,
        helium_pco_label: None,
    }
}

fn row_country_id(values: &[i16]) -> Row {
    let raw = flatten_i16(values);
    let pipeline = Pipeline::new(
        DataType::I16,
        vec![
            blk(DeltaMin::new(DataType::I16).unwrap()),
            blk(BitpackAuto::new(DataType::I16).unwrap()),
            blk(Zstd::default()),
        ],
    )
    .unwrap();
    Row {
        column: "CountryID".into(),
        shape: "i16 skewed".into(),
        raw_bytes: raw.len(),
        starrocks: bench_starrocks_numeric_generic(
            values,
            sr_shuffle_i16,
            sr_unshuffle_i16,
            ITERS,
            "sr_bitshuffle_i16",
        ),
        starrocks_label: "BIT_SHUFFLE+ZSTD".into(),
        gzip: bench_gzip(&raw, ITERS),
        lz4: bench_lz4(&raw, ITERS),
        zstd: bench_zstd(&raw, ITERS),
        pcodec: None,
        helium: bench_helium(&pipeline, ColumnData::I16(values.to_vec()), ITERS),
        helium_label: "deltamin+bitpack_auto+zstd".into(),
        helium_pco: None,
        helium_pco_label: None,
    }
}

fn row_is_refresh(values: &[i8]) -> Row {
    let raw = flatten_i8(values);
    let pipeline = Pipeline::new(
        DataType::I8,
        vec![
            nb(Rle::new(DataType::I8).unwrap()),
            nb(Leb128::new(DataType::I8).unwrap()),
            blk(Zstd::default()),
        ],
    )
    .unwrap();
    Row {
        column: "IsRefresh".into(),
        shape: "bool (i8)".into(),
        raw_bytes: raw.len(),
        starrocks: bench_starrocks_rle_bool(values, ITERS),
        starrocks_label: "RLE+ZSTD".into(),
        gzip: bench_gzip(&raw, ITERS),
        lz4: bench_lz4(&raw, ITERS),
        zstd: bench_zstd(&raw, ITERS),
        pcodec: None,
        helium: bench_helium(&pipeline, ColumnData::I8(values.to_vec()), ITERS),
        helium_label: "rle+leb128+zstd".into(),
        helium_pco: None,
        helium_pco_label: None,
    }
}

fn row_title(values: &[String]) -> Row {
    let raw = flatten_strings(values);
    let dict_lt = helium::LogicalType::Dictionary {
        inner: Box::new(helium::LogicalType::Utf8),
    };
    let schema = Schema::new(vec![helium::ColumnSpec::new(
        "s",
        dict_lt.clone(),
        vec![
            vec![CoderSpec::new("leb128"), CoderSpec::new("zstd")],
            vec![CoderSpec::new("zstd")],
            vec![CoderSpec::new("bitpack_auto"), CoderSpec::new("zstd")],
        ],
    )]);
    let schema_pco = Schema::new(vec![helium::ColumnSpec::new(
        "s",
        dict_lt,
        vec![
            vec![CoderSpec::new("leb128"), CoderSpec::new("zstd")],
            vec![CoderSpec::new("zstd")],
            vec![CoderSpec::new("pcodec")],
        ],
    )]);
    let col = LogicalColumn::dict_encode_utf8(values.to_vec());
    Row {
        column: "Title".into(),
        shape: "utf8 medium-card".into(),
        raw_bytes: raw.len(),
        starrocks: bench_starrocks_dict_utf8(values, ITERS),
        starrocks_label: "DICT+ZSTD".into(),
        gzip: bench_gzip(&raw, ITERS),
        lz4: bench_lz4(&raw, ITERS),
        zstd: bench_zstd(&raw, ITERS),
        pcodec: None,
        helium: bench_helium_logical(&schema, col.clone(), "s", ITERS),
        helium_label: "Dict<Utf8>(bitpack_auto+zstd)".into(),
        helium_pco: Some(bench_helium_logical(&schema_pco, col, "s", ITERS)),
        helium_pco_label: Some("Dict<Utf8>(pcodec)".into()),
    }
}

fn row_url(values: &[String]) -> Row {
    let raw = flatten_strings(values);
    // High-cardinality URLs — plain Utf8 with zstd, no dict (dict wouldn't
    // help if every URL is distinct, matching StarRocks's fallback to PLAIN).
    let schema = Schema::new(vec![helium::ColumnSpec::utf8(
        "s",
        vec![
            CoderSpec::new("delta"),
            CoderSpec::new("leb128"),
            CoderSpec::new("zstd"),
        ],
        vec![CoderSpec::new("zstd").with_param("level", 6)],
    )]);
    let col = LogicalColumn::Utf8(values.to_vec());
    Row {
        column: "URL".into(),
        shape: "utf8 high-card".into(),
        raw_bytes: raw.len(),
        starrocks: bench_starrocks_dict_utf8(values, ITERS),
        starrocks_label: "DICT+ZSTD".into(),
        gzip: bench_gzip(&raw, ITERS),
        lz4: bench_lz4(&raw, ITERS),
        zstd: bench_zstd(&raw, ITERS),
        pcodec: None,
        helium: bench_helium_logical(&schema, col, "s", ITERS),
        helium_label: "Utf8(zstd L6)".into(),
        helium_pco: None,
        helium_pco_label: None,
    }
}

fn row_search_phrase(values: &[String]) -> Row {
    let raw = flatten_strings(values);
    let dict_lt = helium::LogicalType::Dictionary {
        inner: Box::new(helium::LogicalType::Utf8),
    };
    let schema = Schema::new(vec![helium::ColumnSpec::new(
        "s",
        dict_lt,
        vec![
            vec![CoderSpec::new("leb128"), CoderSpec::new("zstd")],
            vec![CoderSpec::new("zstd")],
            vec![
                CoderSpec::new("rle"),
                CoderSpec::new("leb128"),
                CoderSpec::new("zstd"),
            ],
        ],
    )]);
    let col = LogicalColumn::dict_encode_utf8(values.to_vec());
    Row {
        column: "SearchPhrase".into(),
        shape: "utf8 low-card (many empty)".into(),
        raw_bytes: raw.len(),
        starrocks: bench_starrocks_dict_utf8(values, ITERS),
        starrocks_label: "DICT+ZSTD".into(),
        gzip: bench_gzip(&raw, ITERS),
        lz4: bench_lz4(&raw, ITERS),
        zstd: bench_zstd(&raw, ITERS),
        pcodec: None,
        helium: bench_helium_logical(&schema, col, "s", ITERS),
        helium_label: "Dict<Utf8>(rle+leb128+zstd)".into(),
        helium_pco: None,
        helium_pco_label: None,
    }
}

// ============================================================================
// Test body
// ============================================================================

fn ua_col_from_u8(v: Vec<u8>) -> Row {
    let raw = flatten_u8(&v);
    let pipeline = Pipeline::new(
        DataType::U8,
        vec![
            nb(Rle::new(DataType::U8).unwrap()),
            nb(Leb128::new(DataType::U8).unwrap()),
            blk(Zstd::default()),
        ],
    )
    .unwrap();
    Row {
        column: "UserAgent".into(),
        shape: "u8 low-card".into(),
        raw_bytes: raw.len(),
        starrocks: bench_starrocks_numeric_generic(
            &v,
            |x| x.to_vec(),
            |b, n| b[..n].to_vec(),
            ITERS,
            "sr_u8_identity_lz4",
        ),
        starrocks_label: "BIT_SHUFFLE+ZSTD".into(),
        gzip: bench_gzip(&raw, ITERS),
        lz4: bench_lz4(&raw, ITERS),
        zstd: bench_zstd(&raw, ITERS),
        pcodec: None,
        helium: bench_helium(&pipeline, ColumnData::U8(v), ITERS),
        helium_label: "rle+leb128+zstd".into(),
        helium_pco: None,
        helium_pco_label: None,
    }
}

// --- Time-series / sensor float + timestamp columns (pcodec's main case) ---

fn row_sensor_value(values: &[f64]) -> Row {
    let raw = flatten_f64(values);
    let pipeline = Pipeline::new(
        DataType::F64,
        vec![nb(GorillaXor::new(DataType::F64).unwrap()), blk(Zstd::default())],
    )
    .unwrap();
    let pipeline_pco =
        Pipeline::new(DataType::F64, vec![blk(Pcodec::new(DataType::F64, None).unwrap())]).unwrap();
    Row {
        column: "SensorValue".into(),
        shape: "f64 slow-varying".into(),
        raw_bytes: raw.len(),
        starrocks: bench_starrocks_numeric_generic(
            values,
            sr_shuffle_f64,
            sr_unshuffle_f64,
            ITERS,
            "sr_shuffle_f64",
        ),
        starrocks_label: "BIT_SHUFFLE+ZSTD".into(),
        gzip: bench_gzip(&raw, ITERS),
        lz4: bench_lz4(&raw, ITERS),
        zstd: bench_zstd(&raw, ITERS),
        pcodec: Some(bench_pcodec_typed(values, ITERS)),
        helium: bench_helium(&pipeline, ColumnData::F64(values.to_vec()), ITERS),
        helium_label: "gorilla+zstd".into(),
        helium_pco: Some(bench_helium(&pipeline_pco, ColumnData::F64(values.to_vec()), ITERS)),
        helium_pco_label: Some("pcodec".into()),
    }
}

fn row_cpu_pct(values: &[f32]) -> Row {
    let raw = flatten_f32(values);
    let pipeline = Pipeline::new(
        DataType::F32,
        vec![nb(GorillaXor::new(DataType::F32).unwrap()), blk(Zstd::default())],
    )
    .unwrap();
    let pipeline_pco =
        Pipeline::new(DataType::F32, vec![blk(Pcodec::new(DataType::F32, None).unwrap())]).unwrap();
    Row {
        column: "CpuPct".into(),
        shape: "f32 slow-varying".into(),
        raw_bytes: raw.len(),
        starrocks: bench_starrocks_numeric_generic(
            values,
            sr_shuffle_f32,
            sr_unshuffle_f32,
            ITERS,
            "sr_shuffle_f32",
        ),
        starrocks_label: "BIT_SHUFFLE+ZSTD".into(),
        gzip: bench_gzip(&raw, ITERS),
        lz4: bench_lz4(&raw, ITERS),
        zstd: bench_zstd(&raw, ITERS),
        pcodec: Some(bench_pcodec_typed(values, ITERS)),
        helium: bench_helium(&pipeline, ColumnData::F32(values.to_vec()), ITERS),
        helium_label: "gorilla+zstd".into(),
        helium_pco: Some(bench_helium(&pipeline_pco, ColumnData::F32(values.to_vec()), ITERS)),
        helium_pco_label: Some("pcodec".into()),
    }
}

fn row_event_ts(values: &[i64]) -> Row {
    let raw = flatten_i64(values);
    // Near-constant interval → delta-of-delta collapses to near-zero.
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
        vec![nb(Delta::new(DataType::I64).unwrap()), blk(Pcodec::new(DataType::I64, None).unwrap())],
    )
    .unwrap();
    Row {
        column: "EventTsMicros".into(),
        shape: "i64 ts ~1ms".into(),
        raw_bytes: raw.len(),
        starrocks: bench_starrocks_numeric_generic(
            values,
            sr_shuffle_i64,
            sr_unshuffle_i64,
            ITERS,
            "sr_shuffle_i64",
        ),
        starrocks_label: "BIT_SHUFFLE+ZSTD".into(),
        gzip: bench_gzip(&raw, ITERS),
        lz4: bench_lz4(&raw, ITERS),
        zstd: bench_zstd(&raw, ITERS),
        pcodec: Some(bench_pcodec_typed(values, ITERS)),
        helium: bench_helium(&pipeline, ColumnData::I64(values.to_vec()), ITERS),
        helium_label: "dod+leb128+zstd".into(),
        helium_pco: Some(bench_helium(&pipeline_pco, ColumnData::I64(values.to_vec()), ITERS)),
        helium_pco_label: Some("delta+pcodec".into()),
    }
}

// ============================================================================
// Decision summary — the number the go/no-go rests on
// ============================================================================

/// For each column, take Helium's **best** pipeline (higher compression ratio
/// of the two variants) and score it against the StarRocks (enc+ZSTD) baseline
/// on two thresholds: ratio gain ≥ 15% AND decode-throughput retention ≥ 70%
/// (i.e. ≤ 30% slowdown). This is the table that says whether a column is worth
/// a C++ prototype.
fn write_decision_summary(out: &mut String, rows: &[Row]) {
    writeln!(out, "\n## Decision summary\n").unwrap();
    writeln!(
        out,
        "Helium's best pipeline per column vs StarRocks (enc+ZSTD). \
         **PASS** = compression-ratio gain ≥ 15% **and** decode throughput ≥ 70% of baseline.\n"
    )
    .unwrap();
    writeln!(
        out,
        "| Column | sr ratio | best helium | h. ratio | ratio gain | sr decode MB/s | h. decode MB/s | decode kept | verdict |"
    )
    .unwrap();
    writeln!(out, "|---|---:|---|---:|---:|---:|---:|---:|:--:|").unwrap();

    let mut passes = 0usize;
    for r in rows {
        let sr_ratio = ratio(r.raw_bytes, r.starrocks.encoded);
        let sr_dec = mbps(r.raw_bytes, r.starrocks.decode_ns);

        // Pick the higher-ratio Helium variant.
        let mut best = (r.helium, r.helium_label.clone());
        if let (Some(p), Some(pl)) = (r.helium_pco, r.helium_pco_label.as_deref())
            && ratio(r.raw_bytes, p.encoded) > ratio(r.raw_bytes, best.0.encoded)
        {
            best = (p, pl.to_string());
        }
        let h_ratio = ratio(r.raw_bytes, best.0.encoded);
        let h_dec = mbps(r.raw_bytes, best.0.decode_ns);
        let gain = (h_ratio / sr_ratio - 1.0) * 100.0;
        let kept = (h_dec / sr_dec) * 100.0;
        let pass = gain >= 15.0 && kept >= 70.0;
        if pass {
            passes += 1;
        }
        writeln!(
            out,
            "| {} | {:.1}x | {} | {:.1}x | {:+.0}% | {:.0} | {:.0} | {:.0}% | {} |",
            r.column,
            sr_ratio,
            best.1,
            h_ratio,
            gain,
            sr_dec,
            h_dec,
            kept,
            if pass { "✅ PASS" } else { "—" },
        )
        .unwrap();
    }
    writeln!(
        out,
        "\n**{passes}/{} columns clear both thresholds.** These are the prototype candidates — \
         float/timestamp columns are expected to dominate (StarRocks' BIT_SHUFFLE is weakest there). \
         Decode retention is the risk signal to watch: scalar Helium decode that already lags badly \
         here will lag worse in StarRocks' vectorised scan.\n",
        rows.len()
    )
    .unwrap();
}

#[test]
fn starrocks_report() {
    let parquet_path = std::env::var("HELIUM_PARQUET_PATH").ok();

    let mut rows = Vec::new();
    if let Some(path) = parquet_path.as_deref() {
        eprintln!("Using Parquet input: {path} (reading {N_ROWS} rows)");
        let p = std::path::PathBuf::from(path);
        let batch = parquet_loader::load_batch(&p, N_ROWS)
            .unwrap_or_else(|| panic!("failed to read parquet file {path}"));

        // Strings
        if let Some(url) = parquet_loader::strings(&batch, "URL") {
            rows.push(row_url(&url));
        }
        if let Some(title) = parquet_loader::strings(&batch, "Title") {
            rows.push(row_title(&title));
        }
        if let Some(sp) = parquet_loader::strings(&batch, "SearchPhrase") {
            rows.push(row_search_phrase(&sp));
        }

        // 64-bit integers
        if let Some(wid) = parquet_loader::i64s(&batch, "WatchID") {
            rows.push(row_watch_id(&wid));
        }
        if let Some(uid) = parquet_loader::i64s(&batch, "UserID") {
            rows.push(row_user_id(&uid));
        }
        if let Some(et) = parquet_loader::i64s(&batch, "EventTime") {
            // ClickBench stores EventTime as INT64 (unix seconds-ish).
            let as_i32: Vec<i32> = et.into_iter().map(|x| x as i32).collect();
            rows.push(row_event_time(&as_i32));
        }

        // 32-bit
        if let Some(ip) = parquet_loader::u32s(&batch, "ClientIP") {
            rows.push(row_client_ip(&ip));
        }

        // 16-bit (ClickBench stores these as INT32 with Int16 logical)
        if let Some(d) = parquet_loader::i16s(&batch, "EventDate") {
            rows.push(row_event_date(&d));
        }
        if let Some(os) = parquet_loader::u8s(&batch, "OS") {
            rows.push(row_os_code(&os));
        }
        if let Some(ua) = parquet_loader::u8s(&batch, "UserAgent") {
            rows.push(ua_col_from_u8(ua));
        }
        if let Some(refresh) = parquet_loader::i8s(&batch, "IsRefresh") {
            rows.push(row_is_refresh(&refresh));
        }
        if let Some(rc) = parquet_loader::i16s(&batch, "RefererCategoryID") {
            rows.push(Row {
                column: "RefererCategoryID".into(),
                shape: "i16 low-card".into(),
                ..row_country_id(&rc)
            });
        }

        if rows.is_empty() {
            panic!("HELIUM_PARQUET_PATH set but no recognized columns loaded from {path}");
        }
    } else {
        eprintln!(
            "HELIUM_PARQUET_PATH not set — using synthetic ClickBench-shape dataset ({N_ROWS} rows)."
        );
        rows.push(row_watch_id(&gen_watch_id(N_ROWS)));
        rows.push(row_user_id(&gen_user_id(N_ROWS)));
        rows.push(row_client_ip(&gen_client_ip(N_ROWS)));
        rows.push(row_event_time(&gen_event_time(N_ROWS)));
        rows.push(row_event_date(&gen_event_date(N_ROWS)));
        rows.push(row_os_code(&gen_os_code(N_ROWS)));
        rows.push(ua_col_from_u8(gen_user_agent(N_ROWS)));
        rows.push(row_country_id(&gen_country_id(N_ROWS)));
        rows.push(row_is_refresh(&gen_is_refresh(N_ROWS)));
        rows.push(row_title(&gen_title(N_ROWS)));
        rows.push(row_url(&gen_url(N_ROWS)));
        rows.push(row_search_phrase(&gen_search_phrase(N_ROWS)));
        // Time-series float/timestamp columns — pcodec's main case (no
        // ClickBench equivalent, so synthetic-only).
        rows.push(row_sensor_value(&gen_sensor_f64(N_ROWS)));
        rows.push(row_cpu_pct(&gen_cpu_f32(N_ROWS)));
        rows.push(row_event_ts(&gen_event_ts_micros(N_ROWS)));
    }

    let mut report = String::new();
    writeln!(
        &mut report,
        "# StarRocks offline compression report (decision-grade)"
    )
    .unwrap();
    writeln!(
        &mut report,
        "\nCompares Helium pipelines against StarRocks' **best-achievable** per-column \
        config — BIT_SHUFFLE + **ZSTD** for numerics, DICT + ZSTD for strings, RLE + ZSTD \
        for booleans — plus four generic baselines (gzip / lz4 / zstd / pcodec), on a \
        ClickBench-shape column set augmented with time-series float/timestamp columns \
        (pcodec's main case).\n\
        \n\
        StarRocks ships LZ4 by default; baselining against ZSTD makes any Helium win \
        **conservative**. The decision summary below applies two thresholds \
        (compression-ratio gain ≥ 15% AND decode-throughput retention ≥ 70% of the \
        StarRocks baseline) to decide which columns justify a C++ prototype.\n\
        \n\
        **Every entry is round-trip verified** before timing — a broken coder fails \
        the test rather than producing pretty numbers.\n\
        \n\
        Dataset: {}\
        \n\n_PoC caveat: byte-shuffle stands in for the upstream bit-shuffle library \
        (no extra dep); ratio error versus real StarRocks is typically under 10%._",
        parquet_path
            .as_deref()
            .unwrap_or(&format!("synthetic ({N_ROWS} rows)"))
    )
    .unwrap();

    write_decision_summary(&mut report, &rows);
    write_ratio_table(&mut report, &rows);
    write_throughput_table(&mut report, &rows, "Encode", |m| m.encode_ns);
    write_throughput_table(&mut report, &rows, "Decode", |m| m.decode_ns);

    print!("{report}");
    std::fs::create_dir_all("target").ok();
    std::fs::write("target/starrocks-report.md", &report).expect("write report");

    // Sanity: every helium pipeline must round-trip (asserted above) AND not be
    // catastrophically worse than baseline. Floor: ratio >= 0.5x raw (no pipeline
    // should more than double data size).
    for r in &rows {
        let hrat = ratio(r.raw_bytes, r.helium.encoded);
        assert!(
            hrat > 0.5,
            "{} helium pipeline expanded data to {:.2}x raw, pipeline: {}",
            r.column,
            1.0 / hrat,
            r.helium_label
        );
    }
}
