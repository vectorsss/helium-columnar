# Performance baseline

Numbers measured at `helium v0.2.x` (synthetic / ClickBench sections originally at v0.2.0 `b6f24a8`; the real-data DoNext section added at v0.2.4). Methodology + reproduction commands below each table — paste them at your shell, get matching results within ±10% on similar hardware.

> **TL;DR**: helium's compression edge comes from the **optimizer**, not the default schema. On real telemetry (DoNext 5G) helium-optimized beats `csv.zst` by ~30% and the `Avro+zstd` 5G storage anchor by ~42%; helium-*default* roughly ties or slightly loses to both. See §1.5 for the real-data numbers.

## Test environment

| Item | Value |
|---|---|
| CPU | Apple M-series (aarch64) |
| OS | macOS 25.4.0 |
| Rust | rustc 1.91.1 (stable) |
| Build | `--release`, `cargo` default LTO settings |
| Dataset | ClickBench `hits_1.parquet`, first 1 M rows × 105 flat columns (8.7 MB CSV equivalent / 408 MB Parquet+snappy from upstream) |
| `helium` features | `cli,datafusion` |

For Linux numbers, use `/usr/bin/time -v` instead of `/usr/bin/time -l` and substitute `time` itself for shell wall-clock. Throughput numbers will be similar within ±20% on equivalent x86_64.

## Get the dataset

```bash
# Easiest: fetch hits_1.parquet (+ convert to hits_1.he) via the helper.
scripts/fetch-fixtures.sh

# Or download ClickBench's hits_1.parquet directly (~408 MB, 1M rows × 105 cols):
curl -O https://datasets.clickhouse.com/hits_compatible/athena_partitioned/hits_1.parquet
# Or use any Parquet file with predominantly flat columns; substitute the path
# below as needed.
```

## 1. Compression ratios — Helium vs Parquet vs pure zstd / lz4

```bash
# Default 10k rows (~5 min):
HELIUM_PARQUET_PATH=hits_1.parquet \
  cargo test --test format_comparison_report --release --all-features -- --nocapture

# Full or representative scale via env override (100k rows ≈ 2 min; 1M rows >1h):
HELIUM_REPORT_MAX_ROWS=100000 HELIUM_PARQUET_PATH=hits_1.parquet \
  cargo test --test format_comparison_report --release --all-features -- --nocapture
# Persists to target/format-comparison.md.
```

**Compression ratios scale with row count**. The default 10k row cap is too small for Helium's pipelines (delta, gorilla, pcodec, dict) to pull ahead — at that scale per-stripe footer overhead and pipeline-init costs dominate. **100k rows is the representative number** for "real-world wide-Parquet workload"; results shift meaningfully between the two.

### At 100 k rows × 105 flat columns (representative)

| Format / Configuration | bytes | vs csv | vs parquet/snappy |
|---|---:|---:|---:|
| **helium-v6 optimized** | **7.46 MB** | 11.8 × | **1.57 ×** |
| pure zstd (raw concatenated bytes) | 7.56 MB | 11.6 × | 1.55 × |
| helium-v5 default | 8.65 MB | 10.2 × | 1.36 × |
| parquet (zstd) | 8.93 MB | 9.9 × | 1.31 × |
| helium-v6 optimized (10k-row stripes) | 8.98 MB | 9.8 × | 1.31 × |
| helium-v6 default (10k-row stripes) | 10.08 MB | 8.7 × | 1.16 × |
| helium-v6 optimized (lz4 terminal) | 10.86 MB | 8.1 × | 1.08 × |
| pure lz4 (raw concatenated bytes) | 11.32 MB | 7.8 × | 1.04 × |
| parquet (lz4_raw) | 11.34 MB | 7.7 × | 1.03 × |
| parquet (snappy, default) | 11.73 MB | 7.5 × | 1.00 × |
| csv.zst | 12.23 MB | 7.2 × | 0.96 × |
| helium-v6 default (lz4 terminal) | 12.02 MB | 7.3 × | 0.98 × |
| ndjson.zst | 13.71 MB | 6.4 × | 0.86 × |
| avro (deflate) | 14.62 MB | 6.0 × | 0.80 × |
| csv (raw) | 87.91 MB | 1.00 × | 0.13 × |
| ndjson (raw) | 235.25 MB | 0.37 × | 0.05 × |

Headline takeaways:

- **Helium-v6 optimized vs parquet+zstd**: 7.46 MB vs 8.93 MB — **16.5% smaller**. This is the real headline number; earlier "1% gap" was measured at 10 k rows where Helium's pipelines hadn't broken even on framing cost yet.
- **Helium-v6 optimized beats pure-zstd by 1.3 %** at 100 k rows. The columnar shaping (delta + leb128 / gorilla / dict / pcodec) pulls ahead once there's enough data per column for the pipelines to amortize their per-column framing.
- **Helium-v5 *default* still loses to pure-zstd** (-14.4 %) — the optimizer's coder selection is doing real work, not just scheduling. Default schema is good for "I just want a binary format that works"; pay attention to `optimize-schema` for any workload that's compression-sensitive.
- **Terminal zstd vs lz4 on the same pipeline**: lz4 39 % larger on default schema. Picking `lz4_terminal` is essentially "I want fast decompression, accept the size cost".
- **`LZ4_RAW` is the right Parquet lz4 variant**: pure-lz4 (11.32 MB) and parquet/lz4_raw (11.34 MB) within 0.2 %.
- **Multi-stripe overhead at 100 k rows / 10 stripes**: helium-v6 optimized goes 7.46 MB → 8.98 MB (+20.4 %) — bigger than the 10 k case (+5 %) because we now have 10 stripes' worth of footer entries. The trade is real: choose single-stripe for cold storage, multi-stripe when SQL pruning matters at query time.

### At 10 k rows × 105 flat columns (small-data artifact, NOT representative)

| Format | bytes | vs parquet/snappy |
|---|---:|---:|
| **helium-v6 optimized** | 922 KB | 1.31 × |
| parquet (zstd) | 931 KB | 1.30 × |
| pure zstd (raw bytes) | 806 KB | 1.50 × |

At 10 k rows, helium-v6 optimized is essentially tied with parquet+zstd (within 1 %), and pure-zstd-on-raw-bytes actually beats both. **Don't use 10 k row numbers as a proxy for production scale** — they understate Helium's compression by ~15 %.

The reason: each stripe carries fixed footer overhead (per-physical-column entry × 105 cols ≈ 25 KB/stripe). At 10 k rows that's ~25 KB / 1 MB ≈ 2-3 % overhead; at 100 k rows it's ~25 KB / 10 MB ≈ 0.25 % — almost free. And the pipeline shaping (delta, leb128) needs enough redundancy in the data stream for zstd to extract real value.

### At 1 M rows (full ClickBench `hits_1.parquet`)

Reproduce with `HELIUM_REPORT_MAX_ROWS=1000000` — takes >1 hour on Apple Silicon (the optimizer's measure-encoding path is O(rows × candidates)). For a faster estimate of full-scale numbers, extrapolate from 100 k: the pipeline-amortization effect is non-decreasing past ~100 k rows, so the 16.5 % advantage over parquet+zstd is approximately the floor.

## 1.5 Real-data — DoNext 5G/4G, vs csv.zst and the Avro+zstd anchor

The ClickBench numbers above are web-analytics data. This section uses real
**telecom Measurement Report** data — the workload helium's "Avro+zstd 5G MR
replacement" anchor targets. Dataset: **DoNext** (TU Dortmund,
doi:10.17877/TUDODATA-2026-T6MYPO, CC BY 4.0, ~4.5 GB, semicolon-CSV). The
`Avro+zstd` row models real 5G MR storage: Avro-serialize (null codec) then
compress the blob with zstd-3. Avro is produced with helium-inferred types
(apples-to-apples). H-Bahn `cell_data`, first 100 k rows × 34 cols:

| format | bytes | vs raw | vs csv.zst |
|---|---:|---:|---:|
| raw csv | 19.70 MB | 1.00× | — |
| csv.zst (-3) | 4.18 MB | 4.72× | 1.00× |
| avro (deflate) | 6.11 MB | 3.22× | 0.68× |
| avro (null) + zstd-3  ← 5G anchor | 5.07 MB | 3.89× | 0.82× |
| helium default | 5.19 MB | 3.79× | 0.80× |
| **helium optimized** | **2.94 MB** | **6.71×** | **1.42×** |

- **helium-optimized vs csv.zst: +30% smaller. vs the Avro+zstd anchor: +42% smaller.** vs Avro's native deflate: +52%.
- **helium-default loses to both csv.zst (0.80×) and Avro+zstd (1.02×)** — same story as ClickBench: the per-column framing overhead exceeds the default pipeline's savings; the optimizer (pcodec / delta / gorilla per column) is what wins. RSRP/RSRQ/SINR/timestamp columns are exactly its sweet spot.
- `neighboring_data` (19 cols) is "messier" (more low-cardinality/string): helium-optimized still wins csv.zst (+30%) and Avro+zstd (+30%), but the margin over csv.zst is narrower.

Reproduce (after downloading DoNext):
```bash
cargo build --release --features cli
scripts/donext_benchmark.sh /path/to/DoNext/H-Bahn/cell_data.csv 100000 ';'
```
The script prints this exact table (csv.zst / avro / helium) and is the
canonical way to re-run it. See `scripts/README.md`.

**The real value of testing on DoNext was robustness, not ratios**: real messy
input (non-UTF8 binary columns, European `;` delimiters, sparse nulls beyond
the inference sample window, Datetime columns) exposed and fixed **four
production bugs** the synthetic suite called green. Validate on real data
before claiming production-readiness.

## 2. SQL query latency — DataFusion via `helium sql`

Build a multi-stripe fixture once (4 stripes for a 1 M-row file):

```bash
cargo run --release --features cli -- \
  convert hits_1.parquet -o /tmp/hits_1_ms.he --stripe-rows 10000
# 100 stripes × 10k rows. ~31 s wall-clock; peak RSS ~322 MB
# (vs ~6.65 GB without --stripe-rows due to in-memory load).
```

All queries below run with:

```bash
cargo run --release --features cli,datafusion -- sql "<QUERY>" /tmp/hits_1_ms.he
```

Steady-state numbers (after one warmup run; cold-start adds ~1 s for Tokio runtime + file open):

| Query | Wall-clock | Cost driver |
|---|---:|---|
| `SELECT count(*) FROM hits_1_ms` | **28 ms** | metadata only — `Statistics::num_rows` constant-folded by DataFusion |
| `SELECT max(EventTime), min(EventTime) FROM hits_1_ms` | **31 ms** | metadata only — `ColumnStatistics::{min,max}_value` constant-folded |
| `SELECT WatchID, EventTime FROM hits_1_ms LIMIT 5` | 90 ms | reads only the projected columns × first stripe |
| `SELECT count(*) FROM hits_1_ms WHERE EventTime > 9999999999` | 30 ms | all 100 stripes pruned by min/max — body bytes never read |
| `SELECT count(*) FROM hits_1_ms WHERE EventTime > 1373893800` | 210 ms | matches ~22 % of rows; reads only EventTime per stripe (column pruning) |
| `SELECT count(*) FROM hits_1_ms WHERE RegionID = 9999999` | 360 ms | impossible value, all stripes pruned — but 100 partitions × Tokio dispatch ~3.5 ms ≈ floor |
| `SELECT count(*) FROM hits_1_ms WHERE RegionID = 229` | 6.2 s | matches ~20 % of rows; filter can't help — column scan dominates |

Takeaways:
- **`COUNT(*)` / `MIN` / `MAX` are O(1)** — they read footer bytes only. No data scan.
- **Range pushdown works** — `WHERE EventTime > X` stripes whose `[min, max]` doesn't intersect get skipped. All-pruned case is ~30 ms vs 210 ms for partial-prune.
- **Equality pushdown works for impossible values** via the per-stripe DistinctSet / Bloom filter (introduced in `e598624`); 360 ms is the 100-partition Tokio overhead floor, not actual data scan.
- **Filter that doesn't prune anything** (`RegionID = 229` matches values inside every stripe's range) falls back to full scan of the predicate column — ~6 s for 100 stripes × 10 k rows. Not currently parallel across stripes; partition-scheduling optimization is a future task.

## 3. Memory profile — `convert` with chunked I/O

```bash
# Streaming (--stripe-rows N triggers chunked load + write):
/usr/bin/time -l cargo run --release --features cli -- \
  convert hits_1.parquet -o /tmp/hits_1.he --stripe-rows 10000

# Look for "maximum resident set size" line.
```

| Mode | Peak RSS | Wall-clock |
|---|---:|---:|
| **`--stripe-rows 10000` (streaming)** | **322 MB** | 31 s |
| `--stripe-rows 0` (default; in-memory) | ~6.65 GB | OOMs on small machines |

**21 × memory reduction.** The remaining 322 MB is dominated by Parquet's row-API yielding strings for all 105 columns × 10 k rows simultaneously during chunk building. Sub-100 MB peaks would require a typed-array Parquet reader path — separate task.

This is the file size you can safely `helium convert` on a 16 GB laptop:
- With streaming (`--stripe-rows 10000`): no practical limit. Footer entries grow with stripe count, but bounded.
- Without streaming: ~2 GB Parquet input is the realistic ceiling.

## 4. Standalone codec API throughput

```rust
// Reproduce inline. Requires `--release` for representative numbers.
use helium::{ColumnData, compress, decompress};

let ts: Vec<i64> = (0..10_000)
    .map(|i| 1_700_000_000_000_000i64 + i * 10_000_000)
    .collect();
let raw_bytes = ts.len() * 8;                // 80,000 bytes
let encoded   = compress(ColumnData::I64(ts.clone())).unwrap();
let decoded   = decompress(&encoded).unwrap();

assert_eq!(decoded, ColumnData::I64(ts));
println!("{} → {}  ({:.0}× compression)",
         raw_bytes, encoded.len(),
         raw_bytes as f64 / encoded.len() as f64);
```

Representative numbers (Apple M, release):

| Input shape | Pipeline picked | Raw bytes | Compressed | Ratio |
|---|---|---:|---:|---:|
| 10 k uniform timestamps (i64) | delta_of_delta + leb128 + zstd | 80,000 | < 4,000 | > 20 × |
| 10 k jittered timestamps (i64) | delta + leb128 + zstd | 80,000 | ~12,000 | ~7 × |
| 10 k f64 sensor gauges | gorilla + zstd | 80,000 | ~600 | ~130 × |
| 10 k random u64 | delta + leb128 + zstd | 80,000 | ~80,000 | ~1 × (no leverage on random) |

Encode/decode throughput is **bounded by zstd in most cases** — typically 100-500 MB/s depending on column shape. Run `cargo bench` for criterion-statistical numbers per coder.

## 5. File-format overhead reference

For a 1 M-row × 105-column `.he` file with default writer settings:

| Component | Size | % of file | Notes |
|---|---:|---:|---|
| Schema header (zstd-compressed JSON) | 1.1 KB | < 0.01 % | `SCHEMA_ZSTD_LEVEL = 3` |
| Body (encoded column data) | ~93 MB | ~99.6 % | Per-column pipeline output |
| Footer (zstd-compressed JSON) | ~2.3 KB single-stripe / ~340 KB at 100 stripes | < 0.5 % | Per-stripe per-column index entries |
| Per-stripe min/max stats | included in footer | — | ~3 bytes per leaf after zstd |
| Per-stripe Bloom / DistinctSet | included in footer | — | ~10 % of file size on bloom-heavy schemas |

On `hits_1.he` with `--stripe-rows 10000`:
- `helium stats hits_1_ms.he --no-values` reports actual layout in ~400 ms (no body bytes read).
- `helium stats hits_1_ms.he` (with min/max scan) reports the same plus per-column min/max — uses footer stats first, falls back to scan if stats absent. ~1.5 s on 1 M rows.

## 6. Reproducing this document

Numbers in this file span `helium v0.2.0`–`v0.2.4`; the real-data DoNext
section (§1.5) is current as of v0.2.4. To rebuild the full picture:

```bash
# 0. Real-data DoNext benchmark (§1.5) — after downloading the dataset:
scripts/donext_benchmark.sh /path/to/DoNext/H-Bahn/cell_data.csv 100000 ';'

# 1. Compression numbers (writes target/format-comparison.md):
HELIUM_PARQUET_PATH=hits_1.parquet \
  cargo test --test format_comparison_report --release --all-features -- --nocapture

# 2. Build the multi-stripe fixture once:
cargo run --release --features cli -- convert hits_1.parquet \
  -o /tmp/hits_1_ms.he --stripe-rows 10000

# 3. SQL latencies (run each twice; second is steady-state):
for q in \
  'SELECT count(*) FROM hits_1_ms' \
  'SELECT max(EventTime), min(EventTime) FROM hits_1_ms' \
  'SELECT count(*) FROM hits_1_ms WHERE EventTime > 9999999999' \
  'SELECT count(*) FROM hits_1_ms WHERE EventTime > 1373893800' \
  'SELECT count(*) FROM hits_1_ms WHERE RegionID = 9999999' \
  'SELECT count(*) FROM hits_1_ms WHERE RegionID = 229'
do
  echo "=== $q ==="
  time cargo run --release --features cli,datafusion -- sql "$q" /tmp/hits_1_ms.he
done

# 4. Memory profile (streaming convert):
/usr/bin/time -l cargo run --release --features cli -- \
  convert hits_1.parquet -o /tmp/hits_1_streamed.he --stripe-rows 10000
# Linux: /usr/bin/time -v
```

## 7. Known limitations / where helium is slow

Honest accounting of where the current implementation underperforms:

- **`WHERE col = X` where X exists in many stripes**: column scan is per-partition serial inside DataFusion's `SerializedFileWriter`-style execution. ~6 s for 100 stripes × 10 k rows on a single column. Adding parallel partition execution would help; not yet done.
- **`COUNT(*) WHERE filter`**: even when all stripes are prunable, the 100-partition × Tokio dispatch adds up to ~350 ms floor. Below that requires a fast-path in `HeliumExec::execute` for "this partition is fully pruned, return PlaceholderRow with row_count=0".
- **Convert peak RSS still ~322 MB** on 1 M rows × 105 cols: the chunk's intermediate string representation (Parquet row API) is the bulk. A typed-array Parquet reader integration would drop this another 5-10 ×.
- **All coders are scalar.** TurboPFor-style SIMD bitpacking is documented in `../helium-turbopfor-amendment.md` but not implemented. For column-encode-throughput-bound workloads, expect 2-5 × headroom unrealized.
- **No external storage backend.** Only local filesystem. Adding `object_store` integration (S3 / GCS) is a known follow-up.

For each, there's a tracking note in `updates.md` or the source code's TODO comments. None are currently a blocker for the demonstrated use cases (analytical query workloads on fits-in-memory-or-streaming Parquet conversion).
