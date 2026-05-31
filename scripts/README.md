# scripts/

Reproducible benchmark tooling. Not part of the published crate (excluded
from `cargo publish`).

## `fetch-fixtures.sh` — download the optional test fixtures

None of the large fixtures are committed (they're gitignored). This script
fetches the **ClickBench** `hits_1.parquet` partition (public, ~1 M rows ×
105 cols) and converts it to `hits_1.he`, so you can run the Arrow examples
and the `HELIUM_PARQUET_PATH`-gated tests/reports against real data.

```bash
cargo build --release --features cli   # needed for the .he conversion
scripts/fetch-fixtures.sh
# → hits_1.parquet + hits_1.he at the crate root
HELIUM_PARQUET_PATH=hits_1.parquet cargo test --release -- --nocapture
```

Source URL (public):
`https://datasets.clickhouse.com/hits_compatible/athena_partitioned/hits_1.parquet`

The **DoNext** 5G/4G real-data set is *not* fetched by this script (~4.5 GB,
CC BY 4.0) — see `donext_benchmark.sh` below and the DOI.

## `donext_benchmark.sh` — real-data compression benchmark

Runs an apples-to-apples compression comparison on a real telecom CSV:
`csv.zst` vs `helium default` vs `helium optimized` vs `avro+zstd` /
`avro+deflate`. Built around the **DoNext** open 5G/4G measurement dataset.

### Easiest: `--fetch` (auto-download just the file it needs)

The full DoNext set is ~4.6 GB, but the benchmark only needs one file —
`H-Bahn/cell_data.csv` (~100 MB, the serving-cell Measurement-Report
workload). `--fetch` downloads exactly that file from TU Dortmund's
Dataverse (cached under `target/donext/`, gitignored) and runs the
benchmark on it:

```bash
cargo build --release --features cli      # one-time
scripts/donext_benchmark.sh --fetch        # downloads + benchmarks 100k rows
scripts/donext_benchmark.sh --fetch 20000  # fewer rows = faster
```

### Or point it at your own DoNext download

The dataset is CC BY 4.0: <https://doi.org/10.17877/TUDODATA-2026-T6MYPO>.
It is semicolon-separated CSV, split into `H-Bahn/`, `Mobile/`, `static/`
× `{cell,neighboring,latency,datarate,iperf}_data.csv`. The `cell_data.csv`
files are serving-cell Measurement Reports (RSRP / RSRQ / SINR / timestamp /
GNSS — the MR-shaped workload); `neighboring_data.csv` is the neighbor table.

```bash
cargo build --release --features cli
scripts/donext_benchmark.sh /path/to/DoNext/H-Bahn/cell_data.csv 100000 ';'
```

Args: `--fetch [max_rows]`, or `<csv> [max_rows] [delimiter]` (delimiter
defaults to `;`; max_rows defaults to 100000 with `--fetch`, whole file
otherwise — 100k keeps `optimize-schema` fast).

The `avro+zstd` / `avro+deflate` rows model how 5G Measurement Reports are
stored in production (Avro serialization, then the blob compressed). They
require `python3` + `fastavro` (`pip install fastavro`); if absent, the
script prints the other rows and notes the Avro rows were skipped.

### Sample output (DoNext H-Bahn cell_data, 100k rows × 34 cols)

```
format                          bytes     vs raw   vs csv.zst
raw csv                      19697621      1.00x            -
csv.zst (-3)                  4177224      4.72x        1.00x
avro (deflate)                6109903      3.22x        0.68x
avro (null)+zstd              5065726      3.89x        0.82x
helium default                5192497      3.79x        0.80x
helium optimized              2937033      6.71x        1.42x

helium-optimized vs csv.zst:       +30%
helium-optimized vs avro+zstd:     +42%
round-trip verify (opt.he):        OK
```

Read: helium **default** loses to plain `csv.zst` (the per-column framing
overhead exceeds the pipeline savings) — the **optimized** schema is the
differentiator, beating both `csv.zst` (+30%) and the `avro+zstd` 5G
storage anchor (+42%) on this real telemetry data. See
[`../docs/PERFORMANCE.md`](../docs/PERFORMANCE.md) for the broader picture.

## `donext_benchmark_all.sh` — comprehensive (all categories × types)

Runs `donext_benchmark.sh` over **every** DoNext data CSV (H-Bahn / Mobile /
static × cell / latency / iperf / datarate / neighboring) and prints one
combined table — so you can see how helium fares across the whole dataset,
not just `cell_data`.

```bash
cargo build --release --features cli

# Point it at a local DoNext download (the dir with H-Bahn/, Mobile/, static/):
scripts/donext_benchmark_all.sh /path/to/DoNext 100000

# …or let it download every *_data.csv (~4.8 GB) from Dataverse first:
scripts/donext_benchmark_all.sh --fetch-all 100000
```

Args: `<dataset_dir> [max_rows]` or `--fetch-all [max_rows]` (default 100000
rows per file). The combined table prints to stdout and is saved to
`target/donext-benchmark-all.md`. Per-file the takeaway is consistent: helium
**optimized** beats the `avro+zstd` 5G-storage anchor on every file type, and
beats `csv.zst` once there are enough rows to amortize per-column framing
(≥ ~100k; at small caps the lighter formats can win — see *When NOT to use
Helium* in the README).

## `donext_format_comparison.py` — helium vs parquet vs orc vs zstd

Writes the first N rows of every DoNext CSV as Helium (optimizer-chosen
encodings), Parquet (pyarrow, zstd level 3), ORC (pyarrow, zstd), and `csv.zst`
(raw zstd-3 baseline), all in a single row-group/stripe so the stripe size is
identical across the three columnar formats. Prints a per-file table of
compression ratios and Helium's size advantage vs Parquet/ORC.

```bash
cargo build --release --features cli
pip install pyarrow            # ORC writing needs pyarrow (no Rust ORC writer)
python3 scripts/donext_format_comparison.py /path/to/DoNext 100000
```

Args: `<dataset_dir> [n_rows=100000]`. Both Parquet and Helium are pinned to
zstd level 3; pyarrow's ORC writer does not expose a zstd level, so the ORC
column uses Apache ORC's default level (the Parquet comparison is the exact one).
On the 12 DoNext files (100k rows): Helium is smaller than Parquet on 11/12
(median ~13%, up to 58%) and smaller than ORC on 10/12 (median ~9%).

## `csv_to_avro.py`

Helper used by the benchmark to encode a CSV as Avro, mirroring a
helium-inferred schema (so the comparison uses identical logical types).
Requires `fastavro`. Standalone usage:

```bash
python3 scripts/csv_to_avro.py <input.csv> <helium_schema.json> <out_prefix> [delimiter]
```
