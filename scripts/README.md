# scripts/

Reproducible benchmark tooling. Not part of the published crate (excluded
from `cargo publish`).

## `donext_benchmark.sh` — real-data compression benchmark

Runs an apples-to-apples compression comparison on a real telecom CSV:
`csv.zst` vs `helium default` vs `helium optimized` vs `avro+zstd` /
`avro+deflate`. Built around the **DoNext** open 5G/4G measurement dataset.

### Get the data

DoNext is **not** bundled (CC BY 4.0, ~4.5 GB). Download from TU Dortmund:

> https://doi.org/10.17877/TUDODATA-2026-T6MYPO

It is semicolon-separated CSV, split into `H-Bahn/`, `Mobile/`, `static/`
× `{cell,neighboring,latency,datarate,iperf}_data.csv`. The
`cell_data.csv` files are serving-cell Measurement Reports (RSRP / RSRQ /
SINR / timestamp / GNSS — the MR-shaped workload); `neighboring_data.csv`
is the neighbor-cell table.

### Run

```bash
# Build the CLI first.
cargo build --release --features cli

# Benchmark the first 100k rows of a DoNext cell-data file (recommended:
# 100k keeps optimize-schema fast; omit the row cap to use the whole file).
scripts/donext_benchmark.sh /path/to/DoNext/H-Bahn/cell_data.csv 100000 ';'
```

Args: `<csv> [max_rows] [delimiter]` (delimiter defaults to `;` for DoNext).

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

## `csv_to_avro.py`

Helper used by the benchmark to encode a CSV as Avro, mirroring a
helium-inferred schema (so the comparison uses identical logical types).
Requires `fastavro`. Standalone usage:

```bash
python3 scripts/csv_to_avro.py <input.csv> <helium_schema.json> <out_prefix> [delimiter]
```
