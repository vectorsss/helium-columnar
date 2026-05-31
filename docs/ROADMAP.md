# Roadmap

Forward-looking plans for Helium beyond the **v0.2.0** release. None of these
block the current use cases (analytical queries over fits-in-memory or
streaming-converted columnar data); they are where the largest headroom is.
This document is self-contained.

## Encoding throughput — SIMD integer compression

Every coder today is **scalar**. The biggest unrealized win is SIMD
bit-packing and the TurboPFor algorithm family (group-varint / "TurboByte",
SIMD frame-of-reference + bit-packing, patched-FOR). Reference numbers put
SIMD bit-pack decode around 10 G integers/s (~40 GB/s) versus the scalar
path's hundreds of MB/s — a 1–2 order-of-magnitude gap for
encode/decode-throughput-bound workloads.

Two design implications:

- **A third coder category.** The current block / non-block split doesn't model
  fixed-size SIMD batches well. A `MicroBlockCoder` operating on fixed batches
  (e.g. 128 / 256 values) would sit between `NonBlockCoder` (per-element) and
  `BlockCoder` (whole-buffer), with pipelines composing
  per-element → micro-block → block.
- **License constraint (important).** Helium is MIT-licensed. The canonical
  TurboPFor implementation is **not** permissively licensed, so it cannot be
  vendored. The algorithms themselves aren't copyrightable, so the realistic
  path is a permissive implementation or a clean-room one. Candidates:
  **FastLanes** (MIT, CWI), **simdcomp** / **FastPFor** / **StreamVByte**
  (Apache-2.0, D. Lemire). Plan: prototype against FastLanes/simdcomp and adopt
  only if it reaches a large fraction of TurboPFor's throughput.

## Random access as a first-class coder property

Point lookups within a stripe currently require decoding the whole physical
column. Exposing an access pattern per coder (sequential vs randomly
addressable) — where a pipeline's capability is that of its weakest stage —
would let selective queries (`WHERE pk = X`) decode only the needed values.
This pairs naturally with the SIMD/MicroBlock work, since fixed batches are
randomly addressable.

## Query engine

- **Parallel partition execution.** A non-pruning `WHERE col = X` that matches
  values in every stripe currently scans the predicate column serially across
  partitions (~6 s for 100 stripes × 10 k rows). Parallelizing partition
  execution in `HeliumExec` is the single biggest query-latency win.
- **Fully-pruned fast path.** Even when every stripe is pruned, ~350 ms of
  Tokio per-partition dispatch remains; a "partition fully pruned → empty
  result" short-circuit would remove that floor.

## Bindings (pyhelium / helium-duckdb)

The two language bindings live in `python/` and `duckdb/` as independent Cargo
projects (path-deps on `helium-columnar`, not published, gated by the `bindings`
CI job). **helium-duckdb** can *read* the full recursive type set but cannot write or
push down — that is its decisive next step. **pyhelium** has gained Arrow /
pandas interop: `write_table` / `read_table` round-trip nullable, nested
(Struct / List), and semantic (Date / Datetime / Decimal) columns, with
optimizer-chosen encodings and streaming / projected reads.

### helium-duckdb — pushdown status

The extension used to read *every column and every row* of an `.he` file. Most
of that gap is now closed; what landed and what remains:

- **Projection pushdown — done.** The extension advertises projection pushdown,
  reads the projected column indices in `init`, and decodes only those columns
  per stripe via the reader's column pruning. Selecting 1 of N columns decodes
  1, not N (verified: decoding 4 wide columns scales ~linearly over 1; see
  `duckdb/smoke.sh`).
- **One reader held open — done.** The reader is opened once per scan and held
  across stripes (in the init data), instead of re-opening the file per stripe.
- **Nested types — done.** `Struct`, `List`, and `Map` map onto DuckDB
  STRUCT / LIST / MAP vectors. `Union` and the legacy flat `ArrayOf` / `ArrayOfUtf8`
  legacy variants still error at bind time with a clear message.
- **Catalog-mode support — done.** `read_he(path, catalog := '…')` resolves the
  schema through `HeliumReader::new_with_resolver`; a catalog-mode file without
  the parameter errors with the documented resolver message.
- **Correctness — done.** `smoke.sh` covers nullable + multi-stripe: a
  `Nullable<I64>` column whose nulls straddle stripe (5000) and chunk (2048)
  boundaries, verified by both aggregate sums and boundary spot-checks. This
  exercises the absolute-row indexing of the compacted inner values.
- **Predicate pushdown + stripe pruning — partial (ABI-blocked).** The
  stripe-pruning logic for scalar comparisons is implemented and unit-tested
  (`duckdb/src/prune.rs`), but the DuckDB *loadable* extension C-API exposes
  projection pushdown and **no** filter-pushdown hook — DuckDB never hands a
  loadable extension the `WHERE` bounds. So the machinery is ready but cannot be
  auto-driven. This is a known upstream gap stuck on an unresolved filter-API
  design (two C-API filter PRs, [duckdb#14591] / [duckdb#19093], were closed
  unmerged); reaching it via a native C++ shim would trade away the portable
  cross-version loadable extension. The full analysis — why the C API omits it,
  the C++/ABI tradeoff, and the manual-range stopgap — is in
  [`duckdb/README.md`](../duckdb/README.md) → *A note on predicate pushdown*.
  Until the C-API catches up, DuckDB applies `WHERE` after the scan.

  [duckdb#14591]: https://github.com/duckdb/duckdb/pull/14591
  [duckdb#19093]: https://github.com/duckdb/duckdb/pull/19093
- **Distribution — scaffolded.** `duckdb/packaging/package.sh` builds + packages
  one platform; `duckdb/packaging/matrix.md` documents the platform tuples and
  the C-API/`Cargo.lock` ABI coupling; `duckdb/ci/extension-matrix.yml` is a
  ready-to-merge per-platform build+load CI job; and
  `duckdb/packaging/community-extension-submission.md` documents the manual
  community-extensions submission (including the one prerequisite: swap the
  `path = ".."` core dependency for a pinned git/crates.io ref). No external
  submission has been made.

### pyhelium — Arrow / pandas interop

**Done.** `write_table(table_or_df)` / `read_table() -> pyarrow.Table` reuse
Helium's `arrow` bridge over the Arrow C Data Interface, lifting the old
flat-only limit: nullable, nested (Struct / List / Map), and semantic
(Date / Datetime / Decimal) columns round-trip. The original numpy API
(`compress` / `decompress`, `write_he` / `read_he`) is unchanged. Also landed:

- **Encoding control.** `write_table` runs Helium's optimizer by default
  (smallest measured pipeline per column); `optimize=False` selects fast
  defaults.
- **Streaming + projection.** `write_table(..., stripe_rows=N)` writes in
  bounded-memory stripes; `read_table(..., columns=[...], stripe_range=(a, b))`
  decodes only the requested columns / stripes.
- **Packaging.** abi3 wheels (`cp39-abi3`, one per platform) + a `cibuildwheel`
  matrix and a `pyhelium-wheels.yml` workflow; PyPI publishing is prepared and
  documented in `python/PACKAGING.md` but left disabled until the project is
  registered.

Remaining:

- **Dictionary-encoding control** — exposing dict encoding as a Python option.
- Turning on the **PyPI publish** job once the package is registered.

### Shared

- **Benchmarks.** Neither binding has throughput/latency numbers. The DuckDB
  numbers are only meaningful *after* pushdown lands, so sequence it that way.
- **CI depth.** Upgrade the duckdb compile gate to a real load+query smoke over
  a DuckDB-version matrix. The pyhelium wheel-build matrix now exists
  (`pyhelium-wheels.yml`, opt-in on demand / on a `pyhelium-v*` tag).

## Memory

- **Typed-array Parquet reader.** `convert` peaks at ~322 MB on 1 M × 105
  columns because the Parquet row API yields strings for all columns of a chunk
  at once. A typed (column-array) Parquet read path would cut this another
  5–10×.

## Type system

- **`f16` (half-precision float).** `pco` supports it and it is common in ML /
  sensor data, but Helium has no `F16` logical/physical type yet. Adding the
  type (plus `gorilla` / `pcodec` support) is a self-contained extension.
- Decimal128 / Date / Datetime semantic types already ship in v0.1.0.

## Storage backends

- **`object_store` integration** (S3 / GCS / Azure) so `.he` files can be read
  and written directly against cloud object storage, not just the local
  filesystem.

## Format / catalog

- The shared-schema catalog is filesystem-backed. A networked / shared
  catalog service is a possible direction for multi-writer deployments.

## StarRocks integration (external)

The longer-term aim is to bring Helium's encoding wins into StarRocks' storage
engine. `tests/starrocks_report.rs` is the offline evidence: it scores Helium's
best pipeline per column against StarRocks' *best-achievable* config
(BIT_SHUFFLE/DICT/RLE **+ ZSTD**, not its weaker LZ4 default) on two thresholds
— compression-ratio gain ≥ 15% **and** decode throughput ≥ 70% of baseline.

What the report says:

- **The win is concentrated on float/double and near-regular timestamp
  columns** — exactly where StarRocks' BIT_SHUFFLE is weakest. On synthetic
  time-series data, pcodec compresses ~2–8× better than the StarRocks baseline
  **and decodes faster than it** (so the usual "better ratio, worse scan"
  tradeoff does not apply here). ClickBench has no float columns, which is why
  that win only shows up in the synthetic rows.
- The realistic path is to **contribute one or two targeted page builders** (a
  pcodec float/double builder, later a delta+pcodec timestamp builder), rather
  than porting Helium's whole pipeline abstraction into StarRocks. A full
  pipeline rewrite would fight StarRocks' vectorisation / zero-copy /
  dictionary-in-engine investments for little additional payoff.

### Prototype status — validated end-to-end

A working prototype integration has confirmed this thesis inside a live engine
(not yet upstreamed):

- **pcodec links into the C++ backend as a static library.** The build-system
  question — whether the engine's third-party build can link a Rust-produced
  static library plus a generated C header — is **resolved**: it does. pcodec's
  C interface already exposes the caller-allocates primitives a page builder
  needs; the two gaps were closed in a fork and are the upstream contributions
  to make — a `staticlib` crate-type, and routing `decompress_into` through the
  core's zero-copy path. The version is pinned, the same posture the engine
  takes with zstd / lz4 / bitshuffle.
- **A pcodec float/double page builder + decoder is wired into the segment
  format** under a stable encoding id, as a peer of BIT_SHUFFLE: chosen at write
  time, decoded transparently on read.
- **The encoding is user-selectable per column via DDL** — a table property on
  `CREATE TABLE`, and on `ALTER TABLE` (which re-encodes existing data through a
  schema-change rewrite; an empty value resets a column to the default).
  Validation rejects the codec on non-float columns, and the property
  round-trips through `SHOW CREATE TABLE`.
- **Measured in the live engine:** a pcodec-encoded `DOUBLE` column was several×
  smaller on disk than the BIT_SHUFFLE default — matching the offline report —
  with scan / aggregation latency within noise of the default, so the "better
  ratio, free scan" property held in practice.

### What remains

- **Upstreaming.** The main friction for a merge is the Rust static-library
  build dependency; the page builder, the per-column DDL, and the two pcodec C
  contributions are the pieces to land. Any other BIT_SHUFFLE-style float store
  is a candidate for the same targeted-builder approach.
- **Broader column coverage.** The prototype targets float/double, where the win
  is largest. A delta + pcodec timestamp builder is the natural next column.
- **Sub-page random access.** Selective scans want sub-page seek; this depends
  on the *Random access as a first-class coder property* item above.
- **Fallback if linking Rust proves untenable upstream:** `gorilla` /
  `delta-of-delta` reimplement trivially in C++ and still beat BIT_SHUFFLE on
  floats / timestamps (lower ratio than pcodec, zero external dependency).
