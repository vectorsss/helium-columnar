# Roadmap

Forward-looking plans for Helium beyond the **v0.1.0** release. None of these
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
CI job). They are deliberately asymmetric today: **pyhelium** can *write* but
only flat numeric/string/binary columns; **helium-duckdb** can *read* the full
v3 type set but cannot write or push down. Each has one decisive next step.

### helium-duckdb — close the pushdown gap

The extension currently reads *every column and every row* of an `.he` file and
hands them to DuckDB to filter. That switches off Helium's core advantage
(columnar pruning + per-stripe min/max), so the table function on a `.he` file
is no faster — often slower — than reading the equivalent Parquet. Priorities:

- **Projection pushdown** *(highest leverage, smallest change)*. DuckDB's
  `init` phase exposes the projected column indices; read only those columns via
  the reader's existing column pruning. Selecting 1 of 100 columns should decode
  1, not 100.
- **Predicate pushdown + stripe pruning.** Carry `WHERE` bounds into bind/init
  through DuckDB's filter-pushdown hooks and skip stripes whose footer min/max
  cannot match. This is the binding-side companion to *Random access* above.
- **Nested types.** Map `Struct`/`List`/`Map` onto DuckDB's STRUCT/LIST/MAP
  vectors (they error today) for full v3 fidelity.
- **v6 catalog support.** A `read_he(path, catalog := '…')` parameter wired to
  `HeliumReader::new_with_resolver` (v6 files error today).
- **Hold one reader open across stripes** instead of re-opening the file per
  stripe in the scan callback.
- **Correctness.** Add nullable + multi-stripe coverage to `smoke.sh`: the
  `NullablePrim` write path indexes the (compacted) values array by absolute row
  and needs verification across chunk/stripe boundaries.
- **Distribution.** A DuckDB-version build matrix and community-extensions
  submission. The pinned `duckdb/Cargo.lock` exists because the loadable ABI is
  coupled to a specific DuckDB version — that coupling must be made explicit.

### pyhelium — Arrow / pandas interop

The binding only moves numeric `ndarray`s and `list[str]`/`list[bytes]` today,
with hardcoded pipelines. The decisive step is **Arrow interop**
(`read_he() -> pyarrow.Table`, `write_he(df)`): reusing Helium's `arrow` bridge
lifts the flat-only limitation in one move, bringing nullable, nested, and
semantic (Date/Datetime/Decimal) columns along for free. Then:

- **Encoding control.** Expose the optimizer / coder specs so Python users get
  Helium's actual compression wins instead of fixed defaults.
- **Streaming + projection.** Chunked (multi-stripe) writes and projected,
  by-stripe reads for bounded memory on large files (`read_he` is whole-file,
  in-memory today).
- **Packaging.** abi3 wheels + a `cibuildwheel` matrix + PyPI publishing
  (currently source-only via `maturin`).

### Shared

- **Benchmarks.** Neither binding has throughput/latency numbers. The DuckDB
  numbers are only meaningful *after* pushdown lands, so sequence it that way.
- **CI depth.** Upgrade the duckdb compile gate to a real load+query smoke over
  a DuckDB-version matrix; add a pyhelium wheel-build matrix.

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

- The v6 shared-schema catalog is filesystem-backed. A networked / shared
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
- The realistic path is therefore to **contribute one or two targeted page
  builders** (a pcodec float/double builder, a delta+pcodec timestamp builder),
  rather than porting Helium's whole pipeline abstraction into StarRocks. A full
  pipeline rewrite would fight StarRocks' vectorisation / zero-copy /
  dictionary-in-engine investments for little additional payoff on current
  evidence.

### Bringing pcodec into the BE — the FFI plan

pcodec already ships a C interface with exactly the caller-allocates primitives
a page builder needs — `guarantee_file_size`, `simple_compress_into`, and
`simple_decompress_into` (caller-owned buffers, stateless, thread-safe, no Rust
heap ownership crossing the boundary). So linking pcodec into StarRocks' C++ BE
does **not** require writing our own bindings from scratch. The plan is to use
that interface and contribute the couple of improvements it needs upstream:

- It is an early-stage interface, so **pin a known-good version** — that gives
  us stability regardless of upstream churn, the same posture StarRocks already
  takes with zstd/lz4/bitshuffle.
- Two small upstream contributions cover the gaps: a **`staticlib`** crate-type
  (to link into the C++ BE; it is `dylib`-only today), and making its
  `decompress_into` call the core's zero-copy `simple_decompress_into` instead
  of allocating + copying.
- Build it as a staticlib + a generated header + a thin C++ RAII wrapper; one
  pcodec standalone buffer per StarRocks page (the paging spec maps onto the
  ~1 MB page model).

Remaining open items before a C++ prototype is worthwhile:

- **Build-system integration (the one genuine unknown).** Whether StarRocks'
  BE third-party build can link a Rust-produced staticlib + generated header.
  This is a build-system task, not an FFI-feasibility question — validate it
  before committing to the prototype.
- **Zero-copy decode is available** in pcodec's core (`simple_decompress_into`
  decodes straight into engine-owned memory), so no new algorithm work is
  needed for the common path.
- **Fallback if linking Rust proves untenable:** `gorilla` / `delta-of-delta`
  reimplement trivially in C++ and still beat BIT_SHUFFLE on floats/timestamps
  (lower ratio than pcodec, zero external dependency).
- **Random access.** Selective scans want sub-page seek; this depends on the
  *Random access as a first-class coder property* item above.
