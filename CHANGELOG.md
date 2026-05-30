# Changelog

All notable changes to this project are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/); the project follows
[Semantic Versioning](https://semver.org/) with the usual pre-1.0 caveat —
**0.x minor releases may include breaking API or on-disk format changes.**

## [Unreleased]

### Changed
- `helium optimize-schema` now measures encodings on a **sampled prefix** by
  default (`--sample-rows 200000`); pass `--sample-rows 0` to use the whole
  file. Sampling reads only the prefix, so picking a schema for a large file no
  longer scans it end-to-end. The emitted schema still applies to the full
  dataset via `convert --schema`.
- `pcodec` now accepts **all integer widths** (`i8`–`i64`, `u8`–`u64`) plus
  `f32`/`f64` (was `i32`/`i64`/`u32`/`u64`/`f32`/`f64`). 8-bit types are enabled
  via pco's opt-in. (`f16` still pending — Helium has no `F16` type; see ROADMAP.)

### Added
- `docs/ROADMAP.md` — forward-looking plan (SIMD coders, parallel query
  execution, `f16`, object-store backends).

## [0.1.0] — 2026-05-30

Initial public release.

### Compression

- Typed coder pipeline with a block / non-block split and statically validated
  stage ordering.
- Reference coders: `delta`, `delta_of_delta`, `leb128`, `rle`, `deltamin`,
  `bitpack_fixed`, `bitpack_auto`, `gorilla` (float XOR), `elias_fano`, `zstd`,
  `lz4`, `snappy`, `pcodec`.
- Standard self-describing codec API — `helium::compress` / `helium::decompress`.

### Schema & types

- Logical/physical schema split; JSON-serializable schemas embedded in the file.
- Recursive logical types — `Struct`, `List`, `Map`, `Nullable`, `Union`,
  `Dictionary` — plus semantic `Decimal128`, `Date`, and `Datetime`.
- Source-format adapters: Avro, CSV, JSON, Parquet → `Schema`.
- Measured-optimal encoding picker (optimizer).

### File format (`.he`)

- Versioned container: **v5** (self-contained, default) and **v6** (opt-in
  catalog mode). Schema header and footer are zstd-compressed; per-column and
  footer CRC32C integrity checks.
- Multi-stripe (row groups) with per-stripe min/max, null counts, and
  containment (DistinctSet / Bloom) filters for predicate pushdown.
- Column pruning on read and **zero-copy column projection** ("slice").
- Opt-in shared-schema catalog (hash-referenced v6 files).

### Query

- Arrow `ArrayRef` ↔ `LogicalColumn` bridge.
- DataFusion `TableProvider` — SQL over `.he` files with projection / filter
  pushdown and metadata-only `count(*)` / `min` / `max`.

### Tooling & quality

- `helium` CLI: `convert`, `verify`, `stats`, `slice`, `sql`, `catalog`,
  `infer-schema`, `optimize-schema`.
- Property/fuzz round-trip tests (arbtest), criterion benchmarks, and
  per-coder compression reports.
- Library + binary deny `unwrap`/`expect`/`panic`; `cargo fmt` / `clippy`
  clean.

[0.1.0]: https://github.com/vectorsss/helium-columnar/releases/tag/v0.1.0
