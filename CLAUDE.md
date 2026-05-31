# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

The repo-root `../CLAUDE.md` covers the cross-directory layout (design doc, vendored StarRocks, ClickBench parquets). This file is scoped to development *inside* `helium-core/` and supplements (and in places corrects) the parent file.

## Crate layout — single crate, internal modules

Until 2026-04-30 this directory was a Cargo workspace with 4 sibling crates (`helium-schema`, `helium-optimizer`, `helium-cli`, `helium-catalog`). It is now a **single crate `helium`** with internal modules. The directory is still named `helium-core/` (rename to `helium/` is a future task).

```
helium-core/                ← repo root (single crate, name = "helium")
├── Cargo.toml              ← single [package], no [workspace]
├── src/
│   ├── lib.rs              ← public API; declares pub mod core/coders/schema/optimizer/catalog + re-exports
│   ├── main.rs             ← binary; declares its own `mod cli;` (binary-only tree, separate from lib)
│   ├── core/               ← framework: pipeline / coder traits / file format / Schema type / registry / canonicalize / error
│   ├── coders/             ← concrete coder implementations
│   ├── schema/             ← format adapters (Avro / CSV / JSON / Parquet → Schema)
│   ├── optimizer/          ← measured-optimal encoding picker
│   ├── catalog/            ← opt-in shared-schema catalog
│   └── cli/                ← subcommand handlers (only compiled into the binary)
├── tests/                  ← all integration tests (was: per-crate tests merged)
├── benches/
└── examples/
```

**Naming gotcha — two `schema`s**: `crate::core::schema::Schema` (the type) lives under `core/`; `crate::schema::*` (Avro/CSV/JSON/Parquet adapters) lives at the top level. Inside `src/schema/*.rs`, use `crate::core::Schema` to refer to the type — *not* `crate::schema::Schema` (different module).

**Public API surface** (`lib.rs`): re-exports the core types so `helium::Schema`, `helium::HeliumWriter`, `helium::Pipeline`, `helium::CoderRegistry`, etc., are addressable at crate root. Adapters keep their full path: `helium::schema::avsc_to_schema`, `helium::optimizer::Optimizer`, `helium::catalog::Catalog`.

**`cli/` is binary-only**: `lib.rs` does **not** declare `mod cli;`. `main.rs` is the binary's crate root and declares its own `mod cli;` referring to `src/cli/mod.rs`. Library users never see `cli::*` regardless of which features they enable.

## Cargo features

```
default        = ["schema-avro"]
cli            = clap + anyhow + all schema-* adapters     ← required by [[bin]] helium
schema-avro    = pulls `apache-avro` (data container reader/writer)
schema-csv     = pulls `csv`
schema-json    = (no extra deps)
schema-parquet = pulls `parquet`
arrow          = pulls `arrow` (LogicalColumn ↔ ArrayRef bridge)
datafusion     = pulls `datafusion` (SQL on .he files via TableProvider; transitively enables `arrow`)
```

The binary `helium` declares `required-features = ["cli"]`, so plain `cargo build` (no features) compiles only the library and skips the binary. To build/run the CLI: `cargo build --features cli` or `cargo run --features cli -- <subcommand>`.

## Common commands

```bash
cargo test --all-features                                 # full suite (was: cargo test --workspace)
cargo test --no-default-features                          # smoke check feature gating is sound
cargo test --test roundtrip                               # one integration binary
cargo test --test roundtrip -- name_filter                # one test within a binary
cargo clippy --all-targets --all-features -- -D warnings  # lint everything

# CLI
cargo run --release --features cli -- convert events.csv -o events.he                        # external → helium
cargo run --release --features cli -- convert events.he -o events.avro                       # helium → external (avro / csv / json / parquet)
cargo run --release --features cli -- convert events.json -o events.he                       # NDJSON nested data fully supported
cargo run --release --features cli -- convert in.txt --from csv -o out.he                    # --from/--to override extension
cargo run --release --features cli -- convert euro.csv -o euro.he --delimiter ';'            # non-comma CSV (European-style)
cargo run --release --features cli -- convert events.csv -o events.he --stripe-rows 10000    # multi-stripe + streaming (bounded memory)
cargo run --release --features cli -- convert events.csv -o events.he --catalog ./catalog    # write catalog mode
cargo run --release --features cli -- verify file.he                                          # self-contained
cargo run --release --features cli -- verify file.he --catalog ./catalog                     # catalog mode
cargo run --release --features cli -- slice events.he -o subset.he --columns ts,user_id,label # column projection → new .he
cargo run --release --features cli -- catalog list ./catalog                                  # list registered schema hashes
cargo run --release --features cli -- catalog verify ./catalog                                # consistency check
cargo run --release --features cli -- optimize-schema events.csv --out events.schema.json
cargo run --release --features cli -- stats events.he                                         # per-column bytes + min/max
cargo run --release --features cli -- stats events.he --no-values                             # bytes only, faster
cargo run --release --features cli -- stats events.he --json                                  # machine-readable

# SQL on .he files (requires --features datafusion in addition to cli)
cargo run --release --features cli,datafusion -- sql "SELECT count(*) FROM events" events.he
cargo run --release --features cli,datafusion -- sql "SELECT a.id, b.label FROM a JOIN b ON a.id = b.id" a.he b.he
cargo run --release --features cli,datafusion -- sql "SELECT * FROM x LIMIT 5" x=jan-2026.he   # rename via name=path

# Benchmarks (criterion)
cargo bench
cargo bench -- --quick --warm-up-time 1 --measurement-time 2
cargo bench --bench pipelines -- timestamps

# Reports — print markdown tables, persist to target/
cargo test --test compression_report       --release -- --nocapture   # per-coder ratios vs gzip/lz4/zstd/pcodec
cargo test --test starrocks_report         --release -- --nocapture   # ratios vs StarRocks
cargo test --test format_comparison_report --release -- --nocapture   # whole-file sizes vs parquet/avro/csv/json (incl. catalog mode)

# Real ClickBench data (writer/reader/converter tests fall back to synthetic if unset)
HELIUM_PARQUET_PATH=../parquets/hits_1.parquet cargo test --release -- --nocapture
```

Rust edition is **2024**. Acceptance bar: `cargo test --all-features` green AND `cargo test --no-default-features` compiles AND `cargo clippy --all-targets --all-features -- -D warnings` clean.

## Architecture notes / corrections vs. the parent CLAUDE.md

These supersede the parent file where they conflict.

### File format: stable header, two storage modes

- Every file starts with an 8-byte header: the stable 6-byte magic `b"HELIUM"`, a 1-byte format version (`FORMAT_VERSION = 1`), and a 1-byte flags byte. The reader checks the magic, then the version, then rejects any unknown *incompatible* flag bits (compatible high-nibble flags are ignored if unknown). The storage mode is selected by flag bit 0.
- **Self-contained mode** (`flags = 0x00`, the default) — `HeliumWriter::new` embeds the schema JSON in the header. Both the schema JSON and the footer JSON are zstd-compressed (~92% savings on wide schemas). The on-disk `footer_crc32c` is over the *compressed* bytes — corruption is caught before decompression. Read with `HeliumReader::new`.
- **Catalog mode** (`flags` bit 0 set → `0x01`) — `HeliumWriter::with_catalog_ref` opts in: the header carries a 32-byte BLAKE3 of the canonicalized schema JSON plus a CRC32C of that hash instead of an embedded schema, with a compressed footer. Readers use `HeliumReader::new_with_resolver` to fetch the schema by hash. See `src/catalog/`. Catalog mode is **never** the default — writers must supply an explicit catalog handle.
- The error message when a catalog-mode file is opened without a resolver reads "catalog-mode file requires a schema resolver but none was provided".
- Schema canonicalization (used by catalog-mode hashing) lives in `src/core/canonicalize.rs` and uses `unicode-normalization` for NFC. Bumping that dep major would risk hash drift on non-ASCII field names — treat it as frozen.

### Where the plan lives

`PLAN_V2.md` is the active plan; the original `PLAN.md` is the historical record. Both currently sit in `plan_backup/` (untracked at time of writing). The parent CLAUDE.md's "consult `helium-core/PLAN.md`" line is stale on path. PLAN_V2's §2 "Production anchors" and §3 "Architectural commitments" remain load-bearing.

`CHANGELOG.md` holds the public release notes.

### Coder catalog (built-ins registered by `CoderRegistry::default()`)

`src/coders/`:
`delta`, `delta_of_delta`, `leb128`, `rle`, `deltamin`, `bitpack_fixed` (param: `width`), `bitpack_auto`, `zstd` (param: `level`), `lz4`, `snappy`, `pcodec` (param: `level`; integer i8..i64 / u8..u64 + f32/f64), `gorilla`, `elias_fano`. All scalar — SIMD (TurboPFor family) is a future direction (see `docs/ROADMAP.md`).

When adding a coder: stable string ID (frozen the moment any `.he` ships with it), specialize on `input_type: DataType` in the factory, **new ID for new parameter shapes** (`bitpack_fixed` vs `bitpack_auto` is the canonical split — don't add boolean switches to one ID).

### Recursive `LogicalType` (schema vocabulary)

The schema vocabulary is fully recursive: nullability, lists, and dictionaries compose via `Nullable`, `List`, `Map`, `Union`, and `Dictionary` rather than dedicated flat variants. (Earlier revisions carried flat `NullablePrim` / `ArrayOf` / `DictPrim` etc.; these were removed before 1.0 — the format makes no on-disk stability promise at 0.x, so regenerate any old files from source.) All recursive variants ship as **struct variants** (`Variant { field: Box<LogicalType> }`), not newtype, because `#[serde(tag = "kind")]` cannot wrap newtypes. The `"kind"` value and inner field names are wire-format-frozen the same way coder IDs are — they appear in every `.he` schema JSON.

### Dict columns + multi-stripe gotcha

`HeliumReader::read_column("dict_col")` errors on multi-stripe files because each stripe carries its own dictionary. Use `read_column_at_stripe(name, stripe_idx)` and handle each stripe's dictionary explicitly. Tests covering this live in `tests/file_format.rs` / `file_format_modes.rs`.

## Anti-patterns (do not propose without new evidence)

- **Dremel (r, d) levels** — Parquet-specific; we shred via Arrow-style Struct/List, not levels. `grep -E "repetition_level|definition_level"` should stay empty.
- **Multi-table + `Ref` column types** (old C++ Helium) — explicitly rejected; doesn't map to Arrow.
- **Adding coders to fix small-batch (<256 rows/stripe) workloads** — issue is per-column framing overhead, not coder choice. Document the limitation; recommend zstd + pre-trained dictionary for that regime.
- **Wide `#[allow(...)]` to silence clippy**, **`unwrap()` / `expect()` / `panic!()` in non-test `src/`** — narrow with inline `// SAFETY:` justification if genuinely necessary.
- **Reintroducing the workspace split** without strong evidence: it was removed because the project ships exactly one crate; multi-crate added coordination overhead without payoff. If a future module genuinely needs separate publication, reopen the discussion with concrete numbers.
