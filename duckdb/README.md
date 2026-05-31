# helium-duckdb — DuckDB extension for reading `.he` files

A loadable DuckDB extension that registers a `read_he(path)` table function.
Once loaded, you can query any Helium `.he` file directly from SQL:

```sql
LOAD 'helium_duckdb';
SELECT * FROM read_he('path/to/file.he') LIMIT 5;
SELECT count(*) FROM read_he('path/to/file.he');
SELECT col_a, col_b FROM read_he('path/to/file.he') WHERE col_a > 100;
-- catalog-mode files: pass the catalog directory by named parameter
SELECT * FROM read_he('path/to/file.he', catalog := '/path/to/catalog');
```

## Status

The extension ships the `read_he` table function over the full recursive type set,
read-only, with:

- **Projection pushdown** — only the columns you select are decoded. Selecting
  1 of N columns decodes 1, not N (uses DuckDB's `init`-phase projected column
  indices + the reader's per-column pruning).
- **One reader held open** across all stripes (the file is opened once per scan,
  not once per stripe).
- **Nested types** — `Struct`, `List`, and `Map` map onto DuckDB STRUCT / LIST /
  MAP vectors.
- **Catalog-mode files** — pass `catalog := '<dir>'` to resolve the schema.

Remaining items (replacement scan, scalar UDFs, and the predicate-pushdown
caveat below) are tracked in [`docs/ROADMAP.md`](../docs/ROADMAP.md) → *Bindings*.

### A note on predicate pushdown

Helium's footer carries per-stripe min/max and containment filters, and this
crate implements the stripe-pruning logic for scalar comparisons
(`src/prune.rs`, unit-tested). It is **not auto-driven** today: DuckDB's
*loadable* extension C-API exposes projection pushdown but **no filter-pushdown
hook**, so DuckDB never hands the extension the `WHERE` bounds. Until the C-API
gains a filter accessor, DuckDB applies `WHERE` after the scan; the pruning
machinery is ready for the moment it does.

This is a deliberate upstream gap, not an oversight — and it is unlikely to
change soon. DuckDB's *C++* `TableFunction` API has full filter pushdown
(`filter_pushdown`, `TableFilterSet`, `pushdown_complex_filter`), which the
in-tree parquet/arrow readers use. The *C* extension API does not surface it
(verified against DuckDB `main`, not just the pinned C-API version: the only
pushdown symbol is `duckdb_table_function_supports_projection_pushdown`, and the
init/bind info exposes no filter accessor). The reason is structural: projection
pushdown is a flat list of column indices, trivial to expose stably in C, while
a filter set is a recursive tree of ~13 polymorphic node types (constant,
conjunction, IN, IS [NOT] NULL, expression, dynamic, bloom, …) that is itself
mid-refactor toward a unified expression representation. Two upstream PRs that
added a C filter API ([duckdb#14591], [duckdb#19093]) were both closed without
merging, stalled on the same unresolved design question — whether to mirror the
C++ `TableFilter` hierarchy or expose filters as generic `duckdb_expression`s —
plus the all-or-nothing contract that a function opting into filter pushdown
must handle every filter type.

A native **C++** shim *can* reach the full pushdown API (and a Rust crate can
now compile one more easily, since `libduckdb-sys` exposes the DuckDB headers
via `DEP_DUCKDB_INCLUDE`, [duckdb-rs#753]). That path is not taken here on
purpose: building against the C++ API makes the extension a version-locked C++
ABI artifact (`ExtensionABIType::CPP`, "version needs to match precisely")
rather than a portable loadable `.duckdb_extension` that works across DuckDB
versions — which is the whole reason this extension targets the stable C API.
The pragmatic near-term alternative, if stripe pruning is needed before the
C-API catches up, is an explicit user-supplied range parameter on `read_he`
that feeds `prune.rs` directly; it is a manual stopgap (the caller must keep the
range consistent with the `WHERE` clause) and is intentionally not implemented
yet.

[duckdb#14591]: https://github.com/duckdb/duckdb/pull/14591
[duckdb#19093]: https://github.com/duckdb/duckdb/pull/19093
[duckdb-rs#753]: https://github.com/duckdb/duckdb-rs/pull/753

## Prerequisites

| Tool | Minimum version | Install |
|---|---|---|
| Rust toolchain | 1.85 | `rustup update stable` |
| DuckDB CLI | 1.2.0 | `brew install duckdb` (macOS) |
| `cargo-duckdb-ext-tools` | 0.5.0 | `cargo install cargo-duckdb-ext-tools` |

The extension is an **independent Cargo project** inside `helium-core/duckdb/`.
It depends on the `helium` library via a `path` dependency, so it does not need
to be published to crates.io.

## Building

```bash
# From helium-core/
cd duckdb/
cargo build --release
```

This produces `target/release/libhelium_duckdb.dylib` (macOS) or
`target/release/libhelium_duckdb.so` (Linux).

## Packaging

DuckDB requires extensions to be packaged with metadata appended to the shared
library. The simplest path is the helper script, which builds (`--locked`, so
the committed `Cargo.lock` pins the C-API ABI), installs `cargo-duckdb-ext-tools`
if needed, and packages for the host platform:

```bash
# From helium-core/ — autodetect the host platform
bash duckdb/packaging/package.sh

# Or target an explicit platform (osx_arm64 | osx_amd64 | linux_amd64 | linux_arm64)
bash duckdb/packaging/package.sh -p linux_amd64

# Override the version / C-API stamps
EXT_VERSION=v0.1.0 CAPI_VERSION=v1.2.0 bash duckdb/packaging/package.sh
```

Both produce `duckdb/helium_duckdb.duckdb_extension`. The supported platform
tuples and the ABI-coupling rationale live in
[`packaging/matrix.md`](packaging/matrix.md); the per-platform CI build+load job
is [`ci/extension-matrix.yml`](ci/extension-matrix.yml); the steps to publish
through DuckDB's community-extensions repo are in
[`packaging/community-extension-submission.md`](packaging/community-extension-submission.md).

To package by hand instead of using the script:

```bash
cargo build --release --locked
cargo-duckdb-ext package \
  -i target/release/libhelium_duckdb.dylib \   # .so on Linux
  -o helium_duckdb.duckdb_extension \
  -v v0.1.0 \
  -p osx_arm64 \                                # your platform tuple
  --duckdb-capi-version v1.2.0
```

## Loading the extension in DuckDB

DuckDB requires the `-unsigned` flag (or `SET allow_unsigned_extensions=true`)
to load locally-built extensions that are not signed by DuckDB Labs:

```bash
# Interactive shell
duckdb -unsigned my_database.duckdb

# One-shot query
duckdb -unsigned :memory: -c "LOAD '/path/to/helium_duckdb.duckdb_extension'; SELECT count(*) FROM read_he('file.he');"
```

Within DuckDB:
```sql
-- Load by absolute path
LOAD '/absolute/path/to/helium_duckdb.duckdb_extension';

-- Now query any .he file
SELECT * FROM read_he('/path/to/file.he') LIMIT 10;

-- catalog-mode files: resolve the schema from a catalog directory
SELECT * FROM read_he('/path/to/file.he', catalog := '/path/to/catalog') LIMIT 10;
```

## Smoke test

The `smoke.sh` script builds `.he` fixtures and runs a real `LOAD` + query suite
covering projection pushdown, nullable + multi-stripe correctness (values
straddling stripe/chunk boundaries), nested `Struct` / `List` / `Map`, and
catalog-mode reads:

```bash
# From helium-core/
bash duckdb/smoke.sh
```

Expected output ends with `=== All smoke tests passed ===`.

## Type mapping

| Helium type | DuckDB type |
|---|---|
| `Primitive { I8 }` | `TINYINT` |
| `Primitive { I16 }` | `SMALLINT` |
| `Primitive { I32 }` | `INTEGER` |
| `Primitive { I64 }` | `BIGINT` |
| `Primitive { U8 }` | `UTINYINT` |
| `Primitive { U16 }` | `USMALLINT` |
| `Primitive { U32 }` | `UINTEGER` |
| `Primitive { U64 }` | `UBIGINT` |
| `Primitive { F32 }` | `FLOAT` |
| `Primitive { F64 }` | `DOUBLE` |
| `Utf8` | `VARCHAR` |
| `Binary` | `BLOB` |
| `Dictionary { Primitive { T } }` | same as `T` (dictionary expanded) |
| `Dictionary { Utf8 }` | `VARCHAR` (dictionary expanded) |
| `Nullable { T }` | same as `T`, nullable |
| `Struct { fields }` | `STRUCT(...)` |
| `List { inner }` | `LIST(inner)` |
| `Map { key, value }` | `MAP(key, value)` |
| `Decimal128 { p, s }` | `DECIMAL(p, s)` |
| `Date { Days }` | `DATE` |
| `Date { Millis }` | `BIGINT` (ms since epoch) |
| `Datetime { unit, tz: None }` | `TIMESTAMP_S` / `TIMESTAMP_MS` / `TIMESTAMP` / `TIMESTAMP_NS` |
| `Datetime { unit, tz: Some(_) }` | `TIMESTAMPTZ` |

## Limitations

These are tracked in [`docs/ROADMAP.md`](../docs/ROADMAP.md) → *Bindings*.

- **Predicate pushdown is not auto-driven.** DuckDB applies `WHERE` clauses
  after reading the projected rows. The stripe-pruning logic exists and is
  unit-tested (`src/prune.rs`), but the DuckDB *loadable* extension C-API
  (v1.2.0) exposes no filter-pushdown hook, so DuckDB never passes the
  extension the `WHERE` bounds. See the "note on predicate pushdown" above.
- **`Union` is not yet projected.** It errors at bind time with a clear
  message. `Struct`, `List`, and `Map` are supported.
- **Read-only.** The extension queries existing `.he` files; produce them with
  the `helium` CLI or library.

## DuckDB version compatibility

The extension is compiled against the DuckDB C extension API v1.2.0 and is
compatible with DuckDB 1.2.0 and later.  The `-unsigned` flag is required for
all locally-built, unsigned extensions regardless of DuckDB version.

The Rust crate (`duckdb` 1.10502.0) uses the C extension ABI, not the bundled
DuckDB library, so the compiled extension works with any matching DuckDB CLI
version without re-linking.
