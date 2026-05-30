# helium-duckdb — DuckDB extension for reading `.he` files

A loadable DuckDB extension that registers a `read_he(path)` table function.
Once loaded, you can query any Helium `.he` file directly from SQL:

```sql
LOAD 'helium_duckdb';
SELECT * FROM read_he('path/to/file.he') LIMIT 5;
SELECT count(*) FROM read_he('path/to/file.he');
SELECT col_a, col_b FROM read_he('path/to/file.he') WHERE col_a > 100;
```

## Status

The extension ships the `read_he` table function over the full v3 type set
(read-only). Planned next steps — projection/predicate pushdown, replacement
scan, scalar UDFs, and community-extension submission — are tracked in
[`docs/ROADMAP.md`](../docs/ROADMAP.md) → *Bindings*.

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

DuckDB requires extensions to be packaged with metadata appended to the
shared library.  Use `cargo-duckdb-ext-tools`:

**macOS arm64 (Apple Silicon):**
```bash
cargo-duckdb-ext package \
  -i target/release/libhelium_duckdb.dylib \
  -o helium_duckdb.duckdb_extension \
  -v v0.1.0 \
  -p osx_arm64 \
  --duckdb-capi-version v1.2.0
```

**macOS x86_64:**
```bash
cargo-duckdb-ext package \
  -i target/release/libhelium_duckdb.dylib \
  -o helium_duckdb.duckdb_extension \
  -v v0.1.0 \
  -p osx_amd64 \
  --duckdb-capi-version v1.2.0
```

**Linux x86_64:**
```bash
cargo-duckdb-ext package \
  -i target/release/libhelium_duckdb.so \
  -o helium_duckdb.duckdb_extension \
  -v v0.1.0 \
  -p linux_amd64 \
  --duckdb-capi-version v1.2.0
```

**Linux aarch64:**
```bash
cargo-duckdb-ext package \
  -i target/release/libhelium_duckdb.so \
  -o helium_duckdb.duckdb_extension \
  -v v0.1.0 \
  -p linux_arm64 \
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
```

## Smoke test

The `smoke.sh` script builds a 5-column, 100-row `.he` file and runs three
queries against it:

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
| `NullablePrim { T }` | same as `T`, nullable |
| `NullableUtf8` | `VARCHAR`, nullable |
| `NullableBinary` | `BLOB`, nullable |
| `Dictionary { Primitive { T } }` | same as `T` (dictionary expanded) |
| `Dictionary { Utf8 }` | `VARCHAR` (dictionary expanded) |
| `Nullable { T }` | same as `T`, nullable |
| `Decimal128 { p, s }` | `DECIMAL(p, s)` |
| `Date { Days }` | `DATE` |
| `Date { Millis }` | `BIGINT` (ms since epoch) |
| `Datetime { unit, tz: None }` | `TIMESTAMP_S` / `TIMESTAMP_MS` / `TIMESTAMP` / `TIMESTAMP_NS` |
| `Datetime { unit, tz: Some(_) }` | `TIMESTAMPTZ` |

## Limitations

These are tracked in [`docs/ROADMAP.md`](../docs/ROADMAP.md) → *Bindings*.

- **No predicate pushdown.** DuckDB applies all `WHERE` clauses after reading
  all rows. Closing this via the DuckDB filter-pushdown hooks + stripe min/max
  pruning is the highest-value next step.
- **No projection pushdown.** All columns are read from disk even if only a
  subset is projected; column pruning is the most natural first win.
- **Nested types not supported.** `Struct`, `List`, `Map`, `Union`, `ArrayOf`,
  and `ArrayOfUtf8` columns cause an error at bind time with a clear message.
  Use `helium convert --csv-strict` to flatten nested schemas before loading.
- **Catalog-mode (v6) files not supported.** Files written with
  `HeliumWriter::with_catalog_ref` will error.  Use the standard v5 writer.
- **Read-only.** The extension queries existing `.he` files; produce them with
  the `helium` CLI or library.
- **Multi-stripe files re-open the file per stripe.** `func` opens the file and
  rebuilds the reader each time it advances to a new stripe; holding one reader
  open across stripes is a pending optimization.

## DuckDB version compatibility

The extension is compiled against the DuckDB C extension API v1.2.0 and is
compatible with DuckDB 1.2.0 and later.  The `-unsigned` flag is required for
all locally-built, unsigned extensions regardless of DuckDB version.

The Rust crate (`duckdb` v1.10502.0) uses the C extension ABI, not the bundled
DuckDB library, so the compiled extension works with any matching DuckDB CLI
version without re-linking.
