# Build matrix and distribution

## Why a matrix at all

A packaged `.duckdb_extension` is a shared library with DuckDB metadata
appended. Two axes determine whether it loads into a given DuckDB:

1. **C-API ABI version** — the loadable extension links against DuckDB's C
   extension API. An extension built against C-API `vX` loads into any DuckDB
   that ships a compatible C-API. The ABI is fixed by the `duckdb` /
   `libduckdb-sys` crate versions, which `duckdb/Cargo.lock` pins (committed, so
   the ABI is reproducible across machines and CI).
2. **Platform** — one artifact per OS/arch tuple, using DuckDB's platform names.

## Platform tuples

| DuckDB platform | OS / arch         | Runner            |
|-----------------|-------------------|-------------------|
| `osx_arm64`     | macOS Apple Silicon | `macos-14`       |
| `osx_amd64`     | macOS Intel       | `macos-13`        |
| `linux_amd64`   | Linux x86-64      | `ubuntu-latest`   |
| `linux_arm64`   | Linux aarch64     | `ubuntu-24.04-arm`|
| `windows_amd64` | Windows x86-64    | `windows-latest`  |

Build each with:

```bash
bash duckdb/packaging/package.sh -p <platform>
# or autodetect on the host:
bash duckdb/packaging/package.sh
```

## C-API version

The current pin is **C-API `v1.2.0`** (`CAPI_VERSION` in `package.sh`). It loads
into DuckDB 1.2.0 and later — verified locally against the DuckDB 1.4.x CLI. To
target a different C-API, bump the `duckdb` crate (refresh `Cargo.lock`) and set
`CAPI_VERSION` to match; keep the two in lockstep.

## CI

`duckdb/ci/extension-matrix.yml` is a ready-to-merge GitHub Actions job that, for
each platform tuple, builds + packages the extension and (where a DuckDB CLI is
available) runs a real `LOAD` + query smoke. Merge its `jobs:` entry into the
root `.github/workflows/ci.yml` to replace today's compile-only gate. It is kept
as a separate snippet so the binding work stays inside `duckdb/`.
