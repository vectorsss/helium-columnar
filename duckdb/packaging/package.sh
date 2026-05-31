#!/usr/bin/env bash
# package.sh — build and package the helium DuckDB extension for one platform.
#
# Produces `helium_duckdb.duckdb_extension` (a shared library with DuckDB
# metadata appended) for the current host platform, or a platform passed in.
#
# Usage:
#   bash duckdb/packaging/package.sh                       # autodetect platform
#   bash duckdb/packaging/package.sh -p linux_amd64        # explicit platform
#   EXT_VERSION=v0.1.0 CAPI_VERSION=v1.2.0 bash duckdb/packaging/package.sh
#
# Environment:
#   EXT_VERSION   extension version stamp   (default: v0.1.0)
#   CAPI_VERSION  DuckDB C-API ABI version  (default: v1.2.0)
#   OUT           output path               (default: duckdb/helium_duckdb.duckdb_extension)
#
# The C-API ABI version is the coupling point: a packaged extension only loads
# into DuckDB builds that ship a compatible C-API. `Cargo.lock` is committed so
# the `duckdb`/`libduckdb-sys` crate versions (which determine the ABI) are
# reproducible. See `duckdb/packaging/matrix.md` for the supported matrix.
set -euo pipefail

DUCKDB_DIR="$(cd "$(dirname "$0")/.." && pwd)"

EXT_VERSION="${EXT_VERSION:-v0.1.0}"
CAPI_VERSION="${CAPI_VERSION:-v1.2.0}"
OUT="${OUT:-${DUCKDB_DIR}/helium_duckdb.duckdb_extension}"

# --- Resolve platform (DuckDB's naming) -------------------------------------
PLATFORM=""
while getopts "p:" opt; do
    case "${opt}" in
        p) PLATFORM="${OPTARG}" ;;
        *) echo "usage: $0 [-p platform]"; exit 1 ;;
    esac
done

OS="$(uname -s)"
ARCH="$(uname -m)"
if [ -z "${PLATFORM}" ]; then
    case "${OS}/${ARCH}" in
        Darwin/arm64)  PLATFORM="osx_arm64" ;;
        Darwin/x86_64) PLATFORM="osx_amd64" ;;
        Linux/aarch64) PLATFORM="linux_arm64" ;;
        Linux/x86_64)  PLATFORM="linux_amd64" ;;
        *) echo "ERROR: unknown platform ${OS}/${ARCH}; pass -p <platform>"; exit 1 ;;
    esac
fi

case "${OS}" in
    Darwin) DYLIB="${DUCKDB_DIR}/target/release/libhelium_duckdb.dylib" ;;
    *)      DYLIB="${DUCKDB_DIR}/target/release/libhelium_duckdb.so" ;;
esac

echo "=== Packaging helium_duckdb ==="
echo "platform:     ${PLATFORM}"
echo "ext version:  ${EXT_VERSION}"
echo "capi version: ${CAPI_VERSION}"
echo "output:       ${OUT}"

# --- Build (locked, so the committed Cargo.lock pins the ABI) ---------------
( cd "${DUCKDB_DIR}" && cargo build --release --locked )

# --- Install the packaging tool if missing ----------------------------------
if ! command -v cargo-duckdb-ext &>/dev/null; then
    echo "--- Installing cargo-duckdb-ext-tools..."
    cargo install cargo-duckdb-ext-tools
fi

# --- Append DuckDB metadata to the shared library ---------------------------
cargo-duckdb-ext package \
    -i "${DYLIB}" \
    -o "${OUT}" \
    -v "${EXT_VERSION}" \
    -p "${PLATFORM}" \
    --duckdb-capi-version "${CAPI_VERSION}"

echo "--- Wrote ${OUT}"
