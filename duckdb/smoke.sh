#!/usr/bin/env bash
# smoke.sh — integration smoke test for the helium DuckDB extension.
#
# Usage:
#   cd helium-core/
#   bash duckdb/smoke.sh
#
# Prerequisites (auto-resolved below):
#   - duckdb CLI in PATH            (brew install duckdb)
#   - helium CLI built              (cargo build --release --features cli)
#   - cargo-duckdb-ext-tools        (cargo install cargo-duckdb-ext-tools)
#
# The script will rebuild the extension if the .duckdb_extension file is
# missing or if the source library is newer than the packaged file.
set -euo pipefail

# ---- Paths -----------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DUCKDB_DIR="${REPO_ROOT}/duckdb"
HELIUM_CLI="${REPO_ROOT}/target/release/helium"
EXT="${DUCKDB_DIR}/helium_duckdb.duckdb_extension"

OS="$(uname -s)"
ARCH="$(uname -m)"
if [ "${OS}" = "Darwin" ]; then
    DYLIB="${DUCKDB_DIR}/target/release/libhelium_duckdb.dylib"
else
    DYLIB="${DUCKDB_DIR}/target/release/libhelium_duckdb.so"
fi

TMPDIR_SMOKE="$(mktemp -d)"
HE_FILE="${TMPDIR_SMOKE}/smoke.he"
CSV_FILE="${TMPDIR_SMOKE}/smoke.csv"
trap 'rm -rf "${TMPDIR_SMOKE}"' EXIT

echo "=== helium DuckDB extension smoke test ==="
echo "repo:      ${REPO_ROOT}"
echo "extension: ${EXT}"

# ---- 1. Check prerequisites ------------------------------------------------
if ! command -v duckdb &>/dev/null; then
    echo "ERROR: duckdb not found in PATH. Install with: brew install duckdb"
    exit 1
fi
echo "duckdb:    $(duckdb --version)"

# ---- 2. Build helium CLI if needed -----------------------------------------
if [ ! -f "${HELIUM_CLI}" ]; then
    echo "--- Building helium CLI..."
    (cd "${REPO_ROOT}" && cargo build --release --features cli)
fi

# ---- 3. Build + package extension if needed --------------------------------
if [ ! -f "${EXT}" ] || [ "${DYLIB}" -nt "${EXT}" ]; then
    echo "--- Building extension (this may take a minute on first run)..."
    (cd "${DUCKDB_DIR}" && cargo build --release)

    if ! command -v cargo-duckdb-ext &>/dev/null; then
        echo "--- Installing cargo-duckdb-ext-tools..."
        cargo install cargo-duckdb-ext-tools
    fi

    case "${OS}/${ARCH}" in
        Darwin/arm64)  PLATFORM="osx_arm64" ;;
        Darwin/x86_64) PLATFORM="osx_amd64" ;;
        Linux/aarch64) PLATFORM="linux_arm64" ;;
        Linux/x86_64)  PLATFORM="linux_amd64" ;;
        *)
            echo "WARNING: unknown platform ${OS}/${ARCH}, defaulting to linux_amd64"
            PLATFORM="linux_amd64"
            ;;
    esac
    echo "--- Platform: ${PLATFORM}"

    cargo-duckdb-ext package \
        -i "${DYLIB}" \
        -o "${EXT}" \
        -v v0.1.0 \
        -p "${PLATFORM}" \
        --duckdb-capi-version v1.2.0
fi

# ---- 4. Create test data (5 columns, 100 rows) -----------------------------
echo "--- Creating test .he file (5 columns, 100 rows)..."
python3 - "${CSV_FILE}" <<'PYEOF'
import csv, sys, math

outf = sys.argv[1]
names = ["Alice","Bob","Charlie","Diana","Eve","Frank","Grace","Henry","Iris","Jack"]
cats  = ["A","B","C","D","E"]
with open(outf, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "name", "score", "category", "is_active"])
    for i in range(1, 101):
        w.writerow([
            i,
            names[(i-1) % len(names)],
            round(50.0 + 50.0 * abs(math.sin(i * 0.3)), 2),
            cats[(i-1) % len(cats)],
            1 if i % 2 == 1 else 0,
        ])
PYEOF

"${HELIUM_CLI}" convert "${CSV_FILE}" -o "${HE_FILE}"
echo "    wrote ${HE_FILE}"

# ---- 5. Query 1: basic SELECT with LIMIT -----------------------------------
echo ""
echo "--- Query 1: SELECT id, name, score ... LIMIT 5"
Q1_OUT="$(duckdb -unsigned :memory: << SQL
LOAD '${EXT}';
SELECT id, name, score FROM read_he('${HE_FILE}') ORDER BY id LIMIT 5;
SQL
)"
echo "${Q1_OUT}"

if echo "${Q1_OUT}" | grep -q "Alice"; then
    echo "    [PASS] first row contains 'Alice'"
else
    echo "    [FAIL] expected 'Alice' in first row"
    exit 1
fi

# ---- 6. Query 2: aggregation -----------------------------------------------
echo ""
echo "--- Query 2: SELECT count(*)"
Q2_OUT="$(duckdb -unsigned :memory: << SQL
LOAD '${EXT}';
SELECT count(*) AS total FROM read_he('${HE_FILE}');
SQL
)"
echo "${Q2_OUT}"

if echo "${Q2_OUT}" | grep -q "100"; then
    echo "    [PASS] count = 100"
else
    echo "    [FAIL] expected count = 100"
    exit 1
fi

# ---- 7. Query 3: predicate filtering ---------------------------------------
echo ""
echo "--- Query 3: SELECT ... WHERE score > 95"
Q3_OUT="$(duckdb -unsigned :memory: << SQL
LOAD '${EXT}';
SELECT id, name, score FROM read_he('${HE_FILE}') WHERE score > 95.0 ORDER BY id LIMIT 5;
SQL
)"
echo "${Q3_OUT}"

if echo "${Q3_OUT}" | grep -q "9[5-9]\|100"; then
    echo "    [PASS] filtered result contains high scores"
else
    echo "    [FAIL] expected rows with score > 95"
    exit 1
fi

echo ""
echo "=== All smoke tests passed ==="
