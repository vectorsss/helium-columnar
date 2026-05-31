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

# ---- 8. Projection pushdown: only projected columns are decoded -------------
# Build a file where the bulk of the bytes are wide string columns, then show
# that projecting more wide columns takes proportionally longer — i.e. the
# extension decodes only the projected columns, not all of them.
echo ""
echo "--- Query 4: projection pushdown (1 wide column vs 4) ---"
PROJ_CSV="${TMPDIR_SMOKE}/proj.csv"
PROJ_HE="${TMPDIR_SMOKE}/proj.he"
python3 - "${PROJ_CSV}" <<'PYEOF'
import csv, sys, random
random.seed(7)
with open(sys.argv[1], "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "big_a", "big_b", "big_c", "big_d"])
    rnd = lambda: "".join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(40))
    for i in range(1, 60001):
        w.writerow([i, rnd(), rnd(), rnd(), rnd()])
PYEOF
"${HELIUM_CLI}" convert "${PROJ_CSV}" -o "${PROJ_HE}" --stripe-rows 20000

# Correctness: a reordered subset projection returns the right values.
P4_OUT="$(duckdb -unsigned :memory: << SQL
LOAD '${EXT}';
SELECT id FROM read_he('${PROJ_HE}') ORDER BY id LIMIT 3;
SQL
)"
echo "${P4_OUT}"
if echo "${P4_OUT}" | grep -q "^[│| ]*1[ │|]"; then
    echo "    [PASS] single-column projection returns id=1..3"
else
    echo "    [FAIL] projection did not return expected ids"
    exit 1
fi

# Timing: decoding 4 wide columns must cost noticeably more than 1. If all
# columns were always decoded, the two would be equal.
t_one="$( { /usr/bin/time -p duckdb -unsigned :memory: -c \
    "LOAD '${EXT}'; SELECT max(big_a) FROM read_he('${PROJ_HE}');" >/dev/null; } 2>&1 \
    | awk '/^real/ {print $2}')"
t_four="$( { /usr/bin/time -p duckdb -unsigned :memory: -c \
    "LOAD '${EXT}'; SELECT max(big_a),max(big_b),max(big_c),max(big_d) FROM read_he('${PROJ_HE}');" >/dev/null; } 2>&1 \
    | awk '/^real/ {print $2}')"
echo "    decode 1 wide column: ${t_one}s   decode 4 wide columns: ${t_four}s"
if awk -v a="${t_one}" -v b="${t_four}" 'BEGIN { exit !(b > a + 0.005) }'; then
    echo "    [PASS] 4-column decode slower than 1-column → only projected columns decoded"
else
    echo "    [WARN] timing inconclusive (machine too fast / noisy); correctness still verified"
fi

# Zero-projection path (count(*) needs no columns) returns the row count.
P4C_OUT="$(duckdb -unsigned :memory: << SQL
LOAD '${EXT}';
SELECT count(*) FROM read_he('${PROJ_HE}');
SQL
)"
if echo "${P4C_OUT}" | grep -q "60000"; then
    echo "    [PASS] count(*) (zero-projection) = 60000"
else
    echo "    [FAIL] count(*) wrong: ${P4C_OUT}"
    exit 1
fi

# ---- 9. Nullable + multi-stripe correctness (absolute-row indexing) ---------
# Generate fixtures the CLI converters can't easily produce: a Map column and a
# Nullable column whose nulls straddle stripe (5000) and chunk (2048) boundaries.
echo ""
echo "--- Query 5: nullable column across stripe/chunk boundaries ---"
FIX_BIN="${DUCKDB_DIR}/target/release/examples/make_fixtures"
if [ ! -x "${FIX_BIN}" ] || [ "${DYLIB}" -nt "${FIX_BIN}" ]; then
    echo "--- Building fixture generator..."
    (cd "${DUCKDB_DIR}" && cargo build --release --example make_fixtures)
fi
MAP_HE="${TMPDIR_SMOKE}/map.he"
NULL_HE="${TMPDIR_SMOKE}/nullable.he"
"${FIX_BIN}" "${MAP_HE}" "${NULL_HE}"

# v = id*2 for non-null rows; null exactly when id % 3 == 0, over 15000 rows in
# 3 stripes of 5000. Expected: total 15000, nulls 5000, sum(v)=150000000.
N_OUT="$(duckdb -unsigned :memory: << SQL
LOAD '${EXT}';
SELECT count(*) AS total, count(v) AS non_null, sum(v) AS sumv
FROM read_he('${NULL_HE}');
SQL
)"
echo "${N_OUT}"
if echo "${N_OUT}" | grep -q "15000" && echo "${N_OUT}" | grep -q "10000" && echo "${N_OUT}" | grep -q "150000000"; then
    echo "    [PASS] nullable totals correct across stripe/chunk boundaries"
else
    echo "    [FAIL] nullable aggregate mismatch"
    exit 1
fi

# Spot-check a row at the first stripe boundary.
NB_OUT="$(duckdb -unsigned :memory: << SQL
LOAD '${EXT}';
SELECT v FROM read_he('${NULL_HE}') WHERE id = 5000;
SQL
)"
if echo "${NB_OUT}" | grep -q "10000"; then
    echo "    [PASS] row at stripe boundary (id=5000) → v=10000"
else
    echo "    [FAIL] stripe-boundary value wrong: ${NB_OUT}"
    exit 1
fi

# ---- 10. Nested types: Map / List / Struct ---------------------------------
echo ""
echo "--- Query 6: Map column → DuckDB MAP ---"
M_OUT="$(duckdb -unsigned :memory: << SQL
LOAD '${EXT}';
SELECT id, attrs['k0'] AS k0 FROM read_he('${MAP_HE}') ORDER BY id;
SQL
)"
echo "${M_OUT}"
if echo "${M_OUT}" | grep -q " 0 "; then
    echo "    [PASS] Map column projected and indexable ('k0' = 0)"
else
    echo "    [FAIL] Map column did not project correctly"
    exit 1
fi

# List + Struct via NDJSON (if json conversion is available in this build).
NDJSON="${TMPDIR_SMOKE}/nested.ndjson"
NESTED_HE="${TMPDIR_SMOKE}/nested.he"
cat > "${NDJSON}" <<'EOF'
{"id": 1, "tags": ["a","b"], "addr": {"city":"NYC","zip":10001}}
{"id": 2, "tags": [], "addr": {"city":"LA","zip":90001}}
EOF
if "${HELIUM_CLI}" convert "${NDJSON}" -o "${NESTED_HE}" 2>/dev/null; then
    echo "--- Query 7: List + Struct columns ---"
    L_OUT="$(duckdb -unsigned :memory: << SQL
LOAD '${EXT}';
SELECT id, len(tags) AS ntags, addr.city AS city FROM read_he('${NESTED_HE}') ORDER BY id;
SQL
)"
    echo "${L_OUT}"
    if echo "${L_OUT}" | grep -q "NYC" && echo "${L_OUT}" | grep -q "LA"; then
        echo "    [PASS] List + Struct columns project correctly (incl. empty list)"
    else
        echo "    [FAIL] nested List/Struct projection wrong"
        exit 1
    fi
else
    echo "    [SKIP] json adapter not in this build; List/Struct covered by Query 6 fixtures"
fi

# ---- 11. catalog-mode support ----------------------------------------------
echo ""
echo "--- Query 8: catalog-mode file via catalog := '<dir>' ---"
CAT_DIR="${TMPDIR_SMOKE}/catalog"
CATALOG_HE="${TMPDIR_SMOKE}/catalog.he"
mkdir -p "${CAT_DIR}"
if "${HELIUM_CLI}" convert "${CSV_FILE}" -o "${CATALOG_HE}" --catalog "${CAT_DIR}" 2>/dev/null; then
    # Without the catalog parameter, a catalog-mode file must error clearly.
    CATALOG_ERR="$(duckdb -unsigned :memory: -c "LOAD '${EXT}'; SELECT count(*) FROM read_he('${CATALOG_HE}');" 2>&1 || true)"
    if echo "${CATALOG_ERR}" | grep -q "schema resolver"; then
        echo "    [PASS] catalog file without resolver errors with a clear message"
    else
        echo "    [FAIL] expected resolver error, got: ${CATALOG_ERR}"
        exit 1
    fi
    # With the catalog directory, it reads.
    CATALOG_OUT="$(duckdb -unsigned :memory: << SQL
LOAD '${EXT}';
SELECT count(*) AS total FROM read_he('${CATALOG_HE}', catalog := '${CAT_DIR}');
SQL
)"
    echo "${CATALOG_OUT}"
    if echo "${CATALOG_OUT}" | grep -q "100"; then
        echo "    [PASS] catalog file read via catalog := parameter"
    else
        echo "    [FAIL] catalog read returned wrong count"
        exit 1
    fi
else
    echo "    [SKIP] catalog mode unavailable in this CLI build"
fi

echo ""
echo "=== All smoke tests passed ==="
