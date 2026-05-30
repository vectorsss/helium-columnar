#!/usr/bin/env bash
#
# Reproducible real-data compression benchmark on the DoNext 5G/4G dataset.
#
# Compares, on a real telecom CSV:
#   - csv.zst             (plain zstd on the raw CSV)
#   - helium default      (Tier-1 inferred schema)
#   - helium optimized    (measured-optimal schema)
#   - avro + zstd         (5G "Avro+zstd" storage anchor; needs python+fastavro)
#   - avro + deflate      (Avro's native codec)
#
# The DoNext dataset (CC BY 4.0) is published on TU Dortmund's Dataverse:
#   https://doi.org/10.17877/TUDODATA-2026-T6MYPO
# The full set is ~4.6 GB, but this benchmark only needs one file —
# H-Bahn/cell_data.csv (~100 MB, the serving-cell Measurement-Report workload).
# `--fetch` downloads exactly that file (cached under target/donext/) and runs
# the benchmark on it, so no manual download is required.
#
# Usage:
#   scripts/donext_benchmark.sh --fetch [max_rows]            # auto-download + run
#   scripts/donext_benchmark.sh <path/to/data.csv> [max_rows] [delimiter]
#
#   --fetch    download H-Bahn/cell_data.csv from Dataverse, then benchmark it
#              (max_rows defaults to 100000 in this mode; delimiter is ';').
#   max_rows   optional — take only the first N rows (default: whole file).
#              Recommended: 100000 — keeps `optimize-schema` fast.
#   delimiter  optional — CSV field delimiter (default: ';' for DoNext).
#
# Requires: a release `helium` binary (cargo build --release --features cli),
#           zstd on PATH, and optionally python3 + fastavro for the Avro rows.
#
set -euo pipefail

# --- locate the helium binary ---
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HELIUM="${HELIUM_BIN:-$ROOT/target/release/helium}"
if [[ ! -x "$HELIUM" ]]; then
    echo "helium binary not found at $HELIUM" >&2
    echo "build it:  cargo build --release --features cli" >&2
    echo "or set HELIUM_BIN=/path/to/helium" >&2
    exit 1
fi

# --- args ---
if [[ $# -lt 1 ]]; then
    # Print the leading comment block (the doc header), stopping at the
    # first non-comment line so we never leak code into the usage text.
    awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; next} {exit}' "${BASH_SOURCE[0]}"
    exit 2
fi

# --- optional: auto-download the one file the benchmark needs (~100 MB) ---
# DoNext H-Bahn/cell_data.csv = Dataverse datafile id 81633 on data.tu-dortmund.de
# (full dataset is ~4.6 GB; we fetch just this serving-cell MR file). CC BY 4.0.
if [[ "$1" == "--fetch" ]]; then
    shift
    DATA_DIR="$ROOT/target/donext"
    FETCHED="$DATA_DIR/H-Bahn_cell_data.csv"
    URL="https://data.tu-dortmund.de/api/access/datafile/81633"
    mkdir -p "$DATA_DIR"
    if [[ -f "$FETCHED" ]]; then
        echo "✓ using cached $FETCHED"
    else
        echo "Downloading DoNext H-Bahn/cell_data.csv (~100 MB, CC BY 4.0)…"
        echo "  $URL"
        curl -fL --progress-bar -o "$FETCHED" "$URL"
        echo "✓ wrote $FETCHED"
    fi
    # Rebuild args: <fetched-csv> <max_rows (default 100000)> <delimiter ';'>
    set -- "$FETCHED" "${1:-100000}" ";"
fi

SRC="$1"
MAX_ROWS="${2:-0}"
DELIM="${3:-;}"

if [[ ! -f "$SRC" ]]; then
    echo "input CSV not found: $SRC" >&2
    echo "download DoNext from https://doi.org/10.17877/TUDODATA-2026-T6MYPO" >&2
    exit 1
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# --- optional row cap ---
CSV="$WORK/data.csv"
if [[ "$MAX_ROWS" -gt 0 ]]; then
    head -n "$((MAX_ROWS + 1))" "$SRC" > "$CSV"   # +1 for the header
else
    cp "$SRC" "$CSV"
fi
RAW=$(wc -c < "$CSV" | tr -d ' ')
NROWS=$(($(wc -l < "$CSV" | tr -d ' ') - 1))
NCOLS=$(head -1 "$CSV" | awk -F"$DELIM" '{print NF}')

if [[ "${DONEXT_TSV:-0}" != "1" ]]; then
    echo "=== DoNext compression benchmark ==="
    echo "source : $SRC"
    echo "rows   : $NROWS   cols: $NCOLS   delimiter: '$DELIM'"
    echo
fi

ratio() { awk "BEGIN{printf \"%.2fx\", $1/$2}"; }
pct()   { awk "BEGIN{printf \"%+.0f%%\", (1-$1/$2)*100}"; }

# --- csv.zst baseline ---
zstd -3 -q -f -k "$CSV" -o "$WORK/data.csv.zst"
ZST=$(wc -c < "$WORK/data.csv.zst" | tr -d ' ')

# --- helium default --- (helium prints progress to stderr; silence it)
"$HELIUM" convert "$CSV" -o "$WORK/default.he" --delimiter "$DELIM" >/dev/null 2>&1
HD=$(wc -c < "$WORK/default.he" | tr -d ' ')

# --- helium optimized ---
"$HELIUM" optimize-schema "$CSV" --delimiter "$DELIM" --out "$WORK/opt.json" >/dev/null 2>&1
"$HELIUM" convert "$CSV" -o "$WORK/opt.he" --schema "$WORK/opt.json" --delimiter "$DELIM" >/dev/null 2>&1
HO=$(wc -c < "$WORK/opt.he" | tr -d ' ')

# --- avro baseline (optional) ---
AV_DEFLATE="" ; AV_ZSTD=""
if python3 -c "import fastavro" 2>/dev/null; then
    "$HELIUM" infer-schema "$CSV" --delimiter "$DELIM" --out "$WORK/infer.json" >/dev/null 2>&1
    python3 "$ROOT/scripts/csv_to_avro.py" "$CSV" "$WORK/infer.json" "$WORK/av" "$DELIM" >/dev/null
    AV_DEFLATE=$(wc -c < "$WORK/av_deflate.avro" | tr -d ' ')
    zstd -3 -q -f -k "$WORK/av_null.avro" -o "$WORK/av_null.avro.zst"
    AV_ZSTD=$(wc -c < "$WORK/av_null.avro.zst" | tr -d ' ')
fi

# --- round-trip integrity ---
"$HELIUM" verify "$WORK/opt.he" >/dev/null && RT="OK" || RT="FAILED"

# --- machine-readable single line (used by donext_benchmark_all.sh) ---
if [[ "${DONEXT_TSV:-0}" == "1" ]]; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$SRC" "$NROWS" "$NCOLS" "$RAW" "$ZST" "${AV_DEFLATE:-0}" "${AV_ZSTD:-0}" "$HD" "$HO" "$RT"
    [[ "$RT" == "OK" ]] || exit 1
    exit 0
fi

# --- report ---
printf "%-22s %14s %10s %12s\n" "format" "bytes" "vs raw" "vs csv.zst"
printf "%-22s %14d %10s %12s\n" "raw csv"           "$RAW" "1.00x" "-"
printf "%-22s %14d %10s %12s\n" "csv.zst (-3)"       "$ZST" "$(ratio "$RAW" "$ZST")" "1.00x"
[[ -n "$AV_DEFLATE" ]] && printf "%-22s %14d %10s %12s\n" "avro (deflate)" "$AV_DEFLATE" "$(ratio "$RAW" "$AV_DEFLATE")" "$(ratio "$ZST" "$AV_DEFLATE")"
[[ -n "$AV_ZSTD" ]]    && printf "%-22s %14d %10s %12s\n" "avro (null)+zstd" "$AV_ZSTD" "$(ratio "$RAW" "$AV_ZSTD")" "$(ratio "$ZST" "$AV_ZSTD")"
printf "%-22s %14d %10s %12s\n" "helium default"     "$HD" "$(ratio "$RAW" "$HD")" "$(ratio "$ZST" "$HD")"
printf "%-22s %14d %10s %12s\n" "helium optimized"   "$HO" "$(ratio "$RAW" "$HO")" "$(ratio "$ZST" "$HO")"
echo
echo "helium-optimized vs csv.zst:       $(pct "$HO" "$ZST")"
if [[ -n "$AV_ZSTD" ]]; then
    echo "helium-optimized vs avro+zstd:     $(pct "$HO" "$AV_ZSTD")"
fi
echo "round-trip verify (opt.he):        $RT"
if [[ -z "$AV_ZSTD" ]]; then
    echo "(install python3 + fastavro for the avro baseline rows)"
fi
[[ "$RT" == "OK" ]] || exit 1
exit 0
