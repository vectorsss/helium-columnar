#!/usr/bin/env bash
#
# Comprehensive DoNext benchmark. Runs the per-file compression comparison
# (csv.zst / helium default / helium optimized / avro+zstd / avro+deflate) over
# EVERY category × data-type CSV in the DoNext dataset and prints one combined
# table. Wraps scripts/donext_benchmark.sh (one row per file via DONEXT_TSV).
#
# Usage:
#   scripts/donext_benchmark_all.sh <dataset_dir> [max_rows]
#   scripts/donext_benchmark_all.sh --fetch-all    [max_rows]
#
#   <dataset_dir>  a local DoNext download — the dir containing H-Bahn/, Mobile/,
#                  static/ with their *_data.csv files.
#   --fetch-all    download every *_data.csv (~4.8 GB) from TU Dortmund's
#                  Dataverse into target/donext/full/ first, then benchmark.
#   max_rows       per-file row cap (default 100000; keeps optimize-schema fast).
#
# Requires: release helium (cargo build --release --features cli), zstd, and
#           optionally python3 + fastavro for the avro rows. CC BY 4.0 dataset:
#           https://doi.org/10.17877/TUDODATA-2026-T6MYPO
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH="$ROOT/scripts/donext_benchmark.sh"

if [[ $# -lt 1 ]]; then
    awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; next} {exit}' "${BASH_SOURCE[0]}"
    exit 2
fi

# category/type -> Dataverse datafile id on data.tu-dortmund.de (for --fetch-all)
FETCH_LIST="
H-Bahn/cell_data.csv 81633
H-Bahn/datarate_data.csv 81635
H-Bahn/iperf_data.csv 81634
H-Bahn/latency_data.csv 81631
H-Bahn/neighboring_data.csv 81632
Mobile/cell_data.csv 81638
Mobile/iperf_data.csv 81639
Mobile/latency_data.csv 81637
Mobile/neighboring_data.csv 81636
static/cell_data.csv 121033
static/latency_data.csv 121031
"

# --- resolve the dataset directory ---
if [[ "$1" == "--fetch-all" ]]; then
    shift
    MAX_ROWS="${1:-100000}"
    DIR="$ROOT/target/donext/full"
    echo "$FETCH_LIST" | while read -r rel id; do
        [[ -z "$rel" ]] && continue
        out="$DIR/$rel"
        mkdir -p "$(dirname "$out")"
        if [[ -f "$out" ]]; then
            echo "✓ cached $rel" >&2
        else
            echo "downloading $rel (id $id)…" >&2
            curl -fL --progress-bar -o "$out" "https://data.tu-dortmund.de/api/access/datafile/$id"
        fi
    done
else
    DIR="$1"
    MAX_ROWS="${2:-100000}"
fi
[[ -d "$DIR" ]] || { echo "dataset dir not found: $DIR" >&2; exit 1; }

# --- discover the data CSVs (the *_data.csv files; skips static_locations etc.) ---
FILES=()
while IFS= read -r line; do FILES+=("$line"); done < <(find "$DIR" -type f -name '*_data.csv' | sort)
[[ ${#FILES[@]} -gt 0 ]] || { echo "no *_data.csv files under $DIR" >&2; exit 1; }

echo "Benchmarking ${#FILES[@]} files from $DIR (first $MAX_ROWS rows each)…" >&2

OUT="$ROOT/target/donext-benchmark-all.md"
mkdir -p "$ROOT/target"
{
    echo "# DoNext comprehensive benchmark"
    echo
    echo "Per-file compression, first $MAX_ROWS rows each. Sizes in bytes."
    echo "\`opt\` = helium optimized schema; \`def\` = helium default schema."
    echo
    echo "| file | rows | cols | raw | csv.zst | avro+zstd | helium-def | helium-opt | opt vs zst | opt vs avro |"
    echo "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
} > "$OUT"

for f in "${FILES[@]}"; do
    label="${f#"$DIR"/}"
    echo "  • $label" >&2
    # TSV fields: SRC NROWS NCOLS RAW ZST AV_DEFLATE AV_ZSTD HD HO RT
    line="$(DONEXT_TSV=1 "$BENCH" "$f" "$MAX_ROWS" ';')" || echo "    round-trip FAILED on $label" >&2
    IFS=$'\t' read -r _src nrows ncols raw zst _avd avz hd ho _rt <<<"$line"
    optvszst="$(awk "BEGIN{printf \"%+.0f%%\", (1-$ho/$zst)*100}")"
    if [[ "${avz:-0}" != "0" && -n "${avz:-}" ]]; then
        optvsavro="$(awk "BEGIN{printf \"%+.0f%%\", (1-$ho/$avz)*100}")"
        avzcol="$avz"
    else
        optvsavro="—"; avzcol="—"
    fi
    printf '| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n' \
        "$label" "$nrows" "$ncols" "$raw" "$zst" "$avzcol" "$hd" "$ho" "$optvszst" "$optvsavro" >> "$OUT"
done

echo >&2
cat "$OUT"
echo >&2
echo "Combined table written to $OUT" >&2
