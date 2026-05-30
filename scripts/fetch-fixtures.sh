#!/usr/bin/env bash
#
# Fetch the optional benchmark / example fixtures. None of these are committed
# (they are gitignored: *.parquet, *.he), so this script lets anyone reproduce
# the ClickBench numbers and run the Arrow examples against real data.
#
#   hits_1.parquet  ClickBench partition #1 (~1M rows × 105 cols), public.
#   hits_1.he       the above converted to Helium — used by the arrow examples
#                   and by HELIUM_PARQUET_PATH-gated tests / reports.
#
# The DoNext 5G/4G real-data set is NOT fetched here (it is ~4.5 GB, CC BY 4.0).
# See scripts/donext_benchmark.sh and the DOI printed at the end.
#
# Usage:
#   scripts/fetch-fixtures.sh
#
# Requires: curl. The .he conversion additionally needs a release `helium`
# binary (cargo build --release --features cli); if absent it is skipped with
# instructions.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARQUET_URL="https://datasets.clickhouse.com/hits_compatible/athena_partitioned/hits_1.parquet"
PARQUET="$ROOT/hits_1.parquet"
HE="$ROOT/hits_1.he"

# --- 1. hits_1.parquet (ClickBench, public) ---
if [[ -f "$PARQUET" ]]; then
    echo "✓ hits_1.parquet already present"
else
    echo "Downloading ClickBench hits_1.parquet (~public dataset, large)…"
    echo "  $PARQUET_URL"
    curl -fL --progress-bar -o "$PARQUET" "$PARQUET_URL"
    echo "✓ wrote $PARQUET"
fi

# --- 2. hits_1.he (convert via the CLI) ---
HELIUM="${HELIUM_BIN:-$ROOT/target/release/helium}"
if [[ -f "$HE" ]]; then
    echo "✓ hits_1.he already present"
elif [[ -x "$HELIUM" ]]; then
    echo "Converting hits_1.parquet → hits_1.he…"
    "$HELIUM" convert "$PARQUET" -o "$HE"
    echo "✓ wrote $HE"
else
    echo "• helium binary not found at $HELIUM — skipping .he conversion."
    echo "  Build it then re-run:  cargo build --release --features cli"
    echo "  (or set HELIUM_BIN=/path/to/helium)"
fi

echo
echo "ClickBench-gated tests/reports can now use:"
echo "  HELIUM_PARQUET_PATH=$PARQUET cargo test --release -- --nocapture"
echo
echo "Second, optional fixture — DoNext 5G/4G real-data benchmark (NOT fetched"
echo "here; ~4.5 GB, CC BY 4.0):"
echo "  https://doi.org/10.17877/TUDODATA-2026-T6MYPO"
echo "  then: scripts/donext_benchmark.sh /path/to/DoNext/H-Bahn/cell_data.csv 100000 ';'"
