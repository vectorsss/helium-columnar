#!/usr/bin/env python3
"""DoNext format comparison — Helium vs Parquet vs ORC vs raw-zstd.

For every DoNext CSV file, take the first N rows and write that same slice as:
  - Helium   (.he, optimizer-chosen encodings)
  - Parquet  (pyarrow, zstd level 3)
  - ORC      (pyarrow, zstd)
  - raw zstd (csv.zst, zstd level 3)  -- the "no columnar layout" baseline

All columnar formats use a single row-group / stripe covering the whole N-row
slice, so the stripe size is identical across the three. Reports per-file
compression ratio (raw_csv / encoded) and Helium's size advantage vs Parquet
and ORC.

Caveats:
  - All three read the *same* truncated CSV, but each infers column types its
    own way (pyarrow.csv for Parquet/ORC, helium for Helium); this is the
    realistic "ingest this CSV into format X" comparison.
  - pyarrow's ORC writer does not expose a zstd compression *level*; ORC uses
    Apache ORC's default zstd level (not necessarily 3). Parquet and Helium are
    pinned to level 3.

Usage:
  scripts/donext_format_comparison.py <DoNext-dir> [N_rows=100000]
"""
import os
import sys
import glob
import shutil
import subprocess
import tempfile

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import pyarrow.orc as paorc


def normalize_null_columns(table):
    """A column that is entirely null in the slice infers as Arrow `null` type,
    which Parquet/ORC cannot write. Cast such columns to string (still all-null),
    so every format sees the same all-null column."""
    cols, names = [], []
    for i, field in enumerate(table.schema):
        col = table.column(i)
        if pa.types.is_null(field.type):
            col = col.cast(pa.string())
        cols.append(col)
        names.append(field.name)
    return pa.table(cols, names=names)

HELIUM = os.environ.get(
    "HELIUM_BIN",
    os.path.join(os.path.dirname(__file__), "..", "target", "release", "helium"),
)


def sniff_delimiter(path):
    with open(path, "r", errors="replace") as f:
        head = f.readline()
    return ";" if head.count(";") > head.count(",") else ","


def head_rows(src, dst, n):
    """Copy header + first n data rows of `src` into `dst`. Returns row count."""
    rows = 0
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        for i, line in enumerate(fi):
            fo.write(line)
            if i >= 1:
                rows += 1
            if i >= n:  # header (i=0) + n data rows
                break
    return rows


def size(path):
    return os.path.getsize(path)


def helium_default_size(csv, delim, work):
    """Helium with fixed default encodings (`convert`, no optimizer)."""
    he = os.path.join(work, "default.he")
    subprocess.run(
        [HELIUM, "convert", csv, "-o", he, "--delimiter", delim],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return size(he)


def helium_opt_size(csv, delim, work):
    """Helium with optimizer-chosen per-column encodings."""
    schema = os.path.join(work, "opt.json")
    he = os.path.join(work, "opt.he")
    subprocess.run(
        [HELIUM, "optimize-schema", csv, "--delimiter", delim, "--out", schema],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        [HELIUM, "convert", csv, "-o", he, "--schema", schema, "--delimiter", delim],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return size(he)


def parquet_size(table, n, work):
    p = os.path.join(work, "out.parquet")
    pq.write_table(table, p, compression="zstd", compression_level=3,
                   row_group_size=max(n, 1))
    return size(p)


def orc_size(table, work):
    p = os.path.join(work, "out.orc")
    # Single stripe: make stripe_size large enough to hold the whole slice.
    paorc.write_table(table, p, compression="ZSTD",
                      stripe_size=512 * 1024 * 1024)
    return size(p)


def zstd_size(csv, work):
    z = os.path.join(work, "data.csv.zst")
    with open(z, "wb") as out:
        subprocess.run(["zstd", "-3", "-q", "-c", csv], check=True, stdout=out)
    return size(z)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    root = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 100_000

    files = sorted(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True))
    if not files:
        print(f"no CSV files under {root}")
        sys.exit(1)

    print(f"# DoNext format comparison — first {n:,} rows/file, zstd level 3, single stripe\n")
    print("| file | rows | raw CSV | csv.zst | he-default | he-opt | parquet | orc | "
          "opt vs parquet | opt vs orc |")
    print("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|")

    agg = {"vp": [], "vo": [], "gain": []}
    for f in files:
        delim = sniff_delimiter(f)
        work = tempfile.mkdtemp(prefix="donext_cmp_")
        try:
            slice_csv = os.path.join(work, "slice.csv")
            rows = head_rows(f, slice_csv, n)
            raw = size(slice_csv)
            table = pacsv.read_csv(
                slice_csv,
                parse_options=pacsv.ParseOptions(delimiter=delim),
            )
            table = normalize_null_columns(table)
            zst = zstd_size(slice_csv, work)
            hd = helium_default_size(slice_csv, delim, work)
            ho = helium_opt_size(slice_csv, delim, work)
            pqs = parquet_size(table, rows, work)
            orcs = orc_size(table, work)

            vp = (pqs - ho) / pqs * 100.0    # helium-opt % smaller than parquet
            vo = (orcs - ho) / orcs * 100.0
            gain = (hd - ho) / hd * 100.0    # optimizer gain (opt vs default)
            agg["vp"].append(vp)
            agg["vo"].append(vo)
            agg["gain"].append(gain)

            name = os.path.relpath(f, root)
            print(f"| {name} | {rows:,} | {raw/1e6:.1f}M | {zst/1e6:.2f}M | "
                  f"{hd/1e6:.2f}M | {ho/1e6:.2f}M | {pqs/1e6:.2f}M | {orcs/1e6:.2f}M | "
                  f"{vp:+.1f}% | {vo:+.1f}% |")
        except Exception as e:
            print(f"| {os.path.relpath(f, root)} | — | ERROR: {e} |")
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def med(xs):
        xs = sorted(xs)
        return xs[len(xs) // 2] if xs else 0.0

    if agg["vp"]:
        print(f"\n**Helium-opt vs Parquet**: median {med(agg['vp']):+.1f}% smaller "
              f"(range {min(agg['vp']):+.1f}% … {max(agg['vp']):+.1f}%)")
        print(f"**Helium-opt vs ORC**: median {med(agg['vo']):+.1f}% smaller "
              f"(range {min(agg['vo']):+.1f}% … {max(agg['vo']):+.1f}%)")
        print(f"**Optimizer gain (opt vs default)**: median {med(agg['gain']):+.1f}% smaller "
              f"(range {min(agg['gain']):+.1f}% … {max(agg['gain']):+.1f}%)")
        print("\n_(positive = the left side is that much smaller; ORC zstd level is "
              "pyarrow's default, not pinned to 3.)_")


if __name__ == "__main__":
    main()
