#!/usr/bin/env python3
"""Encode a CSV file as Avro, mirroring a helium-inferred schema.

Used by donext_benchmark.sh to produce the "Avro + zstd" baseline that
models how 5G Measurement Reports are stored in production (Avro
serialization, then the whole blob compressed with zstd).

Column types/nullability come from a helium schema JSON (produced by
`helium infer-schema <csv> --delimiter ...`), so the Avro file carries
the same logical types helium uses — an apples-to-apples comparison.

Requires: fastavro  (pip install fastavro)

Usage:
    python3 csv_to_avro.py <input.csv> <helium_schema.json> <out_prefix> [delimiter]

Writes:
    <out_prefix>_deflate.avro   (Avro container, deflate codec)
    <out_prefix>_null.avro       (Avro container, no codec — compress externally)

Prints one "<path> <bytes>" line per file plus a final "rows <n>".
"""
import sys
import csv
import json
import io

try:
    import fastavro
except ImportError:
    sys.exit("error: fastavro not installed (pip install fastavro)")


def avro_type(lt):
    """Map a helium LogicalType (JSON) to (avro_type, is_nullable)."""
    def base(dt):
        return {
            "i8": "int", "i16": "int", "i32": "int", "i64": "long",
            "u8": "int", "u16": "int", "u32": "long", "u64": "long",
            "f32": "float", "f64": "double",
        }.get(dt, "string")

    kind = lt.get("kind")
    if kind == "primitive":
        return base(lt["data_type"]), False
    if kind == "nullable":
        inner = lt["inner"]
        if inner.get("kind") == "primitive":
            return base(inner["data_type"]), True
        return "string", True
    if kind == "utf8":
        return "string", False
    if kind == "binary":
        return "bytes", False
    # Struct/List/Map/Union/Dict/semantic types — fall back to string.
    return "string", True


def convert(val, at):
    if val == "" or val.lower() in ("null", "na"):
        return None
    if at in ("int", "long"):
        return int(float(val))
    if at in ("float", "double"):
        return float(val)
    if at == "bytes":
        return val.encode()
    return val


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    csv_path, schema_json, out_prefix = sys.argv[1], sys.argv[2], sys.argv[3]
    delim = sys.argv[4] if len(sys.argv) > 4 else ","

    cols = json.load(open(schema_json))["columns"]
    fields, meta = [], []
    for c in cols:
        at, nullable = avro_type(c["logical_type"])
        fields.append({"name": c["name"], "type": (["null", at] if nullable else at)})
        meta.append((c["name"], at, nullable))

    parsed = fastavro.parse_schema(
        {"type": "record", "name": "Row", "fields": fields}
    )

    records = []
    with open(csv_path, newline="") as f:
        r = csv.reader(f, delimiter=delim)
        header = next(r)
        idx = {h: i for i, h in enumerate(header)}
        for row in r:
            rec = {}
            for (name, at, nullable) in meta:
                raw = row[idx[name]] if idx[name] < len(row) else ""
                v = convert(raw, at)
                if v is None and not nullable:
                    v = 0 if at in ("int", "long") else (0.0 if at in ("float", "double") else "")
                rec[name] = v
            records.append(rec)

    for codec in ("deflate", "null"):
        buf = io.BytesIO()
        fastavro.writer(buf, parsed, records, codec=codec)
        data = buf.getvalue()
        path = f"{out_prefix}_{codec}.avro"
        open(path, "wb").write(data)
        print(f"{path} {len(data)}")
    print(f"rows {len(records)}")


if __name__ == "__main__":
    main()
