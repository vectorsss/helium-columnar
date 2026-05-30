# Helium

> Published on crates.io as **`helium-columnar`** (the name `helium` was
> taken); you still `use helium` in code and the CLI is still `helium`.

A columnar compression framework in Rust. Strongly typed coder pipelines,
JSON-driven schemas, a versioned `.he` file format with column pruning, a
DataFusion `TableProvider` so any `.he` file is queryable via SQL, and a
built-in bench suite.

**On ClickBench (100 k rows × 105 cols)**: a Helium catalog-mode (v6)
optimized file is **7.46 MB** vs Parquet+zstd's **8.93 MB** (~**16% smaller**)
and beats pure-zstd-on-raw-bytes (7.56 MB) by ~1%. **`SELECT count(*)`
answers in 28 ms** (metadata-only) and stripe-level filter pushdown skips
non-matching stripes without reading them. See
[`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) for the full breakdown
including 10 k / 1 M scaling, pure zstd / lz4 baselines, and
SQLite-vs-Helium query latency at small scale.

> **Status:** pre-1.0 (0.x). The API and on-disk `.he` format may change
> between minor versions; there is no on-disk stability promise yet —
> regenerate `.he` files from source when upgrading.

---

## 5-minute quickstart — `.csv` → `.he` → SQL

```bash
# 1. Install the CLI from source (clone the repo first).
cargo install --path . --features cli,datafusion

# 2. Convert a CSV / Parquet / JSON / Avro file to .he.
#    --stripe-rows 10000 splits into 10k-row stripes so SQL filter pushdown
#    can skip irrelevant stripes at query time.
helium convert events.csv -o events.he --stripe-rows 10000

# 2b. Slice out just the columns you need into a new file (zero-copy).
helium slice events.he -o subset.he --columns ts,user_id,event_type

# 3. Inspect what you got.
helium stats events.he --no-values
# File:     events.he
# Format:   v5
# Size:     2,398,712 bytes total
#   Schema header: 932 bytes
#   Body:          2,394,288 bytes
#   Footer:        3,492 bytes
# Stripes:  100
# Rows:     1,000,000
# ...

# 4. Run SQL.
helium sql "SELECT count(*) FROM events" events.he
# +----------+
# | count(*) |
# +----------+
# | 1000000  |
# +----------+

helium sql "SELECT user_id, count(*) AS hits
            FROM events
            WHERE event_type = 'click'
            GROUP BY user_id
            ORDER BY hits DESC
            LIMIT 5" events.he
```

That's the end-to-end path. A few things worth knowing:

- **Identifiers preserve case** by default (`SELECT WatchID`, not `watchid`)
  — overrides DataFusion's SQL-standard lowercasing because real `.he` files
  imported from Parquet typically carry PascalCase column names.
- **Pruning kicks in for free** when stripes are present: `WHERE EventTime >
  X` reads only the stripes whose `[min, max]` overlaps the predicate.
  `WHERE col = 'foo'` likewise consults a per-stripe DistinctSet (≤ 256
  cardinality) or Bloom filter.
- **`COUNT(*)` and `MIN` / `MAX` are metadata-only** — file-level
  `Statistics` is exposed to DataFusion's optimizer, so those queries
  constant-fold (no scan, ~28 ms).
- **Library use**: `helium-columnar = { version = "0.2", features = ["arrow",
  "datafusion"] }` then (as `use helium::...`) build a
  `HeliumTableProvider::try_new(path)?` and register it in your own
  `SessionContext`. See `examples/datafusion_smoke.rs`.
- **Standalone codec API** for users who don't want the file format: `let
  bytes = helium::compress(ColumnData::I64(values))?;` followed by
  `helium::decompress(&bytes)?`. Self-describing — `decompress` reads the
  embedded type tag + pipeline. See the `codec` module.
- **Performance numbers** (compression ratios, query latency, memory) are
  in [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md), with reproduction commands
  for each.

---

## Table of contents

1. [When NOT to use Helium](#when-not-to-use-helium)
2. [Quick start](#quick-start)
3. [Core concepts](#core-concepts)
4. [Using the pipeline API (low level)](#using-the-pipeline-api-low-level)
5. [Using the file format (high level)](#using-the-file-format-high-level)
6. [Coder catalog](#coder-catalog)
7. [Logical types catalog](#logical-types-catalog)
8. [Recipes by column shape](#recipes-by-column-shape)
9. [Multi-stripe writes and column pruning](#multi-stripe-writes-and-column-pruning)
10. [Column projection (slice)](#column-projection-slice)
11. [Error model](#error-model)
12. [Extending: custom coders](#extending-custom-coders)
13. [Benchmarking](#benchmarking)
14. [File format on disk](#file-format-on-disk)

---

## When NOT to use Helium

Helium's compression wins come from cross-row columnar encoding amortizing
fixed per-column framing overhead (footer index entries, zstd frame
headers, pipeline stage prefixes, schema JSON in the file header). That
overhead is constant per stripe; it only pays off when the stripe holds
enough rows.

**Rule of thumb**:

- **≥ 1k rows / stripe** — reliable wins on structured data. This is
  Helium's sweet spot.
- **< 256 rows / stripe** — Helium's per-column overhead can exceed the
  data itself and you lose to raw zstd on the concatenated byte stream.

This is a **framing-level** issue, not a coder-choice issue. Adding new
coders does not fix it. Concrete anti-pattern observed in production:
protobuf messages, ~32 per batch — Helium's shredded column overhead
exceeds zstd-direct-on-wire on the same payload.

**For small-batch structured records** (protobuf / ASN.1 / Avro at
< 1k rows / batch) use **zstd with a pre-trained dictionary** instead.
Pre-training amortizes structural redundancy across batches without
per-batch columnar framing.

For the underlying analysis (per-column fixed-overhead breakdown
showing why the threshold lands where it does) see
[`docs/PERFORMANCE.md`](docs/PERFORMANCE.md).

---

## Logical types at a glance

A schema column has a **logical type** that decomposes into one or more
physical columns, each with its own coder pipeline. The recommended
vocabulary:

- **Scalars** — `Primitive { data_type }`, `Utf8`, `Binary`.
- **Recursive / nested** — `Struct { fields }`, `List { inner }`,
  `Map { key, value }`, `Nullable { inner }`, `Union { variants }`. These
  compose arbitrarily (`List<Struct>`, `Nullable<List<T>>`,
  `Map<Utf8, Struct>`, …).
- **Dictionary encoding** — `Dictionary { inner }`: low-cardinality values
  held once in a dictionary column of `inner` type plus a `U32` index per
  row. `inner` can be any type (mirrors Arrow's `DictionaryArray<value>`).
- **Semantic** — `Decimal128`, `Date` (days / millis), `Datetime`
  (unit + optional timezone).

A handful of flat legacy variants (`NullablePrim`, `NullableUtf8`,
`NullableBinary`, `ArrayOf`, `ArrayOfUtf8`) predate the recursive forms and
remain in the type system for now; **new schemas should use the recursive
`Nullable` / `List` and `Dictionary` instead**. The Avro / CSV / JSON /
Parquet adapters (under `src/schema/`) already emit the recursive forms.

See [Logical types catalog](#logical-types-catalog) for each type's physical
decomposition and the matching `LogicalColumn` constructor.

---

## Quick start

```toml
# Cargo.toml — the package is `helium-columnar`, imported as `helium`.
[dependencies]
helium-columnar = { version = "0.2", default-features = true }
```

### 30-second example — pipeline only

```rust
use helium::{
    BlockCoder, ColumnData, DataType, Delta, Leb128, NonBlockCoder, Pipeline,
    StageCoder, Zstd,
};

fn nb<T: 'static + NonBlockCoder>(c: T) -> StageCoder {
    StageCoder::NonBlock(Box::new(c))
}
fn blk<T: 'static + BlockCoder>(c: T) -> StageCoder {
    StageCoder::Block(Box::new(c))
}

let pipeline = Pipeline::new(
    DataType::I64,
    vec![
        nb(Delta::new(DataType::I64).unwrap()),
        nb(Leb128::new(DataType::I64).unwrap()),
        blk(Zstd::default()),
    ],
)
.unwrap();

let timestamps: Vec<i64> = (1_700_000_000..1_700_010_000).collect();
let encoded = pipeline.encode(ColumnData::I64(timestamps.clone())).unwrap();
let decoded = pipeline.decode(encoded).unwrap();
assert_eq!(decoded, ColumnData::I64(timestamps));
```

### 60-second example — file I/O

```rust
use std::io::Cursor;
use helium::{
    ColumnData, ColumnSpec, CoderRegistry, CoderSpec, DataType, HeliumReader,
    HeliumWriter, LogicalColumn, Schema,
};

let schema = Schema::new(vec![
    ColumnSpec::primitive(
        "ts",
        DataType::I64,
        vec![
            CoderSpec::new("delta"),
            CoderSpec::new("leb128"),
            CoderSpec::new("zstd").with_param("level", 5),
        ],
    ),
    ColumnSpec::utf8(
        "name",
        vec![CoderSpec::new("delta"), CoderSpec::new("leb128"), CoderSpec::new("zstd")],
        vec![CoderSpec::new("zstd")],
    ),
]);

let registry = CoderRegistry::default();
let mut writer = HeliumWriter::new(Cursor::new(Vec::new()), schema, &registry).unwrap();
writer.write_column(
    "ts",
    LogicalColumn::Primitive(ColumnData::I64(vec![1, 2, 3])),
).unwrap();
writer.write_column(
    "name",
    LogicalColumn::Utf8(vec!["a".into(), "b".into(), "c".into()]),
).unwrap();
let bytes = writer.finish().unwrap().into_inner();

let mut reader = HeliumReader::new(Cursor::new(bytes), &registry).unwrap();
let name = reader.read_column("name").unwrap();
assert_eq!(name, LogicalColumn::Utf8(vec!["a".into(), "b".into(), "c".into()]));
```

---

## Core concepts

### Coder

A `Coder` is an encoder + decoder pair for one transformation. Every coder is
either:

- `NonBlockCoder` — can emit output as soon as it sees each input value
  (delta, varint, RLE, bit-packing with fixed width, Gorilla XOR)
- `BlockCoder` — needs the whole buffer before emitting output (zstd, lz4,
  pcodec, bit-packing with auto width, deltamin, Elias-Fano)

This split is the framework's central abstraction. See design §2.1.

### Pipeline

A `Pipeline` is an ordered list of coders applied to one physical column.
**Non-block coders must come before block coders** — the `Pipeline::new`
constructor rejects any other order.

```rust
Pipeline::new(
    DataType::I64,
    vec![
        nb(Delta::new(DataType::I64)?),      // non-block
        nb(Leb128::new(DataType::I64)?),     // non-block
        blk(Zstd::default()),                // block
    ],
)?
```

Encoding runs left-to-right, decoding runs right-to-left. The `Pipeline`
validates both the ordering and the type chain (each stage's output type must
match the next stage's accepted input type).

### DataType and ColumnData

`DataType` enumerates the physical types the framework understands:

| kind | variants |
|---|---|
| signed integer | `I8`, `I16`, `I32`, `I64` |
| unsigned integer | `U8`, `U16`, `U32`, `U64` |
| float | `F32`, `F64` |
| opaque | `Bytes` (post-encoding byte buffer, not a user-level column type) |

`ColumnData` carries the actual values: `ColumnData::I64(Vec<i64>)`,
`ColumnData::F32(Vec<f32>)`, etc. It is what flows between pipeline stages.

### Schema, ColumnSpec, LogicalColumn

At the file level, a `Schema` is a list of `ColumnSpec`s. Each column has:

- a **name** (string, unique within the schema)
- a **logical type** (`LogicalType`) — what users think of as the column
  shape: primitive, string, array, nullable, dictionary-encoded
- an **encodings** list — one pipeline (as a `Vec<CoderSpec>`) per *physical
  column* the logical type decomposes into

A user writes a whole column at once as a `LogicalColumn`. For composite
types (strings, arrays, nullable, dict) the writer internally splits the
value into multiple physical columns; each gets its own pipeline.

### CoderRegistry

Maps stable string IDs to factories. `CoderRegistry::default()` / `::with_builtins()`
registers every coder in this crate:

```
delta        leb128        rle          deltamin     bitpack_fixed {width: u32}
bitpack_auto zstd {level}  lz4          pcodec {level}
gorilla      elias_fano    delta_of_delta
```

Factories auto-specialize from the column's physical type — writing
`{"id": "delta"}` in a schema picks the right `Delta::new(DataType::*)`
based on the column's declared type.

---

## Using the pipeline API (low level)

Use this when you need direct control over encode/decode for a single
column, without the file format layer.

### Constructing a pipeline

```rust
use helium::{
    BlockCoder, DataType, DeltaOfDelta, Leb128, NonBlockCoder, Pipeline,
    StageCoder, Zstd,
};

fn nb<T: 'static + NonBlockCoder>(c: T) -> StageCoder { StageCoder::NonBlock(Box::new(c)) }
fn blk<T: 'static + BlockCoder>(c: T) -> StageCoder { StageCoder::Block(Box::new(c)) }

let pipeline = Pipeline::new(
    DataType::I64,
    vec![
        nb(DeltaOfDelta::new(DataType::I64)?),
        nb(Leb128::new(DataType::I64)?),
        blk(Zstd::new(5)),
    ],
)?;
```

### Encoding and decoding

```rust
let xs: Vec<i64> = (0..10_000).map(|i| 1_700_000_000 + i * 30).collect();

let encoded = pipeline.encode(ColumnData::I64(xs.clone()))?;
//  ^ ColumnData::Bytes(...) for pipelines that terminate in Bytes

let decoded = pipeline.decode(encoded)?;
assert_eq!(decoded, ColumnData::I64(xs));
```

The `encode` return type is always the pipeline's declared output type. For
pipelines used with the file format, the output must always be
`DataType::Bytes` (otherwise `HeliumWriter::new` rejects the schema).

### Invariants the pipeline enforces

- Non-block stages precede block stages.
- Each stage's `produced_output_type()` matches the next stage's
  `accepted_input_type()`.
- Runtime `ColumnData` passed to `encode` matches the pipeline's declared
  input type.

Violations surface as targeted `HeliumError` variants (see below).

---

## Using the file format (high level)

### Schema construction — ergonomic helpers

`ColumnSpec` has constructors for every logical shape; no need to build the
`encodings` list manually:

```rust
use helium::{ColumnSpec, CoderSpec, DataType, LogicalType, Schema};

let schema = Schema::new(vec![
    // Simple primitive column.
    ColumnSpec::primitive(
        "ts",
        DataType::I64,
        vec![
            CoderSpec::new("delta"),
            CoderSpec::new("leb128"),
            CoderSpec::new("zstd").with_param("level", 5),
        ],
    ),
    // UTF-8 string — two physical pipelines (offsets, data).
    ColumnSpec::utf8(
        "name",
        vec![CoderSpec::new("delta"), CoderSpec::new("leb128"), CoderSpec::new("zstd")], // offsets
        vec![CoderSpec::new("zstd")],                                                    // data
    ),
    // Nullable float — Nullable<Primitive>: pipelines [present, values].
    ColumnSpec::nullable(
        "weight",
        LogicalType::Primitive { data_type: DataType::F64 },
        vec![
            vec![CoderSpec::new("leb128"), CoderSpec::new("zstd")],   // present (U8)
            vec![CoderSpec::new("gorilla"), CoderSpec::new("zstd")],  // values
        ],
    ),
    // List of i32 — List<Primitive>: pipelines [offsets, values].
    ColumnSpec::list(
        "tags",
        LogicalType::Primitive { data_type: DataType::I32 },
        vec![
            vec![CoderSpec::new("delta"), CoderSpec::new("leb128"), CoderSpec::new("zstd")], // offsets
            vec![CoderSpec::new("pcodec")],                                                  // values
        ],
    ),
    // Dictionary-encoded strings — Dictionary<Utf8>: [dict offsets, dict data, indices].
    ColumnSpec::new(
        "status",
        LogicalType::Dictionary { inner: Box::new(LogicalType::Utf8) },
        vec![
            vec![CoderSpec::new("leb128"), CoderSpec::new("zstd")],       // dict offsets
            vec![CoderSpec::new("zstd")],                                 // dict data
            vec![CoderSpec::new("bitpack_auto"), CoderSpec::new("zstd")], // indices
        ],
    ),
]);
```

Schemas serialize to JSON — this is what the writer embeds in the file:

```rust
let json = schema.to_json()?;     // Vec<u8>
let parsed = Schema::from_json(&json)?;
```

### Writing

```rust
use std::io::Cursor;
use helium::{CoderRegistry, ColumnData, HeliumWriter, LogicalColumn};

let registry = CoderRegistry::default();
let mut writer = HeliumWriter::new(Cursor::new(Vec::new()), schema, &registry)?;

writer.write_column("ts", LogicalColumn::Primitive(ColumnData::I64(ts_values)))?;
writer.write_column("name", LogicalColumn::Utf8(names))?;
writer.write_column(
    "weight",
    LogicalColumn::Nullable {
        present: present_mask,                                            // Vec<bool>
        value: Box::new(LogicalColumn::Primitive(ColumnData::F64(vals))), // compacted: present rows only
    },
)?;
writer.write_column(
    "tags",
    LogicalColumn::List {
        offsets: tag_offsets,                                                   // Vec<u32>, length row_count + 1
        values: Box::new(LogicalColumn::Primitive(ColumnData::I32(tag_values))),
    },
)?;
writer.write_column(
    "status",
    LogicalColumn::dict_encode_utf8(raw_status_strings),
)?;

let output = writer.finish()?; // returns the underlying writer (Cursor, File, …)
```

Invariants the writer enforces:

- every schema column must be written exactly once per stripe before `finish`
- all columns in a stripe must report the same row count
- every physical pipeline must terminate in `DataType::Bytes`
- `LogicalColumn` shape must match the schema's `LogicalType`

### Reading

```rust
use helium::HeliumReader;

let mut reader = HeliumReader::new(file, &registry)?;
println!("rows: {}", reader.row_count());

// Read one column — other columns are never touched.
let ts = reader.read_column("ts")?;

// Read everything.
let all = reader.read_all()?;  // HashMap<String, LogicalColumn>
```

For multi-stripe files (see below), `read_column` concatenates across
stripes automatically for primitive/utf8/binary/array/nullable types. For
dict columns it errors — use `read_column_at_stripe(name, idx)` instead.

---

## Coder catalog

Reference for `CoderRegistry::default()` — every ID here works in a
`CoderSpec`.

| id | kind | accepts | produces | params | notes |
|---|---|---|---|---|---|
| `delta` | non-block | integer | same | — | `out[i] = in[i] - in[i-1]`, wrapping |
| `delta_of_delta` | non-block | integer | same | — | second differences; all-zero on uniform series |
| `leb128` | non-block | integer | `Bytes` | — | zigzag for signed; LEB128 for unsigned |
| `rle` | non-block | integer | same | — | interleaved `[value, count]`; overflow errors |
| `deltamin` | block | integer | same | — | frame-of-reference: subtract `min`, prepend it |
| `bitpack_fixed` | non-block | integer | `Bytes` | `width: u32` | caller promises all values fit in `width` bits |
| `bitpack_auto` | block | integer | `Bytes` | — | scans for max, derives width, prepends it |
| `zstd` | block | `Bytes` | `Bytes` | `level: i32` (default 3) | general-purpose compressor |
| `lz4` | block | `Bytes` | `Bytes` | — | faster than zstd, lower ratio |
| `pcodec` | block | integer (`i8`..`i64`, `u8`..`u64`) / `f32` / `f64` | `Bytes` | `level: usize` (optional) | typed numeric compressor (`pco`) |
| `gorilla` | non-block | `F32 / F64` | `Bytes` | — | XOR-based float coder (Facebook TSDB) |
| `elias_fano` | block | `U32 / U64` | `Bytes` | — | strictly-increasing sequences only |

### Choosing a coder — rules of thumb

- **Timestamps, monotone IDs**: `delta → leb128 → zstd` or `delta → pcodec`.
- **Uniformly sampled timestamps**: `delta_of_delta → leb128 → zstd` — collapses to almost nothing.
- **Measurements with tight range around non-zero center**: `deltamin → bitpack_auto → zstd`.
- **Floats / time-series metrics**: `gorilla → zstd` or `pcodec`.
- **Low-cardinality strings/enums**: dict-encode (`LogicalType::Dictionary { inner: Utf8 }`) then `bitpack_auto → zstd` on the indices.
- **Low-cardinality integers**: `rle → leb128 → zstd` or `bitpack_auto → zstd`.
- **Inverted-index postings (sorted unique)**: `elias_fano`.
- **General bytes / text blobs**: `zstd` (default) or `lz4` (if speed matters more than ratio).

---

## Logical types catalog

Every `LogicalType` decomposes into a fixed list of physical columns. The
writer needs exactly one pipeline per physical column, in the listed order.

| logical type | physical decomposition |
|---|---|
| `Primitive { data_type }` | `values: T` |
| `Utf8` | `offsets: U32`, `data: Bytes` |
| `Binary` | `offsets: U32`, `data: Bytes` |
| `Struct { fields }` | each field's leaves, in order (no leaf of its own) |
| `List { inner }` | `offsets: U32`, then `inner`'s leaves |
| `Map { key, value }` | `offsets: U32`, then `key`'s leaves, then `value`'s leaves |
| `Nullable { inner }` | `present: U8`, then `inner`'s leaves (values compacted to present rows) |
| `Dictionary { inner }` | `inner`'s leaves (the dictionary), then `indices: U32` |
| `Decimal128` | `high: I64`, `low: I64` |
| `Date { unit }` / `Datetime { unit, tz }` | `values: I32` (date days) or `I64` |

The corresponding `LogicalColumn` variant for the common cases:

```rust
LogicalColumn::Primitive(ColumnData::I64(vec![...]))

LogicalColumn::Utf8(vec!["a".into(), "b".into()])

LogicalColumn::Binary(vec![vec![0xff, 0xfe], vec![0x00]])

LogicalColumn::List { offsets: vec![0, 2, 5], values: Box::new(LogicalColumn::Primitive(ColumnData::I32(vec![1, 2, 3, 4, 5]))) }
//  row 0 = [1, 2]    (offsets[0]..offsets[1])
//  row 1 = [3, 4, 5] (offsets[1]..offsets[2])

LogicalColumn::Struct { fields: vec![
    ("id".into(),   LogicalColumn::Primitive(ColumnData::I64(vec![1, 2]))),
    ("name".into(), LogicalColumn::Utf8(vec!["a".into(), "b".into()])),
] }

LogicalColumn::Nullable {
    present: vec![true, false, true],          // length = row_count
    value: Box::new(LogicalColumn::Primitive(ColumnData::F64(vec![1.0, 2.0]))),  // compacted: present rows only
}

LogicalColumn::Dictionary {
    dictionary: Box::new(LogicalColumn::Utf8(vec!["pending".into(), "done".into()])),
    indices: vec![0, 1, 0, 0, 1],
}
// convenience constructors (return a `Dictionary { inner }`):
let dict_col = LogicalColumn::dict_encode_primitive(ColumnData::I64(raw_ints))?;
let dict_col = LogicalColumn::dict_encode_utf8(vec!["pending".into(), "done".into(), "pending".into()]);
```

Offset validation: offsets arrays are length `row_count + 1` with
`offsets[0] == 0`, monotonic non-decreasing, and `offsets[N] == total`. The
writer rejects malformed offsets with a descriptive error.

---

## Recipes by column shape

### Timestamps (uniformly sampled)

```rust
ColumnSpec::primitive(
    "ts",
    DataType::I64,
    vec![
        CoderSpec::new("delta_of_delta"),
        CoderSpec::new("leb128"),
        CoderSpec::new("zstd"),
    ],
)
```

### Timestamps (jittered)

```rust
ColumnSpec::primitive(
    "ts",
    DataType::I64,
    vec![
        CoderSpec::new("delta"),
        CoderSpec::new("leb128"),
        CoderSpec::new("zstd"),
    ],
)
```

### Float gauge metrics

```rust
// Option A: Gorilla (good for drifting values that share mantissa bits)
ColumnSpec::primitive(
    "temp_c",
    DataType::F64,
    vec![CoderSpec::new("gorilla"), CoderSpec::new("zstd")],
)

// Option B: pcodec (often better ratio on typed numeric data)
ColumnSpec::primitive("temp_c", DataType::F64, vec![CoderSpec::new("pcodec")])
```

### Low-cardinality strings (enums, statuses)

```rust
ColumnSpec::new(
    "status",
    LogicalType::Dictionary { inner: Box::new(LogicalType::Utf8) },
    vec![
        vec![CoderSpec::new("leb128"), CoderSpec::new("zstd")],       // dict offsets
        vec![CoderSpec::new("zstd")],                                 // dict data
        vec![CoderSpec::new("bitpack_auto"), CoderSpec::new("zstd")], // indices
    ],
)

// Writer side — dict_encode_utf8 builds a `Dictionary { inner: Utf8 }`:
let col = LogicalColumn::dict_encode_utf8(raw_statuses);
writer.write_column("status", col)?;
```

### Nullable measurements

```rust
ColumnSpec::nullable(
    "reading",
    LogicalType::Primitive { data_type: DataType::F64 },
    vec![
        // present bitmap (U8): leb128 → compress
        vec![CoderSpec::new("leb128"), CoderSpec::new("zstd")],
        // values (compacted to present rows)
        vec![CoderSpec::new("gorilla"), CoderSpec::new("zstd")],
    ],
)
```

### Inverted-index postings

```rust
ColumnSpec::primitive(
    "doc_ids",
    DataType::U32,
    vec![CoderSpec::new("elias_fano")],
)
// writer input must be strictly increasing.
```

### Lists of small integers

```rust
ColumnSpec::list(
    "event_codes",
    LogicalType::Primitive { data_type: DataType::U16 },
    vec![
        vec![CoderSpec::new("delta"), CoderSpec::new("leb128"), CoderSpec::new("zstd")], // offsets
        vec![CoderSpec::new("bitpack_auto"), CoderSpec::new("zstd")],                    // values
    ],
)
```

### General text (high cardinality)

```rust
ColumnSpec::utf8(
    "log_message",
    vec![CoderSpec::new("delta"), CoderSpec::new("leb128"), CoderSpec::new("zstd")], // offsets
    vec![CoderSpec::new("zstd").with_param("level", 6)],                             // data
)
```

---

## Multi-stripe writes and column pruning

A *stripe* is a physical row group inside a `.he` file. Stripes are useful
for bounding encoder memory, enabling range reads, and allowing parallel
decoders. The writer takes explicit stripe boundaries via `finish_stripe()`.

### Writing multiple stripes

```rust
let mut writer = HeliumWriter::new(sink, schema, &registry)?;

// Stripe 1
writer.write_column("ts",   col1_ts)?;
writer.write_column("name", col1_name)?;
writer.finish_stripe()?;

// Stripe 2
writer.write_column("ts",   col2_ts)?;
writer.write_column("name", col2_name)?;
// finish() will close the last stripe automatically.

writer.finish()?;
```

### Reading across stripes

```rust
let mut reader = HeliumReader::new(file, &registry)?;
println!("{} rows across {} stripes", reader.row_count(), reader.stripe_count());

// Concatenated read (all stripes merged into one logical column):
let all_ts = reader.read_column("ts")?;

// Per-stripe read:
for s in 0..reader.stripe_count() {
    let piece = reader.read_column_at_stripe("ts", s)?;
    process(piece);
}
```

**Column pruning**: `read_column` only touches the bytes for the requested
logical column. Header and footer are always read on open, but the body
bytes of unselected columns are never loaded. Verified by the test suite's
counted-reader check.

**Dictionary columns in multi-stripe**: `read_column("dict_col")` errors,
because different stripes may have different dictionaries. Use
`read_column_at_stripe` and handle each stripe's dictionary explicitly.

---

## Column projection (slice)

To write a new `.he` file containing only a subset of an existing file's
columns, use `HeliumReader::project_to` (or the `helium slice` CLI). This is
a **zero-copy** operation: the already-encoded leaf bytes are copied
verbatim and each leaf's stored CRC, min/max, null-count and containment
filter are reused from the footer — no coder runs, no stats are recomputed.
Stripe boundaries are preserved, every logical type is supported (including
`Dictionary`), and the output is a fresh self-contained v5 file.

```rust
let mut reader = HeliumReader::new(File::open("events.he")?, &registry)?;
let out = File::create("subset.he")?;
// columns are written in the given order; missing/duplicate names error.
reader.project_to(&["ts", "user_id", "event_type"], out, &registry)?;
```

```bash
helium slice events.he -o subset.he --columns ts,user_id,event_type
#   --catalog ./catalog   # only needed to READ a v6 (catalog-mode) input
```

`Schema::project(&["a", "b"])` builds the subset schema on its own if you
need it without writing a file. (For *row* slicing of an in-memory column,
see `LogicalColumn::slice(start, len)`.)

Each leaf's CRC is re-verified against the footer while copying, so a corrupt
source is caught at slice time rather than propagated.

---

## Error model

Every failure surfaces as `HeliumError`, with specific variants so operators
can isolate *which* column/stage failed. Common variants:

| variant | when |
|---|---|
| `PipelineOrder` | non-block coder placed after a block coder |
| `TypeMismatch` | pipeline stage's accepted input doesn't match previous stage's output |
| `RuntimeType` | `ColumnData` variant passed at encode/decode time doesn't match pipeline |
| `Corrupted { coder, reason }` | decode found malformed input (bad length, CRC mismatch, truncated stream) |
| `CoderFailed { coder, reason }` | encode precondition violated (value out of range, non-sorted input to Elias-Fano, etc.) |
| `UnknownCoder(String)` | schema references a coder ID the registry doesn't know |
| `InvalidParam { coder, param, reason }` | coder parameter in schema is missing, wrong type, or out of range |
| `Schema { column, reason }` | schema validation (duplicate names, shape mismatches, encodings count) |
| `Format(String)` | file-level: bad magic, truncated, malformed footer |
| `Io(std::io::Error)` | underlying writer/reader IO |
| `Json(serde_json::Error)` | schema or footer JSON couldn't be parsed |

Errors carry enough context (column name, stage ID, position) to pinpoint
the problem. Example: a single-byte corruption inside a column's data
produces an error like:

```
coder 'name': stripe 2 physical column CRC32C mismatch: stored 0xa1b2c3d4, computed 0xf00dfeed
```

---

## Extending: custom coders

To add a coder, implement `Coder` + either `NonBlockCoder` or `BlockCoder`,
then register a factory:

```rust
use helium::{
    BlockCoder, Coder, CoderKind, CoderRegistry, CoderSpec, ColumnData,
    DataType, HeliumError, Result, StageCoder,
};

pub struct MyCoder;

impl Coder for MyCoder {
    fn id(&self) -> &'static str { "my_coder" }
    fn kind(&self) -> CoderKind { CoderKind::Block }
    fn accepted_input_type(&self) -> DataType { DataType::Bytes }
    fn produced_output_type(&self) -> DataType { DataType::Bytes }
}

impl BlockCoder for MyCoder {
    fn encode_block(&self, input: &ColumnData) -> Result<ColumnData> {
        let ColumnData::Bytes(src) = input else {
            return Err(HeliumError::RuntimeType {
                coder: self.id().into(),
                expected: DataType::Bytes,
            });
        };
        // ... your compression ...
        Ok(ColumnData::Bytes(compressed))
    }

    fn decode_block(&self, input: &ColumnData) -> Result<ColumnData> {
        // ... your decompression ...
        Ok(ColumnData::Bytes(original))
    }
}

// Register:
let mut registry = CoderRegistry::with_builtins();
registry.register("my_coder", |_spec, _input_type| {
    Ok(StageCoder::Block(Box::new(MyCoder)))
});
```

Guidelines (from design §6):

1. Pick a **stable string ID**. Once any `.he` file is written with it,
   never reuse or repurpose the ID.
2. Decide block vs non-block by the single question: *can you emit output
   for the i-th input using only the first i values?* Yes → non-block; no
   → block. Needing global stats (min, max, dictionary) forces block.
3. If the same algorithm has "parameters pre-declared" and "parameters
   learned from data" variants, give them **different IDs** (e.g.
   `dict_static` vs `dict_learned`). Don't overload a single ID with a
   boolean switch.
4. Write a round-trip test. Test empty, single-element, all-same-value,
   and extreme-range inputs at minimum.

---

## Benchmarking

There are two complementary suites:

### 1. Criterion (`cargo bench`)

Full statistical throughput measurement for individual coders and
pipelines.

```bash
# Quick smoke run (faster, rougher numbers):
cargo bench -- --quick --warm-up-time 1 --measurement-time 2

# Full statistical run:
cargo bench

# Specific group:
cargo bench --bench pipelines -- timestamps
```

- `benches/coders.rs` — per-coder encode throughput on representative typed
  inputs (10k-element datasets).
- `benches/pipelines.rs` — end-to-end pipelines on realistic column shapes
  (jittered timestamps, gauge floats, low-cardinality tags). Prints
  compression ratios to stderr before the criterion statistical phase.

Synthetic datasets live in `benches/datasets.rs` and are deterministic so
runs are comparable across time.

### 2. Compression report (`cargo test --test compression_report`)

Side-by-side comparison against gzip / lz4 / zstd / pcodec on 12 real-shape
datasets at both 10k and 100k rows. Produces three Markdown tables per
size — compression ratio, encode MB/s, decode MB/s — and persists them to
`target/compression-report.md`. Also asserts a floor ratio per dataset, so
any regression in a coder shows up as a test failure.

```bash
# Run in release (realistic timings) with tables printed:
cargo test --test compression_report --release -- --nocapture

# Faster debug build, still prints tables:
cargo test --test compression_report -- --nocapture
```

Datasets cover the main column shapes:

- integers: monotone timestamps, jittered timestamps, narrow-range signal
  measurements (i32), sorted unique u32 (inverted-index shape), random u64
- floats: quantized gauge drift, simulated stock prices, random f64
- strings: low-cardinality enums (log levels), medium cardinality (user
  agents), templated log messages, high-entropy UUIDs

**Correctness**: every cell in the report is round-trip verified — encode
then decode is asserted byte- or value-identical to the original before
the timing loop runs. A broken coder fails the test rather than silently
producing a pretty number.

The report carries **two helium columns**:

- `helium(zstd)` — helium shaping stages with zstd as the final block.
- `helium(pco)` — same shaping stages, but zstd is replaced by pcodec.
  Reveals whether helium's pre-pcodec shaping gives pcodec extra leverage
  versus pcodec alone. `—` where the swap isn't semantically defined.

Representative ratios (Apple Silicon, release, 100k rows, **winner per row
in bold**):

```
Dataset            Raw      gzip   lz4    zstd    pcodec   helium(zstd)  helium(pco)
ts_uniform_i64     781 KB   4.9x   1.9x   6.4x    14286x   28571*        7921x
ts_jittered_i64    781 KB   2.5x   1.5x   2.9x    18.4*    11.9x         18.4*
rsrp_i32           391 KB   3.3x   2.3x   3.5x    5.0*     4.4x          4.5x
ids_sorted_u32     391 KB   2.9x   1.0x   1.5x    11.4*    8.0x          —
random_u64         781 KB   1.0x   1.0x   1.0*    1.0x     0.9x          —
temp_gauge_f64     781 KB   42.7x  19.5x  67.0x   79.1x    139.7*        —
stock_prices_f64   781 KB   1.9x   1.4x   2.0*    1.6x     1.9x          —
random_f64         781 KB   1.1x   1.0x   1.1x    1.2*     1.0x          —
log_levels_utf8    498 KB   18.8x  4.8x   12.6x   —        29.9x         39.0*
user_agents_utf8   4.9 MB   23.1x  11.2x  21.3x   —        58.9x         61.6*
log_messages_utf8  4.7 MB   5.7*   3.6x   5.3x    —        5.4x          —
uuids_utf8         3.5 MB   1.7x   1.0x   1.9x    —        1.9*          —

(* = best in row. Multi-tool ties go to the leftmost entry.)
```

Takeaways:

- **On structured data helium leads**: uniform timestamps (zstd tail shines
  on the near-all-zero post-dod stream), temperature gauges (gorilla+zstd
  crushes), low-cardinality strings (dict + bitpack/pcodec on indices).
- **`helium(pco)` wins on dict-string columns**: swapping bitpack+zstd for
  pcodec on the indices column compresses ~30% more on log levels and ~5%
  more on user agents. **And decodes 2-3× faster** (see decode throughput
  table in the full report).
- **On pcodec's home turf (typed numerics with moderate entropy) pcodec
  alone wins**: jittered timestamps, rsrp, sorted u32 postings. Helium's
  shaping stages don't help pcodec beyond what it already does internally.
- **On fundamentally random data nothing wins** — helium stays within
  noise of the best byte-level compressor, proving there's no hidden
  framework overhead eating the budget.
- **On generic text (log_messages) gzip/zstd beats helium** by a whisker —
  LZ77's sliding-window repetition model outperforms a pure offsets+data
  split when the repetition is in the data bytes themselves.

**Encode/decode speed**: lz4 dominates raw throughput in almost every row.
Helium's encode wins on a few shapes where its pipeline avoids a heavy
generic compressor (e.g. `dod+leb128+zstd` on uniform timestamps, where
the zstd stage sees near-empty input, hits ~3.9 GB/s). Helium(pco) decode
is *significantly* faster than helium(zstd) decode on dict columns
(~3× faster on user_agents). Full numbers live in
`target/compression-report.md`.

Absolute numbers will shift across hardware; the *ordering* is what
matters for coder selection.

---

## File format on disk

`HeliumWriter::new` emits **v5** (the default, self-contained). The opt-in
`HeliumWriter::with_catalog_ref` emits **v6** for catalog-mode writes
(see the `catalog` module). `HeliumReader::new` accepts v5; v6 requires
`HeliumReader::new_with_resolver`.

Formats **v1–v4** (the pre-compression layouts and the uncompressed-footer
catalog format) were removed before 1.0 — Helium makes no on-disk stability
promise at 0.x, so regenerate old files from source. The reader rejects any
unsupported magic with `HeliumError::Format` — never silent corruption.

### v5 layout (default, self-contained)

The schema JSON in the header **and** the footer JSON are both
zstd-compressed. `footer_len` is the compressed length; `footer_crc32c` is
over the compressed bytes (so corruption is caught before zstd
decompression, mirroring how the schema header is integrity-checked).

```
[0..8]          magic = b"HELIUM\x00\x05"
[8..12]         schema_len: u32 LE          (length of the COMPRESSED schema bytes)
[12..12+S]      zstd-compressed schema JSON

body_start = 12 + S

[body_start..]  per-stripe runs of physical-column encoded bytes

[..tail-20]     zstd-compressed footer JSON; decompressed contents:
                {
                  "stripes": [
                    {
                      "row_count": <u64>,
                      "columns": [
                        { "physical": [ { "offset": <u64>, "length": <u64>, "crc32c": <u32> }, ... ] },
                        ...
                      ]
                    },
                    ...
                  ]
                }
[tail-20..-12]  footer_len: u64 LE          (compressed footer length)
[tail-12..-8]   footer_crc32c: u32 LE       (CRC32C of the compressed footer bytes)
[tail-8..]      magic = b"HELIUM\x00\x05"
```

Footer-compression savings on a wide schema (50 columns × 3 stripes):
9.6 KB raw → 0.7 KB on disk (~92%).

### v6 layout (opt-in catalog mode)

Same as v5 except the header carries a 36-byte schema-hash slot instead of
the embedded schema; the footer is zstd-compressed identically to v5.

```
[0..8]          magic = b"HELIUM\x00\x06"
[8..40]         BLAKE3 hash of canonicalized schema JSON (32 bytes raw)
[40..44]        CRC32C of the 32-byte hash (u32 LE)

body_start = 44

[body_start..]  per-stripe runs of physical-column encoded bytes
                (identical to v5)

[..tail-20]     zstd-compressed footer JSON (identical to v5)
[tail-20..-12]  footer_len: u64 LE          (compressed footer length)
[tail-12..-8]   footer_crc32c: u32 LE
[tail-8..]      magic = b"HELIUM\x00\x06"
```

Catalog-mode files do not embed the schema JSON — readers resolve the
32-byte hash via a caller-provided closure (typically a filesystem lookup
in `<catalog-dir>/<hash-hex>.json`). See the `catalog` module. The
schema-slot CRC32C catches single-bit hash corruption with a clearer
"v6 schema-slot CRC mismatch" error than the per-column body CRC would.

> Earlier formats **v1–v4** were removed before 1.0. The reader accepts only
> v5 and v6; any other magic is rejected with `HeliumError::Format`.
> Regenerate old files from source.

### Why compress the schema and footer?

For files with few rows, the schema JSON is a meaningful fraction of the
total file size. For deeply-nested types (Struct/List/Map), the schema
JSON contains lots of repeated tokens (`"kind"`, `"encodings"`, `"id"`,
coder IDs, etc.) — a workload zstd handles very well. Compressing the
schema header trims 70-80% off the schema overhead on representative
shapes.

The footer JSON has the same shape — `"offset"`, `"length"`, `"crc32c"`,
`"physical"`, `"columns"`, `"row_count"` repeat once per physical column
per stripe. On wide schemas the savings are larger still (~90%), with
negligible impact on read latency (one extra zstd decompression at
file-open, after the existing CRC check).

### Integrity on read

On read the reader:

1. Verifies start magic + end magic.
2. Verifies `CRC32C(footer_bytes_on_disk) == footer_crc32c` **before**
   any decompression — detects tampering of the index structure before the
   compressed footer is decoded.
3. For every physical column it reads, verifies `CRC32C(bytes) ==
   stored_crc32c` before handing bytes to the decode pipeline — detects
   single-byte flips in the body.

CRC failures surface as `HeliumError::Corrupted { coder, reason }` where
`coder` is the affected top-level column name and `reason` includes the
dotted leaf path (e.g. `"outer.item.values"`) so operators can pinpoint
the failing leaf even in a deeply nested type.

The schema and footer bytes themselves are NOT separately CRC'd beyond
the on-disk CRC: corruption that survives the on-disk CRC (vanishingly
unlikely under random bit-flips) surfaces as a zstd decode error or
JSON parse error wrapped in `HeliumError::Format`.

---

## Status and further reading

- **[`CHANGELOG.md`](./CHANGELOG.md)** — release notes.
- **[`docs/PERFORMANCE.md`](./docs/PERFORMANCE.md)** — compression ratios,
  query latency, memory profile, file-format overheads, and reproduction
  commands.
- **[`docs/ROADMAP.md`](./docs/ROADMAP.md)** — planned work (SIMD coders,
  parallel query execution, `f16`, object-store backends).
- **[`CLAUDE.md`](./CLAUDE.md)** — guidance for working inside this repo
  (module layout, gotchas, anti-patterns).

Tests worth browsing for more examples:

- `tests/roundtrip.rs` — every coder, every integer width, edge cases
- `tests/file_format.rs` — full file-format round-trips, multi-stripe,
  CRC detection, logical-type coverage
- `tests/projection.rs` — column slice / projection, incl. zero-copy vs
  decode→re-encode equivalence across nested types
