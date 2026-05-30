# pyhelium

Python bindings for the [Helium](https://github.com/vectorsss/helium-columnar)
columnar compression library.

Two complementary APIs:

- a **numpy** API for numeric arrays and flat string/binary columns
  (`compress` / `decompress`, `write_he` / `read_he`); and
- an **Arrow / pandas** API (`write_table` / `read_table`) that round-trips the
  full Helium type set — nullable, nested (Struct / List), and semantic
  (Date / Datetime / Decimal) columns.

## Installation (development)

```bash
cd python/
python -m venv .venv && source .venv/bin/activate
pip install maturin numpy pytest
pip install pyarrow pandas        # for the Arrow / pandas API
maturin develop
```

## Quick start

### Compress / decompress a numeric array

```python
import numpy as np
import pyhelium

arr = np.arange(100_000, dtype=np.int64)
buf = pyhelium.compress(arr)          # returns bytes (HEC0 format)
out = pyhelium.decompress(buf)        # returns numpy.ndarray
assert (arr == out).all()
```

Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
`f32`, `f64`.

### Write and read a flat .he file (numpy / lists)

```python
import numpy as np
import pyhelium

data = {
    "id":    np.arange(1000, dtype=np.int64),
    "score": np.random.rand(1000).astype(np.float64),
    "label": [f"row-{i}" for i in range(1000)],
}
pyhelium.write_he("events.he", data)

result = pyhelium.read_he("events.he")
print(result["label"][:3])   # ['row-0', 'row-1', 'row-2']
```

`write_he` accepts `numpy.ndarray` (numeric → `Primitive`), `list[str]`
(→ `Utf8`), and `list[bytes]` (→ `Binary`). `read_he` returns numpy arrays /
lists. For nullable, nested, or semantic columns use the Arrow API below;
`read_he` raises `NotImplementedError` and points you to `read_table`.

### Arrow / pandas interop (full type set)

```python
import pyarrow as pa
import pandas as pd
import pyhelium

# Write a pyarrow Table, a RecordBatch, or a pandas DataFrame.
df = pd.DataFrame({
    "id":   pd.array([1, 2, None, 4], dtype="Int64"),                 # nullable
    "when": pd.to_datetime(["2020-01-01", "2020-01-02",
                            "2020-01-03", "2020-01-04"]),             # datetime
    "name": ["alice", "bob", "carol", "dave"],
})
pyhelium.write_table("people.he", df)

table = pyhelium.read_table("people.he")     # -> pyarrow.Table
print(table.to_pandas())
```

Nullable, `Struct`, `List`, `Date32`/`Date64`, `Timestamp`, and `Decimal128`
columns all round-trip. Arrays cross the Rust/Python boundary via the Arrow C
Data Interface (no IPC re-serialization).

#### Encoding control

`write_table` runs Helium's **optimizer** by default — it measures candidate
pipelines on your data and picks the smallest per column, so you get Helium's
real compression instead of fixed defaults. Pass `optimize=False` for the fast
default encodings:

```python
pyhelium.write_table("events.he", table)                  # optimized (default)
pyhelium.write_table("events.he", table, optimize=False)  # fast defaults
```

#### Streaming + projection (bounded memory)

Write large tables in stripes (row groups) and read back only the columns /
stripes you need:

```python
# Multi-stripe write: at most 100k rows per stripe.
pyhelium.write_table("big.he", table, stripe_rows=100_000)

# Column projection: decode only "id" and "label" (Helium reads just those bytes).
subset = pyhelium.read_table("big.he", columns=["id", "label"])

# Stripe-range read: stripes [2, 5) only.
window = pyhelium.read_table("big.he", stripe_range=(2, 5))
```

`read_table` returns a Table with one chunk per stripe.

## Scope & limitations

- **Map columns** do not yet round-trip through a `.he` file (the value type is
  paired with the key column on read). This is a limitation in the main crate's
  Map/Arrow composition, not the binding; `Struct` and `List` nesting work. The
  corresponding test is marked `xfail`.
- **Dictionary** columns are written/read through their decoded form; the
  binding does not yet expose dictionary encoding as a Python-level option.
- `read_he` / `write_he` remain numpy/flat only by design; use `read_table` /
  `write_table` for everything else.

## Packaging

Wheels are built **abi3** (`cp39-abi3`) — one wheel per platform serves all
CPython >= 3.9. The `cibuildwheel` matrix and PyPI publish steps are documented
in [`PACKAGING.md`](PACKAGING.md); publishing is prepared but not yet enabled.

## Roadmap

See [`docs/ROADMAP.md`](../docs/ROADMAP.md) → *Bindings → pyhelium*. The headline
Arrow / pandas interop, encoding control, and streaming/projection have landed;
remaining items are Map round-trip (main-crate fix), dictionary-encoding control,
and turning on PyPI publishing.
