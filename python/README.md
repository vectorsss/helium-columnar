# pyhelium

Python bindings for the [Helium](https://github.com/vectorsss/helium-columnar)
columnar compression library.

## Installation (development)

```bash
cd python/
python -m venv .venv && source .venv/bin/activate
pip install maturin numpy pytest
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

### Write and read a .he file

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

Supported column types:
- `numpy.ndarray` (numeric) → stored as a `Primitive` column.
- `list[str]` → stored as a `Utf8` column, returned as `list[str]`.
- `list[bytes]` → stored as a `Binary` column, returned as `list[bytes]`.

## Scope & limitations

`pyhelium` is an early binding focused on numeric/flat data:

- **Flat columns only.** Nullable, nested (Struct/List/Map/Union), dictionary,
  and semantic (Date/Datetime/Decimal128) columns raise `NotImplementedError`
  on `read_he` and cannot be produced by `write_he`.
- **Fixed encodings.** Pipelines are chosen automatically (integers
  `leb128→zstd`, floats `gorilla→zstd`, strings `delta→leb128→zstd` offsets +
  `zstd` data); there is no per-column tuning or access to the optimizer yet.
- **Whole-file, in-memory.** `read_he` loads the entire file; there is no
  column projection, streaming, or multi-stripe control.

## Roadmap

The headline next step is **Arrow / pandas interop** (`read_he() -> pa.Table`,
`write_he(df)`), which would lift the flat-only limitation by reusing Helium's
`arrow` bridge, followed by encoding control (optimizer access) and streaming
I/O. See [`docs/ROADMAP.md`](../docs/ROADMAP.md) → *Bindings* for the full plan.
