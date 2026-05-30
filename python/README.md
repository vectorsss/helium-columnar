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

Supported column types in Phase 1:
- `numpy.ndarray` (numeric) → stored as a `Primitive` column.
- `list[str]` → stored as a `Utf8` column, returned as `list[str]`.
- `list[bytes]` → stored as a `Binary` column, returned as `list[bytes]`.
