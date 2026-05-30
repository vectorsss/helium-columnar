"""Tests for pyhelium.write_he / pyhelium.read_he round-trips."""
import os

import numpy as np
import pytest

import pyhelium


# ---------------------------------------------------------------------------
# Basic 3-column flat dataset
# ---------------------------------------------------------------------------

def test_write_read_roundtrip(tmp_path):
    data = {
        "id": np.arange(100, dtype=np.int64),
        "score": np.random.rand(100).astype(np.float64),
        "label": [f"row-{i}" for i in range(100)],
    }
    path = str(tmp_path / "test.he")
    pyhelium.write_he(path, data)

    out = pyhelium.read_he(path)

    np.testing.assert_array_equal(data["id"], out["id"])
    np.testing.assert_array_equal(data["score"], out["score"])
    assert data["label"] == out["label"]


# ---------------------------------------------------------------------------
# All numeric dtypes in a single file
# ---------------------------------------------------------------------------

def test_all_numeric_dtypes(tmp_path):
    # All columns must have the same row count within a single stripe.
    N = 50
    rng = np.random.default_rng(7)
    data = {
        "col_i8":  np.arange(N, dtype=np.int8),
        "col_i16": np.arange(N, dtype=np.int16),
        "col_i32": np.arange(N, dtype=np.int32),
        "col_i64": np.arange(N, dtype=np.int64),
        "col_u8":  np.arange(N, dtype=np.uint8),
        "col_u16": np.arange(N, dtype=np.uint16),
        "col_u32": np.arange(N, dtype=np.uint32),
        "col_u64": np.arange(N, dtype=np.uint64),
        "col_f32": rng.random(N).astype(np.float32),
        "col_f64": rng.random(N).astype(np.float64),
    }
    path = str(tmp_path / "all_dtypes.he")
    pyhelium.write_he(path, data)
    out = pyhelium.read_he(path)

    for name, arr in data.items():
        assert name in out, f"missing column {name}"
        np.testing.assert_array_equal(arr, out[name], err_msg=f"mismatch in {name}")
        assert out[name].dtype == arr.dtype, f"dtype mismatch in {name}"


# ---------------------------------------------------------------------------
# String (Utf8) column
# ---------------------------------------------------------------------------

def test_utf8_column_roundtrip(tmp_path):
    strings = ["hello", "world", "foo", "bar", "", "unicode: 中文"]
    data = {"text": strings}
    path = str(tmp_path / "utf8.he")
    pyhelium.write_he(path, data)
    out = pyhelium.read_he(path)
    assert out["text"] == strings


# ---------------------------------------------------------------------------
# Binary column
# ---------------------------------------------------------------------------

def test_binary_column_roundtrip(tmp_path):
    blobs = [b"hello", b"\x00\x01\x02", b"", b"\xff" * 100]
    data = {"raw": blobs}
    path = str(tmp_path / "binary.he")
    pyhelium.write_he(path, data)
    out = pyhelium.read_he(path)
    # read_he returns list of bytes objects
    result = out["raw"]
    assert len(result) == len(blobs)
    for expected, actual in zip(blobs, result):
        assert bytes(actual) == expected


# ---------------------------------------------------------------------------
# File is actually created
# ---------------------------------------------------------------------------

def test_file_is_created(tmp_path):
    path = str(tmp_path / "created.he")
    assert not os.path.exists(path)
    pyhelium.write_he(path, {"x": np.array([1, 2, 3], dtype=np.int32)})
    assert os.path.exists(path)
    assert os.path.getsize(path) > 0


# ---------------------------------------------------------------------------
# Empty column (zero rows) round-trip
# ---------------------------------------------------------------------------

def test_empty_column(tmp_path):
    data = {"empty": np.array([], dtype=np.int64)}
    path = str(tmp_path / "empty.he")
    pyhelium.write_he(path, data)
    out = pyhelium.read_he(path)
    np.testing.assert_array_equal(data["empty"], out["empty"])


# ---------------------------------------------------------------------------
# Overwrite existing file
# ---------------------------------------------------------------------------

def test_overwrite(tmp_path):
    path = str(tmp_path / "overwrite.he")
    pyhelium.write_he(path, {"a": np.array([1, 2, 3], dtype=np.int64)})
    pyhelium.write_he(path, {"b": np.array([9, 8, 7], dtype=np.int64)})
    out = pyhelium.read_he(path)
    assert "b" in out
    assert "a" not in out
