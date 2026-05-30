"""Tests for pyhelium.compress / pyhelium.decompress round-trips."""
import math

import numpy as np
import pytest

import pyhelium


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _roundtrip(arr: np.ndarray) -> np.ndarray:
    """Compress then decompress; return the recovered array."""
    buf = pyhelium.compress(arr)
    assert isinstance(buf, bytes), "compress must return bytes"
    assert len(buf) > 0, "compressed output must be non-empty"
    return pyhelium.decompress(buf)


# ---------------------------------------------------------------------------
# Numeric dtype round-trips
# ---------------------------------------------------------------------------

def test_compress_i8_roundtrip():
    arr = np.array([-128, 0, 1, 127], dtype=np.int8)
    out = _roundtrip(arr)
    np.testing.assert_array_equal(arr, out)
    assert out.dtype == np.int8


def test_compress_i16_roundtrip():
    arr = np.arange(-1000, 1000, dtype=np.int16)
    out = _roundtrip(arr)
    np.testing.assert_array_equal(arr, out)
    assert out.dtype == np.int16


def test_compress_i32_roundtrip():
    arr = np.arange(0, 50_000, dtype=np.int32)
    out = _roundtrip(arr)
    np.testing.assert_array_equal(arr, out)
    assert out.dtype == np.int32


def test_compress_i64_roundtrip():
    arr = np.arange(10_000, dtype=np.int64)
    out = _roundtrip(arr)
    np.testing.assert_array_equal(arr, out)
    assert out.dtype == np.int64


def test_compress_u8_roundtrip():
    arr = np.array([0, 1, 128, 255], dtype=np.uint8)
    out = _roundtrip(arr)
    np.testing.assert_array_equal(arr, out)
    assert out.dtype == np.uint8


def test_compress_u16_roundtrip():
    arr = np.arange(0, 65536, 17, dtype=np.uint16)
    out = _roundtrip(arr)
    np.testing.assert_array_equal(arr, out)
    assert out.dtype == np.uint16


def test_compress_u32_roundtrip():
    arr = np.arange(0, 100_000, dtype=np.uint32)
    out = _roundtrip(arr)
    np.testing.assert_array_equal(arr, out)
    assert out.dtype == np.uint32


def test_compress_u64_roundtrip():
    arr = np.array([0, 2**32, 2**48, 2**63 - 1], dtype=np.uint64)
    out = _roundtrip(arr)
    np.testing.assert_array_equal(arr, out)
    assert out.dtype == np.uint64


def test_compress_f32_roundtrip():
    rng = np.random.default_rng(42)
    arr = rng.random(1000).astype(np.float32)
    out = _roundtrip(arr)
    np.testing.assert_array_equal(arr, out)
    assert out.dtype == np.float32


def test_compress_f64_roundtrip():
    rng = np.random.default_rng(99)
    arr = rng.random(10_000).astype(np.float64)
    out = _roundtrip(arr)
    np.testing.assert_array_equal(arr, out)
    assert out.dtype == np.float64


# ---------------------------------------------------------------------------
# Special float values
# ---------------------------------------------------------------------------

def test_compress_f64_with_nan_and_inf():
    arr = np.array([1.0, float("nan"), 3.0, float("inf"), float("-inf")], dtype=np.float64)
    buf = pyhelium.compress(arr)
    out = pyhelium.decompress(buf)

    assert out[0] == 1.0
    assert math.isnan(float(out[1])), "NaN must survive round-trip"
    assert out[2] == 3.0
    assert math.isinf(float(out[3])) and out[3] > 0
    assert math.isinf(float(out[4])) and out[4] < 0


# ---------------------------------------------------------------------------
# List input with explicit dtype
# ---------------------------------------------------------------------------

def test_compress_list_i64():
    lst = list(range(100))
    buf = pyhelium.compress(lst, dtype="i64")
    out = pyhelium.decompress(buf)
    np.testing.assert_array_equal(np.array(lst, dtype=np.int64), out)


def test_compress_list_f64():
    lst = [0.5, 1.5, 2.5, 3.5]
    buf = pyhelium.compress(lst, dtype="f64")
    out = pyhelium.decompress(buf)
    np.testing.assert_array_almost_equal(lst, out)


# ---------------------------------------------------------------------------
# Large array smoke test
# ---------------------------------------------------------------------------

def test_compress_large_i64():
    arr = np.arange(1_000_000, dtype=np.int64)
    out = _roundtrip(arr)
    np.testing.assert_array_equal(arr, out)


# ---------------------------------------------------------------------------
# compressed output is smaller than raw (sanity check for non-trivial data)
# ---------------------------------------------------------------------------

def test_compress_is_smaller_than_raw_for_sequential():
    arr = np.arange(100_000, dtype=np.int64)
    buf = pyhelium.compress(arr)
    raw_bytes = arr.nbytes
    assert len(buf) < raw_bytes, (
        f"Expected compressed ({len(buf)}) < raw ({raw_bytes})"
    )
