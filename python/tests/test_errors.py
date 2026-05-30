"""Tests for error paths in pyhelium."""
import numpy as np
import pytest

import pyhelium


# ---------------------------------------------------------------------------
# compress() error paths
# ---------------------------------------------------------------------------

def test_compress_list_without_dtype_raises():
    """A plain Python list without dtype raises TypeError."""
    with pytest.raises(TypeError, match="dtype argument required"):
        pyhelium.compress([1, 2, 3])


def test_compress_unsupported_dtype_str_raises():
    """An unrecognised dtype string raises TypeError."""
    with pytest.raises(TypeError):
        pyhelium.compress([1, 2, 3], dtype="complex128")


def test_compress_unsupported_numpy_dtype_raises():
    """A numpy array with an unsupported dtype (e.g. complex) raises TypeError."""
    arr = np.array([1 + 2j, 3 + 4j], dtype=np.complex128)
    with pytest.raises(TypeError):
        pyhelium.compress(arr)


def test_compress_non_array_non_list_raises():
    """Passing a plain int raises TypeError."""
    with pytest.raises(TypeError):
        pyhelium.compress(42)


# ---------------------------------------------------------------------------
# decompress() error paths
# ---------------------------------------------------------------------------

def test_decompress_empty_bytes_raises():
    """Empty bytes should raise ValueError (not a valid HEC0 stream)."""
    with pytest.raises(ValueError):
        pyhelium.decompress(b"")


def test_decompress_garbage_raises():
    """Random bytes that are not HEC0 should raise ValueError."""
    with pytest.raises(ValueError):
        pyhelium.decompress(b"NOT_HEC0_DATA_AT_ALL\xff\xfe")


def test_decompress_truncated_raises():
    """A valid HEC0 header but truncated body should raise ValueError."""
    arr = np.arange(1000, dtype=np.int64)
    buf = pyhelium.compress(arr)
    # Truncate to first 8 bytes (just magic, no payload)
    with pytest.raises(ValueError):
        pyhelium.decompress(buf[:8])


# ---------------------------------------------------------------------------
# write_he() error paths
# ---------------------------------------------------------------------------

def test_write_he_bad_path_raises():
    """Writing to a non-existent directory raises an error."""
    with pytest.raises(Exception):  # RuntimeError or ValueError
        pyhelium.write_he("/nonexistent_dir_xyz/file.he", {"x": np.array([1], dtype=np.int32)})


def test_write_he_unsupported_value_type_raises():
    """Passing a dict value that is neither ndarray nor list raises TypeError."""
    with pytest.raises(TypeError):
        pyhelium.write_he("/tmp/ignored.he", {"x": 42})


def test_write_he_list_with_wrong_element_type_raises():
    """A list of integers (not str or bytes) should raise TypeError."""
    with pytest.raises(TypeError):
        pyhelium.write_he("/tmp/ignored.he", {"x": [1, 2, 3]})


# ---------------------------------------------------------------------------
# read_he() error paths
# ---------------------------------------------------------------------------

def test_read_he_missing_file_raises():
    """Reading a file that does not exist raises ValueError."""
    with pytest.raises(ValueError):
        pyhelium.read_he("/nonexistent_file_xyz.he")


def test_read_he_corrupt_file_raises(tmp_path):
    """Reading a file with garbage content raises ValueError."""
    path = str(tmp_path / "corrupt.he")
    with open(path, "wb") as f:
        f.write(b"\x00" * 64)
    with pytest.raises(ValueError):
        pyhelium.read_he(path)
