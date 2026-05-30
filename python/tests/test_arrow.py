"""Tests for pyhelium.write_table / pyhelium.read_table — Arrow / pandas interop.

These exercise the Arrow-bridge path that lifts the flat-only limitation of
write_he / read_he: nullable, nested (Struct / List / Map), and semantic
(Date / Datetime / Decimal) columns round-trip through .he files.
"""
import datetime
import decimal
import os

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")

import pyhelium


# ---------------------------------------------------------------------------
# Basic Arrow Table round-trip
# ---------------------------------------------------------------------------

def test_table_roundtrip_basic(tmp_path):
    tab = pa.table({
        "id": pa.array([1, 2, 3, 4], type=pa.int64()),
        "score": pa.array([1.5, 2.5, 3.5, 4.5], type=pa.float64()),
        "label": pa.array(["a", "b", "c", "d"], type=pa.string()),
    })
    path = str(tmp_path / "basic.he")
    pyhelium.write_table(path, tab)
    out = pyhelium.read_table(path)
    assert out.schema.equals(tab.schema)
    assert out.equals(tab)


def test_table_roundtrip_all_numeric(tmp_path):
    n = 100
    tab = pa.table({
        "i8": pa.array(np.arange(n, dtype=np.int8)),
        "i16": pa.array(np.arange(n, dtype=np.int16)),
        "i32": pa.array(np.arange(n, dtype=np.int32)),
        "i64": pa.array(np.arange(n, dtype=np.int64)),
        "u8": pa.array(np.arange(n, dtype=np.uint8)),
        "u32": pa.array(np.arange(n, dtype=np.uint32)),
        "f32": pa.array(np.random.rand(n).astype(np.float32)),
        "f64": pa.array(np.random.rand(n).astype(np.float64)),
    })
    path = str(tmp_path / "numeric.he")
    pyhelium.write_table(path, tab)
    out = pyhelium.read_table(path)
    assert out.equals(tab)


# ---------------------------------------------------------------------------
# Nullable column
# ---------------------------------------------------------------------------

def test_nullable_int_column(tmp_path):
    tab = pa.table({
        "a": pa.array([1, None, 3, None, 5], type=pa.int64()),
    })
    path = str(tmp_path / "nullable.he")
    pyhelium.write_table(path, tab)
    out = pyhelium.read_table(path)
    assert out.equals(tab)
    assert out.column("a").null_count == 2


def test_nullable_string_column(tmp_path):
    tab = pa.table({
        "s": pa.array(["x", None, "z", None], type=pa.string()),
    })
    path = str(tmp_path / "nullstr.he")
    pyhelium.write_table(path, tab)
    out = pyhelium.read_table(path)
    assert out.equals(tab)


# ---------------------------------------------------------------------------
# Nested: Struct and List
# ---------------------------------------------------------------------------

def test_struct_column(tmp_path):
    tab = pa.table({
        "person": pa.array(
            [{"name": "ann", "age": 30}, {"name": "bob", "age": 25}, {"name": "cy", "age": 40}],
            type=pa.struct([("name", pa.string()), ("age", pa.int64())]),
        ),
    })
    path = str(tmp_path / "struct.he")
    pyhelium.write_table(path, tab)
    out = pyhelium.read_table(path)
    assert out.equals(tab)


def test_list_column(tmp_path):
    tab = pa.table({
        "tags": pa.array([[1, 2, 3], [], [4], [5, 6]], type=pa.list_(pa.int32())),
    })
    path = str(tmp_path / "list.he")
    pyhelium.write_table(path, tab)
    out = pyhelium.read_table(path)
    assert out.equals(tab)


def test_list_of_string_column(tmp_path):
    tab = pa.table({
        "words": pa.array([["a", "b"], ["c"], [], ["d", "e", "f"]],
                          type=pa.list_(pa.string())),
    })
    path = str(tmp_path / "lstr.he")
    pyhelium.write_table(path, tab)
    out = pyhelium.read_table(path)
    assert out.equals(tab)


@pytest.mark.xfail(
    reason="Map columns do not yet round-trip through a .he file: the value "
           "logical type is paired with the key column variant on read. This is "
           "a limitation in the main crate's Map/Arrow composition (read_record_batch "
           "has no Map coverage), not in the binding. Struct and List nesting work.",
    strict=True,
)
def test_map_column(tmp_path):
    tab = pa.table({
        "m": pa.array([[("k1", 1), ("k2", 2)], [("k3", 3)], []],
                      type=pa.map_(pa.string(), pa.int64())),
    })
    path = str(tmp_path / "map.he")
    pyhelium.write_table(path, tab)
    out = pyhelium.read_table(path)
    assert out.equals(tab)


# ---------------------------------------------------------------------------
# Semantic types: Date, Datetime, Decimal
# ---------------------------------------------------------------------------

def test_date32_column(tmp_path):
    tab = pa.table({
        "d": pa.array(
            [datetime.date(2020, 1, 1), datetime.date(2021, 6, 15), datetime.date(1999, 12, 31)],
            type=pa.date32(),
        ),
    })
    path = str(tmp_path / "date.he")
    pyhelium.write_table(path, tab)
    out = pyhelium.read_table(path)
    assert out.equals(tab)


def test_timestamp_column(tmp_path):
    tab = pa.table({
        "ts": pa.array(
            [datetime.datetime(2020, 1, 1, 12, 0), datetime.datetime(2020, 1, 2, 13, 30)],
            type=pa.timestamp("us"),
        ),
    })
    path = str(tmp_path / "ts.he")
    pyhelium.write_table(path, tab)
    out = pyhelium.read_table(path)
    assert out.equals(tab)


def test_decimal128_column(tmp_path):
    tab = pa.table({
        "amt": pa.array(
            [decimal.Decimal("1.50"), decimal.Decimal("99.99"), decimal.Decimal("0.01")],
            type=pa.decimal128(10, 2),
        ),
    })
    path = str(tmp_path / "dec.he")
    pyhelium.write_table(path, tab)
    out = pyhelium.read_table(path)
    assert out.equals(tab)


def test_decimal_nullable_column(tmp_path):
    tab = pa.table({
        "amt": pa.array(
            [decimal.Decimal("1.50"), None, decimal.Decimal("0.01")],
            type=pa.decimal128(10, 2),
        ),
    })
    path = str(tmp_path / "decnull.he")
    pyhelium.write_table(path, tab)
    out = pyhelium.read_table(path)
    assert out.equals(tab)


# ---------------------------------------------------------------------------
# Combined: a realistic mixed table (the headline demonstration)
# ---------------------------------------------------------------------------

def test_mixed_table_roundtrip(tmp_path):
    tab = pa.table({
        "a": pa.array([1, None, 3], type=pa.int64()),
        "d": pa.array([datetime.date(2020, 1, 1), None, datetime.date(2021, 6, 15)],
                      type=pa.date32()),
        "dec": pa.array([decimal.Decimal("1.50"), decimal.Decimal("2.25"), None],
                        type=pa.decimal128(5, 2)),
        "lst": pa.array([[1, 2], [3], []], type=pa.list_(pa.int32())),
        "s": pa.array([{"x": 1, "y": "a"}, {"x": 2, "y": "b"}, {"x": 3, "y": "c"}]),
    })
    path = str(tmp_path / "mixed.he")
    pyhelium.write_table(path, tab)
    out = pyhelium.read_table(path)
    assert out.schema.equals(tab.schema)
    assert out.equals(tab)


# ---------------------------------------------------------------------------
# pandas DataFrame interop (write a DataFrame with a nullable + a date column,
# read it back, assert equality) — the acceptance-bar demonstration.
# ---------------------------------------------------------------------------

def test_pandas_dataframe_roundtrip(tmp_path):
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({
        "id": pd.array([1, 2, None, 4], dtype="Int64"),       # nullable integer
        "when": pd.to_datetime(["2020-01-01", "2020-01-02",
                                "2020-01-03", "2020-01-04"]),  # datetime
        "name": ["alice", "bob", "carol", "dave"],
    })
    path = str(tmp_path / "frame.he")
    pyhelium.write_table(path, df)

    back = pyhelium.read_table(path).to_pandas()
    # Nullable integer survived (the None is preserved as NA/NaN).
    assert back["id"].isna().tolist() == [False, False, True, False]
    assert back["id"].dropna().tolist() == [1, 2, 4]
    # Datetime values survived.
    assert list(back["when"]) == list(df["when"])
    assert back["name"].tolist() == ["alice", "bob", "carol", "dave"]


def test_pandas_string_column_is_normalized(tmp_path):
    # pandas->pyarrow emits large_string by default; the binding normalizes it
    # to Utf8 so the column writes and round-trips.
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"s": ["alpha", "beta", "gamma"]})
    path = str(tmp_path / "strs.he")
    pyhelium.write_table(path, df)
    back = pyhelium.read_table(path)
    assert back.column("s").to_pylist() == ["alpha", "beta", "gamma"]


# ---------------------------------------------------------------------------
# Streaming (multi-stripe) writes + by-stripe / projected reads
# ---------------------------------------------------------------------------

def test_streaming_multistripe(tmp_path):
    n = 1000
    tab = pa.table({
        "x": pa.array(np.arange(n, dtype=np.int64)),
        "y": pa.array((np.arange(n) * 0.5).astype(np.float64)),
    })
    path = str(tmp_path / "stream.he")
    pyhelium.write_table(path, tab, stripe_rows=100)

    out = pyhelium.read_table(path)
    assert out.num_rows == n
    # 1000 rows / 100 per stripe => 10 stripes => 10 chunks.
    assert out.column("x").num_chunks == 10
    assert out.combine_chunks().equals(tab.combine_chunks())


def test_projection(tmp_path):
    tab = pa.table({
        "a": pa.array(np.arange(300, dtype=np.int64)),
        "b": pa.array(np.arange(300, dtype=np.int64) * 2),
        "c": pa.array(["s"] * 300),
    })
    path = str(tmp_path / "proj.he")
    pyhelium.write_table(path, tab, stripe_rows=100)

    only_b = pyhelium.read_table(path, columns=["b"])
    assert only_b.schema.names == ["b"]
    assert only_b.column("b").combine_chunks().equals(tab.column("b").combine_chunks())


def test_stripe_range(tmp_path):
    tab = pa.table({"x": pa.array(np.arange(500, dtype=np.int64))})
    path = str(tmp_path / "ranges.he")
    pyhelium.write_table(path, tab, stripe_rows=100)  # 5 stripes

    middle = pyhelium.read_table(path, stripe_range=(1, 3))
    assert middle.num_rows == 200
    assert middle.column("x").combine_chunks().to_pylist() == list(range(100, 300))


def test_stripe_range_out_of_bounds(tmp_path):
    tab = pa.table({"x": pa.array(np.arange(100, dtype=np.int64))})
    path = str(tmp_path / "oob.he")
    pyhelium.write_table(path, tab, stripe_rows=50)  # 2 stripes
    with pytest.raises(ValueError):
        pyhelium.read_table(path, stripe_range=(0, 99))


# ---------------------------------------------------------------------------
# Encoding control: optimize=True should be no larger than optimize=False
# ---------------------------------------------------------------------------

def test_optimizer_reduces_or_matches_size(tmp_path):
    n = 50_000
    tab = pa.table({
        "seq": pa.array(np.arange(n, dtype=np.int64)),
        "ts": pa.array((1_600_000_000 + np.arange(n) * 60).astype(np.int64)),
    })
    opt = str(tmp_path / "opt.he")
    dfl = str(tmp_path / "default.he")
    pyhelium.write_table(opt, tab, optimize=True)
    pyhelium.write_table(dfl, tab, optimize=False)

    assert os.path.getsize(opt) <= os.path.getsize(dfl)
    # Both must round-trip identically.
    assert pyhelium.read_table(opt).equals(tab)
    assert pyhelium.read_table(dfl).equals(tab)


# ---------------------------------------------------------------------------
# RecordBatch input
# ---------------------------------------------------------------------------

def test_record_batch_input(tmp_path):
    batch = pa.record_batch({"x": pa.array([10, 20, 30], type=pa.int32())})
    path = str(tmp_path / "rb.he")
    pyhelium.write_table(path, batch)
    out = pyhelium.read_table(path)
    assert out.column("x").to_pylist() == [10, 20, 30]


# ---------------------------------------------------------------------------
# Empty table
# ---------------------------------------------------------------------------

def test_empty_table(tmp_path):
    tab = pa.table({"x": pa.array([], type=pa.int64())})
    path = str(tmp_path / "empty.he")
    pyhelium.write_table(path, tab)
    out = pyhelium.read_table(path)
    assert out.num_rows == 0
    assert out.schema.names == ["x"]


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

def test_write_table_bad_input_raises(tmp_path):
    with pytest.raises(TypeError):
        pyhelium.write_table(str(tmp_path / "x.he"), 42)


def test_read_table_missing_column_raises(tmp_path):
    tab = pa.table({"x": pa.array([1, 2, 3], type=pa.int64())})
    path = str(tmp_path / "mc.he")
    pyhelium.write_table(path, tab)
    with pytest.raises(ValueError):
        pyhelium.read_table(path, columns=["nope"])
