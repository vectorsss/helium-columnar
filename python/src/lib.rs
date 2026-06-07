//! `pyhelium` — Python bindings for the Helium columnar compression library.
//!
//! Two complementary APIs are exposed:
//!
//! Numeric / flat (back-compatible, no Arrow dependency on the Python side):
//! - `compress(values, dtype=None) -> bytes`
//! - `decompress(buf: bytes) -> numpy.ndarray`
//! - `write_he(path, data: dict) -> None`
//! - `read_he(path) -> dict`
//!
//! Arrow / pandas interop (full recursive type set — nullable, nested, semantic):
//! - `write_table(path, table_or_df, optimize=True, stripe_rows=None) -> None`
//! - `read_table(path, columns=None, stripe_range=None) -> pyarrow.Table`
//!
//! The Arrow path reuses Helium's `arrow` bridge (`LogicalColumn` <-> `ArrayRef`),
//! moving arrays across the FFI boundary via the Arrow C Data Interface, so it
//! lifts the flat-only limitation of `write_he`/`read_he`: nullable, nested
//! (Struct/List/Map), and semantic (Date/Datetime/Decimal) columns round-trip.

use std::fs::File;
use std::io::BufReader;
use std::sync::Arc;

use arrow::array::{ArrayRef, RecordBatch};
use arrow::datatypes::{Field as ArrowField, Schema as ArrowSchema};
use arrow::pyarrow::{FromPyArrow, ToPyArrow};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::{PyNotImplementedError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyString};

use helium::arrow::{from_arrow_array, schema_from_arrow, schema_to_arrow, to_arrow_array};
use helium::optimizer::Optimizer;
use helium::{
    CoderRegistry, CoderSpec, ColumnData, ColumnSpec, DataType, HeliumReader, HeliumWriter,
    LogicalColumn, LogicalType, Schema,
};

// ---------------------------------------------------------------------------
// Error helpers
// ---------------------------------------------------------------------------

/// Convert a `helium::HeliumError` to a Python `ValueError`.
fn he_err(e: helium::HeliumError) -> PyErr {
    PyValueError::new_err(format!("{e}"))
}

// ---------------------------------------------------------------------------
// Dtype string parsing
// ---------------------------------------------------------------------------

/// Parse a dtype string ("i8", "i16", …, "f64") into a `DataType`.
///
/// Returns `Err(PyTypeError)` for any unrecognised or unsupported dtype.
fn parse_dtype_str(s: &str) -> PyResult<DataType> {
    match s {
        "i8" => Ok(DataType::I8),
        "i16" => Ok(DataType::I16),
        "i32" => Ok(DataType::I32),
        "i64" => Ok(DataType::I64),
        "u8" => Ok(DataType::U8),
        "u16" => Ok(DataType::U16),
        "u32" => Ok(DataType::U32),
        "u64" => Ok(DataType::U64),
        "f32" => Ok(DataType::F32),
        "f64" => Ok(DataType::F64),
        other => Err(PyTypeError::new_err(format!(
            "unsupported dtype {other:?}: must be one of i8/i16/i32/i64/u8/u16/u32/u64/f32/f64"
        ))),
    }
}

/// Infer a `DataType` from a numpy dtype descriptor string such as
/// `"int64"`, `"float32"`, `">u2"`, etc.
fn infer_dtype_from_numpy(descr: &str) -> PyResult<DataType> {
    // Strip endian prefix: '<', '>', '|', '='
    let stripped = descr.trim_start_matches(['<', '>', '|', '=']);
    match stripped {
        "i1" | "int8" => Ok(DataType::I8),
        "i2" | "int16" => Ok(DataType::I16),
        "i4" | "int32" => Ok(DataType::I32),
        "i8" | "int64" => Ok(DataType::I64),
        "u1" | "uint8" => Ok(DataType::U8),
        "u2" | "uint16" => Ok(DataType::U16),
        "u4" | "uint32" => Ok(DataType::U32),
        "u8" | "uint64" => Ok(DataType::U64),
        "f4" | "float32" => Ok(DataType::F32),
        "f8" | "float64" => Ok(DataType::F64),
        other => Err(PyTypeError::new_err(format!(
            "unsupported numpy dtype {other:?}: \
             compress/decompress only support numeric arrays \
             (i8/i16/i32/i64/u8/u16/u32/u64/f32/f64)"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Numpy array → ColumnData
// ---------------------------------------------------------------------------

/// Extract a `ColumnData` from a numpy ndarray `PyAny`.
///
/// Reads the numpy dtype string to pick the matching variant, then copies the
/// contiguous buffer.  Raises `TypeError` for non-numeric dtypes.
fn array_to_column_data(_py: Python<'_>, arr: &Bound<'_, PyAny>) -> PyResult<ColumnData> {
    // Pull out the dtype name via numpy's `dtype.str` attribute, which gives
    // the canonical two-char type code with endian prefix (e.g. "<i8").
    let dtype_str: String = arr.getattr("dtype")?.getattr("str")?.extract()?;
    let data_type = infer_dtype_from_numpy(&dtype_str)?;

    // Flatten to a contiguous 1-D array via `np.asarray(arr).ravel()`.
    // We call `ravel()` on whatever shape was passed so callers can pass
    // 1-D arrays directly and we tolerate higher-dimensional ones.
    let flat = arr.call_method0("ravel")?;

    macro_rules! extract_vec {
        ($T:ty, $variant:ident) => {{
            let ro: PyReadonlyArray1<$T> = flat.extract()?;
            Ok(ColumnData::$variant(ro.as_slice()?.to_vec()))
        }};
    }

    match data_type {
        DataType::I8 => extract_vec!(i8, I8),
        DataType::I16 => extract_vec!(i16, I16),
        DataType::I32 => extract_vec!(i32, I32),
        DataType::I64 => extract_vec!(i64, I64),
        DataType::U8 => extract_vec!(u8, U8),
        DataType::U16 => extract_vec!(u16, U16),
        DataType::U32 => extract_vec!(u32, U32),
        DataType::U64 => extract_vec!(u64, U64),
        DataType::F32 => extract_vec!(f32, F32),
        DataType::F64 => extract_vec!(f64, F64),
        DataType::Bytes => {
            // Should be unreachable because infer_dtype_from_numpy never
            // returns Bytes, but keep the arm for exhaustiveness.
            Err(PyTypeError::new_err(
                "Bytes dtype is not supported by compress/decompress; \
                 use write_he/read_he for string/binary columns",
            ))
        }
    }
}

// ---------------------------------------------------------------------------
// Python list + explicit dtype → ColumnData
// ---------------------------------------------------------------------------

/// Build a `ColumnData` from a Python list and an explicit dtype string.
fn list_to_column_data(
    _py: Python<'_>,
    list: &Bound<'_, PyList>,
    dtype: &str,
) -> PyResult<ColumnData> {
    let data_type = parse_dtype_str(dtype)?;

    macro_rules! collect_list {
        ($T:ty, $variant:ident) => {{
            let mut v: Vec<$T> = Vec::with_capacity(list.len());
            for item in list.iter() {
                v.push(item.extract::<$T>()?);
            }
            Ok(ColumnData::$variant(v))
        }};
    }

    match data_type {
        DataType::I8 => collect_list!(i8, I8),
        DataType::I16 => collect_list!(i16, I16),
        DataType::I32 => collect_list!(i32, I32),
        DataType::I64 => collect_list!(i64, I64),
        DataType::U8 => collect_list!(u8, U8),
        DataType::U16 => collect_list!(u16, U16),
        DataType::U32 => collect_list!(u32, U32),
        DataType::U64 => collect_list!(u64, U64),
        DataType::F32 => collect_list!(f32, F32),
        DataType::F64 => collect_list!(f64, F64),
        DataType::Bytes => Err(PyTypeError::new_err(
            "dtype='bytes' is not supported by compress; use write_he/read_he",
        )),
    }
}

// ---------------------------------------------------------------------------
// ColumnData → numpy array (for decompress return value)
// ---------------------------------------------------------------------------

/// Convert a `ColumnData` to an owned numpy `ndarray`.
fn column_data_to_array<'py>(py: Python<'py>, cd: ColumnData) -> PyResult<Bound<'py, PyAny>> {
    macro_rules! to_array {
        ($v:expr, $T:ty) => {{
            // PyArray1::from_vec returns Bound<'_, PyArray1<T>>.
            let arr: Bound<'py, PyArray1<$T>> = PyArray1::from_vec(py, $v);
            Ok(arr.into_any())
        }};
    }

    match cd {
        ColumnData::I8(v) => to_array!(v, i8),
        ColumnData::I16(v) => to_array!(v, i16),
        ColumnData::I32(v) => to_array!(v, i32),
        ColumnData::I64(v) => to_array!(v, i64),
        ColumnData::U8(v) => to_array!(v, u8),
        ColumnData::U16(v) => to_array!(v, u16),
        ColumnData::U32(v) => to_array!(v, u32),
        ColumnData::U64(v) => to_array!(v, u64),
        ColumnData::F32(v) => to_array!(v, f32),
        ColumnData::F64(v) => to_array!(v, f64),
        ColumnData::Bytes(_) => Err(PyValueError::new_err(
            "decompress returned a raw Bytes buffer; this should not happen \
             for HEC0 self-describing streams",
        )),
    }
}

// ---------------------------------------------------------------------------
// Schema building helpers for write_he
// ---------------------------------------------------------------------------

/// Default pipeline specs for a numeric primitive column.
///
/// - Integer types (`i8`..`u64`): `leb128` → `zstd`.  `leb128` accepts all
///   integer dtypes and emits `Bytes`; `zstd` then compresses those bytes.
/// - Float types (`f32`, `f64`): `gorilla` → `zstd`.  `gorilla` (XOR-delta)
///   converts floats to `Bytes`; `zstd` does the final compression.
fn default_prim_coders(data_type: DataType) -> Vec<CoderSpec> {
    if data_type.is_float() {
        vec![CoderSpec::new("gorilla"), CoderSpec::new("zstd")]
    } else {
        // All integer variants (signed and unsigned) are supported by leb128.
        vec![CoderSpec::new("leb128"), CoderSpec::new("zstd")]
    }
}

/// Default pipeline specs for offsets (U32).
fn default_offset_coders() -> Vec<CoderSpec> {
    vec![
        CoderSpec::new("delta"),
        CoderSpec::new("leb128"),
        CoderSpec::new("zstd"),
    ]
}

/// Default pipeline specs for string/binary data bytes.
fn default_data_coders() -> Vec<CoderSpec> {
    vec![CoderSpec::new("zstd")]
}

// ---------------------------------------------------------------------------
// Column value helpers for write_he
// ---------------------------------------------------------------------------

/// Detect and extract the `LogicalColumn` and a placeholder `ColumnSpec` for one column in `write_he`.
///
/// Supports:
/// - `numpy.ndarray` (numeric dtypes) → `Primitive`
/// - `list[str]` → `Utf8`
/// - `list[bytes]` → `Binary`
///
/// The returned `ColumnSpec` has `name = "__placeholder__"`; the caller
/// must overwrite `spec.name` before passing to `Schema::new`.
fn extract_logical_column(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
) -> PyResult<(LogicalColumn, ColumnSpec)> {
    // --- numpy ndarray path ---
    // Check if the object has a `dtype` attribute (numpy-like).
    if value.hasattr("dtype")? && value.hasattr("ravel")? {
        let cd = array_to_column_data(py, value)?;
        let data_type = cd.data_type();
        let col = LogicalColumn::Primitive(cd);
        let spec =
            ColumnSpec::primitive("__placeholder__", data_type, default_prim_coders(data_type));
        return Ok((col, spec));
    }

    // --- Python list path ---
    if let Ok(list) = value.downcast::<PyList>() {
        if list.is_empty() {
            // Can't infer type from empty list; default to Utf8.
            let col = LogicalColumn::Utf8(vec![]);
            let spec = ColumnSpec::utf8(
                "__placeholder__",
                default_offset_coders(),
                default_data_coders(),
            );
            return Ok((col, spec));
        }

        // Peek at the first element to decide the column type.
        let first = list.get_item(0)?;
        if first.is_instance_of::<PyString>() {
            let mut strings: Vec<String> = Vec::with_capacity(list.len());
            for item in list.iter() {
                let s: String = item.extract().map_err(|_| {
                    PyTypeError::new_err("list elements must all be str for a string column")
                })?;
                strings.push(s);
            }
            let col = LogicalColumn::Utf8(strings);
            let spec = ColumnSpec::utf8(
                "__placeholder__",
                default_offset_coders(),
                default_data_coders(),
            );
            return Ok((col, spec));
        }

        if first.is_instance_of::<PyBytes>() {
            let mut blobs: Vec<Vec<u8>> = Vec::with_capacity(list.len());
            for item in list.iter() {
                let b: Vec<u8> = item
                    .downcast::<PyBytes>()
                    .map_err(|_| {
                        PyTypeError::new_err("list elements must all be bytes for a binary column")
                    })?
                    .as_bytes()
                    .to_vec();
                blobs.push(b);
            }
            let col = LogicalColumn::Binary(blobs);
            let spec = ColumnSpec::binary(
                "__placeholder__",
                default_offset_coders(),
                default_data_coders(),
            );
            return Ok((col, spec));
        }

        return Err(PyTypeError::new_err(
            "list column elements must be str or bytes; \
             for numeric data pass a numpy.ndarray",
        ));
    }

    Err(PyTypeError::new_err(
        "column values must be a numpy.ndarray (numeric) or a list of str/bytes",
    ))
}

// ---------------------------------------------------------------------------
// Public API — codec
// ---------------------------------------------------------------------------

/// Compress a numeric column to self-describing `HEC0` bytes.
///
/// Parameters
/// ----------
/// values : numpy.ndarray or list
///     The data to compress.  numpy arrays are preferred; their dtype is
///     auto-detected.  Plain Python lists require an explicit ``dtype``
///     keyword argument.
/// dtype : str, optional
///     One of ``"i8"``, ``"i16"``, ``"i32"``, ``"i64"``,
///     ``"u8"``, ``"u16"``, ``"u32"``, ``"u64"``, ``"f32"``, ``"f64"``.
///     Required when *values* is a plain Python list.
///
/// Returns
/// -------
/// bytes
///     Self-describing compressed bytes starting with the ``HEC0`` magic.
///     Pass directly to :func:`decompress`.
#[pyfunction]
#[pyo3(signature = (values, dtype=None))]
fn compress(
    py: Python<'_>,
    values: &Bound<'_, PyAny>,
    dtype: Option<&str>,
) -> PyResult<Py<PyBytes>> {
    let cd = if let Ok(list) = values.downcast::<PyList>() {
        let dt = dtype.ok_or_else(|| {
            PyTypeError::new_err("dtype argument required when values is a list; e.g. dtype='i64'")
        })?;
        list_to_column_data(py, list, dt)?
    } else if values.hasattr("dtype").unwrap_or(false) && values.hasattr("ravel").unwrap_or(false) {
        // numpy array or compatible array-like.
        // unwrap_or(false): if hasattr itself raises (PyErr), treat as absent —
        // the caller will get a clear TypeError from the else branch below.
        array_to_column_data(py, values)?
    } else {
        return Err(PyTypeError::new_err(
            "compress() requires a numpy.ndarray or a list; \
             for lists, also provide dtype='i64' (or other numeric dtype string)",
        ));
    };

    let compressed = helium::compress(cd).map_err(he_err)?;
    Ok(PyBytes::new(py, &compressed).into())
}

/// Decompress `HEC0` bytes back to a numpy array.
#[pyfunction]
fn decompress(py: Python<'_>, buf: &[u8]) -> PyResult<PyObject> {
    let cd = helium::decompress(buf).map_err(he_err)?;
    let arr = column_data_to_array(py, cd)?;
    Ok(arr.into())
}

// ---------------------------------------------------------------------------
// Public API — flat dict file I/O (back-compatible)
// ---------------------------------------------------------------------------

/// Write a flat dict of columns to a ``.he`` file.
///
/// Parameters
/// ----------
/// path : str
///     Destination file path (created or overwritten).
/// data : dict[str, numpy.ndarray | list]
///     Column name → column values.  Values may be:
///
///     - ``numpy.ndarray`` with a numeric dtype → ``Primitive`` column.
///     - ``list[str]`` → ``Utf8`` column.
///     - ``list[bytes]`` → ``Binary`` column.
///
/// For nullable / nested / semantic columns use :func:`write_table` with a
/// pyarrow Table or a pandas DataFrame instead.
#[pyfunction]
fn write_he(py: Python<'_>, path: &str, data: &Bound<'_, PyDict>) -> PyResult<()> {
    // Build schema + collect columns.
    let mut specs: Vec<ColumnSpec> = Vec::new();
    let mut columns: Vec<(String, LogicalColumn)> = Vec::new();

    for (key, value) in data.iter() {
        let name: String = key
            .extract()
            .map_err(|_| PyTypeError::new_err("column names must be strings"))?;

        let (col, mut spec) = extract_logical_column(py, &value)?;
        // Inject the real column name into the spec.
        spec.name = name.clone();
        specs.push(spec);
        columns.push((name, col));
    }

    let schema = Schema::new(specs);
    let registry = CoderRegistry::with_builtins();

    let file = File::create(path).map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;
    let mut writer = HeliumWriter::new(file, schema, &registry).map_err(he_err)?;

    for (name, col) in columns {
        writer.write_column(&name, col).map_err(he_err)?;
    }
    writer.finish().map_err(he_err)?;

    Ok(())
}

/// Read a flat-column ``.he`` file as a dict of numpy arrays / Python lists.
///
/// Primitive columns are returned as numpy ndarrays; ``Utf8`` columns as
/// ``list[str]``; ``Binary`` columns as ``list[bytes]``.
///
/// Raises ``NotImplementedError`` for nullable / nested / dict / semantic
/// columns — use :func:`read_table` for those (returns a pyarrow Table).
#[pyfunction]
fn read_he(py: Python<'_>, path: &str) -> PyResult<PyObject> {
    let file = File::open(path).map_err(|e| PyValueError::new_err(format!("{e}")))?;
    let reader = BufReader::new(file);
    let registry = CoderRegistry::with_builtins();
    let mut helium_reader = HeliumReader::new(reader, &registry).map_err(he_err)?;

    let all = helium_reader.read_all().map_err(he_err)?;

    let dict = PyDict::new(py);
    for (name, col) in all {
        let value = logical_column_to_python(py, col)?;
        dict.set_item(name, value)?;
    }

    Ok(dict.into())
}

/// Convert a `LogicalColumn` to a Python object suitable for the `read_he` return dict.
///
/// Supports only flat (non-nested) columns; nested / nullable / semantic
/// columns raise `NotImplementedError` directing the caller to `read_table`.
fn logical_column_to_python(py: Python<'_>, col: LogicalColumn) -> PyResult<PyObject> {
    match col {
        LogicalColumn::Primitive(cd) => {
            let arr = column_data_to_array(py, cd)?;
            Ok(arr.into())
        }
        LogicalColumn::Utf8(strings) => {
            let list = PyList::new(py, strings.iter().map(|s| s.as_str()))?;
            Ok(list.into())
        }
        LogicalColumn::Binary(blobs) => {
            let list = PyList::new(py, blobs.iter().map(|b| PyBytes::new(py, b)))?;
            Ok(list.into())
        }
        // Nested / nullable / dict / semantic types are bridged to Python via
        // the Arrow path (`read_table`), not the flat dict API.
        LogicalColumn::Dictionary { .. }
        | LogicalColumn::Struct { .. }
        | LogicalColumn::List { .. }
        | LogicalColumn::Map { .. }
        | LogicalColumn::Nullable { .. }
        | LogicalColumn::Union { .. }
        | LogicalColumn::Decimal128 { .. }
        | LogicalColumn::Date32 { .. }
        | LogicalColumn::Date64 { .. }
        | LogicalColumn::Datetime { .. } => Err(PyNotImplementedError::new_err(
            "read_he returns numpy/list only for flat Primitive/Utf8/Binary columns; \
             this file has nullable, nested, dict, or semantic-type columns — \
             use read_table() to get a pyarrow.Table instead",
        )),
    }
}

// ---------------------------------------------------------------------------
// Public API — Arrow / pandas interop
// ---------------------------------------------------------------------------

/// Coerce a Python object into a pyarrow Table, then into a list of RecordBatches.
///
/// Accepts a pyarrow Table, a pyarrow RecordBatch, or a pandas DataFrame
/// (converted with `pyarrow.Table.from_pandas`). Returns the batches plus the
/// table's Arrow schema (carrying the nullable flags that drive Helium's
/// `Nullable` wrapping).
fn coerce_to_batches(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
) -> PyResult<(Arc<ArrowSchema>, Vec<RecordBatch>)> {
    let pa = py.import("pyarrow")?;
    let table_cls = pa.getattr("Table")?;
    let batch_cls = pa.getattr("RecordBatch")?;

    // Normalise the input to a pyarrow.Table.
    let table: Bound<'_, PyAny> = if obj.is_instance(&table_cls)? {
        obj.clone()
    } else if obj.is_instance(&batch_cls)? {
        // Table.from_batches([batch])
        let batches = PyList::new(py, [obj])?;
        table_cls.call_method1("from_batches", (batches,))?
    } else if obj.hasattr("__dataframe__")? || obj.hasattr("to_records")? {
        // Looks like a pandas DataFrame. preserve_index=False keeps the schema
        // to just the user's columns (no implicit __index_level_0__).
        let kwargs = PyDict::new(py);
        kwargs.set_item("preserve_index", false)?;
        table_cls.call_method("from_pandas", (obj,), Some(&kwargs))?
    } else {
        return Err(PyTypeError::new_err(
            "write_table expects a pyarrow.Table, a pyarrow.RecordBatch, \
             or a pandas.DataFrame",
        ));
    };

    // Normalise types the Helium bridge does not model 1:1 (pandas/pyarrow emit
    // 64-bit-offset large variants by default) down to their 32-bit-offset
    // equivalents that Helium supports: LargeUtf8->Utf8, LargeBinary->Binary,
    // LargeList->List, LargeString in nested fields, etc. Done entirely on the
    // pyarrow side via `Table.cast`, so the main crate is untouched.
    let table = normalize_table(py, &table)?;

    // Combine chunks so each column is a single contiguous array, then split
    // into one RecordBatch (the whole table). For streaming the caller asks
    // for stripe_rows and we re-slice below.
    let combined = table.call_method0("combine_chunks")?;
    let batches_obj = combined.call_method0("to_batches")?;
    let batch_list = batches_obj.downcast::<PyList>()?;

    let mut batches = Vec::with_capacity(batch_list.len());
    for b in batch_list.iter() {
        batches.push(RecordBatch::from_pyarrow_bound(&b)?);
    }

    // An empty pyarrow table yields zero batches; synthesize one empty batch so
    // the schema (and an empty .he file) is still written.
    let arrow_schema = if let Some(first) = batches.first() {
        first.schema()
    } else {
        let schema_obj = combined.getattr("schema")?;
        Arc::new(ArrowSchema::from_pyarrow_bound(&schema_obj)?)
    };

    Ok((arrow_schema, batches))
}

/// Cast a pyarrow Table so every column uses a type the Helium Arrow bridge
/// models. pandas / pyarrow default to 64-bit-offset "large" variants
/// (`large_string`, `large_binary`, `large_list`) which Helium maps to its
/// 32-bit-offset `Utf8` / `Binary` / `List`. We build a normalized pyarrow
/// schema (recursively) and `Table.cast` to it. If no field needs changing the
/// original table is returned unchanged.
fn normalize_table<'py>(
    py: Python<'py>,
    table: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let pa = py.import("pyarrow")?;
    let schema = table.getattr("schema")?;
    let num_fields: usize = schema.call_method0("__len__")?.extract()?;

    let mut changed = false;
    let new_fields = PyList::empty(py);
    for i in 0..num_fields {
        let field = schema.call_method1("field", (i,))?;
        let dtype = field.getattr("type")?;
        let (norm_type, field_changed) = normalize_pa_type(py, &pa, &dtype)?;
        if field_changed {
            changed = true;
            let new_field = field.call_method1("with_type", (norm_type,))?;
            new_fields.append(new_field)?;
        } else {
            new_fields.append(field)?;
        }
    }

    if !changed {
        return Ok(table.clone());
    }
    let new_schema = pa.call_method1("schema", (new_fields,))?;
    table.call_method1("cast", (new_schema,))
}

/// Recursively normalize a single pyarrow DataType, returning `(type, changed)`.
fn normalize_pa_type<'py>(
    py: Python<'py>,
    pa: &Bound<'py, PyModule>,
    dtype: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyAny>, bool)> {
    let types = py.import("pyarrow.types")?;
    let is = |name: &str| -> PyResult<bool> {
        types.call_method1(name, (dtype,))?.extract::<bool>()
    };

    if is("is_large_string")? {
        return Ok((pa.call_method0("utf8")?, true));
    }
    if is("is_large_binary")? {
        return Ok((pa.call_method0("binary")?, true));
    }
    if is("is_string_view")? {
        return Ok((pa.call_method0("utf8")?, true));
    }
    if is("is_binary_view")? {
        return Ok((pa.call_method0("binary")?, true));
    }
    if is("is_large_list")? || is("is_list")? {
        let value_field = dtype.getattr("value_field")?;
        let value_type = value_field.getattr("type")?;
        let (norm_inner, inner_changed) = normalize_pa_type(py, pa, &value_type)?;
        let is_large = is("is_large_list")?;
        if is_large || inner_changed {
            let new_value_field = value_field.call_method1("with_type", (norm_inner,))?;
            return Ok((pa.call_method1("list_", (new_value_field,))?, true));
        }
        return Ok((dtype.clone(), false));
    }
    if is("is_struct")? {
        let n: usize = dtype.call_method0("__len__")?.extract()?;
        let new_fields = PyList::empty(py);
        let mut any = false;
        for i in 0..n {
            let f = dtype.call_method1("field", (i,))?;
            let ft = f.getattr("type")?;
            let (nt, c) = normalize_pa_type(py, pa, &ft)?;
            if c {
                any = true;
                new_fields.append(f.call_method1("with_type", (nt,))?)?;
            } else {
                new_fields.append(f)?;
            }
        }
        if any {
            return Ok((pa.call_method1("struct", (new_fields,))?, true));
        }
        return Ok((dtype.clone(), false));
    }
    if is("is_map")? {
        let key_field = dtype.getattr("key_field")?;
        let item_field = dtype.getattr("item_field")?;
        let (nk, kc) = normalize_pa_type(py, pa, &key_field.getattr("type")?)?;
        let (nv, vc) = normalize_pa_type(py, pa, &item_field.getattr("type")?)?;
        if kc || vc {
            return Ok((pa.call_method1("map_", (nk, nv))?, true));
        }
        return Ok((dtype.clone(), false));
    }

    Ok((dtype.clone(), false))
}

/// Build the Helium write `Schema` for a set of batches.
///
/// When `optimize` is true, the optimizer measures candidate pipelines on the
/// concatenated columns and picks the smallest per leaf. Otherwise the default
/// encodings from the Arrow→Helium schema bridge are used.
fn build_write_schema(
    arrow_schema: &ArrowSchema,
    batches: &[RecordBatch],
    optimize: bool,
    zstd_level: Option<i32>,
) -> PyResult<Schema> {
    // Default schema (logical types + default encodings) from the Arrow schema.
    let default_schema = schema_from_arrow(arrow_schema).map_err(he_err)?;

    if !optimize || batches.is_empty() {
        return Ok(default_schema);
    }

    // Convert each column (concatenated across all batches) to a LogicalColumn,
    // then let the optimizer pick pipelines. We feed the whole table as the
    // sample, which gives the most accurate measurement.
    let col_count = arrow_schema.fields().len();
    let mut sample: Vec<(String, LogicalType, LogicalColumn)> = Vec::with_capacity(col_count);

    for (i, spec) in default_schema.columns.iter().enumerate() {
        let lt = spec.logical_type.clone();
        // Concatenate this column across batches via arrow::compute::concat.
        let arrays: Vec<ArrayRef> = batches.iter().map(|b| b.column(i).clone()).collect();
        let array_refs: Vec<&dyn arrow::array::Array> =
            arrays.iter().map(|a| a.as_ref()).collect();
        let concatenated: ArrayRef = if array_refs.len() == 1 {
            arrays[0].clone()
        } else {
            arrow::compute::concat(&array_refs)
                .map_err(|e| PyValueError::new_err(format!("arrow concat failed: {e}")))?
        };
        let lc = from_arrow_array(&concatenated, &lt).map_err(he_err)?;
        sample.push((spec.name.clone(), lt, lc));
    }

    let mut optimizer = Optimizer::new();
    if let Some(level) = zstd_level {
        optimizer = optimizer.with_zstd_level(level);
    }
    optimizer.optimize(sample).map_err(he_err)
}

/// Write a pyarrow Table / RecordBatch / pandas DataFrame to a ``.he`` file.
///
/// Parameters
/// ----------
/// path : str
///     Destination file path (created or overwritten).
/// table : pyarrow.Table | pyarrow.RecordBatch | pandas.DataFrame
///     Tabular data. Nullable, nested (Struct/List/Map) and semantic
///     (Date/Datetime/Decimal) columns are all supported.
/// optimize : bool, default True
///     When true, run Helium's optimizer to choose the smallest per-column
///     encoding pipeline (measured on the data). When false, use fast default
///     encodings.
/// stripe_rows : int, optional
///     When set, the table is written in stripes (row groups) of at most this
///     many rows. Enables bounded-memory streaming for large tables and gives
///     readers per-stripe pruning. When unset the whole table is one stripe.
/// zstd_level : int, optional
///     Global zstd compression level (1–22) for the chosen pipelines. Omitted
///     uses the zstd default (3). The level is a single global setting, not
///     picked per column. Only takes effect when ``optimize`` is True (the
///     default-encoding path always uses level 3).
#[pyfunction]
#[pyo3(signature = (path, table, optimize=true, stripe_rows=None, zstd_level=None))]
fn write_table(
    py: Python<'_>,
    path: &str,
    table: &Bound<'_, PyAny>,
    optimize: bool,
    stripe_rows: Option<usize>,
    zstd_level: Option<i32>,
) -> PyResult<()> {
    let (arrow_schema, batches) = coerce_to_batches(py, table)?;
    let schema = build_write_schema(&arrow_schema, &batches, optimize, zstd_level)?;
    let registry = CoderRegistry::with_builtins();

    let file = File::create(path).map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;
    let mut writer = HeliumWriter::new(file, schema, &registry).map_err(he_err)?;

    let col_names: Vec<String> = arrow_schema
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();

    // Plan the stripe slices. Each entry is (batch_idx, offset, len).
    let stripe_plan = plan_stripes(&batches, stripe_rows);

    if stripe_plan.is_empty() {
        // No rows at all: write one empty stripe so the schema is recorded.
        for (i, name) in col_names.iter().enumerate() {
            // Build an empty LogicalColumn per column from an empty slice.
            let lt = &writer.schema().columns[i].logical_type.clone();
            let empty = empty_logical_column(lt)?;
            writer.write_column(name, empty).map_err(he_err)?;
        }
        writer.finish().map_err(he_err)?;
        return Ok(());
    }

    let total = stripe_plan.len();
    for (s, (batch_idx, offset, len)) in stripe_plan.into_iter().enumerate() {
        let batch = &batches[batch_idx];
        for (i, name) in col_names.iter().enumerate() {
            let lt = writer.schema().columns[i].logical_type.clone();
            let sliced: ArrayRef = batch.column(i).slice(offset, len);
            let lc = from_arrow_array(&sliced, &lt).map_err(he_err)?;
            writer.write_column(name, lc).map_err(he_err)?;
        }
        if s + 1 < total {
            writer.finish_stripe().map_err(he_err)?;
        }
    }
    writer.finish().map_err(he_err)?;
    Ok(())
}

/// Produce an empty `LogicalColumn` matching a logical type (zero rows).
///
/// Used only on the all-empty-table path; built by converting an empty Arrow
/// array of the matching DataType.
fn empty_logical_column(lt: &LogicalType) -> PyResult<LogicalColumn> {
    let arrow_dt = helium::arrow::schema::logical_type_to_arrow(lt);
    let empty: ArrayRef = arrow::array::new_empty_array(&arrow_dt);
    from_arrow_array(&empty, lt).map_err(he_err)
}

/// Split the batches into stripe slices of at most `stripe_rows` rows each.
///
/// Returns a list of `(batch_idx, offset_within_batch, len)`. When
/// `stripe_rows` is `None` each batch becomes exactly one stripe.
fn plan_stripes(
    batches: &[RecordBatch],
    stripe_rows: Option<usize>,
) -> Vec<(usize, usize, usize)> {
    let mut plan = Vec::new();
    for (bi, batch) in batches.iter().enumerate() {
        let rows = batch.num_rows();
        if rows == 0 {
            continue;
        }
        match stripe_rows {
            None => plan.push((bi, 0, rows)),
            Some(0) => plan.push((bi, 0, rows)),
            Some(chunk) => {
                let mut off = 0;
                while off < rows {
                    let len = chunk.min(rows - off);
                    plan.push((bi, off, len));
                    off += len;
                }
            }
        }
    }
    plan
}

/// Read a ``.he`` file into a pyarrow Table, with optional projection and
/// stripe-range selection for bounded-memory reads.
///
/// Parameters
/// ----------
/// path : str
///     Path to a ``.he`` file.
/// columns : list[str], optional
///     When given, only these columns are decoded (column pruning — Helium
///     reads only the bytes for the requested columns). Order is preserved.
/// stripe_range : tuple[int, int], optional
///     ``(start, end)`` half-open stripe index range. When given, only stripes
///     ``start..end`` are read. Defaults to all stripes.
///
/// Returns
/// -------
/// pyarrow.Table
///     A table with one chunk per stripe. Nullable, nested, and semantic
///     columns are reconstructed via Helium's Arrow bridge.
#[pyfunction]
#[pyo3(signature = (path, columns=None, stripe_range=None))]
fn read_table(
    py: Python<'_>,
    path: &str,
    columns: Option<Vec<String>>,
    stripe_range: Option<(usize, usize)>,
) -> PyResult<PyObject> {
    let file = File::open(path).map_err(|e| PyValueError::new_err(format!("{e}")))?;
    let reader = BufReader::new(file);
    let registry = CoderRegistry::with_builtins();
    let mut helium_reader = HeliumReader::new(reader, &registry).map_err(he_err)?;

    let full_schema = helium_reader.schema().clone();
    let stripe_count = helium_reader.stripe_count();

    // Resolve the projected column specs (preserving requested order).
    let projected: Vec<ColumnSpec> = match &columns {
        None => full_schema.columns.clone(),
        Some(names) => {
            let mut out = Vec::with_capacity(names.len());
            for n in names {
                let spec = full_schema
                    .columns
                    .iter()
                    .find(|c| &c.name == n)
                    .ok_or_else(|| {
                        PyValueError::new_err(format!("column {n:?} not found in {path:?}"))
                    })?;
                out.push(spec.clone());
            }
            out
        }
    };

    // Build the Arrow schema for the projected set.
    let proj_schema = Schema::new(projected.clone());
    let arrow_schema = Arc::new(schema_to_arrow(&proj_schema));

    // Resolve the stripe range.
    let (start, end) = match stripe_range {
        None => (0usize, stripe_count),
        Some((s, e)) => {
            if s > e || e > stripe_count {
                return Err(PyValueError::new_err(format!(
                    "stripe_range ({s}, {e}) out of bounds; file has {stripe_count} stripes"
                )));
            }
            (s, e)
        }
    };

    // Read each stripe as a RecordBatch over the projected columns only.
    let mut batches: Vec<RecordBatch> = Vec::new();
    for stripe_idx in start..end {
        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(projected.len());
        for spec in &projected {
            let col = helium_reader
                .read_column_at_stripe(&spec.name, stripe_idx)
                .map_err(he_err)?;
            let arr = to_arrow_array(&col, &spec.logical_type).map_err(he_err)?;
            arrays.push(arr);
        }
        let batch = RecordBatch::try_new(arrow_schema.clone(), arrays)
            .map_err(|e| PyValueError::new_err(format!("RecordBatch::try_new: {e}")))?;
        batches.push(batch);
    }

    // Assemble a pyarrow.Table from the batches (one chunk per stripe).
    let pa = py.import("pyarrow")?;
    let table_cls = pa.getattr("Table")?;
    let py_schema = arrow_schema_to_py(py, &arrow_schema)?;

    let py_batches = PyList::empty(py);
    for b in &batches {
        py_batches.append(b.to_pyarrow(py)?)?;
    }
    let kwargs = PyDict::new(py);
    kwargs.set_item("schema", py_schema)?;
    let table = table_cls.call_method("from_batches", (py_batches,), Some(&kwargs))?;
    Ok(table.into())
}

/// Convert a Rust Arrow schema to a pyarrow Schema object.
fn arrow_schema_to_py<'py>(py: Python<'py>, schema: &ArrowSchema) -> PyResult<Bound<'py, PyAny>> {
    // Build a pyarrow.schema from fields; ToPyArrow is implemented for Field.
    let pa = py.import("pyarrow")?;
    let fields = PyList::empty(py);
    for f in schema.fields() {
        let arc_field: ArrowField = (**f).clone();
        let py_field = arc_field.to_pyarrow(py)?;
        fields.append(py_field)?;
    }
    pa.call_method1("schema", (fields,))
}

// ---------------------------------------------------------------------------
// Module entry point
// ---------------------------------------------------------------------------

/// `pyhelium` — Python bindings for the Helium columnar compression library.
#[pymodule]
fn pyhelium(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compress, m)?)?;
    m.add_function(wrap_pyfunction!(decompress, m)?)?;
    m.add_function(wrap_pyfunction!(write_he, m)?)?;
    m.add_function(wrap_pyfunction!(read_he, m)?)?;
    m.add_function(wrap_pyfunction!(write_table, m)?)?;
    m.add_function(wrap_pyfunction!(read_table, m)?)?;
    Ok(())
}
