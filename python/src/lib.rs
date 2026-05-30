//! `pyhelium` — Python bindings for the Helium columnar compression library.
//!
//! Exposes four top-level functions:
//!
//! - `compress(values, dtype=None) -> bytes`
//! - `decompress(buf: bytes) -> numpy.ndarray`
//! - `write_he(path: str, data: dict) -> None`
//! - `read_he(path: str) -> dict`
//!
//! Only flat (non-nested) column types are supported today; Arrow/pandas
//! interop and the wider type set are tracked in `docs/ROADMAP.md`.

use std::fs::File;
use std::io::BufReader;

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::{PyNotImplementedError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyString};

use helium::{
    ColumnData, ColumnSpec, CoderRegistry, CoderSpec, DataType, HeliumReader, HeliumWriter,
    LogicalColumn, Schema,
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
fn list_to_column_data(_py: Python<'_>, list: &Bound<'_, PyList>, dtype: &str) -> PyResult<ColumnData> {
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
            // PyArray1::from_vec returns Bound<'_, PyArray1<T>> in PyO3 0.23.
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
        let spec = ColumnSpec::primitive("__placeholder__", data_type, default_prim_coders(data_type));
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
                    PyTypeError::new_err(
                        "list elements must all be str for a string column",
                    )
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
                        PyTypeError::new_err(
                            "list elements must all be bytes for a binary column",
                        )
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
// Public API
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
///
/// Raises
/// ------
/// TypeError
///     If *values* is neither a numpy array nor a list, or if the dtype is
///     unsupported or missing for a plain list.
/// ValueError
///     If compression fails.
#[pyfunction]
#[pyo3(signature = (values, dtype=None))]
fn compress(py: Python<'_>, values: &Bound<'_, PyAny>, dtype: Option<&str>) -> PyResult<Py<PyBytes>> {
    let cd = if let Ok(list) = values.downcast::<PyList>() {
        let dt = dtype.ok_or_else(|| {
            PyTypeError::new_err(
                "dtype argument required when values is a list; e.g. dtype='i64'",
            )
        })?;
        list_to_column_data(py, list, dt)?
    } else if values.hasattr("dtype").unwrap_or(false)
        && values.hasattr("ravel").unwrap_or(false)
    {
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
///
/// Parameters
/// ----------
/// buf : bytes
///     Self-describing compressed bytes previously returned by :func:`compress`.
///
/// Returns
/// -------
/// numpy.ndarray
///     A 1-D numpy array with the matching numeric dtype.
///
/// Raises
/// ------
/// ValueError
///     If *buf* is not valid `HEC0` data or decompression fails.
#[pyfunction]
fn decompress(py: Python<'_>, buf: &[u8]) -> PyResult<PyObject> {
    let cd = helium::decompress(buf).map_err(he_err)?;
    let arr = column_data_to_array(py, cd)?;
    Ok(arr.into())
}

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
/// Raises
/// ------
/// TypeError
///     If a column value type is not supported.
/// ValueError
///     If writing fails (I/O error or codec error).
/// NotImplementedError
///     If nested types (Struct / List / Map / Union) are detected
///     (write_he currently supports flat columns only).
#[pyfunction]
fn write_he(py: Python<'_>, path: &str, data: &Bound<'_, PyDict>) -> PyResult<()> {
    // Build schema + collect columns.
    let mut specs: Vec<ColumnSpec> = Vec::new();
    let mut columns: Vec<(String, LogicalColumn)> = Vec::new();

    for (key, value) in data.iter() {
        let name: String = key.extract().map_err(|_| {
            PyTypeError::new_err("column names must be strings")
        })?;

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
/// Parameters
/// ----------
/// path : str
///     Path to a ``.he`` file.
///
/// Returns
/// -------
/// dict[str, numpy.ndarray | list]
///     Column name → values.  Primitive columns are returned as numpy
///     ndarrays; ``Utf8`` columns as ``list[str]``; ``Binary`` columns as
///     ``list[bytes]``.
///
/// Raises
/// ------
/// ValueError
///     If reading fails (I/O, corrupt data, etc.).
/// NotImplementedError
///     If the file contains nested column types not yet supported
///     (Struct, List, Map, Union, Nullable, etc.).
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
/// Supports only flat (non-nested) columns today.
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
            let list = PyList::new(
                py,
                blobs.iter().map(|b| PyBytes::new(py, b)),
            )?;
            Ok(list.into())
        }
        // Nested / nullable / dict / semantic types are not yet bridged to
        // Python — see the Arrow-interop item in docs/ROADMAP.md.
        LogicalColumn::ArrayOf { .. }
        | LogicalColumn::ArrayOfUtf8 { .. }
        | LogicalColumn::NullablePrim { .. }
        | LogicalColumn::NullableUtf8 { .. }
        | LogicalColumn::NullableBinary { .. }
        | LogicalColumn::Dictionary { .. }
        | LogicalColumn::Struct { .. }
        | LogicalColumn::List { .. }
        | LogicalColumn::Map { .. }
        | LogicalColumn::Nullable { .. }
        | LogicalColumn::Union { .. }
        | LogicalColumn::Decimal128 { .. }
        | LogicalColumn::Date32 { .. }
        | LogicalColumn::Date64 { .. }
        | LogicalColumn::Datetime { .. } => Err(PyNotImplementedError::new_err(
            "read_he currently supports only flat Primitive/Utf8/Binary columns; \
             nested, nullable, dict, and semantic-type columns are planned \
             (see docs/ROADMAP.md)",
        )),
    }
}

// ---------------------------------------------------------------------------
// Module entry point
// ---------------------------------------------------------------------------

/// `pyhelium` — Python bindings for the Helium columnar compression library.
///
/// Current API surface:
/// - :func:`compress` / :func:`decompress` — HEC0 self-describing codec.
/// - :func:`write_he` / :func:`read_he` — flat-column ``.he`` file I/O.
#[pymodule]
fn pyhelium(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compress, m)?)?;
    m.add_function(wrap_pyfunction!(decompress, m)?)?;
    m.add_function(wrap_pyfunction!(write_he, m)?)?;
    m.add_function(wrap_pyfunction!(read_he, m)?)?;
    Ok(())
}
