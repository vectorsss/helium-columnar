//! Helium DuckDB loadable extension.
//!
//! Registers the `read_he(path VARCHAR)` table function so DuckDB can query
//! `.he` files directly:
//!
//! ```sql
//! LOAD 'helium_duckdb';
//! SELECT * FROM read_he('path/to/file.he') LIMIT 5;
//! SELECT count(*) FROM read_he('path/to/file.he');
//! SELECT col_a, col_b FROM read_he('path/to/file.he') WHERE col_a > 100;
//! ```
//!
//! ## Current limitations (tracked in `docs/ROADMAP.md`)
//! - No predicate / projection pushdown — DuckDB reads every column and every
//!   row, then applies filters itself. Closing this is the highest-value item.
//! - Catalog-mode (v6) files error out with an explicit message.
//! - Nested types (Struct, List, Map, Union) are not yet projected through
//!   DuckDB's complex-type vectors; they error with a clear message.
//! - Multi-stripe files are read stripe-by-stripe in the `func` callback.

use std::cell::UnsafeCell;

use duckdb::core::{DataChunkHandle, Inserter, LogicalTypeHandle, LogicalTypeId};
use duckdb::vtab::{BindInfo, InitInfo, TableFunctionInfo, VTab};
use duckdb::{Connection, Result as DuckResult, duckdb_entrypoint_c_api};

use helium::{
    ColumnData, CoderRegistry, DateUnit, HeliumReader, LogicalColumn, LogicalType, TimeUnit,
};

// ---------------------------------------------------------------------------
// Bind phase — open file, read schema, emit DuckDB column types
// ---------------------------------------------------------------------------

/// Data produced during the bind phase and shared (read-only) with the scan.
pub struct HeBindData {
    /// Path to the `.he` file.
    path: String,
    /// Column names, in schema order.
    col_names: Vec<String>,
    /// Total number of stripes in the file.
    stripe_count: usize,
    /// Column count.
    col_count: usize,
}

// SAFETY: HeBindData fields are all Send; it is constructed once and then only
// read (never mutated) during the scan phase.
unsafe impl Send for HeBindData {}
unsafe impl Sync for HeBindData {}

// ---------------------------------------------------------------------------
// Init phase — scan cursor
// ---------------------------------------------------------------------------

/// Per-scan cursor held inside an `UnsafeCell` so we can mutate state through
/// the shared `&HeInitData` reference that DuckDB's vtab API gives us.
///
/// SAFETY: DuckDB calls `func` sequentially for a given scan — no concurrent
/// calls on the same cursor.  The `UnsafeCell` is the only way to achieve
/// interior mutability here without `Mutex` overhead.
pub struct HeInitData {
    inner: UnsafeCell<HeInitDataInner>,
}

struct HeInitDataInner {
    /// Next stripe index to load (0-based).
    next_stripe: usize,
    /// Columns decoded for the current stripe, one entry per logical column.
    current_columns: Vec<LogicalColumn>,
    /// Row offset within the current stripe (i.e. next unread row).
    row_in_stripe: usize,
    /// Total rows in the current stripe.
    stripe_rows: usize,
    /// True once all stripes have been exhausted.
    done: bool,
}

// SAFETY: DuckDB accesses the init data from a single scan thread at a time.
unsafe impl Send for HeInitData {}
unsafe impl Sync for HeInitData {}

// ---------------------------------------------------------------------------
// VTab implementation
// ---------------------------------------------------------------------------

/// Table function implementation for `read_he(path VARCHAR)`.
pub struct HeVTab;

impl VTab for HeVTab {
    type InitData = HeInitData;
    type BindData = HeBindData;

    fn bind(bind: &BindInfo) -> DuckResult<Self::BindData, Box<dyn std::error::Error>> {
        let path = bind.get_parameter(0).to_string();

        let file = std::fs::File::open(&path)
            .map_err(|e| format!("read_he: cannot open '{}': {}", path, e))?;
        let registry = CoderRegistry::default();
        let reader = HeliumReader::new(file, &registry)
            .map_err(|e| format!("read_he: cannot read '{}': {}", path, e))?;

        let schema = reader.schema().clone();
        let stripe_count = reader.stripe_count();

        let mut col_names = Vec::new();

        for col_spec in &schema.columns {
            let ltype = logical_type_to_duckdb(&col_spec.logical_type)?;
            bind.add_result_column(col_spec.name.as_str(), ltype);
            col_names.push(col_spec.name.clone());
        }

        let col_count = col_names.len();

        Ok(HeBindData { path, col_names, stripe_count, col_count })
    }

    fn init(_: &InitInfo) -> DuckResult<Self::InitData, Box<dyn std::error::Error>> {
        Ok(HeInitData {
            inner: UnsafeCell::new(HeInitDataInner {
                next_stripe: 0,
                current_columns: Vec::new(),
                row_in_stripe: 0,
                stripe_rows: 0,
                done: false,
            }),
        })
    }

    fn func(
        func: &TableFunctionInfo<Self>,
        output: &mut DataChunkHandle,
    ) -> DuckResult<(), Box<dyn std::error::Error>> {
        let bind_data = func.get_bind_data();
        // SAFETY: DuckDB guarantees sequential calls per scan; no aliasing.
        let state = unsafe { &mut *func.get_init_data().inner.get() };

        if state.done {
            output.set_len(0);
            return Ok(());
        }

        // Load the next stripe if the current one is exhausted.
        if state.row_in_stripe >= state.stripe_rows {
            if state.next_stripe >= bind_data.stripe_count {
                state.done = true;
                output.set_len(0);
                return Ok(());
            }

            let stripe_idx = state.next_stripe;
            state.next_stripe += 1;

            let file = std::fs::File::open(&bind_data.path).map_err(|e| {
                format!("read_he: cannot re-open '{}': {}", bind_data.path, e)
            })?;
            let registry = CoderRegistry::default();
            let mut reader = HeliumReader::new(file, &registry).map_err(|e| {
                format!("read_he: cannot re-read '{}': {}", bind_data.path, e)
            })?;

            let mut loaded = Vec::with_capacity(bind_data.col_count);
            let mut stripe_rows = 0usize;
            for name in &bind_data.col_names {
                let column = reader
                    .read_column_at_stripe(name.as_str(), stripe_idx)
                    .map_err(|e| {
                        format!("read_he: stripe {} col '{}': {}", stripe_idx, name, e)
                    })?;
                stripe_rows = column.row_count();
                loaded.push(column);
            }

            state.current_columns = loaded;
            state.row_in_stripe = 0;
            state.stripe_rows = stripe_rows;

            if stripe_rows == 0 {
                // Empty stripe — try again with the next one.
                return Self::func(func, output);
            }
        }

        // Emit up to DUCKDB_VECTOR_SIZE rows.
        let chunk_size = (state.stripe_rows - state.row_in_stripe).min(2048);
        let row_start = state.row_in_stripe;

        for col_idx in 0..bind_data.col_count {
            write_column_to_chunk(
                output,
                col_idx,
                &state.current_columns[col_idx],
                row_start,
                chunk_size,
            )?;
        }

        output.set_len(chunk_size);
        state.row_in_stripe += chunk_size;

        Ok(())
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![LogicalTypeHandle::from(LogicalTypeId::Varchar)])
    }
}

// ---------------------------------------------------------------------------
// Type mapping: Helium LogicalType → DuckDB LogicalTypeHandle
// ---------------------------------------------------------------------------

fn logical_type_to_duckdb(
    lt: &LogicalType,
) -> Result<LogicalTypeHandle, Box<dyn std::error::Error>> {
    use LogicalType::*;
    match lt {
        Primitive { data_type } => Ok(primitive_dt_to_duckdb(*data_type)),

        Utf8 => Ok(LogicalTypeHandle::from(LogicalTypeId::Varchar)),
        Binary => Ok(LogicalTypeHandle::from(LogicalTypeId::Blob)),

        // v2 legacy array types — not yet supported
        ArrayOf { .. } | ArrayOfUtf8 => Err(
            "read_he: ArrayOf / ArrayOfUtf8 (v2 legacy) not yet supported; \
             convert the schema to v3 List first with the helium CLI"
                .into(),
        ),

        // v2 legacy nullable / dict
        NullablePrim { data_type } => Ok(primitive_dt_to_duckdb(*data_type)),
        NullableUtf8 => Ok(LogicalTypeHandle::from(LogicalTypeId::Varchar)),
        NullableBinary => Ok(LogicalTypeHandle::from(LogicalTypeId::Blob)),

        // v3 nullable wrapper — DuckDB columns are always nullable, unwrap the inner type
        Nullable { inner } => logical_type_to_duckdb(inner),

        // Dictionary maps to its inner value type (materialized on read).
        Dictionary { inner } => logical_type_to_duckdb(inner),

        // Deferred: nested types (see docs/ROADMAP.md)
        Struct { .. } => Err(
            "read_he: Struct columns not yet supported; \
             flatten with `helium convert --csv-strict`"
                .into(),
        ),
        List { .. } => Err(
            "read_he: List columns not yet supported".into(),
        ),
        Map { .. } => Err("read_he: Map columns not yet supported".into()),
        Union { .. } => Err("read_he: Union columns not yet supported".into()),

        // Semantic types
        Decimal128 { precision, scale } => {
            Ok(LogicalTypeHandle::decimal(*precision, *scale))
        }

        Date { unit: DateUnit::Days } => Ok(LogicalTypeHandle::from(LogicalTypeId::Date)),
        Date { unit: DateUnit::Millis } => {
            // DuckDB DATE is day-resolution; represent ms-date as BIGINT
            Ok(LogicalTypeHandle::from(LogicalTypeId::Bigint))
        }

        Datetime { unit, timezone } => match (unit, timezone.as_deref()) {
            (_, Some(_)) => Ok(LogicalTypeHandle::from(LogicalTypeId::TimestampTZ)),
            (TimeUnit::Seconds, None) => {
                Ok(LogicalTypeHandle::from(LogicalTypeId::TimestampS))
            }
            (TimeUnit::Millis, None) => {
                Ok(LogicalTypeHandle::from(LogicalTypeId::TimestampMs))
            }
            (TimeUnit::Micros, None) => {
                Ok(LogicalTypeHandle::from(LogicalTypeId::Timestamp))
            }
            (TimeUnit::Nanos, None) => {
                Ok(LogicalTypeHandle::from(LogicalTypeId::TimestampNs))
            }
        },
    }
}

/// Map a Helium scalar `DataType` to a DuckDB `LogicalTypeHandle`.
fn primitive_dt_to_duckdb(dt: helium::DataType) -> LogicalTypeHandle {
    use helium::DataType::*;
    let id = match dt {
        I8 => LogicalTypeId::Tinyint,
        I16 => LogicalTypeId::Smallint,
        I32 => LogicalTypeId::Integer,
        I64 => LogicalTypeId::Bigint,
        U8 => LogicalTypeId::UTinyint,
        U16 => LogicalTypeId::USmallint,
        U32 => LogicalTypeId::UInteger,
        U64 => LogicalTypeId::UBigint,
        F32 => LogicalTypeId::Float,
        F64 => LogicalTypeId::Double,
        Bytes => LogicalTypeId::Blob,
    };
    LogicalTypeHandle::from(id)
}

// ---------------------------------------------------------------------------
// Scan helpers — write rows into DuckDB output vectors
// ---------------------------------------------------------------------------

/// Write rows `[row_start, row_start + chunk_size)` from `column` into the
/// DuckDB output chunk at the given column index.
fn write_column_to_chunk(
    output: &mut DataChunkHandle,
    col_idx: usize,
    column: &LogicalColumn,
    row_start: usize,
    chunk_size: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    match column {
        LogicalColumn::Primitive(data) => {
            write_flat_data(output, col_idx, data, row_start, chunk_size);
        }

        LogicalColumn::Utf8(strings) => {
            let vec = output.flat_vector(col_idx);
            for (i, s) in strings[row_start..row_start + chunk_size].iter().enumerate() {
                vec.insert(i, s.as_str());
            }
        }

        LogicalColumn::Binary(blobs) => {
            let vec = output.flat_vector(col_idx);
            for (i, blob) in blobs[row_start..row_start + chunk_size].iter().enumerate() {
                vec.insert(i, blob.as_slice());
            }
        }

        // v2 nullable primitives (present bitmap covers all rows, values are dense)
        LogicalColumn::NullablePrim { present, values } => {
            write_flat_data_nullable(output, col_idx, values, present, row_start, chunk_size);
        }

        LogicalColumn::NullableUtf8 { present, strings } => {
            let mut vec = output.flat_vector(col_idx);
            for (i, (is_present, s)) in present[row_start..row_start + chunk_size]
                .iter()
                .zip(strings[row_start..row_start + chunk_size].iter())
                .enumerate()
            {
                if *is_present {
                    vec.insert(i, s.as_str());
                } else {
                    vec.set_null(i);
                }
            }
        }

        LogicalColumn::NullableBinary { present, blobs } => {
            let mut vec = output.flat_vector(col_idx);
            for (i, (is_present, blob)) in present[row_start..row_start + chunk_size]
                .iter()
                .zip(blobs[row_start..row_start + chunk_size].iter())
                .enumerate()
            {
                if *is_present {
                    vec.insert(i, blob.as_slice());
                } else {
                    vec.set_null(i);
                }
            }
        }

        LogicalColumn::Dictionary { dictionary, indices } => match dictionary.as_ref() {
            LogicalColumn::Primitive(cd) => {
                expand_dict_prim(output, col_idx, cd, indices, row_start, chunk_size);
            }
            LogicalColumn::Utf8(dict) => {
                let vec = output.flat_vector(col_idx);
                for (i, &idx) in indices[row_start..row_start + chunk_size].iter().enumerate() {
                    vec.insert(i, dict[idx as usize].as_str());
                }
            }
            _ => {
                return Err("read_he: Dictionary with a non-scalar inner type is not yet \
                            supported (only Primitive / Utf8)"
                    .into());
            }
        },

        // v3 nullable — compacted inner values
        LogicalColumn::Nullable { present, value } => {
            write_nullable_compacted(output, col_idx, present, value, row_start, chunk_size)?;
        }

        // Semantic types
        LogicalColumn::Decimal128 { values } => {
            // DuckDB DECIMAL is backed by hugeint { lower: u64, upper: i64 }
            #[repr(C)]
            struct HugeInt {
                lower: u64,
                upper: i64,
            }
            let vec = output.flat_vector(col_idx);
            // SAFETY: HugeInt is repr(C) with no padding; we write exactly
            // chunk_size slots that DuckDB allocated for this column.
            let ptr = vec.as_mut_ptr::<HugeInt>();
            for (i, &v) in values[row_start..row_start + chunk_size].iter().enumerate() {
                unsafe {
                    ptr.add(i).write(HugeInt {
                        lower: v as u64,
                        upper: (v >> 64) as i64,
                    });
                }
            }
        }

        LogicalColumn::Date32 { values } => {
            // DuckDB DATE is stored as duckdb_date { days: int32_t }.
            #[repr(C)]
            struct DuckDate {
                days: i32,
            }
            let vec = output.flat_vector(col_idx);
            let ptr = vec.as_mut_ptr::<DuckDate>();
            for (i, &v) in values[row_start..row_start + chunk_size].iter().enumerate() {
                // SAFETY: same as Decimal128 above.
                unsafe {
                    ptr.add(i).write(DuckDate { days: v });
                }
            }
        }

        LogicalColumn::Date64 { values } => {
            // Mapped to BIGINT (milliseconds since epoch).
            let vec = output.flat_vector(col_idx);
            let ptr = vec.as_mut_ptr::<i64>();
            for (i, &v) in values[row_start..row_start + chunk_size].iter().enumerate() {
                // SAFETY: i64 is the backing type for BIGINT.
                unsafe {
                    ptr.add(i).write(v);
                }
            }
        }

        LogicalColumn::Datetime { values } => {
            // DuckDB TIMESTAMP* types all store a single i64 in native units.
            // duckdb_timestamp { micros: int64_t } — we pass the raw value as-is.
            #[repr(C)]
            struct DuckTimestamp {
                micros: i64,
            }
            let vec = output.flat_vector(col_idx);
            let ptr = vec.as_mut_ptr::<DuckTimestamp>();
            for (i, &v) in values[row_start..row_start + chunk_size].iter().enumerate() {
                // SAFETY: DuckTimestamp is repr(C) single i64.
                unsafe {
                    ptr.add(i).write(DuckTimestamp { micros: v });
                }
            }
        }

        LogicalColumn::ArrayOf { .. } | LogicalColumn::ArrayOfUtf8 { .. } => {
            return Err("read_he: ArrayOf not supported".into());
        }

        LogicalColumn::Struct { .. }
        | LogicalColumn::List { .. }
        | LogicalColumn::Map { .. }
        | LogicalColumn::Union { .. } => {
            return Err(
                "read_he: nested type in scan — bind should have rejected it (bug)".into(),
            );
        }
    }
    Ok(())
}

/// Write a flat (non-nullable) `ColumnData` slice into an output chunk column.
fn write_flat_data(
    output: &mut DataChunkHandle,
    col_idx: usize,
    data: &ColumnData,
    row_start: usize,
    chunk_size: usize,
) {
    let vec = output.flat_vector(col_idx);
    macro_rules! copy_slice {
        ($vals:expr, $ty:ty) => {{
            let ptr = vec.as_mut_ptr::<$ty>();
            // SAFETY: DuckDB allocated `capacity()` slots of the column's
            // native type; we write exactly chunk_size ≤ capacity() of them.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    $vals[row_start..].as_ptr(),
                    ptr,
                    chunk_size,
                );
            }
        }};
    }
    match data {
        ColumnData::I8(v) => copy_slice!(v, i8),
        ColumnData::I16(v) => copy_slice!(v, i16),
        ColumnData::I32(v) => copy_slice!(v, i32),
        ColumnData::I64(v) => copy_slice!(v, i64),
        ColumnData::U8(v) => copy_slice!(v, u8),
        ColumnData::U16(v) => copy_slice!(v, u16),
        ColumnData::U32(v) => copy_slice!(v, u32),
        ColumnData::U64(v) => copy_slice!(v, u64),
        ColumnData::F32(v) => copy_slice!(v, f32),
        ColumnData::F64(v) => copy_slice!(v, f64),
        ColumnData::Bytes(v) => {
            // Insert as blob.
            vec.insert(0, &v[row_start..row_start + chunk_size]);
        }
    }
}

/// Write a nullable `ColumnData` (v2 `NullablePrim` semantics — dense storage,
/// null bitmap separate) into the output chunk.
fn write_flat_data_nullable(
    output: &mut DataChunkHandle,
    col_idx: usize,
    data: &ColumnData,
    present: &[bool],
    row_start: usize,
    chunk_size: usize,
) {
    let mut vec = output.flat_vector(col_idx);
    macro_rules! write_with_nulls {
        ($vals:expr, $ty:ty) => {{
            let ptr = vec.as_mut_ptr::<$ty>();
            for (i, (&p, &v)) in present[row_start..row_start + chunk_size]
                .iter()
                .zip($vals[row_start..row_start + chunk_size].iter())
                .enumerate()
            {
                if p {
                    // SAFETY: writing within the DuckDB-allocated slot.
                    unsafe { ptr.add(i).write(v) };
                } else {
                    vec.set_null(i);
                }
            }
        }};
    }
    match data {
        ColumnData::I8(v) => write_with_nulls!(v, i8),
        ColumnData::I16(v) => write_with_nulls!(v, i16),
        ColumnData::I32(v) => write_with_nulls!(v, i32),
        ColumnData::I64(v) => write_with_nulls!(v, i64),
        ColumnData::U8(v) => write_with_nulls!(v, u8),
        ColumnData::U16(v) => write_with_nulls!(v, u16),
        ColumnData::U32(v) => write_with_nulls!(v, u32),
        ColumnData::U64(v) => write_with_nulls!(v, u64),
        ColumnData::F32(v) => write_with_nulls!(v, f32),
        ColumnData::F64(v) => write_with_nulls!(v, f64),
        ColumnData::Bytes(_) => { /* Not reachable for NullablePrim */ }
    }
}

/// Expand a `Dictionary { Primitive }` column (dictionary + index array) into
/// the output chunk.
fn expand_dict_prim(
    output: &mut DataChunkHandle,
    col_idx: usize,
    dictionary: &ColumnData,
    indices: &[u32],
    row_start: usize,
    chunk_size: usize,
) {
    let vec = output.flat_vector(col_idx);
    let idx_slice = &indices[row_start..row_start + chunk_size];
    macro_rules! expand_typed {
        ($dict:expr, $ty:ty) => {{
            let ptr = vec.as_mut_ptr::<$ty>();
            for (i, &idx) in idx_slice.iter().enumerate() {
                // SAFETY: DuckDB slot within capacity; dict access is
                // bounds-checked by the helium encoder (panic on corrupt data).
                unsafe { ptr.add(i).write($dict[idx as usize]) };
            }
        }};
    }
    match dictionary {
        ColumnData::I8(d) => expand_typed!(d, i8),
        ColumnData::I16(d) => expand_typed!(d, i16),
        ColumnData::I32(d) => expand_typed!(d, i32),
        ColumnData::I64(d) => expand_typed!(d, i64),
        ColumnData::U8(d) => expand_typed!(d, u8),
        ColumnData::U16(d) => expand_typed!(d, u16),
        ColumnData::U32(d) => expand_typed!(d, u32),
        ColumnData::U64(d) => expand_typed!(d, u64),
        ColumnData::F32(d) => expand_typed!(d, f32),
        ColumnData::F64(d) => expand_typed!(d, f64),
        ColumnData::Bytes(_) => { /* Not reachable for a primitive dictionary */ }
    }
}

/// Write a v3 `Nullable { present, value }` (compacted inner) column into the
/// output chunk at `[row_start, row_start + chunk_size)`.
///
/// The inner `value` box contains only the non-null rows in compacted form.
/// We count how many non-null rows precede `row_start` to find our offset
/// into the compacted buffer.
fn write_nullable_compacted(
    output: &mut DataChunkHandle,
    col_idx: usize,
    present: &[bool],
    value: &LogicalColumn,
    row_start: usize,
    chunk_size: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let compact_start: usize = present[..row_start].iter().filter(|&&p| p).count();

    match value {
        LogicalColumn::Primitive(data) => {
            let mut vec = output.flat_vector(col_idx);
            let mut ci = compact_start;
            macro_rules! write_compact_nullable {
                ($vals:expr, $ty:ty) => {{
                    let ptr = vec.as_mut_ptr::<$ty>();
                    for (i, &p) in
                        present[row_start..row_start + chunk_size].iter().enumerate()
                    {
                        if p {
                            // SAFETY: ci < compacted length (helium invariant);
                            // ptr points into DuckDB-allocated vector storage.
                            unsafe { ptr.add(i).write($vals[ci]) };
                            ci += 1;
                        } else {
                            vec.set_null(i);
                        }
                    }
                }};
            }
            match data {
                ColumnData::I8(v) => write_compact_nullable!(v, i8),
                ColumnData::I16(v) => write_compact_nullable!(v, i16),
                ColumnData::I32(v) => write_compact_nullable!(v, i32),
                ColumnData::I64(v) => write_compact_nullable!(v, i64),
                ColumnData::U8(v) => write_compact_nullable!(v, u8),
                ColumnData::U16(v) => write_compact_nullable!(v, u16),
                ColumnData::U32(v) => write_compact_nullable!(v, u32),
                ColumnData::U64(v) => write_compact_nullable!(v, u64),
                ColumnData::F32(v) => write_compact_nullable!(v, f32),
                ColumnData::F64(v) => write_compact_nullable!(v, f64),
                ColumnData::Bytes(_) => {}
            }
        }

        LogicalColumn::Utf8(strings) => {
            let mut vec = output.flat_vector(col_idx);
            let mut ci = compact_start;
            for (i, &p) in
                present[row_start..row_start + chunk_size].iter().enumerate()
            {
                if p {
                    vec.insert(i, strings[ci].as_str());
                    ci += 1;
                } else {
                    vec.set_null(i);
                }
            }
        }

        LogicalColumn::Binary(blobs) => {
            let mut vec = output.flat_vector(col_idx);
            let mut ci = compact_start;
            for (i, &p) in
                present[row_start..row_start + chunk_size].iter().enumerate()
            {
                if p {
                    vec.insert(i, blobs[ci].as_slice());
                    ci += 1;
                } else {
                    vec.set_null(i);
                }
            }
        }

        other => {
            return Err(format!(
                "read_he: Nullable<{}> not yet supported",
                other.variant_name()
            )
            .into());
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Trait helper for diagnostic messages
// ---------------------------------------------------------------------------

trait VariantName {
    fn variant_name(&self) -> &'static str;
}

impl VariantName for LogicalColumn {
    fn variant_name(&self) -> &'static str {
        match self {
            LogicalColumn::Primitive(_) => "Primitive",
            LogicalColumn::Utf8(_) => "Utf8",
            LogicalColumn::Binary(_) => "Binary",
            LogicalColumn::ArrayOf { .. } => "ArrayOf",
            LogicalColumn::ArrayOfUtf8 { .. } => "ArrayOfUtf8",
            LogicalColumn::NullablePrim { .. } => "NullablePrim",
            LogicalColumn::NullableUtf8 { .. } => "NullableUtf8",
            LogicalColumn::NullableBinary { .. } => "NullableBinary",
            LogicalColumn::Dictionary { .. } => "Dictionary",
            LogicalColumn::Struct { .. } => "Struct",
            LogicalColumn::List { .. } => "List",
            LogicalColumn::Map { .. } => "Map",
            LogicalColumn::Nullable { .. } => "Nullable",
            LogicalColumn::Union { .. } => "Union",
            LogicalColumn::Decimal128 { .. } => "Decimal128",
            LogicalColumn::Date32 { .. } => "Date32",
            LogicalColumn::Date64 { .. } => "Date64",
            LogicalColumn::Datetime { .. } => "Datetime",
        }
    }
}

// ---------------------------------------------------------------------------
// Extension entrypoint
// ---------------------------------------------------------------------------

/// DuckDB extension entrypoint — called once when `LOAD 'helium_duckdb'` runs.
#[duckdb_entrypoint_c_api(ext_name = "helium_duckdb", min_duckdb_version = "v1.2.0")]
pub fn extension_entrypoint(con: Connection) -> DuckResult<(), Box<dyn std::error::Error>> {
    con.register_table_function::<HeVTab>("read_he")
        .map_err(|e| format!("helium_duckdb: failed to register read_he: {}", e))?;
    Ok(())
}
