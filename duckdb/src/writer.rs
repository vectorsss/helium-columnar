//! Writing decoded Helium [`LogicalColumn`] windows into DuckDB output vectors.
//!
//! The scan emits up to `DUCKDB_VECTOR_SIZE` rows at a time, so every writer
//! takes a `[row_start, row_start + chunk_size)` window into an already-decoded
//! stripe column and copies just that window into the DuckDB chunk.

use duckdb::core::{DataChunkHandle, FlatVector, Inserter};

use helium::{ColumnData, LogicalColumn};

use crate::BoxErr;
use crate::VariantName;
use crate::nested::{write_list_window, write_map_window, write_struct_window};

/// Write rows `[row_start, row_start + chunk_size)` from `column` into the
/// DuckDB output chunk at the given (output) column index.
///
/// This is the top-level entry: it dispatches on the column variant, recursing
/// into nested-type writers for `Struct` / `List` / `Map`.
pub(crate) fn write_column_window(
    output: &mut DataChunkHandle,
    col_idx: usize,
    column: &LogicalColumn,
    row_start: usize,
    chunk_size: usize,
) -> Result<(), BoxErr> {
    match column {
        LogicalColumn::Struct { .. } => {
            let sv = output.struct_vector(col_idx);
            write_struct_window(&sv, column, row_start, chunk_size)?;
        }
        LogicalColumn::List { .. } => {
            let mut lv = output.list_vector(col_idx);
            write_list_window(&mut lv, column, row_start, chunk_size)?;
        }
        LogicalColumn::Map { .. } => {
            let mut lv = output.list_vector(col_idx);
            write_map_window(&mut lv, column, row_start, chunk_size)?;
        }
        _ => {
            let mut vec = output.flat_vector(col_idx);
            write_flat_column_window(&mut vec, column, row_start, chunk_size)?;
        }
    }
    Ok(())
}

/// Write a non-nested `LogicalColumn` window into a single `FlatVector`.
///
/// Used both for top-level flat columns and as the leaf writer for the
/// children of nested types.
pub(crate) fn write_flat_column_window(
    vec: &mut FlatVector,
    column: &LogicalColumn,
    row_start: usize,
    chunk_size: usize,
) -> Result<(), BoxErr> {
    match column {
        LogicalColumn::Primitive(data) => {
            write_flat_data(vec, data, row_start, chunk_size);
        }

        LogicalColumn::Utf8(strings) => {
            for (i, s) in strings[row_start..row_start + chunk_size].iter().enumerate() {
                vec.insert(i, s.as_str());
            }
        }

        LogicalColumn::Binary(blobs) => {
            for (i, blob) in blobs[row_start..row_start + chunk_size].iter().enumerate() {
                vec.insert(i, blob.as_slice());
            }
        }

        // legacy flat nullable primitives (present bitmap covers all rows, values dense)
        LogicalColumn::NullablePrim { present, values } => {
            write_flat_data_nullable(vec, values, present, row_start, chunk_size);
        }

        LogicalColumn::NullableUtf8 { present, strings } => {
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
                expand_dict_prim(vec, cd, indices, row_start, chunk_size);
            }
            LogicalColumn::Utf8(dict) => {
                for (i, &idx) in indices[row_start..row_start + chunk_size].iter().enumerate() {
                    let s = dict
                        .get(idx as usize)
                        .ok_or("read_he: dict index out of range")?;
                    vec.insert(i, s.as_str());
                }
            }
            other => {
                return Err(format!(
                    "read_he: Dictionary with inner type {} is not yet supported \
                     (only Primitive / Utf8)",
                    other.variant_name()
                )
                .into());
            }
        },

        // recursive nullable — compacted inner values
        LogicalColumn::Nullable { present, value } => {
            write_nullable_compacted(vec, present, value, row_start, chunk_size)?;
        }

        // Semantic types
        LogicalColumn::Decimal128 { values } => {
            // DuckDB DECIMAL is backed by hugeint { lower: u64, upper: i64 }.
            #[repr(C)]
            struct HugeInt {
                lower: u64,
                upper: i64,
            }
            let ptr = vec.as_mut_ptr::<HugeInt>();
            for (i, &v) in values[row_start..row_start + chunk_size].iter().enumerate() {
                // SAFETY: HugeInt is repr(C) with no padding; we write exactly
                // chunk_size slots that DuckDB allocated for this column.
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
            let ptr = vec.as_mut_ptr::<DuckDate>();
            for (i, &v) in values[row_start..row_start + chunk_size].iter().enumerate() {
                // SAFETY: same as Decimal128 above — within DuckDB-allocated slots.
                unsafe {
                    ptr.add(i).write(DuckDate { days: v });
                }
            }
        }

        LogicalColumn::Date64 { values } => {
            // Mapped to BIGINT (milliseconds since epoch).
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
            #[repr(C)]
            struct DuckTimestamp {
                micros: i64,
            }
            let ptr = vec.as_mut_ptr::<DuckTimestamp>();
            for (i, &v) in values[row_start..row_start + chunk_size].iter().enumerate() {
                // SAFETY: DuckTimestamp is repr(C) single i64.
                unsafe {
                    ptr.add(i).write(DuckTimestamp { micros: v });
                }
            }
        }

        LogicalColumn::ArrayOf { .. } | LogicalColumn::ArrayOfUtf8 { .. } => {
            return Err("read_he: ArrayOf (legacy flat) not supported".into());
        }

        LogicalColumn::Struct { .. } | LogicalColumn::List { .. } | LogicalColumn::Map { .. } => {
            return Err(
                "read_he: nested type passed to flat writer (bug — should route through \
                 the nested writers)"
                    .into(),
            );
        }

        LogicalColumn::Union { .. } => {
            return Err("read_he: Union columns not yet supported".into());
        }
    }
    Ok(())
}

/// Write a flat (non-nullable) `ColumnData` slice into an output vector.
fn write_flat_data(vec: &mut FlatVector, data: &ColumnData, row_start: usize, chunk_size: usize) {
    macro_rules! copy_slice {
        ($vals:expr, $ty:ty) => {{
            let ptr = vec.as_mut_ptr::<$ty>();
            // SAFETY: DuckDB allocated `capacity()` slots of the column's
            // native type; we write exactly chunk_size ≤ capacity() of them
            // from a source slice known to hold at least chunk_size elements.
            unsafe {
                std::ptr::copy_nonoverlapping($vals[row_start..].as_ptr(), ptr, chunk_size);
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
            for (i, b) in v[row_start..row_start + chunk_size].iter().enumerate() {
                vec.insert(i, std::slice::from_ref(b));
            }
        }
    }
}

/// Write a nullable `ColumnData` (legacy flat `NullablePrim` semantics — dense storage,
/// null bitmap separate) into the output vector.
fn write_flat_data_nullable(
    vec: &mut FlatVector,
    data: &ColumnData,
    present: &[bool],
    row_start: usize,
    chunk_size: usize,
) {
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
/// the output vector.
fn expand_dict_prim(
    vec: &mut FlatVector,
    dictionary: &ColumnData,
    indices: &[u32],
    row_start: usize,
    chunk_size: usize,
) {
    let idx_slice = &indices[row_start..row_start + chunk_size];
    macro_rules! expand_typed {
        ($dict:expr, $ty:ty) => {{
            let ptr = vec.as_mut_ptr::<$ty>();
            for (i, &idx) in idx_slice.iter().enumerate() {
                // Defensive: clamp out-of-range indices to the last entry rather
                // than panicking on corrupt data; the per-leaf CRC already
                // guards integrity, so this is belt-and-braces.
                let val = $dict.get(idx as usize).copied().unwrap_or_default();
                // SAFETY: DuckDB slot within capacity.
                unsafe { ptr.add(i).write(val) };
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

/// Write a recursive `Nullable { present, value }` (compacted inner) column window.
///
/// The inner `value` holds only the non-null rows in compacted form. We count
/// how many non-null rows precede `row_start` to find the offset into the
/// compacted buffer — this is the absolute-row indexing path that needs to stay
/// correct across chunk and stripe boundaries.
fn write_nullable_compacted(
    vec: &mut FlatVector,
    present: &[bool],
    value: &LogicalColumn,
    row_start: usize,
    chunk_size: usize,
) -> Result<(), BoxErr> {
    let compact_start: usize = present[..row_start].iter().filter(|&&p| p).count();

    match value {
        LogicalColumn::Primitive(data) => {
            let mut ci = compact_start;
            macro_rules! write_compact_nullable {
                ($vals:expr, $ty:ty) => {{
                    let ptr = vec.as_mut_ptr::<$ty>();
                    for (i, &p) in present[row_start..row_start + chunk_size].iter().enumerate() {
                        if p {
                            let v = $vals
                                .get(ci)
                                .copied()
                                .ok_or("read_he: nullable compact index out of range")?;
                            // SAFETY: ptr points into DuckDB-allocated storage; i < chunk_size.
                            unsafe { ptr.add(i).write(v) };
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
            let mut ci = compact_start;
            for (i, &p) in present[row_start..row_start + chunk_size].iter().enumerate() {
                if p {
                    let s = strings
                        .get(ci)
                        .ok_or("read_he: nullable compact index out of range")?;
                    vec.insert(i, s.as_str());
                    ci += 1;
                } else {
                    vec.set_null(i);
                }
            }
        }

        LogicalColumn::Binary(blobs) => {
            let mut ci = compact_start;
            for (i, &p) in present[row_start..row_start + chunk_size].iter().enumerate() {
                if p {
                    let b = blobs
                        .get(ci)
                        .ok_or("read_he: nullable compact index out of range")?;
                    vec.insert(i, b.as_slice());
                    ci += 1;
                } else {
                    vec.set_null(i);
                }
            }
        }

        other => {
            return Err(format!(
                "read_he: Nullable<{}> not yet supported in flat writer",
                other.variant_name()
            )
            .into());
        }
    }
    Ok(())
}
