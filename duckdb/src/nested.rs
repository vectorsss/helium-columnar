//! Writing nested Helium columns (`Struct` / `List` / `Map`) into DuckDB
//! STRUCT / LIST / MAP vectors, windowed to `[row_start, row_start + chunk_size)`.
//!
//! DuckDB's physical layouts:
//! - **STRUCT**: one child vector per field, all sharing the parent row count.
//!   A window simply forwards `[row_start, chunk_size)` to each child writer.
//! - **LIST**: a `(offset, length)` entry per row plus a single flat child
//!   holding all elements. For a window we emit the contiguous child slice
//!   `[child_lo, child_hi)` (where `child_lo = offsets[row_start]`,
//!   `child_hi = offsets[row_start + chunk_size]`), rebase the per-row entries so
//!   the first row starts at child offset 0, and set the child length.
//! - **MAP**: physically a `LIST(STRUCT(key, value))`. Same offset handling as
//!   LIST, with the child being a struct of two parallel children.

use duckdb::core::{ListVector, StructVector};

use helium::LogicalColumn;

use crate::BoxErr;
use crate::writer::write_flat_column_window;

/// Write a `Struct` window into a DuckDB STRUCT vector.
///
/// Each field is a parallel column with the same row count, so the same
/// `[row_start, chunk_size)` window applies to every child.
pub(crate) fn write_struct_window(
    sv: &StructVector,
    column: &LogicalColumn,
    row_start: usize,
    chunk_size: usize,
) -> Result<(), BoxErr> {
    let LogicalColumn::Struct { fields } = column else {
        return Err("read_he: write_struct_window called on non-Struct column".into());
    };
    for (child_idx, (_name, field_col)) in fields.iter().enumerate() {
        write_child_window(sv, child_idx, field_col, row_start, chunk_size)?;
    }
    Ok(())
}

/// Write a `List` window into a DuckDB LIST vector.
pub(crate) fn write_list_window(
    lv: &mut ListVector,
    column: &LogicalColumn,
    row_start: usize,
    chunk_size: usize,
) -> Result<(), BoxErr> {
    let LogicalColumn::List { offsets, values } = column else {
        return Err("read_he: write_list_window called on non-List column".into());
    };
    write_list_like(lv, offsets, row_start, chunk_size, |lv, lo, count| {
        // List child: one flat-or-nested child vector.
        write_list_child(lv, values, lo, count)
    })
}

/// Write a `Map` window into a DuckDB MAP vector (LIST of STRUCT(key,value)).
pub(crate) fn write_map_window(
    lv: &mut ListVector,
    column: &LogicalColumn,
    row_start: usize,
    chunk_size: usize,
) -> Result<(), BoxErr> {
    let LogicalColumn::Map {
        offsets,
        keys,
        values,
    } = column
    else {
        return Err("read_he: write_map_window called on non-Map column".into());
    };
    write_list_like(lv, offsets, row_start, chunk_size, |lv, lo, count| {
        // Map child is a STRUCT(key, value); reserve and fetch it.
        let struct_child = lv.struct_child(lo + count);
        write_child_window(&struct_child, 0, keys, lo, count)?;
        write_child_window(&struct_child, 1, values, lo, count)?;
        Ok(())
    })
}

/// Shared offset/entry bookkeeping for LIST-shaped vectors (LIST and MAP).
///
/// `offsets` is the Helium row→child-range index (length = parent_rows + 1).
/// We copy the child range `[child_lo, child_hi)` for the window, set the LIST
/// child length to `child_hi`, and write per-row `(offset, length)` entries
/// (absolute into the child, matching the child we materialised).
fn write_list_like<F>(
    lv: &mut ListVector,
    offsets: &[u32],
    row_start: usize,
    chunk_size: usize,
    write_child: F,
) -> Result<(), BoxErr>
where
    F: FnOnce(&ListVector, usize, usize) -> Result<(), BoxErr>,
{
    let end = row_start + chunk_size;
    if end >= offsets.len() {
        return Err("read_he: list offsets shorter than window".into());
    }
    let child_lo = offsets[row_start] as usize;
    let child_hi = offsets[end] as usize;
    let count = child_hi - child_lo;

    // Materialise the child elements covering [child_lo, child_hi). We write the
    // child at its absolute child index so the LIST entries can reference the
    // same absolute offsets; DuckDB only reads [0, set_len).
    write_child(lv, child_lo, count)?;
    lv.set_len(child_hi);

    // Per-row entries: offset/length into the child vector. We keep absolute
    // child offsets (the child was filled at absolute positions up to child_hi).
    for i in 0..chunk_size {
        let lo = offsets[row_start + i] as usize;
        let hi = offsets[row_start + i + 1] as usize;
        lv.set_entry(i, lo, hi - lo);
    }
    Ok(())
}

/// Write a LIST's single child column over `[child_lo, child_lo + count)`.
///
/// The child may itself be flat or nested; reserve `child_lo + count` slots so
/// absolute child offsets line up with the LIST entries.
fn write_list_child(
    lv: &ListVector,
    values: &LogicalColumn,
    child_lo: usize,
    count: usize,
) -> Result<(), BoxErr> {
    match values {
        LogicalColumn::Struct { .. } => {
            let sc = lv.struct_child(child_lo + count);
            write_struct_window(&sc, values, child_lo, count)
        }
        LogicalColumn::List { .. } => {
            let mut lc = lv.list_child();
            write_list_window(&mut lc, values, child_lo, count)
        }
        LogicalColumn::Map { .. } => {
            let mut lc = lv.list_child();
            write_map_window(&mut lc, values, child_lo, count)
        }
        _ => {
            let mut child = lv.child(child_lo + count);
            write_flat_column_window(&mut child, values, child_lo, count)
        }
    }
}

/// Write a struct field's child window — handles both flat and nested fields.
fn write_child_window(
    sv: &StructVector,
    child_idx: usize,
    field_col: &LogicalColumn,
    row_start: usize,
    chunk_size: usize,
) -> Result<(), BoxErr> {
    match field_col {
        LogicalColumn::Struct { .. } => {
            let child = sv.struct_vector_child(child_idx);
            write_struct_window(&child, field_col, row_start, chunk_size)
        }
        LogicalColumn::List { .. } => {
            let mut child = sv.list_vector_child(child_idx);
            write_list_window(&mut child, field_col, row_start, chunk_size)
        }
        LogicalColumn::Map { .. } => {
            let mut child = sv.list_vector_child(child_idx);
            write_map_window(&mut child, field_col, row_start, chunk_size)
        }
        _ => {
            let mut child = sv.child(child_idx, row_start + chunk_size);
            write_flat_column_window(&mut child, field_col, row_start, chunk_size)
        }
    }
}
