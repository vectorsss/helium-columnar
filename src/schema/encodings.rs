//! Default encoding pipelines for Helium column types.
//!
//! This is the **single canonical source** for encoding defaults used by all
//! format adapters (`avro`, `csv`, `json`, `parquet`). Every adapter calls
//! [`crate::schema::encodings::default_encodings`]; do not duplicate the policy in per-adapter code.
//!
//! # Policy summary
//!
//! | Physical column type | Default pipeline |
//! |---|---|
//! | `U8` — booleans, present bitmaps, union tags | `leb128 → zstd` |
//! | `I8`–`I64`, `U16`–`U64`, `F32`, `F64` — numeric **data** | `pcodec` |
//! | `Bytes` — string/binary data payloads | `zstd` |
//! | `U32` offsets (list, map, utf8, binary) | `delta → leb128 → zstd` |
//!
//! Numeric data columns default to `pcodec`, which adapts per chunk internally
//! (no monotonic/time-series assumption); monotonic `offsets` keep `delta`.
//!
//! Composite types prepend one header encoding (offsets/present/tag) then
//! extend with the inner type's encodings.  [`crate::LogicalType::Struct`] always
//! has empty top-level encodings; leaf encodings live in child
//! [`crate::FieldSpec`] children.

use crate::{CoderSpec, DataType, DateUnit, LogicalType};

/// Generate default encoding pipeline vectors for a given [`LogicalType`].
///
/// The returned `Vec<Vec<CoderSpec>>` has one entry per physical encoding slot
/// required by the type (matching [`LogicalType::expected_encodings_len`]).
///
/// All format adapters call this function — it is the single place to update
/// the default encoding policy.
pub fn default_encodings(lt: &LogicalType) -> Vec<Vec<CoderSpec>> {
    match lt {
        LogicalType::Primitive { data_type } => vec![prim_coders(*data_type)],
        LogicalType::Utf8 => vec![offset_coders(), data_coders()],
        LogicalType::Binary => vec![offset_coders(), data_coders()],
        LogicalType::List { inner } => {
            let mut enc = vec![offset_coders()]; // offsets: U32
            enc.extend(default_encodings(inner));
            enc
        }
        LogicalType::Map { key, value } => {
            let mut enc = vec![offset_coders()]; // offsets: U32
            enc.extend(default_encodings(key));
            enc.extend(default_encodings(value));
            enc
        }
        LogicalType::Nullable { inner } => {
            let mut enc = vec![u8_coders()]; // present: U8
            enc.extend(default_encodings(inner));
            enc
        }
        LogicalType::Union { variants } => {
            let mut enc = vec![u8_coders()]; // tag: U8
            for (_, v_lt) in variants {
                enc.extend(default_encodings(v_lt));
            }
            enc
        }
        // Struct: empty top-level; leaf encodings live in FieldSpec children.
        LogicalType::Struct { .. } => vec![],

        // Semantic type extensions.
        // Decimal128: two I64 leaves (high + low), both use the integer pipeline.
        LogicalType::Decimal128 { .. } => vec![
            prim_coders(DataType::I64), // high
            prim_coders(DataType::I64), // low
        ],
        // Date (Days): single I32 leaf.
        LogicalType::Date {
            unit: DateUnit::Days,
        } => vec![prim_coders(DataType::I32)],
        // Date (Millis): single I64 leaf.
        LogicalType::Date {
            unit: DateUnit::Millis,
        } => vec![prim_coders(DataType::I64)],
        // Datetime: single I64 leaf.
        LogicalType::Datetime { .. } => vec![prim_coders(DataType::I64)],
        // Dictionary{inner}: inner type encodings + one U32 indices pipeline.
        LogicalType::Dictionary { inner } => {
            let mut enc = default_encodings(inner);
            enc.push(offset_coders()); // indices: U32
            enc
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers (exported pub(crate) for use in adapter modules)
// ---------------------------------------------------------------------------

/// Pipeline for a primitive integer or float **data** physical type.
///
/// Numeric data columns default to `pcodec`. pcodec adapts internally per chunk
/// (choosing mode/bit-width itself, with no monotonic or time-series
/// assumption), which is a far better blanket default than blindly `delta`-coding
/// every integer (harmful on non-sorted data) or `gorilla`-coding every float.
/// The optimizer can still pick a different per-column pipeline when measured.
///
/// Note: this is for *data* leaves only — monotonic `offsets` keep `delta`
/// (see [`offset_coders`]).
pub(crate) fn prim_coders(dt: DataType) -> Vec<CoderSpec> {
    match dt {
        // present bitmaps / union tags / booleans surface as U8; they don't
        // benefit from pcodec's numeric modes — keep the light boolean pipeline.
        DataType::U8 => u8_coders(),
        // I8/I16/I32/I64/U16/U32/U64 + F32/F64 — pcodec handles them all.
        _ => vec![CoderSpec::new("pcodec")],
    }
}

/// Pipeline for `U8` columns (present bitmaps, union tags, booleans).
///
/// Skips `delta` — 0/1 boolean values and small tag indices don't benefit from
/// differencing (delta of alternating 0/1 produces large unsigned wraps).
pub(crate) fn u8_coders() -> Vec<CoderSpec> {
    vec![CoderSpec::new("leb128"), CoderSpec::new("zstd")]
}

/// Pipeline for `U32` monotonically non-decreasing offset columns.
pub(crate) fn offset_coders() -> Vec<CoderSpec> {
    vec![
        CoderSpec::new("delta"),
        CoderSpec::new("leb128"),
        CoderSpec::new("zstd"),
    ]
}

/// Pipeline for raw `Bytes` payloads (string data, binary blobs).
pub(crate) fn data_coders() -> Vec<CoderSpec> {
    vec![CoderSpec::new("zstd")]
}
