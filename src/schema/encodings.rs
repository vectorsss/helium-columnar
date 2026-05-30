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
//! | `I32`, `I64`, `U32`, `U64`, `I8`, `I16`, `U16` | `delta → leb128 → zstd` |
//! | `F32`, `F64` | `gorilla → zstd` |
//! | `Bytes` — string/binary data payloads | `zstd` |
//! | `U32` offsets (list, map, utf8, binary) | `delta → leb128 → zstd` |
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

        // v2 compatibility types — not produced by the format adapters but
        // handled gracefully so this function is safe to call on any LogicalType.
        LogicalType::ArrayOf { data_type } => {
            vec![offset_coders(), prim_coders(*data_type)]
        }
        LogicalType::ArrayOfUtf8 => {
            vec![offset_coders(), offset_coders(), data_coders()]
        }
        LogicalType::NullablePrim { data_type } => {
            vec![u8_coders(), prim_coders(*data_type)]
        }
        LogicalType::NullableUtf8 => vec![u8_coders(), offset_coders(), data_coders()],
        LogicalType::NullableBinary => vec![u8_coders(), offset_coders(), data_coders()],
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

/// Pipeline for a primitive integer or float physical type.
pub(crate) fn prim_coders(dt: DataType) -> Vec<CoderSpec> {
    match dt {
        DataType::F32 | DataType::F64 => {
            // gorilla XOR-encodes consecutive floats → Bytes, then zstd compresses.
            vec![CoderSpec::new("gorilla"), CoderSpec::new("zstd")]
        }
        DataType::U8 => u8_coders(),
        _ => {
            // I8, I16, I32, I64, U16, U32, U64:
            // delta → leb128 (integer → Bytes) → zstd.
            vec![
                CoderSpec::new("delta"),
                CoderSpec::new("leb128"),
                CoderSpec::new("zstd"),
            ]
        }
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
