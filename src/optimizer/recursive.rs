//! Recursive v3 nested-type optimizer.
//!
//! The core function [`optimize_type`] walks any `LogicalType`/`LogicalColumn`
//! pair and returns:
//! - An updated `LogicalType` with `FieldSpec::encodings` filled in for nested
//!   `Struct` nodes (which carry their field encodings inside the type itself).
//! - A flat `Vec<Vec<CoderSpec>>` of encoding vectors matching the type's
//!   [`LogicalType::expected_encodings_len()`].
//!
//! [`optimize_column`] wraps this into a full [`ColumnSpec`].

use crate::{
    CoderRegistry, CoderSpec, ColumnData, ColumnSpec, DateUnit, FieldSpec, HeliumError,
    LogicalColumn, LogicalType,
};

type Result<T> = std::result::Result<T, HeliumError>;

use super::picker::pick_best_leaf;

/// Recursively pick optimal encodings for `lt` using `lc` as sample data.
///
/// Returns:
/// - The (possibly updated) `LogicalType`.  For `Struct`, the returned type
///   has all `FieldSpec::encodings` populated.  For other container types, the
///   returned type has an updated inner `LogicalType` if it contained a `Struct`.
/// - The flat list of encoding vectors (`Vec<Vec<CoderSpec>>`) matching
///   `lt.expected_encodings_len()`.
///
/// Both outputs together are sufficient to construct a valid `ColumnSpec` or
/// `FieldSpec` for `lt`.
pub fn optimize_type(
    lt: LogicalType,
    lc: LogicalColumn,
    terminal: &str,
    registry: &CoderRegistry,
    context: &str,
) -> Result<(LogicalType, Vec<Vec<CoderSpec>>)> {
    match (lt, lc) {
        // ----------------------------------------------------------------
        // Primitive leaf
        // ----------------------------------------------------------------
        (LogicalType::Primitive { data_type }, LogicalColumn::Primitive(cd)) => {
            let coders = pick_best_leaf("values", cd, terminal, registry, context)?;
            Ok((LogicalType::Primitive { data_type }, vec![coders]))
        }

        // ----------------------------------------------------------------
        // Utf8
        // ----------------------------------------------------------------
        (LogicalType::Utf8, LogicalColumn::Utf8(strings)) => {
            let (offsets, bytes) = flatten_strings(&strings).map_err(|e| HeliumError::Schema {
                column: context.into(),
                reason: e,
            })?;
            let off_enc = pick_best_leaf(
                "offsets",
                ColumnData::U32(offsets),
                terminal,
                registry,
                context,
            )?;
            let data_enc = pick_best_leaf(
                "data",
                ColumnData::Bytes(bytes),
                terminal,
                registry,
                context,
            )?;
            Ok((LogicalType::Utf8, vec![off_enc, data_enc]))
        }

        // ----------------------------------------------------------------
        // Binary
        // ----------------------------------------------------------------
        (LogicalType::Binary, LogicalColumn::Binary(blobs)) => {
            let (offsets, bytes) = flatten_binary(&blobs).map_err(|e| HeliumError::Schema {
                column: context.into(),
                reason: e,
            })?;
            let off_enc = pick_best_leaf(
                "offsets",
                ColumnData::U32(offsets),
                terminal,
                registry,
                context,
            )?;
            let data_enc = pick_best_leaf(
                "data",
                ColumnData::Bytes(bytes),
                terminal,
                registry,
                context,
            )?;
            Ok((LogicalType::Binary, vec![off_enc, data_enc]))
        }

        // ----------------------------------------------------------------
        // v2 NullablePrim
        // ----------------------------------------------------------------
        (
            LogicalType::NullablePrim { data_type },
            LogicalColumn::NullablePrim { present, values },
        ) => {
            let present_bytes = present_to_bytes(&present);
            let pres_enc = pick_best_leaf(
                "present",
                ColumnData::U8(present_bytes),
                terminal,
                registry,
                context,
            )?;
            let val_enc = pick_best_leaf("values", values, terminal, registry, context)?;
            Ok((
                LogicalType::NullablePrim { data_type },
                vec![pres_enc, val_enc],
            ))
        }

        // ----------------------------------------------------------------
        // v2 NullableUtf8
        // ----------------------------------------------------------------
        (LogicalType::NullableUtf8, LogicalColumn::NullableUtf8 { present, strings }) => {
            let present_bytes = present_to_bytes(&present);
            let (offsets, bytes) = flatten_strings(&strings).map_err(|e| HeliumError::Schema {
                column: context.into(),
                reason: e,
            })?;
            let pres_enc = pick_best_leaf(
                "present",
                ColumnData::U8(present_bytes),
                terminal,
                registry,
                context,
            )?;
            let off_enc = pick_best_leaf(
                "offsets",
                ColumnData::U32(offsets),
                terminal,
                registry,
                context,
            )?;
            let data_enc = pick_best_leaf(
                "data",
                ColumnData::Bytes(bytes),
                terminal,
                registry,
                context,
            )?;
            Ok((LogicalType::NullableUtf8, vec![pres_enc, off_enc, data_enc]))
        }

        // ----------------------------------------------------------------
        // v2 NullableBinary
        // ----------------------------------------------------------------
        (LogicalType::NullableBinary, LogicalColumn::NullableBinary { present, blobs }) => {
            let present_bytes = present_to_bytes(&present);
            let (offsets, bytes) = flatten_binary(&blobs).map_err(|e| HeliumError::Schema {
                column: context.into(),
                reason: e,
            })?;
            let pres_enc = pick_best_leaf(
                "present",
                ColumnData::U8(present_bytes),
                terminal,
                registry,
                context,
            )?;
            let off_enc = pick_best_leaf(
                "offsets",
                ColumnData::U32(offsets),
                terminal,
                registry,
                context,
            )?;
            let data_enc = pick_best_leaf(
                "data",
                ColumnData::Bytes(bytes),
                terminal,
                registry,
                context,
            )?;
            Ok((
                LogicalType::NullableBinary,
                vec![pres_enc, off_enc, data_enc],
            ))
        }

        // ----------------------------------------------------------------
        // v2 ArrayOf
        // ----------------------------------------------------------------
        (LogicalType::ArrayOf { data_type }, LogicalColumn::ArrayOf { offsets, values }) => {
            let off_enc = pick_best_leaf(
                "offsets",
                ColumnData::U32(offsets),
                terminal,
                registry,
                context,
            )?;
            let val_enc = pick_best_leaf("values", values, terminal, registry, context)?;
            Ok((LogicalType::ArrayOf { data_type }, vec![off_enc, val_enc]))
        }

        // ----------------------------------------------------------------
        // v2 ArrayOfUtf8
        // ----------------------------------------------------------------
        (LogicalType::ArrayOfUtf8, LogicalColumn::ArrayOfUtf8 { offsets, strings }) => {
            let (inner_off, data_bytes) =
                flatten_strings(&strings).map_err(|e| HeliumError::Schema {
                    column: context.into(),
                    reason: e,
                })?;
            let outer_off_enc = pick_best_leaf(
                "outer_offsets",
                ColumnData::U32(offsets),
                terminal,
                registry,
                context,
            )?;
            let inner_off_enc = pick_best_leaf(
                "inner_offsets",
                ColumnData::U32(inner_off),
                terminal,
                registry,
                context,
            )?;
            let data_enc = pick_best_leaf(
                "data",
                ColumnData::Bytes(data_bytes),
                terminal,
                registry,
                context,
            )?;
            Ok((
                LogicalType::ArrayOfUtf8,
                vec![outer_off_enc, inner_off_enc, data_enc],
            ))
        }

        // ----------------------------------------------------------------
        // v3 Struct — encodings live in FieldSpec, not in ColumnSpec::encodings
        // ----------------------------------------------------------------
        (
            LogicalType::Struct {
                fields: spec_fields,
            },
            LogicalColumn::Struct { fields: col_fields },
        ) => {
            if spec_fields.len() != col_fields.len() {
                return Err(HeliumError::Schema {
                    column: context.into(),
                    reason: format!(
                        "Struct: schema has {} fields, data has {} fields",
                        spec_fields.len(),
                        col_fields.len()
                    ),
                });
            }
            let mut new_fields = Vec::with_capacity(spec_fields.len());
            for (spec_f, (col_name, col_lc)) in spec_fields.into_iter().zip(col_fields) {
                if spec_f.name != col_name {
                    return Err(HeliumError::Schema {
                        column: context.into(),
                        reason: format!(
                            "Struct field name mismatch: schema='{}', data='{}'",
                            spec_f.name, col_name
                        ),
                    });
                }
                let field_ctx = format!("{context}.{}", spec_f.name);
                let (updated_lt, encodings) =
                    optimize_type(spec_f.logical_type, col_lc, terminal, registry, &field_ctx)?;
                new_fields.push(FieldSpec::new(spec_f.name, updated_lt, encodings));
            }
            Ok((LogicalType::Struct { fields: new_fields }, vec![]))
        }

        // ----------------------------------------------------------------
        // v3 List
        // ----------------------------------------------------------------
        (LogicalType::List { inner }, LogicalColumn::List { offsets, values }) => {
            let off_enc = pick_best_leaf(
                "offsets",
                ColumnData::U32(offsets),
                terminal,
                registry,
                context,
            )?;
            let inner_ctx = format!("{context}[item]");
            let (updated_inner, inner_encs) =
                optimize_type(*inner, *values, terminal, registry, &inner_ctx)?;
            let mut encodings = vec![off_enc];
            encodings.extend(inner_encs);
            Ok((
                LogicalType::List {
                    inner: Box::new(updated_inner),
                },
                encodings,
            ))
        }

        // ----------------------------------------------------------------
        // v3 Map
        // ----------------------------------------------------------------
        (
            LogicalType::Map { key, value },
            LogicalColumn::Map {
                offsets,
                keys,
                values,
            },
        ) => {
            let off_enc = pick_best_leaf(
                "offsets",
                ColumnData::U32(offsets),
                terminal,
                registry,
                context,
            )?;
            let key_ctx = format!("{context}[key]");
            let val_ctx = format!("{context}[value]");
            let (updated_key, key_encs) = optimize_type(*key, *keys, terminal, registry, &key_ctx)?;
            let (updated_value, val_encs) =
                optimize_type(*value, *values, terminal, registry, &val_ctx)?;
            let mut encodings = vec![off_enc];
            encodings.extend(key_encs);
            encodings.extend(val_encs);
            Ok((
                LogicalType::Map {
                    key: Box::new(updated_key),
                    value: Box::new(updated_value),
                },
                encodings,
            ))
        }

        // ----------------------------------------------------------------
        // v3 Nullable
        // ----------------------------------------------------------------
        (LogicalType::Nullable { inner }, LogicalColumn::Nullable { present, value }) => {
            let present_bytes = present_to_bytes(&present);
            let pres_enc = pick_best_leaf(
                "present",
                ColumnData::U8(present_bytes),
                terminal,
                registry,
                context,
            )?;
            let inner_ctx = format!("{context}[nullable]");
            let (updated_inner, inner_encs) =
                optimize_type(*inner, *value, terminal, registry, &inner_ctx)?;
            let mut encodings = vec![pres_enc];
            encodings.extend(inner_encs);
            Ok((
                LogicalType::Nullable {
                    inner: Box::new(updated_inner),
                },
                encodings,
            ))
        }

        // ----------------------------------------------------------------
        // v3 Union
        // ----------------------------------------------------------------
        (
            LogicalType::Union {
                variants: spec_variants,
            },
            LogicalColumn::Union {
                tags,
                variants: col_variants,
            },
        ) => {
            if spec_variants.len() != col_variants.len() {
                return Err(HeliumError::Schema {
                    column: context.into(),
                    reason: format!(
                        "Union: schema has {} variants, data has {} variants",
                        spec_variants.len(),
                        col_variants.len()
                    ),
                });
            }
            let tag_enc = pick_best_leaf("tag", ColumnData::U8(tags), terminal, registry, context)?;
            let mut encodings = vec![tag_enc];
            let mut new_variants = Vec::with_capacity(spec_variants.len());
            for ((v_name, v_lt), (col_v_name, col_v_lc)) in
                spec_variants.into_iter().zip(col_variants)
            {
                if v_name != col_v_name {
                    return Err(HeliumError::Schema {
                        column: context.into(),
                        reason: format!(
                            "Union variant name mismatch: schema='{v_name}', data='{col_v_name}'"
                        ),
                    });
                }
                let v_ctx = format!("{context}[union:{v_name}]");
                let (updated_v_lt, v_encs) =
                    optimize_type(v_lt, col_v_lc, terminal, registry, &v_ctx)?;
                encodings.extend(v_encs);
                new_variants.push((v_name, updated_v_lt));
            }
            Ok((
                LogicalType::Union {
                    variants: new_variants,
                },
                encodings,
            ))
        }

        // ----------------------------------------------------------------
        // Decimal128 → two I64 leaves (high = (v >> 64) as i64, low = v as i64)
        // This split exactly mirrors the writer's decompose path in schema.rs.
        // ----------------------------------------------------------------
        (LogicalType::Decimal128 { precision, scale }, LogicalColumn::Decimal128 { values }) => {
            let highs: Vec<i64> = values.iter().map(|&v| (v >> 64) as i64).collect();
            let lows: Vec<i64> = values.iter().map(|&v| v as i64).collect();
            let high_enc =
                pick_best_leaf("high", ColumnData::I64(highs), terminal, registry, context)?;
            let low_enc =
                pick_best_leaf("low", ColumnData::I64(lows), terminal, registry, context)?;
            Ok((
                LogicalType::Decimal128 { precision, scale },
                vec![high_enc, low_enc],
            ))
        }

        // ----------------------------------------------------------------
        // Date32 → single I32 leaf (days since epoch).
        // ----------------------------------------------------------------
        (
            LogicalType::Date {
                unit: DateUnit::Days,
            },
            LogicalColumn::Date32 { values },
        ) => {
            let enc = pick_best_leaf(
                "values",
                ColumnData::I32(values),
                terminal,
                registry,
                context,
            )?;
            Ok((
                LogicalType::Date {
                    unit: DateUnit::Days,
                },
                vec![enc],
            ))
        }

        // ----------------------------------------------------------------
        // Date64 → single I64 leaf (milliseconds since epoch).
        // ----------------------------------------------------------------
        (
            LogicalType::Date {
                unit: DateUnit::Millis,
            },
            LogicalColumn::Date64 { values },
        ) => {
            let enc = pick_best_leaf(
                "values",
                ColumnData::I64(values),
                terminal,
                registry,
                context,
            )?;
            Ok((
                LogicalType::Date {
                    unit: DateUnit::Millis,
                },
                vec![enc],
            ))
        }

        // ----------------------------------------------------------------
        // Datetime → single I64 leaf (unit and timezone are schema metadata).
        // ----------------------------------------------------------------
        (LogicalType::Datetime { unit, timezone }, LogicalColumn::Datetime { values }) => {
            let enc = pick_best_leaf(
                "values",
                ColumnData::I64(values),
                terminal,
                registry,
                context,
            )?;
            Ok((LogicalType::Datetime { unit, timezone }, vec![enc]))
        }

        // ----------------------------------------------------------------
        // Dictionary{inner} — v3 recursive dictionary encoding
        // ----------------------------------------------------------------
        (
            LogicalType::Dictionary { inner },
            LogicalColumn::Dictionary {
                dictionary,
                indices,
            },
        ) => {
            // Optimize inner type for the dictionary column.
            let inner_ctx = format!("{context}[dict]");
            let (updated_inner, inner_encs) =
                optimize_type(*inner, *dictionary, terminal, registry, &inner_ctx)?;
            // Optimize the indices leaf.
            let idx_enc = pick_best_leaf(
                "indices",
                ColumnData::U32(indices),
                terminal,
                registry,
                context,
            )?;
            let mut encodings = inner_encs;
            encodings.push(idx_enc);
            Ok((
                LogicalType::Dictionary {
                    inner: Box::new(updated_inner),
                },
                encodings,
            ))
        }

        // ----------------------------------------------------------------
        // Type/data mismatch
        // ----------------------------------------------------------------
        (lt, _lc) => Err(HeliumError::Schema {
            column: context.into(),
            reason: format!(
                "LogicalType/LogicalColumn mismatch: type={lt:?}, data variant is incompatible"
            ),
        }),
    }
}

/// Entry point: optimize a single top-level column.
///
/// `lt` is the structural skeleton; `lc` is the sample data.  Returns a
/// `ColumnSpec` with all encoding vectors chosen for minimal compressed size.
pub fn optimize_column(
    name: &str,
    lt: LogicalType,
    lc: LogicalColumn,
    terminal: &str,
    registry: &CoderRegistry,
) -> Result<ColumnSpec> {
    let (updated_lt, encodings) = optimize_type(lt, lc, terminal, registry, name)?;
    Ok(ColumnSpec::new(name, updated_lt, encodings))
}

// ---------------------------------------------------------------------------
// Internal helpers (mirrors schema.rs private helpers)
// ---------------------------------------------------------------------------

/// Convert a `&[bool]` present bitmap to a `Vec<u8>` (1 = present, 0 = null).
pub(crate) fn present_to_bytes(present: &[bool]) -> Vec<u8> {
    present.iter().map(|&p| if p { 1u8 } else { 0u8 }).collect()
}

/// Flatten strings into (cumulative offsets, concatenated UTF-8 bytes).
pub(crate) fn flatten_strings(
    strings: &[String],
) -> std::result::Result<(Vec<u32>, Vec<u8>), String> {
    let mut offsets = Vec::with_capacity(strings.len() + 1);
    let mut data = Vec::new();
    offsets.push(0u32);
    for s in strings {
        data.extend_from_slice(s.as_bytes());
        let len = data.len();
        let off = u32::try_from(len).map_err(|_| "string data exceeds 4 GiB".to_string())?;
        offsets.push(off);
    }
    Ok((offsets, data))
}

/// Flatten blobs into (cumulative offsets, concatenated bytes).
pub(crate) fn flatten_binary(
    blobs: &[Vec<u8>],
) -> std::result::Result<(Vec<u32>, Vec<u8>), String> {
    let mut offsets = Vec::with_capacity(blobs.len() + 1);
    let mut data = Vec::new();
    offsets.push(0u32);
    for b in blobs {
        data.extend_from_slice(b);
        let len = data.len();
        let off = u32::try_from(len).map_err(|_| "binary data exceeds 4 GiB".to_string())?;
        offsets.push(off);
    }
    Ok((offsets, data))
}
