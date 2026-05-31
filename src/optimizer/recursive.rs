//! Recursive nested-type optimizer.
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
    CoderRegistry, CoderSpec, ColumnData, ColumnSpec, DataType, DateUnit, FieldSpec, HeliumError,
    LogicalColumn, LogicalType,
};

type Result<T> = std::result::Result<T, HeliumError>;

use super::candidates::{distinct_i32, distinct_i64, distinct_str};
use super::measure_encoding;
use super::picker::pick_best_leaf;

/// Upper bound on the number of distinct values for which a column is still a
/// dictionary candidate. The actual limit is `min(row_count / 2, MAX)`: a
/// column whose values are at least half-unique cannot benefit from dictionary
/// encoding (the index column alone would cost ~as much as the data), and the
/// 65 536 ceiling keeps the cardinality probe cheap on high-cardinality data —
/// the `distinct_*` helpers early-exit the moment the limit is exceeded.
const DICT_CARDINALITY_MAX: usize = 65_536;

/// Columns shorter than this are skipped entirely: per-column framing overhead
/// dominates and dictionary encoding cannot pay for the extra index leaf.
const DICT_MIN_ROWS: usize = 64;

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
    terminal: &CoderSpec,
    registry: &CoderRegistry,
    context: &str,
) -> Result<(LogicalType, Vec<Vec<CoderSpec>>)> {
    // Top-level entry: dictionary promotion is allowed.
    optimize_type_inner(lt, lc, terminal, registry, context, true)
}

/// Internal worker for [`optimize_type`].
///
/// `dict_allowed` is `false` in positions where a `Dictionary<T>` leaf would be
/// structurally illegal — notably **Map keys**, which the writer requires to be
/// `Primitive`/`Utf8`/`Binary`. The flag is reset to `true` whenever recursion
/// descends into a position that *can* hold a dictionary (struct fields, list /
/// map values, nullable / union inner).
fn optimize_type_inner(
    lt: LogicalType,
    lc: LogicalColumn,
    terminal: &CoderSpec,
    registry: &CoderRegistry,
    context: &str,
    dict_allowed: bool,
) -> Result<(LogicalType, Vec<Vec<CoderSpec>>)> {
    match (lt, lc) {
        // ----------------------------------------------------------------
        // Primitive leaf
        // ----------------------------------------------------------------
        (LogicalType::Primitive { data_type }, LogicalColumn::Primitive(cd)) => {
            // Plain variant (always valid).
            let plain_lt = LogicalType::Primitive { data_type };
            let plain_coders = pick_best_leaf("values", cd.clone(), terminal, registry, context)?;
            let plain = (plain_lt, vec![plain_coders]);

            // Try dictionary promotion for integer leaves only (floats are poor
            // dict candidates and pcodec already covers low-cardinality numerics).
            if dict_allowed
                && is_dict_candidate_int(&cd)
                && let Some(dict) = try_dict_primitive(&cd, data_type, terminal, registry, context)?
            {
                return pick_smaller_variant(
                    plain,
                    dict,
                    LogicalColumn::Primitive(cd),
                    registry,
                    context,
                );
            }
            Ok(plain)
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
            let plain = (LogicalType::Utf8, vec![off_enc, data_enc]);

            // Try dictionary promotion when cardinality is low.
            let limit = dict_limit(strings.len());
            if dict_allowed && limit > 0 && distinct_str(&strings, limit).is_some() {
                let dict_col = LogicalColumn::dict_encode_utf8(strings.clone());
                let dict_lt = LogicalType::Dictionary {
                    inner: Box::new(LogicalType::Utf8),
                };
                let dict = optimize_type(
                    dict_lt,
                    dict_col,
                    terminal,
                    registry,
                    &format!("{context}[dict-candidate]"),
                )?;
                return pick_smaller_variant(
                    plain,
                    dict,
                    LogicalColumn::Utf8(strings),
                    registry,
                    context,
                );
            }
            Ok(plain)
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
        // recursive Struct — encodings live in FieldSpec, not in ColumnSpec::encodings
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
                let (updated_lt, encodings) = optimize_type_inner(
                    spec_f.logical_type,
                    col_lc,
                    terminal,
                    registry,
                    &field_ctx,
                    true,
                )?;
                new_fields.push(FieldSpec::new(spec_f.name, updated_lt, encodings));
            }
            Ok((LogicalType::Struct { fields: new_fields }, vec![]))
        }

        // ----------------------------------------------------------------
        // recursive List
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
                optimize_type_inner(*inner, *values, terminal, registry, &inner_ctx, true)?;
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
        // recursive Map
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
            // Map keys must remain Primitive/Utf8/Binary — never promote to Dictionary.
            let (updated_key, key_encs) =
                optimize_type_inner(*key, *keys, terminal, registry, &key_ctx, false)?;
            let (updated_value, val_encs) =
                optimize_type_inner(*value, *values, terminal, registry, &val_ctx, true)?;
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
        // recursive Nullable
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
                optimize_type_inner(*inner, *value, terminal, registry, &inner_ctx, true)?;
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
        // recursive Union
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
                    optimize_type_inner(v_lt, col_v_lc, terminal, registry, &v_ctx, true)?;
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
        // Dictionary{inner} — recursive dictionary encoding
        // ----------------------------------------------------------------
        (
            LogicalType::Dictionary { inner },
            LogicalColumn::Dictionary {
                dictionary,
                indices,
            },
        ) => {
            // Optimize inner type for the dictionary column. dict_allowed = false:
            // the dictionary values are already unique by construction, so nesting
            // a second Dictionary inside them is never a win and the inner leaf
            // (Utf8 / Primitive) must stay plain.
            let inner_ctx = format!("{context}[dict]");
            let (updated_inner, inner_encs) =
                optimize_type_inner(*inner, *dictionary, terminal, registry, &inner_ctx, false)?;
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
    terminal: &CoderSpec,
    registry: &CoderRegistry,
) -> Result<ColumnSpec> {
    let (updated_lt, encodings) = optimize_type(lt, lc, terminal, registry, name)?;
    Ok(ColumnSpec::new(name, updated_lt, encodings))
}

// ---------------------------------------------------------------------------
// Dictionary-promotion helpers
// ---------------------------------------------------------------------------

/// The distinct-count limit for a column of `row_count` rows, or `0` if the
/// column is too small to be a dictionary candidate at all.
///
/// A column only benefits from dict encoding when its values repeat heavily, so
/// we cap the distinct count at `row_count / 2` (above that the index leaf costs
/// roughly as much as the data) and at [`DICT_CARDINALITY_MAX`] (keeps the probe
/// cheap). Columns under [`DICT_MIN_ROWS`] are excluded.
fn dict_limit(row_count: usize) -> usize {
    if row_count < DICT_MIN_ROWS {
        return 0;
    }
    (row_count / 2).min(DICT_CARDINALITY_MAX)
}

/// Whether `cd` is an integer column eligible for dictionary promotion **and**
/// passes the cheap cardinality gate. Floats and bytes are never candidates.
fn is_dict_candidate_int(cd: &ColumnData) -> bool {
    let limit = dict_limit(cd.len());
    if limit == 0 {
        return false;
    }
    match cd {
        ColumnData::I8(v) => distinct_i32(&v.iter().map(|&x| x as i32).collect::<Vec<_>>(), limit),
        ColumnData::I16(v) => distinct_i32(&v.iter().map(|&x| x as i32).collect::<Vec<_>>(), limit),
        ColumnData::I32(v) => distinct_i32(v, limit),
        ColumnData::I64(v) => distinct_i64(v, limit),
        ColumnData::U8(v) => distinct_i64(&v.iter().map(|&x| x as i64).collect::<Vec<_>>(), limit),
        ColumnData::U16(v) => distinct_i64(&v.iter().map(|&x| x as i64).collect::<Vec<_>>(), limit),
        ColumnData::U32(v) => distinct_i64(&v.iter().map(|&x| x as i64).collect::<Vec<_>>(), limit),
        // u64 may exceed i64 range; build the distinct set inline with the same limit.
        ColumnData::U64(v) => distinct_u64(v, limit),
        ColumnData::F32(_) | ColumnData::F64(_) | ColumnData::Bytes(_) => None,
    }
    .is_some()
}

/// Returns the number of distinct U64 values if ≤ `limit`, else `None`.
/// (`distinct_i64` would lose the top half of the u64 range to truncation.)
fn distinct_u64(values: &[u64], limit: usize) -> Option<usize> {
    let mut set = std::collections::HashSet::new();
    for &v in values {
        set.insert(v);
        if set.len() > limit {
            return None;
        }
    }
    Some(set.len())
}

/// Build and optimize the dictionary variant of an integer primitive column.
///
/// Returns `Ok(None)` if `cd` cannot be dictionary-encoded (e.g. non-integer —
/// already filtered, but kept defensive). The cardinality gate is assumed to
/// have passed at the call site.
fn try_dict_primitive(
    cd: &ColumnData,
    data_type: DataType,
    terminal: &CoderSpec,
    registry: &CoderRegistry,
    context: &str,
) -> Result<Option<(LogicalType, Vec<Vec<CoderSpec>>)>> {
    let dict_col = match LogicalColumn::dict_encode_primitive(cd.clone()) {
        Ok(c) => c,
        // Non-integer input — not a candidate. (Unreachable given the gate.)
        Err(_) => return Ok(None),
    };
    let dict_lt = LogicalType::Dictionary {
        inner: Box::new(LogicalType::Primitive { data_type }),
    };
    let dict = optimize_type(
        dict_lt,
        dict_col,
        terminal,
        registry,
        &format!("{context}[dict-candidate]"),
    )?;
    Ok(Some(dict))
}

/// Measure both encoded variants of a column against the original sample data
/// and return whichever produces fewer bytes (plain on a tie, to avoid the
/// dictionary's extra runtime indirection when there is no size win).
fn pick_smaller_variant(
    plain: (LogicalType, Vec<Vec<CoderSpec>>),
    dict: (LogicalType, Vec<Vec<CoderSpec>>),
    original: LogicalColumn,
    registry: &CoderRegistry,
    context: &str,
) -> Result<(LogicalType, Vec<Vec<CoderSpec>>)> {
    let plain_spec = ColumnSpec::new(context, plain.0.clone(), plain.1.clone());
    let plain_size = measure_encoding(&plain_spec, original.clone(), registry)?;

    // The dict variant must be measured against the dict-encoded column, not the
    // plain one. Rebuild it from the same source so both measurements are honest.
    let dict_original = match &dict.0 {
        LogicalType::Dictionary { inner } => match (inner.as_ref(), &original) {
            (LogicalType::Utf8, LogicalColumn::Utf8(strings)) => {
                LogicalColumn::dict_encode_utf8(strings.clone())
            }
            (LogicalType::Primitive { .. }, LogicalColumn::Primitive(cd)) => {
                LogicalColumn::dict_encode_primitive(cd.clone())?
            }
            _ => return Ok(plain),
        },
        _ => return Ok(plain),
    };
    let dict_spec = ColumnSpec::new(context, dict.0.clone(), dict.1.clone());
    let dict_size = measure_encoding(&dict_spec, dict_original, registry)?;

    if dict_size < plain_size {
        Ok(dict)
    } else {
        Ok(plain)
    }
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
