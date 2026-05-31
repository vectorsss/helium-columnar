//! `LogicalColumn → ArrayRef` conversion.
//!
//! Converts every Helium logical-column variant (legacy flat and recursive) to an Arrow
//! [`ArrayRef`]. See the module-level doc in `super` for the type mapping
//! table and null-handling semantics.

use std::sync::Arc;

use arrow::array::{
    ArrayRef, BinaryArray, BooleanBufferBuilder, DictionaryArray, Float32Array, Float64Array,
    Int8Array, Int16Array, Int32Array, Int64Array, ListArray, MapArray, StringArray, StructArray,
    UInt8Array, UInt16Array, UInt32Array, UInt64Array, UnionArray,
};
use arrow::buffer::{NullBuffer, OffsetBuffer, ScalarBuffer};
use arrow::datatypes::{Field, Fields, UnionFields};

use crate::core::coder::ColumnData;
use crate::core::error::{HeliumError, Result};
use crate::core::schema::{LogicalColumn, LogicalType};

/// Convert a Helium [`LogicalColumn`] to an Arrow [`ArrayRef`].
///
/// The `logical_type` parameter drives the Arrow schema of the returned array
/// (i.e. which DataType fields are emitted). For nullable wrappers the
/// present mask is converted to Arrow's null bitmap.
///
/// # Errors
///
/// Returns `HeliumError::RuntimeType` for type mismatches (e.g., data doesn't
/// match the declared `logical_type`), or `HeliumError::Format` for overflow
/// conditions (e.g., list offsets exceed `i32::MAX`).
pub fn to_arrow_array(col: &LogicalColumn, lt: &LogicalType) -> Result<ArrayRef> {
    match (col, lt) {
        // ------------------------------------------------------------------ //
        // Primitive types                                                     //
        // ------------------------------------------------------------------ //
        (LogicalColumn::Primitive(ColumnData::I8(v)), LogicalType::Primitive { .. }) => {
            Ok(Arc::new(Int8Array::from(v.clone())))
        }
        (LogicalColumn::Primitive(ColumnData::I16(v)), LogicalType::Primitive { .. }) => {
            Ok(Arc::new(Int16Array::from(v.clone())))
        }
        (LogicalColumn::Primitive(ColumnData::I32(v)), LogicalType::Primitive { .. }) => {
            Ok(Arc::new(Int32Array::from(v.clone())))
        }
        (LogicalColumn::Primitive(ColumnData::I64(v)), LogicalType::Primitive { .. }) => {
            Ok(Arc::new(Int64Array::from(v.clone())))
        }
        (LogicalColumn::Primitive(ColumnData::U8(v)), LogicalType::Primitive { .. }) => {
            Ok(Arc::new(UInt8Array::from(v.clone())))
        }
        (LogicalColumn::Primitive(ColumnData::U16(v)), LogicalType::Primitive { .. }) => {
            Ok(Arc::new(UInt16Array::from(v.clone())))
        }
        (LogicalColumn::Primitive(ColumnData::U32(v)), LogicalType::Primitive { .. }) => {
            Ok(Arc::new(UInt32Array::from(v.clone())))
        }
        (LogicalColumn::Primitive(ColumnData::U64(v)), LogicalType::Primitive { .. }) => {
            Ok(Arc::new(UInt64Array::from(v.clone())))
        }
        (LogicalColumn::Primitive(ColumnData::F32(v)), LogicalType::Primitive { .. }) => {
            Ok(Arc::new(Float32Array::from(v.clone())))
        }
        (LogicalColumn::Primitive(ColumnData::F64(v)), LogicalType::Primitive { .. }) => {
            Ok(Arc::new(Float64Array::from(v.clone())))
        }

        // ------------------------------------------------------------------ //
        // Utf8 and Binary                                                     //
        // ------------------------------------------------------------------ //
        (LogicalColumn::Utf8(v), LogicalType::Utf8) => Ok(Arc::new(StringArray::from(
            v.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        ))),
        (LogicalColumn::Binary(v), LogicalType::Binary) => Ok(Arc::new(BinaryArray::from(
            v.iter().map(|b| b.as_slice()).collect::<Vec<_>>(),
        ))),

        // ------------------------------------------------------------------ //
        // Nullable (recursive) — compact inner, expanded Arrow representation       //
        // ------------------------------------------------------------------ //
        (LogicalColumn::Nullable { present, value }, LogicalType::Nullable { inner }) => {
            to_arrow_nullable(present, value, inner)
        }

        // ------------------------------------------------------------------ //
        // List (recursive) — offsets + inner values                                 //
        // ------------------------------------------------------------------ //
        (LogicalColumn::List { offsets, values }, LogicalType::List { inner }) => {
            to_arrow_list(offsets, values, inner)
        }

        // ------------------------------------------------------------------ //
        // Map (recursive)                                                            //
        // ------------------------------------------------------------------ //
        (
            LogicalColumn::Map {
                offsets,
                keys,
                values,
            },
            LogicalType::Map {
                key: key_lt,
                value: val_lt,
            },
        ) => to_arrow_map(offsets, keys, values, key_lt, val_lt),

        // ------------------------------------------------------------------ //
        // Struct (recursive)                                                         //
        // ------------------------------------------------------------------ //
        (
            LogicalColumn::Struct { fields },
            LogicalType::Struct {
                fields: spec_fields,
            },
        ) => to_arrow_struct(fields, spec_fields),

        // ------------------------------------------------------------------ //
        // Union (recursive) — dense union                                           //
        // ------------------------------------------------------------------ //
        (
            LogicalColumn::Union { tags, variants },
            LogicalType::Union {
                variants: spec_variants,
            },
        ) => to_arrow_union(tags, variants, spec_variants),

        // ------------------------------------------------------------------ //
        // Dictionary{inner} — recursive                                   //
        // ------------------------------------------------------------------ //
        (
            LogicalColumn::Dictionary {
                dictionary,
                indices,
            },
            LogicalType::Dictionary { inner },
        ) => to_arrow_dict(dictionary, indices, inner),

        // ------------------------------------------------------------------ //
        // Semantic types                                                       //
        // ------------------------------------------------------------------ //
        (LogicalColumn::Decimal128 { values }, LogicalType::Decimal128 { precision, scale }) => {
            use arrow::array::Decimal128Array;
            let arr = Decimal128Array::from(values.clone())
                .with_precision_and_scale(*precision, *scale as i8)
                .map_err(|e| HeliumError::Format(format!("Decimal128 precision/scale: {e}")))?;
            Ok(Arc::new(arr) as ArrayRef)
        }

        (
            LogicalColumn::Date32 { values },
            LogicalType::Date {
                unit: crate::core::schema::DateUnit::Days,
            },
        ) => {
            use arrow::array::Date32Array;
            Ok(Arc::new(Date32Array::from(values.clone())) as ArrayRef)
        }

        (
            LogicalColumn::Date64 { values },
            LogicalType::Date {
                unit: crate::core::schema::DateUnit::Millis,
            },
        ) => {
            use arrow::array::Date64Array;
            Ok(Arc::new(Date64Array::from(values.clone())) as ArrayRef)
        }

        (LogicalColumn::Datetime { values }, LogicalType::Datetime { unit, timezone }) => {
            use crate::core::schema::TimeUnit;
            use arrow::array::TimestampMicrosecondArray;
            use arrow::array::TimestampMillisecondArray;
            use arrow::array::TimestampNanosecondArray;
            use arrow::array::TimestampSecondArray;
            let tz: Option<Arc<str>> = timezone.as_deref().map(Arc::from);
            match unit {
                TimeUnit::Seconds => {
                    let arr = TimestampSecondArray::from(values.clone()).with_timezone_opt(tz);
                    Ok(Arc::new(arr) as ArrayRef)
                }
                TimeUnit::Millis => {
                    let arr = TimestampMillisecondArray::from(values.clone()).with_timezone_opt(tz);
                    Ok(Arc::new(arr) as ArrayRef)
                }
                TimeUnit::Micros => {
                    let arr = TimestampMicrosecondArray::from(values.clone()).with_timezone_opt(tz);
                    Ok(Arc::new(arr) as ArrayRef)
                }
                TimeUnit::Nanos => {
                    let arr = TimestampNanosecondArray::from(values.clone()).with_timezone_opt(tz);
                    Ok(Arc::new(arr) as ArrayRef)
                }
            }
        }

        // ------------------------------------------------------------------ //
        // Type mismatch                                                       //
        // ------------------------------------------------------------------ //
        _ => Err(HeliumError::Format(format!(
            "to_arrow_array: logical type {lt:?} does not match column variant {}",
            col_type_name(col)
        ))),
    }
}

// ---------------------------------------------------------------------------
// Nullable → Arrow with null bitmap
// ---------------------------------------------------------------------------

fn to_arrow_nullable(
    present: &[bool],
    value: &LogicalColumn,
    inner: &LogicalType,
) -> Result<ArrayRef> {
    let null_buf = build_null_buffer(present);

    // For inner types we need to expand compact → full-length first.
    match (value, inner) {
        (LogicalColumn::Primitive(data), LogicalType::Primitive { .. }) => {
            let expanded = expand_nullable_primitive(present, data);
            primitive_with_nulls(expanded, &null_buf, inner)
        }
        (LogicalColumn::Utf8(strings), LogicalType::Utf8) => {
            let expanded = expand_nullable_strings(present, strings);
            let arr = StringArray::from(expanded.iter().map(|s| s.as_deref()).collect::<Vec<_>>());
            Ok(Arc::new(arr.with_validity_opt(null_buf)) as ArrayRef)
        }
        (LogicalColumn::Binary(blobs), LogicalType::Binary) => {
            let expanded = expand_nullable_binary(present, blobs);
            let arr = BinaryArray::from(expanded.iter().map(|b| b.as_deref()).collect::<Vec<_>>());
            Ok(Arc::new(arr.with_validity_opt(null_buf)) as ArrayRef)
        }
        // For nested types (List, Struct, Map, Nullable, Union) we expand the
        // compact column to full length using Arrow's `take` kernel:
        // 1. Convert the compact inner column to Arrow (k rows).
        // 2. Build a one-element "zero row" array for the inner type.
        // 3. Use `take` to scatter: null positions get index 0 from the
        //    zero-row array (appended at the tail), valid positions get
        //    indices 1..k from the compact array.
        // 4. Apply the null bitmap.
        _ => {
            let n = present.len();
            // Convert compact inner to Arrow (k rows)
            let compact_arr = to_arrow_array(value, inner)?;
            let k = compact_arr.len();

            // Build a zero-row (1 element) for the inner type as a fallback
            // for null positions
            let zero_inner = make_zero_row(inner)?;
            let zero_arr = to_arrow_array(&zero_inner, inner)?;

            // Concatenate: [zero_row, ...compact_rows]
            let combined = arrow::compute::concat(&[zero_arr.as_ref(), compact_arr.as_ref()])
                .map_err(|e| HeliumError::Format(format!("arrow concat: {e}")))?;

            // Build take-indices: null → 0 (zero row); valid → 1..k in compact order
            let mut take_indices: Vec<u32> = Vec::with_capacity(n);
            let mut compact_idx = 1u32; // compact array starts at index 1 in combined
            for &p in present {
                if p {
                    take_indices.push(compact_idx);
                    compact_idx += 1;
                } else {
                    take_indices.push(0); // points to the zero row
                }
            }
            let _ = k; // compact_idx validates we consumed exactly k elements

            let indices_arr = UInt32Array::from(take_indices);
            let expanded = arrow::compute::take(combined.as_ref(), &indices_arr, None)
                .map_err(|e| HeliumError::Format(format!("arrow take: {e}")))?;

            // Apply the null buffer
            let data = expanded
                .to_data()
                .into_builder()
                .null_bit_buffer(null_buf.map(|nb| nb.into_inner().into_inner()))
                .build()
                .map_err(|e| HeliumError::Format(format!("Arrow ArrayData build: {e}")))?;
            Ok(arrow::array::make_array(data))
        }
    }
}

/// Build an Arrow `NullBuffer` from a Helium present slice.
/// Returns `None` if all values are present (no nulls), which Arrow
/// treats as "fully valid."
fn build_null_buffer(present: &[bool]) -> Option<NullBuffer> {
    let all_present = present.iter().all(|&p| p);
    if all_present {
        return None;
    }
    let mut bb = BooleanBufferBuilder::new(present.len());
    for &p in present {
        bb.append(p);
    }
    Some(NullBuffer::new(bb.finish()))
}

/// Expand a compact ColumnData (only non-null values) to full-length,
/// inserting 0 / false placeholders at null positions.
fn expand_nullable_primitive(present: &[bool], compact: &ColumnData) -> ColumnData {
    let n = present.len();
    match compact {
        ColumnData::I8(v) => {
            let mut out = vec![0i8; n];
            let mut src = v.iter();
            for (i, &p) in present.iter().enumerate() {
                if p {
                    out[i] = *src.next().unwrap_or(&0);
                }
            }
            ColumnData::I8(out)
        }
        ColumnData::I16(v) => {
            let mut out = vec![0i16; n];
            let mut src = v.iter();
            for (i, &p) in present.iter().enumerate() {
                if p {
                    out[i] = *src.next().unwrap_or(&0);
                }
            }
            ColumnData::I16(out)
        }
        ColumnData::I32(v) => {
            let mut out = vec![0i32; n];
            let mut src = v.iter();
            for (i, &p) in present.iter().enumerate() {
                if p {
                    out[i] = *src.next().unwrap_or(&0);
                }
            }
            ColumnData::I32(out)
        }
        ColumnData::I64(v) => {
            let mut out = vec![0i64; n];
            let mut src = v.iter();
            for (i, &p) in present.iter().enumerate() {
                if p {
                    out[i] = *src.next().unwrap_or(&0);
                }
            }
            ColumnData::I64(out)
        }
        ColumnData::U8(v) => {
            let mut out = vec![0u8; n];
            let mut src = v.iter();
            for (i, &p) in present.iter().enumerate() {
                if p {
                    out[i] = *src.next().unwrap_or(&0);
                }
            }
            ColumnData::U8(out)
        }
        ColumnData::U16(v) => {
            let mut out = vec![0u16; n];
            let mut src = v.iter();
            for (i, &p) in present.iter().enumerate() {
                if p {
                    out[i] = *src.next().unwrap_or(&0);
                }
            }
            ColumnData::U16(out)
        }
        ColumnData::U32(v) => {
            let mut out = vec![0u32; n];
            let mut src = v.iter();
            for (i, &p) in present.iter().enumerate() {
                if p {
                    out[i] = *src.next().unwrap_or(&0);
                }
            }
            ColumnData::U32(out)
        }
        ColumnData::U64(v) => {
            let mut out = vec![0u64; n];
            let mut src = v.iter();
            for (i, &p) in present.iter().enumerate() {
                if p {
                    out[i] = *src.next().unwrap_or(&0);
                }
            }
            ColumnData::U64(out)
        }
        ColumnData::F32(v) => {
            let mut out = vec![0.0f32; n];
            let mut src = v.iter();
            for (i, &p) in present.iter().enumerate() {
                if p {
                    out[i] = *src.next().unwrap_or(&0.0);
                }
            }
            ColumnData::F32(out)
        }
        ColumnData::F64(v) => {
            let mut out = vec![0.0f64; n];
            let mut src = v.iter();
            for (i, &p) in present.iter().enumerate() {
                if p {
                    out[i] = *src.next().unwrap_or(&0.0);
                }
            }
            ColumnData::F64(out)
        }
        ColumnData::Bytes(_) => compact.clone(), // Should not happen for primitives
    }
}

/// Expand compact `Option<String>` list to full-length with `None` at null positions.
fn expand_nullable_strings(present: &[bool], compact: &[String]) -> Vec<Option<String>> {
    let n = present.len();
    let mut out = vec![None; n];
    let mut src = compact.iter();
    for (i, &p) in present.iter().enumerate() {
        if p {
            out[i] = src.next().cloned();
        }
    }
    out
}

/// Expand compact `Option<Vec<u8>>` list to full-length with `None` at null positions.
fn expand_nullable_binary(present: &[bool], compact: &[Vec<u8>]) -> Vec<Option<Vec<u8>>> {
    let n = present.len();
    let mut out = vec![None; n];
    let mut src = compact.iter();
    for (i, &p) in present.iter().enumerate() {
        if p {
            out[i] = src.next().cloned();
        }
    }
    out
}

/// Build a single "zero" row for a given LogicalType.
fn make_zero_row(lt: &LogicalType) -> Result<LogicalColumn> {
    Ok(match lt {
        LogicalType::Primitive { data_type } => LogicalColumn::Primitive(match data_type {
            crate::core::coder::DataType::I8 => ColumnData::I8(vec![0]),
            crate::core::coder::DataType::I16 => ColumnData::I16(vec![0]),
            crate::core::coder::DataType::I32 => ColumnData::I32(vec![0]),
            crate::core::coder::DataType::I64 => ColumnData::I64(vec![0]),
            crate::core::coder::DataType::U8 => ColumnData::U8(vec![0]),
            crate::core::coder::DataType::U16 => ColumnData::U16(vec![0]),
            crate::core::coder::DataType::U32 => ColumnData::U32(vec![0]),
            crate::core::coder::DataType::U64 => ColumnData::U64(vec![0]),
            crate::core::coder::DataType::F32 => ColumnData::F32(vec![0.0]),
            crate::core::coder::DataType::F64 => ColumnData::F64(vec![0.0]),
            crate::core::coder::DataType::Bytes => ColumnData::Bytes(vec![]),
        }),
        LogicalType::Utf8 => LogicalColumn::Utf8(vec![String::new()]),
        LogicalType::Binary => LogicalColumn::Binary(vec![vec![]]),
        LogicalType::List { .. } => LogicalColumn::List {
            offsets: vec![0, 0],
            values: Box::new(make_empty_inner_for_list(lt)?),
        },
        LogicalType::Map { key, value } => LogicalColumn::Map {
            offsets: vec![0, 0],
            keys: Box::new(make_empty_col(key)),
            values: Box::new(make_empty_col(value)),
        },
        LogicalType::Nullable { inner } => LogicalColumn::Nullable {
            present: vec![false],
            value: Box::new(make_zero_row(inner)?),
        },
        LogicalType::Struct { fields } => {
            let f: Result<Vec<_>> = fields
                .iter()
                .map(|f| Ok((f.name.clone(), make_zero_row(&f.logical_type)?)))
                .collect();
            LogicalColumn::Struct { fields: f? }
        }
        LogicalType::Union { variants } => {
            // Use tag 0 with zero row for variant 0
            let vars: Result<Vec<_>> = variants
                .iter()
                .enumerate()
                .map(|(i, (n, v_lt))| {
                    let data = if i == 0 {
                        make_zero_row(v_lt)?
                    } else {
                        make_empty_col(v_lt)
                    };
                    Ok((n.clone(), data))
                })
                .collect();
            LogicalColumn::Union {
                tags: vec![0],
                variants: vars?,
            }
        }
        // Semantic type extensions
        LogicalType::Decimal128 { .. } => LogicalColumn::Decimal128 {
            values: vec![0i128],
        },
        LogicalType::Date {
            unit: crate::core::schema::DateUnit::Days,
        } => LogicalColumn::Date32 { values: vec![0i32] },
        LogicalType::Date {
            unit: crate::core::schema::DateUnit::Millis,
        } => LogicalColumn::Date64 { values: vec![0i64] },
        LogicalType::Datetime { .. } => LogicalColumn::Datetime { values: vec![0i64] },
        // Dictionary{inner} — 1-row dict: dictionary has 1 entry, index 0.
        LogicalType::Dictionary { inner } => LogicalColumn::Dictionary {
            dictionary: Box::new(make_zero_row(inner)?),
            indices: vec![0],
        },
    })
}

/// Build an empty (0-row) LogicalColumn for the given type.
fn make_empty_col(lt: &LogicalType) -> LogicalColumn {
    match lt {
        LogicalType::Primitive { data_type } => LogicalColumn::Primitive(match data_type {
            crate::core::coder::DataType::I8 => ColumnData::I8(vec![]),
            crate::core::coder::DataType::I16 => ColumnData::I16(vec![]),
            crate::core::coder::DataType::I32 => ColumnData::I32(vec![]),
            crate::core::coder::DataType::I64 => ColumnData::I64(vec![]),
            crate::core::coder::DataType::U8 => ColumnData::U8(vec![]),
            crate::core::coder::DataType::U16 => ColumnData::U16(vec![]),
            crate::core::coder::DataType::U32 => ColumnData::U32(vec![]),
            crate::core::coder::DataType::U64 => ColumnData::U64(vec![]),
            crate::core::coder::DataType::F32 => ColumnData::F32(vec![]),
            crate::core::coder::DataType::F64 => ColumnData::F64(vec![]),
            crate::core::coder::DataType::Bytes => ColumnData::Bytes(vec![]),
        }),
        LogicalType::Utf8 => LogicalColumn::Utf8(vec![]),
        LogicalType::Binary => LogicalColumn::Binary(vec![]),
        LogicalType::List { inner } => LogicalColumn::List {
            offsets: vec![0],
            values: Box::new(make_empty_col(inner)),
        },
        LogicalType::Map { key, value } => LogicalColumn::Map {
            offsets: vec![0],
            keys: Box::new(make_empty_col(key)),
            values: Box::new(make_empty_col(value)),
        },
        LogicalType::Nullable { inner } => LogicalColumn::Nullable {
            present: vec![],
            value: Box::new(make_empty_col(inner)),
        },
        LogicalType::Union { variants } => LogicalColumn::Union {
            tags: vec![],
            variants: variants
                .iter()
                .map(|(n, v_lt)| (n.clone(), make_empty_col(v_lt)))
                .collect(),
        },
        LogicalType::Struct { fields } => LogicalColumn::Struct {
            fields: fields
                .iter()
                .map(|f| (f.name.clone(), make_empty_col(&f.logical_type)))
                .collect(),
        },
        // Semantic types
        LogicalType::Decimal128 { .. } => LogicalColumn::Decimal128 { values: vec![] },
        LogicalType::Date {
            unit: crate::core::schema::DateUnit::Days,
        } => LogicalColumn::Date32 { values: vec![] },
        LogicalType::Date {
            unit: crate::core::schema::DateUnit::Millis,
        } => LogicalColumn::Date64 { values: vec![] },
        LogicalType::Datetime { .. } => LogicalColumn::Datetime { values: vec![] },
        // Dictionary{inner}: empty dictionary, no indices.
        LogicalType::Dictionary { inner } => LogicalColumn::Dictionary {
            dictionary: Box::new(make_empty_col(inner)),
            indices: vec![],
        },
    }
}

/// Helper for making empty inner column for a 1-row empty list/map.
fn make_empty_inner_for_list(lt: &LogicalType) -> Result<LogicalColumn> {
    Ok(match lt {
        LogicalType::List { inner } => make_empty_col(inner),
        LogicalType::Map { key, .. } => make_empty_col(key),
        _ => make_empty_col(lt),
    })
}

// ---------------------------------------------------------------------------
// Primitive with null buffer
// ---------------------------------------------------------------------------

fn primitive_with_nulls(
    expanded: ColumnData,
    null_buf: &Option<NullBuffer>,
    _lt: &LogicalType,
) -> Result<ArrayRef> {
    let nulls = null_buf.clone();
    Ok(match expanded {
        ColumnData::I8(v) => Arc::new(Int8Array::new(ScalarBuffer::from(v), nulls)),
        ColumnData::I16(v) => Arc::new(Int16Array::new(ScalarBuffer::from(v), nulls)),
        ColumnData::I32(v) => Arc::new(Int32Array::new(ScalarBuffer::from(v), nulls)),
        ColumnData::I64(v) => Arc::new(Int64Array::new(ScalarBuffer::from(v), nulls)),
        ColumnData::U8(v) => Arc::new(UInt8Array::new(ScalarBuffer::from(v), nulls)),
        ColumnData::U16(v) => Arc::new(UInt16Array::new(ScalarBuffer::from(v), nulls)),
        ColumnData::U32(v) => Arc::new(UInt32Array::new(ScalarBuffer::from(v), nulls)),
        ColumnData::U64(v) => Arc::new(UInt64Array::new(ScalarBuffer::from(v), nulls)),
        ColumnData::F32(v) => Arc::new(Float32Array::new(ScalarBuffer::from(v), nulls)),
        ColumnData::F64(v) => Arc::new(Float64Array::new(ScalarBuffer::from(v), nulls)),
        ColumnData::Bytes(_) => {
            return Err(HeliumError::Format(
                "primitive_with_nulls: Bytes is not a valid primitive".into(),
            ));
        }
    })
}

// ---------------------------------------------------------------------------
// List
// ---------------------------------------------------------------------------

fn to_arrow_list(
    offsets: &[u32],
    values: &LogicalColumn,
    inner_lt: &LogicalType,
) -> Result<ArrayRef> {
    // Convert offsets from u32 to i32 (Arrow uses i32 for List)
    let i32_offsets: Vec<i32> = offsets
        .iter()
        .map(|&o| {
            i32::try_from(o).map_err(|_| {
                HeliumError::Format(format!(
                    "list offset {o} exceeds i32::MAX; cannot represent as Arrow List"
                ))
            })
        })
        .collect::<Result<_>>()?;

    let inner_arr = to_arrow_array(values, inner_lt)?;
    let arrow_inner_field = Arc::new(Field::new("item", inner_arr.data_type().clone(), true));
    let offset_buf = OffsetBuffer::new(ScalarBuffer::from(i32_offsets));
    let arr = ListArray::new(arrow_inner_field, offset_buf, inner_arr, None);
    Ok(Arc::new(arr))
}

// ---------------------------------------------------------------------------
// Map
// ---------------------------------------------------------------------------

fn to_arrow_map(
    offsets: &[u32],
    keys: &LogicalColumn,
    values: &LogicalColumn,
    key_lt: &LogicalType,
    val_lt: &LogicalType,
) -> Result<ArrayRef> {
    let key_arr = to_arrow_array(keys, key_lt)?;
    let val_arr = to_arrow_array(values, val_lt)?;

    // Arrow Map is internally List<Struct<keys: K, values: V>>
    // keys are NOT NULL, values are NULLABLE per Arrow convention
    let key_field = Arc::new(Field::new("key", key_arr.data_type().clone(), false));
    let val_field = Arc::new(Field::new("value", val_arr.data_type().clone(), true));
    let struct_fields = Fields::from(vec![key_field.as_ref().clone(), val_field.as_ref().clone()]);
    let entries = StructArray::new(struct_fields, vec![key_arr, val_arr], None);

    let i32_offsets: Vec<i32> = offsets
        .iter()
        .map(|&o| {
            i32::try_from(o)
                .map_err(|_| HeliumError::Format(format!("map offset {o} exceeds i32::MAX")))
        })
        .collect::<Result<_>>()?;

    let offset_buf = OffsetBuffer::new(ScalarBuffer::from(i32_offsets));
    let entries_dt = {
        use arrow::array::Array;
        entries.data_type().clone()
    };
    let arr = MapArray::new(
        Arc::new(Field::new("entries", entries_dt, false)),
        offset_buf,
        entries,
        None,
        false, // keys_sorted = false
    );
    Ok(Arc::new(arr))
}

// ---------------------------------------------------------------------------
// Struct
// ---------------------------------------------------------------------------

fn to_arrow_struct(
    fields: &[(String, LogicalColumn)],
    spec_fields: &[crate::core::schema::FieldSpec],
) -> Result<ArrayRef> {
    if fields.len() != spec_fields.len() {
        return Err(HeliumError::Format(format!(
            "to_arrow_struct: expected {} fields but got {}",
            spec_fields.len(),
            fields.len()
        )));
    }

    let mut arrow_fields = Vec::with_capacity(fields.len());
    let mut arrays = Vec::with_capacity(fields.len());

    for ((name, col), spec) in fields.iter().zip(spec_fields.iter()) {
        let arr = to_arrow_array(col, &spec.logical_type)?;
        let nullable = matches!(spec.logical_type, LogicalType::Nullable { .. });
        arrow_fields.push(Field::new(name.clone(), arr.data_type().clone(), nullable));
        arrays.push(arr);
    }

    let arrow_struct_fields = Fields::from(arrow_fields);
    let arr = StructArray::new(arrow_struct_fields, arrays, None);
    Ok(Arc::new(arr))
}

// ---------------------------------------------------------------------------
// Union (dense)
// ---------------------------------------------------------------------------

fn to_arrow_union(
    tags: &[u8],
    variants: &[(String, LogicalColumn)],
    spec_variants: &[(String, LogicalType)],
) -> Result<ArrayRef> {
    if variants.len() != spec_variants.len() {
        return Err(HeliumError::Format(format!(
            "to_arrow_union: expected {} variants but got {}",
            spec_variants.len(),
            variants.len()
        )));
    }

    // Arrow dense union: type_ids + offsets + per-type child arrays.
    // Helium stores compact per-variant columns; we need per-variant
    // row index tracking (offset within that variant's child array).
    let n_variants = variants.len();

    let mut union_fields_vec: Vec<(i8, Arc<Field>)> = Vec::with_capacity(n_variants);
    let mut children: Vec<ArrayRef> = Vec::with_capacity(n_variants);

    for (i, ((v_name, v_col), (_, v_lt))) in variants.iter().zip(spec_variants.iter()).enumerate() {
        let child_arr = to_arrow_array(v_col, v_lt)?;
        let type_id = i as i8;
        let field = Arc::new(Field::new(
            v_name.clone(),
            child_arr.data_type().clone(),
            true,
        ));
        union_fields_vec.push((type_id, field));
        children.push(child_arr);
    }

    // Dense union: offsets[i] = index into the child array for type_ids[i]
    let mut variant_offsets: Vec<i32> = vec![0; n_variants];
    let mut type_ids: Vec<i8> = Vec::with_capacity(tags.len());
    let mut offsets: Vec<i32> = Vec::with_capacity(tags.len());

    for &tag in tags {
        let t = tag as usize;
        let off = variant_offsets[t];
        type_ids.push(tag as i8);
        offsets.push(off);
        variant_offsets[t] += 1;
    }

    let union_fields = UnionFields::new(
        union_fields_vec.iter().map(|(id, _)| *id),
        union_fields_vec.iter().map(|(_, f)| f.as_ref().clone()),
    );

    let arr = UnionArray::try_new(
        union_fields,
        ScalarBuffer::from(type_ids),
        Some(ScalarBuffer::from(offsets)),
        children,
    )
    .map_err(|e| HeliumError::Format(format!("Arrow UnionArray::try_new: {e}")))?;

    Ok(Arc::new(arr))
}

// ---------------------------------------------------------------------------
// Dictionary{inner} — recursive
// ---------------------------------------------------------------------------

/// Convert a Helium recursive `Dictionary { dictionary, indices }` to an Arrow
/// `DictionaryArray<UInt32>`.
///
/// The `dictionary` LogicalColumn is converted to an Arrow array via
/// [`to_arrow_array`], then used as the dictionary values buffer.  If the
/// inner type is not directly supported as an Arrow dictionary value type
/// this returns a clear `HeliumError::Format("unsupported")` rather than a
/// wrong mapping.
fn to_arrow_dict(
    dictionary: &LogicalColumn,
    indices: &[u32],
    inner: &LogicalType,
) -> Result<ArrayRef> {
    use arrow::datatypes::UInt32Type;

    let dict_arr = to_arrow_array(dictionary, inner)?;
    let index_arr = UInt32Array::from(indices.to_vec());
    let dict_final = DictionaryArray::<UInt32Type>::try_new(index_arr, dict_arr)
        .map_err(|e| HeliumError::Format(format!("Dictionary{{inner}} arrow error: {e}")))?;
    Ok(Arc::new(dict_final))
}

// ---------------------------------------------------------------------------
// Helper: variant name for error messages
// ---------------------------------------------------------------------------

fn col_type_name(col: &LogicalColumn) -> &'static str {
    match col {
        LogicalColumn::Primitive(_) => "Primitive",
        LogicalColumn::Utf8(_) => "Utf8",
        LogicalColumn::Binary(_) => "Binary",
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

// ---------------------------------------------------------------------------
// Trait extension: add validity from an optional NullBuffer
// ---------------------------------------------------------------------------

trait WithValidityOpt: Sized {
    fn with_validity_opt(self, nb: Option<NullBuffer>) -> Self;
}

impl WithValidityOpt for StringArray {
    fn with_validity_opt(self, nb: Option<NullBuffer>) -> Self {
        match nb {
            None => self,
            Some(nulls) => {
                let (off, data, _) = self.into_parts();
                StringArray::new(off, data, Some(nulls))
            }
        }
    }
}

impl WithValidityOpt for BinaryArray {
    fn with_validity_opt(self, nb: Option<NullBuffer>) -> Self {
        match nb {
            None => self,
            Some(nulls) => {
                let (off, data, _) = self.into_parts();
                BinaryArray::new(off, data, Some(nulls))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primitive_i64_no_nulls() {
        let col = LogicalColumn::Primitive(ColumnData::I64(vec![1, 2, 3]));
        let lt = LogicalType::Primitive {
            data_type: crate::core::coder::DataType::I64,
        };
        let arr = to_arrow_array(&col, &lt).unwrap();
        assert_eq!(arr.len(), 3);
        assert_eq!(arr.null_count(), 0);
    }

    #[test]
    fn nullable_i64_with_nulls() {
        let col = LogicalColumn::Nullable {
            present: vec![true, false, true],
            value: Box::new(LogicalColumn::Primitive(ColumnData::I64(vec![10, 30]))),
        };
        let lt = LogicalType::Nullable {
            inner: Box::new(LogicalType::Primitive {
                data_type: crate::core::coder::DataType::I64,
            }),
        };
        let arr = to_arrow_array(&col, &lt).unwrap();
        assert_eq!(arr.len(), 3);
        assert_eq!(arr.null_count(), 1);
        let int_arr = arr.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(int_arr.value(0), 10);
        use arrow::array::Array;
        assert!(Array::is_null(int_arr, 1));
        assert_eq!(int_arr.value(2), 30);
    }

    #[test]
    fn utf8_basic() {
        let col = LogicalColumn::Utf8(vec!["hello".to_string(), "world".to_string()]);
        let lt = LogicalType::Utf8;
        let arr = to_arrow_array(&col, &lt).unwrap();
        assert_eq!(arr.len(), 2);
        let str_arr = arr.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(str_arr.value(0), "hello");
        assert_eq!(str_arr.value(1), "world");
    }

    #[test]
    fn list_primitive() {
        let col = LogicalColumn::List {
            offsets: vec![0, 2, 2, 3],
            values: Box::new(LogicalColumn::Primitive(ColumnData::I32(vec![1, 2, 3]))),
        };
        let lt = LogicalType::List {
            inner: Box::new(LogicalType::Primitive {
                data_type: crate::core::coder::DataType::I32,
            }),
        };
        let arr = to_arrow_array(&col, &lt).unwrap();
        assert_eq!(arr.len(), 3); // 3 rows
        let list_arr = arr.as_any().downcast_ref::<ListArray>().unwrap();
        assert_eq!(list_arr.value_length(0), 2); // row 0 has 2 elements
        assert_eq!(list_arr.value_length(1), 0); // row 1 is empty
        assert_eq!(list_arr.value_length(2), 1); // row 2 has 1 element
    }
}
