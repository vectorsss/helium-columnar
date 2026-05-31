//! `ArrayRef → LogicalColumn` conversion.
//!
//! Converts an Arrow [`ArrayRef`] to a Helium [`LogicalColumn`]. Always
//! produces **recursive-shaped** variants (`Nullable`, `List`, `Struct`, etc.) —
//! never legacy flat variants. The legacy flat variants are only emitted by legacy Helium readers.
//!
//! # Null handling
//!
//! Arrow uses expanded representation (full-length buffer + null bitmap).
//! Helium `Nullable` uses compact inner storage (only non-null rows).
//! This module contracts the Arrow representation: it extracts only the valid
//! rows from the Arrow array to form the compact Helium inner column.

use arrow::array::{
    Array, ArrayRef, BinaryArray, DictionaryArray, Float32Array, Float64Array, Int8Array,
    Int16Array, Int32Array, Int64Array, ListArray, MapArray, StringArray, StructArray, UInt8Array,
    UInt16Array, UInt32Array, UInt64Array, UnionArray,
};
use arrow::datatypes::{DataType as ArrowDataType, UInt32Type, UnionMode};

use crate::core::coder::ColumnData;
use crate::core::error::{HeliumError, Result};
use crate::core::schema::{FieldSpec, LogicalColumn, LogicalType};

/// Convert an Arrow [`ArrayRef`] to a Helium [`LogicalColumn`].
///
/// The `hint` parameter provides the target Helium `LogicalType`, which is
/// used to guide the conversion (e.g., to distinguish `Dictionary`
/// from `Primitive` or `Utf8` values).
///
/// When `hint` is `None`, the conversion is purely Arrow-schema-driven and
/// always produces recursive-shaped output.
///
/// # Inverse direction note
///
/// The returned `LogicalColumn` is always recursive-shaped:
/// - `Nullable { present, value }` (never `NullablePrim`/`NullableUtf8`)
/// - `List { offsets, values }` (never `ArrayOf`/`ArrayOfUtf8`)
///
/// Legacy flat variants round-tripped through Arrow will be returned as their
/// equivalent recursive shapes.
pub fn from_arrow_array(array: &ArrayRef, hint: &LogicalType) -> Result<LogicalColumn> {
    from_arrow_inner(array, hint)
}

fn from_arrow_inner(array: &ArrayRef, hint: &LogicalType) -> Result<LogicalColumn> {
    // Check if the array has nulls; if so, wrap in Nullable
    // UNLESS the hint is already Nullable (then extract present from the
    // null buffer and delegate to the inner type).
    match hint {
        LogicalType::Nullable { inner } => {
            return from_arrow_nullable(array, inner);
        }
        // legacy flat compatibility hints: map to recursive forms
        LogicalType::NullablePrim { data_type } => {
            let inner_lt = LogicalType::Primitive {
                data_type: *data_type,
            };
            return from_arrow_nullable(array, &inner_lt);
        }
        LogicalType::NullableUtf8 => {
            return from_arrow_nullable(array, &LogicalType::Utf8);
        }
        LogicalType::NullableBinary => {
            return from_arrow_nullable(array, &LogicalType::Binary);
        }
        _ => {}
    }

    // Non-nullable types: if the Arrow array has nulls, we still extract
    // non-null values only (treating it as if there were no null wrapper).
    // This is consistent with "Arrow -> Helium always produces a recursive Nullable
    // when there are nulls" — but the caller requested a non-nullable type,
    // so we just ignore the null buffer.

    match (array.data_type(), hint) {
        // ------------------------------------------------------------------ //
        // Primitive types                                                     //
        // ------------------------------------------------------------------ //
        (ArrowDataType::Int8, LogicalType::Primitive { .. }) => {
            let arr = array
                .as_any()
                .downcast_ref::<Int8Array>()
                .ok_or_else(|| type_err("Int8"))?;
            Ok(LogicalColumn::Primitive(ColumnData::I8(
                arr.values().to_vec(),
            )))
        }
        (ArrowDataType::Int16, LogicalType::Primitive { .. }) => {
            let arr = array
                .as_any()
                .downcast_ref::<Int16Array>()
                .ok_or_else(|| type_err("Int16"))?;
            Ok(LogicalColumn::Primitive(ColumnData::I16(
                arr.values().to_vec(),
            )))
        }
        (ArrowDataType::Int32, LogicalType::Primitive { .. }) => {
            let arr = array
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| type_err("Int32"))?;
            Ok(LogicalColumn::Primitive(ColumnData::I32(
                arr.values().to_vec(),
            )))
        }
        (ArrowDataType::Int64, LogicalType::Primitive { .. }) => {
            let arr = array
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| type_err("Int64"))?;
            Ok(LogicalColumn::Primitive(ColumnData::I64(
                arr.values().to_vec(),
            )))
        }
        (ArrowDataType::UInt8, LogicalType::Primitive { .. }) => {
            let arr = array
                .as_any()
                .downcast_ref::<UInt8Array>()
                .ok_or_else(|| type_err("UInt8"))?;
            Ok(LogicalColumn::Primitive(ColumnData::U8(
                arr.values().to_vec(),
            )))
        }
        (ArrowDataType::UInt16, LogicalType::Primitive { .. }) => {
            let arr = array
                .as_any()
                .downcast_ref::<UInt16Array>()
                .ok_or_else(|| type_err("UInt16"))?;
            Ok(LogicalColumn::Primitive(ColumnData::U16(
                arr.values().to_vec(),
            )))
        }
        (ArrowDataType::UInt32, LogicalType::Primitive { .. }) => {
            let arr = array
                .as_any()
                .downcast_ref::<UInt32Array>()
                .ok_or_else(|| type_err("UInt32"))?;
            Ok(LogicalColumn::Primitive(ColumnData::U32(
                arr.values().to_vec(),
            )))
        }
        (ArrowDataType::UInt64, LogicalType::Primitive { .. }) => {
            let arr = array
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| type_err("UInt64"))?;
            Ok(LogicalColumn::Primitive(ColumnData::U64(
                arr.values().to_vec(),
            )))
        }
        (ArrowDataType::Float32, LogicalType::Primitive { .. }) => {
            let arr = array
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| type_err("Float32"))?;
            Ok(LogicalColumn::Primitive(ColumnData::F32(
                arr.values().to_vec(),
            )))
        }
        (ArrowDataType::Float64, LogicalType::Primitive { .. }) => {
            let arr = array
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| type_err("Float64"))?;
            Ok(LogicalColumn::Primitive(ColumnData::F64(
                arr.values().to_vec(),
            )))
        }

        // ------------------------------------------------------------------ //
        // Utf8 and Binary                                                     //
        // ------------------------------------------------------------------ //
        (ArrowDataType::Utf8, LogicalType::Utf8) => {
            let arr = array
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| type_err("Utf8"))?;
            let strings: Vec<String> = (0..arr.len()).map(|i| arr.value(i).to_string()).collect();
            Ok(LogicalColumn::Utf8(strings))
        }
        (ArrowDataType::Binary, LogicalType::Binary) => {
            let arr = array
                .as_any()
                .downcast_ref::<BinaryArray>()
                .ok_or_else(|| type_err("Binary"))?;
            let blobs: Vec<Vec<u8>> = (0..arr.len()).map(|i| arr.value(i).to_vec()).collect();
            Ok(LogicalColumn::Binary(blobs))
        }

        // ------------------------------------------------------------------ //
        // List and legacy flat ArrayOf / ArrayOfUtf8                                  //
        // ------------------------------------------------------------------ //
        (ArrowDataType::List(_), LogicalType::List { inner }) => from_arrow_list(array, inner),
        (ArrowDataType::List(_), LogicalType::ArrayOf { data_type }) => {
            let inner_lt = LogicalType::Primitive {
                data_type: *data_type,
            };
            let result = from_arrow_list(array, &inner_lt)?;
            // Return as recursive List (per spec: inverse always returns the recursive form)
            Ok(result)
        }
        (ArrowDataType::List(_), LogicalType::ArrayOfUtf8) => {
            let inner_lt = LogicalType::Utf8;
            from_arrow_list(array, &inner_lt)
        }

        // ------------------------------------------------------------------ //
        // Map                                                                 //
        // ------------------------------------------------------------------ //
        (
            ArrowDataType::Map(_, _),
            LogicalType::Map {
                key: key_lt,
                value: val_lt,
            },
        ) => from_arrow_map(array, key_lt, val_lt),

        // ------------------------------------------------------------------ //
        // Struct                                                              //
        // ------------------------------------------------------------------ //
        (
            ArrowDataType::Struct(_),
            LogicalType::Struct {
                fields: spec_fields,
            },
        ) => from_arrow_struct(array, spec_fields),

        // ------------------------------------------------------------------ //
        // Union (dense)                                                       //
        // ------------------------------------------------------------------ //
        (ArrowDataType::Union(_, UnionMode::Dense), LogicalType::Union { variants }) => {
            from_arrow_union(array, variants)
        }

        // ------------------------------------------------------------------ //
        // Dictionary{inner} — recursive                                   //
        // ------------------------------------------------------------------ //
        (ArrowDataType::Dictionary(_, _), LogicalType::Dictionary { inner }) => {
            from_arrow_dict(array, inner)
        }

        // ------------------------------------------------------------------ //
        // Semantic types                                                       //
        // ------------------------------------------------------------------ //
        (ArrowDataType::Decimal128(_, _), LogicalType::Decimal128 { .. }) => {
            use arrow::array::Decimal128Array;
            let arr = array
                .as_any()
                .downcast_ref::<Decimal128Array>()
                .ok_or_else(|| type_err("Decimal128"))?;
            let values: Vec<i128> = (0..arr.len()).map(|i| arr.value(i)).collect();
            Ok(LogicalColumn::Decimal128 { values })
        }

        (
            ArrowDataType::Date32,
            LogicalType::Date {
                unit: crate::core::schema::DateUnit::Days,
            },
        ) => {
            use arrow::array::Date32Array;
            let arr = array
                .as_any()
                .downcast_ref::<Date32Array>()
                .ok_or_else(|| type_err("Date32"))?;
            let values: Vec<i32> = arr.values().to_vec();
            Ok(LogicalColumn::Date32 { values })
        }

        (
            ArrowDataType::Date64,
            LogicalType::Date {
                unit: crate::core::schema::DateUnit::Millis,
            },
        ) => {
            use arrow::array::Date64Array;
            let arr = array
                .as_any()
                .downcast_ref::<Date64Array>()
                .ok_or_else(|| type_err("Date64"))?;
            let values: Vec<i64> = arr.values().to_vec();
            Ok(LogicalColumn::Date64 { values })
        }

        (ArrowDataType::Timestamp(_, _), LogicalType::Datetime { unit, .. }) => {
            use crate::core::schema::TimeUnit;
            use arrow::array::{
                TimestampMicrosecondArray, TimestampMillisecondArray, TimestampNanosecondArray,
                TimestampSecondArray,
            };
            let values: Vec<i64> = match unit {
                TimeUnit::Seconds => array
                    .as_any()
                    .downcast_ref::<TimestampSecondArray>()
                    .ok_or_else(|| type_err("TimestampSecond"))?
                    .values()
                    .to_vec(),
                TimeUnit::Millis => array
                    .as_any()
                    .downcast_ref::<TimestampMillisecondArray>()
                    .ok_or_else(|| type_err("TimestampMillisecond"))?
                    .values()
                    .to_vec(),
                TimeUnit::Micros => array
                    .as_any()
                    .downcast_ref::<TimestampMicrosecondArray>()
                    .ok_or_else(|| type_err("TimestampMicrosecond"))?
                    .values()
                    .to_vec(),
                TimeUnit::Nanos => array
                    .as_any()
                    .downcast_ref::<TimestampNanosecondArray>()
                    .ok_or_else(|| type_err("TimestampNanosecond"))?
                    .values()
                    .to_vec(),
            };
            Ok(LogicalColumn::Datetime { values })
        }

        // ------------------------------------------------------------------ //
        // Type mismatch                                                       //
        // ------------------------------------------------------------------ //
        _ => Err(HeliumError::Format(format!(
            "from_arrow_array: Arrow type {} does not match Helium hint {hint:?}",
            array.data_type()
        ))),
    }
}

// ---------------------------------------------------------------------------
// Nullable — extract present bitmap, compact inner values
// ---------------------------------------------------------------------------

fn from_arrow_nullable(array: &ArrayRef, inner_lt: &LogicalType) -> Result<LogicalColumn> {
    let n = array.len();
    let present: Vec<bool> = (0..n).map(|i| array.is_valid(i)).collect();
    let null_count = array.null_count();

    // Build compact inner column: only the valid rows.
    if null_count == 0 {
        // No nulls: all rows present, inner column = full column
        let inner_col = from_arrow_inner(array, inner_lt)?;
        return Ok(LogicalColumn::Nullable {
            present,
            value: Box::new(inner_col),
        });
    }

    // Extract only valid rows to form compact inner.
    let compact = extract_valid_rows(array, inner_lt, &present)?;
    Ok(LogicalColumn::Nullable {
        present,
        value: Box::new(compact),
    })
}

/// Extract only the rows where `present[i] == true` from an Arrow array,
/// returning a compact LogicalColumn.
fn extract_valid_rows(
    array: &ArrayRef,
    lt: &LogicalType,
    present: &[bool],
) -> Result<LogicalColumn> {
    match (array.data_type(), lt) {
        (ArrowDataType::Int8, _) => {
            let arr = array
                .as_any()
                .downcast_ref::<Int8Array>()
                .ok_or_else(|| type_err("Int8"))?;
            let v: Vec<i8> = present
                .iter()
                .enumerate()
                .filter_map(|(i, &p)| if p { Some(arr.value(i)) } else { None })
                .collect();
            Ok(LogicalColumn::Primitive(ColumnData::I8(v)))
        }
        (ArrowDataType::Int16, _) => {
            let arr = array
                .as_any()
                .downcast_ref::<Int16Array>()
                .ok_or_else(|| type_err("Int16"))?;
            let v: Vec<i16> = present
                .iter()
                .enumerate()
                .filter_map(|(i, &p)| if p { Some(arr.value(i)) } else { None })
                .collect();
            Ok(LogicalColumn::Primitive(ColumnData::I16(v)))
        }
        (ArrowDataType::Int32, _) => {
            let arr = array
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| type_err("Int32"))?;
            let v: Vec<i32> = present
                .iter()
                .enumerate()
                .filter_map(|(i, &p)| if p { Some(arr.value(i)) } else { None })
                .collect();
            Ok(LogicalColumn::Primitive(ColumnData::I32(v)))
        }
        (ArrowDataType::Int64, _) => {
            let arr = array
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| type_err("Int64"))?;
            let v: Vec<i64> = present
                .iter()
                .enumerate()
                .filter_map(|(i, &p)| if p { Some(arr.value(i)) } else { None })
                .collect();
            Ok(LogicalColumn::Primitive(ColumnData::I64(v)))
        }
        (ArrowDataType::UInt8, _) => {
            let arr = array
                .as_any()
                .downcast_ref::<UInt8Array>()
                .ok_or_else(|| type_err("UInt8"))?;
            let v: Vec<u8> = present
                .iter()
                .enumerate()
                .filter_map(|(i, &p)| if p { Some(arr.value(i)) } else { None })
                .collect();
            Ok(LogicalColumn::Primitive(ColumnData::U8(v)))
        }
        (ArrowDataType::UInt16, _) => {
            let arr = array
                .as_any()
                .downcast_ref::<UInt16Array>()
                .ok_or_else(|| type_err("UInt16"))?;
            let v: Vec<u16> = present
                .iter()
                .enumerate()
                .filter_map(|(i, &p)| if p { Some(arr.value(i)) } else { None })
                .collect();
            Ok(LogicalColumn::Primitive(ColumnData::U16(v)))
        }
        (ArrowDataType::UInt32, _) => {
            let arr = array
                .as_any()
                .downcast_ref::<UInt32Array>()
                .ok_or_else(|| type_err("UInt32"))?;
            let v: Vec<u32> = present
                .iter()
                .enumerate()
                .filter_map(|(i, &p)| if p { Some(arr.value(i)) } else { None })
                .collect();
            Ok(LogicalColumn::Primitive(ColumnData::U32(v)))
        }
        (ArrowDataType::UInt64, _) => {
            let arr = array
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| type_err("UInt64"))?;
            let v: Vec<u64> = present
                .iter()
                .enumerate()
                .filter_map(|(i, &p)| if p { Some(arr.value(i)) } else { None })
                .collect();
            Ok(LogicalColumn::Primitive(ColumnData::U64(v)))
        }
        (ArrowDataType::Float32, _) => {
            let arr = array
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| type_err("Float32"))?;
            let v: Vec<f32> = present
                .iter()
                .enumerate()
                .filter_map(|(i, &p)| if p { Some(arr.value(i)) } else { None })
                .collect();
            Ok(LogicalColumn::Primitive(ColumnData::F32(v)))
        }
        (ArrowDataType::Float64, _) => {
            let arr = array
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| type_err("Float64"))?;
            let v: Vec<f64> = present
                .iter()
                .enumerate()
                .filter_map(|(i, &p)| if p { Some(arr.value(i)) } else { None })
                .collect();
            Ok(LogicalColumn::Primitive(ColumnData::F64(v)))
        }
        (ArrowDataType::Utf8, _) => {
            let arr = array
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| type_err("Utf8"))?;
            let v: Vec<String> = present
                .iter()
                .enumerate()
                .filter_map(|(i, &p)| {
                    if p {
                        Some(arr.value(i).to_string())
                    } else {
                        None
                    }
                })
                .collect();
            Ok(LogicalColumn::Utf8(v))
        }
        (ArrowDataType::Binary, _) => {
            let arr = array
                .as_any()
                .downcast_ref::<BinaryArray>()
                .ok_or_else(|| type_err("Binary"))?;
            let v: Vec<Vec<u8>> = present
                .iter()
                .enumerate()
                .filter_map(|(i, &p)| if p { Some(arr.value(i).to_vec()) } else { None })
                .collect();
            Ok(LogicalColumn::Binary(v))
        }
        _ => {
            // For complex types, we need to filter rows. Use a simple index-based approach.
            // Build a new array from valid rows only, then convert.
            // This is less efficient but correct for nested types.
            let valid_indices: Vec<usize> = present
                .iter()
                .enumerate()
                .filter_map(|(i, &p)| if p { Some(i) } else { None })
                .collect();
            let compact_arr = arrow::compute::take(
                array.as_ref(),
                &UInt32Array::from(valid_indices.iter().map(|&i| i as u32).collect::<Vec<_>>()),
                None,
            )
            .map_err(|e| HeliumError::Format(format!("arrow::compute::take: {e}")))?;
            from_arrow_inner(&compact_arr, lt)
        }
    }
}

// ---------------------------------------------------------------------------
// List
// ---------------------------------------------------------------------------

fn from_arrow_list(array: &ArrayRef, inner_lt: &LogicalType) -> Result<LogicalColumn> {
    let arr = array
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or_else(|| type_err("ListArray"))?;

    // Arrow offsets are i32; convert to u32
    let offsets_i32 = arr.offsets();
    let offsets_u32: Vec<u32> = offsets_i32
        .iter()
        .map(|&o| {
            u32::try_from(o).map_err(|_| {
                HeliumError::Format(format!("Arrow list offset {o} does not fit in u32"))
            })
        })
        .collect::<Result<_>>()?;

    let values_arr = arr.values();
    let inner_col = from_arrow_inner(values_arr, inner_lt)?;

    Ok(LogicalColumn::List {
        offsets: offsets_u32,
        values: Box::new(inner_col),
    })
}

// ---------------------------------------------------------------------------
// Map
// ---------------------------------------------------------------------------

fn from_arrow_map(
    array: &ArrayRef,
    key_lt: &LogicalType,
    val_lt: &LogicalType,
) -> Result<LogicalColumn> {
    let arr = array
        .as_any()
        .downcast_ref::<MapArray>()
        .ok_or_else(|| type_err("MapArray"))?;

    let offsets_i32 = arr.offsets();
    let offsets_u32: Vec<u32> = offsets_i32
        .iter()
        .map(|&o| {
            u32::try_from(o).map_err(|_| {
                HeliumError::Format(format!("Arrow map offset {o} does not fit in u32"))
            })
        })
        .collect::<Result<_>>()?;

    let entries = arr.entries();
    // entries is a StructArray with 2 fields: key and value
    let key_col_ref = entries.column(0);
    let val_col_ref = entries.column(1);

    let keys_col = from_arrow_inner(key_col_ref, key_lt)?;
    let vals_col = from_arrow_inner(val_col_ref, val_lt)?;

    Ok(LogicalColumn::Map {
        offsets: offsets_u32,
        keys: Box::new(keys_col),
        values: Box::new(vals_col),
    })
}

// ---------------------------------------------------------------------------
// Struct
// ---------------------------------------------------------------------------

fn from_arrow_struct(array: &ArrayRef, spec_fields: &[FieldSpec]) -> Result<LogicalColumn> {
    let arr = array
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| type_err("StructArray"))?;

    if arr.num_columns() != spec_fields.len() {
        return Err(HeliumError::Format(format!(
            "from_arrow_struct: expected {} fields but got {}",
            spec_fields.len(),
            arr.num_columns()
        )));
    }

    let mut fields = Vec::with_capacity(spec_fields.len());
    for (i, spec) in spec_fields.iter().enumerate() {
        let col_arr = arr.column(i);
        let col = from_arrow_inner(col_arr, &spec.logical_type)?;
        fields.push((spec.name.clone(), col));
    }

    Ok(LogicalColumn::Struct { fields })
}

// ---------------------------------------------------------------------------
// Union (dense)
// ---------------------------------------------------------------------------

fn from_arrow_union(
    array: &ArrayRef,
    spec_variants: &[(String, LogicalType)],
) -> Result<LogicalColumn> {
    let arr = array
        .as_any()
        .downcast_ref::<UnionArray>()
        .ok_or_else(|| type_err("UnionArray"))?;

    let n_variants = spec_variants.len();

    // Extract tags
    let type_ids = arr.type_ids();
    let tags: Vec<u8> = type_ids
        .iter()
        .map(|&t| {
            u8::try_from(t)
                .map_err(|_| HeliumError::Format(format!("Union type_id {t} doesn't fit in u8")))
        })
        .collect::<Result<_>>()?;

    // Each variant's child array in Arrow is already compacted (dense union)
    let mut variants = Vec::with_capacity(n_variants);
    for (v_idx, (v_name, v_lt)) in spec_variants.iter().enumerate() {
        let child = arr.child(v_idx as i8);
        let col = from_arrow_inner(child, v_lt)?;
        variants.push((v_name.clone(), col));
    }

    Ok(LogicalColumn::Union { tags, variants })
}

// ---------------------------------------------------------------------------
// Dictionary{inner} — recursive
// ---------------------------------------------------------------------------

/// Convert an Arrow `DictionaryArray<UInt32>` to a Helium recursive
/// [`LogicalColumn::Dictionary`].
///
/// The Arrow `values()` buffer becomes the Helium `dictionary` (converted
/// via [`from_arrow_inner`] with the `inner` hint), and `keys()` become
/// the `indices`.  If the Arrow key type is not UInt32 this function
/// returns a clear `HeliumError::Format("unsupported")` rather than a wrong
/// mapping.
fn from_arrow_dict(array: &ArrayRef, inner: &LogicalType) -> Result<LogicalColumn> {
    let arr = array
        .as_any()
        .downcast_ref::<DictionaryArray<UInt32Type>>()
        .ok_or_else(|| {
            HeliumError::Format(
                "Dictionary{inner}: unsupported Arrow dict key type — only UInt32 keys supported"
                    .into(),
            )
        })?;

    let indices: Vec<u32> = arr.keys().values().to_vec();
    let dict_array: ArrayRef = arr.values().clone();
    let dictionary = from_arrow_inner(&dict_array, inner)?;

    Ok(LogicalColumn::Dictionary {
        dictionary: Box::new(dictionary),
        indices,
    })
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn type_err(expected: &str) -> HeliumError {
    HeliumError::Format(format!(
        "from_arrow: expected {expected} but got a different array type"
    ))
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arrow::to_arrow::to_arrow_array;
    use crate::core::coder::DataType as HDataType;
    use crate::core::schema::FieldSpec;

    fn roundtrip(col: LogicalColumn, lt: LogicalType) -> LogicalColumn {
        let arr = to_arrow_array(&col, &lt).unwrap();
        from_arrow_array(&arr, &lt).unwrap()
    }

    // ---------------------------------------------------------------------- //
    // 1. Primitive types — all numeric DataTypes                             //
    // ---------------------------------------------------------------------- //

    #[test]
    fn rt_primitive_i8() {
        let col = LogicalColumn::Primitive(ColumnData::I8(vec![-1, 0, 127]));
        let lt = LogicalType::Primitive {
            data_type: HDataType::I8,
        };
        assert_eq!(roundtrip(col.clone(), lt), col);
    }

    #[test]
    fn rt_primitive_i16() {
        let col = LogicalColumn::Primitive(ColumnData::I16(vec![-300, 0, 300]));
        let lt = LogicalType::Primitive {
            data_type: HDataType::I16,
        };
        assert_eq!(roundtrip(col.clone(), lt), col);
    }

    #[test]
    fn rt_primitive_i32() {
        let col = LogicalColumn::Primitive(ColumnData::I32(vec![i32::MIN, 0, i32::MAX]));
        let lt = LogicalType::Primitive {
            data_type: HDataType::I32,
        };
        assert_eq!(roundtrip(col.clone(), lt), col);
    }

    #[test]
    fn rt_primitive_i64() {
        let col = LogicalColumn::Primitive(ColumnData::I64(vec![1, 2, 3]));
        let lt = LogicalType::Primitive {
            data_type: HDataType::I64,
        };
        assert_eq!(
            roundtrip(col, lt),
            LogicalColumn::Primitive(ColumnData::I64(vec![1, 2, 3]))
        );
    }

    #[test]
    fn rt_primitive_u8() {
        let col = LogicalColumn::Primitive(ColumnData::U8(vec![0, 128, 255]));
        let lt = LogicalType::Primitive {
            data_type: HDataType::U8,
        };
        assert_eq!(roundtrip(col.clone(), lt), col);
    }

    #[test]
    fn rt_primitive_u16() {
        let col = LogicalColumn::Primitive(ColumnData::U16(vec![0, 1000, 65535]));
        let lt = LogicalType::Primitive {
            data_type: HDataType::U16,
        };
        assert_eq!(roundtrip(col.clone(), lt), col);
    }

    #[test]
    fn rt_primitive_u32() {
        let col = LogicalColumn::Primitive(ColumnData::U32(vec![0, u32::MAX / 2, u32::MAX]));
        let lt = LogicalType::Primitive {
            data_type: HDataType::U32,
        };
        assert_eq!(roundtrip(col.clone(), lt), col);
    }

    #[test]
    fn rt_primitive_u64() {
        let col = LogicalColumn::Primitive(ColumnData::U64(vec![0, 1_000_000_000_000]));
        let lt = LogicalType::Primitive {
            data_type: HDataType::U64,
        };
        assert_eq!(roundtrip(col.clone(), lt), col);
    }

    #[test]
    fn rt_primitive_f32() {
        let col = LogicalColumn::Primitive(ColumnData::F32(vec![0.0, 1.5, -2.5]));
        let lt = LogicalType::Primitive {
            data_type: HDataType::F32,
        };
        assert_eq!(roundtrip(col.clone(), lt), col);
    }

    #[test]
    fn rt_primitive_f64() {
        let col = LogicalColumn::Primitive(ColumnData::F64(vec![0.0, 1.234_567_89]));
        let lt = LogicalType::Primitive {
            data_type: HDataType::F64,
        };
        assert_eq!(roundtrip(col.clone(), lt), col);
    }

    // ---------------------------------------------------------------------- //
    // 2. Utf8 and Binary                                                      //
    // ---------------------------------------------------------------------- //

    #[test]
    fn rt_utf8() {
        let col = LogicalColumn::Utf8(vec!["hello".into(), "world".into(), "".into()]);
        let lt = LogicalType::Utf8;
        assert_eq!(roundtrip(col.clone(), lt), col);
    }

    #[test]
    fn rt_binary() {
        let col = LogicalColumn::Binary(vec![vec![1, 2, 3], vec![], vec![255]]);
        let lt = LogicalType::Binary;
        assert_eq!(roundtrip(col.clone(), lt), col);
    }

    // ---------------------------------------------------------------------- //
    // 3. Nullable<Primitive/Utf8/Binary> — with and without nulls            //
    // ---------------------------------------------------------------------- //

    #[test]
    fn rt_nullable_i32_with_nulls() {
        let col = LogicalColumn::Nullable {
            present: vec![true, false, true, false, true],
            value: Box::new(LogicalColumn::Primitive(ColumnData::I32(vec![10, 30, 50]))),
        };
        let lt = LogicalType::Nullable {
            inner: Box::new(LogicalType::Primitive {
                data_type: HDataType::I32,
            }),
        };
        assert_eq!(roundtrip(col.clone(), lt), col);
    }

    #[test]
    fn rt_nullable_all_present() {
        let col = LogicalColumn::Nullable {
            present: vec![true, true, true],
            value: Box::new(LogicalColumn::Primitive(ColumnData::I64(vec![1, 2, 3]))),
        };
        let lt = LogicalType::Nullable {
            inner: Box::new(LogicalType::Primitive {
                data_type: HDataType::I64,
            }),
        };
        assert_eq!(roundtrip(col.clone(), lt), col);
    }

    #[test]
    fn rt_nullable_utf8_with_nulls() {
        let col = LogicalColumn::Nullable {
            present: vec![true, false, true],
            value: Box::new(LogicalColumn::Utf8(vec!["hello".into(), "world".into()])),
        };
        let lt = LogicalType::Nullable {
            inner: Box::new(LogicalType::Utf8),
        };
        assert_eq!(roundtrip(col.clone(), lt), col);
    }

    #[test]
    fn rt_nullable_binary_with_nulls() {
        let col = LogicalColumn::Nullable {
            present: vec![false, true, true],
            value: Box::new(LogicalColumn::Binary(vec![vec![1, 2], vec![3, 4, 5]])),
        };
        let lt = LogicalType::Nullable {
            inner: Box::new(LogicalType::Binary),
        };
        assert_eq!(roundtrip(col.clone(), lt), col);
    }

    // ---------------------------------------------------------------------- //
    // 4. List<Primitive>, List<Utf8> with empty rows                         //
    // ---------------------------------------------------------------------- //

    #[test]
    fn rt_list_primitive_with_empty_row() {
        let col = LogicalColumn::List {
            offsets: vec![0, 2, 2, 3],
            values: Box::new(LogicalColumn::Primitive(ColumnData::I32(vec![1, 2, 3]))),
        };
        let lt = LogicalType::List {
            inner: Box::new(LogicalType::Primitive {
                data_type: HDataType::I32,
            }),
        };
        assert_eq!(roundtrip(col.clone(), lt), col);
    }

    #[test]
    fn rt_list_utf8_with_empty() {
        let col = LogicalColumn::List {
            offsets: vec![0, 2, 2],
            values: Box::new(LogicalColumn::Utf8(vec!["a".into(), "b".into()])),
        };
        let lt = LogicalType::List {
            inner: Box::new(LogicalType::Utf8),
        };
        assert_eq!(roundtrip(col.clone(), lt), col);
    }

    // ---------------------------------------------------------------------- //
    // 5. Map<Utf8, Primitive>                                                 //
    // ---------------------------------------------------------------------- //

    #[test]
    fn rt_map_utf8_to_i64() {
        let col = LogicalColumn::Map {
            offsets: vec![0, 2, 3],
            keys: Box::new(LogicalColumn::Utf8(vec![
                "a".into(),
                "b".into(),
                "c".into(),
            ])),
            values: Box::new(LogicalColumn::Primitive(ColumnData::I64(vec![10, 20, 30]))),
        };
        let lt = LogicalType::Map {
            key: Box::new(LogicalType::Utf8),
            value: Box::new(LogicalType::Primitive {
                data_type: HDataType::I64,
            }),
        };
        assert_eq!(roundtrip(col.clone(), lt), col);
    }

    // ---------------------------------------------------------------------- //
    // 6. Struct with primitive + utf8 fields                                  //
    // ---------------------------------------------------------------------- //

    #[test]
    fn rt_struct_primitive_and_utf8() {
        let col = LogicalColumn::Struct {
            fields: vec![
                (
                    "id".into(),
                    LogicalColumn::Primitive(ColumnData::I32(vec![1, 2, 3])),
                ),
                (
                    "name".into(),
                    LogicalColumn::Utf8(vec!["a".into(), "b".into(), "c".into()]),
                ),
            ],
        };
        let lt = LogicalType::Struct {
            fields: vec![
                FieldSpec::primitive("id", HDataType::I32, vec![]),
                FieldSpec::utf8("name", vec![], vec![]),
            ],
        };
        assert_eq!(roundtrip(col.clone(), lt), col);
    }

    // ---------------------------------------------------------------------- //
    // 7. Union with two variants                                              //
    // ---------------------------------------------------------------------- //

    #[test]
    fn rt_union_two_variants() {
        let col = LogicalColumn::Union {
            tags: vec![0, 1, 0, 1],
            variants: vec![
                (
                    "int_val".into(),
                    LogicalColumn::Primitive(ColumnData::I32(vec![10, 30])),
                ),
                (
                    "str_val".into(),
                    LogicalColumn::Utf8(vec!["x".into(), "y".into()]),
                ),
            ],
        };
        let lt = LogicalType::Union {
            variants: vec![
                (
                    "int_val".into(),
                    LogicalType::Primitive {
                        data_type: HDataType::I32,
                    },
                ),
                ("str_val".into(), LogicalType::Utf8),
            ],
        };
        let back = roundtrip(col.clone(), lt);
        // Compare tags and variant counts
        let LogicalColumn::Union {
            tags: back_tags,
            variants: back_vars,
        } = back
        else {
            panic!("expected Union");
        };
        let LogicalColumn::Union {
            tags: orig_tags,
            variants: orig_vars,
        } = col
        else {
            panic!("expected Union");
        };
        assert_eq!(back_tags, orig_tags);
        assert_eq!(back_vars.len(), orig_vars.len());
        for ((bn, bc), (on, oc)) in back_vars.iter().zip(orig_vars.iter()) {
            assert_eq!(bn, on);
            assert_eq!(bc, oc);
        }
    }

    // ---------------------------------------------------------------------- //
    // 8. Dict types                                                           //
    // ---------------------------------------------------------------------- //

    #[test]
    fn rt_dict_utf8() {
        let col = LogicalColumn::Dictionary {
            dictionary: Box::new(LogicalColumn::Utf8(vec!["apple".into(), "banana".into()])),
            indices: vec![0, 1, 0],
        };
        let lt = LogicalType::Dictionary {
            inner: Box::new(LogicalType::Utf8),
        };
        assert_eq!(roundtrip(col.clone(), lt), col);
    }

    #[test]
    fn rt_dict_prim() {
        let col = LogicalColumn::Dictionary {
            dictionary: Box::new(LogicalColumn::Primitive(ColumnData::I32(vec![10, 20, 30]))),
            indices: vec![2, 0, 1],
        };
        let lt = LogicalType::Dictionary {
            inner: Box::new(LogicalType::Primitive {
                data_type: HDataType::I32,
            }),
        };
        assert_eq!(roundtrip(col.clone(), lt), col);
    }

    // ---------------------------------------------------------------------- //
    // 9. Legacy flat variants roundtrip → recursive shape                     //
    // ---------------------------------------------------------------------- //

    #[test]
    fn legacy_nullable_prim_to_arrow_then_back_recursive() {
        // NullablePrim is a legacy flat variant. to_arrow converts it to an Arrow
        // Int64Array with nulls. from_arrow with a Nullable hint returns the recursive form.
        let col = LogicalColumn::NullablePrim {
            present: vec![true, false, true],
            values: ColumnData::I64(vec![42, 99]),
        };
        let lt = LogicalType::NullablePrim {
            data_type: HDataType::I64,
        };
        let arr = to_arrow_array(&col, &lt).unwrap();
        // Round-trip through Arrow with the Nullable<Primitive> recursive hint
        let recursive_lt = LogicalType::Nullable {
            inner: Box::new(LogicalType::Primitive {
                data_type: HDataType::I64,
            }),
        };
        let back = from_arrow_array(&arr, &recursive_lt).unwrap();
        let expected = LogicalColumn::Nullable {
            present: vec![true, false, true],
            value: Box::new(LogicalColumn::Primitive(ColumnData::I64(vec![42, 99]))),
        };
        assert_eq!(back, expected);
    }

    // ---------------------------------------------------------------------- //
    // 9b. Nullable<List<Struct>> — demanding composition                    //
    // ---------------------------------------------------------------------- //

    #[test]
    fn rt_nullable_list_struct() {
        // Nullable<List<Struct{id: I32, val: I64}>>
        // Row 0: [{ id=1, val=10 }, { id=2, val=20 }]
        // Row 1: null
        // Row 2: [{ id=3, val=30 }]
        let inner_struct_lt = LogicalType::Struct {
            fields: vec![
                FieldSpec::primitive("id", HDataType::I32, vec![]),
                FieldSpec::primitive("val", HDataType::I64, vec![]),
            ],
        };
        let list_lt = LogicalType::List {
            inner: Box::new(inner_struct_lt.clone()),
        };
        let lt = LogicalType::Nullable {
            inner: Box::new(list_lt.clone()),
        };

        // compact inner: rows 0 and 2 (row 1 is null)
        let compact_inner_list = LogicalColumn::List {
            offsets: vec![0, 2, 3],
            values: Box::new(LogicalColumn::Struct {
                fields: vec![
                    (
                        "id".into(),
                        LogicalColumn::Primitive(ColumnData::I32(vec![1, 2, 3])),
                    ),
                    (
                        "val".into(),
                        LogicalColumn::Primitive(ColumnData::I64(vec![10, 20, 30])),
                    ),
                ],
            }),
        };
        let col = LogicalColumn::Nullable {
            present: vec![true, false, true],
            value: Box::new(compact_inner_list),
        };

        let arr = to_arrow_array(&col, &lt).unwrap();
        assert_eq!(arr.len(), 3);
        assert_eq!(arr.null_count(), 1);
        let back = from_arrow_array(&arr, &lt).unwrap();
        assert_eq!(back, col);
    }

    #[test]
    fn legacy_array_of_to_arrow_then_back_recursive_list() {
        // ArrayOf is a legacy flat variant mapping to Arrow List<Int32>.
        // Back-conversion with the List hint returns the recursive form.
        let col = LogicalColumn::ArrayOf {
            offsets: vec![0, 2, 3],
            values: ColumnData::I32(vec![1, 2, 3]),
        };
        let lt = LogicalType::ArrayOf {
            data_type: HDataType::I32,
        };
        let arr = to_arrow_array(&col, &lt).unwrap();
        let recursive_lt = LogicalType::List {
            inner: Box::new(LogicalType::Primitive {
                data_type: HDataType::I32,
            }),
        };
        let back = from_arrow_array(&arr, &recursive_lt).unwrap();
        let expected = LogicalColumn::List {
            offsets: vec![0, 2, 3],
            values: Box::new(LogicalColumn::Primitive(ColumnData::I32(vec![1, 2, 3]))),
        };
        assert_eq!(back, expected);
    }
}
