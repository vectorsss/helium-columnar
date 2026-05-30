//! Schema ↔ Arrow schema conversion.
//!
//! Converts between Helium [`Schema`] and Arrow [`arrow::datatypes::Schema`].
//!
//! # Inverse direction note
//!
//! `schema_from_arrow` fills in **default encodings** for each leaf using
//! [`crate::schema::encodings::default_encodings`]. The bridge does not
//! optimize encodings — that is the optimizer's job.

use std::sync::Arc;

use arrow::datatypes::{
    DataType as ArrowDataType, Field as ArrowField, Fields, Schema as ArrowSchema,
    TimeUnit as ArrowTimeUnit, UnionFields, UnionMode,
};

use crate::core::coder::DataType as HeliumDataType;
use crate::core::error::{HeliumError, Result};
use crate::core::schema::{ColumnSpec, DateUnit, FieldSpec, LogicalType, Schema, TimeUnit};
use crate::schema::encodings::default_encodings;

/// Convert a Helium [`Schema`] to an Arrow [`ArrowSchema`].
///
/// Each top-level column becomes an Arrow `Field`. The `nullable` flag is
/// `true` if the `LogicalType` is `Nullable`; otherwise `false`. Inside
/// Struct fields the same rule applies recursively.
pub fn schema_to_arrow(schema: &Schema) -> ArrowSchema {
    let fields: Vec<ArrowField> = schema
        .columns
        .iter()
        .map(|col| {
            let dt = logical_type_to_arrow(&col.logical_type);
            let nullable = matches!(
                col.logical_type,
                LogicalType::Nullable { .. }
                    | LogicalType::NullablePrim { .. }
                    | LogicalType::NullableUtf8
                    | LogicalType::NullableBinary
            );
            ArrowField::new(col.name.clone(), dt, nullable)
        })
        .collect();
    ArrowSchema::new(fields)
}

/// Convert an Arrow [`ArrowSchema`] to a Helium [`Schema`].
///
/// Encodings are filled with defaults via [`default_encodings`]. No
/// optimization is performed.
///
/// Returns `HeliumError::Schema` if an Arrow DataType has no Helium
/// equivalent (e.g., `Decimal128`, `Date32`).
pub fn schema_from_arrow(arrow_schema: &ArrowSchema) -> Result<Schema> {
    let columns: Result<Vec<ColumnSpec>> = arrow_schema
        .fields()
        .iter()
        .map(|field| arrow_field_to_column_spec(field))
        .collect();
    Ok(Schema::new(columns?))
}

// ---------------------------------------------------------------------------
// Helium → Arrow DataType
// ---------------------------------------------------------------------------

/// Convert a Helium `LogicalType` to an Arrow `DataType`.
pub fn logical_type_to_arrow(lt: &LogicalType) -> ArrowDataType {
    match lt {
        LogicalType::Primitive { data_type } => helium_data_type_to_arrow(*data_type),
        LogicalType::Utf8 => ArrowDataType::Utf8,
        LogicalType::Binary => ArrowDataType::Binary,

        // v2 variants mapped to v3 equivalents
        LogicalType::ArrayOf { data_type } => {
            let inner_dt = helium_data_type_to_arrow(*data_type);
            ArrowDataType::List(Arc::new(ArrowField::new("item", inner_dt, true)))
        }
        LogicalType::ArrayOfUtf8 => {
            ArrowDataType::List(Arc::new(ArrowField::new("item", ArrowDataType::Utf8, true)))
        }
        LogicalType::NullablePrim { data_type } => helium_data_type_to_arrow(*data_type),
        LogicalType::NullableUtf8 => ArrowDataType::Utf8,
        LogicalType::NullableBinary => ArrowDataType::Binary,

        // v3 Dictionary{inner}
        LogicalType::Dictionary { inner } => ArrowDataType::Dictionary(
            Box::new(ArrowDataType::UInt32),
            Box::new(logical_type_to_arrow(inner)),
        ),

        // v3 types
        LogicalType::Struct { fields } => {
            let arrow_fields: Vec<ArrowField> = fields
                .iter()
                .map(|f| {
                    let dt = logical_type_to_arrow(&f.logical_type);
                    let nullable = matches!(
                        f.logical_type,
                        LogicalType::Nullable { .. }
                            | LogicalType::NullablePrim { .. }
                            | LogicalType::NullableUtf8
                            | LogicalType::NullableBinary
                    );
                    ArrowField::new(f.name.clone(), dt, nullable)
                })
                .collect();
            ArrowDataType::Struct(Fields::from(arrow_fields))
        }

        LogicalType::List { inner } => {
            let inner_dt = logical_type_to_arrow(inner);
            // Always mark list items nullable=true to match the Arrow convention
            // used by `to_arrow_array`. Arrow's ListArray always uses nullable=true
            // for the item field because null list elements are represented via the
            // null buffer, not a non-nullable item type.
            ArrowDataType::List(Arc::new(ArrowField::new("item", inner_dt, true)))
        }

        LogicalType::Map { key, value } => {
            let key_dt = logical_type_to_arrow(key);
            let val_dt = logical_type_to_arrow(value);
            let val_nullable = matches!(
                value.as_ref(),
                LogicalType::Nullable { .. }
                    | LogicalType::NullablePrim { .. }
                    | LogicalType::NullableUtf8
                    | LogicalType::NullableBinary
            );
            let struct_fields = Fields::from(vec![
                ArrowField::new("key", key_dt, false),
                ArrowField::new("value", val_dt, val_nullable),
            ]);
            let entries_field = Arc::new(ArrowField::new(
                "entries",
                ArrowDataType::Struct(struct_fields),
                false,
            ));
            ArrowDataType::Map(entries_field, false)
        }

        LogicalType::Nullable { inner } => {
            // The DataType itself doesn't change; nullability is a field property.
            logical_type_to_arrow(inner)
        }

        LogicalType::Union { variants } => {
            let union_fields: UnionFields = UnionFields::new(
                variants.iter().enumerate().map(|(i, _)| i as i8),
                variants.iter().map(|(name, v_lt)| {
                    let dt = logical_type_to_arrow(v_lt);
                    ArrowField::new(name.clone(), dt, true)
                }),
            );
            ArrowDataType::Union(union_fields, UnionMode::Dense)
        }

        // Semantic type extensions.
        // Arrow uses `u8` for precision and `i8` for scale; precision fits u8 (1..=38);
        // scale fits i8 (0..=precision ≤ 38 ≤ 127).
        LogicalType::Decimal128 { precision, scale } => {
            ArrowDataType::Decimal128(*precision, *scale as i8)
        }
        LogicalType::Date {
            unit: DateUnit::Days,
        } => ArrowDataType::Date32,
        LogicalType::Date {
            unit: DateUnit::Millis,
        } => ArrowDataType::Date64,
        LogicalType::Datetime { unit, timezone } => {
            let arrow_unit = helium_time_unit_to_arrow(*unit);
            let tz = timezone.as_deref().map(Arc::from);
            ArrowDataType::Timestamp(arrow_unit, tz)
        }
    }
}

fn helium_time_unit_to_arrow(unit: TimeUnit) -> ArrowTimeUnit {
    match unit {
        TimeUnit::Seconds => ArrowTimeUnit::Second,
        TimeUnit::Millis => ArrowTimeUnit::Millisecond,
        TimeUnit::Micros => ArrowTimeUnit::Microsecond,
        TimeUnit::Nanos => ArrowTimeUnit::Nanosecond,
    }
}

fn arrow_time_unit_to_helium(unit: &ArrowTimeUnit) -> TimeUnit {
    match unit {
        ArrowTimeUnit::Second => TimeUnit::Seconds,
        ArrowTimeUnit::Millisecond => TimeUnit::Millis,
        ArrowTimeUnit::Microsecond => TimeUnit::Micros,
        ArrowTimeUnit::Nanosecond => TimeUnit::Nanos,
    }
}

fn helium_data_type_to_arrow(dt: HeliumDataType) -> ArrowDataType {
    match dt {
        HeliumDataType::I8 => ArrowDataType::Int8,
        HeliumDataType::I16 => ArrowDataType::Int16,
        HeliumDataType::I32 => ArrowDataType::Int32,
        HeliumDataType::I64 => ArrowDataType::Int64,
        HeliumDataType::U8 => ArrowDataType::UInt8,
        HeliumDataType::U16 => ArrowDataType::UInt16,
        HeliumDataType::U32 => ArrowDataType::UInt32,
        HeliumDataType::U64 => ArrowDataType::UInt64,
        HeliumDataType::F32 => ArrowDataType::Float32,
        HeliumDataType::F64 => ArrowDataType::Float64,
        HeliumDataType::Bytes => ArrowDataType::Binary,
    }
}

// ---------------------------------------------------------------------------
// Arrow → Helium LogicalType
// ---------------------------------------------------------------------------

fn arrow_data_type_to_helium(dt: &ArrowDataType) -> Result<LogicalType> {
    match dt {
        ArrowDataType::Int8 => Ok(LogicalType::Primitive {
            data_type: HeliumDataType::I8,
        }),
        ArrowDataType::Int16 => Ok(LogicalType::Primitive {
            data_type: HeliumDataType::I16,
        }),
        ArrowDataType::Int32 => Ok(LogicalType::Primitive {
            data_type: HeliumDataType::I32,
        }),
        ArrowDataType::Int64 => Ok(LogicalType::Primitive {
            data_type: HeliumDataType::I64,
        }),
        ArrowDataType::UInt8 => Ok(LogicalType::Primitive {
            data_type: HeliumDataType::U8,
        }),
        ArrowDataType::UInt16 => Ok(LogicalType::Primitive {
            data_type: HeliumDataType::U16,
        }),
        ArrowDataType::UInt32 => Ok(LogicalType::Primitive {
            data_type: HeliumDataType::U32,
        }),
        ArrowDataType::UInt64 => Ok(LogicalType::Primitive {
            data_type: HeliumDataType::U64,
        }),
        ArrowDataType::Float32 => Ok(LogicalType::Primitive {
            data_type: HeliumDataType::F32,
        }),
        ArrowDataType::Float64 => Ok(LogicalType::Primitive {
            data_type: HeliumDataType::F64,
        }),
        ArrowDataType::Utf8 => Ok(LogicalType::Utf8),
        ArrowDataType::Binary => Ok(LogicalType::Binary),

        ArrowDataType::List(inner_field) => {
            // Arrow always marks list item fields as nullable=true by convention
            // (even when the Helium inner type is not Nullable). We use the
            // data-type recursion only and ignore the outer `nullable` flag here
            // — nullability of list elements is represented via Helium's
            // `Nullable { inner }` wrapper on the item data type, not on the
            // Arrow field's nullable flag.
            let inner_lt = arrow_data_type_to_helium(inner_field.data_type())?;
            Ok(LogicalType::List {
                inner: Box::new(inner_lt),
            })
        }

        ArrowDataType::Map(entries_field, _) => {
            // entries_field should be a Struct with 2 fields: key, value
            let ArrowDataType::Struct(struct_fields) = entries_field.data_type() else {
                return Err(HeliumError::Schema {
                    column: "<arrow_schema>".into(),
                    reason: "Arrow Map entries field must be a Struct".into(),
                });
            };
            if struct_fields.len() != 2 {
                return Err(HeliumError::Schema {
                    column: "<arrow_schema>".into(),
                    reason: "Arrow Map struct must have exactly 2 fields (key, value)".into(),
                });
            }
            let key_lt = arrow_data_type_to_helium(struct_fields[0].data_type())?;
            let val_lt = if struct_fields[1].is_nullable() {
                LogicalType::Nullable {
                    inner: Box::new(arrow_data_type_to_helium(struct_fields[1].data_type())?),
                }
            } else {
                arrow_data_type_to_helium(struct_fields[1].data_type())?
            };
            Ok(LogicalType::Map {
                key: Box::new(key_lt),
                value: Box::new(val_lt),
            })
        }

        ArrowDataType::Struct(fields) => {
            let spec_fields: Result<Vec<FieldSpec>> = fields
                .iter()
                .map(|f| arrow_field_to_field_spec(f))
                .collect();
            Ok(LogicalType::Struct {
                fields: spec_fields?,
            })
        }

        ArrowDataType::Union(union_fields, UnionMode::Dense) => {
            let variants: Result<Vec<(String, LogicalType)>> = union_fields
                .iter()
                .map(|(_, f)| {
                    let lt = arrow_data_type_to_helium(f.data_type())?;
                    Ok((f.name().to_string(), lt))
                })
                .collect();
            Ok(LogicalType::Union {
                variants: variants?,
            })
        }

        ArrowDataType::Dictionary(key_type, val_type) => {
            // We only support UInt32 index type
            if key_type.as_ref() != &ArrowDataType::UInt32 {
                return Err(HeliumError::Schema {
                    column: "<arrow_schema>".into(),
                    reason: format!(
                        "Arrow Dictionary key type {key_type} not supported; only UInt32 is"
                    ),
                });
            }
            // Produce v3 Dictionary{inner} for all value types.
            let inner_lt = arrow_data_type_to_helium(val_type.as_ref())?;
            Ok(LogicalType::Dictionary {
                inner: Box::new(inner_lt),
            })
        }

        // Semantic type extensions.
        ArrowDataType::Decimal128(precision, scale) => {
            // Arrow scale is i8 (can be negative for "super-integer" scaling);
            // Helium restricts to 0..=precision. Reject negative scale.
            if *scale < 0 {
                return Err(HeliumError::Schema {
                    column: "<arrow_schema>".into(),
                    reason: format!(
                        "Arrow Decimal128 scale {scale} is negative; \
                         Helium Decimal128 requires 0 ≤ scale ≤ precision"
                    ),
                });
            }
            Ok(LogicalType::Decimal128 {
                precision: *precision,
                scale: *scale as u8,
            })
        }

        ArrowDataType::Date32 => Ok(LogicalType::Date {
            unit: DateUnit::Days,
        }),
        ArrowDataType::Date64 => Ok(LogicalType::Date {
            unit: DateUnit::Millis,
        }),

        ArrowDataType::Timestamp(unit, tz) => {
            let helium_unit = arrow_time_unit_to_helium(unit);
            let timezone = tz.as_deref().map(str::to_owned);
            Ok(LogicalType::Datetime {
                unit: helium_unit,
                timezone,
            })
        }

        other => Err(HeliumError::Schema {
            column: "<arrow_schema>".into(),
            reason: format!(
                "Arrow DataType {other} has no Helium equivalent; \
                 supported types include all integers, floats, Utf8, Binary, \
                 Decimal128, Date32/64, Timestamp, List, Map, Struct, Union"
            ),
        }),
    }
}

fn arrow_field_to_column_spec(field: &ArrowField) -> Result<ColumnSpec> {
    let lt = if field.is_nullable() {
        LogicalType::Nullable {
            inner: Box::new(arrow_data_type_to_helium(field.data_type())?),
        }
    } else {
        arrow_data_type_to_helium(field.data_type())?
    };
    let encodings = default_encodings(&lt);
    Ok(ColumnSpec::new(field.name().clone(), lt, encodings))
}

fn arrow_field_to_field_spec(field: &ArrowField) -> Result<FieldSpec> {
    let lt = if field.is_nullable() {
        LogicalType::Nullable {
            inner: Box::new(arrow_data_type_to_helium(field.data_type())?),
        }
    } else {
        arrow_data_type_to_helium(field.data_type())?
    };
    let encodings = default_encodings(&lt);
    Ok(FieldSpec::new(field.name().clone(), lt, encodings))
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::schema::Schema;

    #[test]
    fn schema_to_arrow_basic() {
        let schema = Schema::new(vec![
            ColumnSpec::primitive("id", HeliumDataType::I64, vec![]),
            ColumnSpec::utf8("name", vec![], vec![]),
        ]);
        let arrow_schema = schema_to_arrow(&schema);
        assert_eq!(arrow_schema.fields().len(), 2);
        assert_eq!(arrow_schema.field(0).name(), "id");
        assert_eq!(arrow_schema.field(0).data_type(), &ArrowDataType::Int64);
        assert!(!arrow_schema.field(0).is_nullable());
        assert_eq!(arrow_schema.field(1).name(), "name");
        assert_eq!(arrow_schema.field(1).data_type(), &ArrowDataType::Utf8);
    }

    #[test]
    fn schema_from_arrow_basic() {
        let arrow_schema = ArrowSchema::new(vec![
            ArrowField::new("id", ArrowDataType::Int64, false),
            ArrowField::new("label", ArrowDataType::Utf8, true),
        ]);
        let schema = schema_from_arrow(&arrow_schema).unwrap();
        assert_eq!(schema.columns.len(), 2);
        assert_eq!(schema.columns[0].name, "id");
        assert!(matches!(
            schema.columns[0].logical_type,
            LogicalType::Primitive {
                data_type: HeliumDataType::I64
            }
        ));
        assert_eq!(schema.columns[1].name, "label");
        // nullable=true in Arrow → Nullable { inner: Utf8 } in Helium
        assert!(matches!(
            schema.columns[1].logical_type,
            LogicalType::Nullable { .. }
        ));
    }

    #[test]
    fn schema_roundtrip_logical_types() {
        // Build a 5-column schema with varied types
        let schema = Schema::new(vec![
            ColumnSpec::primitive("a", HeliumDataType::I32, vec![]),
            ColumnSpec::utf8("b", vec![], vec![]),
            ColumnSpec::binary("c", vec![], vec![]),
            ColumnSpec::nullable(
                "d",
                LogicalType::Primitive {
                    data_type: HeliumDataType::F64,
                },
                vec![vec![], vec![]],
            ),
            ColumnSpec::list(
                "e",
                LogicalType::Primitive {
                    data_type: HeliumDataType::U32,
                },
                vec![vec![], vec![]],
            ),
        ]);
        let arrow_schema = schema_to_arrow(&schema);
        let back = schema_from_arrow(&arrow_schema).unwrap();
        // Compare logical types only (encodings differ by design)
        for (orig, rebuilt) in schema.columns.iter().zip(back.columns.iter()) {
            assert_eq!(orig.name, rebuilt.name, "column name mismatch");
            assert_eq!(
                orig.logical_type, rebuilt.logical_type,
                "logical type mismatch for column '{}'",
                orig.name
            );
        }
    }
}
