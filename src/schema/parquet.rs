//! Parquet schema → Helium [`crate::Schema`] translator.
//!
//! # Feature flag: `parquet`
//!
//! ```toml
//! helium = { features = ["schema-parquet"] }
//! ```
//!
//! # Overview
//!
//! Reads the Parquet file schema from the **footer** (no row-data scanning) and
//! translates it to a Helium [`crate::Schema`].  Each top-level field of the Parquet
//! message becomes a top-level [`crate::ColumnSpec`].
//!
//! # Parquet → Helium type mapping
//!
//! | Parquet physical type | Logical / converted type | Helium `LogicalType` |
//! |---|---|---|
//! | `BOOLEAN` | — | `Primitive(U8)` (0 = false, 1 = true) |
//! | `INT32` | — | `Primitive(I32)` |
//! | `INT32` | `DATE` | `Date { Days }` |
//! | `INT32` | `TIME` | `Primitive(I32)` (time-of-day out of scope) |
//! | `INT32` | `Integer { bits=8, signed=false }` | `Primitive(U8)` |
//! | `INT32` | `Integer { bits=8, signed=true }` | `Primitive(I8)` |
//! | `INT32` | `Integer { bits=16, signed=false }` | `Primitive(U16)` |
//! | `INT32` | `Integer { bits=16, signed=true }` | `Primitive(I16)` |
//! | `INT32` | `Integer { bits=32, signed=false }` | `Primitive(U32)` |
//! | `INT64` | — | `Primitive(I64)` |
//! | `INT64` | `TIMESTAMP` | `Datetime { unit, timezone }` |
//! | `INT64` | `Integer { bits=64, signed=false }` | `Primitive(U64)` |
//! | `INT96` | — | `Primitive(I64)` (truncated legacy timestamp) |
//! | `FLOAT` | — | `Primitive(F32)` |
//! | `DOUBLE` | — | `Primitive(F64)` |
//! | `BYTE_ARRAY` | `UTF8` / `String` | `Utf8` |
//! | `BYTE_ARRAY` | `ENUM` / `Enum` | `Dictionary { inner: Utf8 }` |
//! | `BYTE_ARRAY` | `DECIMAL` | `Decimal128 { precision, scale }` |
//! | `BYTE_ARRAY` | — | `Binary` |
//! | `FIXED_LEN_BYTE_ARRAY` | `UUID` | `Binary` |
//! | `FIXED_LEN_BYTE_ARRAY` | — | `Binary` |
//! | Group with `LIST` annotation | — | `List { inner }` |
//! | Group with `MAP` annotation | — | `Map { key, value }` |
//! | Group (no annotation) | — | `Struct { fields }` |
//!
//! **Nullability**: Parquet `OPTIONAL` fields become `Nullable { inner }`
//! ([`crate::LogicalType::Nullable`]); `REQUIRED` fields are non-nullable.
//!
//! # Limitations
//!
//! * `Time` (time-of-day, `INT32`/`INT64` with `TIME` logical type) is out of
//!   scope for this version — Helium has no time-of-day type yet.  These columns
//!   fall back to `Primitive(I32)` / `Primitive(I64)`.
//! * `UUID` (`FIXED_LEN_BYTE_ARRAY(16)`) maps to `Binary` (no semantic hint yet).
//! * Non-standard `LIST` / `MAP` nesting patterns fall back to treating the
//!   group as a `Struct`.
//! * Recursive / self-referential Parquet schemas are not supported.

use std::path::Path;

use crate::{
    ColumnSpec, DataType, DateUnit, FieldSpec, HeliumError, LogicalType, Result, Schema, TimeUnit,
};
use parquet::basic::{ConvertedType, LogicalType as PqLogical, Repetition, Type as PqPhysical};
use parquet::schema::types::Type as PqType;

use super::encodings::default_encodings;

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Infer a Helium [`Schema`] from a Parquet file at `path`.
///
/// Only the file footer / schema metadata is read — no row data is scanned.
///
/// # Errors
///
/// Returns [`HeliumError::Io`] if the file cannot be opened,
/// [`HeliumError::Format`] for Parquet parse errors, or
/// [`HeliumError::Schema`] if translation or validation fails.
pub fn schema_from_parquet(path: &Path) -> Result<Schema> {
    use parquet::file::reader::{FileReader, SerializedFileReader};
    use std::fs::File;

    let file = File::open(path).map_err(HeliumError::Io)?;
    let reader = SerializedFileReader::new(file)
        .map_err(|e| HeliumError::Format(format!("Parquet open error: {e}")))?;
    let schema = reader.metadata().file_metadata().schema().clone();
    schema_from_parquet_schema(&schema)
}

/// Translate a [`parquet::schema::types::Type`] (root message group) to a
/// Helium [`Schema`].
///
/// Each field of the message becomes one top-level [`ColumnSpec`].  Use this
/// function when you already have a Parquet schema object (e.g. from
/// `parquet::schema::parser::parse_message_type`).
pub fn schema_from_parquet_schema(message: &PqType) -> Result<Schema> {
    if !message.is_group() {
        return Err(HeliumError::Schema {
            column: "<parquet>".into(),
            reason: "Parquet schema root must be a message (group) type".into(),
        });
    }
    let mut columns = Vec::new();
    for field in message.get_fields() {
        let name = field.name().to_string();
        let lt = parquet_field_to_logical_type(field, 0).map_err(|e| match e {
            HeliumError::Schema { reason, .. } => HeliumError::Schema {
                column: name.clone(),
                reason,
            },
            other => other,
        })?;
        let enc = default_encodings(&lt);
        columns.push(ColumnSpec::new(name, lt, enc));
    }
    let schema = Schema::new(columns);
    schema.validate()?;
    Ok(schema)
}

// ---------------------------------------------------------------------------
// Core translation
// ---------------------------------------------------------------------------

/// Maximum nesting depth for Parquet type translation.
const MAX_DEPTH: usize = 64;

/// Translate a single Parquet [`PqType`] field to a Helium [`LogicalType`].
fn parquet_field_to_logical_type(field: &PqType, depth: usize) -> Result<LogicalType> {
    if depth >= MAX_DEPTH {
        return Err(HeliumError::Schema {
            column: field.name().into(),
            reason: format!("Parquet schema nesting exceeds maximum depth of {MAX_DEPTH}"),
        });
    }

    let info = field.get_basic_info();
    let optional = info.has_repetition() && info.repetition() == Repetition::OPTIONAL;
    let repeated = info.has_repetition() && info.repetition() == Repetition::REPEATED;

    let lt = if field.is_primitive() {
        translate_primitive(field)?
    } else {
        // Group type.
        let logical = info.logical_type_ref();
        let converted = info.converted_type();

        let is_list = matches!(logical, Some(PqLogical::List)) || converted == ConvertedType::LIST;
        let is_map = matches!(logical, Some(PqLogical::Map)) || converted == ConvertedType::MAP;

        if is_list {
            translate_list_group(field, depth)?
        } else if is_map {
            translate_map_group(field, depth)?
        } else if repeated {
            // Bare repeated group: treat as List<Struct>.
            let inner = translate_plain_group(field, depth)?;
            LogicalType::List {
                inner: Box::new(inner),
            }
        } else {
            translate_plain_group(field, depth)?
        }
    };

    if optional {
        Ok(LogicalType::Nullable {
            inner: Box::new(lt),
        })
    } else {
        Ok(lt)
    }
}

// ---------------------------------------------------------------------------
// Primitive translation
// ---------------------------------------------------------------------------

fn translate_primitive(field: &PqType) -> Result<LogicalType> {
    let info = field.get_basic_info();
    let phys = field.get_physical_type();
    let logical = info.logical_type_ref();
    let converted = info.converted_type();

    let lt = match phys {
        PqPhysical::BOOLEAN => LogicalType::Primitive {
            data_type: DataType::U8,
        },
        PqPhysical::INT32 => translate_int32(logical, converted),
        PqPhysical::INT64 => translate_int64(logical, converted),
        // INT96 is a legacy 96-bit timestamp used by Impala/Hive.
        // TODO(meta): carry semantic hint "int96_timestamp" once ColumnSpec::meta exists.
        PqPhysical::INT96 => LogicalType::Primitive {
            data_type: DataType::I64,
        },
        PqPhysical::FLOAT => LogicalType::Primitive {
            data_type: DataType::F32,
        },
        PqPhysical::DOUBLE => LogicalType::Primitive {
            data_type: DataType::F64,
        },
        PqPhysical::BYTE_ARRAY => translate_byte_array(logical, converted),
        PqPhysical::FIXED_LEN_BYTE_ARRAY => translate_fixed(logical),
    };
    Ok(lt)
}

fn translate_int32(logical: Option<&PqLogical>, converted: ConvertedType) -> LogicalType {
    if let Some(lt) = logical {
        return match lt {
            PqLogical::Date => LogicalType::Date {
                unit: DateUnit::Days,
            },
            // Time: out of scope — no Helium Time-of-day type yet; fall back to I32.
            PqLogical::Time { .. } => LogicalType::Primitive {
                data_type: DataType::I32,
            },
            PqLogical::Integer {
                bit_width,
                is_signed,
            } => match (bit_width, is_signed) {
                (8, false) => LogicalType::Primitive {
                    data_type: DataType::U8,
                },
                (8, true) => LogicalType::Primitive {
                    data_type: DataType::I8,
                },
                (16, false) => LogicalType::Primitive {
                    data_type: DataType::U16,
                },
                (16, true) => LogicalType::Primitive {
                    data_type: DataType::I16,
                },
                (32, false) => LogicalType::Primitive {
                    data_type: DataType::U32,
                },
                _ => LogicalType::Primitive {
                    data_type: DataType::I32,
                },
            },
            PqLogical::Decimal { precision, scale } => {
                // Parquet Decimal over INT32 — precision fits u8 (max 9 for INT32).
                // Scale is always ≥ 0 in the Parquet spec.
                let p = (*precision).min(38) as u8;
                let s = (*scale).max(0).min(p as i32) as u8;
                LogicalType::Decimal128 {
                    precision: p,
                    scale: s,
                }
            }
            _ => LogicalType::Primitive {
                data_type: DataType::I32,
            },
        };
    }
    match converted {
        ConvertedType::DATE => LogicalType::Date {
            unit: DateUnit::Days,
        },
        ConvertedType::TIME_MILLIS => LogicalType::Primitive {
            data_type: DataType::I32,
        },
        ConvertedType::UINT_8 => LogicalType::Primitive {
            data_type: DataType::U8,
        },
        ConvertedType::UINT_16 => LogicalType::Primitive {
            data_type: DataType::U16,
        },
        ConvertedType::UINT_32 => LogicalType::Primitive {
            data_type: DataType::U32,
        },
        ConvertedType::INT_8 => LogicalType::Primitive {
            data_type: DataType::I8,
        },
        ConvertedType::INT_16 => LogicalType::Primitive {
            data_type: DataType::I16,
        },
        ConvertedType::INT_32 | ConvertedType::DECIMAL => LogicalType::Primitive {
            data_type: DataType::I32,
        },
        _ => LogicalType::Primitive {
            data_type: DataType::I32,
        },
    }
}

fn translate_int64(logical: Option<&PqLogical>, converted: ConvertedType) -> LogicalType {
    if let Some(lt) = logical {
        return match lt {
            PqLogical::Timestamp {
                unit,
                is_adjusted_to_u_t_c,
            } => {
                use parquet::basic::TimeUnit as PqTimeUnit;
                // Parquet's TimeUnit enum has MILLIS, MICROS, NANOS.
                // SECONDS is not a standard Parquet timestamp unit; treat as MILLIS.
                let helium_unit = match unit {
                    PqTimeUnit::MILLIS => TimeUnit::Millis,
                    PqTimeUnit::MICROS => TimeUnit::Micros,
                    PqTimeUnit::NANOS => TimeUnit::Nanos,
                };
                // Parquet `is_adjusted_to_UTC` true → timezone = Some("UTC").
                let timezone = if *is_adjusted_to_u_t_c {
                    Some("UTC".to_string())
                } else {
                    None
                };
                LogicalType::Datetime {
                    unit: helium_unit,
                    timezone,
                }
            }
            // Time of day: out of scope — no Helium Time-of-day type yet; fall back to I64.
            PqLogical::Time { .. } => LogicalType::Primitive {
                data_type: DataType::I64,
            },
            PqLogical::Integer {
                bit_width,
                is_signed,
            } => match (bit_width, is_signed) {
                (64, false) => LogicalType::Primitive {
                    data_type: DataType::U64,
                },
                _ => LogicalType::Primitive {
                    data_type: DataType::I64,
                },
            },
            PqLogical::Decimal { precision, scale } => {
                let p = (*precision).min(38) as u8;
                let s = (*scale).max(0).min(p as i32) as u8;
                LogicalType::Decimal128 {
                    precision: p,
                    scale: s,
                }
            }
            _ => LogicalType::Primitive {
                data_type: DataType::I64,
            },
        };
    }
    match converted {
        ConvertedType::TIMESTAMP_MILLIS => LogicalType::Datetime {
            unit: TimeUnit::Millis,
            timezone: Some("UTC".to_string()),
        },
        ConvertedType::TIMESTAMP_MICROS => LogicalType::Datetime {
            unit: TimeUnit::Micros,
            timezone: Some("UTC".to_string()),
        },
        ConvertedType::TIME_MICROS => LogicalType::Primitive {
            data_type: DataType::I64,
        },
        ConvertedType::UINT_64 => LogicalType::Primitive {
            data_type: DataType::U64,
        },
        ConvertedType::INT_64 | ConvertedType::DECIMAL => LogicalType::Primitive {
            data_type: DataType::I64,
        },
        _ => LogicalType::Primitive {
            data_type: DataType::I64,
        },
    }
}

fn translate_byte_array(logical: Option<&PqLogical>, converted: ConvertedType) -> LogicalType {
    if let Some(lt) = logical {
        return match lt {
            PqLogical::String | PqLogical::Json | PqLogical::Bson => LogicalType::Utf8,
            PqLogical::Enum => LogicalType::Dictionary {
                inner: Box::new(LogicalType::Utf8),
            },
            PqLogical::Decimal { precision, scale } => {
                let p = (*precision).min(38) as u8;
                let s = (*scale).max(0).min(p as i32) as u8;
                LogicalType::Decimal128 {
                    precision: p,
                    scale: s,
                }
            }
            PqLogical::Uuid => LogicalType::Binary,
            _ => LogicalType::Binary,
        };
    }
    match converted {
        ConvertedType::UTF8 | ConvertedType::JSON | ConvertedType::BSON => LogicalType::Utf8,
        ConvertedType::ENUM => LogicalType::Dictionary {
            inner: Box::new(LogicalType::Utf8),
        },
        ConvertedType::DECIMAL => LogicalType::Binary,
        _ => LogicalType::Binary,
    }
}

fn translate_fixed(logical: Option<&PqLogical>) -> LogicalType {
    if let Some(lt) = logical {
        match lt {
            PqLogical::Uuid => return LogicalType::Binary,
            PqLogical::Decimal { precision, scale } => {
                let p = (*precision).min(38) as u8;
                let s = (*scale).max(0).min(p as i32) as u8;
                return LogicalType::Decimal128 {
                    precision: p,
                    scale: s,
                };
            }
            _ => {}
        }
    }
    LogicalType::Binary
}

// ---------------------------------------------------------------------------
// Group type translation
// ---------------------------------------------------------------------------

/// Translate a group with `LIST` annotation → [`LogicalType::List`].
///
/// Handles the standard Parquet 3-level LIST structure:
/// ```text
/// <list-rep> group <name> (LIST) {
///   repeated group list {
///     <element-rep> <type> element;
///   }
/// }
/// ```
/// Falls back to treating the group as a `Struct` for non-standard patterns.
fn translate_list_group(group: &PqType, depth: usize) -> Result<LogicalType> {
    let fields = group.get_fields();
    if fields.len() == 1 {
        let middle = &fields[0];
        if middle.is_group() {
            let middle_info = middle.get_basic_info();
            let is_repeated =
                middle_info.has_repetition() && middle_info.repetition() == Repetition::REPEATED;
            if is_repeated {
                let inner_fields = middle.get_fields();
                if inner_fields.len() == 1 {
                    let elem = &inner_fields[0];
                    let inner_lt = parquet_field_to_logical_type(elem, depth + 1)?;
                    return Ok(LogicalType::List {
                        inner: Box::new(inner_lt),
                    });
                }
            }
        } else {
            // Two-level deprecated LIST pattern.
            let inner_lt = parquet_field_to_logical_type(middle, depth + 1)?;
            return Ok(LogicalType::List {
                inner: Box::new(inner_lt),
            });
        }
    }
    // Non-standard: fall back to Struct.
    translate_plain_group(group, depth)
}

/// Translate a group with `MAP` annotation → [`LogicalType::Map`].
///
/// Handles the standard Parquet MAP structure:
/// ```text
/// <map-rep> group <name> (MAP) {
///   repeated group key_value {
///     required <key-type> key;
///     <value-rep> <value-type> value;
///   }
/// }
/// ```
fn translate_map_group(group: &PqType, depth: usize) -> Result<LogicalType> {
    let fields = group.get_fields();
    if fields.len() == 1 {
        let kv_group = &fields[0];
        if kv_group.is_group() {
            let kv_fields = kv_group.get_fields();
            if kv_fields.len() == 2 {
                let key_field = &kv_fields[0];
                let val_field = &kv_fields[1];
                let key_lt = parquet_field_to_logical_type(key_field, depth + 1)?;
                let value_lt = parquet_field_to_logical_type(val_field, depth + 1)?;
                return Ok(LogicalType::Map {
                    key: Box::new(key_lt),
                    value: Box::new(value_lt),
                });
            }
        }
    }
    // Non-standard: fall back to Struct.
    translate_plain_group(group, depth)
}

/// Translate an unannotated group → [`LogicalType::Struct`].
fn translate_plain_group(group: &PqType, depth: usize) -> Result<LogicalType> {
    if depth >= MAX_DEPTH {
        return Err(HeliumError::Schema {
            column: group.name().into(),
            reason: format!("Parquet schema nesting exceeds maximum depth of {MAX_DEPTH}"),
        });
    }
    let mut field_specs = Vec::new();
    for field in group.get_fields() {
        let name = field.name().to_string();
        let lt = parquet_field_to_logical_type(field, depth + 1)?;
        let enc = default_encodings(&lt);
        field_specs.push(FieldSpec::new(name, lt, enc));
    }
    Ok(LogicalType::Struct {
        fields: field_specs,
    })
}

// ---------------------------------------------------------------------------
// Parquet writer
// ---------------------------------------------------------------------------

/// Write a Helium schema + columns to a Parquet file.
///
/// Only flat schemas are supported: `Primitive`, `Utf8`, `Binary`, and their
/// `Nullable<>` wrappers. Nested types (Struct, List, Map, Union) return
/// [`HeliumError::Schema`] with a message noting the limitation.
///
/// # Parquet file structure
///
/// A single row group is written. Each Helium column maps to one Parquet
/// leaf column. `REQUIRED` repetition is used for non-nullable columns;
/// `OPTIONAL` for `Nullable<T>`.
///
/// # Errors
///
/// - [`HeliumError::Schema`] for unsupported types or missing columns.
/// - [`HeliumError::Io`] / [`HeliumError::Format`] for Parquet-level errors.
pub fn write_parquet<W: std::io::Write + Send>(
    schema: &Schema,
    columns: &std::collections::HashMap<String, crate::LogicalColumn>,
    writer: W,
) -> Result<()> {
    use parquet::file::properties::WriterProperties;
    use parquet::file::writer::SerializedFileWriter;
    use std::sync::Arc;

    // Build Parquet schema from Helium schema.
    let pq_fields: Vec<Arc<PqType>> = schema
        .columns
        .iter()
        .map(|spec| helium_logical_type_to_parquet_field(&spec.name, &spec.logical_type))
        .collect::<Result<_>>()?;

    let root = PqType::group_type_builder("schema")
        .with_fields(pq_fields)
        .build()
        .map_err(|e| HeliumError::Format(format!("Parquet schema build error: {e}")))?;
    let pq_schema = Arc::new(root);
    let props = Arc::new(WriterProperties::builder().build());
    let mut file_writer = SerializedFileWriter::new(writer, pq_schema, props)
        .map_err(|e| HeliumError::Format(format!("Parquet writer init error: {e}")))?;

    // Determine row count.
    let row_count = schema
        .columns
        .first()
        .and_then(|spec| columns.get(&spec.name))
        .map(|lc| lc.row_count())
        .unwrap_or(0);

    // Write a single row group.
    let mut rg_writer = file_writer
        .next_row_group()
        .map_err(|e| HeliumError::Format(format!("Parquet row group error: {e}")))?;

    for spec in &schema.columns {
        let lc = columns.get(&spec.name).ok_or_else(|| HeliumError::Schema {
            column: spec.name.clone(),
            reason: "column in schema but missing from data map".into(),
        })?;
        if lc.row_count() != row_count {
            return Err(HeliumError::Schema {
                column: spec.name.clone(),
                reason: format!(
                    "row count mismatch: expected {row_count}, got {}",
                    lc.row_count()
                ),
            });
        }

        let mut col_writer = rg_writer
            .next_column()
            .map_err(|e| HeliumError::Format(format!("Parquet column writer error: {e}")))?
            .ok_or_else(|| HeliumError::Format("fewer Parquet columns than schema".into()))?;

        write_parquet_column(&mut col_writer, &spec.logical_type, lc, &spec.name)?;

        col_writer
            .close()
            .map_err(|e| HeliumError::Format(format!("Parquet column close error: {e}")))?;
    }

    rg_writer
        .close()
        .map_err(|e| HeliumError::Format(format!("Parquet row group close error: {e}")))?;
    file_writer
        .close()
        .map_err(|e| HeliumError::Format(format!("Parquet file close error: {e}")))?;
    Ok(())
}

/// Write one Helium logical column to a Parquet `SerializedColumnWriter`.
fn write_parquet_column(
    col: &mut parquet::file::writer::SerializedColumnWriter<'_>,
    lt: &LogicalType,
    lc: &crate::LogicalColumn,
    col_name: &str,
) -> Result<()> {
    use crate::LogicalColumn;
    use parquet::data_type::{ByteArray, ByteArrayType};

    match (lt, lc) {
        // ── Non-nullable primitives ────────────────────────────────────────
        (LogicalType::Primitive { .. }, LogicalColumn::Primitive(cd)) => {
            write_column_data(col, cd, None, col_name)
        }
        // ── Utf8 (required) ───────────────────────────────────────────────
        (LogicalType::Utf8, LogicalColumn::Utf8(strings)) => {
            let data: Vec<ByteArray> = strings
                .iter()
                .map(|s| ByteArray::from(s.as_bytes()))
                .collect();
            col.typed::<ByteArrayType>()
                .write_batch(&data, None, None)
                .map(|_| ())
                .map_err(|e| HeliumError::Format(format!("Parquet UTF8 write error: {e}")))
        }
        // ── Binary (required) ─────────────────────────────────────────────
        (LogicalType::Binary, LogicalColumn::Binary(blobs)) => {
            let data: Vec<ByteArray> = blobs
                .iter()
                .map(|b| ByteArray::from(b.as_slice()))
                .collect();
            col.typed::<ByteArrayType>()
                .write_batch(&data, None, None)
                .map(|_| ())
                .map_err(|e| HeliumError::Format(format!("Parquet Binary write error: {e}")))
        }
        // ── recursive Nullable: dispatch by inner type ───────────────────────────
        // This is the path real callers hit: HeliumReader returns the recursive LogicalColumn::Nullable
        // for the recursive LogicalType::Nullable, so write_parquet from a HeliumReader → CLI flow
        // lands here. The `value` box carries the *compacted* inner column (only the
        // non-null rows), which is exactly what parquet's write_batch wants:
        // `values.len() == count(def_level == max_def_level)`.
        (LogicalType::Nullable { inner }, LogicalColumn::Nullable { present, value }) => {
            let def_levels: Vec<i16> = present.iter().map(|&p| if p { 1 } else { 0 }).collect();
            match (inner.as_ref(), value.as_ref()) {
                (LogicalType::Primitive { .. }, LogicalColumn::Primitive(cd)) => {
                    write_column_data(col, cd, Some(&def_levels), col_name)
                }
                (LogicalType::Utf8, LogicalColumn::Utf8(strings)) => {
                    let data: Vec<ByteArray> = strings
                        .iter()
                        .map(|s| ByteArray::from(s.as_bytes()))
                        .collect();
                    col.typed::<ByteArrayType>()
                        .write_batch(&data, Some(&def_levels), None)
                        .map(|_| ())
                        .map_err(|e| {
                            HeliumError::Format(format!("Parquet Nullable<Utf8> write error: {e}"))
                        })
                }
                (LogicalType::Binary, LogicalColumn::Binary(blobs)) => {
                    let data: Vec<ByteArray> = blobs
                        .iter()
                        .map(|b| ByteArray::from(b.as_slice()))
                        .collect();
                    col.typed::<ByteArrayType>()
                        .write_batch(&data, Some(&def_levels), None)
                        .map(|_| ())
                        .map_err(|e| {
                            HeliumError::Format(format!(
                                "Parquet Nullable<Binary> write error: {e}"
                            ))
                        })
                }
                (other_inner, _) => Err(HeliumError::Schema {
                    column: col_name.to_string(),
                    reason: format!(
                        "parquet writer does not yet support Nullable<{:?}> types \
                         (only Primitive / Utf8 / Binary inner)",
                        other_inner
                    ),
                }),
            }
        }
        // ── legacy flat compat: NullablePrim ────────────────────────────────────────
        (LogicalType::NullablePrim { .. }, LogicalColumn::NullablePrim { present, values }) => {
            let def_levels: Vec<i16> = present.iter().map(|&p| if p { 1 } else { 0 }).collect();
            write_column_data(col, values, Some(&def_levels), col_name)
        }
        // ── legacy flat compat: NullableUtf8 ───────────────────────────────────────
        (LogicalType::NullableUtf8, LogicalColumn::NullableUtf8 { present, strings }) => {
            let def_levels: Vec<i16> = present.iter().map(|&p| if p { 1 } else { 0 }).collect();
            let data: Vec<ByteArray> = strings
                .iter()
                .map(|s| ByteArray::from(s.as_bytes()))
                .collect();
            col.typed::<ByteArrayType>()
                .write_batch(&data, Some(&def_levels), None)
                .map(|_| ())
                .map_err(|e| HeliumError::Format(format!("Parquet NullableUtf8 write error: {e}")))
        }
        // ── legacy flat compat: NullableBinary ─────────────────────────────────────
        (LogicalType::NullableBinary, LogicalColumn::NullableBinary { present, blobs }) => {
            let def_levels: Vec<i16> = present.iter().map(|&p| if p { 1 } else { 0 }).collect();
            let data: Vec<ByteArray> = blobs
                .iter()
                .map(|b| ByteArray::from(b.as_slice()))
                .collect();
            col.typed::<ByteArrayType>()
                .write_batch(&data, Some(&def_levels), None)
                .map(|_| ())
                .map_err(|e| {
                    HeliumError::Format(format!("Parquet NullableBinary write error: {e}"))
                })
        }
        // ── Unsupported nested types ───────────────────────────────────────
        (lt, _) => Err(HeliumError::Schema {
            column: col_name.to_string(),
            reason: format!("parquet writer does not yet support {:?} types", lt),
        }),
    }
}

/// Write [`ColumnData`] to a Parquet column writer.
///
/// `def_levels` is `None` for REQUIRED columns and `Some(levels)` for OPTIONAL.
fn write_column_data(
    col: &mut parquet::file::writer::SerializedColumnWriter<'_>,
    cd: &crate::ColumnData,
    def_levels: Option<&[i16]>,
    _col_name: &str,
) -> Result<()> {
    use crate::ColumnData;
    use parquet::data_type::{
        ByteArray, ByteArrayType, DoubleType, FloatType, Int32Type, Int64Type,
    };

    match cd {
        ColumnData::I8(v) => {
            let data: Vec<i32> = v.iter().map(|x| *x as i32).collect();
            col.typed::<Int32Type>()
                .write_batch(&data, def_levels, None)
                .map(|_| ())
                .map_err(|e| HeliumError::Format(format!("Parquet I8 write error: {e}")))
        }
        ColumnData::I16(v) => {
            let data: Vec<i32> = v.iter().map(|x| *x as i32).collect();
            col.typed::<Int32Type>()
                .write_batch(&data, def_levels, None)
                .map(|_| ())
                .map_err(|e| HeliumError::Format(format!("Parquet I16 write error: {e}")))
        }
        ColumnData::I32(v) => col
            .typed::<Int32Type>()
            .write_batch(v, def_levels, None)
            .map(|_| ())
            .map_err(|e| HeliumError::Format(format!("Parquet I32 write error: {e}"))),
        ColumnData::I64(v) => col
            .typed::<Int64Type>()
            .write_batch(v, def_levels, None)
            .map(|_| ())
            .map_err(|e| HeliumError::Format(format!("Parquet I64 write error: {e}"))),
        ColumnData::U8(v) => {
            let data: Vec<i32> = v.iter().map(|x| *x as i32).collect();
            col.typed::<Int32Type>()
                .write_batch(&data, def_levels, None)
                .map(|_| ())
                .map_err(|e| HeliumError::Format(format!("Parquet U8 write error: {e}")))
        }
        ColumnData::U16(v) => {
            let data: Vec<i32> = v.iter().map(|x| *x as i32).collect();
            col.typed::<Int32Type>()
                .write_batch(&data, def_levels, None)
                .map(|_| ())
                .map_err(|e| HeliumError::Format(format!("Parquet U16 write error: {e}")))
        }
        ColumnData::U32(v) => {
            let data: Vec<i64> = v.iter().map(|x| *x as i64).collect();
            col.typed::<Int64Type>()
                .write_batch(&data, def_levels, None)
                .map(|_| ())
                .map_err(|e| HeliumError::Format(format!("Parquet U32 write error: {e}")))
        }
        ColumnData::U64(v) => {
            // Parquet INT64 is the closest representation for U64 (values > i64::MAX wrap).
            let data: Vec<i64> = v.iter().map(|x| *x as i64).collect();
            col.typed::<Int64Type>()
                .write_batch(&data, def_levels, None)
                .map(|_| ())
                .map_err(|e| HeliumError::Format(format!("Parquet U64 write error: {e}")))
        }
        ColumnData::F32(v) => col
            .typed::<FloatType>()
            .write_batch(v, def_levels, None)
            .map(|_| ())
            .map_err(|e| HeliumError::Format(format!("Parquet F32 write error: {e}"))),
        ColumnData::F64(v) => col
            .typed::<DoubleType>()
            .write_batch(v, def_levels, None)
            .map(|_| ())
            .map_err(|e| HeliumError::Format(format!("Parquet F64 write error: {e}"))),
        ColumnData::Bytes(v) => {
            let ba = ByteArray::from(v.as_slice());
            col.typed::<ByteArrayType>()
                .write_batch(&[ba], def_levels, None)
                .map(|_| ())
                .map_err(|e| HeliumError::Format(format!("Parquet Bytes write error: {e}")))
        }
    }
}

/// Build a Parquet field type from a Helium [`LogicalType`].
///
/// Only flat types are supported; nested types return [`HeliumError::Schema`].
fn helium_logical_type_to_parquet_field(
    name: &str,
    lt: &LogicalType,
) -> Result<std::sync::Arc<PqType>> {
    use parquet::basic::{ConvertedType, Repetition, Type as PqPhysical};
    use std::sync::Arc;

    match lt {
        LogicalType::Primitive { data_type } => {
            let (phys, conv) = helium_dt_to_parquet_physical(*data_type);
            let field = PqType::primitive_type_builder(name, phys)
                .with_repetition(Repetition::REQUIRED)
                .with_converted_type(conv)
                .build()
                .map_err(|e| HeliumError::Format(format!("Parquet field build error: {e}")))?;
            Ok(Arc::new(field))
        }
        LogicalType::Utf8 => {
            let field = PqType::primitive_type_builder(name, PqPhysical::BYTE_ARRAY)
                .with_repetition(Repetition::REQUIRED)
                .with_converted_type(ConvertedType::UTF8)
                .build()
                .map_err(|e| HeliumError::Format(format!("Parquet Utf8 field build error: {e}")))?;
            Ok(Arc::new(field))
        }
        LogicalType::Binary => {
            let field = PqType::primitive_type_builder(name, PqPhysical::BYTE_ARRAY)
                .with_repetition(Repetition::REQUIRED)
                .build()
                .map_err(|e| {
                    HeliumError::Format(format!("Parquet Binary field build error: {e}"))
                })?;
            Ok(Arc::new(field))
        }
        // Nullable<Primitive>.
        LogicalType::Nullable { inner }
            if matches!(inner.as_ref(), LogicalType::Primitive { .. }) =>
        {
            let data_type = match inner.as_ref() {
                LogicalType::Primitive { data_type } => *data_type,
                _ => unreachable!(), // guarded by pattern above
            };
            let (phys, conv) = helium_dt_to_parquet_physical(data_type);
            let field = PqType::primitive_type_builder(name, phys)
                .with_repetition(Repetition::OPTIONAL)
                .with_converted_type(conv)
                .build()
                .map_err(|e| {
                    HeliumError::Format(format!("Parquet Nullable<Prim> field build error: {e}"))
                })?;
            Ok(Arc::new(field))
        }
        // Nullable<Utf8>.
        LogicalType::Nullable { inner } if matches!(inner.as_ref(), LogicalType::Utf8) => {
            let field = PqType::primitive_type_builder(name, PqPhysical::BYTE_ARRAY)
                .with_repetition(Repetition::OPTIONAL)
                .with_converted_type(ConvertedType::UTF8)
                .build()
                .map_err(|e| {
                    HeliumError::Format(format!("Parquet Nullable<Utf8> field build error: {e}"))
                })?;
            Ok(Arc::new(field))
        }
        // Nullable<Binary>.
        LogicalType::Nullable { inner } if matches!(inner.as_ref(), LogicalType::Binary) => {
            let field = PqType::primitive_type_builder(name, PqPhysical::BYTE_ARRAY)
                .with_repetition(Repetition::OPTIONAL)
                .build()
                .map_err(|e| {
                    HeliumError::Format(format!("Parquet Nullable<Binary> field build error: {e}"))
                })?;
            Ok(Arc::new(field))
        }
        // Any other Nullable<nested> is unsupported.
        LogicalType::Nullable { inner } => Err(HeliumError::Schema {
            column: name.to_string(),
            reason: format!(
                "parquet writer does not yet support Nullable<{:?}> types",
                inner
            ),
        }),
        // legacy flat NullablePrim.
        LogicalType::NullablePrim { data_type } => {
            let (phys, conv) = helium_dt_to_parquet_physical(*data_type);
            let field = PqType::primitive_type_builder(name, phys)
                .with_repetition(Repetition::OPTIONAL)
                .with_converted_type(conv)
                .build()
                .map_err(|e| {
                    HeliumError::Format(format!("Parquet NullablePrim field build error: {e}"))
                })?;
            Ok(Arc::new(field))
        }
        // legacy flat NullableUtf8.
        LogicalType::NullableUtf8 => {
            let field = PqType::primitive_type_builder(name, PqPhysical::BYTE_ARRAY)
                .with_repetition(Repetition::OPTIONAL)
                .with_converted_type(ConvertedType::UTF8)
                .build()
                .map_err(|e| {
                    HeliumError::Format(format!("Parquet NullableUtf8 field build error: {e}"))
                })?;
            Ok(Arc::new(field))
        }
        // legacy flat NullableBinary.
        LogicalType::NullableBinary => {
            let field = PqType::primitive_type_builder(name, PqPhysical::BYTE_ARRAY)
                .with_repetition(Repetition::OPTIONAL)
                .build()
                .map_err(|e| {
                    HeliumError::Format(format!("Parquet NullableBinary field build error: {e}"))
                })?;
            Ok(Arc::new(field))
        }
        // Unsupported nested types.
        other => Err(HeliumError::Schema {
            column: name.to_string(),
            reason: format!("parquet writer does not yet support {:?} types", other),
        }),
    }
}

/// Map a Helium [`DataType`] to a Parquet physical type + converted type.
fn helium_dt_to_parquet_physical(
    dt: crate::DataType,
) -> (parquet::basic::Type, parquet::basic::ConvertedType) {
    use crate::DataType;
    use parquet::basic::{ConvertedType, Type as PqPhysical};
    match dt {
        DataType::I8 => (PqPhysical::INT32, ConvertedType::INT_8),
        DataType::I16 => (PqPhysical::INT32, ConvertedType::INT_16),
        DataType::I32 => (PqPhysical::INT32, ConvertedType::INT_32),
        DataType::I64 => (PqPhysical::INT64, ConvertedType::INT_64),
        DataType::U8 => (PqPhysical::INT32, ConvertedType::UINT_8),
        DataType::U16 => (PqPhysical::INT32, ConvertedType::UINT_16),
        DataType::U32 => (PqPhysical::INT64, ConvertedType::UINT_32),
        DataType::U64 => (PqPhysical::INT64, ConvertedType::UINT_64),
        DataType::F32 => (PqPhysical::FLOAT, ConvertedType::NONE),
        DataType::F64 => (PqPhysical::DOUBLE, ConvertedType::NONE),
        DataType::Bytes => (PqPhysical::BYTE_ARRAY, ConvertedType::NONE),
    }
}

// ---------------------------------------------------------------------------
// Unit tests for write_parquet
// ---------------------------------------------------------------------------

#[cfg(test)]
mod writer_tests {
    use super::*;
    use crate::schema::encodings::default_encodings;
    use crate::{ColumnData, ColumnSpec, DataType, LogicalColumn, LogicalType, Schema};
    use std::collections::HashMap;

    fn make_schema(cols: &[(&str, LogicalType)]) -> Schema {
        Schema::new(
            cols.iter()
                .map(|(name, lt)| {
                    let enc = default_encodings(lt);
                    ColumnSpec::new((*name).to_string(), lt.clone(), enc)
                })
                .collect(),
        )
    }

    #[test]
    fn write_parquet_flat_primitives() {
        let lt_i64 = LogicalType::Primitive {
            data_type: DataType::I64,
        };
        let lt_f64 = LogicalType::Primitive {
            data_type: DataType::F64,
        };
        let lt_utf8 = LogicalType::Utf8;
        let schema = make_schema(&[
            ("id", lt_i64.clone()),
            ("score", lt_f64.clone()),
            ("label", lt_utf8.clone()),
        ]);

        let mut cols: HashMap<String, LogicalColumn> = HashMap::new();
        cols.insert(
            "id".into(),
            LogicalColumn::Primitive(ColumnData::I64(vec![1, 2, 3])),
        );
        cols.insert(
            "score".into(),
            LogicalColumn::Primitive(ColumnData::F64(vec![1.1, 2.2, 3.3])),
        );
        cols.insert(
            "label".into(),
            LogicalColumn::Utf8(vec!["a".into(), "b".into(), "c".into()]),
        );

        // Write to a named temp file so we can re-open it for reading.
        let tmp = tempfile::NamedTempFile::new().unwrap();
        write_parquet(&schema, &cols, tmp.as_file().try_clone().unwrap()).unwrap();

        // Verify it's a valid Parquet file by reading back.
        use parquet::file::reader::{FileReader, SerializedFileReader};
        let file = std::fs::File::open(tmp.path()).unwrap();
        let reader = SerializedFileReader::new(file).unwrap();
        assert_eq!(reader.metadata().file_metadata().num_rows(), 3);
    }

    #[test]
    fn write_parquet_nullable_column() {
        // recursive shape — what HeliumReader actually returns for recursive Nullable schemas
        // and therefore what the CLI hands to write_parquet on real data.
        let lt = LogicalType::Nullable {
            inner: Box::new(LogicalType::Primitive {
                data_type: DataType::I32,
            }),
        };
        let schema = make_schema(&[("x", lt.clone())]);

        let mut cols: HashMap<String, LogicalColumn> = HashMap::new();
        cols.insert(
            "x".into(),
            LogicalColumn::Nullable {
                present: vec![true, false, true],
                value: Box::new(LogicalColumn::Primitive(ColumnData::I32(vec![10, 30]))),
            },
        );

        let tmp = tempfile::NamedTempFile::new().unwrap();
        write_parquet(&schema, &cols, tmp.as_file().try_clone().unwrap()).unwrap();

        use parquet::file::reader::{FileReader, SerializedFileReader};
        let file = std::fs::File::open(tmp.path()).unwrap();
        let reader = SerializedFileReader::new(file).unwrap();
        assert_eq!(reader.metadata().file_metadata().num_rows(), 3);
    }

    #[test]
    fn write_parquet_nullable_utf8_and_binary() {
        // Regression: Nullable<Utf8> and Nullable<Binary> previously expanded
        // their compact value array to full row count with empty-ByteArray
        // placeholders, which made the parquet ByteArray writer panic on
        // `assert!(self.data.is_some())` for the placeholder rows. The fix is
        // to pass the COMPACT values directly — parquet's write_batch already
        // expects values.len() == count(def_level == max).
        let schema = make_schema(&[
            (
                "name",
                LogicalType::Nullable {
                    inner: Box::new(LogicalType::Utf8),
                },
            ),
            (
                "blob",
                LogicalType::Nullable {
                    inner: Box::new(LogicalType::Binary),
                },
            ),
        ]);

        let mut cols: HashMap<String, LogicalColumn> = HashMap::new();
        cols.insert(
            "name".into(),
            LogicalColumn::Nullable {
                present: vec![true, false, true, false, true],
                value: Box::new(LogicalColumn::Utf8(vec![
                    "alpha".into(),
                    "gamma".into(),
                    "epsilon".into(),
                ])),
            },
        );
        cols.insert(
            "blob".into(),
            LogicalColumn::Nullable {
                present: vec![true, false, true, false, true],
                value: Box::new(LogicalColumn::Binary(vec![
                    vec![0xde, 0xad],
                    vec![0xbe, 0xef],
                    vec![0xfe, 0xed],
                ])),
            },
        );

        let tmp = tempfile::NamedTempFile::new().unwrap();
        write_parquet(&schema, &cols, tmp.as_file().try_clone().unwrap()).unwrap();

        use parquet::file::reader::{FileReader, SerializedFileReader};
        let file = std::fs::File::open(tmp.path()).unwrap();
        let reader = SerializedFileReader::new(file).unwrap();
        assert_eq!(reader.metadata().file_metadata().num_rows(), 5);
    }

    #[test]
    fn write_parquet_nested_type_error() {
        let lt = LogicalType::Struct { fields: vec![] };
        let schema = make_schema(&[("s", lt.clone())]);
        let mut cols: HashMap<String, LogicalColumn> = HashMap::new();
        cols.insert("s".into(), LogicalColumn::Struct { fields: vec![] });

        let tmp = tempfile::NamedTempFile::new().unwrap();
        let err = write_parquet(&schema, &cols, tmp.as_file().try_clone().unwrap());
        assert!(err.is_err(), "expected error for nested type");
        let msg = format!("{:?}", err.unwrap_err());
        assert!(
            msg.contains("parquet writer does not yet support"),
            "unexpected error: {msg}"
        );
    }
}
