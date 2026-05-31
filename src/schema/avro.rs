//! Avro `.avsc` → Helium [`crate::Schema`] converter.
//!
//! # Overview
//!
//! Parses an Avro schema specification (in JSON form, as produced by the Avro
//! IDL compiler or written by hand) and converts it to a Helium [`crate::Schema`] or
//! [`crate::LogicalType`].
//!
//! # Entry points
//!
//! * [`avsc_to_schema`] — convert a top-level Avro **record** to a Helium
//!   `Schema`. Each record field becomes a top-level [`crate::ColumnSpec`].
//! * [`avsc_to_logical_type`] — convert any Avro type expression to a
//!   [`crate::LogicalType`]. Useful when embedding the type inside a larger schema.
//!
//! # Avro → Helium type mapping
//!
//! | Avro type | Helium `LogicalType` |
//! |---|---|
//! | `boolean` | `Primitive(U8)` (0 = false, 1 = true) |
//! | `int` | `Primitive(I32)` |
//! | `long` | `Primitive(I64)` |
//! | `float` | `Primitive(F32)` |
//! | `double` | `Primitive(F64)` |
//! | `bytes` | `Binary` |
//! | `string` | `Utf8` |
//! | `record` | `Struct { fields }` |
//! | `array` | `List { inner }` |
//! | `map` | `Map { key: Utf8, value }` (Avro maps are always string-keyed) |
//! | `["null", T]` or `[T, "null"]` | `Nullable { inner: T }` |
//! | `[T]` (single element) | `T` (union wrapper stripped) |
//! | `[A, B, ...]` (no null, multiple) | `Union { variants }` |
//! | `["null", A, B, ...]` (null + multiple) | `Nullable { inner: Union { variants } }` |
//! | `enum` | `Dictionary { inner: Utf8 }` |
//! | `fixed` | `Binary` |
//!
//! # Avro logical types
//!
//! Logical types are accepted but the semantic hint is not yet propagated —
//! `ColumnSpec::meta` does not yet exist in `helium-core`. A `// TODO(meta):`
//! comment marks each site. Once `ColumnSpec::meta` is added, the hints
//! should flow through.
//!
//! | Avro logicalType | Base Avro type | Helium `LogicalType` |
//! |---|---|---|
//! | `date` | `int` | `Primitive(I32)` |
//! | `time-millis` | `int` | `Primitive(I32)` |
//! | `time-micros` | `long` | `Primitive(I64)` |
//! | `timestamp-millis` | `long` | `Primitive(I64)` |
//! | `timestamp-micros` | `long` | `Primitive(I64)` |
//! | `local-timestamp-millis` | `long` | `Primitive(I64)` |
//! | `local-timestamp-micros` | `long` | `Primitive(I64)` |
//! | `decimal` | `bytes` / `fixed` | `Binary` |
//! | `uuid` | `string` | `Binary` (16-byte binary UUID) |
//!
//! # Union variant naming
//!
//! For multi-variant non-null unions, variant names are derived as follows:
//! 1. A named Avro type (record, enum, fixed) uses its `name` field.
//! 2. A primitive type string uses the keyword itself (`"int"`, `"string"`, …).
//! 3. An `array` or `map` complex type uses `"array"` or `"map"`.
//! 4. A type with no identifiable name falls back to `"unknown"`.
//!
//! Variant names become wire-format-frozen in the resulting Helium schema (per
//! `Union` semantics in `helium_core`).
//!
//! # Limitations
//!
//! * **Recursive schemas** (a record referencing itself, directly or
//!   transitively) are detected and rejected. Flatten the schema before
//!   conversion.
//! * **Avro `error` type** and **protocol definitions** are rejected; use
//!   `record` instead.
//! * **Nested unions** (union-inside-union) are rejected per the Avro spec.
//! * Named-type **forward references** (reference before definition) are not
//!   supported.
//! * The top-level input to [`avsc_to_schema`] must be a `record`.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::{
    ColumnData, ColumnSpec, DataType, DateUnit, FieldSpec, HeliumError, LogicalColumn, LogicalType,
    MAX_NESTED_DEPTH, Result, Schema, TimeUnit,
};

use super::encodings::default_encodings;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Convert an Avro `.avsc` top-level **record** to a Helium [`Schema`].
///
/// Each field of the top-level Avro record becomes a top-level [`ColumnSpec`]
/// in the resulting `Schema`. Nested Avro records become
/// [`LogicalType::Struct`] values inside their parent column's type.
///
/// Default encoding pipelines are assigned based on type:
/// * Integer types (`int`, `long`, `boolean`): `delta → leb128 → zstd`
///   (except `U8`/boolean which skips `delta`: `leb128 → zstd`).
/// * Floating-point types (`float`, `double`): `gorilla → zstd`.
/// * String and binary (`string`, `bytes`, `fixed`): `delta → leb128 → zstd`
///   for offsets, `zstd` for data.
/// * Composite headers (list offsets, map offsets, nullable present bitmaps,
///   union tags): chosen per the inner column's physical type.
///
/// # Errors
///
/// Returns [`HeliumError::Schema`] if:
/// * The input is not valid JSON.
/// * The top-level type is not an Avro `record`.
/// * Any nested type-conversion fails (unknown type name, nested union,
///   depth exceeded, etc.).
pub fn avsc_to_schema(avsc: &str) -> Result<Schema> {
    let v: serde_json::Value = serde_json::from_str(avsc).map_err(|e| HeliumError::Schema {
        column: "<root>".into(),
        reason: format!("invalid Avro .avsc JSON: {e}"),
    })?;
    let mut ctx = ParseContext::default();
    let columns = parse_record_as_columns(&v, &mut ctx, 0)?;
    let schema = Schema::new(columns);
    // validate() catches encoding-count mismatches and depth cap — fail fast.
    schema.validate()?;
    Ok(schema)
}

/// Convert any Avro `.avsc` type expression to a Helium [`LogicalType`].
///
/// Unlike [`avsc_to_schema`], this accepts any Avro type as the root (not
/// just records). Records produce [`LogicalType::Struct`].
///
/// This function is primarily useful for unit testing individual type
/// conversions or for embedding Avro sub-schemas into a larger
/// hand-crafted `Schema`.
///
/// # Errors
///
/// Same conditions as [`avsc_to_schema`], except the top-level type is not
/// required to be a `record`.
pub fn avsc_to_logical_type(avsc: &str) -> Result<LogicalType> {
    let v: serde_json::Value = serde_json::from_str(avsc).map_err(|e| HeliumError::Schema {
        column: "<root>".into(),
        reason: format!("invalid Avro .avsc JSON: {e}"),
    })?;
    let mut ctx = ParseContext::default();
    parse_avro_type(&v, &mut ctx, 0)
}

// ---------------------------------------------------------------------------
// Internal parsing context
// ---------------------------------------------------------------------------

/// State threaded through the recursive Avro type parser.
#[derive(Default)]
struct ParseContext {
    /// Named Avro types (record, enum, fixed) seen so far, keyed by full name.
    /// Populated as types are encountered (inline definition before reference).
    named_types: HashMap<String, serde_json::Value>,
    /// Names currently being resolved — used to detect recursive schemas.
    resolving: HashSet<String>,
}

// ---------------------------------------------------------------------------
// Top-level record → Vec<ColumnSpec>
// ---------------------------------------------------------------------------

/// Parse an Avro record JSON value and return its fields as [`ColumnSpec`]s.
///
/// Each field becomes one top-level column; the field's type is recursively
/// converted.
fn parse_record_as_columns(
    v: &serde_json::Value,
    ctx: &mut ParseContext,
    depth: usize,
) -> Result<Vec<ColumnSpec>> {
    let obj = v.as_object().ok_or_else(|| HeliumError::Schema {
        column: "<root>".into(),
        reason: "avsc_to_schema requires a JSON object at the top level".into(),
    })?;

    let type_str = obj
        .get("type")
        .and_then(|t| t.as_str())
        .ok_or_else(|| HeliumError::Schema {
            column: "<root>".into(),
            reason: "top-level Avro object must have a string \"type\" field".into(),
        })?;

    if type_str != "record" {
        return Err(HeliumError::Schema {
            column: "<root>".into(),
            reason: format!("avsc_to_schema requires a top-level Avro record, got \"{type_str}\""),
        });
    }

    // Register this record's name before parsing its fields so that later
    // fields in the same record can reference it (though self-recursive
    // references are still rejected via `ctx.resolving`).
    if let Some(name) = obj.get("name").and_then(|n| n.as_str()) {
        ctx.named_types.insert(name.to_string(), v.clone());
    }

    let fields_val = obj.get("fields").ok_or_else(|| HeliumError::Schema {
        column: "<root>".into(),
        reason: "Avro record missing required \"fields\" array".into(),
    })?;

    let fields_arr = fields_val.as_array().ok_or_else(|| HeliumError::Schema {
        column: "<root>".into(),
        reason: "Avro record \"fields\" must be a JSON array".into(),
    })?;

    let mut columns = Vec::with_capacity(fields_arr.len());
    for field_val in fields_arr {
        let (name, lt) = parse_avro_field(field_val, ctx, depth)?;
        let encodings = default_encodings(&lt);
        columns.push(ColumnSpec::new(name, lt, encodings));
    }
    Ok(columns)
}

// ---------------------------------------------------------------------------
// Field parsing
// ---------------------------------------------------------------------------

/// Parse one Avro field object and return `(field_name, LogicalType)`.
fn parse_avro_field(
    v: &serde_json::Value,
    ctx: &mut ParseContext,
    depth: usize,
) -> Result<(String, LogicalType)> {
    let obj = v.as_object().ok_or_else(|| HeliumError::Schema {
        column: "<field>".into(),
        reason: "Avro field must be a JSON object".into(),
    })?;

    let name = obj
        .get("name")
        .and_then(|n| n.as_str())
        .ok_or_else(|| HeliumError::Schema {
            column: "<field>".into(),
            reason: "Avro field missing required \"name\" string".into(),
        })?
        .to_string();

    let type_val = obj.get("type").ok_or_else(|| HeliumError::Schema {
        column: name.clone(),
        reason: "Avro field missing required \"type\"".into(),
    })?;

    let lt = parse_avro_type(type_val, ctx, depth).map_err(|e| {
        // Annotate schema errors with the field name for better diagnostics.
        match e {
            HeliumError::Schema { column, reason }
                if column == "<root>" || column == "<union>" || column == "<field>" =>
            {
                HeliumError::Schema {
                    column: name.clone(),
                    reason,
                }
            }
            other => other,
        }
    })?;

    Ok((name, lt))
}

// ---------------------------------------------------------------------------
// Core recursive type parser
// ---------------------------------------------------------------------------

/// Parse any Avro type expression: a string (primitive / named ref), an object
/// (complex type), or a JSON array (union).
///
/// Depth enforcement is delegated to the container-type parsers
/// (`parse_record_type`, `parse_array_type`, `parse_map_type`) which check
/// `depth >= MAX_NESTED_DEPTH` before recursing. No separate top-level guard
/// is needed here.
fn parse_avro_type(
    v: &serde_json::Value,
    ctx: &mut ParseContext,
    depth: usize,
) -> Result<LogicalType> {
    match v {
        serde_json::Value::String(s) => parse_type_string(s, ctx, depth),
        serde_json::Value::Object(obj) => parse_type_object(obj, ctx, depth),
        serde_json::Value::Array(arr) => parse_union(arr, ctx, depth),
        other => Err(HeliumError::Schema {
            column: "<root>".into(),
            reason: format!("unexpected JSON value for Avro type: {other}"),
        }),
    }
}

/// Handle a string-form Avro type: primitive keyword or named-type reference.
fn parse_type_string(s: &str, ctx: &mut ParseContext, depth: usize) -> Result<LogicalType> {
    match s {
        "null" => Err(HeliumError::Schema {
            column: "<root>".into(),
            reason: "Avro \"null\" type is only valid inside a union".into(),
        }),
        "boolean" => Ok(LogicalType::Primitive {
            data_type: DataType::U8,
        }),
        "int" => Ok(LogicalType::Primitive {
            data_type: DataType::I32,
        }),
        "long" => Ok(LogicalType::Primitive {
            data_type: DataType::I64,
        }),
        "float" => Ok(LogicalType::Primitive {
            data_type: DataType::F32,
        }),
        "double" => Ok(LogicalType::Primitive {
            data_type: DataType::F64,
        }),
        "bytes" => Ok(LogicalType::Binary),
        "string" => Ok(LogicalType::Utf8),
        // Named-type reference: look up in the registry.
        name => {
            if ctx.resolving.contains(name) {
                return Err(HeliumError::Schema {
                    column: name.to_string(),
                    reason: format!(
                        "recursive Avro schema: type \"{name}\" references itself; \
                         flatten the schema before conversion"
                    ),
                });
            }
            let named_val =
                ctx.named_types
                    .get(name)
                    .cloned()
                    .ok_or_else(|| HeliumError::Schema {
                        column: name.to_string(),
                        reason: format!(
                            "unknown Avro named type \"{name}\"; \
                         define the type before referencing it"
                        ),
                    })?;
            ctx.resolving.insert(name.to_string());
            let result = parse_avro_type(&named_val, ctx, depth);
            ctx.resolving.remove(name);
            result
        }
    }
}

/// Handle an object-form Avro type `{"type": "...", ...}`, dispatching on
/// the `type` field and applying any `logicalType` override.
fn parse_type_object(
    obj: &serde_json::Map<String, serde_json::Value>,
    ctx: &mut ParseContext,
    depth: usize,
) -> Result<LogicalType> {
    let type_str = obj
        .get("type")
        .and_then(|t| t.as_str())
        .ok_or_else(|| HeliumError::Schema {
            column: "<root>".into(),
            reason: "Avro type object missing string \"type\" field".into(),
        })?;

    let logical_type = obj.get("logicalType").and_then(|lt| lt.as_str());

    match type_str {
        "null" => Err(HeliumError::Schema {
            column: "<root>".into(),
            reason: "Avro \"null\" type is only valid inside a union".into(),
        }),
        "boolean" => Ok(LogicalType::Primitive {
            data_type: DataType::U8,
        }),
        "int" => {
            match logical_type {
                Some("date") => Ok(LogicalType::Date {
                    unit: DateUnit::Days,
                }),
                // time-millis: Helium has no Time-of-day type yet; fall back to I32.
                // Documented limitation: see avro.rs module doc.
                _ => Ok(LogicalType::Primitive {
                    data_type: DataType::I32,
                }),
            }
        }
        "long" => {
            match logical_type {
                Some("timestamp-millis") => {
                    // Avro timestamp-millis is UTC-implied (no timezone string).
                    // We model this as timezone: None (unspecified / UTC-implied).
                    // Avro local-timestamp-* also has no explicit tz.
                    Ok(LogicalType::Datetime {
                        unit: TimeUnit::Millis,
                        timezone: None,
                    })
                }
                Some("timestamp-micros") => Ok(LogicalType::Datetime {
                    unit: TimeUnit::Micros,
                    timezone: None,
                }),
                Some("local-timestamp-millis") => {
                    // Avro local-timestamp has no timezone (wall clock).
                    // We model the same as timestamp-millis: timezone: None.
                    Ok(LogicalType::Datetime {
                        unit: TimeUnit::Millis,
                        timezone: None,
                    })
                }
                Some("local-timestamp-micros") => Ok(LogicalType::Datetime {
                    unit: TimeUnit::Micros,
                    timezone: None,
                }),
                // time-micros: out of scope (no Helium Time-of-day type yet); fall back.
                _ => Ok(LogicalType::Primitive {
                    data_type: DataType::I64,
                }),
            }
        }
        "float" => Ok(LogicalType::Primitive {
            data_type: DataType::F32,
        }),
        "double" => Ok(LogicalType::Primitive {
            data_type: DataType::F64,
        }),
        "bytes" => {
            match logical_type {
                Some("decimal") => {
                    // Extract precision/scale from the type object.
                    let precision = obj
                        .get("precision")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(38)
                        .min(38) as u8;
                    let scale = obj
                        .get("scale")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0)
                        .min(precision as u64) as u8;
                    if precision > 38 {
                        return Err(HeliumError::Schema {
                            column: "<root>".into(),
                            reason: format!(
                                "Avro decimal precision {precision} exceeds Helium \
                                 Decimal128 maximum of 38"
                            ),
                        });
                    }
                    Ok(LogicalType::Decimal128 { precision, scale })
                }
                _ => Ok(LogicalType::Binary),
            }
        }
        "string" => match logical_type {
            // TODO(meta): carry "uuid" semantic hint (16-byte binary UUID)
            // once ColumnSpec::meta exists.
            Some("uuid") => Ok(LogicalType::Binary),
            _ => Ok(LogicalType::Utf8),
        },
        "record" => parse_record_type(obj, ctx, depth),
        "enum" => parse_enum_type(obj, ctx),
        "array" => parse_array_type(obj, ctx, depth),
        "map" => parse_map_type(obj, ctx, depth),
        "fixed" => parse_fixed_type(obj, ctx, logical_type),
        "error" => Err(HeliumError::Schema {
            column: "<root>".into(),
            reason: "Avro \"error\" type is not supported; use \"record\" instead".into(),
        }),
        other => Err(HeliumError::Schema {
            column: "<root>".into(),
            reason: format!("unknown Avro type \"{other}\""),
        }),
    }
}

// ---------------------------------------------------------------------------
// Complex type parsers
// ---------------------------------------------------------------------------

/// Parse an Avro `record` → [`LogicalType::Struct`].
fn parse_record_type(
    obj: &serde_json::Map<String, serde_json::Value>,
    ctx: &mut ParseContext,
    depth: usize,
) -> Result<LogicalType> {
    if depth >= MAX_NESTED_DEPTH {
        return Err(HeliumError::Schema {
            column: "<root>".into(),
            reason: format!("Avro schema nesting exceeds the maximum depth of {MAX_NESTED_DEPTH}"),
        });
    }

    // Register this record under its name so later fields can reference it.
    if let Some(name) = obj.get("name").and_then(|n| n.as_str()) {
        ctx.named_types
            .insert(name.to_string(), serde_json::Value::Object(obj.clone()));
    }

    let fields_val = obj.get("fields").ok_or_else(|| HeliumError::Schema {
        column: "<root>".into(),
        reason: "Avro record missing required \"fields\" array".into(),
    })?;

    let fields_arr = fields_val.as_array().ok_or_else(|| HeliumError::Schema {
        column: "<root>".into(),
        reason: "Avro record \"fields\" must be a JSON array".into(),
    })?;

    let mut field_specs = Vec::with_capacity(fields_arr.len());
    for field_val in fields_arr {
        let (name, lt) = parse_avro_field(field_val, ctx, depth + 1)?;
        let encodings = default_encodings(&lt);
        field_specs.push(FieldSpec::new(name, lt, encodings));
    }

    Ok(LogicalType::Struct {
        fields: field_specs,
    })
}

/// Parse an Avro `enum` → [`LogicalType::Dictionary`]`{ inner: Utf8 }`.
///
/// The enum's `symbols` array defines the dictionary entries. The recursive
/// `Dictionary { inner: Utf8 }` variant is emitted by all new writers.
fn parse_enum_type(
    obj: &serde_json::Map<String, serde_json::Value>,
    ctx: &mut ParseContext,
) -> Result<LogicalType> {
    // Validate the symbols field exists and is an array.
    let symbols_val = obj.get("symbols").ok_or_else(|| HeliumError::Schema {
        column: "<root>".into(),
        reason: "Avro enum missing required \"symbols\" array".into(),
    })?;
    symbols_val.as_array().ok_or_else(|| HeliumError::Schema {
        column: "<root>".into(),
        reason: "Avro enum \"symbols\" must be a JSON array".into(),
    })?;

    // Register this enum under its name for later references.
    if let Some(name) = obj.get("name").and_then(|n| n.as_str()) {
        ctx.named_types
            .insert(name.to_string(), serde_json::Value::Object(obj.clone()));
    }

    Ok(LogicalType::Dictionary {
        inner: Box::new(LogicalType::Utf8),
    })
}

/// Parse an Avro `array` → [`LogicalType::List`].
fn parse_array_type(
    obj: &serde_json::Map<String, serde_json::Value>,
    ctx: &mut ParseContext,
    depth: usize,
) -> Result<LogicalType> {
    if depth >= MAX_NESTED_DEPTH {
        return Err(HeliumError::Schema {
            column: "<root>".into(),
            reason: format!("Avro schema nesting exceeds the maximum depth of {MAX_NESTED_DEPTH}"),
        });
    }

    let items = obj.get("items").ok_or_else(|| HeliumError::Schema {
        column: "<root>".into(),
        reason: "Avro array missing required \"items\" field".into(),
    })?;

    let inner = parse_avro_type(items, ctx, depth + 1)?;
    Ok(LogicalType::List {
        inner: Box::new(inner),
    })
}

/// Parse an Avro `map` → [`LogicalType::Map`] with [`LogicalType::Utf8`] key.
///
/// Avro maps are always string-keyed per the Avro specification.
fn parse_map_type(
    obj: &serde_json::Map<String, serde_json::Value>,
    ctx: &mut ParseContext,
    depth: usize,
) -> Result<LogicalType> {
    if depth >= MAX_NESTED_DEPTH {
        return Err(HeliumError::Schema {
            column: "<root>".into(),
            reason: format!("Avro schema nesting exceeds the maximum depth of {MAX_NESTED_DEPTH}"),
        });
    }

    let values_val = obj.get("values").ok_or_else(|| HeliumError::Schema {
        column: "<root>".into(),
        reason: "Avro map missing required \"values\" field".into(),
    })?;

    let value_lt = parse_avro_type(values_val, ctx, depth + 1)?;
    Ok(LogicalType::Map {
        key: Box::new(LogicalType::Utf8),
        value: Box::new(value_lt),
    })
}

/// Parse an Avro `fixed` → [`LogicalType::Binary`].
///
/// The fixed size is validated but not stored in the `LogicalType` since there
/// is no `Binary(size)` variant in helium-core. The size constraint would be
/// carried as a semantic hint once `ColumnSpec::meta` is available.
///
/// # TODO(meta)
/// Once `ColumnSpec::meta: Option<HashMap<String, String>>` is added to
/// `helium-core`, carry `{"fixed_size": "<N>"}` for downstream validation.
fn parse_fixed_type(
    obj: &serde_json::Map<String, serde_json::Value>,
    ctx: &mut ParseContext,
    logical_type: Option<&str>,
) -> Result<LogicalType> {
    // size is required by the Avro spec.
    obj.get("size")
        .and_then(|s| s.as_u64())
        .ok_or_else(|| HeliumError::Schema {
            column: "<root>".into(),
            reason: "Avro fixed missing required integer \"size\" field".into(),
        })?;

    // Register this fixed type under its name.
    if let Some(name) = obj.get("name").and_then(|n| n.as_str()) {
        ctx.named_types
            .insert(name.to_string(), serde_json::Value::Object(obj.clone()));
    }

    // Avro `decimal` logicalType on `fixed` → Decimal128 when precision is present.
    if matches!(logical_type, Some("decimal"))
        && let Some(precision) = obj.get("precision").and_then(|v| v.as_u64())
    {
        let precision = precision.min(38) as u8;
        let scale = obj.get("scale").and_then(|v| v.as_u64()).unwrap_or(0) as u8;
        return Ok(LogicalType::Decimal128 { precision, scale });
    }

    Ok(LogicalType::Binary)
}

// ---------------------------------------------------------------------------
// Union parser
// ---------------------------------------------------------------------------

/// Parse an Avro union (JSON array of type expressions).
///
/// Canonicalization rules:
/// * `["null"]` — rejected (no usable type).
/// * `[T]` (single non-null) — unwrap: return `parse(T)`.
/// * `["null", T]` / `[T, "null"]` — canonicalize to `Nullable { inner: T }`.
/// * `["null", A, B, …]` (null + multiple non-null) — produce
///   `Nullable { inner: Union { variants } }` where variants come from the
///   non-null entries.
/// * `[A, B, …]` (all non-null) — produce `Union { variants }`.
///
/// Nested unions (a union as an element of another union) are rejected per
/// the Avro specification.
///
/// Variant names follow the rules documented in [`avro_variant_name`].
fn parse_union(
    arr: &[serde_json::Value],
    ctx: &mut ParseContext,
    depth: usize,
) -> Result<LogicalType> {
    // Reject union-inside-union (Avro spec §Union).
    for item in arr {
        if item.is_array() {
            return Err(HeliumError::Schema {
                column: "<union>".into(),
                reason: "nested Avro unions are not permitted per the Avro specification".into(),
            });
        }
    }

    if arr.is_empty() {
        return Err(HeliumError::Schema {
            column: "<union>".into(),
            reason: "Avro union must contain at least one type".into(),
        });
    }

    // Single-element union.
    if arr.len() == 1 {
        if arr[0].as_str() == Some("null") {
            return Err(HeliumError::Schema {
                column: "<union>".into(),
                reason: "a union of only \"null\" has no usable Helium representation".into(),
            });
        }
        return parse_avro_type(&arr[0], ctx, depth);
    }

    // Separate null branches from non-null branches (preserving order of
    // non-null items for deterministic variant naming).
    let has_null = arr.iter().any(|v| v.as_str() == Some("null"));
    let non_null: Vec<&serde_json::Value> =
        arr.iter().filter(|v| v.as_str() != Some("null")).collect();

    // ["null", T] or [T, "null"] → Nullable(T).
    if has_null && non_null.len() == 1 {
        let inner = parse_avro_type(non_null[0], ctx, depth)?;
        return Ok(LogicalType::Nullable {
            inner: Box::new(inner),
        });
    }

    // Multiple non-null variants → Union (possibly wrapped in Nullable).
    let mut variants: Vec<(String, LogicalType)> = Vec::with_capacity(non_null.len());
    for item in &non_null {
        let name = avro_variant_name(item);
        let lt = parse_avro_type(item, ctx, depth)?;
        variants.push((name, lt));
    }

    let union_lt = LogicalType::Union { variants };

    if has_null {
        // ["null", A, B, …]: outer Nullable carries the null branch.
        Ok(LogicalType::Nullable {
            inner: Box::new(union_lt),
        })
    } else {
        Ok(union_lt)
    }
}

/// Derive a stable variant name for an Avro type appearing in a union.
///
/// Naming rule (in priority order):
/// 1. Named types (record, enum, fixed): use the type's `name` field.
/// 2. Primitive string keywords (`"int"`, `"string"`, etc.): use the keyword.
/// 3. Complex types without a `name` field: use the `type` keyword
///    (`"array"`, `"map"`).
/// 4. Fallback: `"unknown"`.
///
/// These names become **wire-format-frozen** variant identifiers in the
/// resulting Helium `Union` schema. Changing them produces an incompatible
/// schema (per `Union` semantics in `helium_core`).
fn avro_variant_name(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Object(obj) => {
            // Named type: use its declared name.
            if let Some(name) = obj.get("name").and_then(|n| n.as_str()) {
                return name.to_string();
            }
            // Unnamed complex type: use the type keyword.
            if let Some(type_kw) = obj.get("type").and_then(|t| t.as_str()) {
                return type_kw.to_string();
            }
            "unknown".to_string()
        }
        _ => "unknown".to_string(),
    }
}

// ---------------------------------------------------------------------------
// Avro Object Container File reader / writer
// ---------------------------------------------------------------------------

/// Read an Avro Object Container Format (`.avro`) file.
///
/// Returns the Helium [`Schema`] inferred from the file's embedded Avro schema,
/// and a [`HashMap`] of column name → [`LogicalColumn`] with the data for all
/// rows.
///
/// # Avro → Helium type mapping
///
/// | Avro type | Helium `LogicalType` |
/// |---|---|
/// | `record` | `Struct { fields }` |
/// | `array` | `List { inner }` |
/// | `map` | `Map { key: Utf8, value }` |
/// | `["null", T]` / `[T, "null"]` | `Nullable { inner: T }` |
/// | `["null", A, B, ...]` | `Nullable { inner: Union { variants } }` |
/// | `[A, B, ...]` (no null) | `Union { variants }` |
/// | `boolean` | `Primitive(U8)` |
/// | `int` / `long` | `Primitive(I32 / I64)` |
/// | `float` / `double` | `Primitive(F32 / F64)` |
/// | `bytes` / `fixed` | `Binary` |
/// | `string` / `enum` | `Utf8` |
/// | logical types | base physical type (metadata dropped) |
///
/// # Errors
///
/// Returns [`HeliumError::Format`] if the file cannot be read, the embedded
/// schema cannot be converted, or any record value fails type conversion.
pub fn read_avro_data<P: AsRef<Path>>(path: P) -> Result<(Schema, HashMap<String, LogicalColumn>)> {
    use apache_avro::Reader;
    use std::fs::File;

    let path = path.as_ref();
    let file = File::open(path).map_err(|e| {
        HeliumError::Format(format!("cannot open Avro file '{}': {e}", path.display()))
    })?;

    let reader = Reader::new(file).map_err(|e| {
        HeliumError::Format(format!(
            "cannot open Avro reader for '{}': {e}",
            path.display()
        ))
    })?;

    // Extract the writer schema embedded in the OCF header and convert it to a
    // Helium schema via the existing avsc parser.
    let avro_schema = reader.writer_schema().clone();
    let schema_json = avro_schema_to_json(&avro_schema);
    let helium_schema = avsc_to_schema(&schema_json)?;

    // Accumulate all records.
    let mut all_rows: Vec<apache_avro::types::Value> = Vec::new();
    for result in reader {
        let value =
            result.map_err(|e| HeliumError::Format(format!("error reading Avro record: {e}")))?;
        all_rows.push(value);
    }

    let columns = avro_rows_to_columns(&all_rows, &helium_schema)?;
    Ok((helium_schema, columns))
}

/// Read an Avro Object Container Format (`.avro`) file in chunks.
///
/// Invokes `on_chunk` once per chunk of up to `chunk_rows` records with a
/// `(schema, columns)` pair.  The schema is the same for every chunk; only the
/// column data differs.  This keeps peak memory proportional to
/// `chunk_rows × column_count` rather than the full file size.
///
/// The returned schema is the Helium schema derived from the Avro writer
/// schema embedded in the OCF header.
///
/// # Fallback
///
/// Unlike CSV/JSON, Avro already reads one record at a time internally (the
/// library buffers one OCF *block* ≈ a few KB–few MB, not the whole file),
/// so this function provides genuine streaming without an in-memory fallback.
///
/// # Errors
///
/// Returns [`HeliumError::Format`] on any I/O, schema-parse, or type-
/// conversion error.
pub fn read_avro_data_chunked<P, F>(path: P, chunk_rows: usize, mut on_chunk: F) -> Result<Schema>
where
    P: AsRef<Path>,
    F: FnMut(HashMap<String, LogicalColumn>) -> Result<()>,
{
    use apache_avro::Reader;
    use std::fs::File;

    let path = path.as_ref();
    let file = File::open(path).map_err(|e| {
        HeliumError::Format(format!("cannot open Avro file '{}': {e}", path.display()))
    })?;

    let reader = Reader::new(file).map_err(|e| {
        HeliumError::Format(format!(
            "cannot open Avro reader for '{}': {e}",
            path.display()
        ))
    })?;

    let avro_schema = reader.writer_schema().clone();
    let schema_json = avro_schema_to_json(&avro_schema);
    let helium_schema = avsc_to_schema(&schema_json)?;

    let mut chunk: Vec<apache_avro::types::Value> = Vec::with_capacity(chunk_rows);
    for result in reader {
        let value =
            result.map_err(|e| HeliumError::Format(format!("error reading Avro record: {e}")))?;
        chunk.push(value);
        if chunk.len() == chunk_rows {
            let columns = avro_rows_to_columns(&chunk, &helium_schema)?;
            on_chunk(columns)?;
            chunk = Vec::with_capacity(chunk_rows);
        }
    }
    // Flush final partial chunk.
    if !chunk.is_empty() {
        let columns = avro_rows_to_columns(&chunk, &helium_schema)?;
        on_chunk(columns)?;
    }

    Ok(helium_schema)
}

/// Write a Helium dataset to an Avro Object Container Format (`.avro`) file.
///
/// # Helium → Avro type mapping
///
/// | Helium `LogicalType` | Avro type |
/// |---|---|
/// | `Struct { fields }` | `record` |
/// | `List { inner }` | `array` |
/// | `Map { key: Utf8, value }` | `map` (non-Utf8 key is an error) |
/// | `Nullable { inner }` | `["null", inner_type]` union |
/// | `Union { variants }` | multi-variant union |
/// | `Primitive(U8)` | `boolean` |
/// | `Primitive(I32)` | `int` |
/// | `Primitive(I64 / U64)` | `long` |
/// | `Primitive(F32)` | `float` |
/// | `Primitive(F64)` | `double` |
/// | `Utf8` / `Dictionary { inner: Utf8 }` | `string` |
/// | `Binary` | `bytes` |
///
/// Compression: `Deflate` (matches the format-comparison report).
///
/// # Errors
///
/// Returns [`HeliumError::Format`] if a `Map` column has a non-`Utf8` key
/// type or if the output file cannot be created.
pub fn write_avro_data<P: AsRef<Path>>(
    path: P,
    schema: &Schema,
    columns: &HashMap<String, LogicalColumn>,
) -> Result<()> {
    use apache_avro::{Codec, DeflateSettings, Schema as AvroSchema, Writer as AvroWriter};
    use std::fs::File;

    let path = path.as_ref();

    let avsc_json = helium_schema_to_avsc_json(schema)?;
    let avro_schema = AvroSchema::parse_str(&avsc_json).map_err(|e| {
        HeliumError::Format(format!(
            "generated Avro schema is invalid: {e}\nschema JSON: {avsc_json}"
        ))
    })?;

    let file = File::create(path).map_err(|e| {
        HeliumError::Format(format!("cannot create Avro file '{}': {e}", path.display()))
    })?;

    let codec = Codec::Deflate(DeflateSettings::default());
    let mut writer = AvroWriter::with_codec(&avro_schema, file, codec);

    let row_count = column_row_count(columns, schema);

    for row_idx in 0..row_count {
        let mut record = apache_avro::types::Record::new(&avro_schema).ok_or_else(|| {
            HeliumError::Format(
                "Avro schema is not a record — cannot create Record value".to_string(),
            )
        })?;
        for col_spec in &schema.columns {
            let lc = columns.get(&col_spec.name).ok_or_else(|| {
                HeliumError::Format(format!("column '{}' missing from data map", col_spec.name))
            })?;
            let avro_val = logical_column_to_avro_value(lc, &col_spec.logical_type, row_idx)?;
            record.put(&col_spec.name, avro_val);
        }
        writer.append(record).map_err(|e| {
            HeliumError::Format(format!("error appending Avro record at row {row_idx}: {e}"))
        })?;
    }

    writer
        .into_inner()
        .map_err(|e| HeliumError::Format(format!("error flushing Avro writer: {e}")))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Internal: Avro schema → JSON (for avsc_to_schema)
// ---------------------------------------------------------------------------

fn avro_schema_to_json(avro_schema: &apache_avro::Schema) -> String {
    serde_json::to_string(avro_schema)
        .unwrap_or_else(|_| r#"{"type":"record","name":"unknown","fields":[]}"#.to_string())
}

// ---------------------------------------------------------------------------
// Internal: Avro Value rows → Helium LogicalColumn per column
// ---------------------------------------------------------------------------

fn avro_rows_to_columns(
    rows: &[apache_avro::types::Value],
    schema: &Schema,
) -> Result<HashMap<String, LogicalColumn>> {
    let mut result = HashMap::with_capacity(schema.columns.len());
    for col_spec in &schema.columns {
        let col_values: Vec<&apache_avro::types::Value> = rows
            .iter()
            .map(|row| project_avro_field(row, &col_spec.name))
            .collect();
        let lc = avro_values_to_logical_column(&col_values, &col_spec.logical_type).map_err(
            |e| match e {
                HeliumError::Format(msg) => {
                    HeliumError::Format(format!("column '{}': {msg}", col_spec.name))
                }
                other => other,
            },
        )?;
        result.insert(col_spec.name.clone(), lc);
    }
    Ok(result)
}

fn project_avro_field<'a>(
    row: &'a apache_avro::types::Value,
    field_name: &str,
) -> &'a apache_avro::types::Value {
    static NULL_VALUE: apache_avro::types::Value = apache_avro::types::Value::Null;
    match row {
        apache_avro::types::Value::Record(fields) => fields
            .iter()
            .find(|(k, _)| k == field_name)
            .map(|(_, v)| v)
            .unwrap_or(&NULL_VALUE),
        _ => &NULL_VALUE,
    }
}

fn avro_values_to_logical_column(
    values: &[&apache_avro::types::Value],
    lt: &LogicalType,
) -> Result<LogicalColumn> {
    use apache_avro::types::Value as AV;

    match lt {
        // ----------------------------------------------------------------
        // Primitive
        // ----------------------------------------------------------------
        LogicalType::Primitive { data_type } => {
            let col = match data_type {
                DataType::U8 => {
                    let v: Vec<u8> = values
                        .iter()
                        .enumerate()
                        .map(|(i, val)| avro_to_u8(val, i))
                        .collect::<Result<_>>()?;
                    ColumnData::U8(v)
                }
                DataType::I8 => {
                    let v: Vec<i8> = values
                        .iter()
                        .enumerate()
                        .map(|(i, val)| avro_to_i64(val, i).map(|x| x as i8))
                        .collect::<Result<_>>()?;
                    ColumnData::I8(v)
                }
                DataType::I16 => {
                    let v: Vec<i16> = values
                        .iter()
                        .enumerate()
                        .map(|(i, val)| avro_to_i64(val, i).map(|x| x as i16))
                        .collect::<Result<_>>()?;
                    ColumnData::I16(v)
                }
                DataType::I32 => {
                    let v: Vec<i32> = values
                        .iter()
                        .enumerate()
                        .map(|(i, val)| avro_to_i32(val, i))
                        .collect::<Result<_>>()?;
                    ColumnData::I32(v)
                }
                DataType::I64 => {
                    let v: Vec<i64> = values
                        .iter()
                        .enumerate()
                        .map(|(i, val)| avro_to_i64(val, i))
                        .collect::<Result<_>>()?;
                    ColumnData::I64(v)
                }
                DataType::U16 => {
                    let v: Vec<u16> = values
                        .iter()
                        .enumerate()
                        .map(|(i, val)| avro_to_i64(val, i).map(|x| x as u16))
                        .collect::<Result<_>>()?;
                    ColumnData::U16(v)
                }
                DataType::U32 => {
                    let v: Vec<u32> = values
                        .iter()
                        .enumerate()
                        .map(|(i, val)| avro_to_i64(val, i).map(|x| x as u32))
                        .collect::<Result<_>>()?;
                    ColumnData::U32(v)
                }
                DataType::U64 => {
                    let v: Vec<u64> = values
                        .iter()
                        .enumerate()
                        .map(|(i, val)| avro_to_i64(val, i).map(|x| x as u64))
                        .collect::<Result<_>>()?;
                    ColumnData::U64(v)
                }
                DataType::F32 => {
                    let v: Vec<f32> = values
                        .iter()
                        .enumerate()
                        .map(|(i, val)| avro_to_f32(val, i))
                        .collect::<Result<_>>()?;
                    ColumnData::F32(v)
                }
                DataType::F64 => {
                    let v: Vec<f64> = values
                        .iter()
                        .enumerate()
                        .map(|(i, val)| avro_to_f64(val, i))
                        .collect::<Result<_>>()?;
                    ColumnData::F64(v)
                }
                DataType::Bytes => {
                    return Err(HeliumError::Format(
                        "DataType::Bytes cannot be a Primitive logical type".to_string(),
                    ));
                }
            };
            Ok(LogicalColumn::Primitive(col))
        }

        // ----------------------------------------------------------------
        // Utf8
        // ----------------------------------------------------------------
        LogicalType::Utf8 => {
            let v: Vec<String> = values
                .iter()
                .enumerate()
                .map(|(i, val)| avro_to_string(val, i))
                .collect::<Result<_>>()?;
            Ok(LogicalColumn::Utf8(v))
        }

        // ----------------------------------------------------------------
        // Binary
        // ----------------------------------------------------------------
        LogicalType::Binary => {
            let v: Vec<Vec<u8>> = values
                .iter()
                .enumerate()
                .map(|(i, val)| avro_to_bytes(val, i))
                .collect::<Result<_>>()?;
            Ok(LogicalColumn::Binary(v))
        }

        // ----------------------------------------------------------------
        // Nullable
        // ----------------------------------------------------------------
        LogicalType::Nullable { inner } => {
            let mut present = Vec::with_capacity(values.len());
            let mut non_null: Vec<&AV> = Vec::new();
            for val in values {
                let is_null = matches!(val, AV::Null)
                    || matches!(val, AV::Union(_, iv) if matches!(iv.as_ref(), AV::Null));
                if is_null {
                    present.push(false);
                } else if let AV::Union(_, iv) = val {
                    present.push(true);
                    non_null.push(iv.as_ref());
                } else {
                    present.push(true);
                    non_null.push(val);
                }
            }
            let value = avro_values_to_logical_column(&non_null, inner)?;
            Ok(LogicalColumn::Nullable {
                present,
                value: Box::new(value),
            })
        }

        // ----------------------------------------------------------------
        // List
        // ----------------------------------------------------------------
        LogicalType::List { inner } => {
            let mut offsets: Vec<u32> = Vec::with_capacity(values.len() + 1);
            offsets.push(0);
            let mut flat_items: Vec<&AV> = Vec::new();
            for (i, val) in values.iter().enumerate() {
                match val {
                    AV::Array(arr) => {
                        flat_items.extend(arr.iter());
                        let len = u32::try_from(flat_items.len()).map_err(|_| {
                            HeliumError::Format(format!("row {i}: list offset overflows u32"))
                        })?;
                        offsets.push(len);
                    }
                    other => {
                        return Err(HeliumError::Format(format!(
                            "row {i}: expected Avro Array for List, got {:?}",
                            avro_value_kind(other)
                        )));
                    }
                }
            }
            let inner_col = avro_values_to_logical_column(&flat_items, inner)?;
            Ok(LogicalColumn::List {
                offsets,
                values: Box::new(inner_col),
            })
        }

        // ----------------------------------------------------------------
        // Map
        // ----------------------------------------------------------------
        LogicalType::Map { key, value } => {
            let mut offsets: Vec<u32> = Vec::with_capacity(values.len() + 1);
            offsets.push(0);
            let mut flat_keys: Vec<AV> = Vec::new();
            let mut flat_vals: Vec<&AV> = Vec::new();
            for (i, val) in values.iter().enumerate() {
                match val {
                    AV::Map(map) => {
                        for (k, v) in map {
                            flat_keys.push(AV::String(k.clone()));
                            flat_vals.push(v);
                        }
                        let len = u32::try_from(flat_keys.len()).map_err(|_| {
                            HeliumError::Format(format!("row {i}: map offset overflows u32"))
                        })?;
                        offsets.push(len);
                    }
                    other => {
                        return Err(HeliumError::Format(format!(
                            "row {i}: expected Avro Map, got {:?}",
                            avro_value_kind(other)
                        )));
                    }
                }
            }
            let flat_key_refs: Vec<&AV> = flat_keys.iter().collect();
            let keys_col = avro_values_to_logical_column(&flat_key_refs, key)?;
            let vals_col = avro_values_to_logical_column(&flat_vals, value)?;
            Ok(LogicalColumn::Map {
                offsets,
                keys: Box::new(keys_col),
                values: Box::new(vals_col),
            })
        }

        // ----------------------------------------------------------------
        // Struct
        // ----------------------------------------------------------------
        LogicalType::Struct { fields } => {
            let mut field_columns: Vec<(String, LogicalColumn)> = Vec::with_capacity(fields.len());
            for fspec in fields {
                let field_vals: Vec<&AV> = values
                    .iter()
                    .map(|row| project_avro_field(row, &fspec.name))
                    .collect();
                let fc = avro_values_to_logical_column(&field_vals, &fspec.logical_type).map_err(
                    |e| match e {
                        HeliumError::Format(msg) => {
                            HeliumError::Format(format!("struct field '{}': {msg}", fspec.name))
                        }
                        other => other,
                    },
                )?;
                field_columns.push((fspec.name.clone(), fc));
            }
            Ok(LogicalColumn::Struct {
                fields: field_columns,
            })
        }

        // ----------------------------------------------------------------
        // Union
        // ----------------------------------------------------------------
        LogicalType::Union { variants } => {
            let n = variants.len();
            let mut tags: Vec<u8> = Vec::with_capacity(values.len());
            let mut variant_bufs: Vec<Vec<&AV>> = vec![Vec::new(); n];

            for (row_i, val) in values.iter().enumerate() {
                let (tag, inner_val): (usize, &AV) = match val {
                    AV::Union(idx, inner) => (*idx as usize, inner.as_ref()),
                    AV::Null => (0, val),
                    other => {
                        let t = pick_union_variant_by_shape(other, variants, row_i)?;
                        (t, other)
                    }
                };
                if tag >= n {
                    return Err(HeliumError::Format(format!(
                        "row {row_i}: union variant index {tag} >= {n}"
                    )));
                }
                let tag_u8 = u8::try_from(tag).map_err(|_| {
                    HeliumError::Format(format!(
                        "row {row_i}: union variant index {tag} exceeds u8::MAX"
                    ))
                })?;
                tags.push(tag_u8);
                variant_bufs[tag].push(inner_val);
            }

            let mut result_variants = Vec::with_capacity(n);
            for (vi, (vname, vlt)) in variants.iter().enumerate() {
                let vc =
                    avro_values_to_logical_column(&variant_bufs[vi], vlt).map_err(|e| match e {
                        HeliumError::Format(msg) => {
                            HeliumError::Format(format!("union variant '{vname}': {msg}"))
                        }
                        other => other,
                    })?;
                result_variants.push((vname.clone(), vc));
            }
            Ok(LogicalColumn::Union {
                tags,
                variants: result_variants,
            })
        }

        // ----------------------------------------------------------------
        // Dictionary{inner} — recursive dictionary encoding
        // ----------------------------------------------------------------
        LogicalType::Dictionary { inner } => {
            // Read all logical values into the inner type, then
            // dictionary-encode them to produce the cardinality + indices.
            let all_lc = avro_values_to_logical_column(values, inner)?;
            // Use the UTF-8 fast path when inner is Utf8; otherwise materialize
            // via decompose/compose to build a general dictionary.
            match all_lc {
                LogicalColumn::Utf8(strings) => {
                    // dict_encode_utf8 now returns Dictionary{Utf8} directly.
                    Ok(LogicalColumn::dict_encode_utf8(strings))
                }
                other => {
                    // For primitive and other inner types, we collect rows and
                    // build a compact dictionary by walking the materialized column.
                    // Degenerate but correct: each distinct position gets its own slot.
                    let row_count = other.row_count();
                    let mut dict_rows: Vec<LogicalColumn> = Vec::new();
                    // We just pass all rows as the dictionary (no dedup) for the
                    // general case when called from Avro data path, since Avro
                    // enum/dict types always use Utf8 in practice.  For completeness,
                    // build a simple per-row identity mapping.
                    let indices: Vec<u32> = (0..row_count as u32).collect();
                    for i in 0..row_count {
                        dict_rows.push(other.slice(i, 1).map_err(|e| HeliumError::Schema {
                            column: "<dict>".into(),
                            reason: e.to_string(),
                        })?);
                    }
                    // Reconstruct a column from the per-row slices by simple
                    // concatenation for the dictionary.
                    let dict_inner = if dict_rows.is_empty() {
                        other
                    } else {
                        // Build dictionary as a concatenation of the first row's
                        // type template scaled to row_count. Simplest approach:
                        // use the original column as-is (it IS the dictionary).
                        other.clone()
                    };
                    Ok(LogicalColumn::Dictionary {
                        dictionary: Box::new(dict_inner),
                        indices,
                    })
                }
            }
        }

        // ----------------------------------------------------------------
        // Semantic types
        // ----------------------------------------------------------------
        LogicalType::Decimal128 { .. } => {
            // Avro Bytes / Fixed: big-endian two's-complement i128.
            // When read via OCF, the library may return AV::Decimal (logical-typed).
            let mut out = Vec::with_capacity(values.len());
            for (i, val) in values.iter().enumerate() {
                let v = match val {
                    AV::Bytes(b) | AV::Fixed(_, b) => avro_bytes_to_i128(b),
                    AV::Long(n) => *n as i128,
                    AV::Int(n) => *n as i128,
                    AV::Decimal(d) => {
                        // Convert via big-endian bytes representation.
                        let bytes: Vec<u8> = d.try_into().map_err(|e| {
                            HeliumError::Format(format!("row {i}: Avro Decimal→bytes failed: {e}"))
                        })?;
                        avro_bytes_to_i128(&bytes)
                    }
                    _ => {
                        return Err(HeliumError::Format(format!(
                            "row {i}: expected Avro Bytes or Long for Decimal128, got {:?}",
                            avro_value_kind(val)
                        )));
                    }
                };
                out.push(v);
            }
            Ok(LogicalColumn::Decimal128 { values: out })
        }
        LogicalType::Date {
            unit: DateUnit::Days,
        } => {
            let mut out = Vec::with_capacity(values.len());
            for (i, val) in values.iter().enumerate() {
                let v = match val {
                    AV::Int(n) => *n,
                    AV::Long(n) => *n as i32,
                    // Avro OCF library decodes date-typed fields as AV::Date(i32).
                    AV::Date(n) => *n,
                    _ => {
                        return Err(HeliumError::Format(format!(
                            "row {i}: expected Avro Int/Date for Date32, got {:?}",
                            avro_value_kind(val)
                        )));
                    }
                };
                out.push(v);
            }
            Ok(LogicalColumn::Date32 { values: out })
        }
        LogicalType::Date {
            unit: DateUnit::Millis,
        } => {
            let mut out = Vec::with_capacity(values.len());
            for (i, val) in values.iter().enumerate() {
                let v = match val {
                    AV::Long(n) => *n,
                    AV::Int(n) => *n as i64,
                    AV::Date(n) => *n as i64 * 86_400_000,
                    _ => {
                        return Err(HeliumError::Format(format!(
                            "row {i}: expected Avro Long for Date64, got {:?}",
                            avro_value_kind(val)
                        )));
                    }
                };
                out.push(v);
            }
            Ok(LogicalColumn::Date64 { values: out })
        }
        LogicalType::Datetime { unit, .. } => {
            let mut out = Vec::with_capacity(values.len());
            for (i, val) in values.iter().enumerate() {
                let v = match val {
                    AV::Long(n) => *n,
                    AV::Int(n) => *n as i64,
                    // Avro OCF library may use specific logical-type variants.
                    AV::TimestampMillis(n) => match unit {
                        TimeUnit::Millis => *n,
                        TimeUnit::Micros => *n * 1_000,
                        TimeUnit::Nanos => *n * 1_000_000,
                        TimeUnit::Seconds => *n / 1_000,
                    },
                    AV::TimestampMicros(n) => match unit {
                        TimeUnit::Millis => *n / 1_000,
                        TimeUnit::Micros => *n,
                        TimeUnit::Nanos => *n * 1_000,
                        TimeUnit::Seconds => *n / 1_000_000,
                    },
                    AV::TimestampNanos(n) => match unit {
                        TimeUnit::Millis => *n / 1_000_000,
                        TimeUnit::Micros => *n / 1_000,
                        TimeUnit::Nanos => *n,
                        TimeUnit::Seconds => *n / 1_000_000_000,
                    },
                    AV::LocalTimestampMillis(n)
                    | AV::LocalTimestampMicros(n)
                    | AV::LocalTimestampNanos(n) => *n,
                    _ => {
                        return Err(HeliumError::Format(format!(
                            "row {i}: expected Avro Long/Timestamp for Datetime, got {:?}",
                            avro_value_kind(val)
                        )));
                    }
                };
                out.push(v);
            }
            Ok(LogicalColumn::Datetime { values: out })
        }
    }
}

fn pick_union_variant_by_shape(
    val: &apache_avro::types::Value,
    variants: &[(String, LogicalType)],
    row_i: usize,
) -> Result<usize> {
    use apache_avro::types::Value as AV;
    let shape_matches = |lt: &LogicalType, v: &AV| -> bool {
        matches!(
            (lt, v),
            (
                LogicalType::Primitive {
                    data_type: DataType::U8
                },
                AV::Boolean(_)
            ) | (
                LogicalType::Primitive {
                    data_type: DataType::I32
                },
                AV::Int(_)
            ) | (LogicalType::Primitive { .. }, AV::Long(_))
                | (
                    LogicalType::Primitive {
                        data_type: DataType::F32
                    },
                    AV::Float(_)
                )
                | (
                    LogicalType::Primitive {
                        data_type: DataType::F64
                    },
                    AV::Double(_)
                )
                | (LogicalType::Utf8, AV::String(_) | AV::Enum(_, _))
                | (LogicalType::Binary, AV::Bytes(_) | AV::Fixed(_, _))
                | (LogicalType::Struct { .. }, AV::Record(_))
                | (LogicalType::List { .. }, AV::Array(_))
                | (LogicalType::Map { .. }, AV::Map(_))
        )
    };
    variants
        .iter()
        .position(|(_, lt)| shape_matches(lt, val))
        .ok_or_else(|| {
            HeliumError::Format(format!(
                "row {row_i}: no Union variant matches Avro value shape"
            ))
        })
}

// ---------------------------------------------------------------------------
// Avro value extraction helpers
// ---------------------------------------------------------------------------

/// Decode a big-endian two's-complement byte slice into an `i128`.
///
/// Truncates to 16 bytes if longer; sign-extends if shorter than 16 bytes.
fn avro_bytes_to_i128(b: &[u8]) -> i128 {
    let mut arr = [0u8; 16];
    let src = if b.len() > 16 { &b[b.len() - 16..] } else { b };
    // Sign-extend: fill leading bytes with 0xff if the MSB is set.
    if src.first().copied().unwrap_or(0) & 0x80 != 0 {
        arr = [0xff; 16];
    }
    let offset = 16 - src.len();
    arr[offset..].copy_from_slice(src);
    i128::from_be_bytes(arr)
}

/// Returns a short human-readable name for the variant of an `apache_avro` Value,
/// used in error messages to avoid calling `std::mem::discriminant` on a non-enum.
fn avro_value_kind(v: &apache_avro::types::Value) -> &'static str {
    use apache_avro::types::Value as AV;
    match v {
        AV::Null => "Null",
        AV::Boolean(_) => "Boolean",
        AV::Int(_) => "Int",
        AV::Long(_) => "Long",
        AV::Float(_) => "Float",
        AV::Double(_) => "Double",
        AV::Bytes(_) => "Bytes",
        AV::String(_) => "String",
        AV::Fixed(_, _) => "Fixed",
        AV::Enum(_, _) => "Enum",
        AV::Union(_, _) => "Union",
        AV::Array(_) => "Array",
        AV::Map(_) => "Map",
        AV::Record(_) => "Record",
        AV::Date(_) => "Date",
        AV::Decimal(_) => "Decimal",
        AV::TimeMillis(_) => "TimeMillis",
        AV::TimeMicros(_) => "TimeMicros",
        AV::TimestampMillis(_) => "TimestampMillis",
        AV::TimestampMicros(_) => "TimestampMicros",
        AV::LocalTimestampMillis(_) => "LocalTimestampMillis",
        AV::LocalTimestampMicros(_) => "LocalTimestampMicros",
        AV::Duration(_) => "Duration",
        AV::Uuid(_) => "Uuid",
        AV::BigDecimal(_) => "BigDecimal",
        AV::TimestampNanos(_) => "TimestampNanos",
        AV::LocalTimestampNanos(_) => "LocalTimestampNanos",
    }
}

fn avro_to_u8(val: &apache_avro::types::Value, row: usize) -> Result<u8> {
    use apache_avro::types::Value as AV;
    match val {
        AV::Boolean(b) => Ok(if *b { 1 } else { 0 }),
        AV::Int(n) => Ok(*n as u8),
        AV::Long(n) => Ok(*n as u8),
        AV::Enum(idx, _) => Ok(*idx as u8),
        other => Err(HeliumError::Format(format!(
            "row {row}: expected boolean/int for U8, got {other:?}"
        ))),
    }
}

fn avro_to_i32(val: &apache_avro::types::Value, row: usize) -> Result<i32> {
    use apache_avro::types::Value as AV;
    match val {
        AV::Int(n) => Ok(*n),
        AV::Long(n) => Ok(*n as i32),
        AV::Boolean(b) => Ok(if *b { 1 } else { 0 }),
        AV::Enum(idx, _) => Ok(*idx as i32),
        other => Err(HeliumError::Format(format!(
            "row {row}: expected int for I32, got {other:?}"
        ))),
    }
}

fn avro_to_i64(val: &apache_avro::types::Value, row: usize) -> Result<i64> {
    use apache_avro::types::Value as AV;
    match val {
        AV::Long(n) => Ok(*n),
        AV::Int(n) => Ok(*n as i64),
        AV::Boolean(b) => Ok(if *b { 1 } else { 0 }),
        AV::Enum(idx, _) => Ok(*idx as i64),
        other => Err(HeliumError::Format(format!(
            "row {row}: expected long/int for I64, got {other:?}"
        ))),
    }
}

fn avro_to_f32(val: &apache_avro::types::Value, row: usize) -> Result<f32> {
    use apache_avro::types::Value as AV;
    match val {
        AV::Float(f) => Ok(*f),
        AV::Double(d) => Ok(*d as f32),
        AV::Int(n) => Ok(*n as f32),
        AV::Long(n) => Ok(*n as f32),
        other => Err(HeliumError::Format(format!(
            "row {row}: expected float for F32, got {other:?}"
        ))),
    }
}

fn avro_to_f64(val: &apache_avro::types::Value, row: usize) -> Result<f64> {
    use apache_avro::types::Value as AV;
    match val {
        AV::Double(d) => Ok(*d),
        AV::Float(f) => Ok(*f as f64),
        AV::Int(n) => Ok(*n as f64),
        AV::Long(n) => Ok(*n as f64),
        other => Err(HeliumError::Format(format!(
            "row {row}: expected double/float for F64, got {other:?}"
        ))),
    }
}

fn avro_to_string(val: &apache_avro::types::Value, row: usize) -> Result<String> {
    use apache_avro::types::Value as AV;
    match val {
        AV::String(s) => Ok(s.clone()),
        AV::Enum(_, s) => Ok(s.clone()),
        other => Err(HeliumError::Format(format!(
            "row {row}: expected string/enum for Utf8, got {other:?}"
        ))),
    }
}

fn avro_to_bytes(val: &apache_avro::types::Value, row: usize) -> Result<Vec<u8>> {
    use apache_avro::types::Value as AV;
    match val {
        AV::Bytes(b) => Ok(b.clone()),
        AV::Fixed(_, b) => Ok(b.clone()),
        AV::String(s) => Ok(s.as_bytes().to_vec()),
        other => Err(HeliumError::Format(format!(
            "row {row}: expected bytes/fixed for Binary, got {other:?}"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Internal: Helium Schema → Avro JSON schema
// ---------------------------------------------------------------------------

fn helium_schema_to_avsc_json(schema: &Schema) -> Result<String> {
    let mut fields = Vec::with_capacity(schema.columns.len());
    for col in &schema.columns {
        let avro_type = helium_lt_to_avro_json(&col.logical_type, &col.name)?;
        fields.push(serde_json::json!({
            "name": col.name,
            "type": avro_type,
        }));
    }
    let record = serde_json::json!({
        "type": "record",
        "name": "HeliumRecord",
        "fields": fields,
    });
    serde_json::to_string(&record)
        .map_err(|e| HeliumError::Format(format!("cannot serialize Avro schema JSON: {e}")))
}

fn helium_lt_to_avro_json(lt: &LogicalType, context: &str) -> Result<serde_json::Value> {
    match lt {
        LogicalType::Primitive { data_type } => Ok(serde_json::Value::String(
            helium_dt_to_avro_type_name(*data_type).to_string(),
        )),
        LogicalType::Utf8 => Ok(serde_json::Value::String("string".into())),
        LogicalType::Binary => Ok(serde_json::Value::String("bytes".into())),
        LogicalType::Nullable { inner } => {
            let inner_json = helium_lt_to_avro_json(inner, context)?;
            Ok(serde_json::json!(["null", inner_json]))
        }
        LogicalType::List { inner } => {
            let items_json = helium_lt_to_avro_json(inner, context)?;
            Ok(serde_json::json!({ "type": "array", "items": items_json }))
        }
        LogicalType::Map { key, value } => {
            if !matches!(key.as_ref(), LogicalType::Utf8) {
                return Err(HeliumError::Format(format!(
                    "Avro maps must have string keys; column '{context}' has key type {key:?}"
                )));
            }
            let values_json = helium_lt_to_avro_json(value, context)?;
            Ok(serde_json::json!({ "type": "map", "values": values_json }))
        }
        LogicalType::Struct { fields } => {
            let mut avro_fields = Vec::with_capacity(fields.len());
            for fspec in fields {
                let ft = helium_lt_to_avro_json(&fspec.logical_type, &fspec.name)?;
                avro_fields.push(serde_json::json!({ "name": fspec.name, "type": ft }));
            }
            let safe_name: String = context
                .chars()
                .map(|c| {
                    if c.is_ascii_alphanumeric() || c == '_' {
                        c
                    } else {
                        '_'
                    }
                })
                .collect();
            Ok(serde_json::json!({
                "type": "record",
                "name": format!("{safe_name}_struct"),
                "fields": avro_fields,
            }))
        }
        LogicalType::Union { variants } => {
            let mut union_types = Vec::with_capacity(variants.len());
            for (name, vlt) in variants {
                union_types.push(helium_lt_to_avro_json(vlt, name)?);
            }
            Ok(serde_json::Value::Array(union_types))
        }
        // Dictionary{inner} — serialize as the inner type (expand on Avro round-trip).
        LogicalType::Dictionary { inner } => helium_lt_to_avro_json(inner, context),

        // Semantic types — emit Avro logical type annotations.
        LogicalType::Decimal128 { precision, scale } => Ok(serde_json::json!({
            "type": "bytes",
            "logicalType": "decimal",
            "precision": precision,
            "scale": scale,
        })),
        LogicalType::Date {
            unit: DateUnit::Days,
        } => {
            // Avro `date` logical type is always backed by `int` (days since epoch).
            Ok(serde_json::json!({ "type": "int", "logicalType": "date" }))
        }
        LogicalType::Date {
            unit: DateUnit::Millis,
        } => {
            // No direct Avro equivalent for millis-date; emit as `long` with a note.
            // This is a lossy mapping: the semantic hint is dropped on Avro round-trip.
            Ok(serde_json::Value::String("long".into()))
        }
        LogicalType::Datetime { unit, timezone: _ } => {
            // Avro timestamp logical types:
            // - Millis → "timestamp-millis" over `long`
            // - Micros → "timestamp-micros" over `long`
            // - Seconds / Nanos → no standard Avro equivalent; emit as bare `long`.
            // Timezone info is dropped — Avro only distinguishes UTC vs local via the
            // logical type name, not an explicit tz string.
            match unit {
                TimeUnit::Millis => {
                    Ok(serde_json::json!({ "type": "long", "logicalType": "timestamp-millis" }))
                }
                TimeUnit::Micros => {
                    Ok(serde_json::json!({ "type": "long", "logicalType": "timestamp-micros" }))
                }
                TimeUnit::Seconds | TimeUnit::Nanos => Ok(serde_json::Value::String("long".into())),
            }
        }
    }
}

fn helium_dt_to_avro_type_name(dt: DataType) -> &'static str {
    match dt {
        DataType::U8 => "boolean",
        DataType::I8 | DataType::I16 | DataType::I32 | DataType::U16 => "int",
        DataType::I64 | DataType::U32 | DataType::U64 => "long",
        DataType::F32 => "float",
        DataType::F64 => "double",
        DataType::Bytes => "bytes",
    }
}

// ---------------------------------------------------------------------------
// Internal: Helium LogicalColumn → Avro Value (row-oriented)
// ---------------------------------------------------------------------------

fn logical_column_to_avro_value(
    lc: &LogicalColumn,
    lt: &LogicalType,
    row_idx: usize,
) -> Result<apache_avro::types::Value> {
    use apache_avro::types::Value as AV;

    match (lc, lt) {
        (LogicalColumn::Primitive(cd), LogicalType::Primitive { data_type }) => {
            Ok(column_data_row_to_avro(cd, *data_type, row_idx))
        }
        (LogicalColumn::Utf8(v), LogicalType::Utf8) => {
            Ok(AV::String(v.get(row_idx).cloned().unwrap_or_default()))
        }
        (LogicalColumn::Binary(v), LogicalType::Binary) => {
            Ok(AV::Bytes(v.get(row_idx).cloned().unwrap_or_default()))
        }
        (LogicalColumn::Nullable { present, value }, LogicalType::Nullable { inner }) => {
            let compact_idx = present[..row_idx].iter().filter(|&&p| p).count();
            if present.get(row_idx).copied().unwrap_or(false) {
                let inner_val = logical_column_to_avro_value(value, inner, compact_idx)?;
                Ok(AV::Union(1, Box::new(inner_val)))
            } else {
                Ok(AV::Union(0, Box::new(AV::Null)))
            }
        }
        (LogicalColumn::List { offsets, values }, LogicalType::List { inner }) => {
            let start = offsets.get(row_idx).copied().unwrap_or(0) as usize;
            let end = offsets.get(row_idx + 1).copied().unwrap_or(start as u32) as usize;
            let mut items = Vec::with_capacity(end - start);
            for item_idx in start..end {
                items.push(logical_column_to_avro_value(values, inner, item_idx)?);
            }
            Ok(AV::Array(items))
        }
        (
            LogicalColumn::Map {
                offsets,
                keys,
                values,
            },
            LogicalType::Map { key, value },
        ) => {
            let start = offsets.get(row_idx).copied().unwrap_or(0) as usize;
            let end = offsets.get(row_idx + 1).copied().unwrap_or(start as u32) as usize;
            let mut map = std::collections::HashMap::new();
            for pair_idx in start..end {
                let k = avro_value_to_map_key(&logical_column_to_avro_value(keys, key, pair_idx)?)?;
                let v = logical_column_to_avro_value(values, value, pair_idx)?;
                map.insert(k, v);
            }
            Ok(AV::Map(map))
        }
        (LogicalColumn::Struct { fields }, LogicalType::Struct { fields: fspecs }) => {
            let mut record_fields = Vec::with_capacity(fields.len());
            for (fi, (fname, flc)) in fields.iter().enumerate() {
                let flt = fspecs.get(fi).map(|fs| &fs.logical_type).ok_or_else(|| {
                    HeliumError::Format(format!("struct field '{fname}' has no schema spec"))
                })?;
                let fval = logical_column_to_avro_value(flc, flt, row_idx)?;
                record_fields.push((fname.clone(), fval));
            }
            Ok(AV::Record(record_fields))
        }
        (LogicalColumn::Union { tags, variants }, LogicalType::Union { variants: vspecs }) => {
            let tag = tags.get(row_idx).copied().unwrap_or(0) as usize;
            let compact = tags[..row_idx]
                .iter()
                .filter(|&&t| t as usize == tag)
                .count();
            let (_, variant_lc) = variants.get(tag).ok_or_else(|| {
                HeliumError::Format(format!("row {row_idx}: union tag {tag} out of range"))
            })?;
            let vlt = &vspecs
                .get(tag)
                .ok_or_else(|| {
                    HeliumError::Format(format!(
                        "row {row_idx}: union tag {tag} has no schema spec"
                    ))
                })?
                .1;
            let inner_val = logical_column_to_avro_value(variant_lc, vlt, compact)?;
            Ok(AV::Union(tag as u32, Box::new(inner_val)))
        }
        // Semantic types
        (LogicalColumn::Decimal128 { values }, LogicalType::Decimal128 { .. }) => {
            // Represent as Avro Bytes (big-endian two's-complement i128).
            let v = values.get(row_idx).copied().unwrap_or(0i128);
            Ok(AV::Bytes(v.to_be_bytes().to_vec()))
        }
        (
            LogicalColumn::Date32 { values },
            LogicalType::Date {
                unit: DateUnit::Days,
            },
        ) => {
            let v = values.get(row_idx).copied().unwrap_or(0i32);
            Ok(AV::Int(v))
        }
        (
            LogicalColumn::Date64 { values },
            LogicalType::Date {
                unit: DateUnit::Millis,
            },
        ) => {
            let v = values.get(row_idx).copied().unwrap_or(0i64);
            Ok(AV::Long(v))
        }
        (LogicalColumn::Datetime { values }, LogicalType::Datetime { .. }) => {
            let v = values.get(row_idx).copied().unwrap_or(0i64);
            Ok(AV::Long(v))
        }
        (lc, lt) => Err(HeliumError::Format(format!(
            "cannot convert LogicalColumn to Avro for type {lt:?} (column variant: {lc:?})"
        ))),
    }
}

fn column_data_row_to_avro(cd: &ColumnData, dt: DataType, row: usize) -> apache_avro::types::Value {
    use apache_avro::types::Value as AV;
    match cd {
        ColumnData::U8(v) => match dt {
            DataType::U8 => AV::Boolean(*v.get(row).unwrap_or(&0) != 0),
            _ => AV::Int(*v.get(row).unwrap_or(&0) as i32),
        },
        ColumnData::I8(v) => AV::Int(*v.get(row).unwrap_or(&0) as i32),
        ColumnData::I16(v) => AV::Int(*v.get(row).unwrap_or(&0) as i32),
        ColumnData::I32(v) => AV::Int(*v.get(row).unwrap_or(&0)),
        ColumnData::I64(v) => AV::Long(*v.get(row).unwrap_or(&0)),
        ColumnData::U16(v) => AV::Int(*v.get(row).unwrap_or(&0) as i32),
        ColumnData::U32(v) => AV::Long(*v.get(row).unwrap_or(&0) as i64),
        ColumnData::U64(v) => AV::Long(*v.get(row).unwrap_or(&0) as i64),
        ColumnData::F32(v) => AV::Float(*v.get(row).unwrap_or(&0.0)),
        ColumnData::F64(v) => AV::Double(*v.get(row).unwrap_or(&0.0)),
        ColumnData::Bytes(v) => AV::Bytes(v.clone()),
    }
}

fn avro_value_to_map_key(val: &apache_avro::types::Value) -> Result<String> {
    use apache_avro::types::Value as AV;
    match val {
        AV::String(s) => Ok(s.clone()),
        AV::Enum(_, s) => Ok(s.clone()),
        other => Err(HeliumError::Format(format!(
            "map key must be a string, got {other:?}"
        ))),
    }
}

fn column_row_count(columns: &HashMap<String, LogicalColumn>, schema: &Schema) -> usize {
    let first_col = schema
        .columns
        .first()
        .and_then(|spec| columns.get(&spec.name));
    match first_col {
        None => 0,
        Some(lc) => logical_column_row_count(lc),
    }
}

fn logical_column_row_count(lc: &LogicalColumn) -> usize {
    match lc {
        LogicalColumn::Primitive(cd) => match cd {
            ColumnData::I8(v) => v.len(),
            ColumnData::I16(v) => v.len(),
            ColumnData::I32(v) => v.len(),
            ColumnData::I64(v) => v.len(),
            ColumnData::U8(v) => v.len(),
            ColumnData::U16(v) => v.len(),
            ColumnData::U32(v) => v.len(),
            ColumnData::U64(v) => v.len(),
            ColumnData::F32(v) => v.len(),
            ColumnData::F64(v) => v.len(),
            ColumnData::Bytes(v) => v.len(),
        },
        LogicalColumn::Utf8(v) => v.len(),
        LogicalColumn::Binary(v) => v.len(),
        LogicalColumn::List { offsets, .. } | LogicalColumn::Map { offsets, .. } => {
            offsets.len().saturating_sub(1)
        }
        LogicalColumn::Dictionary { indices, .. } => indices.len(),
        LogicalColumn::Nullable { present, .. } => present.len(),
        LogicalColumn::Struct { fields } => fields
            .first()
            .map(|(_, lc)| logical_column_row_count(lc))
            .unwrap_or(0),
        LogicalColumn::Union { tags, .. } => tags.len(),
        LogicalColumn::Decimal128 { values } => values.len(),
        LogicalColumn::Date32 { values } => values.len(),
        LogicalColumn::Date64 { values } => values.len(),
        LogicalColumn::Datetime { values } => values.len(),
    }
}

// ---------------------------------------------------------------------------
// Unit tests for read_avro_data / write_avro_data
// ---------------------------------------------------------------------------

#[cfg(test)]
mod data_tests {
    use super::*;
    use std::collections::HashMap;
    use tempfile::NamedTempFile;

    fn mk_schema(columns: Vec<ColumnSpec>) -> Schema {
        Schema::new(columns)
    }

    // 1. Round-trip flat record: int + string
    #[test]
    fn roundtrip_flat_record() {
        let lt_id = LogicalType::Primitive {
            data_type: DataType::I32,
        };
        let lt_label = LogicalType::Utf8;
        let schema = mk_schema(vec![
            ColumnSpec::new("id", lt_id.clone(), default_encodings(&lt_id)),
            ColumnSpec::new("label", lt_label.clone(), default_encodings(&lt_label)),
        ]);

        let mut cols: HashMap<String, LogicalColumn> = HashMap::new();
        cols.insert(
            "id".into(),
            LogicalColumn::Primitive(ColumnData::I32(vec![1, 2, 3, 4, 5])),
        );
        cols.insert(
            "label".into(),
            LogicalColumn::Utf8(vec![
                "alpha".into(),
                "beta".into(),
                "gamma".into(),
                "delta".into(),
                "epsilon".into(),
            ]),
        );

        let tmp = NamedTempFile::new().unwrap();
        write_avro_data(tmp.path(), &schema, &cols).expect("write_avro_data failed");

        let (schema2, cols2) = read_avro_data(tmp.path()).expect("read_avro_data failed");
        assert_eq!(schema2.columns.len(), 2);

        let ids = match &cols2["id"] {
            LogicalColumn::Primitive(ColumnData::I32(v)) => v.clone(),
            other => panic!("expected I32, got {other:?}"),
        };
        assert_eq!(ids, vec![1, 2, 3, 4, 5]);

        let labels = match &cols2["label"] {
            LogicalColumn::Utf8(v) => v.clone(),
            other => panic!("expected Utf8, got {other:?}"),
        };
        assert_eq!(labels, vec!["alpha", "beta", "gamma", "delta", "epsilon"]);
    }

    // 2. Round-trip nullable: Nullable<I64>
    #[test]
    fn roundtrip_nullable_i64() {
        let lt_inner = LogicalType::Primitive {
            data_type: DataType::I64,
        };
        let lt = LogicalType::Nullable {
            inner: Box::new(lt_inner),
        };
        let schema = mk_schema(vec![ColumnSpec::new(
            "maybe_long",
            lt.clone(),
            default_encodings(&lt),
        )]);

        let present = vec![true, false, true, true, false];
        let mut cols: HashMap<String, LogicalColumn> = HashMap::new();
        cols.insert(
            "maybe_long".into(),
            LogicalColumn::Nullable {
                present: present.clone(),
                value: Box::new(LogicalColumn::Primitive(ColumnData::I64(vec![10, 30, 40]))),
            },
        );

        let tmp = NamedTempFile::new().unwrap();
        write_avro_data(tmp.path(), &schema, &cols).expect("write failed");

        let (_schema2, cols2) = read_avro_data(tmp.path()).expect("read failed");
        match &cols2["maybe_long"] {
            LogicalColumn::Nullable { present: p2, value } => {
                assert_eq!(*p2, present);
                match value.as_ref() {
                    LogicalColumn::Primitive(ColumnData::I64(v)) => {
                        assert_eq!(*v, vec![10, 30, 40]);
                    }
                    other => panic!("expected I64 values, got {other:?}"),
                }
            }
            other => panic!("expected Nullable, got {other:?}"),
        }
    }

    // 3. Round-trip nested record: Struct { id: I64, address: Struct { city, zip } }
    #[test]
    fn roundtrip_nested_struct() {
        let lt_id = LogicalType::Primitive {
            data_type: DataType::I64,
        };
        let lt_city = LogicalType::Utf8;
        let lt_zip = LogicalType::Utf8;
        let lt_addr = LogicalType::Struct {
            fields: vec![
                FieldSpec::new("city", lt_city.clone(), default_encodings(&lt_city)),
                FieldSpec::new("zip", lt_zip.clone(), default_encodings(&lt_zip)),
            ],
        };
        let schema = mk_schema(vec![
            ColumnSpec::new("id", lt_id.clone(), default_encodings(&lt_id)),
            ColumnSpec::new("address", lt_addr.clone(), default_encodings(&lt_addr)),
        ]);

        let mut cols: HashMap<String, LogicalColumn> = HashMap::new();
        cols.insert(
            "id".into(),
            LogicalColumn::Primitive(ColumnData::I64(vec![100, 200])),
        );
        cols.insert(
            "address".into(),
            LogicalColumn::Struct {
                fields: vec![
                    (
                        "city".into(),
                        LogicalColumn::Utf8(vec!["Paris".into(), "Berlin".into()]),
                    ),
                    (
                        "zip".into(),
                        LogicalColumn::Utf8(vec!["75001".into(), "10115".into()]),
                    ),
                ],
            },
        );

        let tmp = NamedTempFile::new().unwrap();
        write_avro_data(tmp.path(), &schema, &cols).expect("write failed");

        let (_schema2, cols2) = read_avro_data(tmp.path()).expect("read failed");
        match &cols2["address"] {
            LogicalColumn::Struct { fields } => {
                assert_eq!(fields.len(), 2);
                assert_eq!(fields[0].0, "city");
                assert!(
                    matches!(&fields[0].1, LogicalColumn::Utf8(v) if *v == vec!["Paris", "Berlin"])
                );
                assert_eq!(fields[1].0, "zip");
                assert!(
                    matches!(&fields[1].1, LogicalColumn::Utf8(v) if *v == vec!["75001", "10115"])
                );
            }
            other => panic!("expected Struct, got {other:?}"),
        }
    }

    // 4. Round-trip array: List<I64>
    #[test]
    fn roundtrip_list_i64() {
        let lt_inner = LogicalType::Primitive {
            data_type: DataType::I64,
        };
        let lt = LogicalType::List {
            inner: Box::new(lt_inner),
        };
        let schema = mk_schema(vec![ColumnSpec::new(
            "nums",
            lt.clone(),
            default_encodings(&lt),
        )]);

        let mut cols: HashMap<String, LogicalColumn> = HashMap::new();
        cols.insert(
            "nums".into(),
            LogicalColumn::List {
                offsets: vec![0, 3, 3, 5],
                values: Box::new(LogicalColumn::Primitive(ColumnData::I64(vec![
                    10, 20, 30, 40, 50,
                ]))),
            },
        );

        let tmp = NamedTempFile::new().unwrap();
        write_avro_data(tmp.path(), &schema, &cols).expect("write failed");

        let (_schema2, cols2) = read_avro_data(tmp.path()).expect("read failed");
        match &cols2["nums"] {
            LogicalColumn::List { offsets, values } => {
                assert_eq!(*offsets, vec![0, 3, 3, 5]);
                assert!(
                    matches!(values.as_ref(), LogicalColumn::Primitive(ColumnData::I64(v)) if *v == vec![10, 20, 30, 40, 50])
                );
            }
            other => panic!("expected List, got {other:?}"),
        }
    }

    // 5. Round-trip map: Map<Utf8, I64>
    #[test]
    fn roundtrip_map_utf8_i64() {
        let lt = LogicalType::Map {
            key: Box::new(LogicalType::Utf8),
            value: Box::new(LogicalType::Primitive {
                data_type: DataType::I64,
            }),
        };
        let schema = mk_schema(vec![ColumnSpec::new(
            "scores",
            lt.clone(),
            default_encodings(&lt),
        )]);

        // 2 rows: {"a": 1, "b": 2} and {"c": 3}
        let mut cols: HashMap<String, LogicalColumn> = HashMap::new();
        cols.insert(
            "scores".into(),
            LogicalColumn::Map {
                offsets: vec![0, 2, 3],
                keys: Box::new(LogicalColumn::Utf8(vec![
                    "a".into(),
                    "b".into(),
                    "c".into(),
                ])),
                values: Box::new(LogicalColumn::Primitive(ColumnData::I64(vec![1, 2, 3]))),
            },
        );

        let tmp = NamedTempFile::new().unwrap();
        write_avro_data(tmp.path(), &schema, &cols).expect("write failed");

        let (_schema2, cols2) = read_avro_data(tmp.path()).expect("read failed");
        match &cols2["scores"] {
            LogicalColumn::Map {
                offsets,
                keys,
                values,
            } => {
                assert_eq!(offsets.len(), 3);
                assert_eq!(offsets[0], 0);
                assert_eq!(offsets[2], 3);
                assert!(matches!(keys.as_ref(), LogicalColumn::Utf8(_)));
                assert!(matches!(
                    values.as_ref(),
                    LogicalColumn::Primitive(ColumnData::I64(_))
                ));
            }
            other => panic!("expected Map, got {other:?}"),
        }
    }
}

// Encoding helpers are now in the shared `crate::encodings` module.
