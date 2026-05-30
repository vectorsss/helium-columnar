//! JSON / NDJSON → Helium [`crate::Schema`] type inferrer.
//!
//! # Feature flag: `json`
//!
//! ```toml
//! helium-schema = { features = ["json"] }
//! ```
//!
//! # Input format
//!
//! Accepts either:
//! * A JSON **array** of objects: `[{...}, {...}, ...]`
//! * **NDJSON** (newline-delimited JSON): one object per line
//!
//! All top-level values must be JSON objects; other root shapes are rejected.
//!
//! # Type inference
//!
//! For each field in the sampled objects:
//!
//! | Observed JSON value(s) | Inferred [`crate::LogicalType`] |
//! |---|---|
//! | Integer numbers only | `Primitive(I64)` |
//! | Float numbers only | `Primitive(F64)` |
//! | Boolean values only | `Primitive(U8)` (0 / 1) |
//! | Integer + Boolean | `Primitive(I64)` (bool promotes to int) |
//! | Integer + Float | `Primitive(F64)` (int promotes to float) |
//! | Strings only | `Utf8` |
//! | Objects only | `Struct { fields }` (recursive) |
//! | Arrays only | `List { inner }` (element type merged across all rows) |
//! | `null` + one other type | `Nullable { inner: T }` |
//! | 2–3 incompatible non-null types | `Union { variants }` |
//! | ≥ 4 incompatible non-null types | `Utf8` (fallback) |
//! | Only `null` values | `Utf8` (no usable type information) |
//!
//! If a field is **absent** in some records it is considered nullable.
//!
//! # Limitations
//!
//! * Nested types are inferred recursively but depth is capped at
//!   16 levels (`JSON_MAX_DEPTH`) to avoid unbounded recursion.
//! * Arrays of mixed or heterogeneous types fall back to `List<Utf8>`.
//! * Union variant names are fixed: `"bool"`, `"int"`, `"float"`, `"string"`,
//!   `"object"`, `"array"` — these become wire-frozen once written to a `.he`
//!   file (per Helium Union semantics).

use std::collections::HashMap;
use std::path::Path;

use crate::{ColumnSpec, DataType, FieldSpec, HeliumError, LogicalType, Result, Schema};

use super::encodings::default_encodings;

/// Maximum JSON nesting depth for type inference.
///
/// Values deeper than this are treated as `Utf8` (raw JSON string).
/// The value 16 limits recursion during type inference to avoid unbounded
/// stack usage on pathologically nested input.
const JSON_MAX_DEPTH: usize = 16;

/// Maximum number of distinct non-null variant types before a mixed-type
/// field falls back from `Union` to `Utf8`.
const DEFAULT_MAX_UNION_VARIANTS: usize = 3;

// ---------------------------------------------------------------------------
// Public options
// ---------------------------------------------------------------------------

/// Options controlling JSON schema inference.
#[derive(Debug, Clone)]
pub struct JsonInferOptions {
    /// Maximum number of records (objects) to sample. Default: 1 000.
    pub max_records: usize,
    /// Maximum number of distinct non-null variant types in a field before
    /// falling back from [`LogicalType::Union`] to [`LogicalType::Utf8`].
    /// Default: 3.
    pub max_union_variants: usize,
}

impl Default for JsonInferOptions {
    fn default() -> Self {
        Self {
            max_records: 1_000,
            max_union_variants: DEFAULT_MAX_UNION_VARIANTS,
        }
    }
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Infer a Helium [`Schema`] from a JSON or NDJSON file at `path`.
pub fn schema_from_json(path: &Path) -> Result<Schema> {
    let content = std::fs::read_to_string(path).map_err(HeliumError::Io)?;
    schema_from_json_str_with_options(&content, &JsonInferOptions::default())
}

/// Infer a Helium [`Schema`] from a JSON or NDJSON file with custom options.
pub fn schema_from_json_with_options(path: &Path, opts: &JsonInferOptions) -> Result<Schema> {
    let content = std::fs::read_to_string(path).map_err(HeliumError::Io)?;
    schema_from_json_str_with_options(&content, opts)
}

/// Infer a Helium [`Schema`] from an in-memory JSON or NDJSON string.
pub fn schema_from_json_str(json: &str) -> Result<Schema> {
    schema_from_json_str_with_options(json, &JsonInferOptions::default())
}

/// Infer a Helium [`Schema`] from an in-memory JSON or NDJSON string with custom options.
pub fn schema_from_json_str_with_options(json: &str, opts: &JsonInferOptions) -> Result<Schema> {
    let records = parse_json_records(json, opts.max_records)?;
    if records.is_empty() {
        return Err(HeliumError::Schema {
            column: "<json>".into(),
            reason: "JSON input has no objects to infer a schema from".into(),
        });
    }
    let columns = infer_columns_from_records(&records, opts);
    let schema = Schema::new(columns);
    schema.validate()?;
    Ok(schema)
}

// ---------------------------------------------------------------------------
// Record parsing
// ---------------------------------------------------------------------------

/// Parse JSON input into a list of top-level objects.
///
/// Accepts a JSON array of objects or NDJSON (one object per line).
fn parse_json_records(
    input: &str,
    max_records: usize,
) -> Result<Vec<serde_json::Map<String, serde_json::Value>>> {
    let input = input.trim();

    // Try JSON array first.
    if input.starts_with('[') {
        let arr: serde_json::Value = serde_json::from_str(input)
            .map_err(|e| HeliumError::Format(format!("JSON parse error: {e}")))?;
        return match arr {
            serde_json::Value::Array(items) => items
                .into_iter()
                .take(max_records)
                .map(|v| match v {
                    serde_json::Value::Object(m) => Ok(m),
                    _ => Err(HeliumError::Schema {
                        column: "<json>".into(),
                        reason: "JSON array must contain only objects".into(),
                    }),
                })
                .collect(),
            _ => Err(HeliumError::Schema {
                column: "<json>".into(),
                reason: "top-level JSON must be an array of objects or NDJSON".into(),
            }),
        };
    }

    // Fall back to NDJSON.
    let mut records = Vec::new();
    for (line_no, line) in input.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(line).map_err(|e| {
            HeliumError::Format(format!("NDJSON parse error at line {}: {e}", line_no + 1))
        })?;
        match v {
            serde_json::Value::Object(m) => records.push(m),
            _ => {
                return Err(HeliumError::Schema {
                    column: "<json>".into(),
                    reason: format!("NDJSON line {} is not a JSON object", line_no + 1),
                });
            }
        }
        if records.len() >= max_records {
            break;
        }
    }
    Ok(records)
}

// ---------------------------------------------------------------------------
// Type inference internals
// ---------------------------------------------------------------------------

/// Which basic JSON value kinds have been seen for one field.
///
/// Objects and arrays are tracked separately because merging their inner
/// structure requires recursive inference. `ArrAcc` is `Box`ed to break the
/// recursive type cycle (`FieldAcc → ArrAcc → FieldAcc`).
#[derive(Debug, Default)]
struct FieldAcc {
    seen_bool: bool,
    seen_int: bool,
    seen_float: bool,
    seen_str: bool,
    seen_null: bool,
    obj_acc: Option<ObjAcc>,
    arr_acc: Option<Box<ArrAcc>>,
    /// Number of records in which this field appeared (used to detect
    /// fields that are absent in some records → nullable).
    record_count: usize,
}

/// Accumulator for nested JSON object fields.
#[derive(Debug, Default)]
struct ObjAcc {
    field_order: Vec<String>,
    fields: HashMap<String, FieldAcc>,
    total_records: usize,
}

/// Accumulator for JSON array element types.
///
/// `inner` is `Box`ed to break the recursive cycle `FieldAcc → ArrAcc → FieldAcc`.
#[derive(Debug, Default)]
struct ArrAcc {
    inner: Box<FieldAcc>,
}

impl FieldAcc {
    fn observe(&mut self, v: &serde_json::Value, depth: usize) {
        match v {
            serde_json::Value::Null => self.seen_null = true,
            serde_json::Value::Bool(_) => self.seen_bool = true,
            serde_json::Value::Number(n) => {
                if n.is_i64() || n.is_u64() {
                    self.seen_int = true;
                } else {
                    self.seen_float = true;
                }
            }
            serde_json::Value::String(_) => self.seen_str = true,
            serde_json::Value::Object(obj) => {
                if depth < JSON_MAX_DEPTH {
                    let acc = self.obj_acc.get_or_insert_with(ObjAcc::default);
                    acc.total_records += 1;
                    for (k, v) in obj {
                        if acc.field_order.iter().all(|n| n != k) {
                            // New field: record it as nullable because earlier records didn't have it.
                            acc.field_order.push(k.clone());
                        }
                        // field_order and fields are always kept in sync: or_default is a no-op
                        // for existing fields.
                        let entry = acc.fields.entry(k.clone()).or_default();
                        entry.observe(v, depth + 1);
                        entry.record_count += 1;
                    }
                    // Any field NOT present in this record is implicitly nullable.
                }
            }
            serde_json::Value::Array(arr) => {
                if depth < JSON_MAX_DEPTH {
                    let acc = self
                        .arr_acc
                        .get_or_insert_with(|| Box::new(ArrAcc::default()));
                    for elem in arr {
                        acc.inner.observe(elem, depth + 1);
                    }
                }
            }
        }
    }

    /// Convert the accumulated information to a [`LogicalType`].
    fn to_logical_type(&self, total_records: usize, max_union_variants: usize) -> LogicalType {
        let has_null = self.seen_null || (total_records > 0 && self.record_count < total_records);

        // Collect distinct non-null "kinds".
        let has_obj = self.obj_acc.is_some();
        let has_arr = self.arr_acc.is_some();

        let scalar_count = usize::from(self.seen_bool)
            + usize::from(self.seen_int)
            + usize::from(self.seen_float)
            + usize::from(self.seen_str);

        let total_kinds = scalar_count + usize::from(has_obj) + usize::from(has_arr);

        let leaf = if total_kinds == 0 {
            // Only nulls seen.
            LogicalType::Utf8
        } else if total_kinds == 1 {
            // Single non-null type.
            if has_obj {
                if let Some(acc) = self.obj_acc.as_ref() {
                    obj_acc_to_struct(acc, max_union_variants)
                } else {
                    LogicalType::Utf8
                }
            } else if has_arr {
                if let Some(acc) = self.arr_acc.as_ref() {
                    let inner = acc.inner.to_logical_type(0, max_union_variants);
                    LogicalType::List {
                        inner: Box::new(inner),
                    }
                } else {
                    LogicalType::Utf8
                }
            } else {
                single_scalar_type(
                    self.seen_bool,
                    self.seen_int,
                    self.seen_float,
                    self.seen_str,
                )
            }
        } else if total_kinds <= max_union_variants {
            // 2–N compatible or union types.
            // Try numeric promotions first.
            let all_numeric = !self.seen_str && !has_obj && !has_arr;
            if all_numeric {
                // bool → int, int → float promotions.
                if self.seen_float || (self.seen_bool && self.seen_int) {
                    // bool+int+float, or just float, or bool+float: promote to F64.
                    // If only bool+int (no float): promote to I64.
                    if self.seen_float {
                        LogicalType::Primitive {
                            data_type: DataType::F64,
                        }
                    } else {
                        // bool + int only
                        LogicalType::Primitive {
                            data_type: DataType::I64,
                        }
                    }
                } else if self.seen_bool && !self.seen_int {
                    LogicalType::Primitive {
                        data_type: DataType::U8,
                    }
                } else {
                    LogicalType::Primitive {
                        data_type: DataType::I64,
                    }
                }
            } else {
                // Incompatible types → Union (variant count ≤ max).
                build_union(self, max_union_variants)
            }
        } else {
            // Too many variants → fall back to Utf8.
            LogicalType::Utf8
        };

        if has_null {
            LogicalType::Nullable {
                inner: Box::new(leaf),
            }
        } else {
            leaf
        }
    }
}

fn single_scalar_type(
    seen_bool: bool,
    seen_int: bool,
    seen_float: bool,
    seen_str: bool,
) -> LogicalType {
    if seen_str {
        LogicalType::Utf8
    } else if seen_float {
        LogicalType::Primitive {
            data_type: DataType::F64,
        }
    } else if seen_int {
        LogicalType::Primitive {
            data_type: DataType::I64,
        }
    } else if seen_bool {
        LogicalType::Primitive {
            data_type: DataType::U8,
        }
    } else {
        LogicalType::Utf8
    }
}

fn obj_acc_to_struct(acc: &ObjAcc, max_union_variants: usize) -> LogicalType {
    let fields: Vec<FieldSpec> = acc
        .field_order
        .iter()
        .map(|name| {
            let field_acc = &acc.fields[name];
            let lt = field_acc.to_logical_type(acc.total_records, max_union_variants);
            let enc = default_encodings(&lt);
            FieldSpec::new(name.clone(), lt, enc)
        })
        .collect();
    LogicalType::Struct { fields }
}

/// Build a Union from a multi-kind [`FieldAcc`].
///
/// Variant names use stable identifiers that become wire-frozen in the Helium schema.
fn build_union(acc: &FieldAcc, max_union_variants: usize) -> LogicalType {
    let mut variants: Vec<(String, LogicalType)> = Vec::new();
    if acc.seen_bool {
        variants.push((
            "bool".into(),
            LogicalType::Primitive {
                data_type: DataType::U8,
            },
        ));
    }
    if acc.seen_int {
        variants.push((
            "int".into(),
            LogicalType::Primitive {
                data_type: DataType::I64,
            },
        ));
    }
    if acc.seen_float {
        variants.push((
            "float".into(),
            LogicalType::Primitive {
                data_type: DataType::F64,
            },
        ));
    }
    if acc.seen_str {
        variants.push(("string".into(), LogicalType::Utf8));
    }
    if let Some(obj) = &acc.obj_acc {
        variants.push(("object".into(), obj_acc_to_struct(obj, max_union_variants)));
    }
    if let Some(arr) = &acc.arr_acc {
        let inner = arr.inner.to_logical_type(0, max_union_variants);
        variants.push((
            "array".into(),
            LogicalType::List {
                inner: Box::new(inner),
            },
        ));
    }
    if variants.len() > max_union_variants {
        return LogicalType::Utf8;
    }
    LogicalType::Union { variants }
}

// ---------------------------------------------------------------------------
// Top-level column accumulation
// ---------------------------------------------------------------------------

fn infer_columns_from_records(
    records: &[serde_json::Map<String, serde_json::Value>],
    opts: &JsonInferOptions,
) -> Vec<ColumnSpec> {
    let total = records.len();

    // Accumulate field info in field_order + fields map.
    let mut field_order: Vec<String> = Vec::new();
    let mut fields: HashMap<String, FieldAcc> = HashMap::new();

    for record in records {
        for (key, value) in record {
            if field_order.iter().all(|n| n != key) {
                field_order.push(key.clone());
            }
            // field_order and fields are always kept in sync; or_default is a no-op for
            // existing fields.
            let entry = fields.entry(key.clone()).or_default();
            entry.observe(value, 0);
            entry.record_count += 1;
        }
    }

    field_order
        .iter()
        .map(|name| {
            let acc = &fields[name];
            let lt = acc.to_logical_type(total, opts.max_union_variants);
            let enc = default_encodings(&lt);
            ColumnSpec::new(name.clone(), lt, enc)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// JSON / NDJSON writer
// ---------------------------------------------------------------------------

/// Write a Helium schema + columns to NDJSON format (one JSON object per line).
///
/// Each row is emitted as `{"col1": value, "col2": value, ...}\n`.
/// Column order matches `schema.columns`.
///
/// ## Value encoding
///
/// | Helium type | JSON form |
/// |---|---|
/// | `Primitive(int)` | JSON number |
/// | `Primitive(float)` | JSON number (NaN → `null`) |
/// | `Utf8` | JSON string |
/// | `Binary` | JSON string (lowercase hex) |
/// | `Nullable<T>` | `null` or T's form |
/// | `Struct` | JSON object |
/// | `List<T>` | JSON array |
/// | `Map<K,V>` | JSON object (keys coerced to string) |
/// | `Union` | `{"variant_name": value}` |
///
/// # Errors
///
/// Returns [`HeliumError::Schema`] if a column in `schema` is missing from
/// `columns` or if row counts are inconsistent.
pub fn write_json<W: std::io::Write>(
    schema: &Schema,
    columns: &std::collections::HashMap<String, crate::LogicalColumn>,
    mut writer: W,
) -> crate::Result<()> {
    use crate::HeliumError;
    use std::io::Write as IoWrite;

    // Resolve columns in schema order.
    let cols: Vec<(&str, &crate::LogicalColumn)> = schema
        .columns
        .iter()
        .map(|spec| {
            let lc = columns.get(&spec.name).ok_or_else(|| HeliumError::Schema {
                column: spec.name.clone(),
                reason: "column present in schema but missing from data map".into(),
            })?;
            Ok((spec.name.as_str(), lc))
        })
        .collect::<crate::Result<_>>()?;

    if cols.is_empty() {
        return Ok(());
    }

    let row_count = cols[0].1.row_count();
    for (name, lc) in &cols {
        if lc.row_count() != row_count {
            return Err(HeliumError::Schema {
                column: (*name).to_string(),
                reason: format!(
                    "row count mismatch: expected {row_count}, got {}",
                    lc.row_count()
                ),
            });
        }
    }

    for row in 0..row_count {
        let mut obj = serde_json::Map::new();
        for (name, lc) in &cols {
            let val = logical_column_row_to_json_value(lc, row).unwrap_or(serde_json::Value::Null);
            obj.insert((*name).to_string(), val);
        }
        let line = serde_json::to_string(&serde_json::Value::Object(obj))
            .map_err(|e| HeliumError::Format(format!("JSON serialisation error: {e}")))?;
        IoWrite::write_all(&mut writer, line.as_bytes()).map_err(HeliumError::Io)?;
        IoWrite::write_all(&mut writer, b"\n").map_err(HeliumError::Io)?;
    }
    Ok(())
}

/// Convert a single row of a [`LogicalColumn`] to a [`serde_json::Value`].
fn logical_column_row_to_json_value(
    lc: &crate::LogicalColumn,
    row: usize,
) -> Option<serde_json::Value> {
    use crate::LogicalColumn;
    use serde_json::{Value, json};

    match lc {
        LogicalColumn::Primitive(cd) => Some(column_data_to_json(cd, row)),
        LogicalColumn::Utf8(v) => v.get(row).map(|s| Value::String(s.clone())),
        LogicalColumn::Binary(v) => v.get(row).map(|b| Value::String(hex_encode(b))),
        LogicalColumn::NullablePrim { present, values } => {
            if !present.get(row).copied().unwrap_or(false) {
                return Some(Value::Null);
            }
            let idx = present[..row].iter().filter(|&&p| p).count();
            Some(column_data_to_json(values, idx))
        }
        LogicalColumn::NullableUtf8 { present, strings } => {
            if !present.get(row).copied().unwrap_or(false) {
                return Some(Value::Null);
            }
            let idx = present[..row].iter().filter(|&&p| p).count();
            strings.get(idx).map(|s| Value::String(s.clone()))
        }
        LogicalColumn::NullableBinary { present, blobs } => {
            if !present.get(row).copied().unwrap_or(false) {
                return Some(Value::Null);
            }
            let idx = present[..row].iter().filter(|&&p| p).count();
            blobs.get(idx).map(|b| Value::String(hex_encode(b)))
        }
        LogicalColumn::Nullable { present, value } => {
            if !present.get(row).copied().unwrap_or(false) {
                return Some(Value::Null);
            }
            let idx = present[..row].iter().filter(|&&p| p).count();
            logical_column_row_to_json_value(value, idx)
        }
        LogicalColumn::Struct { fields } => {
            let mut obj = serde_json::Map::new();
            for (name, col) in fields {
                let val = logical_column_row_to_json_value(col, row).unwrap_or(Value::Null);
                obj.insert(name.clone(), val);
            }
            Some(Value::Object(obj))
        }
        LogicalColumn::List { offsets, values } => {
            let start = *offsets.get(row)? as usize;
            let end = *offsets.get(row + 1)? as usize;
            let arr: Vec<Value> = (start..end)
                .map(|i| logical_column_row_to_json_value(values, i).unwrap_or(Value::Null))
                .collect();
            Some(Value::Array(arr))
        }
        LogicalColumn::Map {
            offsets,
            keys,
            values,
        } => {
            let start = *offsets.get(row)? as usize;
            let end = *offsets.get(row + 1)? as usize;
            let mut obj = serde_json::Map::new();
            for i in start..end {
                let k = logical_column_row_to_json_value(keys, i)
                    .map(|v| match v {
                        Value::String(s) => s,
                        other => other.to_string(),
                    })
                    .unwrap_or_default();
                let v = logical_column_row_to_json_value(values, i).unwrap_or(Value::Null);
                obj.insert(k, v);
            }
            Some(Value::Object(obj))
        }
        LogicalColumn::Union { tags, variants } => {
            let tag = *tags.get(row)? as usize;
            let (vname, vcol) = variants.get(tag)?;
            let vrow = tags[..row].iter().filter(|&&t| t as usize == tag).count();
            let val = logical_column_row_to_json_value(vcol, vrow).unwrap_or(Value::Null);
            Some(json!({ vname.as_str(): val }))
        }
        LogicalColumn::ArrayOf { offsets, values } => {
            let start = *offsets.get(row)? as usize;
            let end = *offsets.get(row + 1)? as usize;
            let arr: Vec<Value> = (start..end)
                .map(|i| column_data_to_json(values, i))
                .collect();
            Some(Value::Array(arr))
        }
        LogicalColumn::ArrayOfUtf8 { offsets, strings } => {
            let start = *offsets.get(row)? as usize;
            let end = *offsets.get(row + 1)? as usize;
            let arr: Vec<Value> = strings[start..end.min(strings.len())]
                .iter()
                .map(|s| Value::String(s.clone()))
                .collect();
            Some(Value::Array(arr))
        }
        // Dictionary{inner}: look up the dictionary entry for this row's index.
        LogicalColumn::Dictionary {
            dictionary,
            indices,
        } => {
            let idx = *indices.get(row)? as usize;
            logical_column_row_to_json_value(dictionary, idx)
        }
        // Semantic types — render as JSON strings using the shared CSV formatter.
        LogicalColumn::Decimal128 { values } => values
            .get(row)
            .map(|&v| Value::String(crate::schema::csv::format_decimal128(v, 0))),
        LogicalColumn::Date32 { values } => values
            .get(row)
            .map(|&v| Value::String(crate::schema::csv::format_date32(v))),
        LogicalColumn::Date64 { values } => values
            .get(row)
            .map(|&v| Value::String(crate::schema::csv::format_date64(v))),
        LogicalColumn::Datetime { values } => values
            .get(row)
            .map(|&v| Value::String(crate::schema::csv::format_datetime(v))),
    }
}

/// Convert a single element of [`ColumnData`] to a [`serde_json::Value`].
fn column_data_to_json(cd: &crate::ColumnData, row: usize) -> serde_json::Value {
    use crate::ColumnData;
    use serde_json::Value;
    match cd {
        ColumnData::I8(v) => v
            .get(row)
            .map(|x| Value::Number((*x as i64).into()))
            .unwrap_or(Value::Null),
        ColumnData::I16(v) => v
            .get(row)
            .map(|x| Value::Number((*x as i64).into()))
            .unwrap_or(Value::Null),
        ColumnData::I32(v) => v
            .get(row)
            .map(|x| Value::Number((*x as i64).into()))
            .unwrap_or(Value::Null),
        ColumnData::I64(v) => v
            .get(row)
            .map(|x| Value::Number((*x).into()))
            .unwrap_or(Value::Null),
        ColumnData::U8(v) => v
            .get(row)
            .map(|x| Value::Number((*x as u64).into()))
            .unwrap_or(Value::Null),
        ColumnData::U16(v) => v
            .get(row)
            .map(|x| Value::Number((*x as u64).into()))
            .unwrap_or(Value::Null),
        ColumnData::U32(v) => v
            .get(row)
            .map(|x| Value::Number((*x as u64).into()))
            .unwrap_or(Value::Null),
        ColumnData::U64(v) => v
            .get(row)
            .map(|x| Value::Number((*x).into()))
            .unwrap_or(Value::Null),
        ColumnData::F32(v) => v
            .get(row)
            .map(|x| {
                serde_json::Number::from_f64(*x as f64)
                    .map(Value::Number)
                    .unwrap_or(Value::Null)
            })
            .unwrap_or(Value::Null),
        ColumnData::F64(v) => v
            .get(row)
            .map(|x| {
                serde_json::Number::from_f64(*x)
                    .map(Value::Number)
                    .unwrap_or(Value::Null)
            })
            .unwrap_or(Value::Null),
        ColumnData::Bytes(v) => Value::String(hex_encode(v.as_slice())),
    }
}

/// Hex-encode a byte slice.
fn hex_encode(b: &[u8]) -> String {
    b.iter().map(|byte| format!("{byte:02x}")).collect()
}

// ---------------------------------------------------------------------------
// Unit tests for write_json
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ColumnData, ColumnSpec, DataType, LogicalColumn, LogicalType, Schema};
    use std::collections::HashMap;

    fn ndjson_rows(output: &[u8]) -> Vec<serde_json::Value> {
        String::from_utf8_lossy(output)
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }

    #[test]
    fn write_json_basic() {
        use crate::schema::encodings::default_encodings;
        let lt_i64 = LogicalType::Primitive {
            data_type: DataType::I64,
        };
        let lt_utf8 = LogicalType::Utf8;
        let schema = Schema::new(vec![
            ColumnSpec::new("id".to_string(), lt_i64.clone(), default_encodings(&lt_i64)),
            ColumnSpec::new(
                "label".to_string(),
                lt_utf8.clone(),
                default_encodings(&lt_utf8),
            ),
        ]);
        let mut cols: HashMap<String, LogicalColumn> = HashMap::new();
        cols.insert(
            "id".to_string(),
            LogicalColumn::Primitive(ColumnData::I64(vec![10, 20])),
        );
        cols.insert(
            "label".to_string(),
            LogicalColumn::Utf8(vec!["x".to_string(), "y".to_string()]),
        );

        let mut out = Vec::new();
        write_json(&schema, &cols, &mut out).unwrap();
        let rows = ndjson_rows(&out);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0]["id"], serde_json::json!(10));
        assert_eq!(rows[0]["label"], serde_json::json!("x"));
        assert_eq!(rows[1]["id"], serde_json::json!(20));
    }

    #[test]
    fn write_json_nullable_null_cell() {
        use crate::schema::encodings::default_encodings;
        let lt = LogicalType::Nullable {
            inner: Box::new(LogicalType::Primitive {
                data_type: DataType::I32,
            }),
        };
        let schema = Schema::new(vec![ColumnSpec::new(
            "val".to_string(),
            lt.clone(),
            default_encodings(&lt),
        )]);
        let mut cols: HashMap<String, LogicalColumn> = HashMap::new();
        cols.insert(
            "val".to_string(),
            LogicalColumn::NullablePrim {
                present: vec![true, false, true],
                values: ColumnData::I32(vec![1, 3]),
            },
        );

        let mut out = Vec::new();
        write_json(&schema, &cols, &mut out).unwrap();
        let rows = ndjson_rows(&out);
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0]["val"], serde_json::json!(1));
        assert_eq!(rows[1]["val"], serde_json::Value::Null);
        assert_eq!(rows[2]["val"], serde_json::json!(3));
    }
}
