//! CSV → Helium [`crate::Schema`] type inferrer.
//!
//! # Feature flag: `csv`
//!
//! ```toml
//! helium-schema = { features = ["csv"] }
//! ```
//!
//! # Algorithm
//!
//! Reads the **entire** CSV to determine nullability, but samples only the
//! first N rows (default N = 1 000) for type inference.  This ensures that a
//! column whose first null appears beyond the sample window is still correctly
//! inferred as `Nullable<T>` rather than a non-nullable `T`.
//!
//! Type inference priority per column (sampled from first `max_rows` rows):
//!
//! 1. All non-null values parse as `i64`  → `DataType::I64`
//! 2. All non-null values parse as `f64` (at least one fails `i64`) → `DataType::F64`
//! 3. Otherwise → `LogicalType::Utf8`
//!
//! Default null sentinels: `""`, `"NULL"`, `"null"`, `"NA"`.
//!
//! # Limitations
//!
//! * Only flat schemas — no nested types inferred from CSV values.
//! * All columns are either `Primitive(I64)`, `Primitive(F64)`, or `Utf8`.
//! * An all-null column defaults to `Utf8` (not enough information to infer).

use std::path::Path;

use crate::{
    ColumnData, ColumnSpec, DataType, HeliumError, LogicalColumn, LogicalType, Result, Schema,
};

use super::encodings::default_encodings;

// ---------------------------------------------------------------------------
// Public options
// ---------------------------------------------------------------------------

/// Options controlling CSV schema inference.
#[derive(Debug, Clone)]
pub struct CsvInferOptions {
    /// Maximum number of data rows to sample for **type** inference.  Default: 1 000.
    ///
    /// Nullability is always determined from the full file (every row is checked
    /// for null sentinels), so a column whose first null appears after row
    /// `max_rows` is still correctly inferred as `Nullable<T>`.
    pub max_rows: usize,
    /// Whether the first row is a header row with column names. Default: `true`.
    pub has_header: bool,
    /// Field delimiter byte. Default: `b','`.
    pub delimiter: u8,
    /// Values that are treated as null / missing. Default: `["", "NULL", "null", "NA"]`.
    pub null_values: Vec<String>,
}

impl Default for CsvInferOptions {
    fn default() -> Self {
        Self {
            max_rows: 1_000,
            has_header: true,
            delimiter: b',',
            null_values: vec![
                String::new(),
                "NULL".to_string(),
                "null".to_string(),
                "NA".to_string(),
            ],
        }
    }
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Infer a Helium [`Schema`] from a CSV file at `path`.
///
/// Reads up to 1 000 rows with default inference options.
///
/// # Errors
///
/// Returns [`HeliumError::Io`] if the file cannot be opened, [`HeliumError::Format`]
/// for CSV parse errors, or [`HeliumError::Schema`] if the resulting schema
/// fails validation.
pub fn schema_from_csv(path: &Path) -> Result<Schema> {
    let content = std::fs::read_to_string(path).map_err(HeliumError::Io)?;
    schema_from_csv_str_with_options(&content, &CsvInferOptions::default())
}

/// Infer a Helium [`Schema`] from a CSV file with custom [`CsvInferOptions`].
pub fn schema_from_csv_with_options(path: &Path, opts: &CsvInferOptions) -> Result<Schema> {
    let content = std::fs::read_to_string(path).map_err(HeliumError::Io)?;
    schema_from_csv_str_with_options(&content, opts)
}

/// Infer a Helium [`Schema`] from an in-memory CSV string.
///
/// Equivalent to [`schema_from_csv`] but accepts the CSV data as a `&str`.
/// Primarily used in tests and when the CSV is already loaded into memory.
pub fn schema_from_csv_str(csv: &str) -> Result<Schema> {
    schema_from_csv_str_with_options(csv, &CsvInferOptions::default())
}

/// Infer a Helium [`Schema`] from an in-memory CSV string with custom options.
pub fn schema_from_csv_str_with_options(csv: &str, opts: &CsvInferOptions) -> Result<Schema> {
    infer_csv(csv, opts)
}

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

/// Column type accumulator.
///
/// Tracks the widest type that has been inferred from non-null values and
/// whether any null sentinel was seen.
#[derive(Debug)]
struct ColState {
    has_null: bool,
    kind: ScalarKind,
}

/// Type tag for a CSV column — ordered from most specific to least specific.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScalarKind {
    /// No non-null values seen yet (column might be all-null).
    None,
    /// All non-null values parse as `i64`.
    Int,
    /// Some non-null values need `f64` but not wider.
    Float,
    /// At least one value requires a string representation.
    Utf8,
}

impl ColState {
    fn new() -> Self {
        Self {
            has_null: false,
            kind: ScalarKind::None,
        }
    }

    /// Update state for a single cell value.
    ///
    /// `infer_type` controls whether this row is within the type-sampling window.
    /// Nullability (`has_null`) is **always** updated regardless of `infer_type`.
    fn update(&mut self, value: &str, null_values: &[String], infer_type: bool) {
        if null_values.iter().any(|nv| nv == value) {
            self.has_null = true;
            return;
        }
        // Only update the type detector within the sampling window.
        if !infer_type {
            return;
        }
        let new_kind = if value.parse::<i64>().is_ok() {
            ScalarKind::Int
        } else if value.parse::<f64>().is_ok() {
            ScalarKind::Float
        } else {
            ScalarKind::Utf8
        };
        self.kind = match (self.kind, new_kind) {
            (ScalarKind::None, k) => k,
            (ScalarKind::Int, ScalarKind::Int) => ScalarKind::Int,
            (ScalarKind::Int, ScalarKind::Float)
            | (ScalarKind::Float, ScalarKind::Int)
            | (ScalarKind::Float, ScalarKind::Float) => ScalarKind::Float,
            _ => ScalarKind::Utf8,
        };
    }

    fn to_logical_type(&self) -> LogicalType {
        let leaf = match self.kind {
            ScalarKind::Int => LogicalType::Primitive {
                data_type: DataType::I64,
            },
            ScalarKind::Float => LogicalType::Primitive {
                data_type: DataType::F64,
            },
            // All-null or explicitly Utf8.
            ScalarKind::Utf8 | ScalarKind::None => LogicalType::Utf8,
        };
        if self.has_null {
            LogicalType::Nullable {
                inner: Box::new(leaf),
            }
        } else {
            leaf
        }
    }
}

/// Preprocess a CSV string so blank lines are replaced by null-sentinel rows.
///
/// The csv crate may skip completely blank lines without producing records.
/// Replacing them with `n_cols` empty quoted fields (`"","",…`) ensures they
/// are parsed as actual null rows.
fn normalize_blank_lines(csv: &str, n_cols: usize, delimiter: u8) -> String {
    // Build a null row: n_cols quoted empty fields separated by the delimiter.
    // Using `""` avoids ambiguity between "blank line" and "empty field".
    let delim = char::from(delimiter);
    let null_row: String = std::iter::repeat_n("\"\"", n_cols)
        .collect::<Vec<_>>()
        .join(&delim.to_string());

    let mut out = String::with_capacity(csv.len() + 16);
    let mut chars = csv.chars().peekable();
    // We track whether we're at the start of a line to detect blank lines.
    let mut at_line_start = true;
    let mut current_line_blank = true;

    while let Some(ch) = chars.next() {
        if ch == '\n' {
            if at_line_start || current_line_blank {
                // Blank line — replace with null row.
                out.push_str(&null_row);
            }
            out.push('\n');
            at_line_start = true;
            current_line_blank = true;
        } else if ch == '\r' {
            // Skip bare \r (will be followed by \n).
            if at_line_start || current_line_blank {
                if chars.peek() != Some(&'\n') {
                    out.push_str(&null_row);
                    out.push('\r');
                }
            } else {
                out.push(ch);
            }
        } else {
            out.push(ch);
            at_line_start = false;
            current_line_blank = false;
        }
    }
    // If the input ended without a final newline and the last "line" was blank,
    // replace it (rare, but handle gracefully).
    // (We don't add a trailing newline here; the csv crate handles EOFs fine.)
    out
}

fn infer_csv(csv: &str, opts: &CsvInferOptions) -> Result<Schema> {
    // ── Step 1: get column names (without blank-line preprocessing) ─────────
    //   We need the column count BEFORE normalizing blank lines.
    let mut rdr_hdr = csv::ReaderBuilder::new()
        .delimiter(opts.delimiter)
        .has_headers(opts.has_header)
        .flexible(true)
        .from_reader(csv.as_bytes());

    // ── Step 2: get column names ────────────────────────────────────────────
    let col_names: Vec<String> = if opts.has_header {
        rdr_hdr
            .headers()
            .map_err(|e| HeliumError::Format(format!("CSV header error: {e}")))?
            .iter()
            .map(|h| h.to_string())
            .collect()
    } else {
        // No header: peek at first record to determine column count, then
        // rewind by building a new reader from the same string.
        let peek_count = {
            let mut peek_rdr = csv::ReaderBuilder::new()
                .delimiter(opts.delimiter)
                .has_headers(false)
                .flexible(true)
                .from_reader(csv.as_bytes());
            peek_rdr
                .records()
                .next()
                .transpose()
                .map_err(|e| HeliumError::Format(format!("CSV peek error: {e}")))?
                .map(|r| r.len())
                .unwrap_or(0)
        };
        (0..peek_count).map(|i| format!("col_{i}")).collect()
    };

    if col_names.is_empty() {
        return Err(HeliumError::Schema {
            column: "<csv>".into(),
            reason: "CSV has no columns (empty input or header with no fields)".into(),
        });
    }

    // ── Step 3: normalize blank lines and build the data reader ────────────
    // Blank lines in the CSV may be silently skipped by the csv crate instead of
    // producing an empty record. Preprocess the CSV to replace blank lines with
    // explicit null-sentinel rows (n_cols quoted empty fields).
    let normalized = normalize_blank_lines(csv, col_names.len(), opts.delimiter);

    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(opts.delimiter)
        .has_headers(opts.has_header)
        .flexible(true)
        .from_reader(normalized.as_bytes());

    // Re-consume the header so records() starts at data rows.
    if opts.has_header {
        let _ = rdr
            .headers()
            .map_err(|e| HeliumError::Format(format!("CSV re-read header error: {e}")))?;
    }

    // ── Step 4: scan data rows ──────────────────────────────────────────────
    // Iterate ALL rows so nullability is inferred from the full file.
    // Type detection is restricted to the first `max_rows` rows (sampling).
    let mut states: Vec<ColState> = (0..col_names.len()).map(|_| ColState::new()).collect();

    for (row_idx, result) in rdr.records().enumerate() {
        let record = result.map_err(|e| {
            HeliumError::Format(format!("CSV parse error at row {}: {e}", row_idx + 1))
        })?;
        // Type inference is sampled; nullability always uses the full scan.
        let infer_type = row_idx < opts.max_rows;
        for i in 0..col_names.len() {
            // After normalization, missing fields (short rows) are treated as
            // empty-string null sentinels.
            let field = record.get(i).unwrap_or("");
            if let Some(state) = states.get_mut(i) {
                state.update(field, &opts.null_values, infer_type);
            }
        }
    }

    // ── Step 4: build schema ────────────────────────────────────────────────
    let columns: Vec<ColumnSpec> = col_names
        .iter()
        .zip(states.iter())
        .map(|(name, state)| {
            let lt = state.to_logical_type();
            let enc = default_encodings(&lt);
            ColumnSpec::new(name.clone(), lt, enc)
        })
        .collect();

    let schema = Schema::new(columns);
    schema.validate()?;
    Ok(schema)
}

// ---------------------------------------------------------------------------
// CSV writer — options
// ---------------------------------------------------------------------------

/// Options for [`write_csv_with_options`].
#[derive(Debug, Clone)]
pub struct CsvWriteOptions {
    /// When `true`, return an error instead of JSON-stringifying [`LogicalColumn::List`],
    /// [`LogicalColumn::Map`], and [`LogicalColumn::Union`] columns.
    ///
    /// [`LogicalColumn::Struct`] always flattens to dotted sub-columns regardless of this flag.
    pub strict: bool,
    /// Field delimiter byte. Default: `b','`.
    pub delimiter: u8,
}

impl Default for CsvWriteOptions {
    fn default() -> Self {
        Self {
            strict: false,
            delimiter: b',',
        }
    }
}

// ---------------------------------------------------------------------------
// CSV writer — internal flat column representation
// ---------------------------------------------------------------------------

/// A single flat CSV column produced by expanding the top-level schema.
///
/// For `Struct` types we recursively expand into one `FlatCol` per leaf field,
/// naming them `parent.field` (or `parent.middle.leaf` for deeper nesting).
/// For other types each top-level column maps to exactly one `FlatCol`.
struct FlatCol<'a> {
    /// Header label for this CSV column.
    header: String,
    /// The innermost [`LogicalColumn`] that supplies data for this flat column.
    source: &'a LogicalColumn,
    /// When set, checked first: if it returns `true` for a given logical row
    /// index the cell is empty (the row is null in some outer Nullable wrapper).
    ///
    /// Used by `Nullable<Struct>` to propagate nullability to all sub-columns.
    is_null_fn: Option<NullFn<'a>>,
    /// Maps a logical row index to the compacted row index inside `source`.
    ///
    /// For plain (non-nullable) columns this is identity.
    /// For nullable columns it counts the number of `true` entries before `row`.
    compacted_idx_fn: CompactedFn<'a>,
}

type NullFn<'a> = Box<dyn Fn(usize) -> bool + 'a>;
type CompactedFn<'a> = Box<dyn Fn(usize) -> usize + 'a>;

impl<'a> FlatCol<'a> {
    /// Return the CSV cell string for logical row `row`.
    fn cell_at(&self, row: usize) -> String {
        if let Some(ref null_fn) = self.is_null_fn
            && null_fn(row)
        {
            return String::new();
        }
        let idx = (self.compacted_idx_fn)(row);
        format_cell(self.source, idx)
    }
}

/// Recursively flatten a top-level schema column into [`FlatCol`] entries.
///
/// `prefix` is the dotted name accumulated so far (column name at the top level).
///
/// For `Struct { fields }` we recurse into each field.  For
/// `Nullable { inner: Struct }` we recurse into each struct field carrying the
/// present bitmap along so all leaf sub-columns emit empty cells when the struct
/// row is null.  All other types produce exactly one `FlatCol`.
///
/// `outer_present` is non-`None` only when we have entered a `Nullable<Struct>`
/// wrapper; it is the `present` bitmap of that wrapper.  It is used to build
/// the `is_null_fn` and `compacted_idx_fn` for each leaf sub-column.
fn flatten_column<'a>(
    prefix: &str,
    lc: &'a LogicalColumn,
    outer_present: Option<&'a [bool]>,
    opts: &CsvWriteOptions,
    flat: &mut Vec<FlatCol<'a>>,
) -> Result<()> {
    match lc {
        // ── Struct → recurse into fields (no extra nullability context) ───────
        LogicalColumn::Struct { fields } => {
            for (field_name, field_col) in fields {
                let child_prefix = format!("{prefix}.{field_name}");
                flatten_column(&child_prefix, field_col, outer_present, opts, flat)?;
            }
        }

        // ── Nullable<Struct> → flatten struct fields, propagate nullability ──
        LogicalColumn::Nullable { present, value }
            if matches!(value.as_ref(), LogicalColumn::Struct { .. }) =>
        {
            let LogicalColumn::Struct { fields } = value.as_ref() else {
                unreachable!("matched above")
            };
            for (field_name, field_col) in fields {
                let child_prefix = format!("{prefix}.{field_name}");
                // Pass `present` as the outer_present so leaf sub-columns know
                // when their enclosing struct row is null.
                flatten_column(&child_prefix, field_col, Some(present), opts, flat)?;
            }
        }

        // ── Nullable<Nullable<…>> or Nullable<non-Struct> ────────────────────
        // The outer Nullable stays as a single flat column; `format_cell` already
        // handles Nullable variants recursively.
        LogicalColumn::Nullable { .. } => {
            push_simple(prefix, lc, outer_present, flat);
        }

        // ── List / Map / Union → JSON-stringify or strict error ───────────────
        LogicalColumn::List { .. } | LogicalColumn::Map { .. } | LogicalColumn::Union { .. } => {
            if opts.strict {
                let type_name = match lc {
                    LogicalColumn::List { .. } => "List",
                    LogicalColumn::Map { .. } => "Map",
                    LogicalColumn::Union { .. } => "Union",
                    _ => unreachable!(),
                };
                return Err(HeliumError::Schema {
                    column: prefix.to_string(),
                    reason: format!(
                        "CSV strict mode: column '{prefix}' has type {type_name} which cannot \
                         be losslessly represented in CSV; remove --csv-strict to \
                         JSON-stringify, or use --to json / --to parquet for full fidelity"
                    ),
                });
            }
            push_simple(prefix, lc, outer_present, flat);
        }

        // ── Everything else → one flat column ────────────────────────────────
        _ => {
            push_simple(prefix, lc, outer_present, flat);
        }
    }
    Ok(())
}

/// Push a single non-struct flat column, wiring up the `outer_present` context
/// if we are inside a `Nullable<Struct>` wrapper.
///
/// When `outer_present` is `Some(present)`:
/// - `is_null_fn` checks `!present[row]`.
/// - `compacted_idx_fn` counts trues in `present[..row]` (the row index inside
///   the struct value is the compacted index, since struct fields are stored
///   without nulls — only the outer Nullable bitmap introduces the gap).
fn push_simple<'a>(
    header: &str,
    source: &'a LogicalColumn,
    outer_present: Option<&'a [bool]>,
    flat: &mut Vec<FlatCol<'a>>,
) {
    if let Some(present) = outer_present {
        let is_null_fn: NullFn<'a> =
            Box::new(move |row| !present.get(row).copied().unwrap_or(false));
        let compacted_idx_fn: CompactedFn<'a> =
            Box::new(move |row| present[..row].iter().filter(|&&p| p).count());
        flat.push(FlatCol {
            header: header.to_string(),
            source,
            is_null_fn: Some(is_null_fn),
            compacted_idx_fn,
        });
    } else {
        flat.push(FlatCol {
            header: header.to_string(),
            source,
            is_null_fn: None,
            compacted_idx_fn: Box::new(|row| row),
        });
    }
}

// ---------------------------------------------------------------------------
// CSV writer — public API
// ---------------------------------------------------------------------------

/// Write a Helium schema + columns to CSV format.
///
/// Emits a header row with column names (in `schema.columns` order), then one
/// row per logical row with cells formatted according to each column's
/// [`LogicalType`]:
///
/// - **Primitive**: Display representation (e.g. `42`, `3.14`).
/// - **Utf8**: the string, CSV-quoted if it contains commas, quotes, or newlines.
/// - **Binary**: lowercase hex encoding.
/// - **Nullable\<T\>**: empty cell when null, T-formatted when present.
/// - **Struct**: flattened to dotted sub-columns (`addr.street`, `addr.zip`).
///   Nested structs concatenate: `user.addr.street` etc. See
///   [`write_csv_with_options`] for the full flattening rules.
/// - **List / Map / Union** (and legacy flat `ArrayOf`, `ArrayOfUtf8`): serialised as a
///   compact JSON string. This is a lossy representation — it round-trips
///   through JSON, not back through Helium's type system.
///
/// Equivalent to `write_csv_with_options(schema, columns, writer, &CsvWriteOptions::default())`.
///
/// # Errors
///
/// Returns [`HeliumError::Schema`] if a column referenced in `schema` is
/// missing from `columns`, or if the row count is inconsistent.
pub fn write_csv<W: std::io::Write>(
    schema: &Schema,
    columns: &std::collections::HashMap<String, LogicalColumn>,
    writer: W,
) -> Result<()> {
    write_csv_with_options(schema, columns, writer, &CsvWriteOptions::default())
}

/// Write a Helium schema + columns to CSV format with custom [`CsvWriteOptions`].
///
/// Struct columns are always flattened to dotted sub-columns (e.g. `addr.street`).
/// When `options.strict` is `true`, List / Map / Union columns produce an error
/// instead of being JSON-stringified.
///
/// # Errors
///
/// Returns [`HeliumError::Schema`] if:
/// - A column referenced in `schema` is missing from `columns`.
/// - Row counts are inconsistent across top-level columns.
/// - `options.strict` is `true` and a List / Map / Union column is encountered.
pub fn write_csv_with_options<W: std::io::Write>(
    schema: &Schema,
    columns: &std::collections::HashMap<String, LogicalColumn>,
    mut writer: W,
    options: &CsvWriteOptions,
) -> Result<()> {
    use std::io::Write as IoWrite;

    // Resolve top-level columns in schema order.
    let top_cols: Vec<(&str, &LogicalColumn)> = schema
        .columns
        .iter()
        .map(|spec| {
            let lc = columns.get(&spec.name).ok_or_else(|| HeliumError::Schema {
                column: spec.name.clone(),
                reason: "column present in schema but missing from data map".into(),
            })?;
            Ok((spec.name.as_str(), lc))
        })
        .collect::<Result<_>>()?;

    if top_cols.is_empty() {
        return Ok(());
    }

    // Check row count consistency across top-level columns.
    let row_count = top_cols[0].1.row_count();
    for (name, lc) in &top_cols {
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

    // Flatten all top-level columns into a list of FlatCol entries.
    let mut flat: Vec<FlatCol<'_>> = Vec::new();
    for (col_name, lc) in &top_cols {
        flatten_column(col_name, lc, None, options, &mut flat)?;
    }

    // Build a csv::Writer wrapping our writer.
    let mut csv_wtr = csv::WriterBuilder::new()
        .delimiter(options.delimiter)
        .from_writer(vec![]);

    // Header row.
    let header: Vec<&str> = flat.iter().map(|fc| fc.header.as_str()).collect();
    csv_wtr
        .write_record(&header)
        .map_err(|e| HeliumError::Format(format!("CSV write error: {e}")))?;

    // Data rows.
    for row in 0..row_count {
        let record: Vec<String> = flat.iter().map(|fc| fc.cell_at(row)).collect();
        csv_wtr
            .write_record(&record)
            .map_err(|e| HeliumError::Format(format!("CSV write error: {e}")))?;
    }

    let data = csv_wtr
        .into_inner()
        .map_err(|e| HeliumError::Format(format!("CSV flush error: {e}")))?;
    IoWrite::write_all(&mut writer, &data).map_err(HeliumError::Io)?;
    Ok(())
}

/// Format a single cell for the given column at the given row index.
///
/// This is called from [`FlatCol::cell_at`] after null-checks and compaction
/// have already been applied — `row` is the compacted index into `lc`.
fn format_cell(lc: &LogicalColumn, row: usize) -> String {
    match lc {
        LogicalColumn::Primitive(cd) => format_column_data_cell(cd, row),
        LogicalColumn::Utf8(v) => v.get(row).cloned().unwrap_or_default(),
        LogicalColumn::Binary(v) => v.get(row).map(|b| hex_encode(b)).unwrap_or_default(),
        LogicalColumn::Nullable { present, value } => {
            if !present.get(row).copied().unwrap_or(false) {
                return String::new();
            }
            // Count how many `true` entries are before `row` to get the compacted index.
            let compacted_idx = present[..row].iter().filter(|&&p| p).count();
            format_cell(value, compacted_idx)
        }
        LogicalColumn::NullablePrim { present, values } => {
            if !present.get(row).copied().unwrap_or(false) {
                return String::new();
            }
            let idx = present[..row].iter().filter(|&&p| p).count();
            format_column_data_cell(values, idx)
        }
        LogicalColumn::NullableUtf8 { present, strings } => {
            if !present.get(row).copied().unwrap_or(false) {
                return String::new();
            }
            let idx = present[..row].iter().filter(|&&p| p).count();
            strings.get(idx).cloned().unwrap_or_default()
        }
        LogicalColumn::NullableBinary { present, blobs } => {
            if !present.get(row).copied().unwrap_or(false) {
                return String::new();
            }
            let idx = present[..row].iter().filter(|&&p| p).count();
            blobs.get(idx).map(|b| hex_encode(b)).unwrap_or_default()
        }
        // Semantic types — render as human-readable strings.
        LogicalColumn::Decimal128 { values } => values
            .get(row)
            .map(|&v| format_decimal128(v, 0))
            .unwrap_or_default(),
        LogicalColumn::Date32 { values } => values
            .get(row)
            .map(|&v| format_date32(v))
            .unwrap_or_default(),
        LogicalColumn::Date64 { values } => values
            .get(row)
            .map(|&v| format_date64(v))
            .unwrap_or_default(),
        LogicalColumn::Datetime { values } => values
            .get(row)
            .map(|&v| format_datetime(v))
            .unwrap_or_default(),
        // Dictionary{inner}: look up the dictionary entry for this row's index.
        LogicalColumn::Dictionary {
            dictionary,
            indices,
        } => {
            let idx = indices.get(row).copied().unwrap_or(0) as usize;
            format_cell(dictionary, idx)
        }
        // Nested / complex types: stringify as compact JSON.
        LogicalColumn::Struct { .. }
        | LogicalColumn::List { .. }
        | LogicalColumn::Map { .. }
        | LogicalColumn::Union { .. }
        | LogicalColumn::ArrayOf { .. }
        | LogicalColumn::ArrayOfUtf8 { .. } => logical_column_row_to_json(lc, row)
            .map(|v| v.to_string())
            .unwrap_or_default(),
    }
}

/// Render an `i128` decimal value as a fixed-point string.
///
/// `scale` is the number of fractional digits.  If `scale == 0`, renders as an
/// integer.  The decimal point is inserted at the correct position.
///
/// This function is also used by the JSON and Arrow helpers.
pub(crate) fn format_decimal128(value: i128, scale: u8) -> String {
    if scale == 0 {
        return value.to_string();
    }
    let neg = value < 0;
    let mag = value.unsigned_abs(); // u128
    let s = mag.to_string();
    let scale = scale as usize;
    let result = if s.len() <= scale {
        // Pad with leading zeros
        let zeros = scale - s.len();
        format!("0.{}{s}", "0".repeat(zeros))
    } else {
        let (int_part, frac_part) = s.split_at(s.len() - scale);
        format!("{int_part}.{frac_part}")
    };
    if neg { format!("-{result}") } else { result }
}

/// Render an `i32` days-since-epoch value as ISO 8601 `YYYY-MM-DD`.
///
/// Uses the proleptic Gregorian calendar formula (no external deps).
pub(crate) fn format_date32(days: i32) -> String {
    days_to_date_string(days as i64)
}

/// Render an `i64` millis-since-epoch value as ISO 8601 `YYYY-MM-DD`.
pub(crate) fn format_date64(millis: i64) -> String {
    let days = millis.div_euclid(86_400_000);
    days_to_date_string(days)
}

/// Render an `i64` millis-since-epoch as a UTC ISO 8601 timestamp string.
///
/// Renders as `YYYY-MM-DDTHH:MM:SS` or `YYYY-MM-DDTHH:MM:SS.mmm` when the
/// fractional-seconds part is non-zero.  For sub-millisecond units the caller
/// is responsible for passing the correct millisecond count.
pub(crate) fn format_datetime(millis: i64) -> String {
    let total_secs = millis.div_euclid(1000);
    let ms = millis.rem_euclid(1000) as u32;
    let days = total_secs.div_euclid(86400);
    let time_secs = total_secs.rem_euclid(86400) as u32;
    let hh = time_secs / 3600;
    let mm = (time_secs % 3600) / 60;
    let ss = time_secs % 60;
    let date = days_to_date_string(days);
    if ms == 0 {
        format!("{date}T{hh:02}:{mm:02}:{ss:02}Z")
    } else {
        format!("{date}T{hh:02}:{mm:02}:{ss:02}.{ms:03}Z")
    }
}

/// Convert a count of days since 1970-01-01 (proleptic Gregorian) to
/// an ISO 8601 `YYYY-MM-DD` string.  Handles negative days (dates before
/// 1970).
fn days_to_date_string(days: i64) -> String {
    // Algorithm: Julian Day Number → civil date.
    // The Julian Day Number for 1970-01-01 is 2440588.
    let jdn = days + 2_440_588;
    // Richards' algorithm (from Wikipedia "Julian day"):
    let f = jdn + 1401 + (((4 * jdn + 274_277) / 146_097) * 3) / 4 - 38;
    let e = 4 * f + 3;
    let g = (e % 1461) / 4;
    let h = 5 * g + 2;
    let day = (h % 153) / 5 + 1;
    let month = (h / 153 + 2) % 12 + 1;
    let year = e / 1461 - 4716 + (14 - month) / 12;
    format!("{year:04}-{month:02}-{day:02}")
}

/// Format a [`ColumnData`] value at `row` as a display string.
fn format_column_data_cell(cd: &ColumnData, row: usize) -> String {
    match cd {
        ColumnData::I8(v) => v.get(row).map(|x| x.to_string()).unwrap_or_default(),
        ColumnData::I16(v) => v.get(row).map(|x| x.to_string()).unwrap_or_default(),
        ColumnData::I32(v) => v.get(row).map(|x| x.to_string()).unwrap_or_default(),
        ColumnData::I64(v) => v.get(row).map(|x| x.to_string()).unwrap_or_default(),
        ColumnData::U8(v) => v.get(row).map(|x| x.to_string()).unwrap_or_default(),
        ColumnData::U16(v) => v.get(row).map(|x| x.to_string()).unwrap_or_default(),
        ColumnData::U32(v) => v.get(row).map(|x| x.to_string()).unwrap_or_default(),
        ColumnData::U64(v) => v.get(row).map(|x| x.to_string()).unwrap_or_default(),
        ColumnData::F32(v) => v.get(row).map(|x| x.to_string()).unwrap_or_default(),
        ColumnData::F64(v) => v.get(row).map(|x| x.to_string()).unwrap_or_default(),
        ColumnData::Bytes(v) => hex_encode(v.as_slice()),
    }
}

/// Hex-encode a byte slice.
fn hex_encode(b: &[u8]) -> String {
    b.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Attempt to convert a single row of a [`LogicalColumn`] to a
/// [`serde_json::Value`] for use as a CSV cell.
fn logical_column_row_to_json(lc: &LogicalColumn, row: usize) -> Option<serde_json::Value> {
    use serde_json::{Value, json};
    match lc {
        LogicalColumn::Primitive(cd) => Some(column_data_row_to_json(cd, row)),
        LogicalColumn::Utf8(v) => v.get(row).map(|s| Value::String(s.clone())),
        LogicalColumn::Binary(v) => v.get(row).map(|b| Value::String(hex_encode(b))),
        LogicalColumn::Struct { fields } => {
            let mut obj = serde_json::Map::new();
            for (name, col) in fields {
                if let Some(val) = logical_column_row_to_json(col, row) {
                    obj.insert(name.clone(), val);
                }
            }
            Some(Value::Object(obj))
        }
        LogicalColumn::List { offsets, values } => {
            let start = *offsets.get(row)? as usize;
            let end = *offsets.get(row + 1)? as usize;
            let arr: Vec<Value> = (start..end)
                .filter_map(|i| logical_column_row_to_json(values, i))
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
                let k = logical_column_row_to_json(keys, i)
                    .map(|v| match v {
                        Value::String(s) => s,
                        other => other.to_string(),
                    })
                    .unwrap_or_default();
                let v = logical_column_row_to_json(values, i).unwrap_or(Value::Null);
                obj.insert(k, v);
            }
            Some(Value::Object(obj))
        }
        LogicalColumn::Nullable { present, value } => {
            if !present.get(row).copied().unwrap_or(false) {
                return Some(Value::Null);
            }
            let idx = present[..row].iter().filter(|&&p| p).count();
            logical_column_row_to_json(value, idx)
        }
        LogicalColumn::Union { tags, variants } => {
            let tag = *tags.get(row)? as usize;
            let (vname, vcol) = variants.get(tag)?;
            let vrow = tags[..row].iter().filter(|&&t| t as usize == tag).count();
            let val = logical_column_row_to_json(vcol, vrow).unwrap_or(Value::Null);
            Some(json!({ vname.as_str(): val }))
        }
        LogicalColumn::ArrayOf { offsets, values } => {
            let start = *offsets.get(row)? as usize;
            let end = *offsets.get(row + 1)? as usize;
            let arr: Vec<Value> = (start..end)
                .map(|i| column_data_row_to_json(values, i))
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
        LogicalColumn::NullablePrim { present, values } => {
            if !present.get(row).copied().unwrap_or(false) {
                return Some(Value::Null);
            }
            let idx = present[..row].iter().filter(|&&p| p).count();
            Some(column_data_row_to_json(values, idx))
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
        // Dictionary{inner}: look up the dictionary entry for this row's index.
        LogicalColumn::Dictionary {
            dictionary,
            indices,
        } => {
            let idx = *indices.get(row)? as usize;
            logical_column_row_to_json(dictionary, idx)
        }
        // Semantic types — render as JSON strings.
        LogicalColumn::Decimal128 { values } => values
            .get(row)
            .map(|&v| Value::String(format_decimal128(v, 0))),
        LogicalColumn::Date32 { values } => {
            values.get(row).map(|&v| Value::String(format_date32(v)))
        }
        LogicalColumn::Date64 { values } => {
            values.get(row).map(|&v| Value::String(format_date64(v)))
        }
        LogicalColumn::Datetime { values } => {
            values.get(row).map(|&v| Value::String(format_datetime(v)))
        }
    }
}

/// Convert a single [`ColumnData`] element at `row` to a [`serde_json::Value`].
fn column_data_row_to_json(cd: &ColumnData, row: usize) -> serde_json::Value {
    use serde_json::Value;
    match cd {
        ColumnData::I8(v) => v
            .get(row)
            .map(|x| json_number(*x as i64))
            .unwrap_or(Value::Null),
        ColumnData::I16(v) => v
            .get(row)
            .map(|x| json_number(*x as i64))
            .unwrap_or(Value::Null),
        ColumnData::I32(v) => v
            .get(row)
            .map(|x| json_number(*x as i64))
            .unwrap_or(Value::Null),
        ColumnData::I64(v) => v.get(row).map(|x| json_number(*x)).unwrap_or(Value::Null),
        ColumnData::U8(v) => v
            .get(row)
            .map(|x| json_number(*x as i64))
            .unwrap_or(Value::Null),
        ColumnData::U16(v) => v
            .get(row)
            .map(|x| json_number(*x as i64))
            .unwrap_or(Value::Null),
        ColumnData::U32(v) => v
            .get(row)
            .map(|x| json_number(*x as i64))
            .unwrap_or(Value::Null),
        ColumnData::U64(v) => v
            .get(row)
            .map(|x| json_number(*x as i64))
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

fn json_number(n: i64) -> serde_json::Value {
    serde_json::Value::Number(serde_json::Number::from(n))
}

// ---------------------------------------------------------------------------
// Unit tests for write_csv
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ColumnData, ColumnSpec, DataType, LogicalColumn, LogicalType, Schema};
    use std::collections::HashMap;

    fn flat_schema() -> Schema {
        Schema::new(vec![
            ColumnSpec::new(
                "id".to_string(),
                LogicalType::Primitive {
                    data_type: DataType::I64,
                },
                vec![vec![]],
            ),
            ColumnSpec::new("name".to_string(), LogicalType::Utf8, vec![vec![]]),
            ColumnSpec::new(
                "score".to_string(),
                LogicalType::Primitive {
                    data_type: DataType::F64,
                },
                vec![vec![]],
            ),
        ])
    }

    #[test]
    fn write_csv_basic_roundtrip() {
        let schema = flat_schema();
        let mut cols: HashMap<String, LogicalColumn> = HashMap::new();
        cols.insert(
            "id".to_string(),
            LogicalColumn::Primitive(ColumnData::I64(vec![1, 2, 3])),
        );
        cols.insert(
            "name".to_string(),
            LogicalColumn::Utf8(vec![
                "alice".to_string(),
                "bob".to_string(),
                "carol".to_string(),
            ]),
        );
        cols.insert(
            "score".to_string(),
            LogicalColumn::Primitive(ColumnData::F64(vec![1.0, 2.5, 3.99])),
        );

        let mut out = Vec::new();
        write_csv(&schema, &cols, &mut out).unwrap();
        let s = String::from_utf8(out).unwrap();
        assert!(s.starts_with("id,name,score\n"), "missing header: {s}");
        assert!(s.contains("1,alice,1"), "row 0 missing: {s}");
        assert!(s.contains("2,bob,2.5"), "row 1 missing: {s}");
    }

    #[test]
    fn write_csv_nullable_empty_cell() {
        use crate::schema::encodings::default_encodings;
        let lt = LogicalType::Nullable {
            inner: Box::new(LogicalType::Primitive {
                data_type: DataType::I32,
            }),
        };
        let schema = Schema::new(vec![ColumnSpec::new(
            "x".to_string(),
            lt.clone(),
            default_encodings(&lt),
        )]);
        let mut cols: HashMap<String, LogicalColumn> = HashMap::new();
        cols.insert(
            "x".to_string(),
            LogicalColumn::NullablePrim {
                present: vec![true, false, true],
                values: ColumnData::I32(vec![10, 30]),
            },
        );

        let mut out = Vec::new();
        write_csv(&schema, &cols, &mut out).unwrap();
        let s = String::from_utf8(out).unwrap();
        let rows: Vec<&str> = s.lines().collect();
        assert_eq!(rows.len(), 4); // header + 3 data
        // The csv crate emits an empty string as "" (quoted); parse it back to verify it round-trips as empty.
        let parsed: Vec<Vec<String>> = rows[1..]
            .iter()
            .map(|row| {
                csv::ReaderBuilder::new()
                    .has_headers(false)
                    .from_reader(row.as_bytes())
                    .records()
                    .next()
                    .unwrap()
                    .unwrap()
                    .iter()
                    .map(|f| f.to_string())
                    .collect()
            })
            .collect();
        assert_eq!(parsed[1][0], ""); // null row produces empty cell
    }

    #[test]
    fn write_csv_binary_hex() {
        use crate::schema::encodings::default_encodings;
        let lt = LogicalType::Binary;
        let schema = Schema::new(vec![ColumnSpec::new(
            "blob".to_string(),
            lt.clone(),
            default_encodings(&lt),
        )]);
        let mut cols: HashMap<String, LogicalColumn> = HashMap::new();
        cols.insert(
            "blob".to_string(),
            LogicalColumn::Binary(vec![vec![0xde, 0xad], vec![0xbe, 0xef]]),
        );

        let mut out = Vec::new();
        write_csv(&schema, &cols, &mut out).unwrap();
        let s = String::from_utf8(out).unwrap();
        assert!(s.contains("dead"), "expected hex encoding: {s}");
        assert!(s.contains("beef"), "expected hex encoding: {s}");
    }
}
