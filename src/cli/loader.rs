//! Format detection and data loading for helium-cli subcommands.
//!
//! # Supported formats
//!
//! | Extension | Schema inference | Data loading |
//! |---|---|---|
//! | `.csv` | `helium_schema::csv` | Flat columns (Primitive / Utf8 / Nullable) |
//! | `.json` / `.ndjson` | `helium_schema::json` | Recursive nested types |
//! | `.parquet` | `helium_schema::parquet` | Flat columns via row API |
//! | `.avsc` | `helium_schema::avro` | Schema-only; no binary data |
//! | `.avro` | `helium_schema::avro` | Full nested data via Avro OCF reader |
//!
//! # Data loading contract
//!
//! [`load_data`] returns a `Vec<(column_name, LogicalType, LogicalColumn)>` that
//! can be passed directly to [`helium_optimizer::Optimizer::optimize`] or written
//! via [`helium_core::HeliumWriter`].
//!
//! The JSON loader supports all recursive `LogicalType` variants (Struct, List,
//! Map, Nullable, Union).  The CSV and Parquet loaders support flat column types
//! only (Primitive, Utf8, Binary, Nullable of those).
//!
//! # Streaming / chunked loading
//!
//! [`load_data_chunked`] is the streaming companion to [`load_data`].  It reads
//! `chunk_rows` rows at a time and invokes a callback with each chunk's
//! `LoadedData`.  Peak memory is `O(chunk_rows × column_count)` for formats that
//! support true row-at-a-time iteration:
//!
//! - **CSV** — `csv::Reader` row-iterator; true streaming.
//! - **Parquet** — `SerializedFileReader::get_row_iter()`; true streaming.
//! - **NDJSON** — line-at-a-time; true streaming.
//! - **Avro** — `apache_avro::Reader` iterates one record at a time (one OCF
//!   block buffered internally); true streaming.
//! - **JSON-array** (`[{…}, …]` top-level array) — falls back to in-memory
//!   load + `LogicalColumn::slice` chunking; the parse model requires the full
//!   document in memory.  This fallback is documented in the function's doc
//!   comment.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, bail};
use helium::{ColumnData, DataType, LogicalColumn, LogicalType, Schema};

/// A loaded dataset: column name, logical type, and data.
pub type LoadedData = Vec<(String, LogicalType, LogicalColumn)>;

// ---------------------------------------------------------------------------
// Load options
// ---------------------------------------------------------------------------

/// Options that affect data loading.
///
/// Currently only carries a CSV field delimiter; other options may be added
/// without breaking existing call sites that use [`LoadOptions::default`].
#[derive(Debug, Clone, Copy)]
pub struct LoadOptions {
    /// CSV field delimiter byte. Default: `b','`. Ignored for non-CSV inputs.
    pub delimiter: u8,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self { delimiter: b',' }
    }
}

// ---------------------------------------------------------------------------
// Format detection
// ---------------------------------------------------------------------------

/// Supported input formats (detected from file extension).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Csv,
    Json,
    Parquet,
    /// Avro schema file (`.avsc`).  Data loading is not supported.
    AvroSchema,
    /// Avro Object Container Format (`.avro`).  Full data loading supported.
    Avro,
}

/// Detect the [`Format`] from a file extension.
pub fn detect_format(path: &Path) -> anyhow::Result<Format> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "csv" => Ok(Format::Csv),
        "json" | "ndjson" => Ok(Format::Json),
        "parquet" => Ok(Format::Parquet),
        "avsc" => Ok(Format::AvroSchema),
        "avro" => Ok(Format::Avro),
        other => bail!(
            "unsupported file extension '.{other}'; expected .csv / .json / .parquet / .avsc / .avro"
        ),
    }
}

// ---------------------------------------------------------------------------
// Schema inference
// ---------------------------------------------------------------------------

/// Infer a Helium [`Schema`] from a file using an explicitly supplied [`Format`].
///
/// Identical to [`infer_schema`] but bypasses extension detection.
/// `opts.delimiter` is used when `fmt == Format::Csv`; ignored otherwise.
pub fn infer_schema_for_fmt(
    path: &Path,
    fmt: Format,
    opts: &LoadOptions,
) -> anyhow::Result<Schema> {
    match fmt {
        Format::Csv => {
            let csv_opts = helium::schema::csv::CsvInferOptions {
                delimiter: opts.delimiter,
                ..Default::default()
            };
            helium::schema::csv::schema_from_csv_with_options(path, &csv_opts)
                .with_context(|| format!("inferring CSV schema from '{}'", path.display()))
        }
        Format::Json => helium::schema::json::schema_from_json(path)
            .with_context(|| format!("inferring JSON schema from '{}'", path.display())),
        Format::Parquet => helium::schema::parquet::schema_from_parquet(path)
            .with_context(|| format!("inferring Parquet schema from '{}'", path.display())),
        Format::AvroSchema => {
            let content = std::fs::read_to_string(path)
                .with_context(|| format!("reading Avro schema file '{}'", path.display()))?;
            helium::schema::avsc_to_schema(&content)
                .with_context(|| format!("parsing Avro schema '{}'", path.display()))
        }
        Format::Avro => {
            // Read just the schema from the Avro OCF file (discards the data).
            let (schema, _) = helium::schema::read_avro_data(path)
                .with_context(|| format!("reading Avro schema from '{}'", path.display()))?;
            Ok(schema)
        }
    }
    .map_err(|e| anyhow::anyhow!("{e}"))
}

// ---------------------------------------------------------------------------
// Data loading
// ---------------------------------------------------------------------------

/// Load data from a file using an explicitly supplied [`Format`].
///
/// Identical to [`load_data`] but bypasses extension detection.
/// `opts.delimiter` is used when `fmt == Format::Csv`; ignored otherwise.
///
/// This is a thin wrapper: it resolves the schema (if not provided), then
/// delegates to [`load_data_chunked`] with `chunk_rows = usize::MAX`.  Every
/// streaming loader emits exactly one chunk for `usize::MAX`, so the result is
/// the complete in-memory dataset.
pub fn load_data_for_fmt(
    path: &Path,
    fmt: Format,
    schema: Option<&Schema>,
    opts: &LoadOptions,
) -> anyhow::Result<LoadedData> {
    if matches!(fmt, Format::AvroSchema) {
        bail!(
            "Avro schema files (.avsc) contain no row data; use 'helium convert' \
             with a CSV/JSON/Parquet input and an .avsc-derived schema"
        );
    }
    let schema = match schema {
        Some(s) => s.clone(),
        None => infer_schema_for_fmt(path, fmt, opts)?,
    };
    let mut accum: Option<LoadedData> = None;
    load_data_chunked(path, fmt, &schema, usize::MAX, opts, &mut |chunk| {
        // With chunk_rows = usize::MAX every streaming loader emits exactly one
        // chunk containing the full dataset.  If more than one chunk arrives
        // here it is a logic error — surface it rather than silently
        // concatenating.
        if accum.is_some() {
            bail!(
                "internal error: load_data_chunked emitted more than one chunk for \
                 chunk_rows = usize::MAX; this should not happen"
            );
        }
        accum = Some(chunk);
        Ok(())
    })?;
    Ok(accum.unwrap_or_default())
}

/// Load only the first `sample_rows` rows — a representative prefix used by
/// `optimize-schema` to pick per-column encodings without reading the whole
/// file. Stops reading after the first chunk (does not scan the rest of a
/// large file). `sample_rows` must be > 0; for the whole file use
/// [`load_data_for_fmt`].
pub fn load_data_sample(
    path: &Path,
    fmt: Format,
    sample_rows: usize,
    opts: &LoadOptions,
) -> anyhow::Result<LoadedData> {
    if matches!(fmt, Format::AvroSchema) {
        bail!("Avro schema files (.avsc) contain no row data; use a CSV/JSON/Parquet/Avro input");
    }
    let schema = infer_schema_for_fmt(path, fmt, opts)?;
    // Read exactly one chunk of `sample_rows` rows, then stop via a sentinel
    // error so the rest of the file is never read.
    const STOP: &str = "::helium-sample-prefix-collected::";
    let mut first: Option<LoadedData> = None;
    let res = load_data_chunked(path, fmt, &schema, sample_rows, opts, &mut |chunk| {
        first = Some(chunk);
        bail!(STOP)
    });
    if let Err(e) = res
        && !format!("{e}").contains(STOP)
    {
        return Err(e);
    }
    Ok(first.unwrap_or_default())
}

// ---------------------------------------------------------------------------
// Chunked / streaming data loader
// ---------------------------------------------------------------------------

/// Streaming loader: reads `chunk_rows` rows at a time and invokes `on_chunk`
/// with each chunk's `LoadedData`.  Peak memory is
/// `O(chunk_rows × column_count)` for formats that support row-at-a-time
/// iteration (CSV, Parquet, NDJSON, Avro).
///
/// # JSON-array fallback
///
/// When the input is a top-level JSON array (`[{…}, {…}, …]`) the whole file
/// must be parsed before any record is available (no SAX parser is used).  In
/// that case this function falls back to an in-memory load followed by
/// `LogicalColumn::slice` chunking.  The peak memory footprint is the same as
/// `load_data` for JSON-array inputs — only true NDJSON gets the
/// bounded-memory guarantee.
///
/// # Errors
///
/// Returns an error for `.avsc` inputs (schema-only format, no row data) or
/// for any I/O / type-conversion failure.
pub fn load_data_chunked(
    input: &Path,
    fmt: Format,
    schema: &Schema,
    chunk_rows: usize,
    opts: &LoadOptions,
    on_chunk: &mut dyn FnMut(LoadedData) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    match fmt {
        Format::Csv => load_csv_data_chunked(input, schema, chunk_rows, opts.delimiter, on_chunk),
        Format::Parquet => load_parquet_data_chunked(input, schema, chunk_rows, on_chunk),
        Format::Json => load_json_data_chunked(input, schema, chunk_rows, on_chunk),
        Format::Avro => load_avro_data_chunked(input, schema, chunk_rows, on_chunk),
        Format::AvroSchema => anyhow::bail!(
            "Avro schema files (.avsc) contain no row data; use a CSV/JSON/Parquet/Avro input"
        ),
    }
}

// ---------------------------------------------------------------------------
// CSV chunked loader
// ---------------------------------------------------------------------------

fn load_csv_data_chunked(
    path: &Path,
    schema: &Schema,
    chunk_rows: usize,
    delimiter: u8,
    on_chunk: &mut dyn FnMut(LoadedData) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .delimiter(delimiter)
        .from_path(path)
        .with_context(|| format!("opening CSV file '{}'", path.display()))?;

    let headers: Vec<String> = rdr
        .headers()
        .context("reading CSV headers")?
        .iter()
        .map(|h| h.to_string())
        .collect();
    let hdr_map: HashMap<&str, usize> = headers
        .iter()
        .enumerate()
        .map(|(i, h)| (h.as_str(), i))
        .collect();

    let n = schema.columns.len();
    // Use a saturating initial capacity so that chunk_rows = usize::MAX (the
    // "load everything" sentinel used by load_data_for_fmt) doesn't overflow.
    let cap = chunk_rows.min(4096);
    let mut raw: Vec<Vec<String>> = vec![Vec::with_capacity(cap); n];
    let mut count = 0usize;
    let mut any_chunk_emitted = false;

    for (row_idx, result) in rdr.records().enumerate() {
        let record = result.with_context(|| format!("reading CSV row {}", row_idx + 1))?;
        for (ci, spec) in schema.columns.iter().enumerate() {
            let csv_idx = hdr_map.get(spec.name.as_str()).copied();
            let val = csv_idx
                .and_then(|i| record.get(i))
                .unwrap_or("")
                .to_string();
            raw[ci].push(val);
        }
        count += 1;
        if count == chunk_rows {
            let chunk = build_chunk_from_strings(schema, std::mem::take(&mut raw))?;
            on_chunk(chunk)?;
            raw = vec![Vec::with_capacity(cap); n];
            count = 0;
            any_chunk_emitted = true;
        }
    }
    // Flush the final partial chunk (or, for zero-row files, emit one empty
    // chunk so that load_data_for_fmt always receives at least one LoadedData).
    if count > 0 || !any_chunk_emitted {
        let chunk = build_chunk_from_strings(schema, raw)?;
        on_chunk(chunk)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Parquet chunked loader
// ---------------------------------------------------------------------------

/// Per-column accumulator for the Parquet chunked loader.
///
/// Binary and Nullable<Binary> columns collect raw bytes directly (no
/// UTF-8 conversion) to prevent `from_utf8_lossy` from silently corrupting
/// non-UTF-8 byte sequences.  All other columns accumulate strings and go
/// through the existing `build_chunk_from_strings` path.
#[cfg(feature = "schema-parquet")]
enum ColumnAccum {
    /// Numeric, Utf8, and all non-binary columns: values as strings.
    Strings(Vec<String>),
    /// Binary / Nullable<Binary> / NullableBinary columns: raw bytes.
    /// `None` represents a null entry (OPTIONAL Parquet column with no value).
    Bytes(Vec<Option<Vec<u8>>>),
}

/// Returns `true` if `lt` is a binary column type that requires the typed
/// bytes accumulator rather than the string path.
#[cfg(feature = "schema-parquet")]
fn is_binary_type(lt: &LogicalType) -> bool {
    match lt {
        LogicalType::Binary | LogicalType::NullableBinary => true,
        LogicalType::Nullable { inner } => matches!(inner.as_ref(), LogicalType::Binary),
        _ => false,
    }
}

/// Build a [`LogicalColumn`] from a typed bytes accumulator.
///
/// - `LogicalType::Binary` → all entries must be `Some`; builds `LogicalColumn::Binary`.
/// - `LogicalType::Nullable { Binary }` → builds `LogicalColumn::Nullable { present, Binary }`.
/// - `LogicalType::NullableBinary` (legacy) → builds `LogicalColumn::NullableBinary`.
#[cfg(feature = "schema-parquet")]
fn bytes_accum_to_logical_column(
    entries: Vec<Option<Vec<u8>>>,
    lt: &LogicalType,
) -> anyhow::Result<LogicalColumn> {
    match lt {
        LogicalType::Binary => {
            let blobs: Vec<Vec<u8>> = entries
                .into_iter()
                .enumerate()
                .map(|(i, opt)| {
                    opt.ok_or_else(|| {
                        anyhow::anyhow!(
                            "row {i}: null value for required Binary column; \
                             wrap the column in Nullable to allow nulls"
                        )
                    })
                })
                .collect::<anyhow::Result<_>>()?;
            Ok(LogicalColumn::Binary(blobs))
        }
        LogicalType::Nullable { inner } if matches!(inner.as_ref(), LogicalType::Binary) => {
            let mut present = Vec::with_capacity(entries.len());
            let mut blobs: Vec<Vec<u8>> = Vec::new();
            for entry in entries {
                match entry {
                    Some(b) => {
                        present.push(true);
                        blobs.push(b);
                    }
                    None => {
                        present.push(false);
                    }
                }
            }
            Ok(LogicalColumn::Nullable {
                present,
                value: Box::new(LogicalColumn::Binary(blobs)),
            })
        }
        LogicalType::NullableBinary => {
            let mut present = Vec::with_capacity(entries.len());
            let mut blobs: Vec<Vec<u8>> = Vec::new();
            for entry in entries {
                match entry {
                    Some(b) => {
                        present.push(true);
                        blobs.push(b);
                    }
                    None => {
                        present.push(false);
                    }
                }
            }
            Ok(LogicalColumn::NullableBinary { present, blobs })
        }
        other => anyhow::bail!(
            "bytes_accum_to_logical_column called with non-binary type {:?}",
            other
        ),
    }
}

/// Build a [`LoadedData`] chunk from mixed string / bytes accumulators.
///
/// Binary columns are built from the typed `Bytes` accumulator (lossless).
/// All other columns delegate to `strings_to_logical_column`.
#[cfg(feature = "schema-parquet")]
fn build_chunk_from_accums(
    schema: &Schema,
    accums: Vec<ColumnAccum>,
) -> anyhow::Result<LoadedData> {
    let mut result = Vec::with_capacity(schema.columns.len());
    // Consume accums by value so we can move out of each variant without cloning.
    for (accum, spec) in accums.into_iter().zip(schema.columns.iter()) {
        let lc = match accum {
            ColumnAccum::Strings(strings) => {
                strings_to_logical_column(&strings, &spec.logical_type, CSV_NULLS)
                    .with_context(|| format!("converting column '{}'", spec.name))?
            }
            ColumnAccum::Bytes(entries) => {
                bytes_accum_to_logical_column(entries, &spec.logical_type)
                    .with_context(|| format!("converting binary column '{}'", spec.name))?
            }
        };
        result.push((spec.name.clone(), spec.logical_type.clone(), lc));
    }
    Ok(result)
}

/// Reset all per-column accumulators to empty, keeping the correct variant.
#[cfg(feature = "schema-parquet")]
fn reset_accums(schema: &Schema, cap: usize) -> Vec<ColumnAccum> {
    schema
        .columns
        .iter()
        .map(|spec| {
            if is_binary_type(&spec.logical_type) {
                ColumnAccum::Bytes(Vec::with_capacity(cap))
            } else {
                ColumnAccum::Strings(Vec::with_capacity(cap))
            }
        })
        .collect()
}

#[cfg(feature = "schema-parquet")]
fn load_parquet_data_chunked(
    path: &Path,
    schema: &Schema,
    chunk_rows: usize,
    on_chunk: &mut dyn FnMut(LoadedData) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    use parquet::file::reader::{FileReader, SerializedFileReader};
    use parquet::record::Field;
    use std::fs::File;

    let file =
        File::open(path).with_context(|| format!("opening Parquet file '{}'", path.display()))?;
    let reader =
        SerializedFileReader::new(file).map_err(|e| anyhow::anyhow!("Parquet open error: {e}"))?;

    // Use a saturating initial capacity so that chunk_rows = usize::MAX (the
    // "load everything" sentinel used by load_data_for_fmt) doesn't overflow.
    let cap = chunk_rows.min(4096);
    let mut accums: Vec<ColumnAccum> = reset_accums(schema, cap);
    let mut count = 0usize;
    let mut any_chunk_emitted = false;

    for row_result in reader
        .get_row_iter(None)
        .map_err(|e| anyhow::anyhow!("{e}"))?
    {
        let row = row_result.map_err(|e| anyhow::anyhow!("row error: {e}"))?;
        let fields: HashMap<&str, &Field> = row
            .get_column_iter()
            .map(|(n, f)| (n.as_str(), f))
            .collect();
        for (ci, spec) in schema.columns.iter().enumerate() {
            let field = fields.get(spec.name.as_str());
            match &mut accums[ci] {
                ColumnAccum::Strings(strs) => {
                    let val = field
                        .map(|f| parquet_field_to_string(f))
                        .unwrap_or_default();
                    strs.push(val);
                }
                ColumnAccum::Bytes(bytes) => {
                    // Collect raw bytes for Binary / Nullable<Binary> columns,
                    // bypassing the lossy UTF-8 conversion in parquet_field_to_string.
                    let entry = match field {
                        Some(Field::Bytes(b)) => Some(b.data().to_vec()),
                        Some(Field::Null) | None => None,
                        Some(other) => {
                            // Unexpected field type for a Binary column — use the
                            // string fallback converted to bytes (best-effort).
                            Some(parquet_field_to_string(other).into_bytes())
                        }
                    };
                    bytes.push(entry);
                }
            }
        }
        count += 1;
        if count == chunk_rows {
            let chunk = build_chunk_from_accums(
                schema,
                std::mem::replace(&mut accums, reset_accums(schema, cap)),
            )?;
            on_chunk(chunk)?;
            count = 0;
            any_chunk_emitted = true;
        }
    }
    // Flush the final partial chunk (or, for zero-row files, emit one empty
    // chunk so that load_data_for_fmt always receives at least one LoadedData).
    if count > 0 || !any_chunk_emitted {
        let chunk = build_chunk_from_accums(schema, accums)?;
        on_chunk(chunk)?;
    }
    Ok(())
}

// When the parquet feature is not enabled, Parquet loading is unsupported.
// (In practice the `cli` feature always enables `schema-parquet`; this stub
// exists solely to keep the module compilable in hypothetical partial-feature
// builds.)
#[cfg(not(feature = "schema-parquet"))]
fn load_parquet_data_chunked(
    path: &Path,
    _schema: &Schema,
    _chunk_rows: usize,
    _on_chunk: &mut dyn FnMut(LoadedData) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    anyhow::bail!(
        "Parquet loading requires the 'schema-parquet' feature (file: '{}')",
        path.display()
    )
}

// ---------------------------------------------------------------------------
// NDJSON chunked loader
// ---------------------------------------------------------------------------

fn load_json_data_chunked(
    path: &Path,
    schema: &Schema,
    chunk_rows: usize,
    on_chunk: &mut dyn FnMut(LoadedData) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    use std::io::{BufRead, BufReader};

    let file = std::fs::File::open(path)
        .with_context(|| format!("opening JSON file '{}'", path.display()))?;

    // Peek at the first non-whitespace byte to detect array vs NDJSON.
    let content_peek = {
        let mut buf = BufReader::new(&file);
        let mut line = String::new();
        // Read enough bytes to find the first non-whitespace character.
        loop {
            line.clear();
            let n = buf
                .read_line(&mut line)
                .with_context(|| format!("reading JSON file '{}'", path.display()))?;
            if n == 0 {
                break;
            }
            if line.trim().starts_with('[') || !line.trim().is_empty() {
                break;
            }
        }
        line.trim().starts_with('[')
    };

    if content_peek {
        // JSON-array fallback: the whole document must be parsed before any
        // record is available (no SAX parser is used).  Read everything into
        // memory, build one in-memory LoadedData, then slice into chunks via
        // `slice_and_callback`.
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("reading JSON file '{}'", path.display()))?;
        let records = parse_json_records(&content)
            .with_context(|| format!("loading JSON array from '{}'", path.display()))?;
        let n = schema.columns.len();
        let data = build_chunk_from_json_records(schema, &records, n)?;
        return slice_and_callback(data, chunk_rows, on_chunk);
    }

    // NDJSON: line-at-a-time streaming.
    let file2 = std::fs::File::open(path)
        .with_context(|| format!("opening JSON file '{}'", path.display()))?;
    let reader = BufReader::new(file2);

    let n = schema.columns.len();
    // Use a saturating initial capacity so that chunk_rows = usize::MAX (the
    // "load everything" sentinel used by load_data_for_fmt) doesn't overflow.
    let cap = chunk_rows.min(4096);
    let mut chunk_records: Vec<serde_json::Map<String, serde_json::Value>> =
        Vec::with_capacity(cap);
    let mut any_chunk_emitted = false;

    for (line_idx, line_result) in reader.lines().enumerate() {
        let line = line_result
            .with_context(|| format!("reading line {} of '{}'", line_idx + 1, path.display()))?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(line)
            .with_context(|| format!("NDJSON parse error at line {}", line_idx + 1))?;
        match v {
            serde_json::Value::Object(m) => chunk_records.push(m),
            _ => anyhow::bail!("NDJSON line {} is not an object", line_idx + 1),
        }
        if chunk_records.len() == chunk_rows {
            let chunk = build_chunk_from_json_records(schema, &chunk_records, n)?;
            on_chunk(chunk)?;
            chunk_records = Vec::with_capacity(cap);
            any_chunk_emitted = true;
        }
    }
    // Flush the final partial chunk (or, for zero-row files, emit one empty
    // chunk so that load_data_for_fmt always receives at least one LoadedData).
    if !chunk_records.is_empty() || !any_chunk_emitted {
        let chunk = build_chunk_from_json_records(schema, &chunk_records, n)?;
        on_chunk(chunk)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Avro chunked loader
// ---------------------------------------------------------------------------

#[cfg(feature = "schema-avro")]
fn load_avro_data_chunked(
    path: &Path,
    schema: &Schema,
    chunk_rows: usize,
    on_chunk: &mut dyn FnMut(LoadedData) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    // Clamp chunk_rows to avoid capacity overflow in read_avro_data_chunked
    // when called with usize::MAX (the "load everything" sentinel).
    // A value of 0 means "no chunking" — read_avro_data_chunked treats 0 as
    // unbounded, so we use that as the sentinel instead.
    let effective_chunk = if chunk_rows == usize::MAX {
        0
    } else {
        chunk_rows
    };
    // read_avro_data_chunked streams records one-at-a-time internally;
    // peak memory is bounded by chunk_rows × column_count.
    helium::schema::read_avro_data_chunked(path, effective_chunk, |columns| {
        let mut data: LoadedData = Vec::with_capacity(schema.columns.len());
        for spec in &schema.columns {
            let lc = columns.get(&spec.name).cloned().ok_or_else(|| {
                helium::HeliumError::Format(format!("Avro chunk missing column '{}'", spec.name))
            })?;
            data.push((spec.name.clone(), spec.logical_type.clone(), lc));
        }
        on_chunk(data).map_err(|e| helium::HeliumError::Format(format!("on_chunk error: {e}")))
    })
    .map(|_schema| ())
    .map_err(|e| anyhow::anyhow!("{e}"))
}

// When the avro feature is not enabled, Avro loading is unsupported.
// (In practice the `cli` feature always enables `schema-avro`; this stub
// exists solely to keep the module compilable in hypothetical partial-feature
// builds.)
#[cfg(not(feature = "schema-avro"))]
fn load_avro_data_chunked(
    path: &Path,
    _schema: &Schema,
    _chunk_rows: usize,
    _on_chunk: &mut dyn FnMut(LoadedData) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    anyhow::bail!(
        "Avro loading requires the 'schema-avro' feature (file: '{}')",
        path.display()
    )
}

// ---------------------------------------------------------------------------
// Shared chunk-building helpers
// ---------------------------------------------------------------------------

/// Convert a column-major `Vec<Vec<String>>` into a [`LoadedData`] chunk.
///
/// `raw[ci]` must hold exactly the string values for column `ci` in this chunk.
fn build_chunk_from_strings(schema: &Schema, raw: Vec<Vec<String>>) -> anyhow::Result<LoadedData> {
    let mut result = Vec::with_capacity(schema.columns.len());
    for (ci, spec) in schema.columns.iter().enumerate() {
        let strings = &raw[ci];
        let lc = strings_to_logical_column(strings, &spec.logical_type, CSV_NULLS)
            .with_context(|| format!("converting column '{}'", spec.name))?;
        result.push((spec.name.clone(), spec.logical_type.clone(), lc));
    }
    Ok(result)
}

/// Convert a slice of NDJSON record objects into a [`LoadedData`] chunk.
fn build_chunk_from_json_records(
    schema: &Schema,
    records: &[serde_json::Map<String, serde_json::Value>],
    _n: usize,
) -> anyhow::Result<LoadedData> {
    let mut result = Vec::with_capacity(schema.columns.len());
    for spec in &schema.columns {
        let column_values: Vec<serde_json::Value> = records
            .iter()
            .map(|rec| {
                rec.get(&spec.name)
                    .cloned()
                    .unwrap_or(serde_json::Value::Null)
            })
            .collect();
        let lc = json_values_to_logical_column(&column_values, &spec.logical_type)
            .with_context(|| format!("converting column '{}'", spec.name))?;
        result.push((spec.name.clone(), spec.logical_type.clone(), lc));
    }
    Ok(result)
}

/// Fallback: slice an in-memory [`LoadedData`] into chunks and invoke the callback.
fn slice_and_callback(
    data: LoadedData,
    chunk_rows: usize,
    on_chunk: &mut dyn FnMut(LoadedData) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    let total = data.first().map(|(_, _, lc)| lc.row_count()).unwrap_or(0);
    if total == 0 {
        // Emit one empty chunk so the caller knows the schema exists.
        on_chunk(
            data.iter()
                .map(|(n, lt, lc)| {
                    let empty = lc
                        .slice(0, 0)
                        .map_err(|e| anyhow::anyhow!("slice(0,0) failed: {e}"))?;
                    Ok((n.clone(), lt.clone(), empty))
                })
                .collect::<anyhow::Result<_>>()?,
        )?;
        return Ok(());
    }
    let mut offset = 0usize;
    while offset < total {
        let chunk_len = chunk_rows.min(total - offset);
        let chunk: LoadedData = data
            .iter()
            .map(|(name, lt, lc)| {
                let slice = lc
                    .slice(offset, chunk_len)
                    .map_err(|e| anyhow::anyhow!("slicing column '{name}': {e}"))?;
                Ok((name.clone(), lt.clone(), slice))
            })
            .collect::<anyhow::Result<_>>()?;
        on_chunk(chunk)?;
        offset += chunk_len;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// CSV data loader helpers
// ---------------------------------------------------------------------------

/// Null sentinels recognised in CSV fields.
const CSV_NULLS: &[&str] = &["", "NULL", "null", "NA"];

/// Build a recursive-shaped [`LogicalColumn`] directly from a slice of
/// [`serde_json::Value`]s — one per row — matching `lt` recursively.
///
/// This is the canonical JSON data loader for all recursive `LogicalType` variants.
/// Legacy flat variants (`NullablePrim`, `NullableUtf8`, etc.) are also handled
/// by delegating to the flat string path for back-compat.
pub fn json_values_to_logical_column(
    values: &[serde_json::Value],
    lt: &LogicalType,
) -> anyhow::Result<LogicalColumn> {
    match lt {
        // ----------------------------------------------------------------
        // Leaf types
        // ----------------------------------------------------------------
        LogicalType::Primitive { data_type } => {
            let strings: Vec<String> = values
                .iter()
                .enumerate()
                .map(|(i, v)| match v {
                    serde_json::Value::Null => bail!(
                        "row {i}: null value where Primitive({data_type:?}) is required; \
                         wrap the column in Nullable to allow nulls"
                    ),
                    serde_json::Value::Bool(b) => Ok(if *b { "1" } else { "0" }.to_string()),
                    serde_json::Value::Number(n) => Ok(n.to_string()),
                    serde_json::Value::String(s) => Ok(s.clone()),
                    other => bail!(
                        "row {i}: expected a primitive JSON value (number/bool/string), \
                         got {}",
                        other
                    ),
                })
                .collect::<anyhow::Result<_>>()?;
            let cd = parse_primitive_vec(&strings, *data_type)?;
            Ok(LogicalColumn::Primitive(cd))
        }

        LogicalType::Utf8 => {
            let strings: Vec<String> = values
                .iter()
                .enumerate()
                .map(|(i, v)| match v {
                    serde_json::Value::Null => bail!(
                        "row {i}: null value where Utf8 is required; \
                         wrap the column in Nullable to allow nulls"
                    ),
                    serde_json::Value::String(s) => Ok(s.clone()),
                    // Allow Number/Bool coercion to string for Utf8.
                    serde_json::Value::Number(n) => Ok(n.to_string()),
                    serde_json::Value::Bool(b) => Ok(b.to_string()),
                    other => bail!(
                        "row {i}: expected a string JSON value for Utf8, got {}",
                        other
                    ),
                })
                .collect::<anyhow::Result<_>>()?;
            Ok(LogicalColumn::Utf8(strings))
        }

        LogicalType::Binary => {
            // Binary values are hex-encoded strings (mirrors write_json's hex_encode).
            let blobs: Vec<Vec<u8>> = values
                .iter()
                .enumerate()
                .map(|(i, v)| match v {
                    serde_json::Value::String(s) => {
                        hex_decode(s).with_context(|| format!("row {i}: invalid hex for Binary"))
                    }
                    serde_json::Value::Null => bail!(
                        "row {i}: null value where Binary is required; \
                         wrap the column in Nullable to allow nulls"
                    ),
                    other => bail!(
                        "row {i}: expected a hex-encoded string for Binary, got {}",
                        other
                    ),
                })
                .collect::<anyhow::Result<_>>()?;
            Ok(LogicalColumn::Binary(blobs))
        }

        // ----------------------------------------------------------------
        // Nullable wrapper
        // ----------------------------------------------------------------
        LogicalType::Nullable { inner } => {
            let mut present = Vec::with_capacity(values.len());
            let mut non_null: Vec<serde_json::Value> = Vec::new();
            for v in values {
                if matches!(v, serde_json::Value::Null) {
                    present.push(false);
                } else {
                    present.push(true);
                    non_null.push(v.clone());
                }
            }
            let value =
                json_values_to_logical_column(&non_null, inner).context("inside Nullable inner")?;
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
            let mut flat_items: Vec<serde_json::Value> = Vec::new();
            for (i, v) in values.iter().enumerate() {
                match v {
                    serde_json::Value::Array(arr) => {
                        flat_items.extend(arr.iter().cloned());
                        let len = flat_items.len();
                        offsets.push(
                            len.try_into()
                                .with_context(|| format!("row {i}: list offset overflows u32"))?,
                        );
                    }
                    serde_json::Value::Null => bail!(
                        "row {i}: null value where List is required; \
                         wrap the column in Nullable to allow nulls"
                    ),
                    other => bail!("row {i}: expected a JSON array for List, got {}", other),
                }
            }
            let inner_col =
                json_values_to_logical_column(&flat_items, inner).context("inside List inner")?;
            Ok(LogicalColumn::List {
                offsets,
                values: Box::new(inner_col),
            })
        }

        // ----------------------------------------------------------------
        // Map
        // ----------------------------------------------------------------
        LogicalType::Map { key, value } => {
            // Validate that the key type is string-compatible.
            if !matches!(
                key.as_ref(),
                LogicalType::Utf8 | LogicalType::Primitive { .. } | LogicalType::Binary
            ) {
                bail!(
                    "JSON Map loader requires Utf8/Primitive/Binary keys; key type was {:?}",
                    key
                );
            }
            let mut offsets: Vec<u32> = Vec::with_capacity(values.len() + 1);
            offsets.push(0);
            let mut flat_keys: Vec<serde_json::Value> = Vec::new();
            let mut flat_vals: Vec<serde_json::Value> = Vec::new();
            for (i, v) in values.iter().enumerate() {
                match v {
                    serde_json::Value::Object(obj) => {
                        for (k, vv) in obj {
                            flat_keys.push(serde_json::Value::String(k.clone()));
                            flat_vals.push(vv.clone());
                        }
                        let len = flat_keys.len();
                        offsets.push(
                            len.try_into()
                                .with_context(|| format!("row {i}: map offset overflows u32"))?,
                        );
                    }
                    serde_json::Value::Null => bail!(
                        "row {i}: null value where Map is required; \
                         wrap the column in Nullable to allow nulls"
                    ),
                    other => bail!("row {i}: expected a JSON object for Map, got {}", other),
                }
            }
            let keys_col =
                json_values_to_logical_column(&flat_keys, key).context("inside Map key")?;
            let vals_col =
                json_values_to_logical_column(&flat_vals, value).context("inside Map value")?;
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
            // For each field, collect one Value per row.
            let mut field_columns: Vec<(String, LogicalColumn)> = Vec::with_capacity(fields.len());
            for fspec in fields {
                let field_vals: Vec<serde_json::Value> = values
                    .iter()
                    .map(|v| match v {
                        serde_json::Value::Object(obj) => obj
                            .get(&fspec.name)
                            .cloned()
                            .unwrap_or(serde_json::Value::Null),
                        serde_json::Value::Null => serde_json::Value::Null,
                        _ => serde_json::Value::Null,
                    })
                    .collect();
                let fc = json_values_to_logical_column(&field_vals, &fspec.logical_type)
                    .with_context(|| format!("inside Struct field '{}'", fspec.name))?;
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
            // Heuristic variant selection based on JSON value shape.
            // The discriminant is the first variant whose shape matches the value.
            // Special case: null → pick the first Nullable variant if one exists.
            let n_variants = variants.len();
            let mut tags: Vec<u8> = Vec::with_capacity(values.len());
            // Per-variant compacted value buffers.
            let mut variant_bufs: Vec<Vec<serde_json::Value>> = vec![Vec::new(); n_variants];

            for (row_i, v) in values.iter().enumerate() {
                let tag = if matches!(v, serde_json::Value::Null) {
                    // Null: prefer first Nullable variant.
                    variants
                        .iter()
                        .position(|(_, lt)| matches!(lt, LogicalType::Nullable { .. }))
                        .unwrap_or(0)
                } else {
                    pick_union_variant(v, variants).with_context(|| {
                        format!(
                            "row {row_i}: could not match JSON value to any Union variant \
                             (variants: {})",
                            variants
                                .iter()
                                .map(|(n, _)| n.as_str())
                                .collect::<Vec<_>>()
                                .join(", ")
                        )
                    })?
                };
                tags.push(tag.try_into().with_context(|| {
                    format!("row {row_i}: union variant index {tag} exceeds u8::MAX")
                })?);
                variant_bufs[tag].push(v.clone());
            }

            let mut result_variants: Vec<(String, LogicalColumn)> = Vec::with_capacity(n_variants);
            for (vi, (vname, vlt)) in variants.iter().enumerate() {
                let vc = json_values_to_logical_column(&variant_bufs[vi], vlt)
                    .with_context(|| format!("inside Union variant '{vname}'"))?;
                result_variants.push((vname.clone(), vc));
            }
            Ok(LogicalColumn::Union {
                tags,
                variants: result_variants,
            })
        }

        // ----------------------------------------------------------------
        // Legacy flat variants — delegate to the string path for back-compat
        // ----------------------------------------------------------------
        LogicalType::NullablePrim { data_type } => {
            let nulls = json_null_sentinels();
            let strs = json_values_to_strings(values);
            let (present, non_null_strs) = split_nullable(&strs, &nulls);
            let cd = parse_primitive_vec(&non_null_strs, *data_type)?;
            Ok(LogicalColumn::NullablePrim {
                present,
                values: cd,
            })
        }
        LogicalType::NullableUtf8 => {
            let nulls = json_null_sentinels();
            let strs = json_values_to_strings(values);
            let (present, non_null_strs) = split_nullable(&strs, &nulls);
            Ok(LogicalColumn::NullableUtf8 {
                present,
                strings: non_null_strs,
            })
        }
        LogicalType::NullableBinary => {
            let nulls = json_null_sentinels();
            let strs = json_values_to_strings(values);
            let (present, non_null_strs) = split_nullable(&strs, &nulls);
            let blobs: Vec<Vec<u8>> = non_null_strs
                .iter()
                .map(|s| s.as_bytes().to_vec())
                .collect();
            Ok(LogicalColumn::NullableBinary { present, blobs })
        }
        LogicalType::ArrayOf { data_type } => {
            let strs = json_values_to_strings(values);
            let nulls = json_null_sentinels();
            strings_to_logical_column(
                &strs,
                &LogicalType::ArrayOf {
                    data_type: *data_type,
                },
                &nulls,
            )
        }
        LogicalType::ArrayOfUtf8 => {
            let strs = json_values_to_strings(values);
            let nulls = json_null_sentinels();
            strings_to_logical_column(&strs, &LogicalType::ArrayOfUtf8, &nulls)
        }
        LogicalType::Dictionary { .. } => {
            bail!(
                "dict types are not directly loadable from JSON; \
                 the optimizer or a pre-built dict pipeline is required"
            )
        }
        // Semantic types: delegate to string parser.
        LogicalType::Decimal128 { .. }
        | LogicalType::Date { .. }
        | LogicalType::Datetime { .. } => {
            let strs = json_values_to_strings(values);
            let nulls = json_null_sentinels();
            strings_to_logical_column(&strs, lt, &nulls)
        }
    }
}

/// Pick the Union variant index whose shape best matches the JSON value.
///
/// Matching priority (first match wins):
/// 1. Null values are handled by the caller (prefer Nullable variant).
/// 2. Number → Primitive variant.
/// 3. Bool → Primitive(U8) variant, else Utf8.
/// 4. String → Utf8 variant.
/// 5. Object → Struct or Map variant.
/// 6. Array → List variant.
fn pick_union_variant(
    v: &serde_json::Value,
    variants: &[(String, LogicalType)],
) -> anyhow::Result<usize> {
    let shape_ok = |lt: &LogicalType, val: &serde_json::Value| -> bool {
        matches!(
            (val, lt),
            (serde_json::Value::Number(_), LogicalType::Primitive { .. })
                | (serde_json::Value::Bool(_), LogicalType::Primitive { .. })
                | (serde_json::Value::String(_), LogicalType::Utf8)
                | (serde_json::Value::String(_), LogicalType::Binary)
                | (serde_json::Value::Object(_), LogicalType::Struct { .. })
                | (serde_json::Value::Object(_), LogicalType::Map { .. })
                | (serde_json::Value::Array(_), LogicalType::List { .. })
        )
    };
    variants
        .iter()
        .position(|(_, lt)| shape_ok(lt, v))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "no Union variant matches JSON shape '{}'",
                match v {
                    serde_json::Value::Number(_) => "number",
                    serde_json::Value::Bool(_) => "bool",
                    serde_json::Value::String(_) => "string",
                    serde_json::Value::Object(_) => "object",
                    serde_json::Value::Array(_) => "array",
                    serde_json::Value::Null => "null",
                }
            )
        })
}

/// Null sentinels used when falling back to string-based loading for legacy flat types.
fn json_null_sentinels() -> Vec<&'static str> {
    let mut nulls: Vec<&str> = CSV_NULLS.to_vec();
    nulls.push("null");
    nulls
}

/// Flatten JSON values to strings for legacy flat type back-compat paths.
fn json_values_to_strings(values: &[serde_json::Value]) -> Vec<String> {
    values.iter().map(json_value_to_string).collect()
}

/// Convert a [`serde_json::Value`] to a plain string (legacy flat-loading helper).
fn json_value_to_string(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::Null => String::new(),
        serde_json::Value::Bool(b) => if *b { "1" } else { "0" }.to_string(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::String(s) => s.clone(),
        // Nested objects / arrays → serialise as JSON string.
        other => other.to_string(),
    }
}

/// Hex-decode a lowercase hex string to bytes.
fn hex_decode(s: &str) -> anyhow::Result<Vec<u8>> {
    if !s.len().is_multiple_of(2) {
        bail!("hex string has odd length: {s:?}");
    }
    (0..s.len() / 2)
        .map(|i| {
            u8::from_str_radix(&s[2 * i..2 * i + 2], 16)
                .with_context(|| format!("invalid hex byte at position {}", 2 * i))
        })
        .collect()
}

/// Parse a JSON string into top-level objects (array or NDJSON).
fn parse_json_records(
    input: &str,
) -> anyhow::Result<Vec<serde_json::Map<String, serde_json::Value>>> {
    let input = input.trim();
    if input.starts_with('[') {
        let arr: serde_json::Value = serde_json::from_str(input).context("parsing JSON array")?;
        match arr {
            serde_json::Value::Array(items) => items
                .into_iter()
                .map(|v| match v {
                    serde_json::Value::Object(m) => Ok(m),
                    _ => bail!("JSON array must contain objects"),
                })
                .collect(),
            _ => bail!("top-level JSON must be an array of objects or NDJSON"),
        }
    } else {
        let mut records = Vec::new();
        for (i, line) in input.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let v: serde_json::Value = serde_json::from_str(line)
                .with_context(|| format!("NDJSON parse error at line {}", i + 1))?;
            match v {
                serde_json::Value::Object(m) => records.push(m),
                _ => bail!("NDJSON line {} is not an object", i + 1),
            }
        }
        Ok(records)
    }
}

/// Convert a Parquet [`Field`] value to a plain string.
///
/// Note: `Field::Bytes` is handled here as a defensive fallback only.
/// For columns whose `LogicalType` is `Binary` or `Nullable<Binary>`, the
/// Parquet loader routes through `ColumnAccum::Bytes` and never calls this
/// function, so the `Field::Bytes` arm below is unreachable for correctly
/// typed schemas.  It is kept as a last-resort fallback for unexpected cases
/// (e.g. a schema type mismatch at runtime) to avoid a panic, but callers
/// should not rely on it for binary fidelity — use the typed bytes path instead.
fn parquet_field_to_string(f: &parquet::record::Field) -> String {
    use parquet::record::Field;
    match f {
        Field::Null => String::new(),
        Field::Bool(b) => if *b { "1" } else { "0" }.to_string(),
        Field::Byte(x) => x.to_string(),
        Field::Short(x) => x.to_string(),
        Field::Int(x) => x.to_string(),
        Field::Long(x) => x.to_string(),
        Field::UByte(x) => x.to_string(),
        Field::UShort(x) => x.to_string(),
        Field::UInt(x) => x.to_string(),
        Field::ULong(x) => x.to_string(),
        Field::Float(x) => x.to_string(),
        Field::Double(x) => x.to_string(),
        Field::Decimal(d) => format!("{:?}", d),
        Field::Str(s) => s.clone(),
        // Defensive fallback: Binary columns are handled via the typed
        // ColumnAccum::Bytes path and should not reach here for correct schemas.
        Field::Bytes(b) => String::from_utf8_lossy(b.data()).into_owned(),
        Field::Date(d) => d.to_string(),
        Field::TimestampMillis(t) => t.to_string(),
        Field::TimestampMicros(t) => t.to_string(),
        // Nested types: serialise as display string.
        other => format!("{other}"),
    }
}

// ---------------------------------------------------------------------------
// String → LogicalColumn converter (shared by CSV and JSON loaders)
// ---------------------------------------------------------------------------

/// Convert accumulated string values to a [`LogicalColumn`] given a target
/// [`LogicalType`].
///
/// Only flat types are supported:
/// - `Primitive { data_type }`
/// - `Utf8`
/// - `Binary`
/// - `Nullable { inner: Primitive | Utf8 | Binary }`
///
/// Other types return an error with a clear message.
pub fn strings_to_logical_column(
    values: &[String],
    lt: &LogicalType,
    nulls: &[&str],
) -> anyhow::Result<LogicalColumn> {
    match lt {
        LogicalType::Primitive { data_type } => {
            let cd = parse_primitive_vec(values, *data_type)?;
            Ok(LogicalColumn::Primitive(cd))
        }
        LogicalType::Utf8 => Ok(LogicalColumn::Utf8(values.to_vec())),
        LogicalType::Binary => {
            let blobs: Vec<Vec<u8>> = values.iter().map(|s| s.as_bytes().to_vec()).collect();
            Ok(LogicalColumn::Binary(blobs))
        }
        LogicalType::Nullable { inner } => match inner.as_ref() {
            LogicalType::Primitive { data_type } => {
                let (present, non_null_strs) = split_nullable(values, nulls);
                let cd = parse_primitive_vec(&non_null_strs, *data_type)?;
                Ok(LogicalColumn::Nullable {
                    present,
                    value: Box::new(LogicalColumn::Primitive(cd)),
                })
            }
            LogicalType::Utf8 => {
                let (present, non_null_strs) = split_nullable(values, nulls);
                Ok(LogicalColumn::Nullable {
                    present,
                    value: Box::new(LogicalColumn::Utf8(non_null_strs)),
                })
            }
            LogicalType::Binary => {
                let (present, non_null_strs) = split_nullable(values, nulls);
                let blobs: Vec<Vec<u8>> = non_null_strs
                    .iter()
                    .map(|s| s.as_bytes().to_vec())
                    .collect();
                Ok(LogicalColumn::Nullable {
                    present,
                    value: Box::new(LogicalColumn::Binary(blobs)),
                })
            }
            inner => bail!(
                "unsupported nullable inner type for data loading: {:?} \
                 (only Primitive / Utf8 / Binary are supported for flat-file loading)",
                inner
            ),
        },
        // Structured types are not supported by the flat loader.
        LogicalType::Struct { .. }
        | LogicalType::List { .. }
        | LogicalType::Map { .. }
        | LogicalType::Union { .. } => bail!(
            "nested/structured type {:?} is not supported by the flat CSV/JSON loader; \
             use a Parquet source with a matching nested schema instead",
            lt
        ),
        // Legacy flat types — treat same as the recursive equivalents.
        LogicalType::NullablePrim { data_type } => {
            let (present, non_null_strs) = split_nullable(values, nulls);
            let cd = parse_primitive_vec(&non_null_strs, *data_type)?;
            Ok(LogicalColumn::NullablePrim {
                present,
                values: cd,
            })
        }
        LogicalType::NullableUtf8 => {
            let (present, non_null_strs) = split_nullable(values, nulls);
            Ok(LogicalColumn::NullableUtf8 {
                present,
                strings: non_null_strs,
            })
        }
        LogicalType::NullableBinary => {
            let (present, non_null_strs) = split_nullable(values, nulls);
            let blobs: Vec<Vec<u8>> = non_null_strs
                .iter()
                .map(|s| s.as_bytes().to_vec())
                .collect();
            Ok(LogicalColumn::NullableBinary { present, blobs })
        }
        LogicalType::Dictionary { .. } => {
            bail!(
                "dict types are not directly loadable from flat files; \
                 the optimizer or a pre-built dict pipeline is required"
            )
        }
        LogicalType::ArrayOf { .. } | LogicalType::ArrayOfUtf8 => {
            bail!("legacy flat array types are not supported by the flat loader")
        }
        // Semantic types: parse from string representation.
        LogicalType::Decimal128 { .. } => {
            let v: Vec<i128> = values
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    parse_decimal128_str(s).ok_or_else(|| {
                        anyhow::anyhow!("row {i}: cannot parse {:?} as Decimal128", s)
                    })
                })
                .collect::<anyhow::Result<_>>()?;
            Ok(LogicalColumn::Decimal128 { values: v })
        }
        LogicalType::Date {
            unit: helium::DateUnit::Days,
        } => {
            let v: Vec<i32> = values
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    parse_date_days(s).ok_or_else(|| {
                        anyhow::anyhow!("row {i}: cannot parse {:?} as Date (Days)", s)
                    })
                })
                .collect::<anyhow::Result<_>>()?;
            Ok(LogicalColumn::Date32 { values: v })
        }
        LogicalType::Date {
            unit: helium::DateUnit::Millis,
        } => {
            let v: Vec<i64> = values
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    parse_date_millis(s).ok_or_else(|| {
                        anyhow::anyhow!("row {i}: cannot parse {:?} as Date (Millis)", s)
                    })
                })
                .collect::<anyhow::Result<_>>()?;
            Ok(LogicalColumn::Date64 { values: v })
        }
        LogicalType::Datetime { .. } => {
            let v: Vec<i64> = values
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    s.parse::<i64>()
                        .ok()
                        .or_else(|| parse_datetime_millis(s))
                        .ok_or_else(|| anyhow::anyhow!("row {i}: cannot parse {:?} as Datetime", s))
                })
                .collect::<anyhow::Result<_>>()?;
            Ok(LogicalColumn::Datetime { values: v })
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Split nullable: return `(present_bitmap, non_null_values)`.
///
/// `present[i] = true` means row `i` is not null; the non-null values are
/// collected in order into `non_null_values` (same contract as `LogicalColumn`
/// nullable variants).
fn split_nullable(values: &[String], nulls: &[&str]) -> (Vec<bool>, Vec<String>) {
    let mut present = Vec::with_capacity(values.len());
    let mut non_null = Vec::new();
    for v in values {
        if nulls.contains(&v.as_str()) {
            present.push(false);
        } else {
            present.push(true);
            non_null.push(v.clone());
        }
    }
    (present, non_null)
}

/// Parse a slice of string values into a [`ColumnData`] of the requested type.
fn parse_primitive_vec(values: &[String], dt: DataType) -> anyhow::Result<ColumnData> {
    macro_rules! parse {
        ($rust_ty:ty, $ctor:ident) => {{
            let v: Vec<$rust_ty> = values
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    s.parse::<$rust_ty>().with_context(|| {
                        format!("row {i}: cannot parse {:?} as {}", s, stringify!($rust_ty))
                    })
                })
                .collect::<anyhow::Result<_>>()?;
            Ok(ColumnData::$ctor(v))
        }};
    }

    match dt {
        DataType::I8 => parse!(i8, I8),
        DataType::I16 => parse!(i16, I16),
        DataType::I32 => parse!(i32, I32),
        DataType::I64 => parse!(i64, I64),
        DataType::U8 => parse!(u8, U8),
        DataType::U16 => parse!(u16, U16),
        DataType::U32 => parse!(u32, U32),
        DataType::U64 => parse!(u64, U64),
        DataType::F32 => parse!(f32, F32),
        DataType::F64 => parse!(f64, F64),
        DataType::Bytes => bail!("DataType::Bytes is an internal type and cannot be loaded"),
    }
}

// ---------------------------------------------------------------------------
// Raw size estimate (used by the compare subcommand)
// ---------------------------------------------------------------------------

/// Estimate the uncompressed byte size of a [`LogicalColumn`].
///
/// Used by the `compare` subcommand to compute compression ratios.
/// The estimate counts payload bytes only (values + offsets for variable-length
/// columns); framing overhead is excluded.
pub fn raw_bytes(lc: &LogicalColumn) -> usize {
    match lc {
        LogicalColumn::Primitive(cd) => column_data_bytes(cd),
        LogicalColumn::Utf8(v) => v.iter().map(|s| s.len() + 4).sum::<usize>(),
        LogicalColumn::Binary(v) => v.iter().map(|b| b.len() + 4).sum::<usize>(),
        LogicalColumn::NullablePrim { present, values } => {
            present.len().div_ceil(8) + column_data_bytes(values)
        }
        LogicalColumn::NullableUtf8 { present, strings } => {
            present.len().div_ceil(8) + strings.iter().map(|s| s.len() + 4).sum::<usize>()
        }
        LogicalColumn::NullableBinary { present, blobs } => {
            present.len().div_ceil(8) + blobs.iter().map(|b| b.len() + 4).sum::<usize>()
        }
        LogicalColumn::ArrayOf { offsets, values } => offsets.len() * 4 + column_data_bytes(values),
        LogicalColumn::ArrayOfUtf8 { offsets, strings } => {
            offsets.len() * 4 + strings.iter().map(|s| s.len() + 4).sum::<usize>()
        }
        // Recursive types: recurse into physical sub-columns via a rough estimate.
        LogicalColumn::Struct { fields } => fields.iter().map(|(_, lc)| raw_bytes(lc)).sum(),
        LogicalColumn::List { offsets, values } => offsets.len() * 4 + raw_bytes(values),
        LogicalColumn::Map {
            offsets,
            keys,
            values,
        } => offsets.len() * 4 + raw_bytes(keys) + raw_bytes(values),
        LogicalColumn::Nullable { present, value } => present.len().div_ceil(8) + raw_bytes(value),
        LogicalColumn::Union { tags, variants } => {
            tags.len() + variants.iter().map(|(_, lc)| raw_bytes(lc)).sum::<usize>()
        }
        // Semantic types: fixed-width payloads.
        LogicalColumn::Decimal128 { values } => values.len() * 16,
        LogicalColumn::Date32 { values } => values.len() * 4,
        LogicalColumn::Date64 { values } | LogicalColumn::Datetime { values } => values.len() * 8,
        // Dictionary{inner}: dictionary bytes + 4 bytes per index.
        LogicalColumn::Dictionary {
            dictionary,
            indices,
        } => raw_bytes(dictionary) + indices.len() * 4,
    }
}

fn column_data_bytes(cd: &ColumnData) -> usize {
    match cd {
        ColumnData::I8(v) => v.len(),
        ColumnData::U8(v) => v.len(),
        ColumnData::I16(v) => v.len() * 2,
        ColumnData::U16(v) => v.len() * 2,
        ColumnData::I32(v) => v.len() * 4,
        ColumnData::U32(v) => v.len() * 4,
        ColumnData::I64(v) => v.len() * 8,
        ColumnData::U64(v) => v.len() * 8,
        ColumnData::F32(v) => v.len() * 4,
        ColumnData::F64(v) => v.len() * 8,
        ColumnData::Bytes(b) => b.len(),
    }
}

// ---------------------------------------------------------------------------
// Semantic type parse helpers (Decimal128, Date, Datetime)
// ---------------------------------------------------------------------------

/// Parse a fixed-point decimal string (e.g. `"12.345"`, `"-0.01"`, `"42"`)
/// as an `i128` **unscaled integer**.  Returns `None` if parsing fails.
///
/// For now we treat the stored value as a raw integer and accept plain `i128`
/// string representations only.  CSV round-trip restores the human-readable
/// form via `format_decimal128`.
fn parse_decimal128_str(s: &str) -> Option<i128> {
    s.trim().parse::<i128>().ok().or_else(|| {
        // Allow "12.345" style: strip the decimal point and parse.
        // This is a best-effort loader; callers that need precise scale
        // should use the Parquet or Avro path.
        let s = s.trim();
        let neg = s.starts_with('-');
        let abs = s.trim_start_matches('-');
        if let Some(dot_pos) = abs.find('.') {
            let int_part = &abs[..dot_pos];
            let frac_part = &abs[dot_pos + 1..];
            let combined = format!("{int_part}{frac_part}");
            let n: i128 = combined.parse().ok()?;
            Some(if neg { -n } else { n })
        } else {
            None
        }
    })
}

/// Parse a date string (ISO 8601 `YYYY-MM-DD`) as days since Unix epoch (i32).
/// Also accepts plain integer strings.
fn parse_date_days(s: &str) -> Option<i32> {
    s.trim()
        .parse::<i32>()
        .ok()
        .or_else(|| date_string_to_days(s.trim()).map(|d| d as i32))
}

/// Parse a date string (ISO 8601 `YYYY-MM-DD`) as milliseconds since Unix epoch.
/// Also accepts plain integer strings.
fn parse_date_millis(s: &str) -> Option<i64> {
    s.trim()
        .parse::<i64>()
        .ok()
        .or_else(|| date_string_to_days(s.trim()).map(|d| d * 86_400_000))
}

/// Parse a datetime string (ISO 8601 `YYYY-MM-DDTHH:MM:SSZ` or similar) as
/// milliseconds since Unix epoch.  Also accepts plain `i64` strings.
fn parse_datetime_millis(s: &str) -> Option<i64> {
    // Try the compact form emitted by format_datetime: YYYY-MM-DDTHH:MM:SSZ
    // or YYYY-MM-DDTHH:MM:SS.mmmZ.
    let s = s.trim();
    if s.len() < 20 {
        return None;
    }
    let (date_part, rest) = s.split_once('T')?;
    let days = date_string_to_days(date_part)?;
    // Strip trailing 'Z'
    let rest = rest.strip_suffix('Z').unwrap_or(rest);
    // Parse HH:MM:SS or HH:MM:SS.mmm
    let (time_part, ms_part) = if let Some((t, m)) = rest.split_once('.') {
        (t, m)
    } else {
        (rest, "")
    };
    let parts: Vec<&str> = time_part.splitn(3, ':').collect();
    if parts.len() != 3 {
        return None;
    }
    let hh: i64 = parts[0].parse().ok()?;
    let mm: i64 = parts[1].parse().ok()?;
    let ss: i64 = parts[2].parse().ok()?;
    let ms: i64 = if ms_part.is_empty() {
        0
    } else {
        // Treat up to 3 digits as milliseconds, padding or truncating as needed.
        let padded = format!("{ms_part:0<3}");
        padded[..3].parse().ok()?
    };
    let total_ms = days * 86_400_000 + hh * 3_600_000 + mm * 60_000 + ss * 1_000 + ms;
    Some(total_ms)
}

/// Convert an ISO 8601 `YYYY-MM-DD` date string to days since Unix epoch.
/// Uses the proleptic Gregorian calendar (inverse of `days_to_date_string`).
fn date_string_to_days(s: &str) -> Option<i64> {
    // Expect exactly "YYYY-MM-DD" (10 chars).
    if s.len() < 10 {
        return None;
    }
    let year: i64 = s[0..4].parse().ok()?;
    let month: i64 = s[5..7].parse().ok()?;
    let day: i64 = s[8..10].parse().ok()?;
    // Convert civil date to Julian Day Number (Richards' algorithm, reversed).
    // jdn = d + (153*m + 2)/5 + 365*y + y/4 - y/100 + y/400 - 32045
    let a = (14 - month) / 12;
    let y = year + 4800 - a;
    let m = month + 12 * a - 3;
    let jdn = day + (153 * m + 2) / 5 + 365 * y + y / 4 - y / 100 + y / 400 - 32045;
    Some(jdn - 2_440_588)
}

// ---------------------------------------------------------------------------
// Unit tests for json_values_to_logical_column
// ---------------------------------------------------------------------------

#[cfg(test)]
mod nested_json_loader_tests {
    use super::*;
    use helium::{ColumnData, DataType, FieldSpec, LogicalColumn, LogicalType};
    use serde_json::json;

    fn v(values: impl IntoIterator<Item = serde_json::Value>) -> Vec<serde_json::Value> {
        values.into_iter().collect()
    }

    // 1. Struct { id: I32, name: Utf8 } over 3 rows.
    #[test]
    fn struct_id_name_three_rows() {
        let lt = LogicalType::Struct {
            fields: vec![
                FieldSpec::new(
                    "id",
                    LogicalType::Primitive {
                        data_type: DataType::I32,
                    },
                    vec![],
                ),
                FieldSpec::new("name", LogicalType::Utf8, vec![]),
            ],
        };
        let rows = v([
            json!({"id": 1, "name": "alice"}),
            json!({"id": 2, "name": "bob"}),
            json!({"id": 3, "name": "carol"}),
        ]);
        let lc = json_values_to_logical_column(&rows, &lt).unwrap();
        match lc {
            LogicalColumn::Struct { fields } => {
                assert_eq!(fields.len(), 2);
                assert_eq!(fields[0].0, "id");
                assert!(
                    matches!(&fields[0].1, LogicalColumn::Primitive(ColumnData::I32(v)) if *v == vec![1, 2, 3])
                );
                assert_eq!(fields[1].0, "name");
                assert!(
                    matches!(&fields[1].1, LogicalColumn::Utf8(v) if *v == vec!["alice", "bob", "carol"])
                );
            }
            other => panic!("expected Struct, got {other:?}"),
        }
    }

    // 2. List<I32> over 3 rows including an empty list.
    #[test]
    fn list_i32_with_empty_row() {
        let lt = LogicalType::List {
            inner: Box::new(LogicalType::Primitive {
                data_type: DataType::I32,
            }),
        };
        let rows = v([json!([1, 2, 3]), json!([]), json!([4, 5])]);
        let lc = json_values_to_logical_column(&rows, &lt).unwrap();
        match lc {
            LogicalColumn::List { offsets, values } => {
                assert_eq!(offsets, vec![0, 3, 3, 5]);
                assert!(
                    matches!(values.as_ref(), LogicalColumn::Primitive(ColumnData::I32(v)) if *v == vec![1, 2, 3, 4, 5])
                );
            }
            other => panic!("expected List, got {other:?}"),
        }
    }

    // 3. Map<Utf8, I32> over 2 rows.
    #[test]
    fn map_utf8_to_i32_two_rows() {
        let lt = LogicalType::Map {
            key: Box::new(LogicalType::Utf8),
            value: Box::new(LogicalType::Primitive {
                data_type: DataType::I32,
            }),
        };
        let rows = v([json!({"a": 1, "b": 2}), json!({"c": 3})]);
        let lc = json_values_to_logical_column(&rows, &lt).unwrap();
        match lc {
            LogicalColumn::Map {
                offsets,
                keys,
                values,
            } => {
                assert_eq!(offsets.len(), 3); // N+1
                assert_eq!(offsets[0], 0);
                assert_eq!(offsets[2], 3); // 2+1=3 total key-value pairs
                // Keys should be strings.
                assert!(matches!(keys.as_ref(), LogicalColumn::Utf8(_)));
                // Values should be I32.
                assert!(matches!(
                    values.as_ref(),
                    LogicalColumn::Primitive(ColumnData::I32(_))
                ));
            }
            other => panic!("expected Map, got {other:?}"),
        }
    }

    // 4. Nullable<List<Utf8>>: one null row and one non-null 2-element list.
    #[test]
    fn nullable_list_of_utf8() {
        let lt = LogicalType::Nullable {
            inner: Box::new(LogicalType::List {
                inner: Box::new(LogicalType::Utf8),
            }),
        };
        let rows = v([serde_json::Value::Null, json!(["hello", "world"])]);
        let lc = json_values_to_logical_column(&rows, &lt).unwrap();
        match lc {
            LogicalColumn::Nullable { present, value } => {
                assert_eq!(present, vec![false, true]);
                match value.as_ref() {
                    LogicalColumn::List { offsets, values } => {
                        assert_eq!(offsets, &vec![0, 2]);
                        assert!(
                            matches!(values.as_ref(), LogicalColumn::Utf8(v) if *v == vec!["hello", "world"])
                        );
                    }
                    other => panic!("expected List inside Nullable, got {other:?}"),
                }
            }
            other => panic!("expected Nullable, got {other:?}"),
        }
    }

    // 5. Struct { tags: List<Utf8>, score: Nullable<F64> } — composition test.
    #[test]
    fn struct_with_list_and_nullable_f64() {
        let lt = LogicalType::Struct {
            fields: vec![
                FieldSpec::new(
                    "tags",
                    LogicalType::List {
                        inner: Box::new(LogicalType::Utf8),
                    },
                    vec![],
                ),
                FieldSpec::new(
                    "score",
                    LogicalType::Nullable {
                        inner: Box::new(LogicalType::Primitive {
                            data_type: DataType::F64,
                        }),
                    },
                    vec![],
                ),
            ],
        };
        let rows = v([
            json!({"tags": ["rust", "fast"], "score": 9.5}),
            json!({"tags": [], "score": null}),
            json!({"tags": ["slow"], "score": 3.15}),
        ]);
        let lc = json_values_to_logical_column(&rows, &lt).unwrap();
        match lc {
            LogicalColumn::Struct { fields } => {
                // Check tags field.
                assert!(matches!(&fields[0].1, LogicalColumn::List { .. }));
                // Check score field.
                match &fields[1].1 {
                    LogicalColumn::Nullable { present, value } => {
                        assert_eq!(present, &vec![true, false, true]);
                        assert!(
                            matches!(value.as_ref(), LogicalColumn::Primitive(ColumnData::F64(v)) if v.len() == 2)
                        );
                    }
                    other => panic!("expected Nullable<F64>, got {other:?}"),
                }
            }
            other => panic!("expected Struct, got {other:?}"),
        }
    }

    // 6. Schema mismatch error: feed a Number where Schema says Struct.
    #[test]
    fn mismatch_number_for_struct_is_error() {
        let lt = LogicalType::Struct {
            fields: vec![FieldSpec::new(
                "x",
                LogicalType::Primitive {
                    data_type: DataType::I32,
                },
                vec![],
            )],
        };
        let rows = v([json!(42)]);
        let result = json_values_to_logical_column(&rows, &lt);
        assert!(
            result.is_err(),
            "expected error for number where Struct required"
        );
        let msg = result.unwrap_err().to_string();
        // Error bubbles up from the field projection — "inside Struct field" context.
        assert!(
            msg.contains("Primitive")
                || msg.contains("null")
                || msg.contains("Struct")
                || msg.contains("inside"),
            "error should mention the type mismatch: {msg}"
        );
    }

    // 7. Flat Primitive round-trip (regression: new path handles flat JSON too).
    #[test]
    fn flat_primitive_json_values() {
        let lt = LogicalType::Primitive {
            data_type: DataType::I64,
        };
        let rows = v([json!(1), json!(2), json!(3)]);
        let lc = json_values_to_logical_column(&rows, &lt).unwrap();
        assert!(matches!(lc, LogicalColumn::Primitive(ColumnData::I64(v)) if v == vec![1, 2, 3]));
    }
}
