//! `helium stats` — file size and value statistics for a `.he` file.
//!
//! Reads the file's footer metadata to compute per-column on-disk byte sizes,
//! then optionally reads each column to compute min/max value statistics.
//! Supports markdown table output (default) and machine-readable JSON output.
//!
//! When footer-embedded min/max stats are present (files written with the
//! default stats-enabled writer), statistics are read directly from the
//! footer without scanning column data — this makes `helium stats` near-
//! instant even for very large files.

use std::fs::File;
use std::path::Path;

use anyhow::Context;
use helium::catalog::Catalog;
use helium::{CoderRegistry, HeliumReader, LogicalColumn, LogicalType, MinMaxValue};

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Run the `stats` subcommand.
///
/// Reads `path` and prints either a markdown table (default) or JSON to stdout.
/// When `no_values` is `true`, min/max computation is skipped and only byte
/// sizes are reported.  When `catalog_dir` is `Some`, uses
/// [`HeliumReader::new_with_resolver`] so catalog-mode files are
/// readable.
pub fn run(
    path: &Path,
    no_values: bool,
    as_json: bool,
    catalog_dir: Option<&Path>,
) -> anyhow::Result<()> {
    let registry = CoderRegistry::default();
    let file = File::open(path).with_context(|| format!("opening '{}'", path.display()))?;

    let mut reader = if let Some(cat_dir) = catalog_dir {
        let catalog = Catalog::open(cat_dir)
            .with_context(|| format!("opening catalog at '{}'", cat_dir.display()))?;
        HeliumReader::new_with_resolver(file, &registry, catalog.resolver())
            .with_context(|| format!("reading '{}' with catalog resolver", path.display()))?
    } else {
        HeliumReader::new(file, &registry)
            .with_context(|| format!("reading '{}'", path.display()))?
    };

    // Gather structural metadata (no I/O beyond what was done at open time).
    let meta = FileMeta {
        path: path.display().to_string(),
        format_version: reader.version_str().to_string(),
        row_count: reader.row_count(),
        stripe_count: reader.stripe_count(),
        region: reader.region_sizes(),
    };
    let col_byte_sizes = reader.column_byte_sizes();
    let schema = reader.schema().clone();

    if meta.row_count == 0 || meta.stripe_count == 0 {
        print_empty(&meta, as_json)?;
        return Ok(());
    }

    // Build per-column stats table (initially with Skipped min/max).
    let mut col_stats: Vec<ColumnStats> = col_byte_sizes
        .iter()
        .zip(schema.columns.iter())
        .map(|((name, bytes), spec)| ColumnStats {
            name: name.clone(),
            type_str: fmt_logical_type(&spec.logical_type),
            bytes: *bytes,
            min: StatValue::Skipped,
            max: StatValue::Skipped,
            rows_total: meta.row_count,
            rows_non_null: meta.row_count,
        })
        .collect();

    if !no_values {
        for (i, spec) in schema.columns.iter().enumerate() {
            // Try to read from footer stats first (zero I/O on column data).
            if let Some((mn, mx, null_count)) =
                stats_from_footer(&reader, &spec.name, meta.stripe_count)
            {
                let non_null = meta.row_count.saturating_sub(null_count);
                col_stats[i].min = mn;
                col_stats[i].max = mx;
                col_stats[i].rows_non_null = non_null;
            } else {
                // Fall back to reading the column data.
                let (mn, mx, non_null) = compute_column_stats(
                    &mut reader,
                    &spec.name,
                    &spec.logical_type,
                    meta.row_count,
                );
                col_stats[i].min = mn;
                col_stats[i].max = mx;
                col_stats[i].rows_non_null = non_null;
            }
        }
    }

    // Render output.
    if as_json {
        print_json(&meta, &col_stats)?;
    } else {
        print_markdown(&meta, &col_stats);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// File metadata bundle (reduces argument counts on rendering functions)
// ---------------------------------------------------------------------------

/// File-level metadata gathered at open time, used by rendering functions.
struct FileMeta {
    path: String,
    format_version: String,
    row_count: u64,
    stripe_count: usize,
    /// `(header_bytes, body_bytes, footer_bytes)` from [`HeliumReader::region_sizes`].
    region: (u64, u64, u64),
}

impl FileMeta {
    fn size_bytes(&self) -> u64 {
        self.region.0 + self.region.1 + self.region.2
    }
    fn header_bytes(&self) -> u64 {
        self.region.0
    }
    fn body_bytes(&self) -> u64 {
        self.region.1
    }
    fn footer_bytes(&self) -> u64 {
        self.region.2
    }
}

// ---------------------------------------------------------------------------
// Column statistics computation
// ---------------------------------------------------------------------------

/// A computed or deferred statistic value for a single column endpoint.
#[derive(Debug, Clone)]
enum StatValue {
    /// Value computation was skipped (`--no-values` flag).
    Skipped,
    /// The column is empty, all-null, or a type that has no canonical min/max.
    NotAvailable,
    /// A numeric value (signed integer, unsigned integer, or float), stored as
    /// f64 for uniformity. Integers that round-trip without loss are serialized
    /// as JSON integers by [`stat_to_json`].
    Number(f64),
    /// A string value (Utf8 column).
    Str(String),
    /// A binary value — display as `<N bytes>`.
    Bytes(usize),
}

struct ColumnStats {
    name: String,
    type_str: String,
    bytes: u64,
    min: StatValue,
    max: StatValue,
    rows_total: u64,
    rows_non_null: u64,
}

/// Attempt to read (min, max, total_null_count) for `column_name` from
/// the footer statistics across all stripes.
///
/// Returns `None` if:
/// - The column has no stats in the footer (older file or stats disabled).
/// - The column type produces no scalar leaf stats (e.g. Struct, List, Map).
///
/// When `Some` is returned, the values are the overall min/max aggregated
/// across all stripes, and total_null_count is the sum of per-stripe null
/// counts from the "best" physical leaf.
fn stats_from_footer(
    reader: &HeliumReader<File>,
    column_name: &str,
    stripe_count: usize,
) -> Option<(StatValue, StatValue, u64)> {
    // Collect per-stripe physical stats.
    let mut global_min: Option<MinMaxValue> = None;
    let mut global_max: Option<MinMaxValue> = None;
    let mut total_null: u64 = 0;
    let mut any_stats = false;

    for s_idx in 0..stripe_count {
        let phys = reader.stripe_column_stats(s_idx, column_name)?;
        // Find the first physical leaf that has actual min/max stats.
        // For numeric Primitive columns this is leaf 0 (the only leaf).
        // For Utf8, it's leaf 1 (data). For Nullable, it's leaf 1 (values).
        // We pick the first leaf that has a non-None min.
        let (leaf_min, leaf_max, leaf_null) = phys
            .iter()
            .find(|l| l.min.is_some())
            .map(|l| (l.min.clone(), l.max.clone(), l.null_count))
            .unwrap_or((None, None, phys.first().and_then(|l| l.null_count)));

        if leaf_min.is_some() {
            any_stats = true;
            global_min = merge_min(global_min.take(), leaf_min);
            global_max = merge_max(global_max.take(), leaf_max);
        }
        // Accumulate null count from the leaf that carries it.
        // For Nullable types the present-bitmap leaf (index 0) has null_count.
        if let Some(null_c) = leaf_null {
            total_null += null_c;
        }
    }

    if !any_stats {
        return None;
    }

    let mn = mmv_to_stat(global_min);
    let mx = mmv_to_stat(global_max);
    Some((mn, mx, total_null))
}

/// Take the minimum of two `Option<MinMaxValue>`.
fn merge_min(a: Option<MinMaxValue>, b: Option<MinMaxValue>) -> Option<MinMaxValue> {
    match (a, b) {
        (None, x) | (x, None) => x,
        (Some(x), Some(y)) => Some(mmv_min(x, y)),
    }
}

/// Take the maximum of two `Option<MinMaxValue>`.
fn merge_max(a: Option<MinMaxValue>, b: Option<MinMaxValue>) -> Option<MinMaxValue> {
    match (a, b) {
        (None, x) | (x, None) => x,
        (Some(x), Some(y)) => Some(mmv_max(x, y)),
    }
}

/// Return the minimum of two `MinMaxValue` (same variant).
fn mmv_min(a: MinMaxValue, b: MinMaxValue) -> MinMaxValue {
    use MinMaxValue as M;
    match (a, b) {
        (M::I8(x), M::I8(y)) => M::I8(x.min(y)),
        (M::I16(x), M::I16(y)) => M::I16(x.min(y)),
        (M::I32(x), M::I32(y)) => M::I32(x.min(y)),
        (M::I64(x), M::I64(y)) => M::I64(x.min(y)),
        (M::U8(x), M::U8(y)) => M::U8(x.min(y)),
        (M::U16(x), M::U16(y)) => M::U16(x.min(y)),
        (M::U32(x), M::U32(y)) => M::U32(x.min(y)),
        (M::U64(x), M::U64(y)) => M::U64(x.min(y)),
        (M::F32(x), M::F32(y)) => M::F32(if x <= y { x } else { y }),
        (M::F64(x), M::F64(y)) => M::F64(if x <= y { x } else { y }),
        (M::Utf8(x), M::Utf8(y)) => M::Utf8(if x <= y { x } else { y }),
        (M::Binary(x), M::Binary(y)) => M::Binary(if x <= y { x } else { y }),
        (
            M::Decimal128 {
                high: ah,
                low: al,
                precision: ap,
                scale: as_,
            },
            M::Decimal128 {
                high: bh, low: bl, ..
            },
        ) => {
            let av: i128 = ((ah as i128) << 64) | (al as u64 as i128);
            let bv: i128 = ((bh as i128) << 64) | (bl as u64 as i128);
            if av <= bv {
                M::Decimal128 {
                    high: ah,
                    low: al,
                    precision: ap,
                    scale: as_,
                }
            } else {
                M::Decimal128 {
                    high: bh,
                    low: bl,
                    precision: ap,
                    scale: as_,
                }
            }
        }
        (
            M::Date {
                value: xv,
                unit: xu,
            },
            M::Date { value: yv, .. },
        ) => M::Date {
            value: xv.min(yv),
            unit: xu,
        },
        (
            M::Datetime {
                value: xv,
                unit: xu,
                timezone: xtz,
            },
            M::Datetime { value: yv, .. },
        ) => M::Datetime {
            value: xv.min(yv),
            unit: xu,
            timezone: xtz,
        },
        // Mismatched variants: shouldn't happen for a single column; return first.
        (x, _) => x,
    }
}

/// Return the maximum of two `MinMaxValue` (same variant).
fn mmv_max(a: MinMaxValue, b: MinMaxValue) -> MinMaxValue {
    use MinMaxValue as M;
    match (a, b) {
        (M::I8(x), M::I8(y)) => M::I8(x.max(y)),
        (M::I16(x), M::I16(y)) => M::I16(x.max(y)),
        (M::I32(x), M::I32(y)) => M::I32(x.max(y)),
        (M::I64(x), M::I64(y)) => M::I64(x.max(y)),
        (M::U8(x), M::U8(y)) => M::U8(x.max(y)),
        (M::U16(x), M::U16(y)) => M::U16(x.max(y)),
        (M::U32(x), M::U32(y)) => M::U32(x.max(y)),
        (M::U64(x), M::U64(y)) => M::U64(x.max(y)),
        (M::F32(x), M::F32(y)) => M::F32(if x >= y { x } else { y }),
        (M::F64(x), M::F64(y)) => M::F64(if x >= y { x } else { y }),
        (M::Utf8(x), M::Utf8(y)) => M::Utf8(if x >= y { x } else { y }),
        (M::Binary(x), M::Binary(y)) => M::Binary(if x >= y { x } else { y }),
        (
            M::Decimal128 {
                high: ah,
                low: al,
                precision: ap,
                scale: as_,
            },
            M::Decimal128 {
                high: bh, low: bl, ..
            },
        ) => {
            let av: i128 = ((ah as i128) << 64) | (al as u64 as i128);
            let bv: i128 = ((bh as i128) << 64) | (bl as u64 as i128);
            if av >= bv {
                M::Decimal128 {
                    high: ah,
                    low: al,
                    precision: ap,
                    scale: as_,
                }
            } else {
                M::Decimal128 {
                    high: bh,
                    low: bl,
                    precision: ap,
                    scale: as_,
                }
            }
        }
        (
            M::Date {
                value: xv,
                unit: xu,
            },
            M::Date { value: yv, .. },
        ) => M::Date {
            value: xv.max(yv),
            unit: xu,
        },
        (
            M::Datetime {
                value: xv,
                unit: xu,
                timezone: xtz,
            },
            M::Datetime { value: yv, .. },
        ) => M::Datetime {
            value: xv.max(yv),
            unit: xu,
            timezone: xtz,
        },
        // Mismatched variants: shouldn't happen for a single column; return first.
        (x, _) => x,
    }
}

/// Convert a `MinMaxValue` to a `StatValue` for rendering.
fn mmv_to_stat(v: Option<MinMaxValue>) -> StatValue {
    match v {
        None => StatValue::NotAvailable,
        Some(MinMaxValue::I8(x)) => StatValue::Number(x as f64),
        Some(MinMaxValue::I16(x)) => StatValue::Number(x as f64),
        Some(MinMaxValue::I32(x)) => StatValue::Number(x as f64),
        Some(MinMaxValue::I64(x)) => StatValue::Number(x as f64),
        Some(MinMaxValue::U8(x)) => StatValue::Number(x as f64),
        Some(MinMaxValue::U16(x)) => StatValue::Number(x as f64),
        Some(MinMaxValue::U32(x)) => StatValue::Number(x as f64),
        Some(MinMaxValue::U64(x)) => StatValue::Number(x as f64),
        Some(MinMaxValue::F32(x)) => StatValue::Number(x as f64),
        Some(MinMaxValue::F64(x)) => StatValue::Number(x),
        Some(MinMaxValue::Utf8(s)) => StatValue::Str(s),
        Some(MinMaxValue::Binary(b)) => {
            StatValue::Str(format!("<base64:{}>", &b[..b.len().min(16)]))
        }
        Some(MinMaxValue::Decimal128 {
            high,
            low,
            precision,
            scale,
        }) => {
            let v: i128 = ((high as i128) << 64) | (low as u64 as i128);
            // Format as a decimal string: `v / 10^scale`.
            let scale_factor = 10i128.pow(scale as u32);
            let integer_part = v / scale_factor;
            let frac_part = (v % scale_factor).abs();
            StatValue::Str(format!(
                "{integer_part}.{frac_part:0>width$}",
                width = precision as usize
            ))
        }
        Some(MinMaxValue::Date { value, .. }) => StatValue::Number(value as f64),
        Some(MinMaxValue::Datetime { value, .. }) => StatValue::Number(value as f64),
    }
}

/// Compute `(min, max, rows_non_null)` for a logical column.
///
/// Returns `(NotAvailable, NotAvailable, row_count)` for types that have no
/// canonical min/max (Struct, List, Map, Union, Dict, Array).
///
/// For multi-stripe dict columns `read_column` returns an error — we catch it
/// and return `NotAvailable` rather than failing the whole stats run, per the
/// spec: "multi-stripe dict columns: catch and report '—' instead of failing
/// the whole stats command".
fn compute_column_stats(
    reader: &mut HeliumReader<File>,
    name: &str,
    logical_type: &LogicalType,
    row_count: u64,
) -> (StatValue, StatValue, u64) {
    // Types that have no canonical min/max.
    match logical_type {
        LogicalType::Struct { .. }
        | LogicalType::List { .. }
        | LogicalType::Map { .. }
        | LogicalType::Union { .. }
        | LogicalType::ArrayOf { .. }
        | LogicalType::ArrayOfUtf8
        | LogicalType::Dictionary { .. } => {
            return (StatValue::NotAvailable, StatValue::NotAvailable, row_count);
        }
        _ => {}
    }

    let col = match reader.read_column(name) {
        Ok(c) => c,
        Err(_) => {
            // Multi-stripe dict or other read error — report gracefully.
            return (StatValue::NotAvailable, StatValue::NotAvailable, row_count);
        }
    };

    stats_from_logical_column(&col)
}

/// Extract `(min, max, rows_non_null)` from a decoded `LogicalColumn`.
fn stats_from_logical_column(col: &LogicalColumn) -> (StatValue, StatValue, u64) {
    match col {
        LogicalColumn::Primitive(data) => {
            let (min, max) = min_max_column_data(data);
            (min, max, data.len() as u64)
        }
        LogicalColumn::Utf8(strings) => {
            if strings.is_empty() {
                return (StatValue::NotAvailable, StatValue::NotAvailable, 0);
            }
            let min = strings
                .iter()
                .min()
                .map(|s| StatValue::Str(s.clone()))
                .unwrap_or(StatValue::NotAvailable);
            let max = strings
                .iter()
                .max()
                .map(|s| StatValue::Str(s.clone()))
                .unwrap_or(StatValue::NotAvailable);
            (min, max, strings.len() as u64)
        }
        LogicalColumn::Binary(blobs) => {
            if blobs.is_empty() {
                return (StatValue::NotAvailable, StatValue::NotAvailable, 0);
            }
            let min_len = blobs.iter().map(|b| b.len()).min().unwrap_or(0);
            let max_len = blobs.iter().map(|b| b.len()).max().unwrap_or(0);
            (
                StatValue::Bytes(min_len),
                StatValue::Bytes(max_len),
                blobs.len() as u64,
            )
        }
        LogicalColumn::NullablePrim { present, values } => {
            let non_null_count = present.iter().filter(|&&b| b).count() as u64;
            if non_null_count == 0 {
                return (StatValue::NotAvailable, StatValue::NotAvailable, 0);
            }
            let (min, max) = min_max_column_data(values);
            (min, max, non_null_count)
        }
        LogicalColumn::NullableUtf8 { present, strings } => {
            let non_null_count = present.iter().filter(|&&b| b).count() as u64;
            if non_null_count == 0 || strings.is_empty() {
                return (StatValue::NotAvailable, StatValue::NotAvailable, 0);
            }
            let min = strings
                .iter()
                .min()
                .map(|s| StatValue::Str(s.clone()))
                .unwrap_or(StatValue::NotAvailable);
            let max = strings
                .iter()
                .max()
                .map(|s| StatValue::Str(s.clone()))
                .unwrap_or(StatValue::NotAvailable);
            (min, max, non_null_count)
        }
        LogicalColumn::NullableBinary { present, blobs } => {
            let non_null_count = present.iter().filter(|&&b| b).count() as u64;
            if non_null_count == 0 || blobs.is_empty() {
                return (StatValue::NotAvailable, StatValue::NotAvailable, 0);
            }
            let min_len = blobs.iter().map(|b| b.len()).min().unwrap_or(0);
            let max_len = blobs.iter().map(|b| b.len()).max().unwrap_or(0);
            (
                StatValue::Bytes(min_len),
                StatValue::Bytes(max_len),
                non_null_count,
            )
        }
        // recursive Nullable wrapper (new-style).
        LogicalColumn::Nullable { present, value } => {
            let non_null_count = present.iter().filter(|&&b| b).count() as u64;
            if non_null_count == 0 {
                return (StatValue::NotAvailable, StatValue::NotAvailable, 0);
            }
            let (min, max, _) = stats_from_logical_column(value);
            (min, max, non_null_count)
        }
        // All other types (Struct, List, Map, Union, Dictionary,
        // ArrayOf, ArrayOfUtf8) have no canonical scalar min/max.
        _ => {
            let row_count = col.row_count() as u64;
            (StatValue::NotAvailable, StatValue::NotAvailable, row_count)
        }
    }
}

/// Compute min/max over a `ColumnData` as `StatValue::Number`.
fn min_max_column_data(data: &helium::ColumnData) -> (StatValue, StatValue) {
    use helium::ColumnData;
    macro_rules! int_min_max {
        ($v:expr) => {{
            if $v.is_empty() {
                return (StatValue::NotAvailable, StatValue::NotAvailable);
            }
            // SAFETY: non-empty slice guaranteed by the is_empty guard above.
            let min = *$v.iter().min().unwrap() as f64;
            let max = *$v.iter().max().unwrap() as f64;
            (StatValue::Number(min), StatValue::Number(max))
        }};
    }
    match data {
        ColumnData::I8(v) => int_min_max!(v),
        ColumnData::I16(v) => int_min_max!(v),
        ColumnData::I32(v) => int_min_max!(v),
        ColumnData::I64(v) => int_min_max!(v),
        ColumnData::U8(v) => int_min_max!(v),
        ColumnData::U16(v) => int_min_max!(v),
        ColumnData::U32(v) => int_min_max!(v),
        ColumnData::U64(v) => int_min_max!(v),
        ColumnData::F32(v) => {
            if v.is_empty() {
                return (StatValue::NotAvailable, StatValue::NotAvailable);
            }
            let min = v.iter().copied().fold(f32::INFINITY, f32::min) as f64;
            let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            (StatValue::Number(min), StatValue::Number(max))
        }
        ColumnData::F64(v) => {
            if v.is_empty() {
                return (StatValue::NotAvailable, StatValue::NotAvailable);
            }
            let min = v.iter().copied().fold(f64::INFINITY, f64::min);
            let max = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            (StatValue::Number(min), StatValue::Number(max))
        }
        ColumnData::Bytes(_) => (StatValue::NotAvailable, StatValue::NotAvailable),
    }
}

// ---------------------------------------------------------------------------
// Rendering — empty-file case
// ---------------------------------------------------------------------------

fn print_empty(meta: &FileMeta, as_json: bool) -> anyhow::Result<()> {
    if as_json {
        let out = serde_json::json!({
            "file": meta.path,
            "format": meta.format_version,
            "size_bytes": meta.size_bytes(),
            "schema_header_bytes": meta.header_bytes(),
            "body_bytes": meta.body_bytes(),
            "footer_bytes": meta.footer_bytes(),
            "stripes": meta.stripe_count,
            "rows": meta.row_count,
            "columns": []
        });
        println!("{}", serde_json::to_string_pretty(&out)?);
    } else {
        println!("File:     {}", meta.path);
        println!("Format:   {}", meta.format_version);
        println!("Size:     {}", fmt_bytes(meta.size_bytes()));
        println!("  Schema header: {}", fmt_bytes(meta.header_bytes()));
        println!("  Body:          {}", fmt_bytes(meta.body_bytes()));
        println!("  Footer:        {}", fmt_bytes(meta.footer_bytes()));
        println!("Stripes:  {}", meta.stripe_count);
        println!("Rows:     {}", fmt_number(meta.row_count));
        println!();
        println!("0 rows; no per-column stats.");
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Rendering — markdown
// ---------------------------------------------------------------------------

fn print_markdown(meta: &FileMeta, col_stats: &[ColumnStats]) {
    println!("File:     {}", meta.path);
    println!("Format:   {}", meta.format_version);
    println!("Size:     {} total", fmt_bytes(meta.size_bytes()));
    println!("  Schema header: {}", fmt_bytes(meta.header_bytes()));
    println!("  Body:          {}", fmt_bytes(meta.body_bytes()));
    println!("  Footer:        {}", fmt_bytes(meta.footer_bytes()));
    println!("Stripes:  {}", meta.stripe_count);
    println!("Rows:     {}", fmt_number(meta.row_count));
    println!();
    println!("Per-column statistics:");
    println!();

    let total_body = meta.body_bytes();

    // Compute column widths.
    let name_w = col_stats
        .iter()
        .map(|c| c.name.len())
        .max()
        .unwrap_or(6)
        .max(6);
    let type_w = col_stats
        .iter()
        .map(|c| c.type_str.len())
        .max()
        .unwrap_or(4)
        .max(4);

    let bytes_header = "Bytes";
    let pct_header = "%";
    let min_header = "Min";
    let max_header = "Max";

    let bytes_w = col_stats
        .iter()
        .map(|c| fmt_bytes(c.bytes).len())
        .max()
        .unwrap_or(5)
        .max(bytes_header.len());
    let pct_w = col_stats
        .iter()
        .map(|c| fmt_pct(c.bytes, total_body).len())
        .max()
        .unwrap_or(5)
        .max(pct_header.len());
    let min_w = col_stats
        .iter()
        .map(|c| stat_display(&c.min).len())
        .max()
        .unwrap_or(3)
        .max(min_header.len());
    let max_w = col_stats
        .iter()
        .map(|c| stat_display(&c.max).len())
        .max()
        .unwrap_or(3)
        .max(max_header.len());

    // Header row.
    println!(
        "| {name:<name_w$} | {type_:<type_w$} | {bytes:>bytes_w$} | {pct:>pct_w$} | {min:<min_w$} | {max:<max_w$} |",
        name = "Column",
        type_ = "Type",
        bytes = bytes_header,
        pct = pct_header,
        min = min_header,
        max = max_header,
    );
    // Separator row.
    println!(
        "|{:-<nw$}|{:-<tw$}|{:-<bw$}|{:-<pw$}|{:-<mnw$}|{:-<mxw$}|",
        "",
        "",
        "",
        "",
        "",
        "",
        nw = name_w + 2,
        tw = type_w + 2,
        bw = bytes_w + 2,
        pw = pct_w + 2,
        mnw = min_w + 2,
        mxw = max_w + 2,
    );

    for c in col_stats {
        println!(
            "| {name:<name_w$} | {type_:<type_w$} | {bytes:>bytes_w$} | {pct:>pct_w$} | {min:<min_w$} | {max:<max_w$} |",
            name = c.name,
            type_ = c.type_str,
            bytes = fmt_bytes(c.bytes),
            pct = fmt_pct(c.bytes, total_body),
            min = stat_display(&c.min),
            max = stat_display(&c.max),
        );
    }
}

// ---------------------------------------------------------------------------
// Rendering — JSON
// ---------------------------------------------------------------------------

fn print_json(meta: &FileMeta, col_stats: &[ColumnStats]) -> anyhow::Result<()> {
    let total_body = meta.body_bytes();
    let columns: Vec<serde_json::Value> = col_stats
        .iter()
        .map(|c| {
            let bytes_pct = if total_body > 0 {
                (c.bytes as f64 / total_body as f64) * 100.0
            } else {
                0.0
            };
            serde_json::json!({
                "name": c.name,
                "type": c.type_str,
                "bytes": c.bytes,
                "bytes_pct": (bytes_pct * 10.0).round() / 10.0,
                "min": stat_to_json(&c.min),
                "max": stat_to_json(&c.max),
                "rows_total": c.rows_total,
                "rows_non_null": c.rows_non_null,
            })
        })
        .collect();

    let out = serde_json::json!({
        "file": meta.path,
        "format": meta.format_version,
        "size_bytes": meta.size_bytes(),
        "schema_header_bytes": meta.header_bytes(),
        "body_bytes": meta.body_bytes(),
        "footer_bytes": meta.footer_bytes(),
        "stripes": meta.stripe_count,
        "rows": meta.row_count,
        "columns": columns,
    });
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Format a `StatValue` for markdown display.
fn stat_display(v: &StatValue) -> String {
    match v {
        StatValue::Skipped | StatValue::NotAvailable => "\u{2014}".to_string(), // "—"
        StatValue::Number(n) => fmt_number_f64(*n),
        StatValue::Str(s) => {
            if s.len() > 64 {
                format!("{}...", &s[..64])
            } else {
                s.clone()
            }
        }
        StatValue::Bytes(n) => format!("<{n} bytes>"),
    }
}

/// Convert a `StatValue` to a JSON value.
fn stat_to_json(v: &StatValue) -> serde_json::Value {
    match v {
        StatValue::Skipped | StatValue::NotAvailable => serde_json::Value::Null,
        StatValue::Number(n) => {
            // Represent as integer if it round-trips without loss.
            let as_i64 = *n as i64;
            if as_i64 as f64 == *n {
                serde_json::Value::Number(as_i64.into())
            } else {
                serde_json::json!(*n)
            }
        }
        StatValue::Str(s) => serde_json::Value::String(s.clone()),
        StatValue::Bytes(n) => serde_json::json!({ "bytes": n }),
    }
}

/// Format a byte count as raw bytes with thousands separators
/// (e.g. `97_898_765` → `"97,898,765 bytes"`). Uniform unit makes
/// columns sortable and comparable; auto-scaling KB/MB is friendlier
/// to the eye but loses precision and breaks copy-paste-into-spreadsheet
/// workflows.
pub fn fmt_bytes(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3 + 6);
    let bytes = s.as_bytes();
    for (i, b) in bytes.iter().enumerate() {
        if i > 0 && (bytes.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(*b as char);
    }
    out.push_str(" bytes");
    out
}

/// Format a percentage to one decimal place.
fn fmt_pct(bytes: u64, total: u64) -> String {
    if total == 0 {
        return "  0.0%".to_string();
    }
    format!("{:.1}%", (bytes as f64 / total as f64) * 100.0)
}

/// Format a u64 with thousands separators (e.g., `1000000` → `"1,000,000"`).
fn fmt_number(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

/// Format an f64 as an integer string if lossless, or as a float otherwise.
fn fmt_number_f64(n: f64) -> String {
    let as_i64 = n as i64;
    if as_i64 as f64 == n {
        as_i64.to_string()
    } else {
        format!("{n}")
    }
}

/// Produce a short display string for a `LogicalType`.
fn fmt_logical_type(lt: &LogicalType) -> String {
    match lt {
        LogicalType::Primitive { data_type } => format!("{data_type:?}"),
        LogicalType::Utf8 => "Utf8".to_string(),
        LogicalType::Binary => "Binary".to_string(),
        LogicalType::ArrayOf { data_type } => format!("Array<{data_type:?}>"),
        LogicalType::ArrayOfUtf8 => "Array<Utf8>".to_string(),
        LogicalType::NullablePrim { data_type } => format!("Nullable<{data_type:?}>"),
        LogicalType::NullableUtf8 => "Nullable<Utf8>".to_string(),
        LogicalType::NullableBinary => "Nullable<Binary>".to_string(),
        LogicalType::Dictionary { inner } => format!("Dictionary<{}>", fmt_logical_type(inner)),
        LogicalType::Struct { .. } => "Struct".to_string(),
        LogicalType::List { inner } => format!("List<{}>", fmt_logical_type(inner)),
        LogicalType::Map { key, value } => {
            format!("Map<{},{}>", fmt_logical_type(key), fmt_logical_type(value))
        }
        LogicalType::Nullable { inner } => format!("Nullable<{}>", fmt_logical_type(inner)),
        LogicalType::Union { variants } => {
            let names: Vec<&str> = variants.iter().map(|(n, _)| n.as_str()).collect();
            format!("Union<{}>", names.join(","))
        }
        LogicalType::Decimal128 { precision, scale } => {
            format!("Decimal128({precision},{scale})")
        }
        LogicalType::Date { unit } => format!("Date<{unit:?}>"),
        LogicalType::Datetime { unit, timezone } => {
            if let Some(tz) = timezone {
                format!("Datetime<{unit:?},{tz}>")
            } else {
                format!("Datetime<{unit:?}>")
            }
        }
    }
}
