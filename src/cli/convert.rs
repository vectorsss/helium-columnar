//! `helium convert` — bidirectional conversion between `.he` and external formats.
//!
//! # Supported formats
//!
//! | Name | Extensions | Read | Write |
//! |---|---|---|---|
//! | `csv` | `.csv` | Yes | Yes |
//! | `json` | `.json`, `.ndjson` | Yes | Yes |
//! | `parquet` | `.parquet` | Yes | Yes |
//! | `avsc` | `.avsc` | Yes (schema-only) | No |
//! | `he` | `.he` | Yes | Yes |
//!
//! # Direction inference
//!
//! Direction is determined by the `--from` / `--to` flag pair, or inferred from
//! file extensions when the flags are omitted.
//!
//! - `from == he, to != he` → export (`.he` → external format)
//! - `from != he, to == he` → import (external format → `.he`)
//! - Both `he` → error
//! - Neither `he` → error

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use anyhow::{Context, bail};
use helium::catalog::Catalog;
use helium::{CoderRegistry, HeliumReader, HeliumWriter, Schema};

// ---------------------------------------------------------------------------
// Format enum
// ---------------------------------------------------------------------------

/// A recognised format name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fmt {
    Csv,
    Json,
    Parquet,
    /// Avro schema (`.avsc`). Import only; export produces a clear error.
    AvroSchema,
    /// Avro Object Container Format (`.avro`). Both import and export supported.
    Avro,
    He,
}

impl Fmt {
    /// Parse a user-supplied format string (case-insensitive).
    fn from_str(s: &str) -> anyhow::Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "csv" => Ok(Self::Csv),
            "json" | "ndjson" => Ok(Self::Json),
            "parquet" => Ok(Self::Parquet),
            "avsc" => Ok(Self::AvroSchema),
            "avro" => Ok(Self::Avro),
            "he" | "helium" => Ok(Self::He),
            other => bail!(
                "convert: unknown format '{other}'; \
                 supported formats are: csv, json, parquet, avsc, avro, he"
            ),
        }
    }

    /// Infer format from a file extension.
    fn from_path(path: &Path) -> anyhow::Result<Self> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        match ext.as_str() {
            "csv" => Ok(Self::Csv),
            "json" | "ndjson" => Ok(Self::Json),
            "parquet" => Ok(Self::Parquet),
            "avsc" => Ok(Self::AvroSchema),
            "avro" => Ok(Self::Avro),
            "he" => Ok(Self::He),
            other => bail!(
                "convert: cannot infer format from extension '.{other}'; \
                 use --from or --to to specify the format. \
                 Supported formats: csv, json, parquet, avsc, avro, he"
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Options for the `convert` subcommand.
///
/// Bundled to keep [`run`]'s argument count within clippy's limit.
pub struct ConvertOptions<'a> {
    pub input: &'a Path,
    pub schema_path: Option<&'a Path>,
    pub output: &'a Path,
    pub from_flag: Option<&'a str>,
    pub to_flag: Option<&'a str>,
    pub catalog_dir: Option<&'a Path>,
    /// CSV-only: error on List/Map/Union types instead of JSON-stringifying them.
    pub csv_strict: bool,
    /// Split the output into stripes of at most this many rows (0 = single stripe).
    pub stripe_rows: Option<usize>,
    /// CSV field delimiter byte. Default: `b','`. Ignored for non-CSV inputs.
    pub delimiter: u8,
}

/// Run the `convert` subcommand.
///
/// Resolves direction from `--from` / `--to` flags or file extensions, then
/// dispatches to the appropriate import or export path.
///
/// When `opts.catalog_dir` is `Some`:
/// - If converting external → `.he`, uses [`Catalog::open_writer`] to emit a
///   catalog-mode file and register the schema as a side-effect.
/// - If converting `.he` → external, uses [`HeliumReader::new_with_resolver`]
///   so catalog-mode inputs are readable.
/// - If neither side is `.he`, prints a warning and continues without using the
///   catalog.
///
/// `opts.csv_strict` applies only when `to == Csv`. When `true`, List/Map/Union
/// columns produce an error instead of being JSON-stringified. Struct columns
/// always flatten to dotted sub-columns regardless of this flag.
pub fn run(opts: ConvertOptions<'_>) -> anyhow::Result<()> {
    let ConvertOptions {
        input,
        schema_path,
        output,
        from_flag,
        to_flag,
        catalog_dir,
        csv_strict,
        stripe_rows,
        delimiter,
    } = opts;

    // Resolve from / to formats.
    let from = if let Some(f) = from_flag {
        Fmt::from_str(f)?
    } else {
        Fmt::from_path(input)?
    };

    let to = if let Some(t) = to_flag {
        Fmt::from_str(t)?
    } else {
        Fmt::from_path(output)?
    };

    // Validate direction.
    match (from, to) {
        (Fmt::He, Fmt::He) => bail!("convert: both --from and --to resolve to .he; nothing to do"),
        (f, t) if f != Fmt::He && t != Fmt::He => {
            // Neither side is .he — warn about --catalog if provided.
            if catalog_dir.is_some() {
                eprintln!("warning: --catalog has no effect when neither side is .he");
            }
            bail!(
                "convert: neither --from nor --to is .he; specify direction \
                 explicitly with --to he or --from he, or use a different tool for \
                 format-to-format conversion (it would be lossy through Helium's \
                 logical-type model)"
            )
        }
        (Fmt::He, to_fmt) => {
            // Export: .he → external format.
            if schema_path.is_some() {
                eprintln!(
                    "warning: --schema is ignored when exporting from .he \
                     (schema is embedded in the .he file)"
                );
            }
            if stripe_rows.is_some() {
                eprintln!("warning: --stripe-rows has no effect when exporting from .he");
            }
            export_he(input, to_fmt, output, catalog_dir, csv_strict, delimiter)
        }
        (from_fmt, Fmt::He) => {
            // Import: external format → .he.
            import_to_he(
                input,
                from_fmt,
                schema_path,
                output,
                catalog_dir,
                stripe_rows,
                delimiter,
            )
        }
        _ => unreachable!(),
    }
}

// ---------------------------------------------------------------------------
// Import path (external → .he)
// ---------------------------------------------------------------------------

fn import_to_he(
    input: &Path,
    from_fmt: Fmt,
    schema_path: Option<&Path>,
    output: &Path,
    catalog_dir: Option<&Path>,
    stripe_rows: Option<usize>,
    delimiter: u8,
) -> anyhow::Result<()> {
    // Validate that the input format can be imported.
    if from_fmt == Fmt::He {
        bail!("convert: --from he with --to he is not meaningful");
    }

    // Map Fmt → the loader's Format.
    let loader_fmt = match from_fmt {
        Fmt::Csv => super::loader::Format::Csv,
        Fmt::Json => super::loader::Format::Json,
        Fmt::Parquet => super::loader::Format::Parquet,
        Fmt::AvroSchema => super::loader::Format::AvroSchema,
        Fmt::Avro => super::loader::Format::Avro,
        Fmt::He => unreachable!(),
    };

    let load_opts = super::loader::LoadOptions { delimiter };

    // Resolve schema.
    let schema = if let Some(sp) = schema_path {
        load_schema_file(sp)?
    } else {
        super::loader::infer_schema_for_fmt(input, loader_fmt, &load_opts)?
    };

    let registry = CoderRegistry::default();
    let out_file = File::create(output)
        .with_context(|| format!("creating output file '{}'", output.display()))?;

    let mut writer = if let Some(cat_dir) = catalog_dir {
        // Catalog mode: emit a catalog-mode file and register schema as a side-effect.
        let catalog = Catalog::open(cat_dir)
            .with_context(|| format!("opening catalog at '{}'", cat_dir.display()))?;
        catalog
            .open_writer(out_file, schema.clone(), &registry)
            .context("initialising catalog-mode HeliumWriter")?
    } else {
        HeliumWriter::new(out_file, schema.clone(), &registry)
            .context("initialising HeliumWriter")?
    };

    let chunk_rows = stripe_rows.unwrap_or(0);
    if chunk_rows == 0 {
        // Single-stripe path (original behaviour): load all data then write.
        let data = super::loader::load_data_for_fmt(input, loader_fmt, Some(&schema), &load_opts)
            .with_context(|| format!("loading data from '{}'", input.display()))?;
        for (name, _lt, lc) in data {
            writer
                .write_column(&name, lc)
                .with_context(|| format!("writing column '{name}'"))?;
        }
    } else {
        // Streaming path: read chunk_rows rows at a time, write each chunk as
        // a separate stripe.  Peak memory is O(chunk_rows × column_count).
        let mut first_chunk = true;
        super::loader::load_data_chunked(
            input,
            loader_fmt,
            &schema,
            chunk_rows,
            &load_opts,
            &mut |chunk| {
                if !first_chunk {
                    writer.finish_stripe().context("finishing stripe")?;
                }
                first_chunk = false;
                for (name, _lt, lc) in chunk {
                    writer
                        .write_column(&name, lc)
                        .with_context(|| format!("writing column '{name}'"))?;
                }
                Ok(())
            },
        )
        .with_context(|| format!("streaming data from '{}'", input.display()))?;

        // If no chunks were emitted (0-row input), write empty columns so the
        // writer has data for every schema column.
        if first_chunk {
            let empty_data =
                super::loader::load_data_for_fmt(input, loader_fmt, Some(&schema), &load_opts)
                    .with_context(|| format!("loading empty data from '{}'", input.display()))?;
            for (name, _lt, lc) in empty_data {
                writer
                    .write_column(&name, lc)
                    .with_context(|| format!("writing empty column '{name}'"))?;
            }
        }
    }
    writer.finish().context("finalising .he file")?;
    eprintln!("wrote '{}'", output.display());
    Ok(())
}

// ---------------------------------------------------------------------------
// Export path (.he → external)
// ---------------------------------------------------------------------------

fn export_he(
    input: &Path,
    to_fmt: Fmt,
    output: &Path,
    catalog_dir: Option<&Path>,
    csv_strict: bool,
    delimiter: u8,
) -> anyhow::Result<()> {
    // Open the .he file.
    let file =
        File::open(input).with_context(|| format!("opening .he file '{}'", input.display()))?;
    let registry = CoderRegistry::default();
    let mut reader = if let Some(cat_dir) = catalog_dir {
        // Catalog mode: use resolver so catalog-mode inputs are readable.
        let catalog = Catalog::open(cat_dir)
            .with_context(|| format!("opening catalog at '{}'", cat_dir.display()))?;
        HeliumReader::new_with_resolver(file, &registry, catalog.resolver()).with_context(|| {
            format!(
                "reading .he file '{}' with catalog resolver",
                input.display()
            )
        })?
    } else {
        HeliumReader::new(file, &registry)
            .with_context(|| format!("reading .he file '{}'", input.display()))?
    };

    let schema = reader.schema().clone();
    let columns: HashMap<String, helium::LogicalColumn> = reader
        .read_all()
        .with_context(|| format!("reading columns from '{}'", input.display()))?;

    match to_fmt {
        Fmt::Csv => {
            let out_file = File::create(output)
                .with_context(|| format!("creating output file '{}'", output.display()))?;
            let opts = helium::schema::csv::CsvWriteOptions {
                strict: csv_strict,
                delimiter,
            };
            helium::schema::csv::write_csv_with_options(&schema, &columns, out_file, &opts)
                .with_context(|| format!("writing CSV to '{}'", output.display()))?;
        }
        Fmt::Json => {
            let out_file = File::create(output)
                .with_context(|| format!("creating output file '{}'", output.display()))?;
            helium::schema::json::write_json(&schema, &columns, out_file)
                .with_context(|| format!("writing JSON to '{}'", output.display()))?;
        }
        Fmt::Parquet => {
            let out_file = File::create(output)
                .with_context(|| format!("creating output file '{}'", output.display()))?;
            helium::schema::parquet::write_parquet(&schema, &columns, out_file)
                .with_context(|| format!("writing Parquet to '{}'", output.display()))?;
        }
        Fmt::AvroSchema => {
            bail!(
                "convert: --to avsc means schema only; use --to avro to export data \
                 (--to avsc writes only a schema JSON, not an Avro data container file)"
            );
        }
        Fmt::Avro => {
            helium::schema::write_avro_data(output, &schema, &columns)
                .with_context(|| format!("writing Avro to '{}'", output.display()))?;
        }
        Fmt::He => unreachable!(),
    }

    eprintln!("wrote '{}'", output.display());
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Load a schema from a JSON file on disk.
fn load_schema_file(path: &Path) -> anyhow::Result<Schema> {
    let bytes =
        std::fs::read(path).with_context(|| format!("reading schema file '{}'", path.display()))?;
    Schema::from_json(&bytes)
        .with_context(|| format!("parsing schema JSON from '{}'", path.display()))
        .map_err(|e| anyhow::anyhow!("{e}"))
}
