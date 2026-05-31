//! `helium` — columnar compression toolkit binary.
//!
//! # Subcommands
//!
//! | Subcommand | Description |
//! |---|---|
//! | `convert` | Convert CSV / JSON / Parquet / Avro ↔ `.he` |
//! | `infer-schema` | Emit a Tier-1 default schema for the given input file |
//! | `optimize-schema` | Emit a measured-optimal schema (runs optimizer) |
//! | `compare` | Compression-rate table across codec terminals |
//! | `verify` | Read all columns from a `.he` file and verify integrity |
//! | `catalog list` | List all BLAKE3 hashes registered in a catalog directory |
//! | `catalog verify` | Verify catalog directory consistency |
//! | `sql` | Run a SQL query against one or more `.he` files (requires `datafusion` feature) |

// Safety-first: deny panicking unwrap/expect/panic in binary code too (the
// library is covered by the same deny in lib.rs). See lib.rs for rationale.
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::PathBuf;

use clap::{Parser, Subcommand};

mod cli;

// ---------------------------------------------------------------------------
// Delimiter helper
// ---------------------------------------------------------------------------

/// Parse a `--delimiter` string into a single ASCII byte.
///
/// Accepts any single ASCII character (e.g. `","`, `";"`, `"|"`).
/// Returns an error for empty strings, multi-character strings, or
/// non-ASCII characters.
fn parse_delimiter(s: &str) -> anyhow::Result<u8> {
    match s.as_bytes() {
        [b] if b.is_ascii() => Ok(*b),
        _ => anyhow::bail!("--delimiter must be a single ASCII character, got {s:?}"),
    }
}

// ---------------------------------------------------------------------------
// CLI definitions
// ---------------------------------------------------------------------------

/// Helium columnar-compression toolkit.
#[derive(Parser)]
#[command(
    name    = "helium",
    about   = "Helium columnar-compression toolkit",
    version,
    propagate_version = true,
    // No-args invocation prints help instead of clap's terse error.
    // Inherited by every subcommand below: `helium`, `helium convert`,
    // `helium catalog`, etc. all show their own help when called bare.
    arg_required_else_help = true,
    subcommand_required = true,
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Convert between `.he` and external formats (CSV / JSON / Parquet / Avro).
    ///
    /// Direction is inferred from the file extensions of `input` and `-o`.
    /// Use `--from` and `--to` to override extension-based detection.
    /// `--catalog DIR` opts into catalog-mode output, or supplies the
    /// resolver when reading a catalog-mode input. `--delimiter` sets the CSV
    /// field separator (e.g. `;` for European CSV).
    ///
    /// Supported format names: `csv`, `json`, `parquet`, `avro`, `avsc`, `he`.
    /// `.avro` (Object Container) is read/write data; `.avsc` is an Avro
    /// schema only — it provides a schema for an external→`.he` import but
    /// carries no rows, so `--to avsc` is rejected (use `--to avro`).
    ///
    /// Examples:
    ///   helium convert data.csv -o data.he
    ///   helium convert data.he  -o out.csv
    ///   helium convert data.he  -o out.avro
    ///   helium convert data.txt -o data.he --from csv --to he
    ///   helium convert euro.csv -o euro.he --delimiter ';'
    ///   helium convert data.csv -o data.he --catalog /path/to/catalog
    #[command(arg_required_else_help = true)]
    Convert {
        /// Input file
        input: PathBuf,
        /// JSON schema file to use for encoding (inferred if omitted; only applies when converting TO .he)
        #[arg(long, value_name = "SCHEMA.json")]
        schema: Option<PathBuf>,
        /// Output file path
        #[arg(short = 'o', long, value_name = "OUTPUT")]
        output: PathBuf,
        /// Override the source format (csv, json, parquet, avsc, he)
        #[arg(long, value_name = "FMT")]
        from: Option<String>,
        /// Override the destination format (csv, json, parquet, avsc, he)
        #[arg(long, value_name = "FMT")]
        to: Option<String>,
        /// Catalog directory for catalog-mode output or resolving a catalog-mode input
        #[arg(long, value_name = "DIR")]
        catalog: Option<PathBuf>,
        /// CSV-only: error on List/Map/Union types instead of JSON-stringifying them.
        /// Struct types always flatten to dotted columns regardless of this flag.
        #[arg(long)]
        csv_strict: bool,
        /// Split the output into stripes of at most N rows each.
        /// Defaults to 0 (single stripe). Has no effect when exporting from .he.
        #[arg(long, value_name = "N")]
        stripe_rows: Option<usize>,
        /// CSV field delimiter (single ASCII character). Default: ','.
        /// Use ';' for European-style CSV. Ignored for non-CSV inputs.
        #[arg(long, value_name = "CHAR", default_value = ",")]
        delimiter: String,
    },

    /// Infer a Tier-1 default schema from an input file and emit it as JSON.
    ///
    /// Writes to `--out` file if specified, otherwise to stdout.
    #[command(arg_required_else_help = true)]
    InferSchema {
        /// Input file (.csv / .json / .parquet / .avsc)
        input: PathBuf,
        /// Write schema JSON to this file (stdout if omitted)
        #[arg(long, value_name = "SCHEMA.json")]
        out: Option<PathBuf>,
        /// CSV field delimiter (single ASCII character). Default: ','.
        /// Use ';' for European-style CSV. Ignored for non-CSV inputs.
        #[arg(long, value_name = "CHAR", default_value = ",")]
        delimiter: String,
    },

    /// Produce a measured-optimal schema by running the optimizer.
    ///
    /// Picks per-column encodings by measuring them on a representative prefix
    /// of the data (the default `--sample-rows 200000`), then emits the schema
    /// as JSON. Apply it to the full dataset with `convert --schema`. Use
    /// `--sample-rows 0` to measure on the whole file (slow on large inputs:
    /// the picker is O(rows × candidate encodings)).
    /// Writes to `--out` file if specified, otherwise to stdout.
    #[command(arg_required_else_help = true)]
    OptimizeSchema {
        /// Input file (.csv / .json / .parquet / .avsc)
        input: PathBuf,
        /// Write schema JSON to this file (stdout if omitted)
        #[arg(long, value_name = "SCHEMA.json")]
        out: Option<PathBuf>,
        /// CSV field delimiter (single ASCII character). Default: ','.
        /// Use ';' for European-style CSV. Ignored for non-CSV inputs.
        #[arg(long, value_name = "CHAR", default_value = ",")]
        delimiter: String,
        /// Rows to sample when measuring encodings. 0 = whole file.
        #[arg(long, value_name = "N", default_value_t = 200_000)]
        sample_rows: usize,
        /// Global zstd compression level for the emitted schema (1–22).
        /// Omitted = the zstd default (3). Applies to every zstd stage; the
        /// optimizer does not pick the level per column.
        #[arg(long, value_name = "LEVEL")]
        zstd_level: Option<i32>,
    },

    /// Compare compression ratios for different codec terminals on input data.
    ///
    /// Emits a Markdown table with one row per codec.
    #[command(arg_required_else_help = true)]
    Compare {
        /// Input file (.csv / .json / .parquet)
        input: PathBuf,
        /// Comma-separated codec names (default: zstd,lz4,snappy)
        #[arg(long, default_value = "zstd,lz4,snappy", value_name = "CODEC,...")]
        codecs: String,
        /// CSV field delimiter (single ASCII character). Default: ','.
        /// Use ';' for European-style CSV. Ignored for non-CSV inputs.
        #[arg(long, value_name = "CHAR", default_value = ",")]
        delimiter: String,
    },

    /// Slice (project) a subset of columns from a `.he` file into a new `.he` file.
    ///
    /// The output is a fresh self-contained file containing only the
    /// listed columns, in the given order, preserving each column's encodings
    /// and the source's stripe boundaries. Columns are decoded and re-encoded
    /// (not a raw byte copy).
    ///
    /// Examples:
    ///   helium slice events.he -o subset.he --columns ts,user_id,label
    ///   helium slice events.he -o subset.he --columns ts --catalog ./catalog
    #[command(arg_required_else_help = true)]
    Slice {
        /// Input `.he` file.
        input: PathBuf,
        /// Output `.he` file path.
        #[arg(short = 'o', long, value_name = "OUTPUT")]
        output: PathBuf,
        /// Comma-separated column names to keep, in output order.
        #[arg(long, value_name = "COL,...", value_delimiter = ',', required = true)]
        columns: Vec<String>,
        /// Catalog directory for resolving a catalog-mode input.
        #[arg(long, value_name = "DIR")]
        catalog: Option<PathBuf>,
    },

    /// Verify a `.he` file: read all columns and check CRC integrity.
    #[command(arg_required_else_help = true)]
    Verify {
        /// `.he` file to verify
        file: PathBuf,
        /// Catalog directory for resolving catalog-mode files
        #[arg(long, value_name = "DIR")]
        catalog: Option<PathBuf>,
    },

    /// File-size and value statistics for a `.he` file.
    ///
    /// By default reads each column to compute min/max — pass --no-values
    /// to skip the data scan and only show the size breakdown.
    #[command(arg_required_else_help = true)]
    Stats {
        /// Path to the `.he` file.
        file: PathBuf,
        /// Skip min/max computation (only show byte sizes).
        #[arg(long)]
        no_values: bool,
        /// Emit machine-readable JSON instead of a markdown table.
        #[arg(long)]
        json: bool,
        /// Catalog directory for catalog-mode files.
        #[arg(long, value_name = "DIR")]
        catalog: Option<PathBuf>,
    },

    /// Catalog (shared-schema) administration.
    ///
    /// Manages the directory-backed schema catalog used by catalog-mode
    /// `.he` files.
    #[command(arg_required_else_help = true, subcommand_required = true)]
    Catalog {
        #[command(subcommand)]
        op: CatalogOp,
    },

    /// Run a SQL query against one or more `.he` files via DataFusion.
    ///
    /// Each input file is registered as a DataFusion table. By default the
    /// table name is the filename without the `.he` extension. To override,
    /// use `name=path.he` syntax.
    ///
    /// Examples:
    ///   helium sql "SELECT count(*) FROM hits_1" hits_1.he
    ///   helium sql "SELECT a.id, b.label FROM a JOIN b ON a.id = b.id" a.he b.he
    ///   helium sql "SELECT * FROM events LIMIT 5" events=2026-01.he
    ///
    /// Requires building with --features datafusion.
    #[cfg(feature = "datafusion")]
    #[command(arg_required_else_help = true)]
    Sql {
        /// SQL query to execute.
        query: String,
        /// One or more `.he` files. Each becomes a registered table.
        /// Use `name=path.he` to override the default table name.
        #[arg(required = true)]
        files: Vec<String>,
    },
}

/// Operations on a schema catalog directory.
#[derive(Subcommand)]
enum CatalogOp {
    /// List the BLAKE3 hashes of all schemas registered in <DIR>.
    ///
    /// Prints one 64-char lowercase hex hash per line, sorted lexicographically.
    #[command(arg_required_else_help = true)]
    List {
        /// Catalog directory
        dir: PathBuf,
    },
    /// Verify that every <hash>.json in <DIR> hashes to its filename.
    ///
    /// Exits 0 with `OK: {n} schema(s) registered — {dir}` on success.
    /// Exits non-zero with an error message on any inconsistency.
    #[command(arg_required_else_help = true)]
    Verify {
        /// Catalog directory
        dir: PathBuf,
    },
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();
    let result = match cli.command {
        Commands::Convert {
            input,
            schema,
            output,
            from,
            to,
            catalog,
            csv_strict,
            stripe_rows,
            delimiter,
        } => parse_delimiter(&delimiter).and_then(|delim| {
            cli::convert::run(cli::convert::ConvertOptions {
                input: &input,
                schema_path: schema.as_deref(),
                output: &output,
                from_flag: from.as_deref(),
                to_flag: to.as_deref(),
                catalog_dir: catalog.as_deref(),
                csv_strict,
                stripe_rows,
                delimiter: delim,
            })
        }),
        Commands::InferSchema {
            input,
            out,
            delimiter,
        } => parse_delimiter(&delimiter)
            .and_then(|delim| cli::infer_schema::run(&input, out.as_deref(), delim)),
        Commands::OptimizeSchema {
            input,
            out,
            delimiter,
            sample_rows,
            zstd_level,
        } => parse_delimiter(&delimiter).and_then(|delim| {
            cli::optimize_schema::run(&input, out.as_deref(), delim, sample_rows, zstd_level)
        }),
        Commands::Compare {
            input,
            codecs,
            delimiter,
        } => {
            let codec_list: Vec<String> = codecs.split(',').map(|s| s.trim().to_string()).collect();
            parse_delimiter(&delimiter)
                .and_then(|delim| cli::compare::run(&input, &codec_list, delim))
        }
        Commands::Slice {
            input,
            output,
            columns,
            catalog,
        } => cli::slice::run(&input, &output, &columns, catalog.as_deref()),
        Commands::Verify { file, catalog } => cli::verify::run(&file, catalog.as_deref()),
        Commands::Stats {
            file,
            no_values,
            json,
            catalog,
        } => cli::stats::run(&file, no_values, json, catalog.as_deref()),
        Commands::Catalog { op } => match op {
            CatalogOp::List { dir } => cli::catalog::run_list(&dir),
            CatalogOp::Verify { dir } => cli::catalog::run_verify(&dir),
        },
        #[cfg(feature = "datafusion")]
        Commands::Sql { query, files } => cli::sql::run(&query, &files),
    };

    if let Err(e) = result {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}
