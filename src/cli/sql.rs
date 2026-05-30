//! `helium sql` — run SQL over `.he` files via DataFusion.
//!
//! Each positional file argument is registered as a DataFusion table. By
//! default the table name is derived from the filename stem (e.g. `hits_1.he`
//! → `hits_1`). Override with `name=path.he` syntax.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, bail};
use datafusion::prelude::{SessionConfig, SessionContext};
use helium::sql::HeliumTableProvider;

/// A parsed `name=path.he` or bare `path.he` argument.
struct TableArg {
    name: String,
    path: PathBuf,
}

/// Parse a single file argument.
///
/// Accepts either `name=path.he` (explicit table name) or a bare path whose
/// table name is derived from the file stem.
fn parse_table_arg(raw: &str) -> anyhow::Result<TableArg> {
    if let Some((name, path)) = raw.split_once('=') {
        let name = name.trim();
        if name.is_empty() {
            bail!("table arg '{raw}': empty name before '='");
        }
        return Ok(TableArg {
            name: name.to_owned(),
            path: PathBuf::from(path.trim()),
        });
    }

    let path = PathBuf::from(raw);
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| anyhow::anyhow!("table arg '{raw}': cannot derive table name from path"))?;
    if stem.is_empty() {
        bail!("table arg '{raw}': filename stem is empty");
    }
    Ok(TableArg {
        name: stem.to_owned(),
        path,
    })
}

/// Run a SQL query against one or more `.he` files.
///
/// Parses the file list, registers each file as a DataFusion table, executes
/// the query, and pretty-prints the results to stdout.
pub fn run(query: &str, files: &[String]) -> anyhow::Result<()> {
    // 1. Parse all file args.
    let table_args: Vec<TableArg> = files
        .iter()
        .map(|s| parse_table_arg(s))
        .collect::<anyhow::Result<_>>()?;

    // Reject duplicate names.
    let mut seen: HashSet<&str> = HashSet::new();
    for ta in &table_args {
        if !seen.insert(ta.name.as_str()) {
            bail!(
                "duplicate table name '{}'; use name=path syntax to disambiguate",
                ta.name
            );
        }
    }

    // 2. Build SessionContext + register each file.
    //
    // Disable identifier normalization so `SELECT EventTime FROM ...` finds
    // the column literally named `EventTime` instead of being lowercased to
    // `eventtime` per SQL-standard default. Real-world `.he` files (Parquet
    // imports, ClickBench, Avro) routinely carry PascalCase / camelCase
    // column names, and the lowercased form would never match. The user can
    // still force lowercase by writing `"event_time"` explicitly.
    let config =
        SessionConfig::new().set_str("datafusion.sql_parser.enable_ident_normalization", "false");
    let ctx = SessionContext::new_with_config(config);
    for ta in &table_args {
        let provider = HeliumTableProvider::try_new(&ta.path)
            .with_context(|| format!("opening '{}' as Helium table", ta.path.display()))?;
        ctx.register_table(ta.name.as_str(), Arc::new(provider))
            .with_context(|| format!("registering table '{}'", ta.name))?;
    }

    // 3. Run the query via a multi-thread Tokio runtime.
    //    A multi-thread runtime is required because `HeliumExec::execute` uses
    //    `tokio::task::block_in_place` — that call panics on a current-thread
    //    (single-threaded) runtime.
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("building Tokio runtime for SQL execution")?;

    let batches = runtime
        .block_on(async {
            let df = ctx.sql(query).await?;
            df.collect().await
        })
        .context("executing SQL query")?;

    // 4. Pretty-print results to stdout.
    let formatted = datafusion::arrow::util::pretty::pretty_format_batches(&batches)
        .context("formatting result batches")?;
    println!("{formatted}");

    Ok(())
}
