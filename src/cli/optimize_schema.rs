//! `helium optimize-schema` — produce a measured-optimal schema via helium-optimizer.

use std::path::Path;

use anyhow::Context;
use helium::optimizer::Optimizer;

/// Run the `optimize-schema` subcommand.
///
/// Loads a sample of the input data, runs [`Optimizer`] to select the best
/// encoding pipeline per leaf column, and writes the resulting schema as JSON
/// to `out_path` (or stdout if `None`).
///
/// `delimiter` is the CSV field delimiter byte (ignored for non-CSV inputs).
pub fn run(input: &Path, out_path: Option<&Path>, delimiter: u8) -> anyhow::Result<()> {
    // Load data (infer schema from file if not provided — the optimizer
    // uses the inferred schema as the structural skeleton and replaces the
    // encoding pipelines).
    let fmt = super::loader::detect_format(input)?;
    let opts = super::loader::LoadOptions { delimiter };
    let data = super::loader::load_data_for_fmt(input, fmt, None, &opts)
        .with_context(|| format!("loading data from '{}'", input.display()))?;

    if data.is_empty() {
        anyhow::bail!("no columns found in '{}'", input.display());
    }

    let schema = Optimizer::new()
        .optimize(data)
        .context("running optimizer")?;

    let json = serde_json::to_string_pretty(&schema).context("serialising schema to JSON")?;

    if let Some(out) = out_path {
        std::fs::write(out, json.as_bytes())
            .with_context(|| format!("writing schema to '{}'", out.display()))?;
        eprintln!("optimised schema written to '{}'", out.display());
    } else {
        println!("{json}");
    }
    Ok(())
}
