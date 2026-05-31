//! `helium optimize-schema` — produce a measured-optimal schema via helium-optimizer.

use std::path::Path;

use anyhow::Context;
use helium::optimizer::Optimizer;

/// Run the `optimize-schema` subcommand.
///
/// Picking per-column encodings is a structural decision (delta vs gorilla vs
/// pcodec, …), so it is measured on a representative prefix of the data rather
/// than the whole file. `sample_rows` caps how many rows are loaded for that
/// measurement: a positive value reads only the first `sample_rows` rows (the
/// default), and `0` means "use the whole file". The emitted schema then
/// applies to the full dataset via `helium convert --schema`.
///
/// `delimiter` is the CSV field delimiter byte (ignored for non-CSV inputs).
pub fn run(
    input: &Path,
    out_path: Option<&Path>,
    delimiter: u8,
    sample_rows: usize,
    zstd_level: Option<i32>,
) -> anyhow::Result<()> {
    // The optimizer uses the inferred schema as the structural skeleton and
    // replaces only the encoding pipelines.
    let fmt = super::loader::detect_format(input)?;
    let opts = super::loader::LoadOptions { delimiter };
    let data = if sample_rows == 0 {
        super::loader::load_data_for_fmt(input, fmt, None, &opts)
    } else {
        super::loader::load_data_sample(input, fmt, sample_rows, &opts)
    }
    .with_context(|| format!("loading data from '{}'", input.display()))?;

    if data.is_empty() {
        anyhow::bail!("no columns found in '{}'", input.display());
    }

    let mut optimizer = Optimizer::new();
    if let Some(level) = zstd_level {
        optimizer = optimizer.with_zstd_level(level);
    }
    let schema = optimizer.optimize(data).context("running optimizer")?;

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
