//! `helium infer-schema` — emit a Tier-1 default schema for an input file.

use std::path::Path;

use anyhow::Context;

/// Run the `infer-schema` subcommand.
///
/// Calls the appropriate `helium-schema` adapter, serialises the resulting
/// schema to JSON, and writes it to `out_path` (or stdout if `None`).
///
/// `delimiter` is the CSV field delimiter byte (ignored for non-CSV inputs).
pub fn run(input: &Path, out_path: Option<&Path>, delimiter: u8) -> anyhow::Result<()> {
    let fmt = super::loader::detect_format(input)?;
    let opts = super::loader::LoadOptions { delimiter };
    let schema = super::loader::infer_schema_for_fmt(input, fmt, &opts)
        .with_context(|| format!("inferring schema from '{}'", input.display()))?;

    // Serialise to pretty JSON.
    let json = serde_json::to_string_pretty(&schema).context("serialising schema to JSON")?;

    if let Some(out) = out_path {
        std::fs::write(out, json.as_bytes())
            .with_context(|| format!("writing schema to '{}'", out.display()))?;
        eprintln!("schema written to '{}'", out.display());
    } else {
        println!("{json}");
    }
    Ok(())
}
