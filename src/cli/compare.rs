//! `helium compare` — compression-rate table for different codec terminals.

use std::path::Path;

use anyhow::Context;
use helium::CoderRegistry;
use helium::optimizer::{Optimizer, measure_encoding};

/// Run the `compare` subcommand.
///
/// Loads the input data, runs [`Optimizer`] with each requested terminal codec,
/// measures the total encoded byte count per codec, and prints a Markdown table.
///
/// `delimiter` is the CSV field delimiter byte (ignored for non-CSV inputs).
pub fn run(input: &Path, codecs: &[String], delimiter: u8) -> anyhow::Result<()> {
    if codecs.is_empty() {
        anyhow::bail!("no codecs specified (use --codecs zstd,lz4,snappy)");
    }

    // Load data (infer schema from file automatically).
    let fmt = super::loader::detect_format(input)?;
    let opts = super::loader::LoadOptions { delimiter };
    let data = super::loader::load_data_for_fmt(input, fmt, None, &opts)
        .with_context(|| format!("loading data from '{}'", input.display()))?;

    if data.is_empty() {
        anyhow::bail!("no columns found in '{}'", input.display());
    }

    // Compute raw (uncompressed) size as baseline.
    let raw_size: usize = data
        .iter()
        .map(|(_, _, lc)| super::loader::raw_bytes(lc))
        .sum();

    let registry = CoderRegistry::default();

    // Measure each codec.
    struct Row {
        codec: String,
        encoded: usize,
    }
    let mut rows: Vec<Row> = Vec::new();

    for codec in codecs {
        let optimizer = Optimizer::with_terminal(codec.as_str());

        // Clone data for this codec run (Optimizer::optimize consumes columns).
        let data_clone: Vec<(String, _, _)> = data
            .iter()
            .map(|(n, lt, lc)| (n.clone(), lt.clone(), lc.clone()))
            .collect();

        let schema = optimizer
            .optimize(data_clone)
            .with_context(|| format!("optimizing with codec '{codec}'"))?;

        let mut total_encoded = 0usize;
        for (i, (_, _, lc)) in data.iter().enumerate() {
            let bytes =
                measure_encoding(&schema.columns[i], lc.clone(), &registry).with_context(|| {
                    format!(
                        "measuring column '{}' with codec '{codec}'",
                        schema.columns[i].name
                    )
                })?;
            total_encoded += bytes;
        }

        rows.push(Row {
            codec: codec.clone(),
            encoded: total_encoded,
        });
    }

    // Print Markdown table.
    println!("| Codec  | Encoded bytes | Ratio vs raw |");
    println!("|--------|--------------|--------------|");
    for row in &rows {
        let ratio = if row.encoded == 0 {
            f64::INFINITY
        } else {
            raw_size as f64 / row.encoded as f64
        };
        println!(
            "| {:<6} | {:>12} | {:>10.2}× |",
            row.codec, row.encoded, ratio
        );
    }
    println!();
    println!("Raw size: {} bytes", raw_size);
    Ok(())
}
