//! `helium slice` — project a subset of columns from a `.he` file into a new
//! `.he` file (a column "slice").

use std::fs::File;
use std::path::Path;

use anyhow::Context;
use helium::catalog::Catalog;
use helium::{CoderRegistry, HeliumReader};

/// Run the `slice` subcommand.
///
/// Opens `input`, projects exactly `columns` (in the given order, preserving
/// each column's encodings and the source's stripe boundaries) and writes them
/// to `output` as a fresh self-contained `.he` file.
///
/// `catalog_dir` is only needed to *read* a catalog-mode input; the
/// output is always self-contained.
pub fn run(
    input: &Path,
    output: &Path,
    columns: &[String],
    catalog_dir: Option<&Path>,
) -> anyhow::Result<()> {
    if columns.is_empty() {
        anyhow::bail!("--columns must list at least one column");
    }
    let registry = CoderRegistry::default();

    let in_file = File::open(input).with_context(|| format!("opening '{}'", input.display()))?;
    let mut reader = if let Some(cat_dir) = catalog_dir {
        let catalog = Catalog::open(cat_dir)
            .with_context(|| format!("opening catalog at '{}'", cat_dir.display()))?;
        HeliumReader::new_with_resolver(in_file, &registry, catalog.resolver())
            .with_context(|| format!("reading '{}' with catalog resolver", input.display()))?
    } else {
        HeliumReader::new(in_file, &registry)
            .with_context(|| format!("reading '{}'", input.display()))?
    };

    let out_file =
        File::create(output).with_context(|| format!("creating '{}'", output.display()))?;
    let col_refs: Vec<&str> = columns.iter().map(String::as_str).collect();
    reader
        .project_to(&col_refs, out_file, &registry)
        .with_context(|| format!("projecting {col_refs:?} into '{}'", output.display()))?;

    println!(
        "OK: sliced {} column(s) → '{}'",
        columns.len(),
        output.display()
    );
    Ok(())
}
