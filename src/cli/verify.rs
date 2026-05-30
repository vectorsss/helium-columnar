//! `helium verify` — read all columns from a `.he` file and verify integrity.

use std::fs::File;
use std::path::Path;

use anyhow::Context;
use helium::catalog::Catalog;
use helium::{CoderRegistry, HeliumReader};

/// Run the `verify` subcommand.
///
/// Opens the `.he` file, reads every column in every stripe, and verifies
/// CRC32C integrity.  The CRC check is built into [`HeliumReader`], so any
/// corruption surfaces as a [`helium::HeliumError::Corrupted`] error.
///
/// When `catalog_dir` is `Some`, uses [`HeliumReader::new_with_resolver`] so
/// v4/v6 catalog-mode files are readable.
///
/// On success, prints a summary:
/// ```text
/// OK: 42 columns × 100000 rows (3 stripes)
/// ```
pub fn run(path: &Path, catalog_dir: Option<&Path>) -> anyhow::Result<()> {
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

    let schema = reader.schema().clone();
    let row_count = reader.row_count();
    let stripe_count = reader.stripe_count();

    let mut ok = 0usize;
    let mut errors = 0usize;

    for spec in &schema.columns {
        // For dict columns in multi-stripe files, read stripe-by-stripe to
        // avoid the cross-stripe dict constraint.
        let is_dict = matches!(spec.logical_type, helium::LogicalType::Dictionary { .. });

        if is_dict && stripe_count > 1 {
            for si in 0..stripe_count {
                match reader.read_column_at_stripe(&spec.name, si) {
                    Ok(_) => ok += 1,
                    Err(e) => {
                        eprintln!("  FAIL stripe {si} column '{}': {e}", spec.name);
                        errors += 1;
                    }
                }
            }
        } else {
            match reader.read_column(&spec.name) {
                Ok(_) => ok += 1,
                Err(e) => {
                    eprintln!("  FAIL column '{}': {e}", spec.name);
                    errors += 1;
                }
            }
        }
    }

    if errors > 0 {
        anyhow::bail!(
            "{} column(s) failed verification in '{}'",
            errors,
            path.display()
        );
    }

    println!(
        "OK: {} column(s) × {row_count} rows ({stripe_count} stripe(s)) — '{}'",
        ok,
        path.display()
    );
    Ok(())
}
