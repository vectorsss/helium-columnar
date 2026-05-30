//! `helium catalog list` / `helium catalog verify` subcommands.
//!
//! Light wrappers around [`helium::catalog::Catalog`].

use std::path::Path;

use anyhow::Context;
use helium::catalog::Catalog;

/// Run `helium catalog list <DIR>`.
///
/// Prints one BLAKE3 hash per line (64 lowercase hex chars), sorted
/// lexicographically for deterministic output.
pub fn run_list(dir: &Path) -> anyhow::Result<()> {
    let catalog =
        Catalog::open(dir).with_context(|| format!("opening catalog at '{}'", dir.display()))?;

    let mut hashes = catalog
        .list_schemas()
        .with_context(|| format!("listing schemas in '{}'", dir.display()))?;

    // Sort lexicographically for deterministic output; library returns
    // filesystem-order which is not stable across platforms.
    hashes.sort_by_key(|h| h.to_hex().to_string());

    for hash in &hashes {
        println!("{}", hash.to_hex());
    }
    Ok(())
}

/// Run `helium catalog verify <DIR>`.
///
/// On success prints `OK: {n} schema(s) registered — {dir}` and exits 0.
/// On inconsistency propagates the error (non-zero exit via main's error handler).
pub fn run_verify(dir: &Path) -> anyhow::Result<()> {
    let catalog =
        Catalog::open(dir).with_context(|| format!("opening catalog at '{}'", dir.display()))?;

    catalog
        .verify_consistency()
        .with_context(|| format!("verifying catalog at '{}'", dir.display()))?;

    let n = catalog
        .list_schemas()
        .with_context(|| format!("listing schemas in '{}'", dir.display()))?
        .len();

    println!("OK: {n} schema(s) registered — {}", dir.display());
    Ok(())
}
