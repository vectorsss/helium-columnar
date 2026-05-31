//! Opt-in shared-schema catalog for Helium `.he` files.
//!
//! A directory of schema-hash-keyed JSON files plus a
//! filesystem-backed resolver suitable for [`crate::HeliumReader::new_with_resolver`].
//!
//! # Why catalog mode
//!
//! For workloads that produce many `.he` files all sharing the same logical
//! schema (canonical example: the Avro-replacement 5G MR pipeline writing one
//! file per partition × per day), embedding the schema JSON in every file's
//! header is wasted bytes. The self-contained format already compresses the
//! schema, but a 32-byte hash reference is still ~10–100× smaller than even a
//! compressed schema for many real shapes.
//!
//! Catalog mode is **opt-in**: the default [`HeliumWriter::new`] continues to
//! embed the (compressed) schema in every file. Catalog mode is for
//! deployments that have a stable schema across many files and own a writable
//! catalog directory.
//!
//! # Architecture
//!
//! - **Catalog format** — a flat directory of `<blake3-hex>.json` files, one
//!   per registered schema. Content-addressed; same schema → same hash → same
//!   bytes; idempotent writes; no lock file (POSIX `tmpfile + rename` is
//!   crash-safe per file).
//! - **Canonicalization** — [`canonicalize_json`] produces the bit-stable JSON
//!   bytes that BLAKE3 hashes over (lex-sorted keys, no whitespace,
//!   NFC-normalized keys, exact-decimal integers, shortest-float).
//!   See `canonicalize.rs` for the full contract.
//! - **Writer split** — [`crate::HeliumWriter::with_catalog_ref`] takes a
//!   precomputed hash. [`crate::catalog::Catalog::add_schema`] computes the hash and persists
//!   the schema; [`crate::catalog::Catalog::open_writer`] is the convenience wrapper that does
//!   both in one call.
//! - **Reader split** — [`crate::HeliumReader::new_with_resolver`] takes a closure
//!   `Fn(&blake3::Hash) -> Result<Schema>`. [`crate::catalog::Catalog::resolver`] provides
//!   the filesystem-backed default.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::{HeliumError, HeliumWriter, Result, Schema};

// The canonicalizer and schema_hash live in the core module so
// `HeliumWriter::with_catalog_ref` can assert hash correctness without
// pulling extra dependencies into the format-only surface. They are
// re-exported here so users only need the catalog module for catalog admin work.
pub use crate::{canonicalize_json, schema_hash};

// ---------------------------------------------------------------------------
// Catalog (filesystem-backed)
// ---------------------------------------------------------------------------

/// A filesystem-backed schema catalog: a directory containing one
/// `<blake3-hex>.json` file per registered schema.
///
/// The catalog is a write/admin path; the read hot path can use direct
/// filesystem lookups via [`Catalog::resolver`] without dragging the rest of
/// helium-catalog through the dependency graph.
#[derive(Debug, Clone)]
pub struct Catalog {
    dir: PathBuf,
}

impl Catalog {
    /// Open a catalog at `dir`. Creates the directory if it doesn't exist.
    /// The directory is treated as the catalog's root — every file directly
    /// inside it whose name matches `<64-hex-chars>.json` is a schema file.
    pub fn open<P: AsRef<Path>>(dir: P) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        if !dir.exists() {
            fs::create_dir_all(&dir).map_err(HeliumError::from)?;
        }
        if !dir.is_dir() {
            return Err(HeliumError::Format(format!(
                "catalog path is not a directory: {}",
                dir.display()
            )));
        }
        Ok(Self { dir })
    }

    /// Path of the schema file for a given hash within this catalog.
    pub fn path_for(&self, hash: &blake3::Hash) -> PathBuf {
        self.dir.join(format!("{}.json", hash.to_hex()))
    }

    /// Add `schema` to the catalog. Returns the BLAKE3 hash of the
    /// canonicalized schema JSON. Idempotent: adding the same schema twice
    /// is a no-op (same content → same hash → same bytes).
    ///
    /// Crash-safety: writes to `<hash>.json.tmp` first, then atomically
    /// renames to `<hash>.json`. POSIX guarantees this rename is atomic
    /// within a single filesystem.
    pub fn add_schema(&self, schema: &Schema) -> Result<blake3::Hash> {
        let raw = schema.to_json()?;
        let canonical = canonicalize_json(&raw)
            .map_err(|e| HeliumError::Format(format!("failed to canonicalize schema JSON: {e}")))?;
        // Use the same hash function helium-core uses in `with_catalog_ref`'s
        // assertion, so registering and then writing-with-catalog-ref always
        // matches.
        let hash = schema_hash(schema)?;
        debug_assert_eq!(hash, blake3::hash(&canonical));
        let final_path = self.path_for(&hash);
        if final_path.exists() {
            // Already registered; content-addressed so we trust the existing file.
            return Ok(hash);
        }
        let tmp_path = self.dir.join(format!("{}.json.tmp", hash.to_hex()));
        // Best-effort cleanup if a previous tmp file exists from a crash.
        let _ = fs::remove_file(&tmp_path);
        let mut f = fs::File::create(&tmp_path).map_err(HeliumError::from)?;
        f.write_all(&canonical).map_err(HeliumError::from)?;
        f.sync_all().map_err(HeliumError::from)?;
        drop(f);
        fs::rename(&tmp_path, &final_path).map_err(HeliumError::from)?;
        Ok(hash)
    }

    /// Look up a registered schema by hash. Surfaces missing-file as
    /// `HeliumError::Format("schema hash <hex> not found by resolver")`.
    pub fn lookup_by_hash(&self, hash: &blake3::Hash) -> Result<Schema> {
        let path = self.path_for(hash);
        let bytes = fs::read(&path).map_err(|_e| {
            HeliumError::Format(format!(
                "schema hash {} not found by resolver",
                hash.to_hex()
            ))
        })?;
        Schema::from_json(&bytes).map_err(|e| {
            HeliumError::Format(format!(
                "catalog schema at hash {} failed to deserialize: {e}",
                hash.to_hex()
            ))
        })
    }

    /// List all registered schemas in the catalog by their BLAKE3 hash.
    /// Order is filesystem-dependent and not stable.
    pub fn list_schemas(&self) -> Result<Vec<blake3::Hash>> {
        let mut out = Vec::new();
        for entry in fs::read_dir(&self.dir).map_err(HeliumError::from)? {
            let entry = entry.map_err(HeliumError::from)?;
            let name = entry.file_name();
            let s = match name.to_str() {
                Some(s) => s,
                None => continue,
            };
            // Match exactly: 64 hex chars + ".json"
            if let Some(stem) = s.strip_suffix(".json")
                && stem.len() == 64
                && stem.bytes().all(|b| b.is_ascii_hexdigit())
            {
                let mut bytes = [0u8; 32];
                if hex_decode_32(stem, &mut bytes).is_ok() {
                    out.push(blake3::Hash::from_bytes(bytes));
                }
            }
        }
        Ok(out)
    }

    /// Verify catalog consistency: for every `<hash>.json` file in the
    /// directory, recompute the canonical hash of its contents and confirm
    /// it matches the filename. Returns `Ok(())` on a clean catalog;
    /// `Err(HeliumError::Format)` with a greppable identifier on the first
    /// mismatch found (the catalog should be repaired before further use).
    pub fn verify_consistency(&self) -> Result<()> {
        for hash in self.list_schemas()? {
            let path = self.path_for(&hash);
            let bytes = match fs::read(&path) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let canonical = canonicalize_json(&bytes).map_err(|e| {
                HeliumError::Format(format!(
                    "catalog inconsistency: file {} is not valid JSON: {e}",
                    path.display(),
                ))
            })?;
            let computed = blake3::hash(&canonical);
            if computed != hash {
                return Err(HeliumError::Format(format!(
                    "catalog inconsistency: filename hash {} does not match content hash {} (file: {})",
                    hash.to_hex(),
                    computed.to_hex(),
                    path.display(),
                )));
            }
        }
        Ok(())
    }

    /// Return a closure suitable for [`crate::HeliumReader::new_with_resolver`].
    /// Captures a clone of this catalog's directory path; safe to use across
    /// multiple reader opens.
    pub fn resolver(&self) -> impl Fn(&blake3::Hash) -> Result<Schema> + use<> {
        let dir = self.dir.clone();
        move |hash: &blake3::Hash| {
            let path = dir.join(format!("{}.json", hash.to_hex()));
            let bytes = fs::read(&path).map_err(|_e| {
                HeliumError::Format(format!(
                    "schema hash {} not found by resolver",
                    hash.to_hex()
                ))
            })?;
            Schema::from_json(&bytes).map_err(|e| {
                HeliumError::Format(format!(
                    "catalog schema at hash {} failed to deserialize: {e}",
                    hash.to_hex()
                ))
            })
        }
    }

    /// Convenience: add `schema` (if not already present) and open a
    /// `HeliumWriter` that emits a catalog-mode (external-schema) header pointing at the
    /// freshly registered schema.
    ///
    /// Equivalent to:
    /// ```ignore
    /// let hash = catalog.add_schema(&schema)?;
    /// HeliumWriter::with_catalog_ref(file, schema, hash, registry)
    /// ```
    pub fn open_writer<W: std::io::Write + std::io::Seek>(
        &self,
        file: W,
        schema: Schema,
        registry: &crate::CoderRegistry,
    ) -> Result<HeliumWriter<W>> {
        let hash = self.add_schema(&schema)?;
        HeliumWriter::with_catalog_ref(file, schema, hash, registry)
    }
}

fn hex_decode_32(s: &str, out: &mut [u8; 32]) -> std::result::Result<(), ()> {
    if s.len() != 64 {
        return Err(());
    }
    for (i, byte) in out.iter_mut().enumerate() {
        let hi = hex_digit(s.as_bytes()[i * 2])?;
        let lo = hex_digit(s.as_bytes()[i * 2 + 1])?;
        *byte = (hi << 4) | lo;
    }
    Ok(())
}

fn hex_digit(b: u8) -> std::result::Result<u8, ()> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Ok(b - b'A' + 10),
        _ => Err(()),
    }
}
