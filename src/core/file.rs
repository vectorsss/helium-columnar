//! `.he` file format — writer and reader.
//!
//! # Header, version and flags
//!
//! Every file begins with an 8-byte header: a stable 6-byte magic `b"HELIUM"`,
//! a 1-byte format generation (`version`), and a 1-byte `flags`. Matching the
//! magic first lets a reader recognise the file type and report an unsupported
//! generation rather than mistaking it for a foreign file.
//!
//! - `version` (byte 6) is bumped only on a core layout change an older reader
//!   cannot parse; a reader rejects a generation newer than it supports.
//! - `flags` (byte 7) is split into two nibbles. The **low nibble** holds
//!   *incompatible* features: an unknown set bit there means the reader cannot
//!   safely parse the file and must reject it. The **high nibble** holds
//!   *compatible* features an older reader can safely ignore. This is the
//!   forward-compatibility contract — it must stay fixed once shipped.
//!
//! Two storage modes are selected by a flag bit, not by separate versions:
//!
//! - **Self-contained** (`flags = 0x00`) — the schema JSON is embedded in the
//!   header. Emitted by [`HeliumWriter::new`].
//! - **Catalog / external-schema** (`flags` bit 0 set → `0x01`) — the header
//!   carries a 36-byte hash reference (32-byte BLAKE3 + 4-byte CRC32C) instead
//!   of an embedded schema; the reader resolves the hash via a caller-provided
//!   closure. Emitted by [`HeliumWriter::with_catalog_ref`].
//!
//! In both modes the schema header (when present) and the footer JSON are
//! zstd-compressed, every physical column and the footer carry a CRC32C, and
//! readers reject any file whose CRC fails — errors include the column/stripe
//! context so a caller can isolate the bad byte range.
//!
//! # Self-contained binary layout
//!
//! ```text
//! [0..6]          magic = b"HELIUM"
//! [6]             version: u8
//! [7]             flags: u8  (0x00 = self-contained)
//! [8..12]         schema_len: u32 LE        (length of the COMPRESSED schema bytes)
//! [12..12+S]      zstd-compressed schema JSON
//! [body_start..]  stripes laid out in order; each stripe is a contiguous run
//!                 of its logical columns' physical-column encoded bytes.
//! [..tail-20]     footer bytes (zstd-compressed)
//! [tail-20..-12]  footer_len: u64 LE  (= compressed byte length)
//! [tail-12..-8]   footer_crc32c: u32 LE  (CRC32C over the on-disk compressed bytes;
//!                 corruption is caught before decompression)
//! [tail-8..]      the 8-byte header, echoed as a completeness sentinel
//! ```
//!
//! # Catalog binary layout
//!
//! Same as self-contained except `flags` has bit 0 set and the `[8..12+S]`
//! region (`schema_len` + embedded schema JSON) is replaced by a fixed 36-byte
//! schema slot — 32-byte BLAKE3 hash + 4-byte CRC32C of the hash — which the
//! reader resolves to a `Schema` via a caller-provided closure.

use std::collections::HashMap;
use std::fmt;
use std::io::{Read, Seek, SeekFrom, Write};

use serde::{Deserialize, Serialize};

use super::coder::{ColumnData, DataType};
use super::error::{HeliumError, Result};
use super::footer_stats::{
    ContainmentFilter, LeafStats, MinMaxValue, PhysicalColumnStats,
    compute_filters_for_logical_column, compute_stats_for_logical_column,
};
use super::pipeline::Pipeline;
use super::registry::CoderRegistry;
use super::schema::{LogicalColumn, LogicalType, Schema};

pub use super::footer_stats::{
    bloom_might_contain, filter_might_contain_mmv, min_max_value_to_hash_bytes,
};

/// Stable file-type magic — the first 6 bytes of every `.he` file, identical
/// across all format generations. A reader matches this first, so an unknown
/// generation surfaces as "unsupported version" rather than a foreign magic.
pub const MAGIC: &[u8; 6] = b"HELIUM";

/// On-disk format generation, written at byte 6 of the 8-byte header. Bumped
/// only when the core layout changes such that an older reader cannot parse it.
pub const FORMAT_VERSION: u8 = 1;

/// Highest format generation this build can read.
const MAX_FORMAT_VERSION: u8 = 1;

/// Header flags byte (byte 7). The layout splits into two halves so a reader
/// knows how to treat a flag it does not recognise:
/// - **low nibble = incompatible features**: if an unknown bit here is set, the
///   reader cannot safely parse the file and must reject it.
/// - **high nibble = compatible features**: an unknown bit here is additive and
///   safe to ignore (an older reader still reads the data correctly).
const FLAGS_INCOMPAT_MASK: u8 = 0b0000_1111;

/// Incompatible flag: the schema is stored externally (catalog mode) — the
/// header carries a hash slot instead of an embedded schema.
const FLAG_EXTERNAL_SCHEMA: u8 = 0b0000_0001;

/// Incompatible flags this build understands. A set incompatible bit outside
/// this set means the file uses a feature we cannot read.
const KNOWN_INCOMPAT_FLAGS: u8 = FLAG_EXTERNAL_SCHEMA;

/// Build the 8-byte file header: `b"HELIUM"` (6) + version (1) + flags (1).
fn file_header(flags: u8) -> [u8; 8] {
    let mut h = [0u8; 8];
    h[..6].copy_from_slice(MAGIC);
    h[6] = FORMAT_VERSION;
    h[7] = flags;
    h
}

/// zstd compression level for the schema header bytes (v5) and the footer
/// JSON bytes (v5/v6). Level 3 is zstd's default — payloads are small and
/// entropy is high (lots of repeated tag names), so the marginal gain at
/// higher levels is tiny while encoding time grows. Frozen alongside the magic.
const SCHEMA_ZSTD_LEVEL: i32 = 3;

/// zstd compression level for the footer JSON bytes in v5/v6 files.
/// Same rationale as `SCHEMA_ZSTD_LEVEL`: small payload, high key repetition.
const FOOTER_ZSTD_LEVEL: i32 = 3;

/// Total size of the catalog-mode (v6) schema slot on disk: 32 bytes raw
/// BLAKE3 hash plus 4 bytes CRC32C of the hash bytes (LE). Frozen by §6.5
/// Surface F.
pub const CATALOG_SCHEMA_SLOT_LEN: usize = 36;

/// Trait-object form of the catalog (v6) schema resolver — used internally by
/// `HeliumReader::new_inner` to avoid type-complexity lints on
/// `Option<&dyn Fn(...)>`.
type SchemaResolver<'a> = &'a dyn Fn(&blake3::Hash) -> Result<Schema>;

// ---------------------------------------------------------------------------
// Footer structures (optional fields use #[serde(default)] so footers written
// without stats/filters still parse)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Footer {
    stripes: Vec<StripeIndex>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StripeIndex {
    row_count: u64,
    columns: Vec<LogicalLocation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LogicalLocation {
    physical: Vec<PhysicalLocation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PhysicalLocation {
    offset: u64,
    length: u64,
    /// CRC32C of the encoded bytes for this physical column. Always written by
    /// v5/v6; `#[serde(default)]` keeps deserialization total if absent.
    #[serde(default)]
    crc32c: u32,
    /// Minimum value for this physical column. `None` if the column was
    /// empty, all-null, all-NaN (for floats), or stats were disabled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    min: Option<MinMaxValue>,
    /// Maximum value for this physical column. `None` for the same reasons
    /// as `min`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    max: Option<MinMaxValue>,
    /// Count of null/missing rows for this physical column. `None` if stats
    /// were disabled. Zero for non-nullable columns.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    null_count: Option<u64>,
    /// Per-column containment filter for equality pushdown. `None` when
    /// filters were disabled, the column type is unsupported (nested /
    /// Decimal128), or the column was empty.
    ///
    /// Additive field — old files without this field deserialise with
    /// `filter = None` via `serde(default)`, giving conservative "keep all
    /// stripes" behaviour for equality predicates.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    filter: Option<ContainmentFilter>,
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

/// Validate the schema's pipelines and return them. Shared between the v5 and
/// v6 (catalog-mode) writer constructors.
fn resolve_and_validate_pipelines(
    schema: &Schema,
    registry: &CoderRegistry,
) -> Result<Vec<Vec<Pipeline>>> {
    let pipelines = schema.resolve_all(registry)?;
    for (spec, col_pipes) in schema.columns.iter().zip(pipelines.iter()) {
        for (field, pipe) in spec
            .logical_type
            .physical_fields()
            .iter()
            .zip(col_pipes.iter())
        {
            if pipe.output_type() != DataType::Bytes {
                return Err(HeliumError::Schema {
                    column: spec.name.clone(),
                    reason: format!(
                        "physical field '{}' pipeline must terminate in Bytes, got {:?}",
                        field.role,
                        pipe.output_type()
                    ),
                });
            }
        }
    }
    Ok(pipelines)
}

/// Writes a Helium `.he` file column-by-column, stripe-by-stripe.
///
/// Obtain one via [`HeliumWriter::new`]. Call [`write_column`] for every
/// column in every stripe, then [`finish_stripe`] between stripes, and
/// finally [`finish`] to flush the footer. Writing columns out of order or
/// omitting a column raises an error at [`finish_stripe`] time.
///
/// [`write_column`]: HeliumWriter::write_column
/// [`finish_stripe`]: HeliumWriter::finish_stripe
/// [`finish`]: HeliumWriter::finish
pub struct HeliumWriter<W: Write + Seek> {
    inner: W,
    schema: Schema,
    pipelines: Vec<Vec<Pipeline>>,
    column_index: HashMap<String, usize>,
    body_start: u64,
    current_written: Vec<Option<LogicalLocation>>,
    current_row_count: Option<usize>,
    stripes: Vec<StripeIndex>,
    finished: bool,
    /// The 8-byte file header, echoed verbatim at the end of the file as a
    /// completeness sentinel. Mirrors the start header (`b"HELIUM"` + version +
    /// flags) so the reader can detect truncation.
    end_magic: [u8; 8],
    /// Whether to compute and embed per-column min/max statistics in the
    /// footer. Defaults to `true`. Disable via [`HeliumWriter::with_stats_disabled`].
    stats_enabled: bool,
    /// Whether to compute and embed per-column containment filters in the
    /// footer. Defaults to `true`. Disable via [`HeliumWriter::with_filters_disabled`].
    filters_enabled: bool,
}

impl<W: Write + Seek> fmt::Debug for HeliumWriter<W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HeliumWriter")
            .field("schema", &self.schema)
            .field("current_row_count", &self.current_row_count)
            .field("stripes_so_far", &self.stripes.len())
            .field("finished", &self.finished)
            .finish_non_exhaustive()
    }
}

impl<W: Write + Seek> HeliumWriter<W> {
    /// Open a writer that emits v5 magic + zstd-compressed schema JSON in the
    /// header and zstd-compressed footer JSON (the default single-file mode).
    pub fn new(mut inner: W, schema: Schema, registry: &CoderRegistry) -> Result<Self> {
        schema.validate()?;
        let pipelines = resolve_and_validate_pipelines(&schema, registry)?;

        // Self-contained mode: no flags set. Schema JSON is zstd-compressed
        // before being embedded; the compressed length is what the schema_len
        // field on disk reports.
        let header = file_header(0);
        inner.write_all(&header)?;
        let schema_json = schema.to_json()?;
        let compressed_schema = zstd::encode_all(&schema_json[..], SCHEMA_ZSTD_LEVEL)
            .map_err(|e| HeliumError::Format(format!("zstd compress schema: {e}")))?;
        let schema_len: u32 = compressed_schema
            .len()
            .try_into()
            .map_err(|_| HeliumError::Format("compressed schema exceeds u32 length".into()))?;
        inner.write_all(&schema_len.to_le_bytes())?;
        inner.write_all(&compressed_schema)?;

        Self::finish_init(inner, schema, pipelines, header)
    }

    /// Open a writer that emits **v6 catalog-mode magic** + a 36-byte schema
    /// slot (32-byte BLAKE3 hash + 4-byte CRC32C of those bytes). The actual
    /// schema is *not* embedded; readers must resolve the hash via
    /// [`HeliumReader::new_with_resolver`] backed by a catalog directory.
    ///
    /// Per PLAN_V2 §6.5 Surface C, this constructor takes the schema **and**
    /// the precomputed hash separately:
    /// - `schema` is needed in-memory for column write-time validation,
    ///   `physical_fields()` decomposition, and per-leaf pipeline construction.
    /// - `schema_hash` is the on-disk identity — the writer embeds it in the
    ///   v6 schema slot. The caller is responsible for computing the hash via
    ///   the canonicalizer in `helium-catalog`. helium-core does not depend
    ///   on the canonicalizer to keep the format-only surface clean.
    ///
    /// Convenience: use `helium_catalog::Catalog::open_writer(file, schema, registry)`
    /// which canonicalizes and registers the schema, then forwards here.
    ///
    /// Default writer behaviour (`HeliumWriter::new`) is unchanged: catalog
    /// mode is **opt-in**.
    pub fn with_catalog_ref(
        mut inner: W,
        schema: Schema,
        schema_hash: blake3::Hash,
        registry: &CoderRegistry,
    ) -> Result<Self> {
        schema.validate()?;

        // §6.5 Surface C lock: assert the caller-supplied hash matches the
        // schema's canonical hash. Catches the most likely caller bug —
        // passing a hash for the wrong schema, which would silently produce a
        // .he file that resolves to the wrong schema in the catalog.
        let expected_hash = super::canonicalize::schema_hash(&schema)?;
        if expected_hash != schema_hash {
            return Err(HeliumError::Schema {
                column: "<header>".into(),
                reason: format!(
                    "schema_hash {} passed to HeliumWriter::with_catalog_ref does not \
                     match canonicalize_and_hash(&schema) = {}",
                    schema_hash.to_hex(),
                    expected_hash.to_hex(),
                ),
            });
        }

        let pipelines = resolve_and_validate_pipelines(&schema, registry)?;

        // Catalog mode: the external-schema flag, then a 36-byte schema slot
        // = 32-byte BLAKE3 hash + 4-byte CRC32C of those bytes.
        let header = file_header(FLAG_EXTERNAL_SCHEMA);
        inner.write_all(&header)?;
        let hash_bytes: &[u8; 32] = schema_hash.as_bytes();
        inner.write_all(hash_bytes)?;
        let crc = crc32c::crc32c(hash_bytes);
        inner.write_all(&crc.to_le_bytes())?;

        Self::finish_init(inner, schema, pipelines, header)
    }

    /// Finish initialization after the version-specific header has been
    /// written. Computes `body_start`, builds the column-name index, and
    /// constructs the writer.
    fn finish_init(
        mut inner: W,
        schema: Schema,
        pipelines: Vec<Vec<Pipeline>>,
        end_magic: [u8; 8],
    ) -> Result<Self> {
        let body_start = inner.stream_position()?;
        let column_index = schema
            .columns
            .iter()
            .enumerate()
            .map(|(i, c)| (c.name.clone(), i))
            .collect();
        let n = schema.columns.len();

        Ok(Self {
            inner,
            schema,
            pipelines,
            column_index,
            body_start,
            current_written: vec![None; n],
            current_row_count: None,
            stripes: Vec::new(),
            finished: false,
            end_magic,
            stats_enabled: true,
            filters_enabled: true,
        })
    }

    /// Returns the schema this writer was constructed with.
    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// Disable per-column min/max statistics writing.
    ///
    /// When disabled the footer grows by no stats overhead, but no predicate
    /// pushdown information is available to readers. Default is stats-on.
    ///
    /// Must be called before the first [`write_column`] call to take effect.
    ///
    /// [`write_column`]: HeliumWriter::write_column
    pub fn with_stats_disabled(mut self) -> Self {
        self.stats_enabled = false;
        self
    }

    /// Disable per-column containment-filter writing.
    ///
    /// When disabled the footer contains no equality-pushdown information and
    /// `WHERE col = x` / `WHERE col IN (...)` predicates cannot prune stripes.
    /// Min/max range pushdown (`WHERE col > x`) is **not** affected.
    ///
    /// Useful for files where the footer size overhead of Bloom filters is
    /// unacceptable (e.g. very wide schemas × many stripes). Default is
    /// filters-on.
    ///
    /// Must be called before the first [`write_column`] call to take effect.
    ///
    /// [`write_column`]: HeliumWriter::write_column
    pub fn with_filters_disabled(mut self) -> Self {
        self.filters_enabled = false;
        self
    }

    /// Encode and buffer one logical column for the current stripe.
    ///
    /// Each column must be written exactly once per stripe. Call
    /// [`finish_stripe`] after all columns have been written to commit the
    /// stripe to disk.
    ///
    /// [`finish_stripe`]: HeliumWriter::finish_stripe
    pub fn write_column(&mut self, name: &str, data: LogicalColumn) -> Result<()> {
        if self.finished {
            return Err(HeliumError::Format("writer already finished".into()));
        }
        let idx = *self
            .column_index
            .get(name)
            .ok_or_else(|| HeliumError::Schema {
                column: name.into(),
                reason: "column not in schema".into(),
            })?;
        if self.current_written[idx].is_some() {
            return Err(HeliumError::Schema {
                column: name.into(),
                reason: "column written twice in the current stripe".into(),
            });
        }

        let spec = &self.schema.columns[idx];
        let row_len = data.row_count();
        match self.current_row_count {
            None => self.current_row_count = Some(row_len),
            Some(n) if n != row_len => {
                return Err(HeliumError::Schema {
                    column: name.into(),
                    reason: format!("row count {row_len} != first column's {n} in this stripe"),
                });
            }
            _ => {}
        }

        // Compute per-leaf statistics before decompose (while we still have the
        // typed values). We clone only when stats_enabled to avoid any overhead
        // when the feature is turned off.
        let leaf_stats: Option<LeafStats> = if self.stats_enabled {
            Some(compute_stats_for_logical_column(&data, &spec.logical_type))
        } else {
            None
        };

        // Compute per-leaf containment filters before decompose.
        let leaf_filters: Option<Vec<Option<ContainmentFilter>>> = if self.filters_enabled {
            Some(compute_filters_for_logical_column(
                &data,
                &spec.logical_type,
            ))
        } else {
            None
        };

        let physical_parts = data.decompose(&spec.logical_type).map_err(|e| match e {
            HeliumError::Schema { reason, .. } => HeliumError::Schema {
                column: name.into(),
                reason,
            },
            other => other,
        })?;

        let pipes = &self.pipelines[idx];
        if physical_parts.len() != pipes.len() {
            return Err(HeliumError::Schema {
                column: name.into(),
                reason: format!(
                    "decomposed to {} physical parts but schema has {} pipelines",
                    physical_parts.len(),
                    pipes.len()
                ),
            });
        }

        let mut phys_locs = Vec::with_capacity(pipes.len());
        for (leaf_idx, (part, pipe)) in physical_parts.into_iter().zip(pipes.iter()).enumerate() {
            let encoded = pipe.encode(part).map_err(|e| HeliumError::Schema {
                column: name.into(),
                reason: e.to_string(),
            })?;
            let ColumnData::Bytes(bytes) = encoded else {
                unreachable!("pipeline output type validated as Bytes in new()");
            };
            let offset = self.inner.stream_position()? - self.body_start;
            let length = bytes.len() as u64;
            let crc = crc32c::crc32c(&bytes);
            self.inner.write_all(&bytes)?;
            let (min, max, null_count) = leaf_stats
                .as_ref()
                .and_then(|s| s.get(leaf_idx))
                .cloned()
                .unwrap_or((None, None, None));
            let filter = leaf_filters
                .as_ref()
                .and_then(|f| f.get(leaf_idx))
                .and_then(|f| f.clone());
            phys_locs.push(PhysicalLocation {
                offset,
                length,
                crc32c: crc,
                min,
                max,
                null_count,
                filter,
            });
        }
        self.current_written[idx] = Some(LogicalLocation {
            physical: phys_locs,
        });
        Ok(())
    }

    /// Close the current stripe and start a new one. All columns must have
    /// been written in the current stripe. After this call the writer
    /// accepts `write_column` again for the next stripe.
    pub fn finish_stripe(&mut self) -> Result<()> {
        if self.finished {
            return Err(HeliumError::Format("writer already finished".into()));
        }
        for (i, loc) in self.current_written.iter().enumerate() {
            if loc.is_none() {
                return Err(HeliumError::Schema {
                    column: self.schema.columns[i].name.clone(),
                    reason: "column missing from current stripe".into(),
                });
            }
        }
        let row_count = self.current_row_count.unwrap_or(0) as u64;
        // All entries are Some — verified by the loop above.
        let columns: Vec<LogicalLocation> = self
            .current_written
            .iter()
            .filter_map(|w| w.clone())
            .collect();
        self.stripes.push(StripeIndex { row_count, columns });
        self.current_written = vec![None; self.schema.columns.len()];
        self.current_row_count = None;
        Ok(())
    }

    /// Finalize. If there is an in-progress stripe (at least one column
    /// written since the last boundary) it is finalized first.
    pub fn finish(mut self) -> Result<W> {
        let has_in_progress = self.current_written.iter().any(|w| w.is_some());
        if has_in_progress || self.stripes.is_empty() {
            self.finish_stripe()?;
        }

        let footer = Footer {
            stripes: std::mem::take(&mut self.stripes),
        };
        let footer_json = serde_json::to_vec(&footer)?;
        // Both writer outputs (v5, v6) zstd-compress the footer JSON. The CRC
        // is over the compressed bytes so corruption is detected before
        // decompression (mirrors the schema header).
        let footer_bytes = zstd::encode_all(&footer_json[..], FOOTER_ZSTD_LEVEL)
            .map_err(|e| HeliumError::Format(format!("zstd compress footer: {e}")))?;
        let footer_len = footer_bytes.len() as u64;
        let footer_crc = crc32c::crc32c(&footer_bytes);
        self.inner.write_all(&footer_bytes)?;
        self.inner.write_all(&footer_len.to_le_bytes())?;
        self.inner.write_all(&footer_crc.to_le_bytes())?;
        self.inner.write_all(&self.end_magic)?;
        self.inner.flush()?;
        self.finished = true;
        Ok(self.inner)
    }
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

/// Reads a Helium `.he` file column-by-column, stripe-by-stripe.
///
/// Obtain one via [`HeliumReader::new`]. Call [`read_column`] to decode a
/// full logical column across all stripes, or [`read_column_at_stripe`] for
/// single-stripe access (required for dict columns in multi-stripe files).
///
/// [`read_column`]: HeliumReader::read_column
/// [`read_column_at_stripe`]: HeliumReader::read_column_at_stripe
pub struct HeliumReader<R: Read + Seek> {
    inner: R,
    schema: Schema,
    pipelines: Vec<Vec<Pipeline>>,
    body_start: u64,
    /// Total file size in bytes (cached at open time for [`region_sizes`]).
    ///
    /// [`region_sizes`]: HeliumReader::region_sizes
    file_len: u64,
    /// On-disk byte length of the (zstd-compressed) footer payload. Cached at
    /// open time for [`region_sizes`].
    ///
    /// [`region_sizes`]: HeliumReader::region_sizes
    footer_len: u64,
    stripes: Vec<StripeIndex>,
    /// On-disk format generation (header byte 6).
    version: u8,
    /// Whether the schema is stored externally (catalog mode).
    external_schema: bool,
}

impl<R: Read + Seek> fmt::Debug for HeliumReader<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HeliumReader")
            .field("schema", &self.schema)
            .field("version", &self.version)
            .field("stripes", &self.stripes.len())
            .field("total_rows", &self.row_count())
            .finish_non_exhaustive()
    }
}

impl<R: Read + Seek> HeliumReader<R> {
    /// Open a self-contained `.he` file (schema embedded in the header).
    ///
    /// **Catalog-mode files (external schema, opt-in) require
    /// [`new_with_resolver`].** Such a file seen by this constructor surfaces as
    /// `HeliumError::Format("catalog-mode file requires a schema resolver but none was provided")`
    /// — never silent corruption.
    ///
    /// [`new_with_resolver`]: HeliumReader::new_with_resolver
    pub fn new(inner: R, registry: &CoderRegistry) -> Result<Self> {
        Self::new_inner(inner, registry, None)
    }

    /// Open a `.he` file with a schema resolver for catalog-mode files.
    ///
    /// For self-contained files the resolver is ignored — the schema is embedded
    /// in the header. For catalog-mode files, the resolver is called with the
    /// 32-byte BLAKE3 hash from the schema slot and must return the matching
    /// `Schema` (typically by looking it up in a catalog directory).
    ///
    /// Resolver-returned errors propagate as-is.
    pub fn new_with_resolver<F>(inner: R, registry: &CoderRegistry, resolver: F) -> Result<Self>
    where
        F: Fn(&blake3::Hash) -> Result<Schema>,
    {
        Self::new_inner(inner, registry, Some(&resolver))
    }

    fn new_inner(
        mut inner: R,
        registry: &CoderRegistry,
        resolver: Option<SchemaResolver<'_>>,
    ) -> Result<Self> {
        // 8-byte header: b"HELIUM" (6) + version (1) + flags (1).
        let mut header = [0u8; 8];
        inner
            .read_exact(&mut header)
            .map_err(|e| HeliumError::Format(format!("cannot read file header: {e}")))?;
        if &header[..6] != MAGIC {
            return Err(HeliumError::Format(format!(
                "not a Helium file: bad magic {:02x?}",
                &header[..6]
            )));
        }
        let version = header[6];
        if version == 0 || version > MAX_FORMAT_VERSION {
            return Err(HeliumError::Format(format!(
                "unsupported .he format generation {version}: this build reads up to \
                 {MAX_FORMAT_VERSION} — regenerate the file from source"
            )));
        }
        let flags = header[7];
        // An incompatible flag we don't understand means we cannot parse the
        // file safely; compatible (high-nibble) flags are ignored if unknown.
        let unknown_incompat = flags & FLAGS_INCOMPAT_MASK & !KNOWN_INCOMPAT_FLAGS;
        if unknown_incompat != 0 {
            return Err(HeliumError::Format(format!(
                "unsupported .he format features (incompatible flags {unknown_incompat:#010b}) \
                 — regenerate the file from source"
            )));
        }
        let external_schema = flags & FLAG_EXTERNAL_SCHEMA != 0;

        let schema = if external_schema {
            // Catalog mode: a 36-byte schema slot (32-byte BLAKE3 + 4-byte
            // CRC32C of the hash), resolved out-of-band via the resolver.
            let resolver = resolver.ok_or_else(|| {
                HeliumError::Format(
                    "catalog-mode file requires a schema resolver but none was provided".into(),
                )
            })?;
            let mut slot = [0u8; CATALOG_SCHEMA_SLOT_LEN];
            inner.read_exact(&mut slot)?;
            let mut hash_bytes = [0u8; 32];
            hash_bytes.copy_from_slice(&slot[..32]);
            // slot is exactly CATALOG_SCHEMA_SLOT_LEN=36 bytes; [32..36] is 4 bytes.
            let stored_crc = u32::from_le_bytes(
                slot[32..36]
                    .try_into()
                    .map_err(|_| HeliumError::Format("catalog schema-slot read failed".into()))?,
            );
            let actual_crc = crc32c::crc32c(&hash_bytes);
            if actual_crc != stored_crc {
                return Err(HeliumError::Format(format!(
                    "catalog schema-slot CRC mismatch: stored {stored_crc:#x}, computed {actual_crc:#x}"
                )));
            }
            let hash = blake3::Hash::from_bytes(hash_bytes);
            resolver(&hash)?
        } else {
            // Self-contained mode: schema_len (u32 LE) + zstd-compressed schema.
            let mut len_buf = [0u8; 4];
            inner.read_exact(&mut len_buf)?;
            let schema_len = u32::from_le_bytes(len_buf) as usize;
            let mut schema_bytes = vec![0u8; schema_len];
            inner.read_exact(&mut schema_bytes)?;
            let schema_json = zstd::decode_all(&schema_bytes[..])
                .map_err(|e| HeliumError::Format(format!("zstd decompress schema: {e}")))?;
            Schema::from_json(&schema_json)?
        };
        let body_start = inner.stream_position()?;
        let pipelines = schema.resolve_all(registry)?;

        let file_len = inner.seek(SeekFrom::End(0))?;
        // v5/v6 share a 20-byte trailer: footer_len(8) + footer_crc32c(4) + magic(8).
        let trailer_len: u64 = 20;
        if file_len < body_start + trailer_len {
            return Err(HeliumError::Format("file truncated: no footer".into()));
        }
        inner.seek(SeekFrom::End(-(trailer_len as i64)))?;
        let mut fl_buf = [0u8; 8];
        inner.read_exact(&mut fl_buf)?;
        let footer_len = u64::from_le_bytes(fl_buf);

        let mut crc_buf = [0u8; 4];
        inner.read_exact(&mut crc_buf)?;
        let stored_crc = u32::from_le_bytes(crc_buf);

        let mut end_magic = [0u8; 8];
        inner.read_exact(&mut end_magic)?;
        if end_magic != header {
            return Err(HeliumError::Format(format!(
                "bad end magic: {end_magic:02x?}"
            )));
        }
        if footer_len > file_len - trailer_len - body_start {
            return Err(HeliumError::Format(format!(
                "footer length {footer_len} overruns body"
            )));
        }
        inner.seek(SeekFrom::Start(file_len - trailer_len - footer_len))?;
        let mut footer_bytes = vec![0u8; footer_len as usize];
        inner.read_exact(&mut footer_bytes)?;
        // CRC is over the on-disk (zstd-compressed) bytes, so corruption is
        // caught before decompression — same as the schema header CRC.
        let actual = crc32c::crc32c(&footer_bytes);
        if actual != stored_crc {
            return Err(HeliumError::Corrupted {
                coder: "<footer>".into(),
                reason: format!(
                    "footer CRC32C mismatch: expected {stored_crc:#x}, got {actual:#x}"
                ),
            });
        }
        // v5/v6: zstd-decompress the footer before parsing JSON.
        let footer_json: Vec<u8> = zstd::decode_all(&footer_bytes[..])
            .map_err(|e| HeliumError::Format(format!("zstd decompress footer: {e}")))?;
        let footer: Footer = serde_json::from_slice(&footer_json)?;

        if footer.stripes.is_empty() {
            return Err(HeliumError::Format("footer has no stripes".into()));
        }
        for (s_idx, stripe) in footer.stripes.iter().enumerate() {
            if stripe.columns.len() != schema.columns.len() {
                return Err(HeliumError::Format(format!(
                    "stripe {s_idx} has {} column entries but schema has {}",
                    stripe.columns.len(),
                    schema.columns.len()
                )));
            }
            for (c_idx, (spec, loc)) in schema.columns.iter().zip(stripe.columns.iter()).enumerate()
            {
                let expected = spec.logical_type.physical_fields().len();
                if loc.physical.len() != expected {
                    return Err(HeliumError::Format(format!(
                        "stripe {s_idx} column {c_idx} ('{}') has {} physical entries, expected {expected}",
                        spec.name,
                        loc.physical.len()
                    )));
                }
            }
        }

        Ok(Self {
            inner,
            schema,
            pipelines,
            body_start,
            file_len,
            footer_len,
            stripes: footer.stripes,
            version,
            external_schema,
        })
    }

    /// Returns the schema embedded in this file.
    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// Total row count across all stripes.
    pub fn row_count(&self) -> u64 {
        self.stripes.iter().map(|s| s.row_count).sum()
    }

    /// Number of stripes in this file.
    pub fn stripe_count(&self) -> usize {
        self.stripes.len()
    }

    /// Row count for a specific stripe (0-indexed).
    ///
    /// Returns `None` if `stripe_idx` is out of range.
    pub fn stripe_row_count(&self, stripe_idx: usize) -> Option<u64> {
        self.stripes.get(stripe_idx).map(|s| s.row_count)
    }

    /// Per-stripe row counts, in order.
    pub fn stripe_row_counts(&self) -> Vec<u64> {
        self.stripes.iter().map(|s| s.row_count).collect()
    }

    /// Per-stripe per-physical-column statistics for a logical column.
    ///
    /// Returns one [`PhysicalColumnStats`] per **physical** leaf of the
    /// logical column, in the same order as
    /// `Schema::column().logical_type.physical_fields()`. Returns `None`
    /// if `stripe_idx` is out of range or `column_name` is not in the
    /// schema.
    ///
    /// All fields inside a [`PhysicalColumnStats`] are `None` for v5/v6 files
    /// written without stats.
    pub fn stripe_column_stats(
        &self,
        stripe_idx: usize,
        column_name: &str,
    ) -> Option<Vec<PhysicalColumnStats>> {
        let col_idx = self.schema.column_index(column_name)?;
        let stripe = self.stripes.get(stripe_idx)?;
        let locs = &stripe.columns[col_idx].physical;
        Some(
            locs.iter()
                .map(|loc| PhysicalColumnStats {
                    min: loc.min.clone(),
                    max: loc.max.clone(),
                    null_count: loc.null_count,
                })
                .collect(),
        )
    }

    /// Per-stripe per-physical-column containment filter for a logical column.
    ///
    /// Returns one `Option<ContainmentFilter>` per **physical** leaf of the
    /// logical column, in the same order as
    /// `Schema::column().logical_type.physical_fields()`. Returns `None`
    /// if `stripe_idx` is out of range or `column_name` is not in the schema.
    ///
    /// `None` at the inner level means no filter was computed for that leaf
    /// (disabled, unsupported type, or empty column). The caller should treat
    /// a missing filter as "might contain" (conservative).
    pub fn stripe_column_filter(
        &self,
        stripe_idx: usize,
        column_name: &str,
    ) -> Option<Vec<Option<ContainmentFilter>>> {
        let col_idx = self.schema.column_index(column_name)?;
        let stripe = self.stripes.get(stripe_idx)?;
        let locs = &stripe.columns[col_idx].physical;
        Some(locs.iter().map(|loc| loc.filter.clone()).collect())
    }

    /// Iterator over all logical column names in schema order.
    pub fn column_names(&self) -> impl Iterator<Item = &str> {
        self.schema.columns.iter().map(|c| c.name.as_str())
    }

    /// Per-stripe per-logical-column on-disk byte size summary.
    ///
    /// Returns a vector of `(column_name, total_encoded_bytes_across_stripes)`
    /// in schema column order. The byte count is the sum of all physical-leaf
    /// encoded sizes for that logical column, across every stripe. This is the
    /// **on-disk** encoded size, not the in-memory decoded size.
    ///
    /// Useful for `helium stats` and any other tool that needs to understand
    /// the storage breakdown of a `.he` file without reading the column data.
    pub fn column_byte_sizes(&self) -> Vec<(String, u64)> {
        self.schema
            .columns
            .iter()
            .enumerate()
            .map(|(col_idx, spec)| {
                let total: u64 = self
                    .stripes
                    .iter()
                    .map(|stripe| {
                        stripe.columns[col_idx]
                            .physical
                            .iter()
                            .map(|loc| loc.length)
                            .sum::<u64>()
                    })
                    .sum();
                (spec.name.clone(), total)
            })
            .collect()
    }

    /// Approximate on-disk region sizes: `(header_bytes, body_bytes, footer_bytes)`.
    ///
    /// - `header_bytes` = byte offset where the body starts (everything before
    ///   the first stripe: magic + schema slot).
    /// - `body_bytes` = total file size minus header and footer regions.
    /// - `footer_bytes` = on-disk footer payload length plus the 20-byte
    ///   trailer (footer_len field + CRC32C field + end magic).
    ///
    /// These numbers are derived from metadata cached at open time — no extra
    /// I/O is required.
    pub fn region_sizes(&self) -> (u64, u64, u64) {
        // v5/v6 share a 20-byte trailer: footer_len(8) + footer_crc32c(4) + magic(8).
        let trailer_len: u64 = 20;
        let footer_region = self.footer_len + trailer_len;
        let body_bytes = self
            .file_len
            .saturating_sub(self.body_start + footer_region);
        (self.body_start, body_bytes, footer_region)
    }

    /// Return a human-readable format descriptor for display (e.g. `helium
    /// stats`): the generation plus the storage mode, such as
    /// `"v1"` (self-contained) or `"v1 (catalog)"`.
    pub fn version_str(&self) -> String {
        if self.external_schema {
            format!("v{} (catalog)", self.version)
        } else {
            format!("v{}", self.version)
        }
    }

    /// Read a logical column at a specific stripe (0-indexed).
    pub fn read_column_at_stripe(
        &mut self,
        name: &str,
        stripe_idx: usize,
    ) -> Result<LogicalColumn> {
        let idx = self
            .schema
            .column_index(name)
            .ok_or_else(|| HeliumError::Schema {
                column: name.into(),
                reason: "column not in schema".into(),
            })?;
        if stripe_idx >= self.stripes.len() {
            return Err(HeliumError::Format(format!(
                "stripe index {stripe_idx} out of range (have {})",
                self.stripes.len()
            )));
        }
        self.read_column_piece(idx, stripe_idx)
    }

    /// Read a logical column across *all* stripes, concatenating results.
    ///
    /// For `Dictionary` columns in multi-stripe files, this returns an error
    /// because different stripes may have different dictionaries.
    /// Use [`HeliumReader::read_column_at_stripe`] for those.
    pub fn read_column(&mut self, name: &str) -> Result<LogicalColumn> {
        let idx = self
            .schema
            .column_index(name)
            .ok_or_else(|| HeliumError::Schema {
                column: name.into(),
                reason: "column not in schema".into(),
            })?;
        let stripe_count = self.stripes.len();
        if stripe_count == 1 {
            return self.read_column_piece(idx, 0);
        }

        let logical_type = self.schema.columns[idx].logical_type.clone();
        if matches!(logical_type, LogicalType::Dictionary { .. }) {
            return Err(HeliumError::Schema {
                column: name.into(),
                reason:
                    "dict columns cannot be concatenated across stripes; use read_column_at_stripe"
                        .into(),
            });
        }

        let mut pieces = Vec::with_capacity(stripe_count);
        for s_idx in 0..stripe_count {
            pieces.push(self.read_column_piece(idx, s_idx)?);
        }
        concat_logical_columns(pieces, &logical_type).map_err(|e| match e {
            HeliumError::Schema { reason, .. } => HeliumError::Schema {
                column: name.into(),
                reason,
            },
            other => other,
        })
    }

    /// Project a subset of columns into a new `.he` file (a column "slice").
    ///
    /// Writes `columns` (in the given order) to `dst` as a fresh self-contained
    /// (v5) file whose schema is [`Schema::project`]ed to exactly those columns,
    /// preserving each column's encodings and the source's stripe boundaries.
    /// Returns the `dst` sink (like [`HeliumWriter::finish`]).
    ///
    /// **Zero-copy**: the already-encoded leaf bytes are copied verbatim — no
    /// coder runs, and each leaf's stored CRC32C, min/max, null-count and
    /// containment filter are reused from the source footer (they cannot change
    /// when the bytes don't). Only the per-leaf byte offsets are recomputed.
    /// Every leaf's CRC is re-checked against the source footer during the copy,
    /// so corruption surfaces here rather than being propagated. Works for every
    /// logical type, including `Dictionary` columns in multi-stripe files.
    ///
    /// `registry` is used only to validate that the projected schema resolves
    /// (every kept column's pipeline is buildable); no encoding is performed.
    ///
    /// Errors if `columns` is empty, or any name is absent / requested twice.
    ///
    /// [`Schema::project`]: crate::Schema::project
    pub fn project_to<W: Write + Seek>(
        &mut self,
        columns: &[&str],
        mut dst: W,
        registry: &CoderRegistry,
    ) -> Result<W> {
        if columns.is_empty() {
            return Err(HeliumError::Schema {
                column: "<projection>".into(),
                reason: "projection must select at least one column".into(),
            });
        }
        let subset = self.schema.project(columns)?;
        // Validate the projected schema resolves with this registry (no encode).
        resolve_and_validate_pipelines(&subset, registry)?;

        // Source column indices for the kept columns, in output order.
        let mut col_indices = Vec::with_capacity(columns.len());
        for &name in columns {
            let idx = self
                .schema
                .column_index(name)
                .ok_or_else(|| HeliumError::Schema {
                    column: name.into(),
                    reason: "column not in schema".into(),
                })?;
            col_indices.push(idx);
        }

        // Snapshot the leaf metadata we need (offsets/length/crc/stats), which
        // releases the borrow on `self` so we can read from `self.inner` below.
        let body_start = self.body_start;
        let plan: Vec<(u64, Vec<LogicalLocation>)> = self
            .stripes
            .iter()
            .map(|s| {
                let cols: Vec<LogicalLocation> = col_indices
                    .iter()
                    .map(|&ci| s.columns[ci].clone())
                    .collect();
                (s.row_count, cols)
            })
            .collect();

        // --- Write the self-contained header ---
        let header = file_header(0);
        dst.write_all(&header)?;
        let schema_json = subset.to_json()?;
        let compressed_schema = zstd::encode_all(&schema_json[..], SCHEMA_ZSTD_LEVEL)
            .map_err(|e| HeliumError::Format(format!("zstd compress schema: {e}")))?;
        let schema_len: u32 = compressed_schema
            .len()
            .try_into()
            .map_err(|_| HeliumError::Format("compressed schema exceeds u32 length".into()))?;
        dst.write_all(&schema_len.to_le_bytes())?;
        dst.write_all(&compressed_schema)?;
        let new_body_start = dst.stream_position()?;

        // --- Copy leaf bytes verbatim, recomputing offsets, reusing metadata ---
        let mut buf: Vec<u8> = Vec::new();
        let mut new_stripes = Vec::with_capacity(plan.len());
        for (row_count, src_cols) in &plan {
            let mut new_columns = Vec::with_capacity(src_cols.len());
            for loc in src_cols {
                let mut new_physical = Vec::with_capacity(loc.physical.len());
                for p in &loc.physical {
                    self.inner.seek(SeekFrom::Start(body_start + p.offset))?;
                    buf.resize(p.length as usize, 0);
                    self.inner.read_exact(&mut buf)?;
                    let actual = crc32c::crc32c(&buf);
                    if actual != p.crc32c {
                        return Err(HeliumError::Corrupted {
                            coder: "<slice>".into(),
                            reason: format!(
                                "source leaf CRC32C mismatch during slice: stored {:#x}, got {actual:#x}",
                                p.crc32c
                            ),
                        });
                    }
                    let new_offset = dst.stream_position()? - new_body_start;
                    dst.write_all(&buf)?;
                    new_physical.push(PhysicalLocation {
                        offset: new_offset,
                        length: p.length,
                        crc32c: p.crc32c,
                        min: p.min.clone(),
                        max: p.max.clone(),
                        null_count: p.null_count,
                        filter: p.filter.clone(),
                    });
                }
                new_columns.push(LogicalLocation {
                    physical: new_physical,
                });
            }
            new_stripes.push(StripeIndex {
                row_count: *row_count,
                columns: new_columns,
            });
        }

        // --- Write the footer (zstd JSON), trailer, end magic ---
        let footer = Footer {
            stripes: new_stripes,
        };
        let footer_json = serde_json::to_vec(&footer)?;
        let footer_bytes = zstd::encode_all(&footer_json[..], FOOTER_ZSTD_LEVEL)
            .map_err(|e| HeliumError::Format(format!("zstd compress footer: {e}")))?;
        let footer_len = footer_bytes.len() as u64;
        let footer_crc = crc32c::crc32c(&footer_bytes);
        dst.write_all(&footer_bytes)?;
        dst.write_all(&footer_len.to_le_bytes())?;
        dst.write_all(&footer_crc.to_le_bytes())?;
        dst.write_all(&header)?;
        dst.flush()?;
        Ok(dst)
    }

    /// Read all columns across all stripes into a `name → LogicalColumn` map.
    ///
    /// Equivalent to calling [`read_column`] for every column name. Returns an
    /// error if any column fails to decode. Dict columns in multi-stripe files
    /// will error here — use [`read_column_at_stripe`] in that case.
    ///
    /// [`read_column`]: HeliumReader::read_column
    /// [`read_column_at_stripe`]: HeliumReader::read_column_at_stripe
    pub fn read_all(&mut self) -> Result<HashMap<String, LogicalColumn>> {
        let names: Vec<String> = self.schema.columns.iter().map(|c| c.name.clone()).collect();
        let mut out = HashMap::with_capacity(names.len());
        for name in names {
            let data = self.read_column(&name)?;
            out.insert(name, data);
        }
        Ok(out)
    }

    /// Read one stripe as an Arrow `RecordBatch`.
    ///
    /// Convenience for callers that want to pipe Helium data through
    /// Arrow-native consumers (DataFusion, polars, Arrow Flight, etc.).
    ///
    /// Uses [`HeliumReader::read_column_at_stripe`] per column under the hood,
    /// then converts each [`LogicalColumn`] to an Arrow array using
    /// [`crate::arrow::to_arrow_array`], and wraps the result in a
    /// `RecordBatch` with the schema returned by
    /// [`crate::arrow::schema_to_arrow`].
    ///
    /// # Dict columns
    ///
    /// Dict columns are safe to use here because this method reads one stripe
    /// at a time — each stripe's dictionary is self-contained.
    ///
    /// # Errors
    ///
    /// Returns an error if `stripe_idx` is out of range, if any column fails
    /// to decode, or if any [`LogicalColumn`] → Arrow conversion fails.
    #[cfg(feature = "arrow")]
    pub fn read_record_batch(
        &mut self,
        stripe_idx: usize,
    ) -> Result<arrow::record_batch::RecordBatch> {
        use crate::arrow::{schema_to_arrow, to_arrow_array};
        use std::sync::Arc;

        if stripe_idx >= self.stripes.len() {
            return Err(HeliumError::Format(format!(
                "stripe index {stripe_idx} out of range (have {})",
                self.stripes.len()
            )));
        }

        let arrow_schema = Arc::new(schema_to_arrow(&self.schema));
        let col_count = self.schema.columns.len();
        let mut arrays = Vec::with_capacity(col_count);

        for i in 0..col_count {
            let spec = &self.schema.columns[i];
            let name = spec.name.clone();
            let lt = spec.logical_type.clone();
            let col = self.read_column_piece(i, stripe_idx)?;
            let arr = to_arrow_array(&col, &lt).map_err(|e| HeliumError::Schema {
                column: name.clone(),
                reason: format!("Arrow conversion failed: {e}"),
            })?;
            arrays.push(arr);
        }

        arrow::record_batch::RecordBatch::try_new(arrow_schema, arrays)
            .map_err(|e| HeliumError::Format(format!("RecordBatch::try_new: {e}")))
    }

    fn read_column_piece(&mut self, col_idx: usize, stripe_idx: usize) -> Result<LogicalColumn> {
        let spec = &self.schema.columns[col_idx];
        let pipes = &self.pipelines[col_idx];
        let stripe = &self.stripes[stripe_idx];
        let locs = &stripe.columns[col_idx].physical;
        let row_count = stripe.row_count as usize;
        let column_name = spec.name.clone();

        // Leaf-path names (dotted) for richer error context on CRC / decode
        // failures inside a deeply-nested type (§5.8 requirement: failures
        // must pinpoint the failing leaf).
        let physical_field_names = spec.logical_type.physical_fields();

        let mut physical_parts = Vec::with_capacity(locs.len());
        for ((loc, pipe), field) in locs
            .iter()
            .zip(pipes.iter())
            .zip(physical_field_names.iter())
        {
            self.inner
                .seek(SeekFrom::Start(self.body_start + loc.offset))?;
            let mut bytes = vec![0u8; loc.length as usize];
            self.inner
                .read_exact(&mut bytes)
                .map_err(|e| HeliumError::Schema {
                    column: column_name.clone(),
                    reason: format!(
                        "reading physical column bytes for leaf '{}': {e}",
                        field.role
                    ),
                })?;
            // v5/v6 always carry a per-leaf CRC32C.
            let actual = crc32c::crc32c(&bytes);
            if actual != loc.crc32c {
                return Err(HeliumError::Corrupted {
                    coder: column_name.clone(),
                    reason: format!(
                        "stripe {stripe_idx} leaf '{}' CRC32C mismatch: \
                         stored {:#x}, computed {actual:#x}",
                        field.role, loc.crc32c
                    ),
                });
            }
            let decoded =
                pipe.decode(ColumnData::Bytes(bytes))
                    .map_err(|e| HeliumError::Schema {
                        column: column_name.clone(),
                        reason: format!("leaf '{}': {e}", field.role),
                    })?;
            physical_parts.push(decoded);
        }

        LogicalColumn::compose(physical_parts, &spec.logical_type, row_count).map_err(|e| match e {
            HeliumError::Schema { reason, .. } => HeliumError::Schema {
                column: column_name.clone(),
                reason,
            },
            other => other,
        })
    }
}

// ---------------------------------------------------------------------------
// Cross-stripe concatenation
// ---------------------------------------------------------------------------

fn concat_logical_columns(pieces: Vec<LogicalColumn>, lt: &LogicalType) -> Result<LogicalColumn> {
    use LogicalColumn as LC;
    match lt {
        LogicalType::Primitive { .. } => {
            let mut out: Option<ColumnData> = None;
            for p in pieces {
                let LC::Primitive(d) = p else {
                    return Err(schema_err("concat: expected Primitive pieces"));
                };
                out = Some(match out {
                    None => d,
                    Some(acc) => concat_column_data(acc, d)?,
                });
            }
            out.map(LC::Primitive)
                .ok_or_else(|| schema_err("concat: no pieces provided for Primitive"))
        }
        LogicalType::Utf8 => {
            let mut all: Vec<String> = Vec::new();
            for p in pieces {
                let LC::Utf8(v) = p else {
                    return Err(schema_err("concat: expected Utf8 pieces"));
                };
                all.extend(v);
            }
            Ok(LC::Utf8(all))
        }
        LogicalType::Binary => {
            let mut all: Vec<Vec<u8>> = Vec::new();
            for p in pieces {
                let LC::Binary(v) = p else {
                    return Err(schema_err("concat: expected Binary pieces"));
                };
                all.extend(v);
            }
            Ok(LC::Binary(all))
        }
        LogicalType::ArrayOf { .. } => {
            let mut acc_offsets: Vec<u32> = vec![0];
            let mut acc_values: Option<ColumnData> = None;
            for p in pieces {
                let LC::ArrayOf { offsets, values } = p else {
                    return Err(schema_err("concat: expected ArrayOf pieces"));
                };
                // acc_offsets always has at least one element (initialized with vec![0]).
                let Some(&base) = acc_offsets.last() else {
                    return Err(schema_err("concat: acc_offsets unexpectedly empty"));
                };
                for &o in &offsets[1..] {
                    acc_offsets.push(base + o);
                }
                acc_values = Some(match acc_values {
                    None => values,
                    Some(acc) => concat_column_data(acc, values)?,
                });
            }
            Ok(LC::ArrayOf {
                offsets: acc_offsets,
                values: acc_values
                    .ok_or_else(|| schema_err("concat: no pieces provided for ArrayOf"))?,
            })
        }
        LogicalType::ArrayOfUtf8 => {
            let mut acc_offsets: Vec<u32> = vec![0];
            let mut acc_strings: Vec<String> = Vec::new();
            for p in pieces {
                let LC::ArrayOfUtf8 { offsets, strings } = p else {
                    return Err(schema_err("concat: expected ArrayOfUtf8 pieces"));
                };
                // acc_offsets always has at least one element (initialized with vec![0]).
                let Some(&base) = acc_offsets.last() else {
                    return Err(schema_err("concat: acc_offsets unexpectedly empty"));
                };
                for &o in &offsets[1..] {
                    acc_offsets.push(base + o);
                }
                acc_strings.extend(strings);
            }
            Ok(LC::ArrayOfUtf8 {
                offsets: acc_offsets,
                strings: acc_strings,
            })
        }
        LogicalType::NullablePrim { .. } => {
            let mut acc_present: Vec<bool> = Vec::new();
            let mut acc_values: Option<ColumnData> = None;
            for p in pieces {
                let LC::NullablePrim { present, values } = p else {
                    return Err(schema_err("concat: expected NullablePrim pieces"));
                };
                acc_present.extend(present);
                acc_values = Some(match acc_values {
                    None => values,
                    Some(acc) => concat_column_data(acc, values)?,
                });
            }
            Ok(LC::NullablePrim {
                present: acc_present,
                values: acc_values
                    .ok_or_else(|| schema_err("concat: no pieces provided for NullablePrim"))?,
            })
        }
        LogicalType::NullableUtf8 => {
            let mut acc_present: Vec<bool> = Vec::new();
            let mut acc_strings: Vec<String> = Vec::new();
            for p in pieces {
                let LC::NullableUtf8 { present, strings } = p else {
                    return Err(schema_err("concat: expected NullableUtf8 pieces"));
                };
                acc_present.extend(present);
                acc_strings.extend(strings);
            }
            Ok(LC::NullableUtf8 {
                present: acc_present,
                strings: acc_strings,
            })
        }
        LogicalType::NullableBinary => {
            let mut acc_present: Vec<bool> = Vec::new();
            let mut acc_blobs: Vec<Vec<u8>> = Vec::new();
            for p in pieces {
                let LC::NullableBinary { present, blobs } = p else {
                    return Err(schema_err("concat: expected NullableBinary pieces"));
                };
                acc_present.extend(present);
                acc_blobs.extend(blobs);
            }
            Ok(LC::NullableBinary {
                present: acc_present,
                blobs: acc_blobs,
            })
        }
        LogicalType::Dictionary { .. } => Err(schema_err(
            "dict columns cannot be concatenated — use read_column_at_stripe",
        )),
        LogicalType::List { inner } => {
            // Concat outer offsets (rebasing each stripe), then concat inner values.
            let mut acc_offsets: Vec<u32> = vec![0];
            let mut value_pieces: Vec<LogicalColumn> = Vec::new();
            for piece in pieces {
                let LC::List { offsets, values } = piece else {
                    return Err(schema_err("concat: expected List pieces"));
                };
                // acc_offsets always has at least one element (initialized with vec![0]).
                let Some(&base) = acc_offsets.last() else {
                    return Err(schema_err("concat: acc_offsets unexpectedly empty"));
                };
                for &o in &offsets[1..] {
                    acc_offsets.push(base + o);
                }
                value_pieces.push(*values);
            }
            let concatenated_values = concat_logical_columns(value_pieces, inner)?;
            Ok(LC::List {
                offsets: acc_offsets,
                values: Box::new(concatenated_values),
            })
        }
        LogicalType::Union {
            variants: spec_variants,
        } => {
            // Concat tags; concat each variant's compacted data independently.
            let n = spec_variants.len();
            let mut acc_tags: Vec<u8> = Vec::new();
            let mut var_pieces: Vec<Vec<LogicalColumn>> = (0..n).map(|_| Vec::new()).collect();
            for piece in pieces {
                let LC::Union { tags, variants } = piece else {
                    return Err(schema_err("concat: expected Union pieces"));
                };
                acc_tags.extend(tags);
                for (i, (_, v_col)) in variants.into_iter().enumerate() {
                    var_pieces[i].push(v_col);
                }
            }
            let mut result_variants = Vec::with_capacity(n);
            for ((v_name, v_lt), pieces) in spec_variants.iter().zip(var_pieces) {
                let concat = concat_logical_columns(pieces, v_lt)?;
                result_variants.push((v_name.clone(), concat));
            }
            Ok(LC::Union {
                tags: acc_tags,
                variants: result_variants,
            })
        }
        LogicalType::Nullable { inner } => {
            // Concat present bitmaps; concat compacted inner values independently.
            let mut acc_present: Vec<bool> = Vec::new();
            let mut value_pieces: Vec<LogicalColumn> = Vec::new();
            for piece in pieces {
                let LC::Nullable { present, value } = piece else {
                    return Err(schema_err("concat: expected Nullable pieces"));
                };
                acc_present.extend(present);
                value_pieces.push(*value);
            }
            let concat_value = concat_logical_columns(value_pieces, inner)?;
            Ok(LC::Nullable {
                present: acc_present,
                value: Box::new(concat_value),
            })
        }
        LogicalType::Map { key, value } => {
            // Concat offsets (rebasing), then concat keys and values independently.
            let mut acc_offsets: Vec<u32> = vec![0];
            let mut key_pieces: Vec<LogicalColumn> = Vec::new();
            let mut value_pieces: Vec<LogicalColumn> = Vec::new();
            for piece in pieces {
                let LC::Map {
                    offsets,
                    keys,
                    values,
                } = piece
                else {
                    return Err(schema_err("concat: expected Map pieces"));
                };
                // acc_offsets always has at least one element (initialized with vec![0]).
                let Some(&base) = acc_offsets.last() else {
                    return Err(schema_err("concat: acc_offsets unexpectedly empty"));
                };
                for &o in &offsets[1..] {
                    acc_offsets.push(base + o);
                }
                key_pieces.push(*keys);
                value_pieces.push(*values);
            }
            let concat_keys = concat_logical_columns(key_pieces, key)?;
            let concat_values = concat_logical_columns(value_pieces, value)?;
            Ok(LC::Map {
                offsets: acc_offsets,
                keys: Box::new(concat_keys),
                values: Box::new(concat_values),
            })
        }
        LogicalType::Struct {
            fields: spec_fields,
        } => {
            // Collect pieces per field, then concatenate each field independently.
            let n_fields = spec_fields.len();
            let mut per_field: Vec<Vec<LogicalColumn>> =
                (0..n_fields).map(|_| Vec::new()).collect();
            for piece in pieces {
                let LC::Struct {
                    fields: piece_fields,
                } = piece
                else {
                    return Err(schema_err("concat: expected Struct pieces"));
                };
                if piece_fields.len() != n_fields {
                    return Err(schema_err(
                        "concat: Struct field count mismatch across stripes",
                    ));
                }
                for (i, (_, field_col)) in piece_fields.into_iter().enumerate() {
                    per_field[i].push(field_col);
                }
            }
            let mut result_fields = Vec::with_capacity(n_fields);
            for (spec_field, field_pieces) in spec_fields.iter().zip(per_field) {
                let concatenated = concat_logical_columns(field_pieces, &spec_field.logical_type)?;
                result_fields.push((spec_field.name.clone(), concatenated));
            }
            Ok(LC::Struct {
                fields: result_fields,
            })
        }

        // Semantic types: simple Vec concatenation.
        LogicalType::Decimal128 { .. } => {
            let mut all: Vec<i128> = Vec::new();
            for p in pieces {
                let LC::Decimal128 { values } = p else {
                    return Err(schema_err("concat: expected Decimal128 pieces"));
                };
                all.extend(values);
            }
            Ok(LC::Decimal128 { values: all })
        }
        LogicalType::Date {
            unit: crate::core::schema::DateUnit::Days,
        } => {
            let mut all: Vec<i32> = Vec::new();
            for p in pieces {
                let LC::Date32 { values } = p else {
                    return Err(schema_err("concat: expected Date32 pieces"));
                };
                all.extend(values);
            }
            Ok(LC::Date32 { values: all })
        }
        LogicalType::Date {
            unit: crate::core::schema::DateUnit::Millis,
        } => {
            let mut all: Vec<i64> = Vec::new();
            for p in pieces {
                let LC::Date64 { values } = p else {
                    return Err(schema_err("concat: expected Date64 pieces"));
                };
                all.extend(values);
            }
            Ok(LC::Date64 { values: all })
        }
        LogicalType::Datetime { .. } => {
            let mut all: Vec<i64> = Vec::new();
            for p in pieces {
                let LC::Datetime { values } = p else {
                    return Err(schema_err("concat: expected Datetime pieces"));
                };
                all.extend(values);
            }
            Ok(LC::Datetime { values: all })
        }
    }
}

fn schema_err(msg: &str) -> HeliumError {
    HeliumError::Schema {
        column: "<concat>".into(),
        reason: msg.into(),
    }
}

fn concat_column_data(a: ColumnData, b: ColumnData) -> Result<ColumnData> {
    if a.data_type() != b.data_type() {
        return Err(schema_err(&format!(
            "concat: type mismatch {:?} vs {:?}",
            a.data_type(),
            b.data_type()
        )));
    }
    Ok(match (a, b) {
        (ColumnData::I8(mut x), ColumnData::I8(y)) => {
            x.extend(y);
            ColumnData::I8(x)
        }
        (ColumnData::I16(mut x), ColumnData::I16(y)) => {
            x.extend(y);
            ColumnData::I16(x)
        }
        (ColumnData::I32(mut x), ColumnData::I32(y)) => {
            x.extend(y);
            ColumnData::I32(x)
        }
        (ColumnData::I64(mut x), ColumnData::I64(y)) => {
            x.extend(y);
            ColumnData::I64(x)
        }
        (ColumnData::U8(mut x), ColumnData::U8(y)) => {
            x.extend(y);
            ColumnData::U8(x)
        }
        (ColumnData::U16(mut x), ColumnData::U16(y)) => {
            x.extend(y);
            ColumnData::U16(x)
        }
        (ColumnData::U32(mut x), ColumnData::U32(y)) => {
            x.extend(y);
            ColumnData::U32(x)
        }
        (ColumnData::U64(mut x), ColumnData::U64(y)) => {
            x.extend(y);
            ColumnData::U64(x)
        }
        (ColumnData::F32(mut x), ColumnData::F32(y)) => {
            x.extend(y);
            ColumnData::F32(x)
        }
        (ColumnData::F64(mut x), ColumnData::F64(y)) => {
            x.extend(y);
            ColumnData::F64(x)
        }
        (ColumnData::Bytes(mut x), ColumnData::Bytes(y)) => {
            x.extend(y);
            ColumnData::Bytes(x)
        }
        _ => unreachable!("types compared above"),
    })
}
