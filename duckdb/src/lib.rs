//! Helium DuckDB loadable extension.
//!
//! Registers the `read_he(path VARCHAR)` table function so DuckDB can query
//! `.he` files directly:
//!
//! ```sql
//! LOAD 'helium_duckdb';
//! SELECT * FROM read_he('path/to/file.he') LIMIT 5;
//! SELECT count(*) FROM read_he('path/to/file.he');
//! SELECT col_a, col_b FROM read_he('path/to/file.he') WHERE col_a > 100;
//! -- catalog-mode files: pass the catalog directory by named parameter
//! SELECT * FROM read_he('path/to/file.he', catalog := '/path/to/catalog');
//! ```
//!
//! ## What is pushed down
//! - **Projection pushdown** — the extension advertises projection pushdown to
//!   DuckDB, reads the projected logical-column indices in the `init` phase, and
//!   decodes only those columns per stripe via the reader's column pruning.
//!   Selecting 1 of N columns decodes 1, not N.
//! - **One reader held open** across all stripes (in the scan's init data) — the
//!   file is opened once per scan instead of once per stripe.
//! - **Stripe min/max pruning** is available as a tested helper
//!   ([`stripe_matches_range`]) but **cannot be auto-driven** from inside a
//!   loadable extension: the DuckDB loadable C-API (v1.2.0) exposes projection
//!   pushdown but **no** filter-pushdown hook, so DuckDB never hands the
//!   extension the `WHERE` bounds. The machinery is wired and unit-tested so it
//!   is ready the moment the C-API gains filter access; until then DuckDB
//!   applies `WHERE` after the scan. See `docs/ROADMAP.md`.
//!
//! ## Type coverage
//! - Scalar: all primitives, `Utf8`, `Binary`, `Nullable`,
//!   `Dictionary`, and the semantic types (`Decimal128`, `Date`, `Datetime`).
//! - Nested: `Struct`, `List`, and `Map` map onto DuckDB STRUCT / LIST / MAP
//!   vectors. `Union` is rejected at bind time with a clear message.
//! - Catalog-mode files are read by passing `catalog := '<dir>'`.

use std::cell::UnsafeCell;
use std::fs::File;

use duckdb::core::{DataChunkHandle, LogicalTypeHandle, LogicalTypeId};
use duckdb::vtab::{BindInfo, InitInfo, TableFunctionInfo, VTab};
use duckdb::{Connection, Result as DuckResult, duckdb_entrypoint_c_api};

use helium::catalog::Catalog;
use helium::{CoderRegistry, DateUnit, HeliumReader, LogicalColumn, LogicalType, TimeUnit};

mod nested;
mod prune;
mod writer;

pub use prune::stripe_matches_range;
use writer::write_column_window;

// ---------------------------------------------------------------------------
// Bind phase — open file, read schema, emit DuckDB column types
// ---------------------------------------------------------------------------

/// Data produced during the bind phase and shared (read-only) with the scan.
pub struct HeBindData {
    /// Path to the `.he` file.
    path: String,
    /// Optional catalog directory for catalog-mode files. `None` for self-contained files.
    catalog_dir: Option<String>,
    /// Column names, in schema order.
    col_names: Vec<String>,
    /// Total number of stripes in the file.
    stripe_count: usize,
}

// SAFETY: HeBindData fields are all Send + Sync (String / Vec<String> /
// primitive); it is constructed once and then only read during the scan.
unsafe impl Send for HeBindData {}
unsafe impl Sync for HeBindData {}

// ---------------------------------------------------------------------------
// Init phase — scan cursor + held-open reader
// ---------------------------------------------------------------------------

/// Per-scan cursor held inside an `UnsafeCell` so we can mutate state through
/// the shared `&HeInitData` reference that DuckDB's vtab API gives us.
///
/// SAFETY: DuckDB calls `func` sequentially for a given scan with `max_threads`
/// left at the default of 1 — no concurrent calls on the same cursor. The
/// `UnsafeCell` is the only way to achieve interior mutability here without
/// `Mutex` overhead.
pub struct HeInitData {
    inner: UnsafeCell<HeInitDataInner>,
}

struct HeInitDataInner {
    /// The reader, opened once and held open for the whole scan (instead of
    /// re-opening the file per stripe). `None` only if open failed at first use.
    reader: Option<HeliumReader<File>>,
    /// Logical column names to decode, in **output** order. This is the
    /// projection: for `SELECT a, c` over a file with columns `[a, b, c]`,
    /// `projected` is `["a", "c"]` and output column 0 ← a, 1 ← c.
    projected: Vec<String>,
    /// Next stripe index to load (0-based).
    next_stripe: usize,
    /// Columns decoded for the current stripe — one entry per projected column,
    /// in output order.
    current_columns: Vec<LogicalColumn>,
    /// Row offset within the current stripe (i.e. next unread row).
    row_in_stripe: usize,
    /// Total rows in the current stripe.
    stripe_rows: usize,
    /// True once all stripes have been exhausted.
    done: bool,
}

// SAFETY: DuckDB accesses the init data from a single scan thread at a time
// (max_threads defaults to 1). The held reader is `Send` (File + owned state).
unsafe impl Send for HeInitData {}
unsafe impl Sync for HeInitData {}

// ---------------------------------------------------------------------------
// VTab implementation
// ---------------------------------------------------------------------------

/// Table function implementation for `read_he(path VARCHAR [, catalog := VARCHAR])`.
pub struct HeVTab;

/// Build a reader for `path`, resolving catalog-mode files through the
/// catalog directory when one was supplied.
fn open_reader(
    path: &str,
    catalog_dir: Option<&str>,
    registry: &CoderRegistry,
) -> Result<HeliumReader<File>, Box<dyn std::error::Error>> {
    let file = File::open(path).map_err(|e| format!("read_he: cannot open '{path}': {e}"))?;
    match catalog_dir {
        Some(dir) => {
            let catalog = Catalog::open(dir)
                .map_err(|e| format!("read_he: cannot open catalog '{dir}': {e}"))?;
            let resolver = catalog.resolver();
            HeliumReader::new_with_resolver(file, registry, resolver)
                .map_err(|e| format!("read_he: cannot read '{path}' via catalog '{dir}': {e}").into())
        }
        None => HeliumReader::new(file, registry)
            .map_err(|e| format!("read_he: cannot read '{path}': {e}").into()),
    }
}

impl VTab for HeVTab {
    type InitData = HeInitData;
    type BindData = HeBindData;

    fn supports_pushdown() -> bool {
        // Projection pushdown only. The DuckDB loadable C-API does not expose a
        // filter-pushdown hook, so predicate pushdown cannot be advertised here.
        true
    }

    fn bind(bind: &BindInfo) -> DuckResult<Self::BindData, Box<dyn std::error::Error>> {
        let path = bind.get_parameter(0).to_string();
        let catalog_dir = bind.get_named_parameter("catalog").map(|v| v.to_string());

        let registry = CoderRegistry::default();
        let reader = open_reader(&path, catalog_dir.as_deref(), &registry)?;

        let schema = reader.schema().clone();
        let stripe_count = reader.stripe_count();

        let mut col_names = Vec::with_capacity(schema.columns.len());
        for col_spec in &schema.columns {
            let ltype = logical_type_to_duckdb(&col_spec.logical_type)?;
            bind.add_result_column(col_spec.name.as_str(), ltype);
            col_names.push(col_spec.name.clone());
        }

        Ok(HeBindData {
            path,
            catalog_dir,
            col_names,
            stripe_count,
        })
    }

    fn init(init: &InitInfo) -> DuckResult<Self::InitData, Box<dyn std::error::Error>> {
        // SAFETY: `get_bind_data` returns the bind data we set during bind; it
        // outlives init/func and is read-only.
        let bind_data: &HeBindData = unsafe {
            init.get_bind_data::<HeBindData>()
                .as_ref()
                .ok_or("read_he: missing bind data in init")?
        };

        // Projection pushdown: DuckDB hands us the indices (into the bind's
        // result columns, i.e. schema order) of just the columns it needs, in
        // the order it wants them back.
        let indices = init.get_column_indices();
        let projected: Vec<String> = indices
            .iter()
            .map(|&i| {
                bind_data
                    .col_names
                    .get(i as usize)
                    .cloned()
                    .ok_or_else(|| format!("read_he: projected column index {i} out of range"))
            })
            .collect::<Result<_, _>>()?;

        // Open the reader once and hold it open for the entire scan.
        let registry = CoderRegistry::default();
        let reader = open_reader(
            &bind_data.path,
            bind_data.catalog_dir.as_deref(),
            &registry,
        )?;

        Ok(HeInitData {
            inner: UnsafeCell::new(HeInitDataInner {
                reader: Some(reader),
                projected,
                next_stripe: 0,
                current_columns: Vec::new(),
                row_in_stripe: 0,
                stripe_rows: 0,
                done: false,
            }),
        })
    }

    fn func(
        func: &TableFunctionInfo<Self>,
        output: &mut DataChunkHandle,
    ) -> DuckResult<(), Box<dyn std::error::Error>> {
        let bind_data = func.get_bind_data();
        // SAFETY: DuckDB guarantees sequential calls per scan (max_threads = 1);
        // no aliasing of the cursor.
        let state = unsafe { &mut *func.get_init_data().inner.get() };

        if state.done {
            output.set_len(0);
            return Ok(());
        }

        // Load the next stripe if the current one is exhausted.
        if state.row_in_stripe >= state.stripe_rows {
            if state.next_stripe >= bind_data.stripe_count {
                state.done = true;
                output.set_len(0);
                return Ok(());
            }

            let stripe_idx = state.next_stripe;
            state.next_stripe += 1;

            let reader = state
                .reader
                .as_mut()
                .ok_or("read_he: reader not initialized")?;

            let mut loaded = Vec::with_capacity(state.projected.len());
            let mut stripe_rows = 0usize;
            for name in &state.projected {
                let column = reader
                    .read_column_at_stripe(name.as_str(), stripe_idx)
                    .map_err(|e| format!("read_he: stripe {stripe_idx} col '{name}': {e}"))?;
                stripe_rows = column.row_count();
                loaded.push(column);
            }

            // A projection of zero columns (e.g. `SELECT count(*)`) still needs
            // the stripe's logical row count so DuckDB receives the right cardinality.
            if state.projected.is_empty() {
                stripe_rows = reader
                    .stripe_row_count(stripe_idx)
                    .ok_or("read_he: stripe row count unavailable")?
                    as usize;
            }

            state.current_columns = loaded;
            state.row_in_stripe = 0;
            state.stripe_rows = stripe_rows;

            if stripe_rows == 0 {
                // Empty stripe — advance to the next one.
                return Self::func(func, output);
            }
        }

        // Emit up to DUCKDB_VECTOR_SIZE rows.
        let chunk_size = (state.stripe_rows - state.row_in_stripe).min(2048);
        let row_start = state.row_in_stripe;

        for (out_idx, column) in state.current_columns.iter().enumerate() {
            write_column_window(output, out_idx, column, row_start, chunk_size)?;
        }

        output.set_len(chunk_size);
        state.row_in_stripe += chunk_size;

        Ok(())
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![LogicalTypeHandle::from(LogicalTypeId::Varchar)])
    }

    fn named_parameters() -> Option<Vec<(String, LogicalTypeHandle)>> {
        Some(vec![(
            "catalog".to_string(),
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
        )])
    }
}

// ---------------------------------------------------------------------------
// Type mapping: Helium LogicalType → DuckDB LogicalTypeHandle
// ---------------------------------------------------------------------------

pub(crate) fn logical_type_to_duckdb(
    lt: &LogicalType,
) -> Result<LogicalTypeHandle, Box<dyn std::error::Error>> {
    use LogicalType::*;
    match lt {
        Primitive { data_type } => Ok(primitive_dt_to_duckdb(*data_type)),

        Utf8 => Ok(LogicalTypeHandle::from(LogicalTypeId::Varchar)),
        Binary => Ok(LogicalTypeHandle::from(LogicalTypeId::Blob)),

        // recursive nullable wrapper — DuckDB columns are always nullable, unwrap the inner type
        Nullable { inner } => logical_type_to_duckdb(inner),

        // Dictionary maps to its inner value type (materialized on read).
        Dictionary { inner } => logical_type_to_duckdb(inner),

        // Nested types — mapped onto DuckDB STRUCT / LIST / MAP.
        Struct { fields } => {
            // Build owned (name, handle) pairs, then borrow for the call.
            // `struct_type` copies the names into DuckDB-owned CStrings during
            // the call, so borrowing `names`/`handles` for its duration is sound
            // and leak-free.
            let mut names: Vec<String> = Vec::with_capacity(fields.len());
            let mut handles: Vec<LogicalTypeHandle> = Vec::with_capacity(fields.len());
            for f in fields {
                names.push(f.name.clone());
                handles.push(logical_type_to_duckdb(&f.logical_type)?);
            }
            let refs: Vec<(&str, LogicalTypeHandle)> = names
                .iter()
                .zip(handles)
                .map(|(n, h)| (n.as_str(), h))
                .collect();
            Ok(LogicalTypeHandle::struct_type(&refs))
        }
        List { inner } => {
            let child = logical_type_to_duckdb(inner)?;
            Ok(LogicalTypeHandle::list(&child))
        }
        Map { key, value } => {
            let k = logical_type_to_duckdb(key)?;
            let v = logical_type_to_duckdb(value)?;
            Ok(LogicalTypeHandle::map(&k, &v))
        }
        Union { .. } => Err("read_he: Union columns not yet supported".into()),

        // Semantic types
        Decimal128 { precision, scale } => Ok(LogicalTypeHandle::decimal(*precision, *scale)),

        Date { unit: DateUnit::Days } => Ok(LogicalTypeHandle::from(LogicalTypeId::Date)),
        Date { unit: DateUnit::Millis } => {
            // DuckDB DATE is day-resolution; represent ms-date as BIGINT
            Ok(LogicalTypeHandle::from(LogicalTypeId::Bigint))
        }

        Datetime { unit, timezone } => match (unit, timezone.as_deref()) {
            (_, Some(_)) => Ok(LogicalTypeHandle::from(LogicalTypeId::TimestampTZ)),
            (TimeUnit::Seconds, None) => Ok(LogicalTypeHandle::from(LogicalTypeId::TimestampS)),
            (TimeUnit::Millis, None) => Ok(LogicalTypeHandle::from(LogicalTypeId::TimestampMs)),
            (TimeUnit::Micros, None) => Ok(LogicalTypeHandle::from(LogicalTypeId::Timestamp)),
            (TimeUnit::Nanos, None) => Ok(LogicalTypeHandle::from(LogicalTypeId::TimestampNs)),
        },
    }
}

/// Map a Helium scalar `DataType` to a DuckDB `LogicalTypeHandle`.
pub(crate) fn primitive_dt_to_duckdb(dt: helium::DataType) -> LogicalTypeHandle {
    use helium::DataType::*;
    let id = match dt {
        I8 => LogicalTypeId::Tinyint,
        I16 => LogicalTypeId::Smallint,
        I32 => LogicalTypeId::Integer,
        I64 => LogicalTypeId::Bigint,
        U8 => LogicalTypeId::UTinyint,
        U16 => LogicalTypeId::USmallint,
        U32 => LogicalTypeId::UInteger,
        U64 => LogicalTypeId::UBigint,
        F32 => LogicalTypeId::Float,
        F64 => LogicalTypeId::Double,
        Bytes => LogicalTypeId::Blob,
    };
    LogicalTypeHandle::from(id)
}

// ---------------------------------------------------------------------------
// Trait helper for diagnostic messages
// ---------------------------------------------------------------------------

pub(crate) trait VariantName {
    fn variant_name(&self) -> &'static str;
}

impl VariantName for LogicalColumn {
    fn variant_name(&self) -> &'static str {
        match self {
            LogicalColumn::Primitive(_) => "Primitive",
            LogicalColumn::Utf8(_) => "Utf8",
            LogicalColumn::Binary(_) => "Binary",
            LogicalColumn::Dictionary { .. } => "Dictionary",
            LogicalColumn::Struct { .. } => "Struct",
            LogicalColumn::List { .. } => "List",
            LogicalColumn::Map { .. } => "Map",
            LogicalColumn::Nullable { .. } => "Nullable",
            LogicalColumn::Union { .. } => "Union",
            LogicalColumn::Decimal128 { .. } => "Decimal128",
            LogicalColumn::Date32 { .. } => "Date32",
            LogicalColumn::Date64 { .. } => "Date64",
            LogicalColumn::Datetime { .. } => "Datetime",
        }
    }
}

/// Shared boxed-error alias used across the writer / nested / prune modules.
pub(crate) type BoxErr = Box<dyn std::error::Error>;

// ---------------------------------------------------------------------------
// Extension entrypoint
// ---------------------------------------------------------------------------

/// DuckDB extension entrypoint — called once when `LOAD 'helium_duckdb'` runs.
#[duckdb_entrypoint_c_api(ext_name = "helium_duckdb", min_duckdb_version = "v1.2.0")]
pub fn extension_entrypoint(con: Connection) -> DuckResult<(), Box<dyn std::error::Error>> {
    con.register_table_function::<HeVTab>("read_he")
        .map_err(|e| format!("helium_duckdb: failed to register read_he: {e}"))?;
    Ok(())
}
