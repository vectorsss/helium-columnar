//! Per-stripe column statistics and containment filters stored in the `.he` footer.
//!
//! This module owns three related concerns:
//!
//! 1. **`MinMaxValue`** — a typed tagged-union of the statistical values that
//!    can appear as min or max for a physical column.
//! 2. **`PhysicalColumnStats`** / **`ContainmentFilter`** — the public types
//!    returned by [`crate::HeliumReader`] for predicate pushdown.
//! 3. **Computation helpers** — the private functions called by
//!    [`crate::HeliumWriter`] at write time to build these structures from
//!    [`crate::LogicalColumn`] data before it is encoded.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use super::coder::ColumnData;
use super::schema::{DateUnit, LogicalColumn, LogicalType, TimeUnit};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum byte length for min/max string/binary statistics stored in the
/// footer. Values longer than this are silently truncated.
pub(super) const STATS_TRUNCATE_BYTES: usize = 256;

/// Maximum number of distinct values tracked before switching to a Bloom filter.
///
/// When the distinct-value set for a physical column exceeds this threshold
/// during a stripe write, the writer abandons the exact set and builds a Bloom
/// filter instead.
pub(super) const MAX_DISTINCT_SET_SIZE: usize = 256;

/// Maximum byte size for a Bloom filter payload (bits / 8).
///
/// Caps the per-filter memory and footer-size cost at 64 KB per stripe per
/// physical column.  Stripes with more rows will have a higher FPP but
/// equality pruning is still useful for very selective predicates.
pub(super) const MAX_BLOOM_BYTES: usize = 65_536; // 64 KB = 524 288 bits

// ---------------------------------------------------------------------------
// MinMaxValue
// ---------------------------------------------------------------------------

/// A typed min or max value stored per physical column in the footer.
///
/// The `kind` tag is stable JSON wire format (frozen once any `.he` file is
/// written); do not rename or repurpose variants. `content` carries the
/// actual value.
///
/// # Wire-format stability
///
/// All existing variant names (`i8`..`binary`) are frozen.  The three semantic
/// variants added in the recursive-schema era (`decimal128`, `date`, `datetime`) are
/// likewise frozen — do not rename them.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum MinMaxValue {
    /// Signed 8-bit integer.
    I8(i8),
    /// Signed 16-bit integer.
    I16(i16),
    /// Signed 32-bit integer.
    I32(i32),
    /// Signed 64-bit integer.
    I64(i64),
    /// Unsigned 8-bit integer.
    U8(u8),
    /// Unsigned 16-bit integer.
    U16(u16),
    /// Unsigned 32-bit integer.
    U32(u32),
    /// Unsigned 64-bit integer.
    U64(u64),
    /// 32-bit float (NaN excluded — if all values were NaN the field is absent).
    F32(f32),
    /// 64-bit float (NaN excluded — if all values were NaN the field is absent).
    F64(f64),
    /// UTF-8 string. Truncated to 256 bytes if longer (respects UTF-8 boundaries).
    Utf8(String),
    /// Binary blob, base64-encoded for JSON safety. Truncated to
    /// 256 bytes **before** encoding if longer.
    Binary(String),
    /// Decimal128 value stored as (high, low) i64 halves with precision/scale
    /// metadata for unambiguous reconstruction.
    ///
    /// The i128 value = `(high as i128) << 64 | (low as u64 as i128)`.
    Decimal128 {
        /// High 64 bits of the i128 value.
        high: i64,
        /// Low 64 bits of the i128 value (interpret as unsigned to reconstruct).
        low: i64,
        /// Decimal precision (number of significant digits).
        precision: u8,
        /// Decimal scale (digits after the decimal point).
        scale: u8,
    },
    /// Calendar date value.
    ///
    /// - `Days` unit: `value` is an `i32` cast to `i64` (days since 1970-01-01).
    /// - `Millis` unit: `value` is an `i64` (milliseconds since 1970-01-01).
    Date {
        /// Encoded date value (unit-dependent).
        value: i64,
        /// Calendar unit — `Days` or `Millis`.
        unit: DateUnit,
    },
    /// Timestamp value.
    ///
    /// `value` is an `i64` whose interpretation depends on `unit`.
    Datetime {
        /// Encoded timestamp value (unit-dependent).
        value: i64,
        /// Time unit — `Seconds`, `Millis`, `Micros`, or `Nanos`.
        unit: TimeUnit,
        /// Optional IANA time-zone identifier (e.g. `"UTC"`, `"America/New_York"`).
        timezone: Option<String>,
    },
}

// ---------------------------------------------------------------------------
// PhysicalColumnStats
// ---------------------------------------------------------------------------

/// Per-physical-column statistics returned by [`crate::HeliumReader::stripe_column_stats`].
///
/// These are read directly from the footer metadata — no column data I/O is
/// performed. All fields are `None` for files written before stats were added,
/// or for columns where stats were disabled at write time.
#[derive(Debug, Clone)]
pub struct PhysicalColumnStats {
    /// Minimum value across non-null rows. `None` if empty, all-null, or
    /// all-NaN (for floats).
    pub min: Option<MinMaxValue>,
    /// Maximum value across non-null rows. `None` if empty, all-null, or
    /// all-NaN (for floats).
    pub max: Option<MinMaxValue>,
    /// Number of null rows for this physical column. For non-nullable types
    /// this is always `Some(0)`. `None` if stats were not computed.
    pub null_count: Option<u64>,
}

// ---------------------------------------------------------------------------
// ContainmentFilter
// ---------------------------------------------------------------------------

/// Per-physical-column containment filter for equality pushdown.
///
/// Stored alongside min/max in the `.he` footer.  Used to answer
/// `WHERE col = x` and `WHERE col IN (...)` predicates at stripe granularity.
///
/// # Wire format
///
/// The `kind` tag is **frozen** once any `.he` file is written with this
/// filter — do not rename or repurpose variants without a file-format version
/// bump.  The `content` field carries the variant payload.
///
/// # False negatives
///
/// `DistinctSet` is exact (zero false negatives).  `Bloom` may have false
/// positives (≈1% FPP for typical sizing) but **never** false negatives —
/// if a value is present it will always be reported as `might_contain = true`.
///
/// # Size
///
/// - `DistinctSet` with N ≤ 256 entries × ≈40 bytes each (JSON) ≈ 10 KB max.
/// - `Bloom` is capped at 64 KB per filter (524 288 bits).
///   For a 10 000-row stripe at 1 % FPP: m ≈ 9.585 × 10 000 ≈ 96 000 bits
///   ≈ 12 KB — well below the cap.
///
/// Users with very wide (100-col) × very deep (1000-stripe) files should
/// consider [`crate::HeliumWriter::with_filters_disabled`] until per-column
/// opt-in heuristics are added.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum ContainmentFilter {
    /// Exact set of distinct values.  Used when cardinality ≤ 256.
    /// Lookup is `O(N)` where N ≤ 256.  Zero false positives or false negatives.
    DistinctSet(Vec<MinMaxValue>),
    /// Probabilistic membership filter.
    ///
    /// Uses two independent 64-bit hashes (Murmur2/64-style) derived with the
    /// `(h1 + i * h2)` trick to generate `k` bit positions.  Achieves ≈1 %
    /// FPP for the default sizing (`k ≈ 7`, `m ≈ 9.585 × n` bits).
    Bloom {
        /// Raw bit array — `bits.len() == m / 8`.
        bits: Vec<u8>,
        /// Number of bits in the filter (always a multiple of 64).
        m: u32,
        /// Number of hash functions (typically 7 for 1 % FPP).
        k: u8,
    },
}

// ---------------------------------------------------------------------------
// Bloom filter helpers
// ---------------------------------------------------------------------------

/// Murmur-inspired 64-bit finaliser (MurmurHash2_64 body).
///
/// Produces a stable, high-quality 64-bit hash from a byte slice.  Not
/// cryptographic; only used for Bloom filter bit-position derivation.
fn murmur64(data: &[u8]) -> u64 {
    const M: u64 = 0xc6a4_a793_5bd1_e995;
    const R: u32 = 47;
    let mut h: u64 = 0xe17a_1465u64 ^ (data.len() as u64).wrapping_mul(M);
    let mut chunks = data.chunks_exact(8);
    for chunk in chunks.by_ref() {
        // chunks_exact(8) guarantees exactly 8 bytes; try_into cannot fail here.
        let Ok(arr) = chunk.try_into() else { break };
        let mut k = u64::from_le_bytes(arr);
        k = k.wrapping_mul(M);
        k ^= k >> R;
        k = k.wrapping_mul(M);
        h ^= k;
        h = h.wrapping_mul(M);
    }
    let rem = chunks.remainder();
    if !rem.is_empty() {
        let mut tail = 0u64;
        for (i, &b) in rem.iter().enumerate() {
            tail |= (b as u64) << (i * 8);
        }
        h ^= tail;
        h = h.wrapping_mul(M);
    }
    h ^= h >> R;
    h = h.wrapping_mul(M);
    h ^= h >> R;
    h
}

/// Compute two independent 64-bit hashes for a byte slice using the
/// `(h1, h2)` seed-split technique.
///
/// `h1` and `h2` are combined via `h1_i + i * h2_i` to derive the k bit
/// positions without calling the hash function k times.
fn double_hash(data: &[u8]) -> (u64, u64) {
    let h1 = murmur64(data);
    // A second independent hash produced by XOR-shifting h1 with a large
    // prime seed.  This provides sufficient independence for Bloom filters.
    let h2 = murmur64(&h1.to_le_bytes());
    (h1, h2)
}

/// Build a Bloom filter over a slice of byte keys.
///
/// Sizing: `m = ceil(9.585 * n)` bits rounded to the next multiple of 64,
/// then capped at `MAX_BLOOM_BYTES * 8` bits.
/// `k = max(1, round((m / n) * ln(2)))`.
fn build_bloom(keys: &[Vec<u8>]) -> ContainmentFilter {
    let n = keys.len().max(1);
    // m = ceil(-n * ln(0.01) / ln(2)^2) ≈ 9.585 * n bits
    let m_raw = (9.585_f64 * n as f64).ceil() as usize;
    // Round up to next multiple of 64 bits.
    let m_bits = m_raw.div_ceil(64) * 64;
    // Cap at MAX_BLOOM_BYTES * 8 bits.
    let m_bits = m_bits.min(MAX_BLOOM_BYTES * 8);
    let m = m_bits as u32;
    let k = ((m as f64 / n as f64) * std::f64::consts::LN_2)
        .round()
        .clamp(1.0, 30.0) as u8;

    let mut bits = vec![0u8; m_bits / 8];
    for key in keys {
        bloom_insert(&mut bits, m, k, key);
    }
    ContainmentFilter::Bloom { bits, m, k }
}

/// Insert `key` bytes into a Bloom filter bit array.
fn bloom_insert(bits: &mut [u8], m: u32, k: u8, key: &[u8]) {
    let (h1, h2) = double_hash(key);
    for i in 0..k as u64 {
        let bit_pos = (h1.wrapping_add(i.wrapping_mul(h2)) % m as u64) as usize;
        bits[bit_pos / 8] |= 1 << (bit_pos % 8);
    }
}

/// Query whether `key` bytes might be present in a Bloom filter.
///
/// Returns `true` if all k bits are set (might be present, with FPP ≈1 %).
/// Returns `false` only if at least one bit is unset (definitely not present).
pub fn bloom_might_contain(bits: &[u8], m: u32, k: u8, key: &[u8]) -> bool {
    let (h1, h2) = double_hash(key);
    for i in 0..k as u64 {
        let bit_pos = (h1.wrapping_add(i.wrapping_mul(h2)) % m as u64) as usize;
        if bits[bit_pos / 8] & (1 << (bit_pos % 8)) == 0 {
            return false;
        }
    }
    true
}

/// Encode a `MinMaxValue` as the canonical byte sequence used for Bloom
/// filter hashing.
///
/// For numeric types: little-endian binary encoding.
/// For strings: UTF-8 bytes.
/// For binary: raw bytes.
/// For semantic types: a type-prefixed little-endian encoding.
pub fn min_max_value_to_hash_bytes(v: &MinMaxValue) -> Vec<u8> {
    match v {
        MinMaxValue::I8(x) => x.to_le_bytes().to_vec(),
        MinMaxValue::I16(x) => x.to_le_bytes().to_vec(),
        MinMaxValue::I32(x) => x.to_le_bytes().to_vec(),
        MinMaxValue::I64(x) => x.to_le_bytes().to_vec(),
        MinMaxValue::U8(x) => x.to_le_bytes().to_vec(),
        MinMaxValue::U16(x) => x.to_le_bytes().to_vec(),
        MinMaxValue::U32(x) => x.to_le_bytes().to_vec(),
        MinMaxValue::U64(x) => x.to_le_bytes().to_vec(),
        MinMaxValue::F32(x) => x.to_le_bytes().to_vec(),
        MinMaxValue::F64(x) => x.to_le_bytes().to_vec(),
        MinMaxValue::Utf8(s) => s.as_bytes().to_vec(),
        MinMaxValue::Binary(b64) => b64.as_bytes().to_vec(),
        MinMaxValue::Decimal128 {
            high,
            low,
            precision,
            scale,
        } => {
            // Encode as: high(8) + low(8) + precision(1) + scale(1) = 18 bytes.
            let mut out = Vec::with_capacity(18);
            out.extend_from_slice(&high.to_le_bytes());
            out.extend_from_slice(&low.to_le_bytes());
            out.push(*precision);
            out.push(*scale);
            out
        }
        MinMaxValue::Date { value, unit } => {
            let unit_byte: u8 = match unit {
                DateUnit::Days => 0,
                DateUnit::Millis => 1,
            };
            let mut out = Vec::with_capacity(9);
            out.push(unit_byte);
            out.extend_from_slice(&value.to_le_bytes());
            out
        }
        MinMaxValue::Datetime {
            value,
            unit,
            timezone,
        } => {
            let unit_byte: u8 = match unit {
                TimeUnit::Seconds => 0,
                TimeUnit::Millis => 1,
                TimeUnit::Micros => 2,
                TimeUnit::Nanos => 3,
            };
            let mut out = Vec::with_capacity(9 + timezone.as_deref().map_or(0, |tz| tz.len()));
            out.push(unit_byte);
            out.extend_from_slice(&value.to_le_bytes());
            if let Some(tz) = timezone {
                out.extend_from_slice(tz.as_bytes());
            }
            out
        }
    }
}

/// Query a `ContainmentFilter` for a `MinMaxValue` key.
///
/// Returns `true` if the value **might** be in the filter (conservative),
/// `false` only if it is definitely absent.
pub fn filter_might_contain_mmv(filter: &ContainmentFilter, value: &MinMaxValue) -> bool {
    match filter {
        ContainmentFilter::DistinctSet(set) => set.iter().any(|v| v == value),
        ContainmentFilter::Bloom { bits, m, k } => {
            bloom_might_contain(bits, *m, *k, &min_max_value_to_hash_bytes(value))
        }
    }
}

// ---------------------------------------------------------------------------
// Per-column filter computation
// ---------------------------------------------------------------------------

/// Accumulator used during a single stripe write to build a
/// `ContainmentFilter` for one physical column.
enum FilterAccumulator {
    /// Still tracking the exact distinct set.
    Tracking(HashSet<MinMaxValue>),
    /// Overflowed MAX_DISTINCT_SET_SIZE; collecting raw byte keys for Bloom.
    Overflow(Vec<Vec<u8>>),
}

impl FilterAccumulator {
    fn new() -> Self {
        Self::Tracking(HashSet::new())
    }

    fn insert_mmv(&mut self, v: MinMaxValue) {
        match self {
            FilterAccumulator::Tracking(set) => {
                set.insert(v.clone());
                if set.len() > MAX_DISTINCT_SET_SIZE {
                    // Promote to Bloom: re-hash all existing values.
                    let keys: Vec<Vec<u8>> = set.iter().map(min_max_value_to_hash_bytes).collect();
                    *self = FilterAccumulator::Overflow(keys);
                }
            }
            FilterAccumulator::Overflow(keys) => {
                keys.push(min_max_value_to_hash_bytes(&v));
            }
        }
    }

    fn finish(self) -> Option<ContainmentFilter> {
        match self {
            FilterAccumulator::Tracking(set) => {
                if set.is_empty() {
                    None
                } else {
                    let mut sorted: Vec<MinMaxValue> = set.into_iter().collect();
                    // Sort for deterministic serialisation (easier to compare in tests).
                    sorted.sort_by(mmv_cmp);
                    Some(ContainmentFilter::DistinctSet(sorted))
                }
            }
            FilterAccumulator::Overflow(keys) => Some(build_bloom(&keys)),
        }
    }
}

/// Map a `MinMaxValue` variant to a stable integer discriminant for ordering.
fn mmv_discriminant(v: &MinMaxValue) -> u8 {
    match v {
        MinMaxValue::I8(_) => 0,
        MinMaxValue::I16(_) => 1,
        MinMaxValue::I32(_) => 2,
        MinMaxValue::I64(_) => 3,
        MinMaxValue::U8(_) => 4,
        MinMaxValue::U16(_) => 5,
        MinMaxValue::U32(_) => 6,
        MinMaxValue::U64(_) => 7,
        MinMaxValue::F32(_) => 8,
        MinMaxValue::F64(_) => 9,
        MinMaxValue::Utf8(_) => 10,
        MinMaxValue::Binary(_) => 11,
        MinMaxValue::Decimal128 { .. } => 12,
        MinMaxValue::Date { .. } => 13,
        MinMaxValue::Datetime { .. } => 14,
    }
}

/// Total ordering for `MinMaxValue` used to produce a canonical sort order
/// within a `DistinctSet`.  Same-variant values are compared by value;
/// cross-variant ordering falls back to discriminant comparison.
///
/// # Semantic variant handling
///
/// - `Decimal128`: reconstructs the i128 and compares.  Requires same
///   `(precision, scale)` — if they differ the writer invariant is violated;
///   falls back to discriminant ordering.
/// - `Date`: requires same `unit`; compares `value` directly (both represent
///   the same epoch scale).  Unit mismatch → discriminant fallback.
/// - `Datetime`: requires same `unit` and `timezone`; compares `value`.
///   Unit or timezone mismatch → discriminant fallback.
pub(super) fn mmv_cmp(a: &MinMaxValue, b: &MinMaxValue) -> std::cmp::Ordering {
    use MinMaxValue::*;
    match (a, b) {
        (I8(x), I8(y)) => x.cmp(y),
        (I16(x), I16(y)) => x.cmp(y),
        (I32(x), I32(y)) => x.cmp(y),
        (I64(x), I64(y)) => x.cmp(y),
        (U8(x), U8(y)) => x.cmp(y),
        (U16(x), U16(y)) => x.cmp(y),
        (U32(x), U32(y)) => x.cmp(y),
        (U64(x), U64(y)) => x.cmp(y),
        (F32(x), F32(y)) => x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal),
        (F64(x), F64(y)) => x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal),
        (Utf8(x), Utf8(y)) => x.cmp(y),
        (Binary(x), Binary(y)) => x.cmp(y),
        // Decimal128: same precision/scale → compare reconstructed i128.
        (
            Decimal128 {
                high: ah,
                low: al,
                precision: ap,
                scale: as_,
            },
            Decimal128 {
                high: bh,
                low: bl,
                precision: bp,
                scale: bs,
            },
        ) if ap == bp && as_ == bs => {
            let av: i128 = ((*ah as i128) << 64) | (*al as u64 as i128);
            let bv: i128 = ((*bh as i128) << 64) | (*bl as u64 as i128);
            av.cmp(&bv)
        }
        // Date: same unit → compare values directly (same epoch + same scale).
        (
            Date {
                value: av,
                unit: au,
            },
            Date {
                value: bv,
                unit: bu,
            },
        ) if au == bu => av.cmp(bv),
        // Datetime: same unit AND timezone → compare values.
        (
            Datetime {
                value: av,
                unit: au,
                timezone: atz,
            },
            Datetime {
                value: bv,
                unit: bu,
                timezone: btz,
            },
        ) if au == bu && atz == btz => av.cmp(bv),
        // Mixed variants or mismatched semantic metadata: stable ordering by discriminant.
        _ => mmv_discriminant(a).cmp(&mmv_discriminant(b)),
    }
}

// `MinMaxValue` needs to implement `Eq` and `Hash` for use in a `HashSet`.
impl Eq for MinMaxValue {}

impl std::hash::Hash for MinMaxValue {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            MinMaxValue::I8(v) => v.hash(state),
            MinMaxValue::I16(v) => v.hash(state),
            MinMaxValue::I32(v) => v.hash(state),
            MinMaxValue::I64(v) => v.hash(state),
            MinMaxValue::U8(v) => v.hash(state),
            MinMaxValue::U16(v) => v.hash(state),
            MinMaxValue::U32(v) => v.hash(state),
            MinMaxValue::U64(v) => v.hash(state),
            // For floats: use bit representation so NaN==NaN and -0==+0 are distinct.
            MinMaxValue::F32(v) => v.to_bits().hash(state),
            MinMaxValue::F64(v) => v.to_bits().hash(state),
            MinMaxValue::Utf8(s) => s.hash(state),
            MinMaxValue::Binary(s) => s.hash(state),
            MinMaxValue::Decimal128 {
                high,
                low,
                precision,
                scale,
            } => {
                high.hash(state);
                low.hash(state);
                precision.hash(state);
                scale.hash(state);
            }
            MinMaxValue::Date { value, unit } => {
                value.hash(state);
                unit.hash(state);
            }
            MinMaxValue::Datetime {
                value,
                unit,
                timezone,
            } => {
                value.hash(state);
                unit.hash(state);
                timezone.hash(state);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Filter computation
// ---------------------------------------------------------------------------

/// Compute a `ContainmentFilter` from a `ColumnData` leaf.
///
/// Returns `None` for `Bytes` leaves (no typed values to hash) or empty slices.
fn compute_filter_for_column_data(data: &ColumnData) -> Option<ContainmentFilter> {
    let mut acc = FilterAccumulator::new();
    match data {
        ColumnData::I8(v) => {
            if v.is_empty() {
                return None;
            }
            for &x in v {
                acc.insert_mmv(MinMaxValue::I8(x));
            }
        }
        ColumnData::I16(v) => {
            if v.is_empty() {
                return None;
            }
            for &x in v {
                acc.insert_mmv(MinMaxValue::I16(x));
            }
        }
        ColumnData::I32(v) => {
            if v.is_empty() {
                return None;
            }
            for &x in v {
                acc.insert_mmv(MinMaxValue::I32(x));
            }
        }
        ColumnData::I64(v) => {
            if v.is_empty() {
                return None;
            }
            for &x in v {
                acc.insert_mmv(MinMaxValue::I64(x));
            }
        }
        ColumnData::U8(v) => {
            if v.is_empty() {
                return None;
            }
            for &x in v {
                acc.insert_mmv(MinMaxValue::U8(x));
            }
        }
        ColumnData::U16(v) => {
            if v.is_empty() {
                return None;
            }
            for &x in v {
                acc.insert_mmv(MinMaxValue::U16(x));
            }
        }
        ColumnData::U32(v) => {
            if v.is_empty() {
                return None;
            }
            for &x in v {
                acc.insert_mmv(MinMaxValue::U32(x));
            }
        }
        ColumnData::U64(v) => {
            if v.is_empty() {
                return None;
            }
            for &x in v {
                acc.insert_mmv(MinMaxValue::U64(x));
            }
        }
        ColumnData::F32(v) => {
            if v.is_empty() {
                return None;
            }
            // Skip NaN (same policy as min/max stats).
            for &x in v {
                if x.is_finite() {
                    acc.insert_mmv(MinMaxValue::F32(x));
                }
            }
        }
        ColumnData::F64(v) => {
            if v.is_empty() {
                return None;
            }
            for &x in v {
                if x.is_finite() {
                    acc.insert_mmv(MinMaxValue::F64(x));
                }
            }
        }
        // Bytes leaves have no typed representation; no filter is produced.
        ColumnData::Bytes(_) => return None,
    }
    acc.finish()
}

/// Compute containment filters for a `LogicalColumn`.
///
/// Returns one `Option<ContainmentFilter>` per physical leaf in the same order
/// as `LogicalType::physical_fields()`.  Returns `None` for offset/bitmap leaves
/// and nested types (same policy as min/max stats).
pub(super) fn compute_filters_for_logical_column(
    data: &LogicalColumn,
    lt: &LogicalType,
) -> Vec<Option<ContainmentFilter>> {
    match (data, lt) {
        // --- Primitive leaf ---
        (LogicalColumn::Primitive(d), _) => {
            vec![compute_filter_for_column_data(d)]
        }
        // --- Utf8 ---
        (LogicalColumn::Utf8(strings), LogicalType::Utf8) => {
            // offsets leaf: no filter
            let mut acc = FilterAccumulator::new();
            for s in strings {
                acc.insert_mmv(MinMaxValue::Utf8(s.clone()));
            }
            vec![None, acc.finish()]
        }
        // --- Binary ---
        (LogicalColumn::Binary(blobs), LogicalType::Binary) => {
            // offsets leaf: no filter; data leaf: Binary
            let mut acc = FilterAccumulator::new();
            for b in blobs {
                acc.insert_mmv(MinMaxValue::Binary(encode_binary_stat(b)));
            }
            vec![None, acc.finish()]
        }
        // --- Dictionary{inner} ---
        (LogicalColumn::Dictionary { dictionary, .. }, LogicalType::Dictionary { inner }) => {
            // Delegate to the inner column's filter computation, then append
            // None for the trailing indices leaf.
            let mut result = compute_filters_for_logical_column(dictionary, inner);
            result.push(None); // indices leaf: no filter
            result
        }
        // --- Struct: recurse ---
        (
            LogicalColumn::Struct { fields: col_fields },
            LogicalType::Struct {
                fields: spec_fields,
            },
        ) => {
            let mut result = Vec::new();
            for ((_, col), spec) in col_fields.iter().zip(spec_fields.iter()) {
                result.extend(compute_filters_for_logical_column(col, &spec.logical_type));
            }
            result
        }
        // --- List: offsets + inner ---
        (LogicalColumn::List { values, .. }, LogicalType::List { inner }) => {
            let mut result = vec![None]; // offsets
            result.extend(compute_filters_for_logical_column(values, inner));
            result
        }
        // --- Map: offsets + keys + values ---
        (LogicalColumn::Map { keys, values, .. }, LogicalType::Map { key, value }) => {
            let mut result = vec![None]; // offsets
            result.extend(compute_filters_for_logical_column(keys, key));
            result.extend(compute_filters_for_logical_column(values, value));
            result
        }
        // --- Nullable (recursive): present + inner ---
        (LogicalColumn::Nullable { value, .. }, LogicalType::Nullable { inner }) => {
            let mut result = vec![None]; // present bitmap
            result.extend(compute_filters_for_logical_column(value, inner));
            result
        }
        // --- Union: tag + per-variant ---
        (
            LogicalColumn::Union {
                variants: col_variants,
                ..
            },
            LogicalType::Union {
                variants: spec_variants,
            },
        ) => {
            let mut result = vec![None]; // tag
            for ((_, col), (_, spec_lt)) in col_variants.iter().zip(spec_variants.iter()) {
                result.extend(compute_filters_for_logical_column(col, spec_lt));
            }
            result
        }
        // --- Decimal128: two I64 leaves ---
        (LogicalColumn::Decimal128 { .. }, LogicalType::Decimal128 { .. }) => {
            // Decimal128 splits into two i64 halves — equality on split representation
            // is not meaningful, so skip filters for this type.
            vec![None, None]
        }
        // --- Date32 ---
        (LogicalColumn::Date32 { values }, _) => {
            vec![compute_filter_for_column_data(&ColumnData::I32(
                values.clone(),
            ))]
        }
        // --- Date64 ---
        (LogicalColumn::Date64 { values }, _) => {
            vec![compute_filter_for_column_data(&ColumnData::I64(
                values.clone(),
            ))]
        }
        // --- Datetime ---
        (LogicalColumn::Datetime { values }, _) => {
            vec![compute_filter_for_column_data(&ColumnData::I64(
                values.clone(),
            ))]
        }
        // Fallback: no filters.
        _ => lt.physical_fields().iter().map(|_| None).collect(),
    }
}

// ---------------------------------------------------------------------------
// Min/max computation
// ---------------------------------------------------------------------------

/// Compute (min, max) from a [`ColumnData`].
///
/// For float types, NaN values are excluded. Returns `(None, None)` if the
/// slice is empty or all values are NaN.
fn compute_min_max(part: &ColumnData) -> (Option<MinMaxValue>, Option<MinMaxValue>) {
    // Integer variants are all `min()`/`max()` over `Ord`, re-wrapped in the
    // matching `MinMaxValue` variant. `$variant` names both the `ColumnData`
    // and `MinMaxValue` arm (they share spelling for every integer width).
    macro_rules! int_min_max {
        ($($variant:ident),* $(,)?) => {
            match part {
                $(ColumnData::$variant(v) => {
                    if v.is_empty() { return (None, None); }
                    // SAFETY: v is non-empty as checked above.
                    let min = *v.iter().min().unwrap();
                    let max = *v.iter().max().unwrap();
                    return (Some(MinMaxValue::$variant(min)), Some(MinMaxValue::$variant(max)));
                })*
                _ => {}
            }
        };
    }
    int_min_max!(I8, I16, I32, I64, U8, U16, U32, U64);
    match part {
        ColumnData::F32(v) => {
            // Filter out NaN and non-finite values: JSON cannot represent
            // infinity, so serde_json would serialize them as null which
            // breaks footer deserialization.
            let mut finite = v.iter().copied().filter(|x| x.is_finite());
            let Some(first) = finite.next() else {
                return (None, None);
            };
            let (min, max) = finite.fold((first, first), |(lo, hi), x| {
                (if x < lo { x } else { lo }, if x > hi { x } else { hi })
            });
            (Some(MinMaxValue::F32(min)), Some(MinMaxValue::F32(max)))
        }
        ColumnData::F64(v) => {
            // Filter out NaN and non-finite values: JSON cannot represent
            // infinity, so serde_json would serialize them as null which
            // breaks footer deserialization.
            let mut finite = v.iter().copied().filter(|x| x.is_finite());
            let Some(first) = finite.next() else {
                return (None, None);
            };
            let (min, max) = finite.fold((first, first), |(lo, hi), x| {
                (if x < lo { x } else { lo }, if x > hi { x } else { hi })
            });
            (Some(MinMaxValue::F64(min)), Some(MinMaxValue::F64(max)))
        }
        // Integer variants returned early above; only `Bytes` remains here and
        // carries no orderable min/max.
        _ => (None, None),
    }
}

/// Truncate a string to at most `STATS_TRUNCATE_BYTES` bytes, respecting
/// UTF-8 character boundaries.
fn truncate_str(s: &str) -> String {
    if s.len() <= STATS_TRUNCATE_BYTES {
        s.to_owned()
    } else {
        // Walk back to a valid UTF-8 boundary.
        let mut end = STATS_TRUNCATE_BYTES;
        while !s.is_char_boundary(end) {
            end -= 1;
        }
        s[..end].to_owned()
    }
}

/// Encode binary data for storage: truncate then base64.
pub(super) fn encode_binary_stat(b: &[u8]) -> String {
    use std::fmt::Write as FmtWrite;
    let slice = if b.len() > STATS_TRUNCATE_BYTES {
        &b[..STATS_TRUNCATE_BYTES]
    } else {
        b
    };
    // Simple base64 encoding without an external dep.
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(slice.len().div_ceil(3) * 4);
    for chunk in slice.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let n = (b0 << 16) | (b1 << 8) | b2;
        let _ = write!(out, "{}", CHARS[(n >> 18) as usize] as char);
        let _ = write!(out, "{}", CHARS[((n >> 12) & 0x3f) as usize] as char);
        if chunk.len() > 1 {
            let _ = write!(out, "{}", CHARS[((n >> 6) & 0x3f) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            let _ = write!(out, "{}", CHARS[(n & 0x3f) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

/// One (min, max, null_count) triplet per physical leaf in declaration order.
pub(super) type LeafStats = Vec<(Option<MinMaxValue>, Option<MinMaxValue>, Option<u64>)>;

/// Compute per-leaf statistics for a `LogicalColumn` before it is decomposed
/// into `ColumnData` parts.
///
/// Returns one entry per physical field in the same order as
/// `logical_type.physical_fields()`. For non-leaf-bearing roles (e.g. the
/// `present` bitmap of a Nullable, offsets of a List) the entry has all
/// `None`s so the vector length always matches `physical_fields().len()`.
pub(super) fn compute_stats_for_logical_column(
    data: &LogicalColumn,
    lt: &LogicalType,
) -> LeafStats {
    match (data, lt) {
        // --- Primitive leaf ---
        (LogicalColumn::Primitive(d), _) => {
            let (mn, mx) = compute_min_max(d);
            vec![(mn, mx, Some(0))]
        }
        // --- Utf8 leaf ---
        (LogicalColumn::Utf8(strings), LogicalType::Utf8) => {
            // offsets physical field: no stats
            // data physical field: lex min/max on the strings
            let (mn, mx) = match (strings.iter().min(), strings.iter().max()) {
                (Some(min), Some(max)) => (
                    Some(MinMaxValue::Utf8(truncate_str(min))),
                    Some(MinMaxValue::Utf8(truncate_str(max))),
                ),
                _ => (None, None),
            };
            // [offsets, data]
            vec![(None, None, Some(0)), (mn, mx, Some(0))]
        }
        // --- Binary leaf ---
        (LogicalColumn::Binary(blobs), LogicalType::Binary) => {
            let min_blob = blobs.iter().min_by(|a, b| a.as_slice().cmp(b.as_slice()));
            let max_blob = blobs.iter().max_by(|a, b| a.as_slice().cmp(b.as_slice()));
            let (mn, mx) = match (min_blob, max_blob) {
                (Some(mn_b), Some(mx_b)) => (
                    Some(MinMaxValue::Binary(encode_binary_stat(mn_b))),
                    Some(MinMaxValue::Binary(encode_binary_stat(mx_b))),
                ),
                _ => (None, None),
            };
            vec![(None, None, Some(0)), (mn, mx, Some(0))]
        }
        // --- Dictionary{inner} ---
        (LogicalColumn::Dictionary { dictionary, .. }, LogicalType::Dictionary { inner }) => {
            // Delegate to the inner column's stats, then append no-stats for the indices leaf.
            let mut result = compute_stats_for_logical_column(dictionary, inner);
            result.push((None, None, Some(0))); // indices leaf
            result
        }
        // --- Struct: recurse into each field ---
        (
            LogicalColumn::Struct { fields: col_fields },
            LogicalType::Struct {
                fields: spec_fields,
            },
        ) => {
            let mut result = Vec::new();
            for ((_, col), spec) in col_fields.iter().zip(spec_fields.iter()) {
                result.extend(compute_stats_for_logical_column(col, &spec.logical_type));
            }
            result
        }
        // --- List: offsets (no stats) + inner leaves ---
        (LogicalColumn::List { values, .. }, LogicalType::List { inner }) => {
            let mut result = vec![(None, None, Some(0u64))]; // offsets
            result.extend(compute_stats_for_logical_column(values, inner));
            result
        }
        // --- Map: offsets (no stats) + key leaves + value leaves ---
        (LogicalColumn::Map { keys, values, .. }, LogicalType::Map { key, value }) => {
            let mut result = vec![(None, None, Some(0u64))]; // offsets
            result.extend(compute_stats_for_logical_column(keys, key));
            result.extend(compute_stats_for_logical_column(values, value));
            result
        }
        // --- Nullable (recursive): present bitmap (no stats) + inner leaves ---
        (LogicalColumn::Nullable { present, value }, LogicalType::Nullable { inner }) => {
            let null_count = present.iter().filter(|&&p| !p).count() as u64;
            let mut result = vec![(None, None, Some(null_count))]; // present bitmap
            // Inner stats are computed on the compacted (non-null) inner values.
            result.extend(compute_stats_for_logical_column(value, inner));
            result
        }
        // --- Union: tag (no stats) + per-variant leaves ---
        (
            LogicalColumn::Union {
                variants: col_variants,
                ..
            },
            LogicalType::Union {
                variants: spec_variants,
            },
        ) => {
            let mut result = vec![(None, None, Some(0u64))]; // tag
            for ((_, col), (_, spec_lt)) in col_variants.iter().zip(spec_variants.iter()) {
                result.extend(compute_stats_for_logical_column(col, spec_lt));
            }
            result
        }
        // --- Decimal128: two I64 leaves; stats on both raw halves.
        //     Store as typed Decimal128 variants for correct value-level comparison.
        (LogicalColumn::Decimal128 { values }, LogicalType::Decimal128 { precision, scale }) => {
            let (Some(min_v), Some(max_v)) =
                (values.iter().copied().min(), values.iter().copied().max())
            else {
                return vec![(None, None, Some(0u64)), (None, None, Some(0u64))];
            };
            let min_high = (min_v >> 64) as i64;
            let min_low = min_v as i64;
            let max_high = (max_v >> 64) as i64;
            let max_low = max_v as i64;
            // Both leaves carry the same min/max as typed Decimal128 values.
            // The high-word leaf gets the typed stat; the low-word leaf gets None
            // (the typed variant already encodes both halves unambiguously).
            let min_mmv = MinMaxValue::Decimal128 {
                high: min_high,
                low: min_low,
                precision: *precision,
                scale: *scale,
            };
            let max_mmv = MinMaxValue::Decimal128 {
                high: max_high,
                low: max_low,
                precision: *precision,
                scale: *scale,
            };
            vec![
                (Some(min_mmv), Some(max_mmv), Some(0)),
                (None, None, Some(0)),
            ]
        }
        // --- Date32 ---
        (LogicalColumn::Date32 { values }, LogicalType::Date { unit }) => {
            let (mn, mx) = match (values.iter().copied().min(), values.iter().copied().max()) {
                (Some(min_v), Some(max_v)) => (
                    Some(MinMaxValue::Date {
                        value: min_v as i64,
                        unit: *unit,
                    }),
                    Some(MinMaxValue::Date {
                        value: max_v as i64,
                        unit: *unit,
                    }),
                ),
                _ => (None, None),
            };
            vec![(mn, mx, Some(0))]
        }
        // --- Date64 ---
        (LogicalColumn::Date64 { values }, LogicalType::Date { unit }) => {
            let (mn, mx) = match (values.iter().copied().min(), values.iter().copied().max()) {
                (Some(min_v), Some(max_v)) => (
                    Some(MinMaxValue::Date {
                        value: min_v,
                        unit: *unit,
                    }),
                    Some(MinMaxValue::Date {
                        value: max_v,
                        unit: *unit,
                    }),
                ),
                _ => (None, None),
            };
            vec![(mn, mx, Some(0))]
        }
        // --- Datetime ---
        (LogicalColumn::Datetime { values }, LogicalType::Datetime { unit, timezone }) => {
            let (mn, mx) = match (values.iter().copied().min(), values.iter().copied().max()) {
                (Some(min_v), Some(max_v)) => (
                    Some(MinMaxValue::Datetime {
                        value: min_v,
                        unit: *unit,
                        timezone: timezone.clone(),
                    }),
                    Some(MinMaxValue::Datetime {
                        value: max_v,
                        unit: *unit,
                        timezone: timezone.clone(),
                    }),
                ),
                _ => (None, None),
            };
            vec![(mn, mx, Some(0))]
        }
        // Fallback: produce None entries for each physical field.
        _ => lt
            .physical_fields()
            .iter()
            .map(|_| (None, None, None))
            .collect(),
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify `mmv_cmp` for `Decimal128`: same precision/scale → compare by value.
    #[test]
    fn mmv_cmp_decimal128_same_scale() {
        let small = MinMaxValue::Decimal128 {
            high: 0,
            low: 100,
            precision: 10,
            scale: 2,
        };
        let large = MinMaxValue::Decimal128 {
            high: 0,
            low: 200,
            precision: 10,
            scale: 2,
        };
        let neg = MinMaxValue::Decimal128 {
            high: -1,
            low: i64::MAX,
            precision: 10,
            scale: 2,
        };

        assert_eq!(mmv_cmp(&small, &small), std::cmp::Ordering::Equal);
        assert_eq!(mmv_cmp(&small, &large), std::cmp::Ordering::Less);
        assert_eq!(mmv_cmp(&large, &small), std::cmp::Ordering::Greater);
        // A negative i128 (high = -1 means the top 64 bits are all 1s in two's-complement).
        assert_eq!(mmv_cmp(&neg, &small), std::cmp::Ordering::Less);
    }

    /// Decimal128 with different precision/scale → discriminant fallback (Equal by discriminant).
    #[test]
    fn mmv_cmp_decimal128_different_scale_falls_back() {
        let a = MinMaxValue::Decimal128 {
            high: 0,
            low: 100,
            precision: 10,
            scale: 2,
        };
        let b = MinMaxValue::Decimal128 {
            high: 0,
            low: 200,
            precision: 12,
            scale: 4,
        };
        // Both have discriminant 12 → fallback compares discriminant(12) == discriminant(12) → Equal.
        assert_eq!(mmv_cmp(&a, &b), std::cmp::Ordering::Equal);
    }

    /// `Date` variants with same unit: compare by value.
    #[test]
    fn mmv_cmp_date_same_unit() {
        let d1 = MinMaxValue::Date {
            value: 100,
            unit: DateUnit::Days,
        };
        let d2 = MinMaxValue::Date {
            value: 200,
            unit: DateUnit::Days,
        };
        assert_eq!(mmv_cmp(&d1, &d2), std::cmp::Ordering::Less);
        assert_eq!(mmv_cmp(&d2, &d1), std::cmp::Ordering::Greater);
        assert_eq!(mmv_cmp(&d1, &d1), std::cmp::Ordering::Equal);
    }

    /// `Date` variants with different units: discriminant fallback → Equal.
    #[test]
    fn mmv_cmp_date_different_unit_falls_back() {
        let d_days = MinMaxValue::Date {
            value: 100,
            unit: DateUnit::Days,
        };
        let d_ms = MinMaxValue::Date {
            value: 100,
            unit: DateUnit::Millis,
        };
        // Both discriminant 13 → Equal on fallback.
        assert_eq!(mmv_cmp(&d_days, &d_ms), std::cmp::Ordering::Equal);
    }

    /// `Datetime` with same unit and timezone: compare by value.
    #[test]
    fn mmv_cmp_datetime_same_unit_tz() {
        let t1 = MinMaxValue::Datetime {
            value: 1_000,
            unit: TimeUnit::Millis,
            timezone: Some("UTC".into()),
        };
        let t2 = MinMaxValue::Datetime {
            value: 2_000,
            unit: TimeUnit::Millis,
            timezone: Some("UTC".into()),
        };
        assert_eq!(mmv_cmp(&t1, &t2), std::cmp::Ordering::Less);
        assert_eq!(mmv_cmp(&t2, &t1), std::cmp::Ordering::Greater);
        assert_eq!(mmv_cmp(&t1, &t1), std::cmp::Ordering::Equal);
    }

    /// `Datetime` with different unit: discriminant fallback → Equal.
    #[test]
    fn mmv_cmp_datetime_different_unit_falls_back() {
        let t_ms = MinMaxValue::Datetime {
            value: 1_000,
            unit: TimeUnit::Millis,
            timezone: None,
        };
        let t_us = MinMaxValue::Datetime {
            value: 1_000_000,
            unit: TimeUnit::Micros,
            timezone: None,
        };
        assert_eq!(mmv_cmp(&t_ms, &t_us), std::cmp::Ordering::Equal);
    }

    /// `Datetime` with different timezone: discriminant fallback → Equal.
    #[test]
    fn mmv_cmp_datetime_different_tz_falls_back() {
        let t_utc = MinMaxValue::Datetime {
            value: 1_000,
            unit: TimeUnit::Millis,
            timezone: Some("UTC".into()),
        };
        let t_ny = MinMaxValue::Datetime {
            value: 1_000,
            unit: TimeUnit::Millis,
            timezone: Some("America/New_York".into()),
        };
        assert_eq!(mmv_cmp(&t_utc, &t_ny), std::cmp::Ordering::Equal);
    }

    /// Cross-variant ordering uses discriminant.
    #[test]
    fn mmv_cmp_cross_variant() {
        let i8_val = MinMaxValue::I8(0);
        let i16_val = MinMaxValue::I16(0);
        // discriminant(I8) = 0 < discriminant(I16) = 1
        assert_eq!(mmv_cmp(&i8_val, &i16_val), std::cmp::Ordering::Less);
    }
}
