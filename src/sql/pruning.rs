//! Stripe-level predicate pruning via DataFusion's `PruningPredicate`.
//!
//! [`HeliumPruningStatistics`] wraps the per-stripe min/max/null_count stats
//! stored in a `.he` footer and surfaces them in the vectorised shape that
//! [`PruningStatistics`] expects.  One row per stripe per column.
//!
//! # Supported column types
//!
//! Primitives (`I8`..`U64`, `F32`, `F64`), `Utf8`, `Binary`, `NullablePrim`,
//! `NullableUtf8`, `NullableBinary`, `Date`, `Datetime`, and `Decimal128`.
//! For every supported type the "meaningful" physical leaf is selected:
//!
//! | Logical type | Leaf used for min/max |
//! |---|---|
//! | `Primitive(T)` | the single `values` leaf |
//! | `Utf8` / `Binary` | `data` leaf (lex ordering) |
//! | `NullablePrim(T)` | `values` leaf; `present` leaf for null_count |
//! | `NullableUtf8` / `NullableBinary` | `data` leaf |
//! | `Date` / `Datetime` | `values` leaf |
//! | `Decimal128` | `high` leaf (upper 64 bits) |
//! | `List`, `Map`, `Struct`, `Union`, `Dict*`, `ArrayOf*` | unsupported → `None` |
//!
//! Unsupported types simply return `None` from `min_values` / `max_values`,
//! which causes `PruningPredicate::prune` to conservatively keep those stripes.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow::array::{
    ArrayRef, BooleanArray, Float32Array, Float64Array, Int8Array, Int16Array, Int32Array,
    Int64Array, StringArray, TimestampMicrosecondArray, TimestampMillisecondArray,
    TimestampNanosecondArray, TimestampSecondArray, UInt8Array, UInt16Array, UInt32Array,
    UInt64Array,
};
use datafusion::common::Column;
use datafusion::common::ScalarValue;
use datafusion::physical_optimizer::pruning::PruningStatistics;

use crate::core::schema::{DateUnit, LogicalType, TimeUnit};
use crate::{ContainmentFilter, MinMaxValue, PhysicalColumnStats};

// ---------------------------------------------------------------------------
// Scalar conversion helper (used by both pruning and statistics)
// ---------------------------------------------------------------------------

/// Convert a single [`MinMaxValue`] to a DataFusion [`ScalarValue`].
///
/// The conversion is lossless for all variants except `Decimal128`, which is
/// reconstructed from the stored `(high, low)` split. Returns `None` when the
/// variant has no unambiguous `ScalarValue` mapping (currently: `Binary`).
///
/// This helper is shared between `HeliumPruningStatistics` (which builds
/// per-stripe Arrow arrays) and the file-wide `Statistics` computation in
/// `HeliumTableProvider`.
pub fn min_max_to_scalar(value: &MinMaxValue) -> Option<ScalarValue> {
    match value {
        MinMaxValue::I8(v) => Some(ScalarValue::Int8(Some(*v))),
        MinMaxValue::I16(v) => Some(ScalarValue::Int16(Some(*v))),
        MinMaxValue::I32(v) => Some(ScalarValue::Int32(Some(*v))),
        MinMaxValue::I64(v) => Some(ScalarValue::Int64(Some(*v))),
        MinMaxValue::U8(v) => Some(ScalarValue::UInt8(Some(*v))),
        MinMaxValue::U16(v) => Some(ScalarValue::UInt16(Some(*v))),
        MinMaxValue::U32(v) => Some(ScalarValue::UInt32(Some(*v))),
        MinMaxValue::U64(v) => Some(ScalarValue::UInt64(Some(*v))),
        MinMaxValue::F32(v) => Some(ScalarValue::Float32(Some(*v))),
        MinMaxValue::F64(v) => Some(ScalarValue::Float64(Some(*v))),
        MinMaxValue::Utf8(s) => Some(ScalarValue::Utf8(Some(s.clone()))),
        // Binary stats exist but DataFusion doesn't prune on binary min/max well.
        MinMaxValue::Binary(_) => None,
        // Decimal128: reconstruct i128 from (high, low) halves.
        MinMaxValue::Decimal128 {
            high,
            low,
            precision,
            scale,
        } => {
            let v: i128 = ((*high as i128) << 64) | (*low as u64 as i128);
            Some(ScalarValue::Decimal128(Some(v), *precision, *scale as i8))
        }
        // Date: value is days (Days) or millis (Millis).
        MinMaxValue::Date { value, unit } => match unit {
            crate::DateUnit::Days => Some(ScalarValue::Date32(Some(*value as i32))),
            crate::DateUnit::Millis => Some(ScalarValue::Date64(Some(*value))),
        },
        // Datetime: value is the raw i64 in the stored unit.
        MinMaxValue::Datetime {
            value,
            unit,
            timezone,
        } => {
            let tz: Option<Arc<str>> = timezone.as_deref().map(Arc::from);
            match unit {
                crate::TimeUnit::Seconds => Some(ScalarValue::TimestampSecond(Some(*value), tz)),
                crate::TimeUnit::Millis => {
                    Some(ScalarValue::TimestampMillisecond(Some(*value), tz))
                }
                crate::TimeUnit::Micros => {
                    Some(ScalarValue::TimestampMicrosecond(Some(*value), tz))
                }
                crate::TimeUnit::Nanos => Some(ScalarValue::TimestampNanosecond(Some(*value), tz)),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Struct
// ---------------------------------------------------------------------------

/// Per-stripe min/max/null_count statistics adapter for DataFusion pruning.
///
/// Created in [`crate::sql::HeliumTableProvider`]'s `scan` implementation from
/// the file footer before building [`crate::sql::HeliumExec`].  Implements
/// `PruningStatistics` so DataFusion can evaluate a `PruningPredicate` against it.
#[derive(Debug)]
pub struct HeliumPruningStatistics {
    /// Total number of stripes == `num_containers()`.
    pub stripe_count: usize,
    /// Per-stripe row counts (used by `row_counts()`).
    pub stripe_row_counts: Vec<u64>,
    /// `column_name → per-stripe physical leaf stats`.
    ///
    /// Outer key is the logical column name.  The `Vec<Vec<PhysicalColumnStats>>`
    /// has `stripe_count` outer elements; each inner `Vec` contains one entry
    /// per physical leaf in declaration order.
    pub column_stats: HashMap<String, Vec<Vec<PhysicalColumnStats>>>,
    /// Logical type per column — used to pick the right leaf and build the
    /// correct Arrow array type.
    pub column_types: HashMap<String, LogicalType>,
    /// `column_name → per-stripe per-physical-leaf containment filters`.
    ///
    /// Same shape as `column_stats`.  `None` at the inner level means no
    /// filter was written for that leaf (disabled, unsupported type, or empty).
    pub column_filters: HashMap<String, Vec<Vec<Option<ContainmentFilter>>>>,
}

// ---------------------------------------------------------------------------
// Helpers: MinMaxValue → scalar for Arrow builders
// ---------------------------------------------------------------------------

/// Build an `ArrayRef` of the primitive Arrow type that matches the
/// `MinMaxValue` variant.  `values` has `stripe_count` elements;
/// `None` means the stat is absent for that stripe (emits Arrow null).
///
/// Returns `None` when the `MinMaxValue` variant has no unambiguous Arrow
/// numeric mapping (e.g., `Binary`), or if `values` is empty.
fn build_numeric_array(values: &[Option<MinMaxValue>]) -> Option<ArrayRef> {
    if values.is_empty() {
        return None;
    }
    // Determine the type from the first non-None element.
    let representative = values.iter().find_map(|v| v.as_ref())?;
    match representative {
        MinMaxValue::I8(_) => {
            let arr: Int8Array = values
                .iter()
                .map(|v| {
                    v.as_ref().and_then(|x| {
                        if let MinMaxValue::I8(n) = x {
                            Some(*n)
                        } else {
                            None
                        }
                    })
                })
                .collect();
            Some(Arc::new(arr))
        }
        MinMaxValue::I16(_) => {
            let arr: Int16Array = values
                .iter()
                .map(|v| {
                    v.as_ref().and_then(|x| {
                        if let MinMaxValue::I16(n) = x {
                            Some(*n)
                        } else {
                            None
                        }
                    })
                })
                .collect();
            Some(Arc::new(arr))
        }
        MinMaxValue::I32(_) => {
            let arr: Int32Array = values
                .iter()
                .map(|v| {
                    v.as_ref().and_then(|x| {
                        if let MinMaxValue::I32(n) = x {
                            Some(*n)
                        } else {
                            None
                        }
                    })
                })
                .collect();
            Some(Arc::new(arr))
        }
        MinMaxValue::I64(_) => {
            let arr: Int64Array = values
                .iter()
                .map(|v| {
                    v.as_ref().and_then(|x| {
                        if let MinMaxValue::I64(n) = x {
                            Some(*n)
                        } else {
                            None
                        }
                    })
                })
                .collect();
            Some(Arc::new(arr))
        }
        MinMaxValue::U8(_) => {
            let arr: UInt8Array = values
                .iter()
                .map(|v| {
                    v.as_ref().and_then(|x| {
                        if let MinMaxValue::U8(n) = x {
                            Some(*n)
                        } else {
                            None
                        }
                    })
                })
                .collect();
            Some(Arc::new(arr))
        }
        MinMaxValue::U16(_) => {
            let arr: UInt16Array = values
                .iter()
                .map(|v| {
                    v.as_ref().and_then(|x| {
                        if let MinMaxValue::U16(n) = x {
                            Some(*n)
                        } else {
                            None
                        }
                    })
                })
                .collect();
            Some(Arc::new(arr))
        }
        MinMaxValue::U32(_) => {
            let arr: UInt32Array = values
                .iter()
                .map(|v| {
                    v.as_ref().and_then(|x| {
                        if let MinMaxValue::U32(n) = x {
                            Some(*n)
                        } else {
                            None
                        }
                    })
                })
                .collect();
            Some(Arc::new(arr))
        }
        MinMaxValue::U64(_) => {
            let arr: UInt64Array = values
                .iter()
                .map(|v| {
                    v.as_ref().and_then(|x| {
                        if let MinMaxValue::U64(n) = x {
                            Some(*n)
                        } else {
                            None
                        }
                    })
                })
                .collect();
            Some(Arc::new(arr))
        }
        MinMaxValue::F32(_) => {
            let arr: Float32Array = values
                .iter()
                .map(|v| {
                    v.as_ref().and_then(|x| {
                        if let MinMaxValue::F32(n) = x {
                            Some(*n)
                        } else {
                            None
                        }
                    })
                })
                .collect();
            Some(Arc::new(arr))
        }
        MinMaxValue::F64(_) => {
            let arr: Float64Array = values
                .iter()
                .map(|v| {
                    v.as_ref().and_then(|x| {
                        if let MinMaxValue::F64(n) = x {
                            Some(*n)
                        } else {
                            None
                        }
                    })
                })
                .collect();
            Some(Arc::new(arr))
        }
        MinMaxValue::Utf8(_) => {
            let arr: StringArray = values
                .iter()
                .map(|v| {
                    v.as_ref().and_then(|x| {
                        if let MinMaxValue::Utf8(s) = x {
                            Some(s.as_str())
                        } else {
                            None
                        }
                    })
                })
                .collect();
            Some(Arc::new(arr))
        }
        // Binary stats exist but DataFusion doesn't prune on binary min/max well.
        MinMaxValue::Binary(_) => None,
        // Semantic variants: not expected to appear as the representative for a
        // generic numeric array; return None to let the typed helpers handle them.
        MinMaxValue::Decimal128 { .. }
        | MinMaxValue::Date { .. }
        | MinMaxValue::Datetime { .. } => None,
    }
}

/// Build a typed Arrow array for the `values` physical leaf of a `Date { Days }`
/// column — `Date32Array`.
fn build_date32_array(values: &[Option<MinMaxValue>]) -> Option<ArrayRef> {
    let has_any = values.iter().any(|v| v.is_some());
    if !has_any {
        return None;
    }
    let arr: Int32Array = values
        .iter()
        .map(|v| {
            v.as_ref().and_then(|x| match x {
                // New typed variant (written by current code).
                MinMaxValue::Date {
                    value,
                    unit: crate::DateUnit::Days,
                } => Some(*value as i32),
                // Legacy raw-integer variant (written by older code).
                MinMaxValue::I32(n) => Some(*n),
                _ => None,
            })
        })
        .collect();
    // Re-interpret as Date32 (same underlying Int32).
    let arr = arrow::compute::cast(&arr, &arrow::datatypes::DataType::Date32).ok()?;
    Some(arr)
}

/// Build a typed Arrow array for `Date { Millis }` / `Datetime` / `Decimal128`
/// columns — `Date64Array` or `TimestampXxx`.
fn build_date64_array(values: &[Option<MinMaxValue>]) -> Option<ArrayRef> {
    let has_any = values.iter().any(|v| v.is_some());
    if !has_any {
        return None;
    }
    let arr: Int64Array = values
        .iter()
        .map(|v| {
            v.as_ref().and_then(|x| match x {
                // New typed variant.
                MinMaxValue::Date {
                    value,
                    unit: crate::DateUnit::Millis,
                } => Some(*value),
                // Legacy raw-integer variant.
                MinMaxValue::I64(n) => Some(*n),
                _ => None,
            })
        })
        .collect();
    let arr = arrow::compute::cast(&arr, &arrow::datatypes::DataType::Date64).ok()?;
    Some(arr)
}

fn build_timestamp_array(
    values: &[Option<MinMaxValue>],
    unit: TimeUnit,
    timezone: Option<&str>,
) -> Option<ArrayRef> {
    let has_any = values.iter().any(|v| v.is_some());
    if !has_any {
        return None;
    }
    let raw: Vec<Option<i64>> = values
        .iter()
        .map(|v| {
            v.as_ref().and_then(|x| match x {
                // New typed variant — extract value only (unit/tz come from the schema).
                MinMaxValue::Datetime { value, .. } => Some(*value),
                // Legacy raw-integer variant.
                MinMaxValue::I64(n) => Some(*n),
                _ => None,
            })
        })
        .collect();
    let tz: Option<Arc<str>> = timezone.map(Arc::from);
    let arr: ArrayRef = match unit {
        TimeUnit::Seconds => Arc::new(TimestampSecondArray::from(raw).with_timezone_opt(tz)),
        TimeUnit::Millis => Arc::new(TimestampMillisecondArray::from(raw).with_timezone_opt(tz)),
        TimeUnit::Micros => Arc::new(TimestampMicrosecondArray::from(raw).with_timezone_opt(tz)),
        TimeUnit::Nanos => Arc::new(TimestampNanosecondArray::from(raw).with_timezone_opt(tz)),
    };
    Some(arr)
}

// ---------------------------------------------------------------------------
// Leaf selection helper
// ---------------------------------------------------------------------------

/// Given a logical type and its per-stripe physical leaf stats, return
/// the (min_values, max_values, null_counts) arrays DataFusion needs, as well
/// as whether this column is nullable (for null_count extraction).
///
/// Returns `None` for leaf selection if the logical type is unsupported for
/// pruning (nested types, dict, union, array-of-x, etc.).
struct LeafSelection<'a> {
    logical_type: &'a LogicalType,
    stripe_stats: &'a Vec<Vec<PhysicalColumnStats>>,
}

impl<'a> LeafSelection<'a> {
    fn new(
        logical_type: &'a LogicalType,
        stripe_stats: &'a Vec<Vec<PhysicalColumnStats>>,
        _stripe_count: usize,
    ) -> Self {
        Self {
            logical_type,
            stripe_stats,
        }
    }

    /// Index of the "data" physical leaf to use for min/max, and index
    /// of the "present" leaf to use for null_count (or `None` if not nullable).
    fn leaf_indices(&self) -> Option<(usize, Option<usize>)> {
        match self.logical_type {
            // Primitive: single leaf (index 0)
            LogicalType::Primitive { .. } => Some((0, None)),
            // Utf8 / Binary: offsets at 0, data at 1
            LogicalType::Utf8 | LogicalType::Binary => Some((1, None)),
            // NullablePrim: present at 0, values at 1
            LogicalType::NullablePrim { .. } => Some((1, Some(0))),
            // NullableUtf8 / NullableBinary: present at 0, offsets at 1, data at 2
            LogicalType::NullableUtf8 | LogicalType::NullableBinary => Some((2, Some(0))),
            // v3 Nullable wrapping a single-leaf type: present at 0, leaf at 1
            LogicalType::Nullable { inner } => {
                match inner.as_ref() {
                    LogicalType::Primitive { .. } => Some((1, Some(0))),
                    LogicalType::Utf8 | LogicalType::Binary => {
                        // Nullable(Utf8): present(0) + offsets(1) + data(2)
                        Some((2, Some(0)))
                    }
                    LogicalType::Date {
                        unit: DateUnit::Days,
                    } => Some((1, Some(0))),
                    LogicalType::Date {
                        unit: DateUnit::Millis,
                    } => Some((1, Some(0))),
                    LogicalType::Datetime { .. } => Some((1, Some(0))),
                    LogicalType::Decimal128 { .. } => {
                        // Nullable(Decimal128): present(0) + high(1) + low(2)
                        Some((1, Some(0)))
                    }
                    _ => None,
                }
            }
            // Date / Datetime: single values leaf
            LogicalType::Date { .. } | LogicalType::Datetime { .. } => Some((0, None)),
            // Decimal128: high leaf (0), low leaf (1) — use high for pruning
            LogicalType::Decimal128 { .. } => Some((0, None)),
            // Unsupported — nested types, dicts, unions, etc.
            LogicalType::Struct { .. }
            | LogicalType::List { .. }
            | LogicalType::Map { .. }
            | LogicalType::Union { .. }
            | LogicalType::Dictionary { .. }
            | LogicalType::ArrayOf { .. }
            | LogicalType::ArrayOfUtf8 => None,
        }
    }

    /// Extract per-stripe min values for the selected leaf.
    fn min_values(&self) -> Option<ArrayRef> {
        let (data_idx, _) = self.leaf_indices()?;
        let mins: Vec<Option<MinMaxValue>> = self
            .stripe_stats
            .iter()
            .map(|stripe_leaves| {
                stripe_leaves
                    .get(data_idx)
                    .and_then(|leaf| leaf.min.clone())
            })
            .collect();
        self.build_array_for_type(&mins)
    }

    /// Extract per-stripe max values for the selected leaf.
    fn max_values(&self) -> Option<ArrayRef> {
        let (data_idx, _) = self.leaf_indices()?;
        let maxes: Vec<Option<MinMaxValue>> = self
            .stripe_stats
            .iter()
            .map(|stripe_leaves| {
                stripe_leaves
                    .get(data_idx)
                    .and_then(|leaf| leaf.max.clone())
            })
            .collect();
        self.build_array_for_type(&maxes)
    }

    /// Extract per-stripe null_count values for the present/nullable leaf.
    fn null_counts(&self) -> Option<ArrayRef> {
        let (_, present_idx) = self.leaf_indices()?;
        let present_idx = present_idx?;
        let null_counts: Vec<Option<u64>> = self
            .stripe_stats
            .iter()
            .map(|stripe_leaves| {
                stripe_leaves
                    .get(present_idx)
                    .and_then(|leaf| leaf.null_count)
            })
            .collect();
        let arr: UInt64Array = null_counts.into_iter().collect();
        Some(Arc::new(arr))
    }

    /// Build an Arrow array from `MinMaxValue`s, choosing the right type
    /// based on `self.logical_type`.
    fn build_array_for_type(&self, values: &[Option<MinMaxValue>]) -> Option<ArrayRef> {
        // For date / datetime, we need to produce the correctly typed array.
        match self.logical_type {
            LogicalType::Date {
                unit: DateUnit::Days,
            } => build_date32_array(values),
            LogicalType::Date {
                unit: DateUnit::Millis,
            } => build_date64_array(values),
            LogicalType::Datetime { unit, timezone } => {
                build_timestamp_array(values, *unit, timezone.as_deref())
            }
            LogicalType::Nullable { inner } => match inner.as_ref() {
                LogicalType::Date {
                    unit: DateUnit::Days,
                } => build_date32_array(values),
                LogicalType::Date {
                    unit: DateUnit::Millis,
                } => build_date64_array(values),
                LogicalType::Datetime { unit, timezone } => {
                    build_timestamp_array(values, *unit, timezone.as_deref())
                }
                _ => build_numeric_array(values),
            },
            // For Decimal128 we use the I64 high leaf — DataFusion doesn't have
            // great support for Decimal128 pruning so we let build_numeric_array
            // produce Int64Array; pruning will be approximate (upper 64 bits).
            _ => build_numeric_array(values),
        }
    }
}

// ---------------------------------------------------------------------------
// PruningStatistics impl
// ---------------------------------------------------------------------------

impl PruningStatistics for HeliumPruningStatistics {
    fn num_containers(&self) -> usize {
        self.stripe_count
    }

    fn min_values(&self, column: &Column) -> Option<ArrayRef> {
        let col_name = &column.name;
        let stats = self.column_stats.get(col_name.as_str())?;
        let lt = self.column_types.get(col_name.as_str())?;
        let sel = LeafSelection::new(lt, stats, self.stripe_count);
        sel.min_values()
    }

    fn max_values(&self, column: &Column) -> Option<ArrayRef> {
        let col_name = &column.name;
        let stats = self.column_stats.get(col_name.as_str())?;
        let lt = self.column_types.get(col_name.as_str())?;
        let sel = LeafSelection::new(lt, stats, self.stripe_count);
        sel.max_values()
    }

    fn null_counts(&self, column: &Column) -> Option<ArrayRef> {
        let col_name = &column.name;
        let stats = self.column_stats.get(col_name.as_str())?;
        let lt = self.column_types.get(col_name.as_str())?;
        let sel = LeafSelection::new(lt, stats, self.stripe_count);
        sel.null_counts()
    }

    fn row_counts(&self, _column: &Column) -> Option<ArrayRef> {
        // Row count is per-stripe, not per-column.
        let arr: UInt64Array = self.stripe_row_counts.iter().copied().map(Some).collect();
        Some(Arc::new(arr))
    }

    fn contained(&self, column: &Column, values: &HashSet<ScalarValue>) -> Option<BooleanArray> {
        let col_name = &column.name;
        let lt = self.column_types.get(col_name.as_str())?;
        let per_stripe_filters = self.column_filters.get(col_name.as_str())?;

        // Determine which physical leaf is the one carrying data values (same
        // leaf index used for min/max in `LeafSelection`).
        let (data_leaf_idx, _) = LeafSelection::new(lt, &Vec::new(), 0).leaf_indices()?;

        // Convert the incoming `ScalarValue`s to `MinMaxValue`s (for DistinctSet
        // lookup) and to raw byte sequences (for Bloom lookup).
        // We build both representations up front once.
        let mmvs: Vec<MinMaxValue> = values.iter().filter_map(scalar_to_min_max_value).collect();
        let byte_keys: Vec<Vec<u8>> = mmvs
            .iter()
            .map(crate::min_max_value_to_hash_bytes)
            .collect();

        // If we can't convert any value to MinMaxValue, fall back to None
        // (DataFusion will keep all stripes conservatively).
        if mmvs.is_empty() && !values.is_empty() {
            return None;
        }

        // Check whether at least one stripe has a filter for this leaf.
        let any_filter = per_stripe_filters.iter().any(|stripe_leaves| {
            stripe_leaves
                .get(data_leaf_idx)
                .and_then(|f| f.as_ref())
                .is_some()
        });
        if !any_filter {
            // No filter available for any stripe — return None so DataFusion
            // treats this path as "stat unavailable" and keeps all stripes.
            return None;
        }

        let mut result = Vec::with_capacity(self.stripe_count);
        for stripe_leaves in per_stripe_filters {
            let filter = stripe_leaves.get(data_leaf_idx).and_then(|f| f.as_ref());
            let might_contain = match filter {
                // No filter for this stripe — conservative: keep it.
                None => true,
                Some(f) => filter_might_contain_any(f, &mmvs, &byte_keys),
            };
            result.push(might_contain);
        }

        Some(BooleanArray::from(result))
    }
}

// ---------------------------------------------------------------------------
// ScalarValue → MinMaxValue conversion
// ---------------------------------------------------------------------------

/// Convert a DataFusion `ScalarValue` to a `MinMaxValue` for DistinctSet lookup.
///
/// Returns `None` for scalar types that have no `MinMaxValue` representation.
fn scalar_to_min_max_value(sv: &ScalarValue) -> Option<MinMaxValue> {
    match sv {
        ScalarValue::Int8(Some(v)) => Some(MinMaxValue::I8(*v)),
        ScalarValue::Int16(Some(v)) => Some(MinMaxValue::I16(*v)),
        ScalarValue::Int32(Some(v)) => Some(MinMaxValue::I32(*v)),
        ScalarValue::Int64(Some(v)) => Some(MinMaxValue::I64(*v)),
        ScalarValue::UInt8(Some(v)) => Some(MinMaxValue::U8(*v)),
        ScalarValue::UInt16(Some(v)) => Some(MinMaxValue::U16(*v)),
        ScalarValue::UInt32(Some(v)) => Some(MinMaxValue::U32(*v)),
        ScalarValue::UInt64(Some(v)) => Some(MinMaxValue::U64(*v)),
        ScalarValue::Float32(Some(v)) => {
            if v.is_finite() {
                Some(MinMaxValue::F32(*v))
            } else {
                None
            }
        }
        ScalarValue::Float64(Some(v)) => {
            if v.is_finite() {
                Some(MinMaxValue::F64(*v))
            } else {
                None
            }
        }
        ScalarValue::Utf8(Some(s)) | ScalarValue::LargeUtf8(Some(s)) => {
            Some(MinMaxValue::Utf8(s.clone()))
        }
        // Date32: days since epoch stored as i32
        ScalarValue::Date32(Some(v)) => Some(MinMaxValue::I32(*v)),
        // Date64: millis since epoch stored as i64
        ScalarValue::Date64(Some(v)) => Some(MinMaxValue::I64(*v)),
        // Timestamps: stored as i64
        ScalarValue::TimestampSecond(Some(v), _)
        | ScalarValue::TimestampMillisecond(Some(v), _)
        | ScalarValue::TimestampMicrosecond(Some(v), _)
        | ScalarValue::TimestampNanosecond(Some(v), _) => Some(MinMaxValue::I64(*v)),
        _ => None,
    }
}

/// Return `true` if the filter might contain any of the provided values.
///
/// For `DistinctSet`: exact O(N×K) lookup.
/// For `Bloom`: hash-and-probe for each byte key.
fn filter_might_contain_any(
    filter: &ContainmentFilter,
    mmvs: &[MinMaxValue],
    byte_keys: &[Vec<u8>],
) -> bool {
    match filter {
        ContainmentFilter::DistinctSet(set) => mmvs.iter().any(|v| set.iter().any(|s| s == v)),
        ContainmentFilter::Bloom { bits, m, k } => byte_keys
            .iter()
            .any(|key| crate::bloom_might_contain(bits, *m, *k, key)),
    }
}
