//! Stripe pruning from footer min/max statistics.
//!
//! This is the binding-side companion to Helium's per-stripe min/max index. A
//! scalar comparison predicate (`col <op> literal`) can skip an entire stripe
//! when the stripe's `[min, max]` range provably cannot satisfy the predicate.
//!
//! # Why this is not auto-driven yet
//!
//! DuckDB's **loadable** extension C-API (v1.2.0) exposes projection pushdown
//! (`duckdb_init_get_column_index`) but **no** filter-pushdown accessor — the
//! in-tree C++ `TableFilterSet` is not surfaced through the C-API a loadable
//! extension links against. So DuckDB never hands `read_he` the `WHERE` bounds,
//! and the extension cannot invoke this pruning automatically.
//!
//! The logic lives here, fully unit-tested, so it is ready the moment the C-API
//! gains a filter hook (or a future DataFusion/native integration that does
//! surface filters). Until then DuckDB applies `WHERE` after the scan.

use std::cmp::Ordering;

use helium::MinMaxValue;

/// A scalar comparison operator for stripe-pruning predicates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareOp {
    /// `col = literal`
    Eq,
    /// `col < literal`
    Lt,
    /// `col <= literal`
    Le,
    /// `col > literal`
    Gt,
    /// `col >= literal`
    Ge,
}

/// Decide whether a stripe with statistics `[min, max]` **might** contain a row
/// satisfying `col <op> literal`.
///
/// Returns `true` if the stripe must be scanned (might match) and `false` only
/// when it can be safely skipped. Conservative: returns `true` whenever the
/// statistics are missing, the literal type does not match the column type, or
/// the comparison is otherwise undecidable.
pub fn stripe_matches_range(
    min: Option<&MinMaxValue>,
    max: Option<&MinMaxValue>,
    op: CompareOp,
    literal: &MinMaxValue,
) -> bool {
    let (Some(min), Some(max)) = (min, max) else {
        // No stats → cannot prune.
        return true;
    };
    let (Some(min_cmp), Some(max_cmp)) = (compare(min, literal), compare(max, literal)) else {
        // Type mismatch / incomparable → cannot prune.
        return true;
    };

    match op {
        // min <= literal <= max
        CompareOp::Eq => min_cmp != Ordering::Greater && max_cmp != Ordering::Less,
        // some value < literal  ⇔  min < literal
        CompareOp::Lt => min_cmp == Ordering::Less,
        // some value <= literal ⇔  min <= literal
        CompareOp::Le => min_cmp != Ordering::Greater,
        // some value > literal  ⇔  max > literal
        CompareOp::Gt => max_cmp == Ordering::Greater,
        // some value >= literal ⇔  max >= literal
        CompareOp::Ge => max_cmp != Ordering::Less,
    }
}

/// Compare two `MinMaxValue`s of the same variant, returning `None` when the
/// variants differ or are not totally orderable (NaN floats).
fn compare(a: &MinMaxValue, b: &MinMaxValue) -> Option<Ordering> {
    use MinMaxValue::*;
    match (a, b) {
        (I8(x), I8(y)) => Some(x.cmp(y)),
        (I16(x), I16(y)) => Some(x.cmp(y)),
        (I32(x), I32(y)) => Some(x.cmp(y)),
        (I64(x), I64(y)) => Some(x.cmp(y)),
        (U8(x), U8(y)) => Some(x.cmp(y)),
        (U16(x), U16(y)) => Some(x.cmp(y)),
        (U32(x), U32(y)) => Some(x.cmp(y)),
        (U64(x), U64(y)) => Some(x.cmp(y)),
        (F32(x), F32(y)) => x.partial_cmp(y),
        (F64(x), F64(y)) => x.partial_cmp(y),
        (Utf8(x), Utf8(y)) => Some(x.cmp(y)),
        (Binary(x), Binary(y)) => Some(x.cmp(y)),
        (
            Date { value: x, unit: ux },
            Date { value: y, unit: uy },
        ) if ux == uy => Some(x.cmp(y)),
        (
            Datetime {
                value: x,
                unit: ux,
                timezone: tx,
            },
            Datetime {
                value: y,
                unit: uy,
                timezone: ty,
            },
        ) if ux == uy && tx == ty => Some(x.cmp(y)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use helium::MinMaxValue;

    fn i32v(x: i32) -> MinMaxValue {
        MinMaxValue::I32(x)
    }

    #[test]
    fn eq_prunes_when_literal_outside_range() {
        // stripe [10, 20], col = 5  → cannot match
        assert!(!stripe_matches_range(
            Some(&i32v(10)),
            Some(&i32v(20)),
            CompareOp::Eq,
            &i32v(5)
        ));
        // stripe [10, 20], col = 25 → cannot match
        assert!(!stripe_matches_range(
            Some(&i32v(10)),
            Some(&i32v(20)),
            CompareOp::Eq,
            &i32v(25)
        ));
        // stripe [10, 20], col = 15 → might match
        assert!(stripe_matches_range(
            Some(&i32v(10)),
            Some(&i32v(20)),
            CompareOp::Eq,
            &i32v(15)
        ));
        // boundary: col = 10 and col = 20 both might match
        assert!(stripe_matches_range(
            Some(&i32v(10)),
            Some(&i32v(20)),
            CompareOp::Eq,
            &i32v(10)
        ));
        assert!(stripe_matches_range(
            Some(&i32v(10)),
            Some(&i32v(20)),
            CompareOp::Eq,
            &i32v(20)
        ));
    }

    #[test]
    fn range_ops_prune_correctly() {
        let lo = i32v(10);
        let hi = i32v(20);
        // col > 20 : max(20) not > 20 → prune
        assert!(!stripe_matches_range(Some(&lo), Some(&hi), CompareOp::Gt, &i32v(20)));
        // col > 19 : max(20) > 19 → keep
        assert!(stripe_matches_range(Some(&lo), Some(&hi), CompareOp::Gt, &i32v(19)));
        // col >= 21 : max(20) not >= 21 → prune
        assert!(!stripe_matches_range(Some(&lo), Some(&hi), CompareOp::Ge, &i32v(21)));
        // col < 10 : min(10) not < 10 → prune
        assert!(!stripe_matches_range(Some(&lo), Some(&hi), CompareOp::Lt, &i32v(10)));
        // col < 11 : min(10) < 11 → keep
        assert!(stripe_matches_range(Some(&lo), Some(&hi), CompareOp::Lt, &i32v(11)));
        // col <= 9 : min(10) not <= 9 → prune
        assert!(!stripe_matches_range(Some(&lo), Some(&hi), CompareOp::Le, &i32v(9)));
    }

    #[test]
    fn missing_stats_or_type_mismatch_keeps_stripe() {
        // No stats → keep.
        assert!(stripe_matches_range(None, None, CompareOp::Eq, &i32v(5)));
        // Type mismatch (I64 literal vs I32 stats) → keep (conservative).
        assert!(stripe_matches_range(
            Some(&i32v(10)),
            Some(&i32v(20)),
            CompareOp::Eq,
            &MinMaxValue::I64(5)
        ));
    }

    #[test]
    fn string_range_pruning() {
        let lo = MinMaxValue::Utf8("apple".into());
        let hi = MinMaxValue::Utf8("mango".into());
        // col = "zebra" outside ["apple","mango"] → prune
        assert!(!stripe_matches_range(
            Some(&lo),
            Some(&hi),
            CompareOp::Eq,
            &MinMaxValue::Utf8("zebra".into())
        ));
        // col = "kiwi" inside → keep
        assert!(stripe_matches_range(
            Some(&lo),
            Some(&hi),
            CompareOp::Eq,
            &MinMaxValue::Utf8("kiwi".into())
        ));
    }
}
