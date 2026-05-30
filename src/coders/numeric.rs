//! Shared helpers for integer-typed coders.

/// Dispatch an integer-only, type-preserving transform across every supported
/// `ColumnData` integer variant.
///
/// `$f` is applied to the inner `&[T]` and re-wrapped in the matching variant.
/// Non-integer inputs (or a `data_type` / payload mismatch) yield
/// [`HeliumError::RuntimeType`] tagged with `$coder`. Used by the `delta` and
/// `delta_of_delta` coders, whose `encode`/`decode` differ only in `$f`.
macro_rules! int_dispatch {
    ($coder:expr, $dt:expr, $input:expr, $f:expr) => {{
        use $crate::core::coder::{ColumnData, DataType};
        Ok(match ($dt, $input) {
            (DataType::I8, ColumnData::I8(xs)) => ColumnData::I8($f(xs)),
            (DataType::I16, ColumnData::I16(xs)) => ColumnData::I16($f(xs)),
            (DataType::I32, ColumnData::I32(xs)) => ColumnData::I32($f(xs)),
            (DataType::I64, ColumnData::I64(xs)) => ColumnData::I64($f(xs)),
            (DataType::U8, ColumnData::U8(xs)) => ColumnData::U8($f(xs)),
            (DataType::U16, ColumnData::U16(xs)) => ColumnData::U16($f(xs)),
            (DataType::U32, ColumnData::U32(xs)) => ColumnData::U32($f(xs)),
            (DataType::U64, ColumnData::U64(xs)) => ColumnData::U64($f(xs)),
            _ => {
                return Err($crate::core::error::HeliumError::RuntimeType {
                    coder: $coder.into(),
                    expected: $dt,
                });
            }
        })
    }};
}
pub(crate) use int_dispatch;

pub(crate) trait WrappingArith: Copy + PartialEq {
    fn wrap_sub(self, other: Self) -> Self;
    fn wrap_add(self, other: Self) -> Self;
    fn zero() -> Self;
}

macro_rules! impl_wrap_arith {
    ($($t:ty),* $(,)?) => {
        $(
            impl WrappingArith for $t {
                #[inline] fn wrap_sub(self, other: Self) -> Self { self.wrapping_sub(other) }
                #[inline] fn wrap_add(self, other: Self) -> Self { self.wrapping_add(other) }
                #[inline] fn zero() -> Self { 0 }
            }
        )*
    };
}

impl_wrap_arith!(i8, i16, i32, i64, u8, u16, u32, u64);

/// `out[i] = in[i] - in[i-1]` with `in[-1] := 0`, wrapping on underflow.
pub(crate) fn delta_encode<T: WrappingArith>(xs: &[T]) -> Vec<T> {
    let mut out = Vec::with_capacity(xs.len());
    let mut prev = T::zero();
    for &x in xs {
        out.push(x.wrap_sub(prev));
        prev = x;
    }
    out
}

pub(crate) fn delta_decode<T: WrappingArith>(xs: &[T]) -> Vec<T> {
    let mut out = Vec::with_capacity(xs.len());
    let mut acc = T::zero();
    for &d in xs {
        acc = acc.wrap_add(d);
        out.push(acc);
    }
    out
}
