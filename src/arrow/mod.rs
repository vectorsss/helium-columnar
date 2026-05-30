//! Arrow bridge — bidirectional conversion between Helium and Apache Arrow.
//!
//! This module is gated behind the `arrow` cargo feature. It provides:
//!
//! - [`crate::arrow::to_arrow_array`] — convert a [`crate::LogicalColumn`] to an Arrow `ArrayRef`
//! - [`crate::arrow::from_arrow_array`] — convert an Arrow `ArrayRef` to a [`crate::LogicalColumn`]
//! - [`crate::arrow::schema_to_arrow`] — convert a Helium [`crate::Schema`] to an `arrow::datatypes::Schema`
//! - [`crate::arrow::schema_from_arrow`] — convert an `arrow::datatypes::Schema` to a Helium [`crate::Schema`]
//!
//! # Type mapping
//!
//! | Helium `LogicalType` | Arrow `DataType` |
//! |---|---|
//! | `Primitive { I8 }` | `Int8` |
//! | `Primitive { I16 }` | `Int16` |
//! | `Primitive { I32 }` | `Int32` |
//! | `Primitive { I64 }` | `Int64` |
//! | `Primitive { U8 }` | `UInt8` |
//! | `Primitive { U16 }` | `UInt16` |
//! | `Primitive { U32 }` | `UInt32` |
//! | `Primitive { U64 }` | `UInt64` |
//! | `Primitive { F32 }` | `Float32` |
//! | `Primitive { F64 }` | `Float64` |
//! | `Utf8` | `Utf8` |
//! | `Binary` | `Binary` |
//! | `Nullable<T>` | same as T's mapping, with a null buffer set from the present mask |
//! | `List<T>` | `List<T-mapping>` (i32 offsets; errors if offsets exceed `i32::MAX`) |
//! | `Map<Utf8, V>` | `Map(Struct{ keys: Utf8, values: V })` |
//! | `Struct { fields }` | `Struct(fields)` |
//! | `Union { variants }` | `Union(variants, dense)` — dense union |
//! | `NullablePrim` (v2) | same as `Nullable<Primitive>` |
//! | `NullableUtf8` (v2) | same as `Nullable<Utf8>` |
//! | `NullableBinary` (v2) | same as `Nullable<Binary>` |
//! | `ArrayOf` (v2) | same as `List<Primitive>` |
//! | `ArrayOfUtf8` (v2) | same as `List<Utf8>` |
//! | `Dictionary { inner }` | `Dictionary(UInt32, inner)` |
//!
//! # Null handling
//!
//! Helium's `Nullable` wrapper stores values in **compact** form — only the
//! non-null rows appear in the inner column's data. Arrow uses **expanded**
//! representation with a null bitmap alongside a full-length value buffer.
//!
//! `to_arrow_array` expands the compact values to match Arrow's layout,
//! inserting placeholder zeros (or empty strings) at null positions.
//!
//! `from_arrow_array` on an array with a null buffer always produces the v3
//! `Nullable { present, value }` form (never v2 `NullablePrim` etc.). The
//! inner `value` is **compact** — only valid rows extracted.
//!
//! # Inverse direction note
//!
//! `from_arrow_array` always produces **v3-shaped** `LogicalColumn` variants
//! (`Nullable`, `List`, etc.) — never v2 variants. v2 variants are only
//! produced by legacy Helium file readers; new conversions always use v3.

pub mod from_arrow;
pub mod schema;
pub mod to_arrow;

pub use from_arrow::from_arrow_array;
pub use schema::{schema_from_arrow, schema_to_arrow};
pub use to_arrow::to_arrow_array;
