//! Schema — logical columns + physical decomposition.
//!
//! `ColumnSpec::encodings` has one `Vec<CoderSpec>` per entry returned by
//! [`LogicalType::expected_encodings_len`]:
//!
//! - Leaf types: one per physical leaf.
//! - `Struct`: **empty** — all leaf encodings live in [`FieldSpec::encodings`].
//! - `List { inner }`: 1 (offsets) + `inner.expected_encodings_len()`.
//! - `Map { key, value }`: 1 (offsets) + key + value.
//! - `Nullable { inner }`: 1 (present bitmap) + `inner.expected_encodings_len()`.
//! - `Union { variants }`: 1 (tag) + sum of each variant's `expected_encodings_len()`.
//! - `Dictionary { inner }`: `inner.expected_encodings_len()` + 1 (indices leaf).
//! - `Decimal128`: 2 — `[high: I64, low: I64]`.
//! - `Date { Days }`: 1 — `[values: I32]`.
//! - `Date { Millis }`: 1 — `[values: I64]`.
//! - `Datetime`: 1 — `[values: I64]`.

use std::collections::{HashMap, HashSet};
use std::hash::Hash;

use serde::{Deserialize, Serialize};

use super::coder::{ColumnData, DataType};
use super::error::{HeliumError, Result};
use super::pipeline::Pipeline;
use super::registry::{CoderRegistry, CoderSpec};

// ---------------------------------------------------------------------------
// DateUnit / TimeUnit
// ---------------------------------------------------------------------------

/// Unit for [`LogicalType::Date`] values.
///
/// The `"kind"` discriminant values (`Days`, `Millis`) and the physical integer
/// backing are **wire-format-frozen** — they appear in every `.he` schema JSON
/// that uses a `Date` column.  Do not rename or repurpose these variants;
/// add new variants with new names if a different backing is needed.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum DateUnit {
    /// `i32` days since 1970-01-01 (matches Arrow `Date32`).
    Days,
    /// `i64` milliseconds since 1970-01-01 00:00:00 UTC (matches Arrow `Date64`).
    Millis,
}

/// Unit for [`LogicalType::Datetime`] values.
///
/// The discriminant strings (`Seconds`, `Millis`, `Micros`, `Nanos`) are
/// **wire-format-frozen**.  All Datetime values are backed by an `i64`
/// regardless of unit.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum TimeUnit {
    /// `i64` seconds since Unix epoch.
    Seconds,
    /// `i64` milliseconds since Unix epoch.
    Millis,
    /// `i64` microseconds since Unix epoch.
    Micros,
    /// `i64` nanoseconds since Unix epoch.
    Nanos,
}

/// Current schema JSON version written by new files.
///
/// This is embedded in every `.he` schema JSON as `"version": 1`.  Readers
/// use this to gate forward-compatibility checks — a reader that knows only
/// version 1 rejects files with a higher version number.
pub const CURRENT_SCHEMA_VERSION: u32 = 1;

/// Maximum nesting depth for recursive type validation.
///
/// Container types (Struct, List, Map, Nullable, Union) at depth ≥ this value
/// are rejected by [`Schema::validate`]. Top-level column type is depth 0;
/// each container nesting level adds 1.
///
/// **Public** so cross-crate consumers (notably `helium-schema`'s Avro parser)
/// can fail fast at conversion time rather than waiting for
/// `Schema::validate()`. The constant is part of the public API contract —
/// changing the value is a semver-breaking change.
pub const MAX_NESTED_DEPTH: usize = 64;

fn default_version() -> u32 {
    CURRENT_SCHEMA_VERSION
}

/// Top-level schema: version + ordered list of column specifications.
///
/// Serializes to / from JSON and is embedded in every `.he` file header.
/// Build with [`Schema::new`] or deserialize from JSON with [`Schema::from_json`].
///
/// # Example
///
/// ```rust
/// use helium::{Schema, ColumnSpec, LogicalType, DataType, CoderSpec};
///
/// let schema = Schema::new(vec![
///     ColumnSpec {
///         name: "id".into(),
///         logical_type: LogicalType::Primitive { data_type: DataType::I64 },
///         encodings: vec![vec![CoderSpec::new("delta"), CoderSpec::new("leb128"), CoderSpec::new("zstd")]],
///     },
///     ColumnSpec {
///         name: "label".into(),
///         logical_type: LogicalType::Utf8,
///         encodings: vec![
///             vec![CoderSpec::new("delta"), CoderSpec::new("leb128"), CoderSpec::new("zstd")],
///             vec![CoderSpec::new("zstd")],
///         ],
///     },
/// ]);
/// assert_eq!(schema.columns.len(), 2);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Schema {
    /// Schema format version. Always [`CURRENT_SCHEMA_VERSION`] for newly
    /// written files; older files may have a lower value.
    #[serde(default = "default_version")]
    pub version: u32,
    /// Ordered list of top-level column specifications.
    pub columns: Vec<ColumnSpec>,
}

/// A single top-level logical column in a [`Schema`].
///
/// `encodings` has one `Vec<CoderSpec>` per physical leaf produced by
/// `logical_type.physical_fields()`. For `Struct` types, `encodings` is
/// empty — leaf encodings live in child [`FieldSpec::encodings`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ColumnSpec {
    /// Column name. Must be unique within a schema.
    pub name: String,
    /// Logical type — determines the physical leaf decomposition.
    pub logical_type: LogicalType,
    /// Encoding pipeline for each physical leaf, in declaration order.
    ///
    /// Must have the same length as `logical_type.physical_fields()`
    /// (or zero for `Struct` top-level columns).
    pub encodings: Vec<Vec<CoderSpec>>,
}

/// A named field inside a [`LogicalType::Struct`].
///
/// Mirrors [`ColumnSpec`] but lives inside a `Struct` type rather than at
/// the top level.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FieldSpec {
    /// Field name within the struct. Must be unique within the parent struct.
    pub name: String,
    /// Logical type of this field.
    pub logical_type: LogicalType,
    /// Encoding pipeline for each physical leaf of this field.
    pub encodings: Vec<Vec<CoderSpec>>,
}

/// Logical column type.
///
/// The `serde(tag = "kind")` shape is wire-format-visible and frozen.
/// The legacy flat variants (`ArrayOf`, `ArrayOfUtf8`, `Nullable*`) are kept for
/// read compatibility; new writers use `List`, `Map`, `Nullable`, `Union`, etc.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LogicalType {
    /// A single primitive scalar column (integer or float).
    ///
    /// Physical: one leaf `values: data_type`.
    Primitive {
        /// The physical scalar type (`I8`..`U64`, `F32`, `F64`).
        data_type: DataType,
    },
    /// Variable-length UTF-8 strings.
    ///
    /// Physical: `[offsets: U32, data: Bytes]`.
    Utf8,
    /// Variable-length binary blobs.
    ///
    /// Physical: `[offsets: U32, data: Bytes]`.
    Binary,
    /// Legacy flat variant — readable; new writer uses `List`.
    ArrayOf {
        /// Element type of the array.
        data_type: DataType,
    },
    /// Legacy flat variant — readable; new writer uses `List(Utf8)`.
    ArrayOfUtf8,
    /// Legacy flat variant — readable; new writer uses `Nullable(Primitive(T))`.
    NullablePrim {
        /// The primitive element type.
        data_type: DataType,
    },
    /// Legacy flat variant — readable; new writer uses `Nullable(Utf8)`.
    NullableUtf8,
    /// Legacy flat variant — readable; new writer uses `Nullable(Binary)`.
    NullableBinary,
    /// Composite type. `ColumnSpec::encodings` must be empty; leaf encodings
    /// live in child `FieldSpec::encodings`.
    Struct {
        /// Ordered list of named fields in this struct.
        fields: Vec<FieldSpec>,
    },
    /// Recursive list. Physical: `offsets: U32` + inner leaves (prefix `"item."`).
    List {
        /// Element type of the list.
        inner: Box<LogicalType>,
    },
    /// Key→value map with shared offsets. Key restricted to Primitive/Utf8/Binary.
    /// Physical: `offsets: U32` + key leaves (`"key."`) + value leaves (`"value."`).
    Map {
        /// Key type (must be `Primitive`, `Utf8`, or `Binary`).
        key: Box<LogicalType>,
        /// Value type (any `LogicalType`).
        value: Box<LogicalType>,
    },
    /// Nullable wrapper.
    ///
    /// Physical: `present: U8` (1 = non-null, 0 = null) + inner leaves
    /// (`"item."` prefix). Inner values are stored in **compact** form —
    /// only non-null rows appear in the inner column's storage.
    Nullable {
        /// The inner (non-null) element type.
        inner: Box<LogicalType>,
    },
    /// Tagged union. Each row holds exactly one variant's value, selected by
    /// `tag: U8` (0..variants.len()-1). Must have 1..=255 variants.
    ///
    /// Physical layout: `tag: U8` followed by the concatenated physical leaves
    /// of each variant in declaration order, each prefixed `"v_{name}."`.
    /// Each variant's data is **compacted** — only rows where
    /// `tag == variant_index` appear in that variant's storage.
    ///
    /// **Wire-format commitment**: on-disk JSON is
    /// `{"kind":"union","variants":[["name0",{...type...}],["name1",{...}],...]}`.
    /// Variant names are positional and **frozen** alongside the variant index —
    /// renaming a variant in a new schema produces an incompatible wire format.
    ///
    /// **Avro `["null", T]`** is canonicalized to `Nullable(T)` at import time
    /// and never represented as a `Union` (enforced in `helium-schema`).
    Union {
        /// Named variants: `(variant_name, type)` pairs.  Names are positional
        /// and wire-format-frozen once any `.he` file is written.
        variants: Vec<(String, LogicalType)>,
    },

    /// Dictionary-encoded column: `indices` reference distinct values held in a
    /// dictionary column of `inner` type.
    ///
    /// The recursive dictionary-encoded column type —
    /// `inner` may be any `LogicalType`.
    ///
    /// Physical layout: `inner.physical_fields()` first (verbatim, in order),
    /// then one trailing `indices: U32` leaf.
    ///
    /// **Wire-format-frozen**: on-disk JSON `{"kind":"dictionary","inner":{...}}`.
    /// The discriminant `"dictionary"` and the field name `"inner"` must never
    /// change — they appear in every `.he` schema JSON that uses this type.
    Dictionary {
        /// The type of the dictionary entries (distinct values).
        inner: Box<LogicalType>,
    },

    // -----------------------------------------------------------------------
    // Semantic type extensions (recursive vocabulary — not new physical types;
    // each decomposes into existing `DataType` leaves so the codec is
    // unchanged.  The `"kind"` discriminant values, inner field names, and
    // enum member names for `DateUnit` / `TimeUnit` are wire-format-frozen
    // the same way coder IDs are — never rename or repurpose them.
    // -----------------------------------------------------------------------
    /// Fixed-precision decimal backed by a 128-bit integer split into two
    /// `i64` leaves (`high` = upper 64 bits, `low` = lower 64 bits, both
    /// as signed two's-complement with the high word carrying the sign).
    ///
    /// `precision` is the total number of significant decimal digits (1..=38).
    /// `scale` is the number of digits to the right of the decimal point
    /// (0..=precision).
    ///
    /// **Wire-format-frozen**: on-disk JSON
    /// `{"kind":"decimal128","precision":N,"scale":M}`.
    /// The discriminant `"decimal128"` (snake_case via `rename_all`) and field
    /// names `"precision"`, `"scale"` must never change.
    ///
    /// Physical leaves: `[high: I64, low: I64]`.
    Decimal128 {
        /// Total number of significant decimal digits (1..=38).
        precision: u8,
        /// Number of digits to the right of the decimal point (0..=precision).
        scale: u8,
    },

    /// Calendar date with no time-of-day component.
    ///
    /// `unit` selects the integer backing:
    /// - `Days` → `i32` days since 1970-01-01 (Arrow `Date32`)
    /// - `Millis` → `i64` milliseconds since 1970-01-01 (Arrow `Date64`)
    ///
    /// **Wire-format-frozen**: on-disk JSON `{"kind":"Date","unit":"Days"}` etc.
    ///
    /// Physical leaves: `[values: I32]` for `Days`, `[values: I64]` for `Millis`.
    Date {
        /// Integer backing (`Days` = `i32` since epoch, `Millis` = `i64` since epoch).
        unit: DateUnit,
    },

    /// Timestamp — date + time + optional timezone.
    ///
    /// `unit` gives the resolution (`Seconds` / `Millis` / `Micros` / `Nanos`).
    /// `timezone` is an optional IANA timezone name (`"UTC"`,
    /// `"America/Los_Angeles"`, …). `None` means the timezone is unspecified
    /// (local time or "wall clock" semantics — matching Arrow's semantics for
    /// timestamp columns without explicit timezone).
    ///
    /// **Wire-format-frozen**: on-disk JSON
    /// `{"kind":"Datetime","unit":"Millis","timezone":"UTC"}` etc.
    ///
    /// Physical leaf: `[values: I64]` (one `i64` per row regardless of unit).
    Datetime {
        /// Time resolution (`Seconds`, `Millis`, `Micros`, or `Nanos`).
        unit: TimeUnit,
        /// Optional IANA timezone name (`"UTC"`, `"America/New_York"`, …).
        /// `None` means unspecified / wall-clock time.
        timezone: Option<String>,
    },
}

/// One physical (leaf) column produced by decomposing a logical type.
///
/// Each [`LogicalType`] decomposes into an ordered list of `PhysicalField`s via
/// [`LogicalType::physical_fields`]. The `role` becomes the dotted column name
/// in the footer index (e.g., `"offsets"`, `"data"`, `"present"`, `"item.values"`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhysicalField {
    /// Dotted role name (e.g. `"offsets"`, `"item.values"`, `"key.data"`).
    pub role: String,
    /// Physical [`DataType`] for this leaf.
    pub data_type: DataType,
}

fn is_valid_map_key_type(lt: &LogicalType) -> bool {
    matches!(
        lt,
        LogicalType::Primitive { .. } | LogicalType::Utf8 | LogicalType::Binary
    )
}

impl LogicalType {
    /// Return the ordered list of physical storage fields this logical type
    /// decomposes into. Each field has a `role` label and a [`DataType`].
    ///
    /// The ordering is canonical and must not change — it matches the
    /// `encodings` slot ordering in [`ColumnSpec`] and [`FieldSpec`].
    pub fn physical_fields(&self) -> Vec<PhysicalField> {
        match self {
            Self::Primitive { data_type } => vec![PhysicalField {
                role: "values".to_string(),
                data_type: *data_type,
            }],
            Self::Utf8 | Self::Binary => vec![
                PhysicalField {
                    role: "offsets".to_string(),
                    data_type: DataType::U32,
                },
                PhysicalField {
                    role: "data".to_string(),
                    data_type: DataType::Bytes,
                },
            ],
            Self::ArrayOf { data_type } => vec![
                PhysicalField {
                    role: "offsets".to_string(),
                    data_type: DataType::U32,
                },
                PhysicalField {
                    role: "values".to_string(),
                    data_type: *data_type,
                },
            ],
            Self::ArrayOfUtf8 => vec![
                PhysicalField {
                    role: "outer_offsets".to_string(),
                    data_type: DataType::U32,
                },
                PhysicalField {
                    role: "inner_offsets".to_string(),
                    data_type: DataType::U32,
                },
                PhysicalField {
                    role: "data".to_string(),
                    data_type: DataType::Bytes,
                },
            ],
            Self::NullablePrim { data_type } => vec![
                PhysicalField {
                    role: "present".to_string(),
                    data_type: DataType::U8,
                },
                PhysicalField {
                    role: "values".to_string(),
                    data_type: *data_type,
                },
            ],
            Self::NullableUtf8 | Self::NullableBinary => vec![
                PhysicalField {
                    role: "present".to_string(),
                    data_type: DataType::U8,
                },
                PhysicalField {
                    role: "offsets".to_string(),
                    data_type: DataType::U32,
                },
                PhysicalField {
                    role: "data".to_string(),
                    data_type: DataType::Bytes,
                },
            ],
            Self::Struct { fields } => {
                let mut result = Vec::new();
                for field in fields {
                    for pf in field.logical_type.physical_fields() {
                        result.push(PhysicalField {
                            role: format!("{}.{}", field.name, pf.role),
                            data_type: pf.data_type,
                        });
                    }
                }
                result
            }
            Self::List { inner } => {
                let mut result = vec![PhysicalField {
                    role: "offsets".to_string(),
                    data_type: DataType::U32,
                }];
                for pf in inner.physical_fields() {
                    result.push(PhysicalField {
                        role: format!("item.{}", pf.role),
                        data_type: pf.data_type,
                    });
                }
                result
            }
            Self::Map { key, value } => {
                let mut result = vec![PhysicalField {
                    role: "offsets".to_string(),
                    data_type: DataType::U32,
                }];
                for pf in key.physical_fields() {
                    result.push(PhysicalField {
                        role: format!("key.{}", pf.role),
                        data_type: pf.data_type,
                    });
                }
                for pf in value.physical_fields() {
                    result.push(PhysicalField {
                        role: format!("value.{}", pf.role),
                        data_type: pf.data_type,
                    });
                }
                result
            }
            Self::Nullable { inner } => {
                let mut result = vec![PhysicalField {
                    role: "present".to_string(),
                    data_type: DataType::U8,
                }];
                for pf in inner.physical_fields() {
                    result.push(PhysicalField {
                        role: format!("item.{}", pf.role),
                        data_type: pf.data_type,
                    });
                }
                result
            }
            Self::Union { variants } => {
                let mut result = vec![PhysicalField {
                    role: "tag".to_string(),
                    data_type: DataType::U8,
                }];
                for (v_name, v_lt) in variants {
                    for pf in v_lt.physical_fields() {
                        result.push(PhysicalField {
                            role: format!("v_{}.{}", v_name, pf.role),
                            data_type: pf.data_type,
                        });
                    }
                }
                result
            }
            // Semantic type extensions — decompose into existing primitive leaves.
            Self::Decimal128 { .. } => vec![
                PhysicalField {
                    role: "high".to_string(),
                    data_type: DataType::I64,
                },
                PhysicalField {
                    role: "low".to_string(),
                    data_type: DataType::I64,
                },
            ],
            Self::Date {
                unit: DateUnit::Days,
            } => vec![PhysicalField {
                role: "values".to_string(),
                data_type: DataType::I32,
            }],
            Self::Date {
                unit: DateUnit::Millis,
            } => vec![PhysicalField {
                role: "values".to_string(),
                data_type: DataType::I64,
            }],
            Self::Datetime { .. } => vec![PhysicalField {
                role: "values".to_string(),
                data_type: DataType::I64,
            }],
            Self::Dictionary { inner } => {
                let mut result = inner.physical_fields();
                result.push(PhysicalField {
                    role: "indices".to_string(),
                    data_type: DataType::U32,
                });
                result
            }
        }
    }

    /// Expected number of entries in `ColumnSpec::encodings` for this type.
    ///
    /// - `Struct`: 0 (encodings in FieldSpec).
    /// - `List { inner }`: 1 + inner.
    /// - `Map { key, value }`: 1 + key + value.
    /// - `Nullable { inner }`: 1 + inner.
    /// - `Union { variants }`: 1 (tag) + sum of each variant's encoding count.
    /// - leaf types: `physical_fields().len()`.
    pub fn expected_encodings_len(&self) -> usize {
        match self {
            Self::Struct { .. } => 0,
            Self::List { inner } => 1 + inner.expected_encodings_len(),
            Self::Map { key, value } => {
                1 + key.expected_encodings_len() + value.expected_encodings_len()
            }
            Self::Nullable { inner } => 1 + inner.expected_encodings_len(),
            Self::Union { variants } => {
                1 + variants
                    .iter()
                    .map(|(_, lt)| lt.expected_encodings_len())
                    .sum::<usize>()
            }
            Self::Dictionary { inner } => inner.expected_encodings_len() + 1,
            _ => self.physical_fields().len(),
        }
    }
}

// ---------------------------------------------------------------------------
// FieldSpec constructors
// ---------------------------------------------------------------------------

impl FieldSpec {
    /// Construct a [`FieldSpec`] with an explicit encoding list.
    pub fn new(
        name: impl Into<String>,
        logical_type: LogicalType,
        encodings: Vec<Vec<CoderSpec>>,
    ) -> Self {
        Self {
            name: name.into(),
            logical_type,
            encodings,
        }
    }

    /// Construct a primitive field with a single pipeline.
    pub fn primitive(name: impl Into<String>, data_type: DataType, coders: Vec<CoderSpec>) -> Self {
        Self::new(name, LogicalType::Primitive { data_type }, vec![coders])
    }

    /// Construct a UTF-8 string field with separate `offsets` and `data` pipelines.
    pub fn utf8(name: impl Into<String>, offsets: Vec<CoderSpec>, data: Vec<CoderSpec>) -> Self {
        Self::new(name, LogicalType::Utf8, vec![offsets, data])
    }

    /// Construct a nested struct field.
    pub fn struct_field(name: impl Into<String>, fields: Vec<FieldSpec>) -> Self {
        Self::new(name, LogicalType::Struct { fields }, vec![])
    }

    /// Construct a list field with the given inner type and encoding list.
    pub fn list(
        name: impl Into<String>,
        inner: LogicalType,
        encodings: Vec<Vec<CoderSpec>>,
    ) -> Self {
        Self::new(
            name,
            LogicalType::List {
                inner: Box::new(inner),
            },
            encodings,
        )
    }

    /// Construct a map field with the given key/value types and encoding list.
    pub fn map(
        name: impl Into<String>,
        key: LogicalType,
        value: LogicalType,
        encodings: Vec<Vec<CoderSpec>>,
    ) -> Self {
        Self::new(
            name,
            LogicalType::Map {
                key: Box::new(key),
                value: Box::new(value),
            },
            encodings,
        )
    }

    /// Construct a nullable wrapper field.
    pub fn nullable(
        name: impl Into<String>,
        inner: LogicalType,
        encodings: Vec<Vec<CoderSpec>>,
    ) -> Self {
        Self::new(
            name,
            LogicalType::Nullable {
                inner: Box::new(inner),
            },
            encodings,
        )
    }
}

// ---------------------------------------------------------------------------
// ColumnSpec constructors + validation
// ---------------------------------------------------------------------------

impl ColumnSpec {
    /// Construct a [`ColumnSpec`] with an explicit encoding list.
    pub fn new(
        name: impl Into<String>,
        logical_type: LogicalType,
        encodings: Vec<Vec<CoderSpec>>,
    ) -> Self {
        Self {
            name: name.into(),
            logical_type,
            encodings,
        }
    }

    /// Construct a primitive column with a single pipeline.
    pub fn primitive(name: impl Into<String>, data_type: DataType, coders: Vec<CoderSpec>) -> Self {
        Self::new(name, LogicalType::Primitive { data_type }, vec![coders])
    }

    /// Construct a UTF-8 string column.
    pub fn utf8(name: impl Into<String>, offsets: Vec<CoderSpec>, data: Vec<CoderSpec>) -> Self {
        Self::new(name, LogicalType::Utf8, vec![offsets, data])
    }

    /// Construct a binary column.
    pub fn binary(name: impl Into<String>, offsets: Vec<CoderSpec>, data: Vec<CoderSpec>) -> Self {
        Self::new(name, LogicalType::Binary, vec![offsets, data])
    }

    /// Construct an `ArrayOf` (legacy flat) primitive array column.
    pub fn array_of(
        name: impl Into<String>,
        data_type: DataType,
        offsets: Vec<CoderSpec>,
        values: Vec<CoderSpec>,
    ) -> Self {
        Self::new(
            name,
            LogicalType::ArrayOf { data_type },
            vec![offsets, values],
        )
    }

    /// Construct a `NullablePrim` (legacy flat) nullable primitive column.
    pub fn nullable_prim(
        name: impl Into<String>,
        data_type: DataType,
        present: Vec<CoderSpec>,
        values: Vec<CoderSpec>,
    ) -> Self {
        Self::new(
            name,
            LogicalType::NullablePrim { data_type },
            vec![present, values],
        )
    }

    /// Construct a `NullableUtf8` (legacy flat) nullable string column.
    pub fn nullable_utf8(
        name: impl Into<String>,
        present: Vec<CoderSpec>,
        offsets: Vec<CoderSpec>,
        data: Vec<CoderSpec>,
    ) -> Self {
        Self::new(
            name,
            LogicalType::NullableUtf8,
            vec![present, offsets, data],
        )
    }

    /// Construct a `NullableBinary` (legacy flat) nullable binary column.
    pub fn nullable_binary(
        name: impl Into<String>,
        present: Vec<CoderSpec>,
        offsets: Vec<CoderSpec>,
        data: Vec<CoderSpec>,
    ) -> Self {
        Self::new(
            name,
            LogicalType::NullableBinary,
            vec![present, offsets, data],
        )
    }

    /// Construct an `ArrayOfUtf8` (legacy flat) string-array column.
    pub fn array_of_utf8(
        name: impl Into<String>,
        outer_offsets: Vec<CoderSpec>,
        inner_offsets: Vec<CoderSpec>,
        data: Vec<CoderSpec>,
    ) -> Self {
        Self::new(
            name,
            LogicalType::ArrayOfUtf8,
            vec![outer_offsets, inner_offsets, data],
        )
    }

    /// Construct a struct column (leaf encodings live in child `FieldSpec`s).
    pub fn struct_col(name: impl Into<String>, fields: Vec<FieldSpec>) -> Self {
        Self::new(name, LogicalType::Struct { fields }, vec![])
    }

    /// Construct a list column.
    pub fn list(
        name: impl Into<String>,
        inner: LogicalType,
        encodings: Vec<Vec<CoderSpec>>,
    ) -> Self {
        Self::new(
            name,
            LogicalType::List {
                inner: Box::new(inner),
            },
            encodings,
        )
    }

    /// Construct a map column.
    pub fn map(
        name: impl Into<String>,
        key: LogicalType,
        value: LogicalType,
        encodings: Vec<Vec<CoderSpec>>,
    ) -> Self {
        Self::new(
            name,
            LogicalType::Map {
                key: Box::new(key),
                value: Box::new(value),
            },
            encodings,
        )
    }

    /// Construct a nullable column.
    pub fn nullable(
        name: impl Into<String>,
        inner: LogicalType,
        encodings: Vec<Vec<CoderSpec>>,
    ) -> Self {
        Self::new(
            name,
            LogicalType::Nullable {
                inner: Box::new(inner),
            },
            encodings,
        )
    }

    /// Union column. `encodings[0]` = tag pipeline (U8 input);
    /// `encodings[1..]` = per-variant coders in declaration order (Struct variants
    /// contribute 0 entries; their leaf encodings live in FieldSpec).
    pub fn union(
        name: impl Into<String>,
        variants: Vec<(String, LogicalType)>,
        encodings: Vec<Vec<CoderSpec>>,
    ) -> Self {
        Self::new(name, LogicalType::Union { variants }, encodings)
    }

    /// Decimal128 column with two I64 leaves (high + low).
    /// `high_enc` and `low_enc` are the pipelines for the two physical leaves.
    pub fn decimal128(
        name: impl Into<String>,
        precision: u8,
        scale: u8,
        high_enc: Vec<CoderSpec>,
        low_enc: Vec<CoderSpec>,
    ) -> Self {
        Self::new(
            name,
            LogicalType::Decimal128 { precision, scale },
            vec![high_enc, low_enc],
        )
    }

    /// Date column backed by `i32` days (matches Arrow `Date32`).
    pub fn date32(name: impl Into<String>, enc: Vec<CoderSpec>) -> Self {
        Self::new(
            name,
            LogicalType::Date {
                unit: DateUnit::Days,
            },
            vec![enc],
        )
    }

    /// Date column backed by `i64` milliseconds (matches Arrow `Date64`).
    pub fn date64(name: impl Into<String>, enc: Vec<CoderSpec>) -> Self {
        Self::new(
            name,
            LogicalType::Date {
                unit: DateUnit::Millis,
            },
            vec![enc],
        )
    }

    /// Datetime column. Unit and timezone live in the schema type.
    pub fn datetime(
        name: impl Into<String>,
        unit: TimeUnit,
        timezone: Option<String>,
        enc: Vec<CoderSpec>,
    ) -> Self {
        Self::new(name, LogicalType::Datetime { unit, timezone }, vec![enc])
    }

    fn validate(&self) -> Result<()> {
        if self.name.is_empty() {
            return Err(HeliumError::Schema {
                column: String::new(),
                reason: "empty column name".into(),
            });
        }
        match &self.logical_type {
            LogicalType::Struct { .. } => {
                if !self.encodings.is_empty() {
                    return Err(HeliumError::Schema {
                        column: self.name.clone(),
                        reason: "Struct column must have empty top-level encodings; \
                                 leaf encodings live in FieldSpec"
                            .into(),
                    });
                }
                validate_nested_type(&self.logical_type, &self.name, 0)?;
            }
            lt => {
                if let LogicalType::Map { key, .. } = lt
                    && !is_valid_map_key_type(key)
                {
                    return Err(HeliumError::Schema {
                        column: self.name.clone(),
                        reason: "Map key type must be Primitive, Utf8, or Binary".into(),
                    });
                }
                let expected = lt.expected_encodings_len();
                if self.encodings.len() != expected {
                    return Err(HeliumError::Schema {
                        column: self.name.clone(),
                        reason: format!(
                            "logical type expects {expected} encoding vectors, got {}",
                            self.encodings.len()
                        ),
                    });
                }
                validate_nested_type(lt, &self.name, 0)?;
            }
        }
        Ok(())
    }
}

fn validate_field_spec(field: &FieldSpec, column_name: &str, depth: usize) -> Result<()> {
    if field.name.is_empty() {
        return Err(HeliumError::Schema {
            column: column_name.into(),
            reason: "struct field name cannot be empty".into(),
        });
    }
    match &field.logical_type {
        LogicalType::Struct { .. } => {
            if !field.encodings.is_empty() {
                return Err(HeliumError::Schema {
                    column: column_name.into(),
                    reason: format!(
                        "Struct field '{}' must have empty encodings; \
                         nested field encodings live in child FieldSpecs",
                        field.name
                    ),
                });
            }
            validate_nested_type(&field.logical_type, column_name, depth)?;
        }
        lt => {
            if let LogicalType::Map { key, .. } = lt
                && !is_valid_map_key_type(key)
            {
                return Err(HeliumError::Schema {
                    column: column_name.into(),
                    reason: format!(
                        "field '{}': Map key type must be Primitive, Utf8, or Binary",
                        field.name
                    ),
                });
            }
            let expected = lt.expected_encodings_len();
            if field.encodings.len() != expected {
                return Err(HeliumError::Schema {
                    column: column_name.into(),
                    reason: format!(
                        "field '{}' expects {expected} encoding vectors, got {}",
                        field.name,
                        field.encodings.len()
                    ),
                });
            }
            validate_nested_type(lt, column_name, depth)?;
        }
    }
    Ok(())
}

/// Recursively validate the structure of a logical type.
///
/// `depth` is the depth of `lt` in the type tree (0 = top-level column type).
/// Container types at depth ≥ [`MAX_NESTED_DEPTH`] are rejected.
///
/// Single authoritative location for:
/// - Struct/Union duplicate-name checking and field recursion
/// - Depth cap enforcement
/// - Map key-type restriction for nested Maps
/// - Union: zero-variant and >255-variant rejection, empty variant names
fn validate_nested_type(lt: &LogicalType, column_name: &str, depth: usize) -> Result<()> {
    match lt {
        LogicalType::Struct { fields } => {
            if depth >= MAX_NESTED_DEPTH {
                return depth_error(column_name);
            }
            let mut seen: HashSet<&str> = HashSet::new();
            for field in fields {
                if !seen.insert(field.name.as_str()) {
                    return Err(HeliumError::Schema {
                        column: column_name.into(),
                        reason: format!("duplicate field name '{}'", field.name),
                    });
                }
                validate_field_spec(field, column_name, depth + 1)?;
            }
        }
        LogicalType::List { inner } => {
            if depth >= MAX_NESTED_DEPTH {
                return depth_error(column_name);
            }
            validate_nested_type(inner, column_name, depth + 1)?;
        }
        LogicalType::Map { key, value } => {
            if depth >= MAX_NESTED_DEPTH {
                return depth_error(column_name);
            }
            if !is_valid_map_key_type(key) {
                return Err(HeliumError::Schema {
                    column: column_name.into(),
                    reason: "Map key type must be Primitive, Utf8, or Binary; \
                             got a non-scalar key type"
                        .to_string(),
                });
            }
            validate_nested_type(key, column_name, depth + 1)?;
            validate_nested_type(value, column_name, depth + 1)?;
        }
        LogicalType::Nullable { inner } => {
            if depth >= MAX_NESTED_DEPTH {
                return depth_error(column_name);
            }
            validate_nested_type(inner, column_name, depth + 1)?;
        }
        LogicalType::Dictionary { inner } => {
            if depth >= MAX_NESTED_DEPTH {
                return depth_error(column_name);
            }
            validate_nested_type(inner, column_name, depth + 1)?;
        }
        LogicalType::Union { variants } => {
            if depth >= MAX_NESTED_DEPTH {
                return depth_error(column_name);
            }
            if variants.is_empty() {
                return Err(HeliumError::Schema {
                    column: column_name.into(),
                    reason: "Union must have at least one variant".into(),
                });
            }
            if variants.len() > 255 {
                return Err(HeliumError::Schema {
                    column: column_name.into(),
                    reason: format!(
                        "Union has {} variants; maximum is 255 (tag fits in U8)",
                        variants.len()
                    ),
                });
            }
            let mut seen: HashSet<&str> = HashSet::new();
            for (v_name, v_lt) in variants {
                if v_name.is_empty() {
                    return Err(HeliumError::Schema {
                        column: column_name.into(),
                        reason: "Union variant name cannot be empty".into(),
                    });
                }
                if !seen.insert(v_name.as_str()) {
                    return Err(HeliumError::Schema {
                        column: column_name.into(),
                        reason: format!("duplicate Union variant name '{v_name}'"),
                    });
                }
                validate_nested_type(v_lt, column_name, depth + 1)?;
            }
        }
        _ => {} // leaf types have no nested structure to validate
    }
    Ok(())
}

fn depth_error(column_name: &str) -> Result<()> {
    Err(HeliumError::Schema {
        column: column_name.into(),
        reason: format!("nested type exceeds maximum depth of {MAX_NESTED_DEPTH}"),
    })
}

impl Schema {
    /// Create a new schema from an ordered list of column specs.
    pub fn new(columns: Vec<ColumnSpec>) -> Self {
        Self {
            version: CURRENT_SCHEMA_VERSION,
            columns,
        }
    }

    /// Serialize this schema to JSON bytes (embedded in the `.he` header).
    pub fn to_json(&self) -> Result<Vec<u8>> {
        Ok(serde_json::to_vec(self)?)
    }

    /// Deserialize a schema from JSON bytes and validate it.
    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        let schema: Schema = serde_json::from_slice(bytes)?;
        if schema.version != CURRENT_SCHEMA_VERSION {
            return Err(HeliumError::Format(format!(
                "unsupported schema version {} (this build supports {CURRENT_SCHEMA_VERSION})",
                schema.version
            )));
        }
        schema.validate()?;
        Ok(schema)
    }

    /// Validate all column specs and check for duplicate column names.
    pub fn validate(&self) -> Result<()> {
        let mut seen = HashSet::new();
        for col in &self.columns {
            col.validate()?;
            if !seen.insert(col.name.as_str()) {
                return Err(HeliumError::Schema {
                    column: col.name.clone(),
                    reason: "duplicate column name".into(),
                });
            }
        }
        Ok(())
    }

    /// Look up a column spec by name, returning `None` if not found.
    pub fn column(&self, name: &str) -> Option<&ColumnSpec> {
        self.columns.iter().find(|c| c.name == name)
    }

    /// Look up a column's 0-based position in the schema, returning `None`
    /// if not found.
    pub fn column_index(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|c| c.name == name)
    }

    /// Build a new schema containing only `names`, in the given order,
    /// preserving each column's logical type and encodings.
    ///
    /// This is the schema-level half of a column projection ("slice"): pair it
    /// with [`crate::HeliumReader::project_to`] to write a subset `.he` file.
    /// Errors if any name is absent from this schema, or appears more than once.
    pub fn project(&self, names: &[&str]) -> Result<Schema> {
        let mut seen = HashSet::new();
        let mut columns = Vec::with_capacity(names.len());
        for &name in names {
            if !seen.insert(name) {
                return Err(HeliumError::Schema {
                    column: name.into(),
                    reason: "column requested more than once in projection".into(),
                });
            }
            let spec = self.column(name).ok_or_else(|| HeliumError::Schema {
                column: name.into(),
                reason: "column not in schema".into(),
            })?;
            columns.push(spec.clone());
        }
        Ok(Schema::new(columns))
    }

    /// Build the physical [`Pipeline`]s for a single column using `registry`.
    ///
    /// Returns an error if the column is not in the schema or if any coder
    /// in the column's encodings list is unregistered or misconfigured.
    pub fn resolve_column(&self, name: &str, registry: &CoderRegistry) -> Result<Vec<Pipeline>> {
        let spec = self.column(name).ok_or_else(|| HeliumError::Schema {
            column: name.into(),
            reason: "column not found".into(),
        })?;
        build_physical_pipelines(spec, registry)
    }

    /// Build physical [`Pipeline`]s for every column. Returns one
    /// `Vec<Pipeline>` per column, in schema order.
    pub fn resolve_all(&self, registry: &CoderRegistry) -> Result<Vec<Vec<Pipeline>>> {
        self.columns
            .iter()
            .map(|spec| build_physical_pipelines(spec, registry))
            .collect()
    }
}

fn build_physical_pipelines(spec: &ColumnSpec, registry: &CoderRegistry) -> Result<Vec<Pipeline>> {
    build_pipelines_for_type(&spec.logical_type, &spec.encodings, &spec.name, registry)
}

fn build_pipelines_for_type(
    lt: &LogicalType,
    encodings: &[Vec<CoderSpec>],
    column_name: &str,
    registry: &CoderRegistry,
) -> Result<Vec<Pipeline>> {
    match lt {
        LogicalType::Struct { fields } => {
            let mut result = Vec::new();
            for field_spec in fields {
                result.extend(build_pipelines_for_type(
                    &field_spec.logical_type,
                    &field_spec.encodings,
                    column_name,
                    registry,
                )?);
            }
            Ok(result)
        }
        LogicalType::List { inner } => {
            if encodings.is_empty() {
                return Err(HeliumError::Schema {
                    column: column_name.into(),
                    reason: "List requires at least one encoding vector (for offsets)".into(),
                });
            }
            let mut result = vec![build_single_pipeline(
                DataType::U32,
                &encodings[0],
                column_name,
                registry,
            )?];
            result.extend(build_pipelines_for_type(
                inner,
                &encodings[1..],
                column_name,
                registry,
            )?);
            Ok(result)
        }
        LogicalType::Map { key, value } => {
            if encodings.is_empty() {
                return Err(HeliumError::Schema {
                    column: column_name.into(),
                    reason: "Map requires at least one encoding vector (for offsets)".into(),
                });
            }
            let key_enc_len = key.expected_encodings_len();
            let mut result = vec![build_single_pipeline(
                DataType::U32,
                &encodings[0],
                column_name,
                registry,
            )?];
            result.extend(build_pipelines_for_type(
                key,
                &encodings[1..1 + key_enc_len],
                column_name,
                registry,
            )?);
            result.extend(build_pipelines_for_type(
                value,
                &encodings[1 + key_enc_len..],
                column_name,
                registry,
            )?);
            Ok(result)
        }
        LogicalType::Nullable { inner } => {
            if encodings.is_empty() {
                return Err(HeliumError::Schema {
                    column: column_name.into(),
                    reason: "Nullable requires at least one encoding vector (for present bitmap)"
                        .into(),
                });
            }
            let mut result = vec![build_single_pipeline(
                DataType::U8,
                &encodings[0],
                column_name,
                registry,
            )?];
            result.extend(build_pipelines_for_type(
                inner,
                &encodings[1..],
                column_name,
                registry,
            )?);
            Ok(result)
        }
        LogicalType::Union { variants } => {
            if encodings.is_empty() {
                return Err(HeliumError::Schema {
                    column: column_name.into(),
                    reason: "Union requires at least one encoding vector (for tag)".into(),
                });
            }
            let mut result = vec![build_single_pipeline(
                DataType::U8,
                &encodings[0],
                column_name,
                registry,
            )?];
            let mut offset = 1usize;
            for (_, v_lt) in variants {
                let v_enc_len = v_lt.expected_encodings_len();
                result.extend(build_pipelines_for_type(
                    v_lt,
                    &encodings[offset..offset + v_enc_len],
                    column_name,
                    registry,
                )?);
                offset += v_enc_len;
            }
            Ok(result)
        }
        LogicalType::Dictionary { inner } => {
            let inner_enc_len = inner.expected_encodings_len();
            if encodings.len() < inner_enc_len + 1 {
                return Err(HeliumError::Schema {
                    column: column_name.into(),
                    reason: format!(
                        "Dictionary requires {} encoding vectors (inner) + 1 (indices), got {}",
                        inner_enc_len,
                        encodings.len()
                    ),
                });
            }
            let mut result = build_pipelines_for_type(
                inner,
                &encodings[..inner_enc_len],
                column_name,
                registry,
            )?;
            result.push(build_single_pipeline(
                DataType::U32,
                &encodings[inner_enc_len],
                column_name,
                registry,
            )?);
            Ok(result)
        }
        _ => {
            let fields = lt.physical_fields();
            fields
                .iter()
                .zip(encodings.iter())
                .map(|(field, coders)| {
                    build_single_pipeline(field.data_type, coders, column_name, registry).map_err(
                        |e| match e {
                            HeliumError::Schema { reason, .. } => HeliumError::Schema {
                                column: column_name.into(),
                                reason: format!("physical field '{}': {reason}", field.role),
                            },
                            other => other,
                        },
                    )
                })
                .collect()
        }
    }
}

fn build_single_pipeline(
    input_type: DataType,
    coders: &[CoderSpec],
    column_name: &str,
    registry: &CoderRegistry,
) -> Result<Pipeline> {
    let mut stages = Vec::with_capacity(coders.len());
    let mut current_type = input_type;
    for coder_spec in coders {
        let stage = registry
            .build(coder_spec, current_type)
            .map_err(|e| HeliumError::Schema {
                column: column_name.into(),
                reason: e.to_string(),
            })?;
        current_type = stage.produced_output_type();
        stages.push(stage);
    }
    Pipeline::new(input_type, stages).map_err(|e| HeliumError::Schema {
        column: column_name.into(),
        reason: e.to_string(),
    })
}

// ---------------------------------------------------------------------------
// LogicalColumn
// ---------------------------------------------------------------------------

/// A full logical column's worth of values.
#[derive(Debug, Clone, PartialEq)]
pub enum LogicalColumn {
    /// A flat column of primitive (non-string, non-binary) values.
    Primitive(ColumnData),
    /// A column of UTF-8 strings.
    Utf8(Vec<String>),
    /// A column of arbitrary byte blobs.
    Binary(Vec<Vec<u8>>),
    /// Variable-length array of primitives. `offsets` has length N + 1;
    /// `offsets[i]..offsets[i+1]` is row i's element range in `values`.
    ArrayOf {
        /// Row start/end offsets into `values`. Length is N + 1.
        offsets: Vec<u32>,
        /// Flat buffer of all element values across all rows.
        values: ColumnData,
    },
    /// Variable-length array of UTF-8 strings. `offsets` has length N + 1;
    /// `offsets[i]..offsets[i+1]` is row i's element range in `strings`.
    ArrayOfUtf8 {
        /// Row start/end offsets into `strings`. Length is N + 1.
        offsets: Vec<u32>,
        /// Flat buffer of all string elements across all rows.
        strings: Vec<String>,
    },
    /// Nullable column of primitives. `present[i]` is `true` if row i is
    /// non-null; `values` contains **all** rows (nulls have unspecified value).
    NullablePrim {
        /// `true` = non-null for each row.
        present: Vec<bool>,
        /// All row values, including those for null rows (unspecified).
        values: ColumnData,
    },
    /// Nullable column of UTF-8 strings.
    NullableUtf8 {
        /// `true` = non-null for each row.
        present: Vec<bool>,
        /// All row strings, including those for null rows (unspecified).
        strings: Vec<String>,
    },
    /// Nullable column of byte blobs.
    NullableBinary {
        /// `true` = non-null for each row.
        present: Vec<bool>,
        /// All row blobs, including those for null rows (unspecified).
        blobs: Vec<Vec<u8>>,
    },
    /// Composite struct value. `fields` is a `(name, value)` list in
    /// schema-declaration order; all fields must have the same row count.
    Struct {
        /// Ordered `(field_name, column)` pairs matching the schema's field
        /// declarations.
        fields: Vec<(String, LogicalColumn)>,
    },
    /// Variable-length list. `offsets` has length N + 1; `offsets[i]..offsets[i+1]`
    /// is row i's element range in `values`.
    List {
        /// Row start/end offsets into `values`. Length is N + 1.
        offsets: Vec<u32>,
        /// Flat buffer of all list elements across all rows.
        values: Box<LogicalColumn>,
    },
    /// Key→value map with shared offsets. `offsets` has length N + 1; `keys` and
    /// `values` are 1:1 aligned by flat index within each row's range.
    Map {
        /// Row start/end offsets. Length is N + 1.
        offsets: Vec<u32>,
        /// Flat buffer of all map keys across all rows.
        keys: Box<LogicalColumn>,
        /// Flat buffer of all map values across all rows (parallel to `keys`).
        values: Box<LogicalColumn>,
    },
    /// Nullable wrapper. `present` has length = logical row count;
    /// `value` holds **only the non-null rows**, compacted.
    Nullable {
        /// `true` = non-null for each logical row.
        present: Vec<bool>,
        /// Compacted non-null values — length equals the number of `true`
        /// entries in `present`.
        value: Box<LogicalColumn>,
    },
    /// Tagged union. `tags[i]` is the variant index (0-based) for row i.
    /// `variants[v]` holds **only the rows where `tags[j] == v`**, compacted.
    /// `variants[v].row_count()` must equal the number of `v`-valued tags.
    Union {
        /// Per-row variant index (0-based).
        tags: Vec<u8>,
        /// `(variant_name, compacted_column)` pairs in declaration order.
        variants: Vec<(String, LogicalColumn)>,
    },

    // Semantic type extensions — mirrors the `LogicalType` additions.
    /// Fixed-precision 128-bit decimal. Each `i128` value is the integer
    /// representation (value × 10^(-scale)).  On write this is split into
    /// two `i64` physical leaves (high 64 bits + low 64 bits); on read they
    /// are re-assembled.
    Decimal128 {
        /// The decimal values as unscaled `i128` integers.
        values: Vec<i128>,
    },

    /// Calendar date backed by `i32` days since 1970-01-01 (`Date { unit: Days }`).
    Date32 {
        /// Days since Unix epoch (1970-01-01) for each row.
        values: Vec<i32>,
    },

    /// Calendar date backed by `i64` milliseconds since 1970-01-01 (`Date { unit: Millis }`).
    Date64 {
        /// Milliseconds since Unix epoch for each row.
        values: Vec<i64>,
    },

    /// Timestamp; unit and timezone live in the schema's [`LogicalType::Datetime`].
    Datetime {
        /// Timestamp ticks (unit from schema) since Unix epoch for each row.
        values: Vec<i64>,
    },

    /// Dictionary-encoded column. `dictionary` holds the distinct values
    /// (`dictionary.row_count()` == cardinality); `indices[i]` is the 0-based
    /// dictionary slot for logical row `i`. `indices.len()` == logical row count.
    ///
    /// This is the recursive dictionary-encoded column variant.
    Dictionary {
        /// The distinct values in insertion order.
        dictionary: Box<LogicalColumn>,
        /// Per-row index into `dictionary` (0-based). Length equals the logical row count.
        indices: Vec<u32>,
    },
}

impl LogicalColumn {
    /// Return the number of logical rows this column holds.
    pub fn row_count(&self) -> usize {
        match self {
            Self::Primitive(d) => d.len(),
            Self::Utf8(v) => v.len(),
            Self::Binary(v) => v.len(),
            Self::ArrayOf { offsets, .. } | Self::ArrayOfUtf8 { offsets, .. } => {
                offsets.len().saturating_sub(1)
            }
            Self::NullablePrim { present, .. }
            | Self::NullableUtf8 { present, .. }
            | Self::NullableBinary { present, .. } => present.len(),
            Self::Dictionary { indices, .. } => indices.len(),
            Self::Struct { fields } => fields.first().map_or(0, |(_, col)| col.row_count()),
            Self::List { offsets, .. } | Self::Map { offsets, .. } => {
                offsets.len().saturating_sub(1)
            }
            Self::Nullable { present, .. } => present.len(),
            Self::Union { tags, .. } => tags.len(),
            // Semantic types
            Self::Decimal128 { values } => values.len(),
            Self::Date32 { values } => values.len(),
            Self::Date64 { values } => values.len(),
            Self::Datetime { values } => values.len(),
        }
    }

    /// Dictionary-encode a `Vec<String>` into a `Dictionary { inner: Utf8 }` column.
    pub fn dict_encode_utf8(values: Vec<String>) -> Self {
        let mut dictionary: Vec<String> = Vec::new();
        let mut map: HashMap<String, u32> = HashMap::new();
        let mut indices = Vec::with_capacity(values.len());
        for v in values {
            let idx = match map.get(&v) {
                Some(&i) => i,
                None => {
                    let i = dictionary.len() as u32;
                    map.insert(v.clone(), i);
                    dictionary.push(v);
                    i
                }
            };
            indices.push(idx);
        }
        Self::Dictionary {
            dictionary: Box::new(LogicalColumn::Utf8(dictionary)),
            indices,
        }
    }

    /// Dictionary-encode a primitive `ColumnData` (integer types only) into a
    /// `Dictionary { inner: Primitive(T) }` column. Returns an error for float or bytes inputs.
    pub fn dict_encode_primitive(values: ColumnData) -> Result<Self> {
        fn encode<T: Copy + Eq + Hash>(xs: Vec<T>) -> (Vec<T>, Vec<u32>) {
            let mut dict = Vec::new();
            let mut map: HashMap<T, u32> = HashMap::new();
            let mut indices = Vec::with_capacity(xs.len());
            for v in xs {
                let idx = match map.get(&v) {
                    Some(&i) => i,
                    None => {
                        let i = dict.len() as u32;
                        map.insert(v, i);
                        dict.push(v);
                        i
                    }
                };
                indices.push(idx);
            }
            (dict, indices)
        }
        let (dictionary_data, indices) = match values {
            ColumnData::I8(xs) => {
                let (d, i) = encode(xs);
                (ColumnData::I8(d), i)
            }
            ColumnData::I16(xs) => {
                let (d, i) = encode(xs);
                (ColumnData::I16(d), i)
            }
            ColumnData::I32(xs) => {
                let (d, i) = encode(xs);
                (ColumnData::I32(d), i)
            }
            ColumnData::I64(xs) => {
                let (d, i) = encode(xs);
                (ColumnData::I64(d), i)
            }
            ColumnData::U8(xs) => {
                let (d, i) = encode(xs);
                (ColumnData::U8(d), i)
            }
            ColumnData::U16(xs) => {
                let (d, i) = encode(xs);
                (ColumnData::U16(d), i)
            }
            ColumnData::U32(xs) => {
                let (d, i) = encode(xs);
                (ColumnData::U32(d), i)
            }
            ColumnData::U64(xs) => {
                let (d, i) = encode(xs);
                (ColumnData::U64(d), i)
            }
            ColumnData::F32(_) | ColumnData::F64(_) | ColumnData::Bytes(_) => {
                return Err(schema_error(
                    "dict_encode_primitive only supports integer types",
                ));
            }
        };
        Ok(Self::Dictionary {
            dictionary: Box::new(LogicalColumn::Primitive(dictionary_data)),
            indices,
        })
    }

    /// Expand a `Dictionary { inner: Utf8 }` column back into a plain `Vec<String>`.
    ///
    /// Returns an error if `self` is not `Dictionary { inner: Utf8 }` or if
    /// any index is out of bounds.
    pub fn materialize_dict_utf8(self) -> Result<Vec<String>> {
        match self {
            Self::Dictionary {
                dictionary,
                indices,
            } => match *dictionary {
                Self::Utf8(strings) => indices
                    .iter()
                    .map(|&i| {
                        strings
                            .get(i as usize)
                            .cloned()
                            .ok_or_else(|| schema_error(&format!("dict index {i} out of range")))
                    })
                    .collect(),
                _ => Err(schema_error(
                    "materialize_dict_utf8: dictionary inner is not Utf8",
                )),
            },
            _ => Err(schema_error("not a Dictionary{Utf8} column")),
        }
    }

    /// Decompose this logical column into its ordered list of physical
    /// `ColumnData` parts, validated against `lt`.
    ///
    /// The physical part ordering matches [`LogicalType::physical_fields`].
    pub fn decompose(self, lt: &LogicalType) -> Result<Vec<ColumnData>> {
        match (self, lt) {
            (Self::Primitive(d), LogicalType::Primitive { data_type }) => {
                if d.data_type() != *data_type {
                    return Err(schema_error(&format!(
                        "primitive type {:?} does not match schema {:?}",
                        d.data_type(),
                        data_type
                    )));
                }
                Ok(vec![d])
            }
            (Self::Utf8(strings), LogicalType::Utf8) => {
                let (off, data) = flatten_strings(&strings)?;
                Ok(vec![ColumnData::U32(off), ColumnData::Bytes(data)])
            }
            (Self::Binary(blobs), LogicalType::Binary) => {
                let (off, data) = flatten_binary(&blobs)?;
                Ok(vec![ColumnData::U32(off), ColumnData::Bytes(data)])
            }
            (Self::ArrayOf { offsets, values }, LogicalType::ArrayOf { data_type }) => {
                if values.data_type() != *data_type {
                    return Err(schema_error("array values type mismatch"));
                }
                validate_offsets(&offsets, values.len())?;
                Ok(vec![ColumnData::U32(offsets), values])
            }
            (Self::ArrayOfUtf8 { offsets, strings }, LogicalType::ArrayOfUtf8) => {
                validate_offsets(&offsets, strings.len())?;
                let (inner_off, data) = flatten_strings(&strings)?;
                Ok(vec![
                    ColumnData::U32(offsets),
                    ColumnData::U32(inner_off),
                    ColumnData::Bytes(data),
                ])
            }
            (Self::NullablePrim { present, values }, LogicalType::NullablePrim { data_type }) => {
                if values.data_type() != *data_type {
                    return Err(schema_error("nullable prim type mismatch"));
                }
                let pc = present.iter().filter(|&&p| p).count();
                if pc != values.len() {
                    return Err(schema_error("present/values count mismatch"));
                }
                Ok(vec![ColumnData::U8(present_to_bytes(&present)), values])
            }
            (Self::NullableUtf8 { present, strings }, LogicalType::NullableUtf8) => {
                let pc = present.iter().filter(|&&p| p).count();
                if pc != strings.len() {
                    return Err(schema_error("present/strings count mismatch"));
                }
                let (off, data) = flatten_strings(&strings)?;
                Ok(vec![
                    ColumnData::U8(present_to_bytes(&present)),
                    ColumnData::U32(off),
                    ColumnData::Bytes(data),
                ])
            }
            (Self::NullableBinary { present, blobs }, LogicalType::NullableBinary) => {
                let pc = present.iter().filter(|&&p| p).count();
                if pc != blobs.len() {
                    return Err(schema_error("present/blobs count mismatch"));
                }
                let (off, data) = flatten_binary(&blobs)?;
                Ok(vec![
                    ColumnData::U8(present_to_bytes(&present)),
                    ColumnData::U32(off),
                    ColumnData::Bytes(data),
                ])
            }
            (
                Self::Struct { fields: col_fields },
                LogicalType::Struct {
                    fields: spec_fields,
                },
            ) => {
                if col_fields.len() != spec_fields.len() {
                    return Err(schema_error("Struct field count mismatch"));
                }
                let mut result = Vec::new();
                for ((name, col), spec_field) in col_fields.into_iter().zip(spec_fields.iter()) {
                    if name != spec_field.name {
                        return Err(schema_error(&format!(
                            "Struct field name mismatch: expected '{}', got '{}'",
                            spec_field.name, name
                        )));
                    }
                    result.extend(col.decompose(&spec_field.logical_type).map_err(
                        |e| match e {
                            HeliumError::Schema { reason, .. } => {
                                schema_error(&format!("in struct field '{name}': {reason}"))
                            }
                            other => other,
                        },
                    )?);
                }
                Ok(result)
            }
            (Self::List { offsets, values }, LogicalType::List { inner }) => {
                validate_offsets(&offsets, values.row_count())?;
                let mut result = vec![ColumnData::U32(offsets)];
                result.extend(values.decompose(inner).map_err(|e| match e {
                    HeliumError::Schema { reason, .. } => {
                        schema_error(&format!("in List inner: {reason}"))
                    }
                    other => other,
                })?);
                Ok(result)
            }
            (
                Self::Map {
                    offsets,
                    keys,
                    values,
                },
                LogicalType::Map { key, value },
            ) => {
                let entry_count = keys.row_count();
                if values.row_count() != entry_count {
                    return Err(schema_error(&format!(
                        "Map keys has {entry_count} entries but values has {} entries",
                        values.row_count()
                    )));
                }
                validate_offsets(&offsets, entry_count)?;
                let mut result = vec![ColumnData::U32(offsets)];
                result.extend(keys.decompose(key).map_err(|e| match e {
                    HeliumError::Schema { reason, .. } => {
                        schema_error(&format!("in Map key: {reason}"))
                    }
                    other => other,
                })?);
                result.extend(values.decompose(value).map_err(|e| match e {
                    HeliumError::Schema { reason, .. } => {
                        schema_error(&format!("in Map value: {reason}"))
                    }
                    other => other,
                })?);
                Ok(result)
            }
            (Self::Nullable { present, value }, LogicalType::Nullable { inner }) => {
                let present_count = present.iter().filter(|&&p| p).count();
                if present_count != value.row_count() {
                    return Err(schema_error(&format!(
                        "Nullable present count ({present_count}) != value row count ({})",
                        value.row_count()
                    )));
                }
                let mut result = vec![ColumnData::U8(present_to_bytes(&present))];
                result.extend(value.decompose(inner).map_err(|e| match e {
                    HeliumError::Schema { reason, .. } => {
                        schema_error(&format!("in Nullable inner: {reason}"))
                    }
                    other => other,
                })?);
                Ok(result)
            }
            (
                Self::Union {
                    tags,
                    variants: col_variants,
                },
                LogicalType::Union {
                    variants: spec_variants,
                },
            ) => {
                if col_variants.len() != spec_variants.len() {
                    return Err(schema_error(&format!(
                        "Union data has {} variants but schema has {}",
                        col_variants.len(),
                        spec_variants.len()
                    )));
                }
                let n = spec_variants.len();
                // Validate tag range
                if let Some(&bad) = tags.iter().find(|&&t| (t as usize) >= n) {
                    return Err(schema_error(&format!(
                        "Union tag {bad} out of range for {n}-variant union"
                    )));
                }
                // Validate compaction counts and name alignment
                for (v_idx, ((col_name, col_data), (spec_name, _))) in
                    col_variants.iter().zip(spec_variants.iter()).enumerate()
                {
                    if col_name != spec_name {
                        return Err(schema_error(&format!(
                            "Union variant name mismatch at index {v_idx}: \
                             expected '{spec_name}', got '{col_name}'"
                        )));
                    }
                    let expected_count = tags.iter().filter(|&&t| t as usize == v_idx).count();
                    if col_data.row_count() != expected_count {
                        return Err(schema_error(&format!(
                            "Union variant '{col_name}' has {} rows \
                             but {expected_count} tags point to it",
                            col_data.row_count()
                        )));
                    }
                }
                let mut result = vec![ColumnData::U8(tags)];
                for ((_, col_data), (_, spec_lt)) in
                    col_variants.into_iter().zip(spec_variants.iter())
                {
                    result.extend(col_data.decompose(spec_lt).map_err(|e| match e {
                        HeliumError::Schema { reason, .. } => {
                            schema_error(&format!("in Union variant: {reason}"))
                        }
                        other => other,
                    })?);
                }
                Ok(result)
            }
            // --- Decimal128 ---
            (Self::Decimal128 { values }, LogicalType::Decimal128 { .. }) => {
                let highs: Vec<i64> = values.iter().map(|&v| (v >> 64) as i64).collect();
                let lows: Vec<i64> = values.iter().map(|&v| v as i64).collect();
                Ok(vec![ColumnData::I64(highs), ColumnData::I64(lows)])
            }
            // --- Date (Days → I32) ---
            (
                Self::Date32 { values },
                LogicalType::Date {
                    unit: DateUnit::Days,
                },
            ) => Ok(vec![ColumnData::I32(values)]),
            // --- Date (Millis → I64) ---
            (
                Self::Date64 { values },
                LogicalType::Date {
                    unit: DateUnit::Millis,
                },
            ) => Ok(vec![ColumnData::I64(values)]),
            // --- Datetime → I64 ---
            (Self::Datetime { values }, LogicalType::Datetime { .. }) => {
                Ok(vec![ColumnData::I64(values)])
            }
            // --- Dictionary ---
            (
                Self::Dictionary {
                    dictionary,
                    indices,
                },
                LogicalType::Dictionary { inner },
            ) => {
                validate_dict_indices(&indices, dictionary.row_count())?;
                let mut result = dictionary.decompose(inner).map_err(|e| match e {
                    HeliumError::Schema { reason, .. } => {
                        schema_error(&format!("in Dictionary inner: {reason}"))
                    }
                    other => other,
                })?;
                result.push(ColumnData::U32(indices));
                Ok(result)
            }
            (_, _) => Err(schema_error("logical column shape does not match schema")),
        }
    }

    /// Reconstruct a `LogicalColumn` from its physical `parts`, given the
    /// logical type `lt` and the original `row_count`.
    ///
    /// This is the inverse of [`decompose`]; `parts` must be in the same order
    /// as [`LogicalType::physical_fields`] returns.
    ///
    /// [`decompose`]: LogicalColumn::decompose
    pub fn compose(parts: Vec<ColumnData>, lt: &LogicalType, row_count: usize) -> Result<Self> {
        /// Pull the next physical part from the iterator, returning a typed
        /// error if the iterator is unexpectedly exhausted.
        fn take_part(it: &mut std::vec::IntoIter<ColumnData>) -> Result<ColumnData> {
            it.next()
                .ok_or_else(|| schema_error("internal: missing physical column part"))
        }

        let fields = lt.physical_fields();
        if parts.len() != fields.len() {
            return Err(schema_error(&format!(
                "expected {} physical parts for {lt:?}, got {}",
                fields.len(),
                parts.len()
            )));
        }
        match lt {
            LogicalType::Primitive { data_type } => {
                let mut it = parts.into_iter();
                let v = take_part(&mut it)?;
                check_type(&v, *data_type)?;
                Ok(Self::Primitive(v))
            }
            LogicalType::Utf8 => {
                let mut it = parts.into_iter();
                let off = expect_u32(take_part(&mut it)?)?;
                let data = expect_bytes(take_part(&mut it)?)?;
                Ok(Self::Utf8(unflatten_strings(&off, &data, row_count, true)?))
            }
            LogicalType::Binary => {
                let mut it = parts.into_iter();
                let off = expect_u32(take_part(&mut it)?)?;
                let data = expect_bytes(take_part(&mut it)?)?;
                Ok(Self::Binary(unflatten_binary(&off, &data, row_count)?))
            }
            LogicalType::ArrayOf { data_type } => {
                let mut it = parts.into_iter();
                let offsets = expect_u32(take_part(&mut it)?)?;
                let values = take_part(&mut it)?;
                check_type(&values, *data_type)?;
                if offsets.len() != row_count + 1 {
                    return Err(schema_error("array offsets length mismatch"));
                }
                Ok(Self::ArrayOf { offsets, values })
            }
            LogicalType::ArrayOfUtf8 => {
                let mut it = parts.into_iter();
                let outer = expect_u32(take_part(&mut it)?)?;
                let inner = expect_u32(take_part(&mut it)?)?;
                let data = expect_bytes(take_part(&mut it)?)?;
                if outer.len() != row_count + 1 {
                    return Err(schema_error("outer offsets mismatch"));
                }
                let total = *outer.last().unwrap_or(&0) as usize;
                let strings = unflatten_strings(&inner, &data, total, true)?;
                Ok(Self::ArrayOfUtf8 {
                    offsets: outer,
                    strings,
                })
            }
            LogicalType::NullablePrim { data_type } => {
                let mut it = parts.into_iter();
                let pb = expect_u8(take_part(&mut it)?)?;
                let values = take_part(&mut it)?;
                check_type(&values, *data_type)?;
                let present = bytes_to_present(&pb, row_count)?;
                Ok(Self::NullablePrim { present, values })
            }
            LogicalType::NullableUtf8 => {
                let mut it = parts.into_iter();
                let pb = expect_u8(take_part(&mut it)?)?;
                let off = expect_u32(take_part(&mut it)?)?;
                let data = expect_bytes(take_part(&mut it)?)?;
                let present = bytes_to_present(&pb, row_count)?;
                let pc = present.iter().filter(|&&p| p).count();
                Ok(Self::NullableUtf8 {
                    present,
                    strings: unflatten_strings(&off, &data, pc, true)?,
                })
            }
            LogicalType::NullableBinary => {
                let mut it = parts.into_iter();
                let pb = expect_u8(take_part(&mut it)?)?;
                let off = expect_u32(take_part(&mut it)?)?;
                let data = expect_bytes(take_part(&mut it)?)?;
                let present = bytes_to_present(&pb, row_count)?;
                let pc = present.iter().filter(|&&p| p).count();
                Ok(Self::NullableBinary {
                    present,
                    blobs: unflatten_binary(&off, &data, pc)?,
                })
            }
            LogicalType::Struct {
                fields: spec_fields,
            } => {
                let mut it = parts.into_iter();
                let mut result_fields = Vec::with_capacity(spec_fields.len());
                for spec_field in spec_fields {
                    let leaf_count = spec_field.logical_type.physical_fields().len();
                    let field_parts: Vec<ColumnData> = it.by_ref().take(leaf_count).collect();
                    if field_parts.len() != leaf_count {
                        return Err(schema_error(&format!(
                            "insufficient parts for struct field '{}'",
                            spec_field.name
                        )));
                    }
                    let col =
                        LogicalColumn::compose(field_parts, &spec_field.logical_type, row_count)
                            .map_err(|e| match e {
                                HeliumError::Schema { reason, .. } => schema_error(&format!(
                                    "in struct field '{}': {reason}",
                                    spec_field.name
                                )),
                                other => other,
                            })?;
                    result_fields.push((spec_field.name.clone(), col));
                }
                Ok(Self::Struct {
                    fields: result_fields,
                })
            }
            LogicalType::List { inner } => {
                let mut it = parts.into_iter();
                // ok_or_else instead of unwrap since parts.len() was only checked against
                // physical_fields().len(), and we need to be safe if inner is empty.
                let offsets = expect_u32(
                    it.next()
                        .ok_or_else(|| schema_error("List offsets missing"))?,
                )?;
                if offsets.len() != row_count + 1 {
                    return Err(schema_error("List offsets length mismatch"));
                }
                let inner_count = offsets.last().copied().unwrap_or(0) as usize;
                let inner_parts: Vec<ColumnData> = it.collect();
                let values = LogicalColumn::compose(inner_parts, inner, inner_count).map_err(
                    |e| match e {
                        HeliumError::Schema { reason, .. } => {
                            schema_error(&format!("in List inner: {reason}"))
                        }
                        other => other,
                    },
                )?;
                Ok(Self::List {
                    offsets,
                    values: Box::new(values),
                })
            }
            LogicalType::Map { key, value } => {
                let mut it = parts.into_iter();
                let offsets = expect_u32(
                    it.next()
                        .ok_or_else(|| schema_error("Map offsets missing"))?,
                )?;
                if offsets.len() != row_count + 1 {
                    return Err(schema_error("Map offsets length mismatch"));
                }
                let entry_count = offsets.last().copied().unwrap_or(0) as usize;
                let key_leaf_count = key.physical_fields().len();
                let key_parts: Vec<ColumnData> = it.by_ref().take(key_leaf_count).collect();
                if key_parts.len() != key_leaf_count {
                    return Err(schema_error("insufficient parts for Map keys"));
                }
                let keys_col =
                    LogicalColumn::compose(key_parts, key, entry_count).map_err(|e| match e {
                        HeliumError::Schema { reason, .. } => {
                            schema_error(&format!("in Map key: {reason}"))
                        }
                        other => other,
                    })?;
                let value_parts: Vec<ColumnData> = it.collect();
                let values_col = LogicalColumn::compose(value_parts, value, entry_count).map_err(
                    |e| match e {
                        HeliumError::Schema { reason, .. } => {
                            schema_error(&format!("in Map value: {reason}"))
                        }
                        other => other,
                    },
                )?;
                Ok(Self::Map {
                    offsets,
                    keys: Box::new(keys_col),
                    values: Box::new(values_col),
                })
            }
            LogicalType::Nullable { inner } => {
                let mut it = parts.into_iter();
                let present_bytes = expect_u8(
                    it.next()
                        .ok_or_else(|| schema_error("Nullable present missing"))?,
                )?;
                let present = bytes_to_present(&present_bytes, row_count)?;
                let present_count = present.iter().filter(|&&p| p).count();
                let inner_parts: Vec<ColumnData> = it.collect();
                let value = LogicalColumn::compose(inner_parts, inner, present_count).map_err(
                    |e| match e {
                        HeliumError::Schema { reason, .. } => {
                            schema_error(&format!("in Nullable inner: {reason}"))
                        }
                        other => other,
                    },
                )?;
                Ok(Self::Nullable {
                    present,
                    value: Box::new(value),
                })
            }
            LogicalType::Union {
                variants: spec_variants,
            } => {
                let mut it = parts.into_iter();
                let tag_bytes =
                    expect_u8(it.next().ok_or_else(|| schema_error("Union tag missing"))?)?;
                if tag_bytes.len() != row_count {
                    return Err(schema_error("Union tag length mismatch"));
                }
                let n_variants = spec_variants.len();
                if let Some(&bad) = tag_bytes.iter().find(|&&t| (t as usize) >= n_variants) {
                    return Err(schema_error(&format!(
                        "Union tag {bad} out of range for {n_variants}-variant union"
                    )));
                }
                let mut result_variants = Vec::with_capacity(n_variants);
                for (v_idx, (v_name, v_lt)) in spec_variants.iter().enumerate() {
                    let v_count = tag_bytes.iter().filter(|&&t| t as usize == v_idx).count();
                    let leaf_count = v_lt.physical_fields().len();
                    let v_parts: Vec<ColumnData> = it.by_ref().take(leaf_count).collect();
                    if v_parts.len() != leaf_count {
                        return Err(schema_error(&format!(
                            "insufficient physical parts for union variant '{v_name}'"
                        )));
                    }
                    let v_col =
                        LogicalColumn::compose(v_parts, v_lt, v_count).map_err(|e| match e {
                            HeliumError::Schema { reason, .. } => {
                                schema_error(&format!("in union variant '{v_name}': {reason}"))
                            }
                            other => other,
                        })?;
                    result_variants.push((v_name.clone(), v_col));
                }
                Ok(Self::Union {
                    tags: tag_bytes,
                    variants: result_variants,
                })
            }
            // --- Decimal128 ---
            LogicalType::Decimal128 { .. } => {
                let mut it = parts.into_iter();
                let highs = expect_i64(take_part(&mut it)?)?;
                let lows = expect_i64(take_part(&mut it)?)?;
                if highs.len() != row_count || lows.len() != row_count {
                    return Err(schema_error("Decimal128 high/low leaf length mismatch"));
                }
                let values: Vec<i128> = highs
                    .into_iter()
                    .zip(lows)
                    .map(|(h, l)| ((h as i128) << 64) | (l as u64 as i128))
                    .collect();
                Ok(Self::Decimal128 { values })
            }
            // --- Date (Days) ---
            LogicalType::Date {
                unit: DateUnit::Days,
            } => {
                let mut it = parts.into_iter();
                let vals = expect_i32(take_part(&mut it)?)?;
                if vals.len() != row_count {
                    return Err(schema_error("Date32 leaf length mismatch"));
                }
                Ok(Self::Date32 { values: vals })
            }
            // --- Date (Millis) ---
            LogicalType::Date {
                unit: DateUnit::Millis,
            } => {
                let mut it = parts.into_iter();
                let vals = expect_i64(take_part(&mut it)?)?;
                if vals.len() != row_count {
                    return Err(schema_error("Date64 leaf length mismatch"));
                }
                Ok(Self::Date64 { values: vals })
            }
            // --- Datetime ---
            LogicalType::Datetime { .. } => {
                let mut it = parts.into_iter();
                let vals = expect_i64(take_part(&mut it)?)?;
                if vals.len() != row_count {
                    return Err(schema_error("Datetime leaf length mismatch"));
                }
                Ok(Self::Datetime { values: vals })
            }
            // --- Dictionary ---
            LogicalType::Dictionary { inner } => {
                let inner_leaf_count = inner.physical_fields().len();
                // Last part is the indices leaf.
                let mut it = parts.into_iter();
                let inner_parts: Vec<ColumnData> = it.by_ref().take(inner_leaf_count).collect();
                let indices_cd = it
                    .next()
                    .ok_or_else(|| schema_error("Dictionary indices missing"))?;
                let indices = expect_u32(indices_cd)?;
                if indices.len() != row_count {
                    return Err(schema_error(&format!(
                        "Dictionary indices length {} != row_count {row_count}",
                        indices.len()
                    )));
                }
                if inner_parts.len() != inner_leaf_count {
                    return Err(schema_error("insufficient parts for Dictionary inner type"));
                }
                // Recover the cardinality (the dictionary's own row count) from the inner leaves.
                let cardinality = parts_logical_row_count(&inner_parts, inner)?;
                let dictionary = LogicalColumn::compose(inner_parts, inner, cardinality).map_err(
                    |e| match e {
                        HeliumError::Schema { reason, .. } => {
                            schema_error(&format!("in Dictionary inner: {reason}"))
                        }
                        other => other,
                    },
                )?;
                // Validate that every index is within cardinality bounds.
                validate_dict_indices(&indices, cardinality)?;
                Ok(Self::Dictionary {
                    dictionary: Box::new(dictionary),
                    indices,
                })
            }
        }
    }
}

/// Recover the number of logical rows that the physical `parts` represent for
/// `lt`, without having access to the original `row_count` parameter.
///
/// This is used in [`LogicalColumn::compose`] for `Dictionary` columns:
/// the dictionary's own row count (cardinality) differs from the logical row
/// count of the outer column, and must be reconstructed from the inner leaves.
fn parts_logical_row_count(parts: &[ColumnData], lt: &LogicalType) -> Result<usize> {
    if parts.is_empty() {
        return Err(schema_error(
            "parts_logical_row_count: empty parts for non-empty type",
        ));
    }
    match lt {
        // Primitive leaf: row count = length of the single leaf.
        LogicalType::Primitive { .. } | LogicalType::Date { .. } | LogicalType::Datetime { .. } => {
            Ok(parts[0].len())
        }
        // Decimal128 has two I64 leaves of equal length.
        LogicalType::Decimal128 { .. } => Ok(parts[0].len()),
        // Utf8 / Binary: first leaf is offsets; n rows → n+1 offsets.
        LogicalType::Utf8 | LogicalType::Binary => {
            let off = expect_u32(parts[0].clone())?;
            Ok(off.len().saturating_sub(1))
        }
        // Nullable: first leaf is the U8 present bitmap; one entry per row.
        LogicalType::Nullable { .. }
        | LogicalType::NullablePrim { .. }
        | LogicalType::NullableUtf8
        | LogicalType::NullableBinary => Ok(parts[0].len()),
        // List / Map: first leaf is offsets; n rows → n+1 offsets.
        LogicalType::List { .. } | LogicalType::Map { .. } => {
            let off = expect_u32(parts[0].clone())?;
            Ok(off.len().saturating_sub(1))
        }
        // ArrayOf: first leaf is offsets.
        LogicalType::ArrayOf { .. } => {
            let off = expect_u32(parts[0].clone())?;
            Ok(off.len().saturating_sub(1))
        }
        // ArrayOfUtf8: first leaf is outer offsets.
        LogicalType::ArrayOfUtf8 => {
            let off = expect_u32(parts[0].clone())?;
            Ok(off.len().saturating_sub(1))
        }
        // Struct: recurse on the first field's first leaf (all fields share row count).
        LogicalType::Struct { fields } => {
            if fields.is_empty() {
                return Ok(0);
            }
            let first_field_leaf_count = fields[0].logical_type.physical_fields().len();
            let field_parts = &parts[..first_field_leaf_count.min(parts.len())];
            parts_logical_row_count(field_parts, &fields[0].logical_type)
        }
        // Union: first leaf is the U8 tag; one entry per row.
        LogicalType::Union { .. } => Ok(parts[0].len()),
        // Dictionary{inner}: last leaf is the indices.
        LogicalType::Dictionary { .. } => {
            if parts.is_empty() {
                return Err(schema_error(
                    "parts_logical_row_count: Dictionary needs at least 1 part",
                ));
            }
            let indices = expect_u32(parts[parts.len() - 1].clone())?;
            Ok(indices.len())
        }
    }
}

fn validate_dict_indices(indices: &[u32], dict_len: usize) -> Result<()> {
    if let Some(&bad) = indices.iter().find(|&&i| (i as usize) >= dict_len) {
        return Err(schema_error(&format!(
            "dict index {bad} out of range for dictionary of size {dict_len}"
        )));
    }
    Ok(())
}

fn schema_error(msg: &str) -> HeliumError {
    HeliumError::Schema {
        column: "<unspecified>".into(),
        reason: msg.into(),
    }
}

fn flatten_strings(strings: &[String]) -> Result<(Vec<u32>, Vec<u8>)> {
    let mut offsets = Vec::with_capacity(strings.len() + 1);
    offsets.push(0u32);
    let mut data = Vec::new();
    for s in strings {
        data.extend_from_slice(s.as_bytes());
        let off =
            u32::try_from(data.len()).map_err(|_| schema_error("string data exceeds u32::MAX"))?;
        offsets.push(off);
    }
    Ok((offsets, data))
}

fn flatten_binary(blobs: &[Vec<u8>]) -> Result<(Vec<u32>, Vec<u8>)> {
    let mut offsets = Vec::with_capacity(blobs.len() + 1);
    offsets.push(0u32);
    let mut data = Vec::new();
    for b in blobs {
        data.extend_from_slice(b);
        let off =
            u32::try_from(data.len()).map_err(|_| schema_error("binary data exceeds u32::MAX"))?;
        offsets.push(off);
    }
    Ok((offsets, data))
}

fn unflatten_strings(
    offsets: &[u32],
    data: &[u8],
    row_count: usize,
    validate_utf8: bool,
) -> Result<Vec<String>> {
    if offsets.len() != row_count + 1 {
        return Err(schema_error(&format!(
            "string offsets must have {} entries, got {}",
            row_count + 1,
            offsets.len()
        )));
    }
    let mut out = Vec::with_capacity(row_count);
    for i in 0..row_count {
        let start = offsets[i] as usize;
        let end = offsets[i + 1] as usize;
        if end < start || end > data.len() {
            return Err(schema_error("string offsets out of range"));
        }
        let slice = &data[start..end];
        if validate_utf8 {
            out.push(
                std::str::from_utf8(slice)
                    .map_err(|e| schema_error(&e.to_string()))?
                    .to_string(),
            );
        } else {
            out.push(String::from_utf8_lossy(slice).into_owned());
        }
    }
    Ok(out)
}

fn unflatten_binary(offsets: &[u32], data: &[u8], row_count: usize) -> Result<Vec<Vec<u8>>> {
    if offsets.len() != row_count + 1 {
        return Err(schema_error("binary offsets length mismatch"));
    }
    let mut out = Vec::with_capacity(row_count);
    for i in 0..row_count {
        let start = offsets[i] as usize;
        let end = offsets[i + 1] as usize;
        if end < start || end > data.len() {
            return Err(schema_error("binary offsets out of range"));
        }
        out.push(data[start..end].to_vec());
    }
    Ok(out)
}

fn validate_offsets(offsets: &[u32], values_len: usize) -> Result<()> {
    let Some(&last) = offsets.last() else {
        return Err(schema_error("offsets cannot be empty"));
    };
    if offsets[0] != 0 {
        return Err(schema_error("offsets[0] must be 0"));
    }
    if (last as usize) != values_len {
        return Err(schema_error(&format!(
            "last offset {last} does not match values len {values_len}",
        )));
    }
    for w in offsets.windows(2) {
        if w[1] < w[0] {
            return Err(schema_error("offsets must be non-decreasing"));
        }
    }
    Ok(())
}

fn present_to_bytes(present: &[bool]) -> Vec<u8> {
    present.iter().map(|&b| b as u8).collect()
}

fn bytes_to_present(bytes: &[u8], row_count: usize) -> Result<Vec<bool>> {
    if bytes.len() != row_count {
        return Err(schema_error(&format!(
            "present column length {} != row_count {row_count}",
            bytes.len()
        )));
    }
    Ok(bytes.iter().map(|&b| b != 0).collect())
}

fn check_type(d: &ColumnData, expected: DataType) -> Result<()> {
    if d.data_type() != expected {
        return Err(schema_error(&format!(
            "expected physical type {expected:?}, got {:?}",
            d.data_type()
        )));
    }
    Ok(())
}

fn expect_u32(d: ColumnData) -> Result<Vec<u32>> {
    if let ColumnData::U32(v) = d {
        Ok(v)
    } else {
        Err(schema_error(&format!(
            "expected U32, got {:?}",
            d.data_type()
        )))
    }
}

fn expect_u8(d: ColumnData) -> Result<Vec<u8>> {
    if let ColumnData::U8(v) = d {
        Ok(v)
    } else {
        Err(schema_error(&format!(
            "expected U8, got {:?}",
            d.data_type()
        )))
    }
}

fn expect_bytes(d: ColumnData) -> Result<Vec<u8>> {
    if let ColumnData::Bytes(v) = d {
        Ok(v)
    } else {
        Err(schema_error(&format!(
            "expected Bytes, got {:?}",
            d.data_type()
        )))
    }
}

fn expect_i64(d: ColumnData) -> Result<Vec<i64>> {
    if let ColumnData::I64(v) = d {
        Ok(v)
    } else {
        Err(schema_error(&format!(
            "expected I64, got {:?}",
            d.data_type()
        )))
    }
}

fn expect_i32(d: ColumnData) -> Result<Vec<i32>> {
    if let ColumnData::I32(v) = d {
        Ok(v)
    } else {
        Err(schema_error(&format!(
            "expected I32, got {:?}",
            d.data_type()
        )))
    }
}

// ---------------------------------------------------------------------------
// ColumnData::slice
// ---------------------------------------------------------------------------

impl ColumnData {
    /// Return the rows `[start..start+len)` as a fresh `ColumnData`.
    ///
    /// Returns a `Format` error if `start + len > self.len()`.
    pub fn slice(&self, start: usize, len: usize) -> crate::core::error::Result<Self> {
        let total = self.len();
        if start.saturating_add(len) > total {
            return Err(HeliumError::Format(format!(
                "ColumnData::slice: start={start} + len={len} > total={total}"
            )));
        }
        Ok(match self {
            Self::I8(v) => Self::I8(v[start..start + len].to_vec()),
            Self::I16(v) => Self::I16(v[start..start + len].to_vec()),
            Self::I32(v) => Self::I32(v[start..start + len].to_vec()),
            Self::I64(v) => Self::I64(v[start..start + len].to_vec()),
            Self::U8(v) => Self::U8(v[start..start + len].to_vec()),
            Self::U16(v) => Self::U16(v[start..start + len].to_vec()),
            Self::U32(v) => Self::U32(v[start..start + len].to_vec()),
            Self::U64(v) => Self::U64(v[start..start + len].to_vec()),
            Self::F32(v) => Self::F32(v[start..start + len].to_vec()),
            Self::F64(v) => Self::F64(v[start..start + len].to_vec()),
            Self::Bytes(v) => Self::Bytes(v[start..start + len].to_vec()),
        })
    }
}

// ---------------------------------------------------------------------------
// LogicalColumn::slice
// ---------------------------------------------------------------------------

impl LogicalColumn {
    /// Return the logical rows `[start..start+len)` as a fresh `LogicalColumn`
    /// of the same variant.
    ///
    /// For `Nullable`, `Union`, and legacy flat nullable variants the inner
    /// **compact** storage is sliced correctly — only the values that actually
    /// exist in the requested row range are copied.
    ///
    /// Returns a `Format` error if `start + len > self.row_count()`.
    pub fn slice(&self, start: usize, len: usize) -> Result<Self> {
        let total = self.row_count();
        if start.saturating_add(len) > total {
            return Err(HeliumError::Format(format!(
                "LogicalColumn::slice: start={start} + len={len} > row_count={total}"
            )));
        }

        match self {
            // ---- Simple cases ------------------------------------------------
            Self::Primitive(cd) => Ok(Self::Primitive(cd.slice(start, len)?)),

            Self::Utf8(v) => Ok(Self::Utf8(v[start..start + len].to_vec())),

            Self::Binary(v) => Ok(Self::Binary(v[start..start + len].to_vec())),

            // ---- Semantic leaf types -----------------------------------------
            Self::Decimal128 { values } => Ok(Self::Decimal128 {
                values: values[start..start + len].to_vec(),
            }),
            Self::Date32 { values } => Ok(Self::Date32 {
                values: values[start..start + len].to_vec(),
            }),
            Self::Date64 { values } => Ok(Self::Date64 {
                values: values[start..start + len].to_vec(),
            }),
            Self::Datetime { values } => Ok(Self::Datetime {
                values: values[start..start + len].to_vec(),
            }),

            // ---- Struct — recurse each field --------------------------------
            Self::Struct { fields } => {
                let sliced: Result<Vec<(String, LogicalColumn)>> = fields
                    .iter()
                    .map(|(name, col)| col.slice(start, len).map(|sc| (name.clone(), sc)))
                    .collect();
                Ok(Self::Struct { fields: sliced? })
            }

            // ---- List --------------------------------------------------------
            // offsets has len = row_count + 1.
            // The values range for rows [start..start+len] is
            //   offsets[start] .. offsets[start + len].
            // New offsets[i] = offsets[start + i] - offsets[start].
            Self::List { offsets, values } => {
                let base = offsets[start] as usize;
                let end = offsets[start + len] as usize;
                let value_len = end - base;
                let new_values = values.slice(base, value_len)?;
                let base32 = offsets[start];
                let new_offsets: Vec<u32> = offsets[start..=start + len]
                    .iter()
                    .map(|&o| o - base32)
                    .collect();
                Ok(Self::List {
                    offsets: new_offsets,
                    values: Box::new(new_values),
                })
            }

            // ---- Map ---------------------------------------------------------
            // Same shape as List but two children.
            Self::Map {
                offsets,
                keys,
                values,
            } => {
                let base = offsets[start] as usize;
                let end = offsets[start + len] as usize;
                let entry_len = end - base;
                let new_keys = keys.slice(base, entry_len)?;
                let new_values = values.slice(base, entry_len)?;
                let base32 = offsets[start];
                let new_offsets: Vec<u32> = offsets[start..=start + len]
                    .iter()
                    .map(|&o| o - base32)
                    .collect();
                Ok(Self::Map {
                    offsets: new_offsets,
                    keys: Box::new(new_keys),
                    values: Box::new(new_values),
                })
            }

            // ---- Nullable (recursive) ----------------------------------------
            // present[i] tells whether row i has a value.
            // value is compact: only rows where present[i]==true.
            // compact_offset = number of true bits before `start`.
            // n_present    = number of true bits in [start..start+len).
            Self::Nullable { present, value } => {
                let compact_offset = present[..start].iter().filter(|&&p| p).count();
                let n_present = present[start..start + len].iter().filter(|&&p| p).count();
                let new_value = value.slice(compact_offset, n_present)?;
                Ok(Self::Nullable {
                    present: present[start..start + len].to_vec(),
                    value: Box::new(new_value),
                })
            }

            // ---- Union -------------------------------------------------------
            // tags[i] is the variant index for row i (compact per-variant).
            // For each variant v, its storage holds only rows where tags[j]==v.
            // compact_offset_for_v = count of v-valued tags in tags[..start].
            // count_in_slice_for_v = count of v-valued tags in tags[start..start+len].
            Self::Union { tags, variants } => {
                let slice_tags = tags[start..start + len].to_vec();
                let sliced_variants: Result<Vec<(String, LogicalColumn)>> = variants
                    .iter()
                    .enumerate()
                    .map(|(v_idx, (v_name, v_col))| {
                        let compact_offset = tags[..start]
                            .iter()
                            .filter(|&&t| t as usize == v_idx)
                            .count();
                        let count_in_slice =
                            slice_tags.iter().filter(|&&t| t as usize == v_idx).count();
                        let new_v = v_col.slice(compact_offset, count_in_slice)?;
                        Ok((v_name.clone(), new_v))
                    })
                    .collect();
                Ok(Self::Union {
                    tags: slice_tags,
                    variants: sliced_variants?,
                })
            }

            // ---- Legacy flat nullable types ----------------------------------
            Self::NullablePrim { present, values } => {
                let compact_offset = present[..start].iter().filter(|&&p| p).count();
                let n_present = present[start..start + len].iter().filter(|&&p| p).count();
                let new_values = values.slice(compact_offset, n_present)?;
                Ok(Self::NullablePrim {
                    present: present[start..start + len].to_vec(),
                    values: new_values,
                })
            }

            Self::NullableUtf8 { present, strings } => {
                let compact_offset = present[..start].iter().filter(|&&p| p).count();
                let n_present = present[start..start + len].iter().filter(|&&p| p).count();
                Ok(Self::NullableUtf8 {
                    present: present[start..start + len].to_vec(),
                    strings: strings[compact_offset..compact_offset + n_present].to_vec(),
                })
            }

            Self::NullableBinary { present, blobs } => {
                let compact_offset = present[..start].iter().filter(|&&p| p).count();
                let n_present = present[start..start + len].iter().filter(|&&p| p).count();
                Ok(Self::NullableBinary {
                    present: present[start..start + len].to_vec(),
                    blobs: blobs[compact_offset..compact_offset + n_present].to_vec(),
                })
            }

            // ---- Dictionary — dictionary stays intact; slice the indices ------
            Self::Dictionary {
                dictionary,
                indices,
            } => Ok(Self::Dictionary {
                dictionary: dictionary.clone(),
                indices: indices[start..start + len].to_vec(),
            }),

            // ---- Legacy flat array types -------------------------------------
            Self::ArrayOf { offsets, values } => {
                let base = offsets[start] as usize;
                let end = offsets[start + len] as usize;
                let value_len = end - base;
                let new_values = values.slice(base, value_len)?;
                let base32 = offsets[start];
                let new_offsets: Vec<u32> = offsets[start..=start + len]
                    .iter()
                    .map(|&o| o - base32)
                    .collect();
                Ok(Self::ArrayOf {
                    offsets: new_offsets,
                    values: new_values,
                })
            }

            Self::ArrayOfUtf8 { offsets, strings } => {
                // offsets has len = row_count + 1; strings is flat per element.
                let base = offsets[start] as usize;
                let end = offsets[start + len] as usize;
                let base32 = offsets[start];
                let new_offsets: Vec<u32> = offsets[start..=start + len]
                    .iter()
                    .map(|&o| o - base32)
                    .collect();
                Ok(Self::ArrayOfUtf8 {
                    offsets: new_offsets,
                    strings: strings[base..end].to_vec(),
                })
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Unit tests for LogicalColumn::slice and ColumnData::slice
// ---------------------------------------------------------------------------

#[cfg(test)]
mod slice_tests {
    use super::*;

    fn prim_i32(v: Vec<i32>) -> LogicalColumn {
        LogicalColumn::Primitive(ColumnData::I32(v))
    }

    fn prim_u8(v: Vec<u8>) -> LogicalColumn {
        LogicalColumn::Primitive(ColumnData::U8(v))
    }

    // ---- ColumnData::slice --------------------------------------------------

    #[test]
    fn column_data_slice_basic() {
        let cd = ColumnData::I32(vec![10, 20, 30, 40, 50]);
        let s = cd.slice(1, 3).unwrap();
        assert_eq!(s, ColumnData::I32(vec![20, 30, 40]));
    }

    #[test]
    fn column_data_slice_empty() {
        let cd = ColumnData::U64(vec![1, 2, 3]);
        let s = cd.slice(1, 0).unwrap();
        assert_eq!(s, ColumnData::U64(vec![]));
    }

    #[test]
    fn column_data_slice_overflow() {
        let cd = ColumnData::I32(vec![1, 2, 3]);
        assert!(cd.slice(2, 2).is_err());
    }

    #[test]
    fn column_data_slice_full() {
        let cd = ColumnData::F64(vec![1.0, 2.0, 3.0]);
        assert_eq!(cd.slice(0, 3).unwrap(), cd);
    }

    // ---- Primitive ----------------------------------------------------------

    #[test]
    fn slice_primitive() {
        let col = prim_i32(vec![1, 2, 3, 4, 5]);
        let s = col.slice(1, 3).unwrap();
        assert_eq!(s, prim_i32(vec![2, 3, 4]));
    }

    #[test]
    fn slice_primitive_full() {
        let col = prim_i32(vec![10, 20, 30]);
        assert_eq!(col.slice(0, 3).unwrap(), col);
    }

    #[test]
    fn slice_primitive_empty() {
        let col = prim_i32(vec![10, 20, 30]);
        let s = col.slice(1, 0).unwrap();
        assert_eq!(s.row_count(), 0);
    }

    #[test]
    fn slice_primitive_overflow() {
        let col = prim_i32(vec![1, 2, 3]);
        assert!(col.slice(2, 2).is_err());
    }

    // ---- Utf8 ---------------------------------------------------------------

    #[test]
    fn slice_utf8() {
        let col = LogicalColumn::Utf8(vec![
            "alpha".to_string(),
            "beta".to_string(),
            "gamma".to_string(),
            "delta".to_string(),
        ]);
        let s = col.slice(1, 2).unwrap();
        assert_eq!(
            s,
            LogicalColumn::Utf8(vec!["beta".to_string(), "gamma".to_string()])
        );
    }

    // ---- Binary -------------------------------------------------------------

    #[test]
    fn slice_binary() {
        let col = LogicalColumn::Binary(vec![vec![1, 2], vec![3, 4], vec![5, 6]]);
        let s = col.slice(1, 1).unwrap();
        assert_eq!(s, LogicalColumn::Binary(vec![vec![3, 4]]));
    }

    // ---- Nullable -----------------------------------------------------------

    #[test]
    fn slice_nullable_primitive() {
        // 6 rows: T, F, T, T, F, T — values (compact): 0, 2, 3, 5
        let col = LogicalColumn::Nullable {
            present: vec![true, false, true, true, false, true],
            value: Box::new(prim_i32(vec![0, 2, 3, 5])),
        };
        // Slice rows 1..4 (present: F, T, T), values compact offset = 1 (one T before index 1)
        // n_present = 2
        let s = col.slice(1, 3).unwrap();
        assert_eq!(
            s,
            LogicalColumn::Nullable {
                present: vec![false, true, true],
                value: Box::new(prim_i32(vec![2, 3])),
            }
        );
    }

    #[test]
    fn slice_nullable_all_null() {
        let col = LogicalColumn::Nullable {
            present: vec![false, false, false],
            value: Box::new(prim_i32(vec![])),
        };
        let s = col.slice(0, 2).unwrap();
        assert_eq!(
            s,
            LogicalColumn::Nullable {
                present: vec![false, false],
                value: Box::new(prim_i32(vec![])),
            }
        );
    }

    #[test]
    fn slice_nullable_full() {
        let col = LogicalColumn::Nullable {
            present: vec![true, false, true],
            value: Box::new(prim_i32(vec![10, 30])),
        };
        assert_eq!(col.slice(0, 3).unwrap(), col);
    }

    // ---- Struct -------------------------------------------------------------

    #[test]
    fn slice_struct() {
        let col = LogicalColumn::Struct {
            fields: vec![
                ("a".to_string(), prim_i32(vec![1, 2, 3])),
                ("b".to_string(), prim_u8(vec![10, 20, 30])),
            ],
        };
        let s = col.slice(1, 2).unwrap();
        assert_eq!(
            s,
            LogicalColumn::Struct {
                fields: vec![
                    ("a".to_string(), prim_i32(vec![2, 3])),
                    ("b".to_string(), prim_u8(vec![20, 30])),
                ],
            }
        );
    }

    // ---- List ---------------------------------------------------------------

    #[test]
    fn slice_list() {
        // 4 rows: [1,2], [3], [], [4,5,6]
        let col = LogicalColumn::List {
            offsets: vec![0, 2, 3, 3, 6],
            values: Box::new(prim_i32(vec![1, 2, 3, 4, 5, 6])),
        };
        // Slice rows 1..3: [3], []  → values = [3], offsets = [0,1,1]
        let s = col.slice(1, 2).unwrap();
        assert_eq!(
            s,
            LogicalColumn::List {
                offsets: vec![0, 1, 1],
                values: Box::new(prim_i32(vec![3])),
            }
        );
    }

    #[test]
    fn slice_list_full() {
        let col = LogicalColumn::List {
            offsets: vec![0, 2, 3, 3, 6],
            values: Box::new(prim_i32(vec![1, 2, 3, 4, 5, 6])),
        };
        assert_eq!(col.slice(0, 4).unwrap(), col);
    }

    // ---- Map ----------------------------------------------------------------

    #[test]
    fn slice_map() {
        // 3 rows: {1→10}, {2→20, 3→30}, {}
        let col = LogicalColumn::Map {
            offsets: vec![0, 1, 3, 3],
            keys: Box::new(prim_i32(vec![1, 2, 3])),
            values: Box::new(prim_i32(vec![10, 20, 30])),
        };
        // Slice rows 1..2: {2→20, 3→30} → entries=[2,3],[20,30], offsets=[0,2]
        let s = col.slice(1, 1).unwrap();
        assert_eq!(
            s,
            LogicalColumn::Map {
                offsets: vec![0, 2],
                keys: Box::new(prim_i32(vec![2, 3])),
                values: Box::new(prim_i32(vec![20, 30])),
            }
        );
    }

    // ---- Union --------------------------------------------------------------

    #[test]
    fn slice_union() {
        // 5 rows with 2 variants (0 = int, 1 = str)
        // tags = [0, 1, 0, 0, 1]
        // variant 0 (compact): rows 0,2,3 → values [10, 30, 40]
        // variant 1 (compact): rows 1,4   → strings ["a","b"]
        let col = LogicalColumn::Union {
            tags: vec![0, 1, 0, 0, 1],
            variants: vec![
                ("int".to_string(), prim_i32(vec![10, 30, 40])),
                (
                    "str".to_string(),
                    LogicalColumn::Utf8(vec!["a".to_string(), "b".to_string()]),
                ),
            ],
        };
        // Slice rows 1..4 (inclusive end=4, len=3): tags=[1,0,0]
        // variant 0 offset=1 (one 0-tag before index 1), count_in_slice=2 → [30,40]
        // variant 1 offset=1 (one 1-tag before index 1), count_in_slice=1 → ["a"]
        let s = col.slice(1, 3).unwrap();
        assert_eq!(
            s,
            LogicalColumn::Union {
                tags: vec![1, 0, 0],
                variants: vec![
                    ("int".to_string(), prim_i32(vec![30, 40])),
                    (
                        "str".to_string(),
                        LogicalColumn::Utf8(vec!["a".to_string()])
                    ),
                ],
            }
        );
    }

    // ---- Dict ---------------------------------------------------------------

    #[test]
    fn slice_dict_utf8() {
        let col = LogicalColumn::Dictionary {
            dictionary: Box::new(LogicalColumn::Utf8(vec![
                "x".to_string(),
                "y".to_string(),
                "z".to_string(),
            ])),
            indices: vec![0, 1, 2, 0, 1],
        };
        let s = col.slice(1, 3).unwrap();
        assert_eq!(
            s,
            LogicalColumn::Dictionary {
                dictionary: Box::new(LogicalColumn::Utf8(vec![
                    "x".to_string(),
                    "y".to_string(),
                    "z".to_string(),
                ])),
                indices: vec![1, 2, 0],
            }
        );
    }

    // ---- Semantic types -----------------------------------------------------

    #[test]
    fn slice_decimal128() {
        let col = LogicalColumn::Decimal128 {
            values: vec![100i128, 200, 300],
        };
        let s = col.slice(1, 2).unwrap();
        assert_eq!(
            s,
            LogicalColumn::Decimal128 {
                values: vec![200, 300]
            }
        );
    }

    #[test]
    fn slice_date32() {
        let col = LogicalColumn::Date32 {
            values: vec![0i32, 1, 2, 3],
        };
        assert_eq!(
            col.slice(2, 2).unwrap(),
            LogicalColumn::Date32 { values: vec![2, 3] }
        );
    }

    #[test]
    fn slice_datetime() {
        let col = LogicalColumn::Datetime {
            values: vec![1000i64, 2000, 3000],
        };
        assert_eq!(col.slice(0, 3).unwrap(), col);
    }
}
