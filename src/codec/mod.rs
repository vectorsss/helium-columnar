//! Simple typed compression API — helium's "zstd-style" surface.
//!
//! This module provides a high-level compression API that requires no
//! knowledge of `Schema`, `Pipeline`, or `CoderRegistry`.  Two usage tiers:
//!
//! ## Tier 1 — self-describing (recommended)
//!
//! ```rust
//! use helium::{compress, decompress, ColumnData};
//!
//! let values = ColumnData::I64(vec![100, 200, 300, 400, 500]);
//! let compressed = compress(values.clone()).unwrap();
//! let recovered  = decompress(&compressed).unwrap();
//! assert_eq!(recovered, values);
//! ```
//!
//! The output starts with the `HEC0` magic and carries the type tag + pipeline
//! definition in its header.  `decompress` reconstructs everything from the
//! bytes alone — no out-of-band metadata.
//!
//! ## Tier 2 — custom pipeline, still self-describing
//!
//! ```rust
//! use helium::{compress_with, decompress, ColumnData, DataType, Pipeline};
//! use helium::{CoderRegistry, CoderSpec};
//!
//! let reg = CoderRegistry::with_builtins();
//! let specs = [CoderSpec::new("pcodec")];
//! let mut current = DataType::I32;
//! let mut stages = Vec::new();
//! for spec in &specs {
//!     let stage = reg.build(spec, current).unwrap();
//!     current = stage.produced_output_type();
//!     stages.push(stage);
//! }
//! let pipeline = Pipeline::new(DataType::I32, stages).unwrap();
//!
//! let values = ColumnData::I32(vec![1, 2, 3, 4, 5]);
//! let compressed = compress_with(values.clone(), &pipeline).unwrap();
//! let recovered  = decompress(&compressed).unwrap();
//! assert_eq!(recovered, values);
//! ```
//!
//! ## Tier 3 — bare bytes (expert path)
//!
//! ```rust
//! use helium::{compress_with_pipeline, decompress_with_pipeline, ColumnData, DataType, Pipeline};
//! use helium::{CoderRegistry, CoderSpec};
//!
//! let reg = CoderRegistry::with_builtins();
//! let specs = [CoderSpec::new("zstd")];
//! let mut current = DataType::Bytes;
//! let mut stages = Vec::new();
//! for spec in &specs {
//!     let stage = reg.build(spec, current).unwrap();
//!     current = stage.produced_output_type();
//!     stages.push(stage);
//! }
//! let pipeline = Pipeline::new(DataType::Bytes, stages).unwrap();
//!
//! let payload = ColumnData::Bytes(b"hello world".to_vec());
//! let raw = compress_with_pipeline(payload.clone(), &pipeline).unwrap();
//! // `raw` has NO HEC0 header — caller must supply the same pipeline at decode time.
//! let recovered = decompress_with_pipeline(&raw, &pipeline).unwrap();
//! assert_eq!(recovered, payload);
//! ```
//!
//! ## Named pre-fabricated codecs
//!
//! ```rust
//! use helium::{TypedCodec, ColumnData};
//!
//! let codec = TypedCodec::for_i64_timestamps_uniform().unwrap();
//! let ts: Vec<i64> = (0..1000).map(|i| 1_700_000_000_000_000i64 + i * 10_000).collect();
//! let compressed = codec.compress(ColumnData::I64(ts.clone())).unwrap();
//! let recovered  = codec.decompress(&compressed).unwrap();
//! assert_eq!(recovered, ColumnData::I64(ts));
//! ```

pub mod default;
pub(crate) mod header;
pub(crate) mod top_level;
pub mod typed;

pub use top_level::{
    compress, compress_with, compress_with_pipeline, decompress, decompress_with_pipeline,
};
pub use typed::TypedCodec;
