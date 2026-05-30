//! Named, pre-fabricated compression pipelines via [`TypedCodec`].
//!
//! Instead of building a `Pipeline` by hand, pick a named recipe that matches
//! your column shape. Each constructor returns a `TypedCodec` whose
//! `compress` / `decompress` methods use the self-describing HEC0 format, so
//! round-trips work even without keeping the codec object alive between calls.

use crate::core::coder::{ColumnData, DataType};
use crate::core::error::Result;
use crate::core::pipeline::Pipeline;
use crate::core::registry::CoderSpec;

use super::default::build_pipeline;
use super::top_level::{compress_with, decompress};

/// A pre-fabricated compression pipeline for a known column shape.
///
/// ## Self-describing contract
///
/// `TypedCodec::compress` always produces a self-describing HEC0 buffer whose
/// header records the embedded pipeline.  `TypedCodec::decompress` reads the
/// header and uses **the pipeline embedded in the bytes** — the codec's own
/// internal pipeline is ignored at decode time.  This means:
///
/// - Misuse is at worst slow (e.g. compressing with an expert pipeline then
///   decoding through the simpler auto-detected one), never silently wrong.
/// - The bytes are always round-trippable with the top-level
///   [`crate::decompress`] function, no `TypedCodec` instance needed.
pub struct TypedCodec {
    pipeline: Pipeline,
}

impl TypedCodec {
    /// Build a `TypedCodec` around any `Pipeline`.
    pub fn from_pipeline(pipeline: Pipeline) -> Self {
        Self { pipeline }
    }

    // -------------------------------------------------------------------------
    // Named constructors
    // -------------------------------------------------------------------------

    /// Uniformly-spaced timestamps (constant clock-tick interval).
    ///
    /// Pipeline: `delta_of_delta → leb128 → zstd`
    pub fn for_i64_timestamps_uniform() -> Result<Self> {
        let specs = [
            CoderSpec::new("delta_of_delta"),
            CoderSpec::new("leb128"),
            CoderSpec::new("zstd"),
        ];
        Ok(Self {
            pipeline: build_pipeline(DataType::I64, &specs)?,
        })
    }

    /// Jittered timestamps (e.g. event-arrival times).
    ///
    /// Pipeline: `delta → leb128 → zstd`
    pub fn for_i64_timestamps_jittered() -> Result<Self> {
        let specs = [
            CoderSpec::new("delta"),
            CoderSpec::new("leb128"),
            CoderSpec::new("zstd"),
        ];
        Ok(Self {
            pipeline: build_pipeline(DataType::I64, &specs)?,
        })
    }

    /// Drifting `f64` gauge metrics (e.g. temperature, CPU utilisation).
    ///
    /// Pipeline: `gorilla → zstd`
    pub fn for_f64_gauges() -> Result<Self> {
        let specs = [CoderSpec::new("gorilla"), CoderSpec::new("zstd")];
        Ok(Self {
            pipeline: build_pipeline(DataType::F64, &specs)?,
        })
    }

    /// General-purpose numeric column (best when the value distribution is
    /// unclear). Uses `pcodec`, which self-tunes to the data.
    pub fn for_numeric_general(data_type: DataType) -> Result<Self> {
        let specs = [CoderSpec::new("pcodec")];
        Ok(Self {
            pipeline: build_pipeline(data_type, &specs)?,
        })
    }

    /// Low-cardinality string indices (status codes, categories, log levels).
    ///
    /// The caller is expected to have dict-encoded the strings externally; this
    /// pipeline compresses the resulting `U32` index column.
    ///
    /// Pipeline: `bitpack_auto → zstd`
    pub fn for_low_cardinality_indices() -> Result<Self> {
        let specs = [CoderSpec::new("bitpack_auto"), CoderSpec::new("zstd")];
        Ok(Self {
            pipeline: build_pipeline(DataType::U32, &specs)?,
        })
    }

    /// Sorted unique `u32` values (postings lists, sorted ID sets).
    ///
    /// Pipeline: `elias_fano` (Elias-Fano is self-contained; no zstd needed).
    pub fn for_u32_sorted_unique() -> Result<Self> {
        let specs = [CoderSpec::new("elias_fano")];
        Ok(Self {
            pipeline: build_pipeline(DataType::U32, &specs)?,
        })
    }

    // -------------------------------------------------------------------------
    // Compress / decompress
    // -------------------------------------------------------------------------

    /// Compress `data` using this codec's pipeline.
    ///
    /// The output is a self-describing HEC0 buffer.  The embedded pipeline
    /// header makes it round-trippable with
    /// [`crate::decompress`] without any additional context.
    pub fn compress(&self, data: ColumnData) -> Result<Vec<u8>> {
        compress_with(data, &self.pipeline)
    }

    /// Decompress a HEC0 buffer.
    ///
    /// The pipeline embedded in the bytes is always used — `self.pipeline` is
    /// **ignored** at decode time.  This is intentional: it prevents silent
    /// data corruption when the encoding and decoding codec instances diverge.
    pub fn decompress(&self, bytes: &[u8]) -> Result<ColumnData> {
        decompress(bytes)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_pipeline_round_trips() {
        let specs = [
            CoderSpec::new("delta"),
            CoderSpec::new("leb128"),
            CoderSpec::new("zstd"),
        ];
        let pipeline = build_pipeline(DataType::I64, &specs).unwrap();
        let codec = TypedCodec::from_pipeline(pipeline);
        let values: Vec<i64> = (0..50).map(|i| i * 10_000).collect();
        let compressed = codec.compress(ColumnData::I64(values.clone())).unwrap();
        let recovered = codec.decompress(&compressed).unwrap();
        assert_eq!(recovered, ColumnData::I64(values));
    }

    #[test]
    fn timestamps_uniform_round_trips() {
        let codec = TypedCodec::for_i64_timestamps_uniform().unwrap();
        let values: Vec<i64> = (0..200).map(|i| 1_700_000_000i64 + i * 1_000_000).collect();
        let compressed = codec.compress(ColumnData::I64(values.clone())).unwrap();
        let recovered = codec.decompress(&compressed).unwrap();
        assert_eq!(recovered, ColumnData::I64(values));
    }

    #[test]
    fn timestamps_jittered_round_trips() {
        let codec = TypedCodec::for_i64_timestamps_jittered().unwrap();
        let values: Vec<i64> = (0..200)
            .map(|i| 1_700_000_000i64 + i * 998_543 + (i % 7) * 13)
            .collect();
        let compressed = codec.compress(ColumnData::I64(values.clone())).unwrap();
        let recovered = codec.decompress(&compressed).unwrap();
        assert_eq!(recovered, ColumnData::I64(values));
    }

    #[test]
    fn f64_gauges_round_trips() {
        let codec = TypedCodec::for_f64_gauges().unwrap();
        let values: Vec<f64> = (0..200).map(|i| 23.5 + (i as f64) * 0.01).collect();
        let compressed = codec.compress(ColumnData::F64(values.clone())).unwrap();
        let recovered = codec.decompress(&compressed).unwrap();
        assert_eq!(recovered, ColumnData::F64(values));
    }

    #[test]
    fn low_cardinality_indices_round_trips() {
        let codec = TypedCodec::for_low_cardinality_indices().unwrap();
        let values: Vec<u32> = (0..200).map(|i| i % 5).collect();
        let compressed = codec.compress(ColumnData::U32(values.clone())).unwrap();
        let recovered = codec.decompress(&compressed).unwrap();
        assert_eq!(recovered, ColumnData::U32(values));
    }

    #[test]
    fn u32_sorted_unique_round_trips() {
        let codec = TypedCodec::for_u32_sorted_unique().unwrap();
        let values: Vec<u32> = (0u32..200).map(|i| i * 3).collect(); // sorted, gaps
        let compressed = codec.compress(ColumnData::U32(values.clone())).unwrap();
        let recovered = codec.decompress(&compressed).unwrap();
        assert_eq!(recovered, ColumnData::U32(values));
    }

    #[test]
    fn bytes_embedded_pipeline_wins_over_codec_pipeline() {
        // Compress with uniform-timestamp codec, decode with jittered codec.
        // The bytes' embedded pipeline should win, not the decoder's pipeline.
        let enc_codec = TypedCodec::for_i64_timestamps_uniform().unwrap();
        let dec_codec = TypedCodec::for_i64_timestamps_jittered().unwrap();
        let values: Vec<i64> = (0..100).map(|i| 1_700_000_000i64 + i * 1_000_000).collect();
        let compressed = enc_codec.compress(ColumnData::I64(values.clone())).unwrap();
        let recovered = dec_codec.decompress(&compressed).unwrap();
        // Should still equal original because embedded pipeline (delta_of_delta) is used.
        assert_eq!(recovered, ColumnData::I64(values));
    }
}
