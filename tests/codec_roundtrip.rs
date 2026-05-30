//! Integration tests for the `helium::codec` API.
//!
//! Covers every `TypedCodec::for_*` constructor, the self-describing contract,
//! and compression-ratio sanity for timestamp columns.

use helium::{CoderRegistry, CoderSpec, Pipeline};
use helium::{ColumnData, DataType, TypedCodec, compress, compress_with, decompress};

// ---------------------------------------------------------------------------
// Helper: build a pipeline from spec IDs.
// ---------------------------------------------------------------------------

fn make_pipeline(input_type: DataType, ids: &[&str]) -> Pipeline {
    let reg = CoderRegistry::with_builtins();
    let mut stages = Vec::new();
    let mut current = input_type;
    for id in ids {
        let spec = CoderSpec::new(*id);
        let stage = reg.build(&spec, current).unwrap();
        current = stage.produced_output_type();
        stages.push(stage);
    }
    Pipeline::new(input_type, stages).unwrap()
}

// ---------------------------------------------------------------------------
// TypedCodec constructors: compress + decompress for 1000 representative values
// ---------------------------------------------------------------------------

#[test]
fn typed_codec_uniform_timestamps_round_trip() {
    let codec = TypedCodec::for_i64_timestamps_uniform().unwrap();
    let values: Vec<i64> = (0..1000)
        .map(|i| 1_700_000_000_000_000i64 + i * 10_000)
        .collect();
    let compressed = codec.compress(ColumnData::I64(values.clone())).unwrap();
    let recovered = codec.decompress(&compressed).unwrap();
    assert_eq!(recovered, ColumnData::I64(values));
}

#[test]
fn typed_codec_jittered_timestamps_round_trip() {
    let codec = TypedCodec::for_i64_timestamps_jittered().unwrap();
    let values: Vec<i64> = (0..1000)
        .map(|i| 1_700_000_000_000_000i64 + i * 10_000 + (i % 13) * 37)
        .collect();
    let compressed = codec.compress(ColumnData::I64(values.clone())).unwrap();
    let recovered = codec.decompress(&compressed).unwrap();
    assert_eq!(recovered, ColumnData::I64(values));
}

#[test]
fn typed_codec_f64_gauges_round_trip() {
    let codec = TypedCodec::for_f64_gauges().unwrap();
    let values: Vec<f64> = (0..1000).map(|i| 21.5 + (i as f64) * 0.003).collect();
    let compressed = codec.compress(ColumnData::F64(values.clone())).unwrap();
    let recovered = codec.decompress(&compressed).unwrap();
    assert_eq!(recovered, ColumnData::F64(values));
}

#[test]
fn typed_codec_numeric_general_i64_round_trip() {
    let codec = TypedCodec::for_numeric_general(DataType::I64).unwrap();
    let values: Vec<i64> = (0..1000).map(|i| i * 12_345 - 5_000_000).collect();
    let compressed = codec.compress(ColumnData::I64(values.clone())).unwrap();
    let recovered = codec.decompress(&compressed).unwrap();
    assert_eq!(recovered, ColumnData::I64(values));
}

#[test]
fn typed_codec_numeric_general_f32_round_trip() {
    let codec = TypedCodec::for_numeric_general(DataType::F32).unwrap();
    let values: Vec<f32> = (0..1000).map(|i| (i as f32) * 0.5 - 250.0).collect();
    let compressed = codec.compress(ColumnData::F32(values.clone())).unwrap();
    let recovered = codec.decompress(&compressed).unwrap();
    assert_eq!(recovered, ColumnData::F32(values));
}

#[test]
fn typed_codec_low_cardinality_indices_round_trip() {
    let codec = TypedCodec::for_low_cardinality_indices().unwrap();
    // 5 distinct categories repeated across 1000 rows.
    let values: Vec<u32> = (0..1000).map(|i| (i % 5) as u32).collect();
    let compressed = codec.compress(ColumnData::U32(values.clone())).unwrap();
    let recovered = codec.decompress(&compressed).unwrap();
    assert_eq!(recovered, ColumnData::U32(values));
}

#[test]
fn typed_codec_u32_sorted_unique_round_trip() {
    let codec = TypedCodec::for_u32_sorted_unique().unwrap();
    // Sorted with gaps — typical postings list.
    let values: Vec<u32> = (0..1000).map(|i| i * 7).collect();
    let compressed = codec.compress(ColumnData::U32(values.clone())).unwrap();
    let recovered = codec.decompress(&compressed).unwrap();
    assert_eq!(recovered, ColumnData::U32(values));
}

// ---------------------------------------------------------------------------
// Compression-ratio sanity: uniform timestamps should compress well.
// ---------------------------------------------------------------------------

#[test]
fn uniform_timestamps_compression_ratio() {
    let codec = TypedCodec::for_i64_timestamps_uniform().unwrap();
    let n = 10_000usize;
    let values: Vec<i64> = (0..n as i64)
        .map(|i| 1_700_000_000_000_000 + i * 10_000)
        .collect();
    let raw_bytes = n * 8; // i64 = 8 bytes each
    let compressed = codec.compress(ColumnData::I64(values)).unwrap();
    // The spec claims 28000×; even 20× should pass with margin.
    // raw / compressed > 20 means compressed < raw / 20 = 4000 bytes for 80kB input.
    assert!(
        compressed.len() < raw_bytes / 20,
        "expected compression ratio > 20× but got {raw_bytes}/{} = {:.1}×",
        compressed.len(),
        raw_bytes as f64 / compressed.len() as f64
    );
}

// ---------------------------------------------------------------------------
// Self-describing contract: bytes survive without the original pipeline object.
// ---------------------------------------------------------------------------

#[test]
fn self_describing_round_trip_without_pipeline() {
    let pipeline = make_pipeline(DataType::I64, &["delta", "leb128", "zstd"]);
    let values: Vec<i64> = (0..500).map(|i| i * 1_000_000).collect();
    let compressed = compress_with(ColumnData::I64(values.clone()), &pipeline).unwrap();
    // Drop the pipeline — only the bytes remain.
    drop(pipeline);
    let recovered = decompress(&compressed).unwrap();
    assert_eq!(recovered, ColumnData::I64(values));
}

// ---------------------------------------------------------------------------
// Bytes column round-trip.
// ---------------------------------------------------------------------------

#[test]
fn bytes_column_round_trip() {
    let data = ColumnData::Bytes(vec![1, 2, 3, 4, 5]);
    let compressed = compress(data.clone()).unwrap();
    let recovered = decompress(&compressed).unwrap();
    assert_eq!(recovered, data);
}

#[test]
fn bytes_column_large_payload_round_trip() {
    let payload: Vec<u8> = (0..10_000).map(|i| (i % 256) as u8).collect();
    let data = ColumnData::Bytes(payload.clone());
    let compressed = compress(data).unwrap();
    let recovered = decompress(&compressed).unwrap();
    assert_eq!(recovered, ColumnData::Bytes(payload));
}

// ---------------------------------------------------------------------------
// TypedCodec::decompress uses bytes' embedded pipeline, not the codec's own.
// ---------------------------------------------------------------------------

#[test]
fn typed_codec_decompress_uses_embedded_pipeline() {
    // Compress with uniform-timestamp codec (delta_of_delta).
    let enc_codec = TypedCodec::for_i64_timestamps_uniform().unwrap();
    // Attempt to decompress with a completely different codec (f64 gauges).
    // The f64 gauges codec ignores its own pipeline at decode time — the bytes win.
    // (The bytes are I64 data with delta_of_delta, not F64, so the codec's pipeline
    //  is irrelevant. We use jittered codec which also operates on I64.)
    let dec_codec = TypedCodec::for_i64_timestamps_jittered().unwrap();
    let values: Vec<i64> = (0..200).map(|i| 1_700_000_000i64 + i * 1_000_000).collect();
    let compressed = enc_codec.compress(ColumnData::I64(values.clone())).unwrap();
    let recovered = dec_codec.decompress(&compressed).unwrap();
    assert_eq!(recovered, ColumnData::I64(values));
}

// ---------------------------------------------------------------------------
// Empty columns survive round-trip.
// ---------------------------------------------------------------------------

#[test]
fn empty_i32_codec_round_trip() {
    let compressed = compress(ColumnData::I32(Vec::new())).unwrap();
    let recovered = decompress(&compressed).unwrap();
    assert_eq!(recovered, ColumnData::I32(Vec::new()));
}

#[test]
fn empty_bytes_codec_round_trip() {
    let compressed = compress(ColumnData::Bytes(Vec::new())).unwrap();
    let recovered = decompress(&compressed).unwrap();
    assert_eq!(recovered, ColumnData::Bytes(Vec::new()));
}

// ---------------------------------------------------------------------------
// HEC0 magic verification: hexdump check on a small I32 column.
// ---------------------------------------------------------------------------

#[test]
fn hec0_magic_bytes_present_at_offset_0() {
    let data = ColumnData::I32(vec![10, 20, 30, 40, 50]);
    let compressed = compress(data).unwrap();
    // Bytes 0..4 must be the HEC0 magic.
    assert_eq!(
        &compressed[0..4],
        b"HEC0",
        "expected HEC0 magic at offset 0, got: {:?}",
        &compressed[0..4]
    );
}

#[test]
fn hec0_structure_5_row_i32() {
    // Freeze-test the wire format for a 5-element I32 column.
    // Default pipeline for I32: delta → leb128 → zstd
    // Layout: [0..4]=HEC0 [4]=header_len [5]=data_type(3=I32) [6]=stage_count(3)
    //         then stage blocks…
    let data = ColumnData::I32(vec![10, 20, 30, 40, 50]);
    let compressed = compress(data).unwrap();

    assert_eq!(&compressed[0..4], b"HEC0"); // magic
    let header_len = compressed[4] as usize;
    assert!(header_len > 0, "header_len should be > 0");
    assert_eq!(compressed[5], 3u8, "DataType::I32 discriminant should be 3");
    assert_eq!(compressed[6], 3u8, "3 stages: delta, leb128, zstd");

    // Confirm round-trip still works from the same buffer.
    let recovered = decompress(&compressed).unwrap();
    assert_eq!(recovered, ColumnData::I32(vec![10, 20, 30, 40, 50]));
}
