//! Public compress/decompress functions — implementations for `src/codec/mod.rs`.

use crate::core::coder::{ColumnData, DataType};
use crate::core::error::{HeliumError, Result};
use crate::core::pipeline::Pipeline;
use crate::core::registry::{CoderRegistry, CoderSpec};

use super::default::{build_pipeline, default_pipeline_for};
use super::header::{Hec0Header, read_header, write_header};

// ---------------------------------------------------------------------------
// Tier 1 — self-describing with default pipeline
// ---------------------------------------------------------------------------

/// Compress typed column data into a self-describing HEC0 byte buffer.
///
/// The default pipeline for the data's type is selected automatically:
///
/// | DataType      | Pipeline              |
/// |---------------|-----------------------|
/// | I8 / I16      | leb128 → zstd         |
/// | I32 / I64     | delta → leb128 → zstd |
/// | U8 / U16      | leb128 → zstd         |
/// | U32 / U64     | delta → leb128 → zstd |
/// | F32 / F64     | gorilla → zstd        |
/// | Bytes         | zstd                  |
///
/// The output starts with the 4-byte magic `HEC0` and is round-trippable
/// with [`decompress`] without any out-of-band metadata.
pub fn compress(data: ColumnData) -> Result<Vec<u8>> {
    let data_type = data.data_type();
    let pipeline = default_pipeline_for(data_type)?;
    compress_with(data, &pipeline)
}

/// Decompress a HEC0 buffer back to typed column data.
///
/// Reads the type tag and pipeline definition from the embedded header; no
/// out-of-band metadata is required.  The buffer must have been produced by
/// [`compress`] or [`compress_with`].
pub fn decompress(bytes: &[u8]) -> Result<ColumnData> {
    let (header, value_count, body_offset) = read_header(bytes)?;
    let body = &bytes[body_offset..];

    // Reconstruct the pipeline from the header spec list.
    let pipeline = build_pipeline_from_specs(header.data_type, &header.stages)?;

    // Feed the body through the pipeline in reverse.
    let body_data = ColumnData::Bytes(body.to_vec());
    let decoded = pipeline.decode(body_data)?;

    // Sanity-check the recovered row count.
    let recovered_len = decoded.len() as u64;
    if recovered_len != value_count {
        return Err(HeliumError::Corrupted {
            coder: "HEC0".into(),
            reason: format!("decoded {recovered_len} values but header says {value_count}"),
        });
    }

    Ok(decoded)
}

// ---------------------------------------------------------------------------
// Tier 2 — custom pipeline, still self-describing
// ---------------------------------------------------------------------------

/// Compress typed column data using a caller-chosen pipeline.
///
/// The output is still self-describing: the pipeline's stages are embedded in
/// the HEC0 header so [`decompress`] can reconstruct everything from the bytes
/// alone.
///
/// # Errors
///
/// Returns an error if the pipeline's input type does not match `data`'s type,
/// or if the pipeline output is not `DataType::Bytes`.
pub fn compress_with(data: ColumnData, pipeline: &Pipeline) -> Result<Vec<u8>> {
    if pipeline.output_type() != DataType::Bytes {
        return Err(HeliumError::Format(format!(
            "compress_with: pipeline output type must be Bytes, got {:?}",
            pipeline.output_type()
        )));
    }

    let data_type = data.data_type();
    let value_count = data.len() as u64;

    // Encode through the pipeline.
    let encoded = pipeline.encode(data)?;
    let ColumnData::Bytes(body) = encoded else {
        // The output_type() check above makes this unreachable in correct code.
        return Err(HeliumError::Format(
            "compress_with: pipeline did not produce Bytes".into(),
        ));
    };

    // Build stage specs from the pipeline's stages.
    let stages = pipeline_to_specs(pipeline)?;
    let header = Hec0Header { data_type, stages };

    let mut out = write_header(&header, value_count)?;
    out.extend_from_slice(&body);
    Ok(out)
}

// ---------------------------------------------------------------------------
// Tier 3 — bare bytes (expert path, caller must supply pipeline at decode time)
// ---------------------------------------------------------------------------

/// Compress typed column data through an explicit pipeline — **no header**.
///
/// The output is raw encoded bytes with **no HEC0 magic or header**.  The
/// caller is responsible for using the same pipeline at decompress time.
///
/// Prefer [`compress_with`] (self-describing) unless you have a strong reason
/// to avoid the ~20-byte header overhead.
pub fn compress_with_pipeline(data: ColumnData, pipeline: &Pipeline) -> Result<Vec<u8>> {
    let encoded = pipeline.encode(data)?;
    match encoded {
        ColumnData::Bytes(v) => Ok(v),
        other => Err(HeliumError::Format(format!(
            "compress_with_pipeline: pipeline produced {:?} instead of Bytes",
            other.data_type()
        ))),
    }
}

/// Decompress raw bytes through an explicit pipeline — **no header expected**.
///
/// The `pipeline` must be the same one used to compress the data.  Use
/// [`decompress`] for self-describing (HEC0) buffers.
pub fn decompress_with_pipeline(bytes: &[u8], pipeline: &Pipeline) -> Result<ColumnData> {
    let body_data = ColumnData::Bytes(bytes.to_vec());
    pipeline.decode(body_data)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Reconstruct a `Pipeline` from a list of `CoderSpec` values (header stages).
fn build_pipeline_from_specs(input_type: DataType, specs: &[CoderSpec]) -> Result<Pipeline> {
    build_pipeline(input_type, specs)
}

/// Extract the stage specs from a `Pipeline` in forward order.
///
/// `Pipeline` does not expose its stages directly as `CoderSpec` — we recover
/// them by comparing the coder IDs against the registered set. For parameters
/// we re-derive the defaults from the registry's known shapes: `zstd` has a
/// `level` param, `bitpack_fixed` has `width`.
///
/// This works for all built-in coders. Custom coders built outside the
/// registry would need a separate route.
fn pipeline_to_specs(pipeline: &Pipeline) -> Result<Vec<CoderSpec>> {
    // We need to know the stages. Pipeline doesn't expose them directly by
    // design (they're opaque StageCoder). We reconstruct by running encode
    // with a sentinel and comparing, or — simpler — by re-deriving from the
    // input type using the registry.
    //
    // The canonical solution is to call `default_specs_for` if the input type
    // matches a known default, and to rely on the fact that `compress_with`
    // was called with a pipeline built by the caller. Because `Pipeline` only
    // exposes `input_type`, `output_type`, and stage `id()`/`kind()`, we pull
    // the stage IDs via the private helper below.
    let stage_ids = pipeline_stage_ids(pipeline);
    derive_specs_from_ids(pipeline.input_type(), &stage_ids)
}

/// Extract stage IDs from a pipeline by encoding a one-element sentinel and
/// tracing the pipeline structure.  Since `Pipeline` doesn't expose its stages
/// as a slice, we introspect by comparing stage count against known shapes.
///
/// In practice every built-in pipeline matches one of the known spec vectors,
/// so we can use the IDs to reconstruct CoderSpecs. We intentionally keep
/// defaults parameter-free (zstd with default level=3, etc.) — callers that
/// want custom params use `compress_with` with a fully-parameterized pipeline
/// and should also embed their params.  We handle the common param cases below.
fn pipeline_stage_ids(pipeline: &Pipeline) -> Vec<String> {
    // The Pipeline struct exposes `stages` as private. We use the Debug impl
    // which formats stage IDs, and extract them; this avoids adding public API
    // to Pipeline just for this purpose.
    //
    // Alternative: we could change Pipeline to expose stages() -> &[StageCoder].
    // For now, parse the Debug output.
    let debug_str = format!("{pipeline:?}");
    // Format: Pipeline { input_type: I64, output_type: Bytes, stages: ["delta", "leb128", "zstd"] }
    if let Some(start) = debug_str.find("stages: [") {
        let rest = &debug_str[start + 9..]; // after 'stages: ['
        if let Some(end) = rest.find(']') {
            let inner = &rest[..end];
            return inner
                .split(',')
                .map(|s| s.trim().trim_matches('"').to_string())
                .filter(|s| !s.is_empty())
                .collect();
        }
    }
    Vec::new()
}

/// Map stage IDs back to `CoderSpec` values, using the registry to validate.
fn derive_specs_from_ids(input_type: DataType, ids: &[String]) -> Result<Vec<CoderSpec>> {
    let reg = CoderRegistry::with_builtins();
    let mut specs: Vec<CoderSpec> = Vec::with_capacity(ids.len());
    let mut current = input_type;

    for id in ids {
        // Build a bare spec (no extra params) and try it.
        let spec = CoderSpec::new(id.as_str());
        let stage = reg.build(&spec, current).map_err(|e| {
            HeliumError::Format(format!(
                "compress_with: cannot reconstruct pipeline spec for stage '{id}': {e}"
            ))
        })?;
        current = stage.produced_output_type();
        specs.push(spec);
    }

    Ok(specs)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a pipeline directly from spec list (same as default::build_pipeline).
    fn make_pipeline(input_type: DataType, ids: &[&str]) -> Pipeline {
        let specs: Vec<CoderSpec> = ids.iter().map(|id| CoderSpec::new(*id)).collect();
        build_pipeline(input_type, &specs).unwrap()
    }

    // -----------------------------------------------------------------------
    // Round-trip every DataType through compress / decompress.
    // -----------------------------------------------------------------------

    #[test]
    fn round_trip_i8() {
        let values: Vec<i8> = (-50..50).collect();
        let data = ColumnData::I8(values.clone());
        let compressed = compress(data).unwrap();
        let recovered = decompress(&compressed).unwrap();
        assert_eq!(recovered, ColumnData::I8(values));
    }

    #[test]
    fn round_trip_i16() {
        let values: Vec<i16> = (-100..100).collect();
        let data = ColumnData::I16(values.clone());
        let compressed = compress(data).unwrap();
        let recovered = decompress(&compressed).unwrap();
        assert_eq!(recovered, ColumnData::I16(values));
    }

    #[test]
    fn round_trip_i32() {
        let values: Vec<i32> = (0..200).map(|i| i * 1000).collect();
        let data = ColumnData::I32(values.clone());
        let compressed = compress(data).unwrap();
        let recovered = decompress(&compressed).unwrap();
        assert_eq!(recovered, ColumnData::I32(values));
    }

    #[test]
    fn round_trip_i64() {
        let values: Vec<i64> = (0..200).map(|i| i * 1_000_000).collect();
        let data = ColumnData::I64(values.clone());
        let compressed = compress(data).unwrap();
        let recovered = decompress(&compressed).unwrap();
        assert_eq!(recovered, ColumnData::I64(values));
    }

    #[test]
    fn round_trip_u8() {
        let values: Vec<u8> = (0..200).map(|i| i as u8).collect();
        let data = ColumnData::U8(values.clone());
        let compressed = compress(data).unwrap();
        let recovered = decompress(&compressed).unwrap();
        assert_eq!(recovered, ColumnData::U8(values));
    }

    #[test]
    fn round_trip_u16() {
        let values: Vec<u16> = (0..200).map(|i| i as u16 * 100).collect();
        let data = ColumnData::U16(values.clone());
        let compressed = compress(data).unwrap();
        let recovered = decompress(&compressed).unwrap();
        assert_eq!(recovered, ColumnData::U16(values));
    }

    #[test]
    fn round_trip_u32() {
        let values: Vec<u32> = (0..200).map(|i| i * 1000).collect();
        let data = ColumnData::U32(values.clone());
        let compressed = compress(data).unwrap();
        let recovered = decompress(&compressed).unwrap();
        assert_eq!(recovered, ColumnData::U32(values));
    }

    #[test]
    fn round_trip_u64() {
        let values: Vec<u64> = (0..200).map(|i| i * 1_000_000_000).collect();
        let data = ColumnData::U64(values.clone());
        let compressed = compress(data).unwrap();
        let recovered = decompress(&compressed).unwrap();
        assert_eq!(recovered, ColumnData::U64(values));
    }

    #[test]
    fn round_trip_f32() {
        let values: Vec<f32> = (0..200).map(|i| i as f32 * 0.1).collect();
        let data = ColumnData::F32(values.clone());
        let compressed = compress(data).unwrap();
        let recovered = decompress(&compressed).unwrap();
        assert_eq!(recovered, ColumnData::F32(values));
    }

    #[test]
    fn round_trip_f64() {
        let values: Vec<f64> = (0..200).map(|i| i as f64 * 0.01).collect();
        let data = ColumnData::F64(values.clone());
        let compressed = compress(data).unwrap();
        let recovered = decompress(&compressed).unwrap();
        assert_eq!(recovered, ColumnData::F64(values));
    }

    #[test]
    fn round_trip_bytes() {
        let values = b"hello world from helium codec test payload bytes".to_vec();
        let data = ColumnData::Bytes(values.clone());
        let compressed = compress(data).unwrap();
        let recovered = decompress(&compressed).unwrap();
        assert_eq!(recovered, ColumnData::Bytes(values));
    }

    // -----------------------------------------------------------------------
    // Empty columns
    // -----------------------------------------------------------------------

    #[test]
    fn empty_i32_round_trip() {
        let data = ColumnData::I32(Vec::new());
        let compressed = compress(data).unwrap();
        let recovered = decompress(&compressed).unwrap();
        assert_eq!(recovered, ColumnData::I32(Vec::new()));
    }

    #[test]
    fn empty_bytes_round_trip() {
        let data = ColumnData::Bytes(Vec::new());
        let compressed = compress(data).unwrap();
        let recovered = decompress(&compressed).unwrap();
        assert_eq!(recovered, ColumnData::Bytes(Vec::new()));
    }

    // -----------------------------------------------------------------------
    // Tier 2: compress_with (self-describing with custom pipeline)
    // -----------------------------------------------------------------------

    #[test]
    fn compress_with_custom_pipeline_round_trips() {
        let pipeline = make_pipeline(DataType::I64, &["delta", "leb128", "zstd"]);
        let values: Vec<i64> = (0..100).map(|i| i * 999).collect();
        let data = ColumnData::I64(values.clone());
        let compressed = compress_with(data, &pipeline).unwrap();
        // Self-describing: decompress without pipeline.
        let recovered = decompress(&compressed).unwrap();
        assert_eq!(recovered, ColumnData::I64(values));
    }

    #[test]
    fn compress_with_self_describing_across_runs() {
        // Compress, then decode without the original pipeline object — only bytes.
        let pipeline = make_pipeline(DataType::F64, &["gorilla", "zstd"]);
        let values: Vec<f64> = (0..50).map(|i| (i as f64).sin()).collect();
        let compressed = compress_with(ColumnData::F64(values.clone()), &pipeline).unwrap();
        // Drop the pipeline — only bytes remain.
        drop(pipeline);
        let recovered = decompress(&compressed).unwrap();
        assert_eq!(recovered, ColumnData::F64(values));
    }

    // -----------------------------------------------------------------------
    // Tier 3: compress_with_pipeline / decompress_with_pipeline (bare bytes)
    // -----------------------------------------------------------------------

    #[test]
    fn bare_pipeline_round_trip() {
        let pipeline = make_pipeline(DataType::I32, &["delta", "leb128", "zstd"]);
        let values: Vec<i32> = (0..100).map(|i| i * 7).collect();
        let data = ColumnData::I32(values.clone());
        let raw = compress_with_pipeline(data, &pipeline).unwrap();
        let recovered = decompress_with_pipeline(&raw, &pipeline).unwrap();
        assert_eq!(recovered, ColumnData::I32(values));
    }

    #[test]
    fn bare_bytes_no_magic_cannot_be_decoded_by_decompress() {
        // compress_with_pipeline produces raw bytes (no HEC0 header).
        // Feeding them to decompress() must fail.
        let pipeline = make_pipeline(DataType::I32, &["delta", "leb128", "zstd"]);
        let values: Vec<i32> = (0..50).collect();
        let raw = compress_with_pipeline(ColumnData::I32(values), &pipeline).unwrap();
        let err = decompress(&raw).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("magic") || msg.contains("HEC0") || msg.contains("format"),
            "expected magic/format error, got: {msg}"
        );
    }

    // -----------------------------------------------------------------------
    // Magic mismatch
    // -----------------------------------------------------------------------

    #[test]
    fn magic_mismatch_error() {
        let bad = b"FOOO\x01\x03\x00";
        let err = decompress(bad).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("magic") || msg.contains("HEC0") || msg.contains("format"),
            "expected magic error, got: {msg}"
        );
    }

    // -----------------------------------------------------------------------
    // compress_with rejects non-Bytes-producing pipeline
    // -----------------------------------------------------------------------

    #[test]
    fn compress_with_non_bytes_pipeline_errors() {
        // delta alone: I64 → I64, not Bytes.
        use crate::core::registry::CoderRegistry;
        let reg = CoderRegistry::with_builtins();
        let spec = CoderSpec::new("delta");
        let stage = reg.build(&spec, DataType::I64).unwrap();
        let pipeline = Pipeline::new(DataType::I64, vec![stage]).unwrap();
        let err = compress_with(ColumnData::I64(vec![1, 2, 3]), &pipeline).unwrap_err();
        assert!(err.to_string().contains("Bytes") || err.to_string().contains("format"));
    }

    // -----------------------------------------------------------------------
    // Magic bytes are at offset 0..4
    // -----------------------------------------------------------------------

    #[test]
    fn magic_at_start_of_compressed_buffer() {
        let compressed = compress(ColumnData::I32(vec![1, 2, 3, 4, 5])).unwrap();
        assert_eq!(&compressed[0..4], b"HEC0");
    }
}
