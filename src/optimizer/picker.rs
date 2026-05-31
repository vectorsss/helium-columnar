//! Pipeline measurement and candidate selection.
//!
//! The core primitive is [`measure_pipeline`] which encodes a column with a
//! given pipeline and returns the compressed byte count.  The selector
//! [`pick_best_leaf`] wraps it and runs the candidate set returned by the
//! [`crate::optimizer::candidates`] module.

use crate::{CoderRegistry, CoderSpec, ColumnData, DataType, HeliumError, Pipeline, Result};

use super::candidates::{LeafCandidate, data_candidates, structural_candidates};

/// Encode `data` through the pipeline described by `coders` and return the
/// number of output bytes.
///
/// Returns `Err` if any stage fails (type mismatch, unknown coder, etc.).
/// The pipeline must terminate with a `Bytes`-producing stage; otherwise an
/// error is returned.
pub fn measure_pipeline(
    input_type: DataType,
    coders: &[CoderSpec],
    data: ColumnData,
    registry: &CoderRegistry,
) -> Result<usize> {
    // Build stages
    let mut stages = Vec::with_capacity(coders.len());
    let mut current_type = input_type;
    for spec in coders {
        let stage = registry.build(spec, current_type)?;
        current_type = stage.produced_output_type();
        stages.push(stage);
    }
    let pipeline = Pipeline::new(input_type, stages)?;
    let encoded = pipeline.encode(data)?;
    match encoded {
        ColumnData::Bytes(b) => Ok(b.len()),
        other => Err(HeliumError::CoderFailed {
            coder: "optimizer".into(),
            reason: format!(
                "pipeline did not terminate in Bytes (output type: {:?})",
                other.data_type()
            ),
        }),
    }
}

/// Try a list of candidates and return `(best_label, best_coders)` for the one
/// producing the smallest encoded byte count.
///
/// Candidates that fail encoding (e.g. deltamin on negative values) are silently
/// skipped.  Returns an error only if **all** candidates fail.
pub fn pick_from_candidates(
    input_type: DataType,
    candidates: Vec<LeafCandidate>,
    data: ColumnData,
    registry: &CoderRegistry,
    context_label: &str,
) -> Result<(String, Vec<CoderSpec>)> {
    let mut best: Option<(usize, String, Vec<CoderSpec>)> = None;
    for cand in candidates {
        match measure_pipeline(input_type, &cand.coders, data.clone(), registry) {
            Ok(size) => {
                if best.as_ref().is_none_or(|(s, _, _)| size < *s) {
                    best = Some((size, cand.label, cand.coders));
                }
            }
            Err(_) => {
                // Skip invalid candidates (e.g. type mismatch, negative values
                // into deltamin, etc.)
            }
        }
    }
    match best {
        Some((_, label, coders)) => Ok((label, coders)),
        None => Err(HeliumError::Schema {
            column: context_label.into(),
            reason: "optimizer: no valid encoding candidate found for this leaf".into(),
        }),
    }
}

/// Pick the best encoding for a **structural** leaf (offsets, present, tag, indices).
pub fn pick_best_structural(
    role: &str,
    data: ColumnData,
    terminal: &CoderSpec,
    registry: &CoderRegistry,
    context: &str,
) -> Result<Vec<CoderSpec>> {
    let input_type = data.data_type();
    let candidates = structural_candidates(role, &data, terminal);
    let (_label, coders) = pick_from_candidates(input_type, candidates, data, registry, context)?;
    Ok(coders)
}

/// Pick the best encoding for a **data** leaf (values, dict values, raw data).
pub fn pick_best_data(
    data: ColumnData,
    terminal: &CoderSpec,
    registry: &CoderRegistry,
    context: &str,
) -> Result<Vec<CoderSpec>> {
    let input_type = data.data_type();
    let candidates = data_candidates(&data, terminal);
    let (_label, coders) = pick_from_candidates(input_type, candidates, data, registry, context)?;
    Ok(coders)
}

/// Pick the best encoding for a leaf given its role name.
///
/// Structural roles (`offsets`, `present`, `tag`, `indices`, `data`, `dict_data`,
/// `dict_offsets`, etc.) use role-driven heuristics; all other roles fall through
/// to data-driven candidates.
pub fn pick_best_leaf(
    role: &str,
    data: ColumnData,
    terminal: &CoderSpec,
    registry: &CoderRegistry,
    context: &str,
) -> Result<Vec<CoderSpec>> {
    let is_structural = matches!(
        role,
        "offsets"
            | "outer_offsets"
            | "inner_offsets"
            | "dict_offsets"
            | "present"
            | "tag"
            | "indices"
            | "data"
            | "dict_data"
    );
    if is_structural {
        pick_best_structural(role, data, terminal, registry, context)
    } else {
        pick_best_data(data, terminal, registry, context)
    }
}
