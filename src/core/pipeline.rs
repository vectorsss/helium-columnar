use std::fmt;

use super::coder::{AccessPattern, CoderKind, ColumnData, DataType, StageCoder};
use super::error::{HeliumError, Result};

/// An ordered chain of [`StageCoder`]s that encodes or decodes a single
/// physical column.
///
/// Pipelines enforce two invariants at construction time (see
/// [`Pipeline::new`]):
///
/// 1. **Block-after-non-block ordering**: no [`NonBlockCoder`] may appear after
///    a [`BlockCoder`] in the same pipeline.
/// 2. **Type compatibility**: the output type of each stage must match the input
///    type of the next stage.
///
/// [`NonBlockCoder`]: crate::NonBlockCoder
/// [`BlockCoder`]: crate::BlockCoder
pub struct Pipeline {
    stages: Vec<StageCoder>,
    input_type: DataType,
    output_type: DataType,
}

impl fmt::Debug for Pipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ids: Vec<&str> = self.stages.iter().map(|s| s.id()).collect();
        f.debug_struct("Pipeline")
            .field("input_type", &self.input_type)
            .field("output_type", &self.output_type)
            .field("stages", &ids)
            .finish()
    }
}

impl Pipeline {
    /// Build a pipeline, validating both the non-block-then-block ordering rule
    /// and the input/output type chain.
    pub fn new(input_type: DataType, stages: Vec<StageCoder>) -> Result<Self> {
        let mut seen_block = false;
        for stage in &stages {
            match stage.kind() {
                CoderKind::Block => seen_block = true,
                CoderKind::NonBlock if seen_block => {
                    return Err(HeliumError::PipelineOrder(stage.id().to_string()));
                }
                CoderKind::NonBlock => {}
            }
        }

        let mut current_type = input_type;
        for stage in &stages {
            if stage.accepted_input_type() != current_type {
                return Err(HeliumError::TypeMismatch {
                    coder: stage.id().to_string(),
                    expected: stage.accepted_input_type(),
                    got: current_type,
                });
            }
            current_type = stage.produced_output_type();
        }

        Ok(Self {
            stages,
            input_type,
            output_type: current_type,
        })
    }

    /// The [`DataType`] this pipeline expects as its input.
    pub fn input_type(&self) -> DataType {
        self.input_type
    }

    /// The [`DataType`] this pipeline produces after encoding (or after
    /// decoding, reading backwards).
    pub fn output_type(&self) -> DataType {
        self.output_type
    }

    /// Weakest-link combination of every stage's access pattern. A pipeline
    /// is `RandomAccess` only if *every* stage is; any `SequentialOnly`
    /// stage downgrades the whole chain. Empty pipeline is trivially RA
    /// (pass-through).
    pub fn access_pattern(&self) -> AccessPattern {
        self.stages
            .iter()
            .fold(AccessPattern::RandomAccess, |acc, s| {
                acc.combine(s.access_pattern())
            })
    }

    /// Encode `input` by passing it through every stage in forward order.
    pub fn encode(&self, input: ColumnData) -> Result<ColumnData> {
        if input.data_type() != self.input_type {
            return Err(HeliumError::RuntimeType {
                coder: "<pipeline input>".into(),
                expected: self.input_type,
            });
        }
        let mut current = input;
        for stage in &self.stages {
            current = match stage {
                StageCoder::NonBlock(c) => c.encode(&current)?,
                StageCoder::Block(c) => c.encode_block(&current)?,
            };
        }
        Ok(current)
    }

    /// Decode by walking the pipeline in reverse. Each stage decodes its own
    /// input to exhaustion — intermediate row counts are not threaded through
    /// this layer; that is a file-format concern and will be
    /// added when the schema / stripe layer lands.
    pub fn decode(&self, input: ColumnData) -> Result<ColumnData> {
        if input.data_type() != self.output_type {
            return Err(HeliumError::RuntimeType {
                coder: "<pipeline output>".into(),
                expected: self.output_type,
            });
        }
        let mut current = input;
        for stage in self.stages.iter().rev() {
            current = match stage {
                StageCoder::Block(c) => c.decode_block(&current)?,
                StageCoder::NonBlock(c) => c.decode(&current)?,
            };
        }
        Ok(current)
    }
}
