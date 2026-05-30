//! Coder registry — maps stable string IDs to type-aware factories so that
//! schemas reference coders by name and the correct typed implementation is
//! materialized at pipeline build time.
//!
//! Each factory receives both the `CoderSpec` (its declared parameters) and
//! the `input_type` flowing into it. Primitive-agnostic coders (delta, rle,
//! leb128, …) use `input_type` to auto-specialize; coders with parameters
//! (bitpack_fixed, zstd) pull them from `CoderSpec::params`.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::coder::{DataType, StageCoder};
use super::error::{HeliumError, Result};
use crate::coders::{
    BitpackAuto, BitpackFixed, Delta, DeltaMin, DeltaOfDelta, EliasFano, GorillaXor, Leb128, Lz4,
    Pcodec, Rle, Snappy, Zstd,
};

/// Declarative description of one coder in a pipeline — the unit that gets
/// serialized into the schema.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoderSpec {
    /// Stable string identifier for the coder (e.g. `"zstd"`, `"leb128"`).
    pub id: String,
    /// Optional key-value parameters (e.g. `{ "level": 3 }` for zstd).
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    pub params: Map<String, Value>,
}

impl CoderSpec {
    /// Create a `CoderSpec` with the given ID and no parameters.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            params: Map::new(),
        }
    }

    /// Add a key-value parameter (builder pattern).
    pub fn with_param(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        self.params.insert(key.into(), value.into());
        self
    }

    /// Look up a `u32` parameter by key, returning an error if absent or
    /// not representable as `u32`.
    pub fn get_u32(&self, key: &str) -> Result<u32> {
        let v = self
            .params
            .get(key)
            .ok_or_else(|| HeliumError::InvalidParam {
                coder: self.id.clone(),
                param: key.into(),
                reason: "missing".into(),
            })?;
        v.as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .ok_or_else(|| HeliumError::InvalidParam {
                coder: self.id.clone(),
                param: key.into(),
                reason: format!("expected u32, got {v}"),
            })
    }

    /// Look up an `i32` parameter, returning `default` if absent.
    pub fn get_i32_or(&self, key: &str, default: i32) -> Result<i32> {
        match self.params.get(key) {
            None => Ok(default),
            Some(v) => v
                .as_i64()
                .and_then(|n| i32::try_from(n).ok())
                .ok_or_else(|| HeliumError::InvalidParam {
                    coder: self.id.clone(),
                    param: key.into(),
                    reason: format!("expected i32, got {v}"),
                }),
        }
    }

    /// Look up an optional `usize` parameter, returning `None` if absent.
    pub fn get_usize_optional(&self, key: &str) -> Result<Option<usize>> {
        match self.params.get(key) {
            None => Ok(None),
            Some(v) => v
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .map(Some)
                .ok_or_else(|| HeliumError::InvalidParam {
                    coder: self.id.clone(),
                    param: key.into(),
                    reason: format!("expected usize, got {v}"),
                }),
        }
    }
}

type CoderFactory = Box<dyn Fn(&CoderSpec, DataType) -> Result<StageCoder> + Send + Sync>;

/// Maps coder IDs to type-aware factories.
pub struct CoderRegistry {
    factories: HashMap<String, CoderFactory>,
}

impl CoderRegistry {
    /// Create an empty registry with no registered coders.
    pub fn new() -> Self {
        Self {
            factories: HashMap::new(),
        }
    }

    /// Register a factory function under `id`.
    ///
    /// The factory receives the full [`CoderSpec`] (for parameter access) and
    /// the `input_type` flowing into this stage, and must return a boxed
    /// [`StageCoder`].
    pub fn register<F>(&mut self, id: impl Into<String>, factory: F)
    where
        F: Fn(&CoderSpec, DataType) -> Result<StageCoder> + Send + Sync + 'static,
    {
        self.factories.insert(id.into(), Box::new(factory));
    }

    /// Build a coder stage given its spec and the DataType flowing into it
    /// (either the column's physical type for the first stage or the previous
    /// stage's output type).
    pub fn build(&self, spec: &CoderSpec, input_type: DataType) -> Result<StageCoder> {
        let factory = self
            .factories
            .get(&spec.id)
            .ok_or_else(|| HeliumError::UnknownCoder(spec.id.clone()))?;
        factory(spec, input_type)
    }

    /// Iterator over all registered coder IDs.
    pub fn ids(&self) -> impl Iterator<Item = &str> {
        self.factories.keys().map(String::as_str)
    }

    /// Registry wired with every coder this crate ships.
    pub fn with_builtins() -> Self {
        let mut r = Self::new();
        r.register("delta", |_spec, input_type| {
            Ok(StageCoder::NonBlock(Box::new(Delta::new(input_type)?)))
        });
        r.register("leb128", |_spec, input_type| {
            Ok(StageCoder::NonBlock(Box::new(Leb128::new(input_type)?)))
        });
        r.register("rle", |_spec, input_type| {
            Ok(StageCoder::NonBlock(Box::new(Rle::new(input_type)?)))
        });
        r.register("deltamin", |_spec, input_type| {
            Ok(StageCoder::Block(Box::new(DeltaMin::new(input_type)?)))
        });
        r.register("bitpack_fixed", |spec, input_type| {
            let width = spec.get_u32("width")?;
            Ok(StageCoder::NonBlock(Box::new(BitpackFixed::new(
                input_type, width,
            )?)))
        });
        r.register("bitpack_auto", |_spec, input_type| {
            Ok(StageCoder::Block(Box::new(BitpackAuto::new(input_type)?)))
        });
        r.register("zstd", |spec, input_type| {
            expect_bytes("zstd", input_type)?;
            let level = spec.get_i32_or("level", 3)?;
            Ok(StageCoder::Block(Box::new(Zstd::new(level))))
        });
        r.register("lz4", |_spec, input_type| {
            expect_bytes("lz4", input_type)?;
            Ok(StageCoder::Block(Box::new(Lz4)))
        });
        r.register("snappy", |_spec, input_type| {
            expect_bytes("snappy", input_type)?;
            Ok(StageCoder::Block(Box::new(Snappy)))
        });
        r.register("pcodec", |spec, input_type| {
            let level = spec.get_usize_optional("level")?;
            Ok(StageCoder::Block(Box::new(Pcodec::new(input_type, level)?)))
        });
        r.register("gorilla", |_spec, input_type| {
            Ok(StageCoder::NonBlock(Box::new(GorillaXor::new(input_type)?)))
        });
        r.register("elias_fano", |_spec, input_type| {
            Ok(StageCoder::Block(Box::new(EliasFano::new(input_type)?)))
        });
        r.register("delta_of_delta", |_spec, input_type| {
            Ok(StageCoder::NonBlock(Box::new(DeltaOfDelta::new(
                input_type,
            )?)))
        });
        r
    }
}

fn expect_bytes(coder: &'static str, input_type: DataType) -> Result<()> {
    if input_type != DataType::Bytes {
        return Err(HeliumError::InvalidParam {
            coder: coder.into(),
            param: "<input_type>".into(),
            reason: format!("expected Bytes, got {input_type:?}"),
        });
    }
    Ok(())
}

impl Default for CoderRegistry {
    fn default() -> Self {
        Self::with_builtins()
    }
}
