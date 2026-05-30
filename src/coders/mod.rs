//! Concrete coder implementations for helium pipelines.
//!
//! All coders are registered by [`CoderRegistry::default()`](crate::CoderRegistry::default)
//! under stable string IDs. Use [`CoderSpec::new("id")`](crate::CoderSpec) to reference
//! them in a schema, or instantiate them directly for unit testing.

mod bitpack;
mod bitstream;
mod delta;
mod delta_of_delta;
mod deltamin;
mod elias_fano;
mod gorilla;
mod leb128;
mod lz4;
mod numeric;
mod pcodec;
mod rle;
mod snappy;
mod zstd_coder;

pub use bitpack::{BitpackAuto, BitpackFixed};
pub use delta::Delta;
pub use delta_of_delta::DeltaOfDelta;
pub use deltamin::DeltaMin;
pub use elias_fano::EliasFano;
pub use gorilla::GorillaXor;
pub use leb128::Leb128;
pub use lz4::Lz4;
pub use pcodec::Pcodec;
pub use rle::Rle;
pub use snappy::Snappy;
pub use zstd_coder::Zstd;
