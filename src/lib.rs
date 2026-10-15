//! Native CLEF inference. The model release is pinned; requests use the
//! release's schema encoding and SystemOne answer contract.
pub mod audio;
mod backbone;
pub mod checkpoint;
mod chunk;
mod embedding;
pub mod encoding;
mod gpu;
mod head;
mod kernel_cache;
pub mod media;
mod memory;
pub mod model;
mod omni;
pub mod omni_media;
mod ops;
#[cfg(feature = "bench-internals")]
mod packed_cache;
mod quantized;
pub mod schema;
mod vision;

#[cfg(feature = "bench-internals")]
#[doc(hidden)]
pub mod benchmarking;

#[cfg(test)]
mod tests;

pub use encoding::{EncodeOptions, EncodedQuestion, EncodedRecord, Encoder};
pub use model::{ClefModel, LoadOptions, Precision, Prediction};
pub use schema::{Question, QuestionType, Request, Response};
pub type Result<T> = anyhow::Result<T>;

pub mod architecture;
pub use architecture::ModelVariant;
