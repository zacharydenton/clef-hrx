//! Native CLEF inference. The model release is pinned; requests use the
//! release's schema encoding and SystemOne answer contract.
mod backbone;
pub mod checkpoint;
mod chunk;
pub mod encoding;
mod gpu;
mod head;
pub mod media;
mod memory;
pub mod model;
mod ops;
pub mod schema;
mod vision;

#[cfg(feature = "bench-internals")]
#[doc(hidden)]
pub mod benchmarking;

#[cfg(test)]
mod tests;

pub use encoding::{EncodeOptions, EncodedQuestion, EncodedRecord, Encoder};
pub use model::{ClefModel, LoadOptions, Prediction};
pub use schema::{Question, QuestionType, Request, Response};
pub type Result<T> = anyhow::Result<T>;
