//! Pipeline 2 — Entity extraction.
//!
//! The atomic ingest path (`crate::ingest::atomic::ingest_message_atomic`)
//! drives entity extraction directly via the helpers in
//! [`dedup`] and [`rules`] / [`onnx`]; the standalone Step adapter has
//! been retired.

#[cfg(feature = "code-parse")]
pub mod code;
pub mod dedup;
pub mod onnx;
pub mod rules;
pub mod types;

pub use types::{EntityMatch, EntityType, ExtractionSource, RawEntity};
