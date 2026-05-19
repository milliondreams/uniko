//! # uniko-store — Layer 1: Graph Storage
//!
//! Wraps uni-db to provide typed graph storage, search (vector, fulltext, hybrid),
//! and Locy runtime for the uniko cognitive memory system.
//!
//! This is the lowest layer. It depends only on uni-db and external utility crates.
//! All other uniko crates access storage through this layer.

pub mod config;
pub mod error;
pub mod id;
pub mod locy;
pub mod operations;
pub mod schema;
pub mod search;
pub mod storage;
pub mod types;

pub use error::{Result, UnikoError};
#[doc(inline)]
pub use storage::KnowledgeBase;
pub use types::*;

// Re-export `ModelRuntime` so callers can name the type without
// taking a direct dependency on `uni_xervo`. Needed for multi-KB
// workflows that share one ONNX session — see
// [`KnowledgeBase::build_shared_runtime`] and
// [`KnowledgeBase::open_with_runtime`]. uni-db wraps `ModelRuntime`
// internally but does not re-export it.
pub use uni_xervo::runtime::ModelRuntime;
