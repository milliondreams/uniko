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
