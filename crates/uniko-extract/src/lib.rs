//! # uniko-extract — Layer 3: Content Processing
//!
//! Content intelligence steps: NER (entity extraction), observation extraction,
//! chunking (recursive, tree-sitter, DOM), ingest pipeline, and embedding computation.
//!
//! Depends on `uniko-pipes` only. Implements the `Step` trait for pipeline integration.

pub mod embedding;
pub mod ingest;
pub mod ner;
pub mod observations;
