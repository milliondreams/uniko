//! Higher-level domain operations layered over [`crate::storage`].
//!
//! Where [`crate::storage`] exposes generic node/edge CRUD, this module
//! provides typed, intention-revealing operations on specific node
//! types.  P4 Consolidation calls these helpers rather than building
//! Cypher by hand at the worker layer.

// Rust guideline compliant

pub mod facts;
