//! Shared type aliases used across all uniko layers.
//!
//! These are domain-level types. `NodeId` and `EdgeId` are `i64` aliases for
//! uniko's own tracking — conversion to/from uni-db's `Vid(u64)` / `Eid(u64)`
//! happens at the storage boundary in the `storage` module.

/// Internal node identifier (matches uni-db's sequential ID range).
pub type NodeId = i64;

/// Internal edge identifier (matches uni-db's sequential ID range).
pub type EdgeId = i64;

/// UTC timestamp used for all temporal fields.
pub type Timestamp = chrono::DateTime<chrono::Utc>;

/// Agent identifier (UUID v7 string or caller-provided).
pub type AgentId = String;

/// Session identifier (UUID v7 string or caller-provided).
pub type SessionId = String;

/// Goal identifier (UUID v7 string or caller-provided).
pub type GoalId = String;

/// Task identifier (UUID v7 string or caller-provided).
pub type TaskId = String;

/// Embedding vector (f32 components, dimensionality depends on model).
pub type EmbeddingVec = Vec<f32>;
