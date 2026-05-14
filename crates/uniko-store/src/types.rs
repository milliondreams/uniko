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

/// Convert a UTC datetime to uni-db's wire-form temporal value.
///
/// Writing `Value::String(dt.to_rfc3339())` into a `DataType::DateTime`
/// column is silently rejected by uni-db's post-commit flush check —
/// the transaction commits, but the row is omitted from the per-label
/// persisted table, leaving it invisible to label-anchored MATCH.  All
/// uniko write paths that target a DateTime property MUST go through
/// this helper instead.
#[must_use]
pub fn datetime_value(dt: chrono::DateTime<chrono::Utc>) -> uni_db::Value {
    uni_db::Value::Temporal(uni_db::common::TemporalValue::DateTime {
        nanos_since_epoch: dt.timestamp_nanos_opt().unwrap_or(0),
        offset_seconds: 0,
        timezone_name: None,
    })
}
