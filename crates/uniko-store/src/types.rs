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

/// Embedding vector (f32 components, dimensionality depends on model).
pub type EmbeddingVec = Vec<f32>;

/// Convert a UTC datetime to uni-db's wire-form temporal value.
///
/// This is the canonical, type-safe way to write a `DataType::DateTime`
/// property: it carries full nanosecond precision and a normalized UTC
/// offset. Prefer it over `Value::String(dt.to_rfc3339())` + uni-db's
/// write-time string→DateTime coercion (added in uni-db 2.0, issue #68),
/// which round-trips through the `datetime()` parser and is not
/// guaranteed to preserve sub-second precision.
///
/// Historically this helper was *mandatory*: pre-2.0, a String written
/// to a DateTime column committed but was silently dropped at flush
/// (row omitted from the per-label table). #68 fixed that — strings are
/// now coerced or loudly rejected — but the typed path here remains the
/// preferred one.
#[must_use]
pub fn datetime_value(dt: chrono::DateTime<chrono::Utc>) -> uni_db::Value {
    uni_db::Value::Temporal(uni_db::common::TemporalValue::DateTime {
        nanos_since_epoch: dt.timestamp_nanos_opt().unwrap_or(0),
        offset_seconds: 0,
        timezone_name: None,
    })
}
