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

/// Decode a `Value::Temporal` into a UTC datetime, surfacing a precise
/// [`UnikoError::Storage`] when it does not yield a valid timestamp.
///
/// The inverse of [`datetime_value`] for the read path. Use this when an
/// absent / malformed timestamp is genuinely an error (e.g. an Episode
/// that *must* carry a write time); use [`optional_datetime_from_row`]
/// when absence is acceptable. uni-db (>= 2.0) returns DateTime properties
/// as typed `Value::Temporal`, so no RFC-3339 string branch is needed.
///
/// # Errors
///
/// Returns [`UnikoError::Storage`](crate::UnikoError::Storage) if the
/// value is not a Temporal or the millis fall outside the supported range.
pub fn datetime_from_value(
    value: &uni_db::Value,
    context: &str,
) -> crate::error::Result<chrono::DateTime<chrono::Utc>> {
    use chrono::{DateTime, Utc};
    match value {
        uni_db::Value::Temporal(t) => {
            let millis = t
                .epoch_millis()
                .ok_or_else(|| crate::UnikoError::Storage(format!("{context} has no epoch")))?;
            DateTime::<Utc>::from_timestamp_millis(millis).ok_or_else(|| {
                crate::UnikoError::Storage(format!("epoch millis {millis} out of range"))
            })
        }
        other => Err(crate::UnikoError::Storage(format!(
            "{context} unexpected type: {other:?}"
        ))),
    }
}

/// Pull an optional UTC datetime out of a uni-db row `column`.
///
/// uni-db (>= 2.0) serialises DateTime properties as `Value::Temporal`.
/// Returns `None` when the column is missing, null, or not a valid
/// Temporal — callers treat absence as "not yet set" rather than an error.
#[must_use]
pub fn optional_datetime_from_row(
    row: &uni_db::Row,
    column: &str,
) -> Option<chrono::DateTime<chrono::Utc>> {
    use chrono::{DateTime, Utc};
    let idx = row.columns().iter().position(|c| c == column)?;
    let value = row.values().get(idx)?;
    match value {
        uni_db::Value::Temporal(t) => {
            let millis = t.epoch_millis()?;
            DateTime::<Utc>::from_timestamp_millis(millis)
        }
        _ => None,
    }
}
