//! Shared conversion helpers between `serde_json` and `uni_db` value
//! trees.  Centralised here so episode/action/rules don't each carry a
//! copy.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde_json::Value as JsonValue;
use uni_db::Value;

use uniko_store::UnikoError;

/// Convert a `serde_json::Value` tree to `uni_db::Value`.
///
/// Maps are preserved as `Value::Map`; arrays as `Value::List`.  Null
/// becomes `Value::Null` per uni-db conventions.  Numbers prefer `Int`
/// when integral, then `Float`, then fall back to the JSON text form.
pub(crate) fn json_to_value(json: &JsonValue) -> Value {
    match json {
        JsonValue::Null => Value::Null,
        JsonValue::Bool(b) => Value::Bool(*b),
        JsonValue::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Int(i)
            } else if let Some(f) = n.as_f64() {
                Value::Float(f)
            } else {
                Value::String(n.to_string())
            }
        }
        JsonValue::String(s) => Value::String(s.clone()),
        JsonValue::Array(arr) => Value::List(arr.iter().map(json_to_value).collect()),
        JsonValue::Object(obj) => {
            let mut m = HashMap::new();
            for (k, v) in obj {
                m.insert(k.clone(), json_to_value(v));
            }
            Value::Map(m)
        }
    }
}

/// Pull an optional UTC datetime out of a uni-db row column.
///
/// uni-db serialises DateTime properties as `Value::Temporal`, but
/// some legacy properties land as RFC-3339 strings.  Falls back to
/// `None` when the column is missing, null, or unparseable — callers
/// treat absence as "not yet set" rather than an error.
pub(crate) fn extract_optional_dt(row: &uni_db::Row, column: &str) -> Option<DateTime<Utc>> {
    let idx = row.columns().iter().position(|c| c == column)?;
    let value = row.values().get(idx)?;
    match value {
        Value::Temporal(t) => {
            let millis = t.epoch_millis()?;
            DateTime::<Utc>::from_timestamp_millis(millis)
        }
        Value::String(s) if !s.is_empty() => DateTime::parse_from_rfc3339(s)
            .ok()
            .map(|d| d.with_timezone(&Utc)),
        _ => None,
    }
}

/// Parse a `Value::Temporal` or RFC-3339 `Value::String` into a UTC
/// datetime, surfacing a precise [`UnikoError::Storage`] when neither
/// branch yields a valid timestamp.
///
/// Use this when an absent / malformed timestamp is genuinely an
/// error (e.g. an Episode that *must* carry a write time); use
/// [`extract_optional_dt`] when absence is fine.
///
/// # Errors
///
/// Returns [`UnikoError::Storage`] if the value is not a Temporal /
/// String, or the millis fall outside the supported range, or the
/// RFC-3339 parse fails.
pub(crate) fn require_datetime(
    value: &Value,
    context: &str,
) -> Result<DateTime<Utc>, UnikoError> {
    match value {
        Value::Temporal(t) => {
            let millis = t
                .epoch_millis()
                .ok_or_else(|| UnikoError::Storage(format!("{context} has no epoch")))?;
            DateTime::<Utc>::from_timestamp_millis(millis)
                .ok_or_else(|| UnikoError::Storage(format!("epoch millis {millis} out of range")))
        }
        Value::String(s) => DateTime::parse_from_rfc3339(s)
            .map(|d| d.with_timezone(&Utc))
            .map_err(|e| UnikoError::Storage(e.to_string())),
        other => Err(UnikoError::Storage(format!(
            "{context} unexpected type: {other:?}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn scalars_roundtrip() {
        assert!(matches!(json_to_value(&JsonValue::Null), Value::Null));
        assert!(matches!(json_to_value(&json!(true)), Value::Bool(true)));
        assert!(matches!(json_to_value(&json!(42)), Value::Int(42)));
        match json_to_value(&json!(2.5)) {
            Value::Float(x) => assert!((x - 2.5).abs() < 1e-9),
            other => panic!("expected Float, got {other:?}"),
        }
    }

    #[test]
    fn nested_object_becomes_map() {
        let v = json_to_value(&json!({ "k": [1, 2] }));
        match v {
            Value::Map(m) => match m.get("k") {
                Some(Value::List(items)) => assert_eq!(items.len(), 2),
                other => panic!("expected nested List, got {other:?}"),
            },
            other => panic!("expected Map, got {other:?}"),
        }
    }
}
