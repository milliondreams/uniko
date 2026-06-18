//! Shared conversion helpers between `serde_json` and `uni_db` value
//! trees.  Centralised here so episode/action/rules don't each carry a
//! copy.

use std::collections::HashMap;

use serde_json::Value as JsonValue;
use uniko_store::Value;

/// Convert a `serde_json::Value` tree to a `uniko_store::Value`.
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

/// Convert a `uniko_store::Value` tree to a `serde_json::Value`.
///
/// The inverse of [`json_to_value`] for the read path (e.g. a Goal's
/// `metrics` blob). `Bytes` / `Temporal` / `Vector` have no natural JSON
/// form, so they degrade to `Null` — read those through typed accessors.
pub(crate) fn value_to_json(value: &Value) -> JsonValue {
    match value {
        Value::Null => JsonValue::Null,
        Value::Bool(b) => JsonValue::Bool(*b),
        Value::Int(i) => JsonValue::Number((*i).into()),
        Value::Float(f) => {
            serde_json::Number::from_f64(*f).map_or(JsonValue::Null, JsonValue::Number)
        }
        Value::String(s) => JsonValue::String(s.clone()),
        Value::List(items) => JsonValue::Array(items.iter().map(value_to_json).collect()),
        Value::Map(m) => {
            let mut obj = serde_json::Map::new();
            for (k, v) in m {
                obj.insert(k.clone(), value_to_json(v));
            }
            JsonValue::Object(obj)
        }
        _ => JsonValue::Null,
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
