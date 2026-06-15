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
