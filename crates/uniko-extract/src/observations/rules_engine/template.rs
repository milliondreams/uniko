//! Tiny `{name}` template renderer.
//!
//! Used by [`super::matcher`] to turn captures into the observation
//! `content` string. Missing captures collapse to empty strings;
//! adjacent whitespace is normalised so a missing `{object}` doesn't
//! leave a double space behind.

use std::collections::HashMap;

/// Render `template` by substituting `{name}` markers with values from
/// `captures`. Missing keys become empty strings, then runs of
/// whitespace are squeezed to a single space and the result is
/// trimmed.
pub fn render(template: &str, captures: &HashMap<String, String>) -> String {
    let mut out = String::with_capacity(template.len());
    let mut chars = template.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '{' {
            let mut key = String::new();
            let mut closed = false;
            for next in chars.by_ref() {
                if next == '}' {
                    closed = true;
                    break;
                }
                key.push(next);
            }
            if !closed {
                out.push('{');
                out.push_str(&key);
                continue;
            }
            if let Some(v) = captures.get(&key) {
                out.push_str(v);
            }
        } else {
            out.push(c);
        }
    }
    squeeze_whitespace(&out)
}

fn squeeze_whitespace(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_space = false;
    for c in s.chars() {
        if c.is_whitespace() {
            if !prev_space {
                out.push(' ');
            }
            prev_space = true;
        } else {
            out.push(c);
            prev_space = false;
        }
    }
    out.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn caps(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect()
    }

    #[test]
    fn substitutes_captures() {
        let t = "{subject} {anchor} {object}";
        let c = caps(&[("subject", "Jon"), ("anchor", "got"), ("object", "a job")]);
        assert_eq!(render(t, &c), "Jon got a job");
    }

    #[test]
    fn missing_capture_collapses_whitespace() {
        let t = "{subject} {anchor} {object} {modifiers}";
        let c = caps(&[("subject", "Jon"), ("anchor", "is"), ("object", "happy")]);
        assert_eq!(render(t, &c), "Jon is happy");
    }

    #[test]
    fn literal_words_in_template() {
        let t = "{subject} is {anchor}";
        let c = caps(&[("subject", "Jon"), ("anchor", "happy")]);
        assert_eq!(render(t, &c), "Jon is happy");
    }
}
