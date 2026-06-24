//! Shared canonical text normalization for name-like graph keys.
//!
//! A single normalizer used wherever a human-text surface form becomes a
//! graph key — `Entity` names, observation subjects/objects, consolidation
//! grouping keys, topic member names — so the same surface form keys
//! identically across the pipeline. Historically each path normalized
//! differently (rules lowercased, the ONNX path title-cased, some paths
//! did neither), which split `"Melanie."`, `"melanie"`, and `"Melanie"`
//! into distinct nodes/groups. This aligns them, and matches
//! [`crate::id::entity_id`], which also lowercases its key.
//!
//! Type-specific forms that must preserve case (URLs, emails, code
//! symbols) deliberately do NOT use this.

/// Normalize a name-like string into a stable canonical key: trim, strip
/// leading/trailing ASCII punctuation, collapse internal whitespace runs
/// to single spaces, and lowercase.
///
/// Idempotent: `normalize_canonical(normalize_canonical(s)) == normalize_canonical(s)`.
///
/// # Examples
/// ```
/// use uniko_store::text::normalize_canonical;
/// assert_eq!(normalize_canonical("Melanie."), "melanie");
/// assert_eq!(normalize_canonical("  ALICE   SMITH! "), "alice smith");
/// assert_eq!(normalize_canonical("yesterday,"), "yesterday");
/// ```
#[must_use]
pub fn normalize_canonical(s: &str) -> String {
    let trimmed = s
        .trim()
        .trim_matches(|c: char| c.is_ascii_punctuation());
    let mut out = String::with_capacity(trimmed.len());
    let mut pending_space = false;
    for ch in trimmed.chars() {
        if ch.is_whitespace() {
            pending_space = !out.is_empty();
        } else {
            if pending_space {
                out.push(' ');
                pending_space = false;
            }
            for lc in ch.to_lowercase() {
                out.push(lc);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::normalize_canonical;

    #[test]
    fn strips_trailing_punctuation() {
        assert_eq!(normalize_canonical("Melanie."), "melanie");
        assert_eq!(normalize_canonical("Mel,"), "mel");
        assert_eq!(normalize_canonical("yesterday!"), "yesterday");
        assert_eq!(normalize_canonical("The Summer?"), "the summer");
    }

    #[test]
    fn lowercases_and_collapses_whitespace() {
        assert_eq!(normalize_canonical("ALICE SMITH"), "alice smith");
        assert_eq!(normalize_canonical("Alice   Smith"), "alice smith");
        assert_eq!(normalize_canonical("  hey   melanie  "), "hey melanie");
    }

    #[test]
    fn collapses_case_and_punct_variants_to_one_key() {
        // The three duplicate forms the probe found must converge.
        let a = normalize_canonical("Melanie.");
        let b = normalize_canonical("melanie");
        let c = normalize_canonical("Melanie");
        assert_eq!(a, b);
        assert_eq!(b, c);
    }

    #[test]
    fn idempotent_and_edge_cases() {
        for s in ["Melanie.", "  ALICE  SMITH! ", "", "...", "café"] {
            let once = normalize_canonical(s);
            assert_eq!(normalize_canonical(&once), once, "not idempotent for {s:?}");
        }
        assert_eq!(normalize_canonical(""), "");
        assert_eq!(normalize_canonical("..."), "");
        assert_eq!(normalize_canonical("Café"), "café");
    }
}
