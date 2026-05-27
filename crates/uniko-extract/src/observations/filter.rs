//! Content filtering to reject non-informative text before extraction.
//!
//! Runs before any observation extraction (< 1 ms).  Filters out
//! greetings, pure questions, system messages, and very short content.

use std::collections::HashSet;
use std::sync::OnceLock;

/// Whether `text` contains potentially informative content worth
/// extracting observations from.
///
/// Returns `false` for greetings, reactions, pure questions, system
/// messages, and content shorter than 5 words.
pub fn is_informative(text: &str, content_type: Option<&str>) -> bool {
    if is_system_message(content_type) {
        return false;
    }
    let trimmed = text.trim();
    if is_too_short(trimmed) {
        return false;
    }
    if is_greeting_or_filler(trimmed) {
        return false;
    }
    if is_pure_question(trimmed) {
        return false;
    }
    true
}

/// Static set of normalized greeting/filler phrases.
fn fillers() -> &'static HashSet<&'static str> {
    static SET: OnceLock<HashSet<&str>> = OnceLock::new();
    SET.get_or_init(|| {
        [
            "hey",
            "hi",
            "hello",
            "hi there",
            "hey there",
            "howdy",
            "thanks",
            "thank you",
            "thx",
            "ty",
            "wow",
            "oh wow",
            "whoa",
            "omg",
            "lol",
            "haha",
            "hehe",
            "lmao",
            "rofl",
            "ok",
            "okay",
            "k",
            "sure",
            "sure thing",
            "yes",
            "yeah",
            "yep",
            "yup",
            "nope",
            "no",
            "right",
            "exactly",
            "agreed",
            "absolutely",
            "no worries",
            "no problem",
            "np",
            "you're welcome",
            "yw",
            "sounds good",
            "sounds great",
            "cool",
            "nice",
            "got it",
            "understood",
            "roger",
            "copy that",
            "bye",
            "goodbye",
            "see you",
            "later",
            "ttyl",
        ]
        .into_iter()
        .collect()
    })
}

fn is_greeting_or_filler(text: &str) -> bool {
    let normalized = text
        .trim_end_matches(|c: char| c.is_ascii_punctuation())
        .trim()
        .to_lowercase();
    fillers().contains(normalized.as_str())
}

fn is_pure_question(text: &str) -> bool {
    let trimmed = text.trim();
    if !trimmed.ends_with('?') {
        return false;
    }

    // Check for embedded-fact patterns first — these contain
    // declarative content even though the sentence ends with `?`.
    let lower = trimmed.to_lowercase();
    let has_embedded_fact = lower.contains("did you know")
        || lower.contains("have you heard")
        || lower.contains("is it true that")
        || lower.contains("do you remember");
    if has_embedded_fact {
        return false;
    }

    // Short questions (< 8 words) without embedded facts are pure.
    let word_count = trimmed.split_whitespace().count();
    word_count < 8
}

fn is_system_message(content_type: Option<&str>) -> bool {
    matches!(content_type, Some("system" | "error"))
}

fn is_too_short(text: &str) -> bool {
    // Count words after stripping leading/trailing punctuation.
    let word_count = text.split_whitespace().count();
    word_count < 5
}

/// Classify informativeness using the NLP model's dialog-act label.
///
/// The cascade emits one of 8 ISO 24617-2 inspired dialog-act labels:
/// `inform`, `request`, `question`, `confirm`, `reject`, `offer`,
/// `social`, `status`.
///
/// Extract from acts that carry propositional content: `inform` and
/// `status` (declarative facts), `request` and `offer` (actionable
/// directives), and `confirm` (positive assertion of a prior fact).
///
/// Reject acts that don't add new information: `question` (asks
/// rather than asserts), `reject` (negates without asserting an
/// alternative), `social` (greetings, phatic).
pub fn is_informative_by_cls(cls_label: &str) -> bool {
    matches!(
        cls_label,
        "inform" | "request" | "confirm" | "offer" | "status"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greetings_rejected() {
        assert!(!is_informative("Hey!", None));
        assert!(!is_informative("Thanks!", None));
        assert!(!is_informative("lol", None));
        assert!(!is_informative("Wow!", None));
        assert!(!is_informative("Sounds good", None));
    }

    #[test]
    fn test_pure_questions_rejected() {
        assert!(!is_informative("Where is the store?", None));
        assert!(!is_informative("What time is it?", None));
    }

    #[test]
    fn test_question_with_fact_kept() {
        assert!(is_informative("Did you know Caroline went to Paris?", None));
    }

    #[test]
    fn test_short_content_rejected() {
        assert!(!is_informative("OK sure", None));
        assert!(!is_informative("yes please", None));
    }

    #[test]
    fn test_system_messages_rejected() {
        assert!(!is_informative("Connection reset by peer", Some("system")));
        assert!(!is_informative("404 Not Found", Some("error")));
    }

    #[test]
    fn test_declarative_kept() {
        assert!(is_informative(
            "Caroline attended an LGBTQ support group yesterday.",
            None
        ));
        assert!(is_informative(
            "The server runs on port 9090 and handles HTTP requests.",
            None
        ));
    }

    #[test]
    fn test_tool_result_text_kept() {
        assert!(is_informative(
            "Build succeeded with 3 warnings on the auth module.",
            Some("tool_result")
        ));
    }
}
