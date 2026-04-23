//! LongMemEval dataset loader and types.
//!
//! Parses the LongMemEval JSON dataset (e.g., `longmemeval_s_cleaned.json`)
//! into typed Rust structures. Each item is self-contained with its own
//! haystack of chat sessions and a question to answer.

use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

/// A single LongMemEval question item with its haystack context.
#[derive(Debug, Deserialize)]
pub struct LongMemEvalItem {
    pub question_id: String,

    #[serde(deserialize_with = "deserialize_question_type")]
    pub question_type: LmeQuestionType,

    pub question: String,

    #[serde(deserialize_with = "crate::string_or_number")]
    pub answer: Option<String>,

    /// Simulated "current date" when the question is asked.
    /// Format: "YYYY/MM/DD (Day) HH:MM"
    pub question_date: String,

    /// Dates for each session, parallel with `haystack_sessions`.
    pub haystack_dates: Vec<String>,

    /// Session IDs, parallel with `haystack_sessions`.
    pub haystack_session_ids: Vec<String>,

    /// The chat history: list of sessions, each a list of turns.
    pub haystack_sessions: Vec<Vec<LmeMessage>>,

    /// Session IDs that contain evidence for the answer.
    pub answer_session_ids: Vec<String>,
}

/// A single chat turn in a LongMemEval session.
#[derive(Debug, Clone, Deserialize)]
pub struct LmeMessage {
    pub role: String,
    pub content: String,
    /// Whether this turn contains evidence for the answer.
    #[serde(default)]
    pub has_answer: bool,
}

/// LongMemEval question types (6 categories mapping to 5 memory abilities).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LmeQuestionType {
    /// Recall something the user said in a single session.
    SingleSessionUser,
    /// Recall something the assistant said/produced.
    SingleSessionAssistant,
    /// Recall implicit user preferences, generate personalized response.
    SingleSessionPreference,
    /// Synthesize information across 2+ sessions.
    MultiSession,
    /// Reason about when events occurred, time-relative queries.
    TemporalReasoning,
    /// Track when user info changes, return latest value.
    KnowledgeUpdate,
}

impl LmeQuestionType {
    /// Display name for reporting.
    pub fn name(self) -> &'static str {
        match self {
            Self::SingleSessionUser => "SSU",
            Self::SingleSessionAssistant => "SSA",
            Self::SingleSessionPreference => "SSP",
            Self::MultiSession => "MS",
            Self::TemporalReasoning => "TR",
            Self::KnowledgeUpdate => "KU",
        }
    }

    /// Long display name for reporting.
    pub fn long_name(self) -> &'static str {
        match self {
            Self::SingleSessionUser => "Single-Session User",
            Self::SingleSessionAssistant => "Single-Session Assistant",
            Self::SingleSessionPreference => "Single-Session Preference",
            Self::MultiSession => "Multi-Session",
            Self::TemporalReasoning => "Temporal Reasoning",
            Self::KnowledgeUpdate => "Knowledge Update",
        }
    }

    /// Whether this question type is included in Phase 1 evaluation.
    ///
    /// Phase 1 skips temporal-reasoning and knowledge-update
    /// (they require BTIC, which is a Phase 2 feature).
    pub fn is_phase1(self) -> bool {
        matches!(
            self,
            Self::SingleSessionUser | Self::SingleSessionAssistant | Self::MultiSession
        )
    }

    /// Whether this is an abstention question (determined by question_id suffix).
    /// Abstention questions have IDs ending with "_abs".
    pub fn is_abstention(question_id: &str) -> bool {
        question_id.ends_with("_abs")
    }

    /// Parse from the string value in the dataset.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "single-session-user" => Some(Self::SingleSessionUser),
            "single-session-assistant" => Some(Self::SingleSessionAssistant),
            "single-session-preference" => Some(Self::SingleSessionPreference),
            "multi-session" => Some(Self::MultiSession),
            "temporal-reasoning" => Some(Self::TemporalReasoning),
            "knowledge-update" => Some(Self::KnowledgeUpdate),
            _ => None,
        }
    }

    /// Parse from CLI shorthand (ssu, ssa, ssp, ms, tr, ku).
    pub fn from_shorthand(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "ssu" => Some(Self::SingleSessionUser),
            "ssa" => Some(Self::SingleSessionAssistant),
            "ssp" => Some(Self::SingleSessionPreference),
            "ms" => Some(Self::MultiSession),
            "tr" => Some(Self::TemporalReasoning),
            "ku" => Some(Self::KnowledgeUpdate),
            _ => None,
        }
    }
}

impl std::fmt::Display for LmeQuestionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// Custom deserializer for `LmeQuestionType` from its string representation.
fn deserialize_question_type<'de, D>(
    deserializer: D,
) -> std::result::Result<LmeQuestionType, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    LmeQuestionType::parse(&s)
        .ok_or_else(|| serde::de::Error::custom(format!("unknown question type: {s}")))
}

/// Load the LongMemEval dataset from a JSON file.
///
/// The file should contain a JSON array of `LongMemEvalItem` objects
/// (e.g., `longmemeval_s_cleaned.json`).
pub fn load_longmemeval(path: impl AsRef<Path>) -> Result<Vec<LongMemEvalItem>> {
    let content = std::fs::read_to_string(path.as_ref())
        .with_context(|| format!("reading {}", path.as_ref().display()))?;
    let items: Vec<LongMemEvalItem> =
        serde_json::from_str(&content).context("parsing LongMemEval JSON")?;
    Ok(items)
}

/// Gold answer text for evaluation.
pub fn gold_answer(item: &LongMemEvalItem) -> &str {
    item.answer.as_deref().unwrap_or("")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_question_types() {
        assert_eq!(
            LmeQuestionType::parse("single-session-user"),
            Some(LmeQuestionType::SingleSessionUser)
        );
        assert_eq!(
            LmeQuestionType::parse("knowledge-update"),
            Some(LmeQuestionType::KnowledgeUpdate)
        );
        assert_eq!(LmeQuestionType::parse("invalid"), None);
    }

    #[test]
    fn test_phase1_filter() {
        assert!(LmeQuestionType::SingleSessionUser.is_phase1());
        assert!(LmeQuestionType::SingleSessionAssistant.is_phase1());
        assert!(LmeQuestionType::MultiSession.is_phase1());
        assert!(!LmeQuestionType::TemporalReasoning.is_phase1());
        assert!(!LmeQuestionType::KnowledgeUpdate.is_phase1());
        assert!(!LmeQuestionType::SingleSessionPreference.is_phase1());
    }

    #[test]
    fn test_abstention_detection() {
        assert!(LmeQuestionType::is_abstention("gpt4_2655b836_abs"));
        assert!(!LmeQuestionType::is_abstention("gpt4_2655b836"));
    }

    #[test]
    fn test_shorthand_parsing() {
        assert_eq!(
            LmeQuestionType::from_shorthand("ssu"),
            Some(LmeQuestionType::SingleSessionUser)
        );
        assert_eq!(
            LmeQuestionType::from_shorthand("TR"),
            Some(LmeQuestionType::TemporalReasoning)
        );
    }

    #[test]
    fn test_parse_fixture() {
        let json = r#"[
            {
                "question_id": "gpt4_test1",
                "question_type": "single-session-user",
                "question": "What color is the sky?",
                "answer": "blue",
                "question_date": "2023/04/10 (Mon) 23:07",
                "haystack_dates": ["2023/04/09 (Sun) 10:00"],
                "haystack_session_ids": ["answer_abc123"],
                "haystack_sessions": [[
                    {"role": "user", "content": "The sky is blue today", "has_answer": true},
                    {"role": "assistant", "content": "Indeed it is!", "has_answer": false}
                ]],
                "answer_session_ids": ["answer_abc123"]
            },
            {
                "question_id": "gpt4_test2",
                "question_type": "multi-session",
                "question": "How many times did we discuss weather?",
                "answer": 3,
                "question_date": "2023/04/10 (Mon) 23:07",
                "haystack_dates": ["2023/04/09 (Sun) 10:00"],
                "haystack_session_ids": ["answer_def456"],
                "haystack_sessions": [[
                    {"role": "user", "content": "Let's talk about weather", "has_answer": false},
                    {"role": "assistant", "content": "Sure!", "has_answer": false}
                ]],
                "answer_session_ids": ["answer_def456"]
            }
        ]"#;

        let items: Vec<LongMemEvalItem> = serde_json::from_str(json).unwrap();
        assert_eq!(items.len(), 2);

        // String answer
        assert_eq!(items[0].question_type, LmeQuestionType::SingleSessionUser);
        assert_eq!(items[0].answer.as_deref(), Some("blue"));
        assert!(items[0].haystack_sessions[0][0].has_answer);
        assert!(!items[0].haystack_sessions[0][1].has_answer);

        // Integer answer coerced to string
        assert_eq!(items[1].question_type, LmeQuestionType::MultiSession);
        assert_eq!(items[1].answer.as_deref(), Some("3"));
    }
}
