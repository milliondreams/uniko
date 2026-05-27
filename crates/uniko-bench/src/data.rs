//! LoCoMo dataset loader and types.
//!
//! Parses `locomo10.json` into typed Rust structures.  Sessions use
//! dynamic JSON keys (`session_1`, `session_1_date_time`, etc.) which
//! are extracted via regex after deserialization.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use regex::Regex;
use serde::Deserialize;

/// A single LoCoMo conversation sample with QA pairs.
#[derive(Debug, Deserialize)]
pub struct LocomoSample {
    pub sample_id: String,
    pub conversation: Conversation,
    pub qa: Vec<QaPair>,
}

/// Raw conversation with dynamic session keys.
#[derive(Debug, Deserialize)]
pub struct Conversation {
    pub speaker_a: String,
    pub speaker_b: String,
    #[serde(flatten)]
    pub raw: HashMap<String, serde_json::Value>,
}

/// A single dialog turn within a session.
#[derive(Debug, Clone, Deserialize)]
pub struct DialogTurn {
    pub speaker: String,
    pub dia_id: String,
    #[serde(default)]
    pub text: String,
    /// Image URLs attached to this turn (LoCoMo multimodal data).
    #[serde(default)]
    pub img_url: Option<Vec<String>>,
    /// BLIP-generated caption describing the image.
    #[serde(default)]
    pub blip_caption: Option<String>,
    /// Search query/intent associated with the image.
    #[serde(default)]
    pub query: Option<String>,
}

use crate::string_or_number;

/// A question-answer pair with evaluation metadata.
#[derive(Debug, Deserialize)]
pub struct QaPair {
    pub question: String,
    #[serde(default, deserialize_with = "string_or_number")]
    pub answer: Option<String>,
    #[serde(default, deserialize_with = "string_or_number")]
    pub adversarial_answer: Option<String>,
    #[serde(default)]
    pub evidence: Vec<String>,
    pub category: u32,
}

impl QaPair {
    /// Gold answer for evaluation (adversarial uses adversarial_answer).
    pub fn gold_answer(&self) -> &str {
        self.answer
            .as_deref()
            .or(self.adversarial_answer.as_deref())
            .unwrap_or("")
    }

    /// Question category as an enum.
    pub fn question_category(&self) -> QuestionCategory {
        QuestionCategory::from_id(self.category)
    }
}

/// LoCoMo question categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuestionCategory {
    MultiHop,
    Temporal,
    OpenDomain,
    SingleHop,
    Adversarial,
}

impl QuestionCategory {
    /// Parse from the integer category ID in the dataset.
    pub fn from_id(id: u32) -> Self {
        match id {
            1 => Self::MultiHop,
            2 => Self::Temporal,
            3 => Self::OpenDomain,
            4 => Self::SingleHop,
            5 => Self::Adversarial,
            _ => Self::SingleHop,
        }
    }

    /// Display name.
    pub fn name(self) -> &'static str {
        match self {
            Self::MultiHop => "Multi-hop",
            Self::Temporal => "Temporal",
            Self::OpenDomain => "Open-domain",
            Self::SingleHop => "Single-hop",
            Self::Adversarial => "Adversarial",
        }
    }
}

/// A parsed session with timestamp and turns.
#[derive(Debug)]
pub struct ParsedSession {
    /// Session number (1-based).
    pub session_number: u32,
    /// Unique session ID for graph storage.
    pub session_id: String,
    /// Raw datetime string from the dataset.
    pub date_time: String,
    /// Ordered dialog turns.
    pub turns: Vec<DialogTurn>,
}

/// Load the LoCoMo dataset from a JSON file.
///
/// # Errors
///
/// Returns an error if the file cannot be read or parsed.
pub fn load_locomo(path: impl AsRef<Path>) -> Result<Vec<LocomoSample>> {
    let content = std::fs::read_to_string(path.as_ref())
        .with_context(|| format!("reading {}", path.as_ref().display()))?;
    let samples: Vec<LocomoSample> =
        serde_json::from_str(&content).context("parsing locomo JSON")?;
    Ok(samples)
}

/// Extract sessions from the flattened conversation JSON.
///
/// Finds `session_N_date_time` and `session_N` keys, pairs them,
/// and returns sorted sessions.
pub fn parse_sessions(sample_id: &str, conv: &Conversation) -> Result<Vec<ParsedSession>> {
    let session_re = Regex::new(r"^session_(\d+)$").unwrap();
    let datetime_re = Regex::new(r"^session_(\d+)_date_time$").unwrap();

    // Collect datetime strings.
    let mut datetimes: HashMap<u32, String> = HashMap::new();
    for (key, val) in &conv.raw {
        if let Some(caps) = datetime_re.captures(key) {
            let n: u32 = caps[1].parse().unwrap_or(0);
            if let Some(dt) = val.as_str() {
                datetimes.insert(n, dt.to_string());
            }
        }
    }

    // Collect session turn arrays.
    let mut sessions = Vec::new();
    for (key, val) in &conv.raw {
        if let Some(caps) = session_re.captures(key) {
            let n: u32 = caps[1].parse().unwrap_or(0);
            let turns: Vec<DialogTurn> = serde_json::from_value(val.clone())
                .with_context(|| format!("parsing turns for {key}"))?;
            let dt = datetimes
                .get(&n)
                .cloned()
                .unwrap_or_else(|| "2023-01-01T00:00:00".to_string());
            sessions.push(ParsedSession {
                session_number: n,
                session_id: format!("{sample_id}-session-{n}"),
                date_time: dt,
                turns,
            });
        }
    }

    sessions.sort_by_key(|s| s.session_number);
    Ok(sessions)
}

/// Build a `dia_id → text` lookup from parsed sessions.
///
/// For image turns, appends `[image: caption | query]` so that
/// evidence substring matching can find image-only evidence.
pub fn build_evidence_lookup(sessions: &[ParsedSession]) -> HashMap<String, String> {
    let mut lookup = HashMap::new();
    for session in sessions {
        for turn in &session.turns {
            let mut text = turn.text.clone();
            if let Some(ref caption) = turn.blip_caption
                && !caption.is_empty()
            {
                if !text.is_empty() {
                    text.push(' ');
                }
                text.push_str(caption);
            }
            if let Some(ref query) = turn.query
                && !query.is_empty()
            {
                if !text.is_empty() {
                    text.push(' ');
                }
                text.push_str(query);
            }
            lookup.insert(turn.dia_id.clone(), text);
        }
    }
    lookup
}

/// Resolve evidence dia_ids to their message texts.
pub fn resolve_evidence(qa: &QaPair, lookup: &HashMap<String, String>) -> Vec<String> {
    qa.evidence
        .iter()
        .filter_map(|dia_id| lookup.get(dia_id).cloned())
        .collect()
}
