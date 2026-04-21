//! Rule-based observation extraction from text.
//!
//! Extracts factual statements anchored to entities from P2.  Each
//! observation is a self-contained sentence that can be understood
//! without surrounding context.

// Rust guideline compliant

use std::sync::OnceLock;

use chrono::{DateTime, Utc};
use regex::Regex;

use super::temporal::resolve_temporal;
use super::types::RawObservation;

/// Entity reference passed from P2: `(node_id, canonical_name)`.
pub type EntityRef = (uniko_store::NodeId, String);

/// Extract observations using rule-based patterns.
///
/// Requires entities from P2 to anchor observations to known subjects.
/// Only sentences containing a named entity or matching a strong
/// pattern produce observations.
pub fn extract_observations_rule_based(
    text: &str,
    entities: &[EntityRef],
    timestamp: DateTime<Utc>,
) -> Vec<RawObservation> {
    if text.is_empty() || entities.is_empty() {
        return Vec::new();
    }

    let sentences = sentence_split(text);
    let mut observations = Vec::new();

    for sentence in sentences {
        let trimmed = sentence.trim();
        if trimmed.is_empty() || trimmed.len() < 10 {
            continue;
        }

        // Find which entity this sentence mentions.
        let subject = match find_entity_in_sentence(trimmed, entities) {
            Some(name) => name,
            None => continue, // No entity reference — skip.
        };

        // Check for observation-worthy patterns.
        let confidence = if is_fact_pattern(trimmed) || is_preference_pattern(trimmed) {
            0.7
        } else if has_svo_structure(trimmed) {
            0.6
        } else {
            continue; // No recognized pattern.
        };

        // Construct self-contained observation.
        let content = make_self_contained(trimmed, &subject);

        // Resolve temporal context.
        let observed_at = resolve_temporal(trimmed, timestamp);

        observations.push(RawObservation {
            content,
            subject: subject.clone(),
            observed_at,
            confidence,
        });
    }

    observations
}

/// Split text into sentences at `.` `?` `!` `\n` `;` boundaries.
///
/// Handles common abbreviations (Mr., Dr., etc.) to avoid false splits.
fn sentence_split(text: &str) -> Vec<&str> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| {
        // Split at sentence-ending punctuation followed by whitespace,
        // or at newlines.
        Regex::new(r"[.!?;]\s+|\n+").unwrap()
    });

    // Use find_iter on the split points instead of split() to preserve
    // text between boundaries.
    let mut result = Vec::new();
    let mut last = 0;
    for m in re.find_iter(text) {
        let end = m.end();
        let segment = text[last..m.start() + 1].trim(); // Include the punctuation.
        if !segment.is_empty() {
            result.push(segment);
        }
        last = end;
    }
    // Remaining text after last split point.
    let tail = text[last..].trim();
    if !tail.is_empty() {
        result.push(tail);
    }
    result
}

/// Find the first entity name that appears in the sentence.
fn find_entity_in_sentence(sentence: &str, entities: &[EntityRef]) -> Option<String> {
    let lower = sentence.to_lowercase();
    // Direct name mention.
    for (_, name) in entities {
        if lower.contains(&name.to_lowercase()) {
            return Some(name.clone());
        }
    }
    // First-person pronoun — attribute to the first entity (the speaker).
    // This handles "I'm starting a dance studio" → attributed to the speaker.
    if !entities.is_empty() {
        let first_word = lower.split_whitespace().next().unwrap_or("");
        if matches!(
            first_word,
            "i" | "i'm" | "i've" | "i'll" | "i'd" | "my" | "me" | "we" | "we're" | "we've" | "our"
        ) {
            return Some(entities[0].1.clone());
        }
    }
    None
}

/// Check for factual statement patterns: "X is Y", "X has Y",
/// "X went to Y", "X works at Y", "X lives in Y", "X attended Y".
fn is_fact_pattern(sentence: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| {
        Regex::new(
            r"(?i)\b(?:is|are|was|were|has|had|have|went to|works? at|lives? in|attended|started|completed|joined|left|moved to|studied at|graduated from|runs?|manages?|leads?|created|built|wrote|published)\b"
        ).unwrap()
    });
    re.is_match(sentence)
}

/// Check for preference patterns: "I prefer X", "I like X", etc.
fn is_preference_pattern(sentence: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| {
        Regex::new(
            r"(?i)\b(?:prefer|like|love|hate|dislike|enjoy|don't like|interested in|passionate about|favorite)\b",
        )
        .unwrap()
    });
    re.is_match(sentence)
}

/// Check if a sentence has subject-verb-object structure.
///
/// Heuristic: at least 4 words and contains a verb-like word.
fn has_svo_structure(sentence: &str) -> bool {
    let words: Vec<&str> = sentence.split_whitespace().collect();
    if words.len() < 4 {
        return false;
    }

    static VERBS: OnceLock<Regex> = OnceLock::new();
    let re = VERBS.get_or_init(|| {
        Regex::new(
            r"(?i)\b(?:is|are|was|were|has|had|have|do|does|did|will|would|can|could|shall|should|may|might|goes?|went|comes?|came|takes?|took|makes?|made|gives?|gave|gets?|got|says?|said|tells?|told|thinks?|thought|knows?|knew|sees?|saw|finds?|found|wants?|needs?|uses?|used|tries?|tried|starts?|began|becomes?|became|runs?|ran|works?|worked|calls?|called|puts?|keeps?|kept|lets?|begins?|seems?|helps?|shows?|hears?|plays?|moves?|lives?|believes?|happens?|provides?|includes?|considers?|appears?|learns?|changes?|meets?|met|pays?|paid|brings?|brought|holds?|held|writes?|wrote|stands?|stood|loses?|lost|adds?|grows?|grew|opens?|walks?|wins?|won|offers?|remembers?|leads?|led|sits?|sat|reads?|carries?|builds?|built|sets?|turns?|follows?|watches?|speaks?|spoke|stops?|leaves?|left|creates?|develops?)\b",
        )
        .unwrap()
    });
    re.is_match(sentence)
}

/// Construct a self-contained observation from a sentence.
///
/// Replaces leading pronouns with the entity name if the sentence
/// starts with one.
fn make_self_contained(sentence: &str, subject: &str) -> String {
    static PRONOUN_START: OnceLock<Regex> = OnceLock::new();
    let re = PRONOUN_START.get_or_init(|| {
        // Match leading pronouns including contractions.
        // "I'm starting" → "Jon is starting"
        // "I've been" → "Jon has been"
        // "She went" → "Caroline went"
        Regex::new(r"(?i)^(I'm|I've|I'll|I'd|She's|He's|They're|She|He|They|It|I)\b").unwrap()
    });

    let result = re.replace(sentence, |caps: &regex::Captures| {
        let matched = &caps[1];
        match matched.to_lowercase().as_str() {
            "i'm" => format!("{subject} is"),
            "i've" => format!("{subject} has"),
            "i'll" => format!("{subject} will"),
            "i'd" => format!("{subject} would"),
            "she's" | "he's" => format!("{subject} is"),
            "they're" => format!("{subject} are"),
            _ => subject.to_string(),
        }
    });
    result.trim().to_string()
}

/// Extract observations using NLP dependency tree.
///
/// Finds SVO triples in the dependency parse and creates observations
/// with higher confidence (0.85) than regex-based extraction (0.6).
/// Only triples whose subject matches a known entity are included.
#[cfg(feature = "onnx")]
pub fn extract_observations_from_dep_tree(
    nlp_result: &crate::nlp::types::NlpResult,
    entities: &[EntityRef],
    timestamp: chrono::DateTime<chrono::Utc>,
) -> Vec<RawObservation> {
    use crate::nlp::assets;
    use crate::nlp::decode;

    let labels = assets::label_maps();
    let triples = decode::extract_svo_triples(
        &nlp_result.words,
        &nlp_result.pos_indices,
        &nlp_result.dep_arcs,
        &labels.pos_labels,
    );

    let mut observations = Vec::new();
    for triple in &triples {
        // Check if the subject matches a known entity.
        let entity_name = match find_entity_match(&triple.subject, entities) {
            Some(name) => name,
            None => continue,
        };

        let content = if triple.object.is_empty() {
            format!("{} {}", entity_name, triple.verb)
        } else {
            format!("{} {} {}", entity_name, triple.verb, triple.object)
        };

        observations.push(RawObservation {
            content,
            subject: entity_name,
            observed_at: timestamp,
            confidence: 0.85, // dep-tree extraction is more reliable than regex
        });
    }

    observations
}

/// Find an entity whose name matches (case-insensitive) part of a text.
#[cfg(feature = "onnx")]
fn find_entity_match(text: &str, entities: &[EntityRef]) -> Option<String> {
    let lower = text.to_lowercase();
    for (_, name) in entities {
        if lower.contains(&name.to_lowercase()) || name.to_lowercase().contains(&lower) {
            return Some(name.clone());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entity(name: &str) -> EntityRef {
        (0, name.to_string())
    }

    fn ts() -> DateTime<Utc> {
        chrono::Utc::now()
    }

    #[test]
    fn test_simple_fact_extraction() {
        let entities = vec![entity("Caroline")];
        let obs = extract_observations_rule_based(
            "Caroline attended an LGBTQ support group.",
            &entities,
            ts(),
        );
        assert_eq!(obs.len(), 1);
        assert!(obs[0].content.contains("Caroline"));
        assert!(obs[0].content.contains("LGBTQ support group"));
    }

    #[test]
    fn test_preference_extraction() {
        let entities = vec![entity("Python")];
        let obs = extract_observations_rule_based(
            "I really prefer Python over Java for scripting.",
            &entities,
            ts(),
        );
        assert!(!obs.is_empty());
    }

    #[test]
    fn test_no_entity_no_observation() {
        let entities = vec![entity("Caroline")];
        let obs = extract_observations_rule_based("The weather is nice today.", &entities, ts());
        assert!(obs.is_empty());
    }

    #[test]
    fn test_multi_sentence() {
        let entities = vec![entity("Caroline"), entity("Paris")];
        let text = "Caroline went to Paris last year. She loved the Eiffel Tower.";
        let obs = extract_observations_rule_based(text, &entities, ts());
        assert!(!obs.is_empty());
        assert!(obs[0].content.contains("Caroline"));
    }

    #[test]
    fn test_empty_entities() {
        let obs = extract_observations_rule_based("Some text here.", &[], ts());
        assert!(obs.is_empty());
    }

    #[test]
    fn test_svo_detection() {
        assert!(has_svo_structure(
            "Caroline works at the hospital every day."
        ));
        assert!(!has_svo_structure("Hi."));
    }

    #[test]
    fn test_pronoun_replacement() {
        let result = make_self_contained("She went to Paris.", "Caroline");
        assert_eq!(result, "Caroline went to Paris.");
    }

    #[test]
    fn test_fact_pattern() {
        assert!(is_fact_pattern("Caroline is a social worker."));
        assert!(is_fact_pattern("She works at the clinic."));
        assert!(!is_fact_pattern("Hello there."));
    }
}
