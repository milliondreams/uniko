//! Rule-based named entity extraction using regex patterns.
//!
//! Fast, dependency-free baseline that always runs (< 10 ms).  Covers
//! URLs, emails, dates, measurements, preferences, proper nouns, and
//! quoted strings.

// Rust guideline compliant

use std::sync::OnceLock;

use regex::Regex;

use super::types::{EntityType, ExtractionSource, RawEntity};

/// Compiled regex pattern set — initialized once on first use.
struct Patterns {
    url: Regex,
    email: Regex,
    date_iso: Regex,
    date_informal: Regex,
    measurement: Regex,
    preference: Regex,
    quoted: Regex,
    proper_noun: Regex,
}

static PATTERNS: OnceLock<Patterns> = OnceLock::new();

fn patterns() -> &'static Patterns {
    PATTERNS.get_or_init(|| Patterns {
        url: Regex::new(r"https?://[^\s<>\[\]()]+").unwrap(),
        email: Regex::new(r"(?i)\b[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b").unwrap(),
        date_iso: Regex::new(
            r"\b\d{4}[-/]\d{2}[-/]\d{2}(?:T\d{2}:\d{2}(?::\d{2})?(?:Z|[+\-]\d{2}:?\d{2})?)?\b",
        )
        .unwrap(),
        date_informal: Regex::new(
            r"(?i)\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?,?\s*\d{4}\b",
        )
        .unwrap(),
        measurement: Regex::new(
            r"\b\d+(?:\.\d+)?\s*(?:kg|g|mg|lb|oz|km|m|cm|mm|mi|ft|in|MB|GB|TB|KB|ms|s|min|hrs?|%|px)\b",
        )
        .unwrap(),
        preference: Regex::new(
            r"(?i)\bI\s+(?:prefer|like|love|hate|dislike|enjoy)\s+(.{2,40}?)(?:[.,;!?\n]|$)",
        )
        .unwrap(),
        quoted: Regex::new(r#""([^"]{2,60})""#).unwrap(),
        proper_noun: Regex::new(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b").unwrap(),
    })
}

/// Extract entities from text using rule-based regex patterns.
///
/// Returns entities sorted by `start_byte`.
pub fn extract_entities_rule_based(text: &str) -> Vec<RawEntity> {
    if text.is_empty() {
        return Vec::new();
    }

    let p = patterns();
    let mut entities = Vec::new();

    // URLs
    for m in p.url.find_iter(text) {
        entities.push(raw(
            m.as_str(),
            m.as_str(), // preserve case for URLs
            EntityType::Url,
            0.95,
            m.start(),
            m.end(),
        ));
    }

    // Emails
    for m in p.email.find_iter(text) {
        entities.push(raw(
            m.as_str(),
            &m.as_str().to_lowercase(),
            EntityType::Email,
            0.95,
            m.start(),
            m.end(),
        ));
    }

    // ISO dates
    for m in p.date_iso.find_iter(text) {
        entities.push(raw(
            m.as_str(),
            m.as_str(),
            EntityType::Date,
            0.9,
            m.start(),
            m.end(),
        ));
    }

    // Informal dates
    for m in p.date_informal.find_iter(text) {
        entities.push(raw(
            m.as_str(),
            &m.as_str().trim().to_lowercase(),
            EntityType::Date,
            0.85,
            m.start(),
            m.end(),
        ));
    }

    // Measurements
    for m in p.measurement.find_iter(text) {
        entities.push(raw(
            m.as_str(),
            &m.as_str().trim().to_lowercase(),
            EntityType::Measurement,
            0.85,
            m.start(),
            m.end(),
        ));
    }

    // Preferences: "I prefer X" → extract X
    for caps in p.preference.captures_iter(text) {
        if let Some(obj) = caps.get(1) {
            entities.push(raw(
                obj.as_str().trim(),
                &obj.as_str().trim().to_lowercase(),
                EntityType::Preference,
                0.7,
                obj.start(),
                obj.end(),
            ));
        }
    }

    // Quoted strings (meaningful content only)
    for caps in p.quoted.captures_iter(text) {
        let inner = &caps[1];
        if inner.len() >= 2 && inner.chars().any(|c| c.is_alphabetic()) {
            let full = caps.get(0).unwrap();
            entities.push(raw(
                inner,
                &inner.trim().to_lowercase(),
                EntityType::QuotedString,
                0.6,
                full.start(),
                full.end(),
            ));
        }
    }

    // Proper nouns — skip if overlapping with already-found entities
    for m in p.proper_noun.find_iter(text) {
        if !entities
            .iter()
            .any(|e| overlaps(e.start_byte, e.end_byte, m.start(), m.end()))
        {
            entities.push(raw(
                m.as_str(),
                &m.as_str().trim().to_lowercase(),
                EntityType::Person,
                0.7,
                m.start(),
                m.end(),
            ));
        }
    }

    entities.sort_by_key(|e| e.start_byte);
    entities
}

fn raw(
    surface: &str,
    canonical: &str,
    entity_type: EntityType,
    confidence: f64,
    start: usize,
    end: usize,
) -> RawEntity {
    RawEntity {
        surface_form: surface.to_string(),
        canonical_name: canonical.to_string(),
        entity_type,
        confidence,
        source: ExtractionSource::RuleBased,
        start_byte: start,
        end_byte: end,
    }
}

fn overlaps(s1: usize, e1: usize, s2: usize, e2: usize) -> bool {
    s1 < e2 && s2 < e1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_url() {
        let ents = extract_entities_rule_based("Visit https://example.com/path today.");
        assert!(ents.iter().any(|e| e.entity_type == EntityType::Url));
    }

    #[test]
    fn test_email() {
        let ents = extract_entities_rule_based("Contact user@example.com for info.");
        assert!(ents.iter().any(|e| e.entity_type == EntityType::Email));
    }

    #[test]
    fn test_date_iso() {
        let ents = extract_entities_rule_based("Due on 2024-03-15.");
        assert!(ents.iter().any(|e| e.entity_type == EntityType::Date));
    }

    #[test]
    fn test_date_informal() {
        let ents = extract_entities_rule_based("Meeting on January 15, 2024 at noon.");
        assert!(ents.iter().any(|e| e.entity_type == EntityType::Date));
    }

    #[test]
    fn test_measurement() {
        let ents = extract_entities_rule_based("The file is 5 GB in size.");
        assert!(
            ents.iter()
                .any(|e| e.entity_type == EntityType::Measurement)
        );
    }

    #[test]
    fn test_preference() {
        let ents = extract_entities_rule_based("I prefer VSCode over Vim.");
        assert!(ents.iter().any(|e| e.entity_type == EntityType::Preference));
        let pref = ents
            .iter()
            .find(|e| e.entity_type == EntityType::Preference)
            .unwrap();
        assert!(pref.canonical_name.contains("vscode"));
    }

    #[test]
    fn test_proper_noun() {
        let ents = extract_entities_rule_based("I met Caroline Smith yesterday.");
        assert!(
            ents.iter().any(
                |e| e.entity_type == EntityType::Person && e.canonical_name == "caroline smith"
            )
        );
    }

    #[test]
    fn test_quoted_string() {
        let ents = extract_entities_rule_based("She researched \"adoption agencies\" online.");
        assert!(
            ents.iter()
                .any(|e| e.entity_type == EntityType::QuotedString)
        );
    }

    #[test]
    fn test_empty_text() {
        let ents = extract_entities_rule_based("");
        assert!(ents.is_empty());
    }

    #[test]
    fn test_no_false_positives() {
        let ents = extract_entities_rule_based("the quick brown fox jumps over the lazy dog");
        assert!(ents.is_empty(), "common words should not be entities");
    }

    #[test]
    fn test_overlap_prevention() {
        // URL should prevent proper noun from matching its host.
        let ents = extract_entities_rule_based("Visit https://Caroline.Smith.com today.");
        let proper = ents
            .iter()
            .filter(|e| e.entity_type == EntityType::Person)
            .count();
        assert_eq!(proper, 0, "proper noun overlapping URL should be skipped");
    }
}
