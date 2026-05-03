//! Intent profile construction for recall queries.

// Rust guideline compliant

use uniko_extract::ner::rules::extract_entities_rule_based;
use uniko_store::{KnowledgeBase, UnikoError};

/// Content POS tags to keep for keyword extraction.
/// NOUN, VERB, PROPN, ADJ, NUM — drop function words.
const CONTENT_POS: &[&str] = &["NOUN", "VERB", "PROPN", "ADJ", "NUM"];

/// Structured representation of a recall query.
#[derive(Debug, Clone)]
pub struct IntentProfile {
    /// Embedding of the full query text.
    pub intent_vec: Vec<f32>,
    /// Content keywords extracted from query via POS filtering.
    /// Used as the fulltext search query instead of the raw question.
    pub keywords: String,
    /// Entity names extracted from the query.
    pub entity_refs: Vec<String>,
    /// Number of facets: `max(entity_refs.len(), 1)`.
    pub facet_count: usize,
    /// Predicted entity_type of the answer, when the question's surface
    /// form gives a clear cue ("where" → location, "who" → person,
    /// "how many" → measurement, …). `None` when no rule fires —
    /// downstream code should treat absence as "no signal", not
    /// "anything goes". Populated by [`predict_answer_type`].
    pub expected_answer_type: Option<&'static str>,
}

/// Build an [`IntentProfile`] from a query string.
///
/// Embeds the query via Xervo and extracts entity references + keywords
/// using the ONNX NLP pipeline (when available) with rule-based fallback.
///
/// # Errors
///
/// Returns [`UnikoError::Embedding`] if the embedding runtime is
/// unavailable.
pub async fn build_intent(kb: &KnowledgeBase, query: &str) -> Result<IntentProfile, UnikoError> {
    // Extract entities and keywords via NLP first.
    let (entity_refs, keywords) = analyze_query(kb, query).await;

    // Embed the keywords (not the raw question) — keyword-stripped text
    // embeds closer to statement-form content in the index.
    let embed_text = if keywords != query { &keywords } else { query };
    let intent_vec = uniko_extract::embedding::embed_query(kb, embed_text)
        .await
        .unwrap_or_default();

    let facet_count = entity_refs.len().max(1);
    let expected_answer_type = predict_answer_type(query);

    if let Some(t) = expected_answer_type {
        tracing::info!(expected_answer_type = %t, query = %query, "intent: answer type predicted");
    }

    Ok(IntentProfile {
        intent_vec,
        keywords,
        entity_refs,
        facet_count,
        expected_answer_type,
    })
}

/// Predict the expected entity_type of a question's answer from its
/// surface form. Returns one of the strings stored in `Entity.entity_type`
/// (`"person"`, `"location"`, `"date"`, `"measurement"`,
/// `"organization"`, `"work_of_art"`), or `None` when no rule fires.
///
/// Pure regex over the question text; no model call. Tuned for high
/// precision over recall — emitting a wrong type would actively
/// mislead downstream reweighting, while emitting `None` just falls
/// back to baseline retrieval.
pub fn predict_answer_type(question: &str) -> Option<&'static str> {
    let q = question.trim().to_lowercase();
    // Order matters: more-specific patterns first.
    const RULES: &[(&str, &str)] = &[
        // Person
        (r"^(who|whose|by whom|with whom|to whom)\b", "person"),
        (r"\bwhich (person|friend|coworker|colleague|teacher|doctor|partner)\b", "person"),
        (r"\bwhat is (his|her|their) name\b", "person"),
        // Location
        (r"^where\b|^from where\b|^to where\b", "location"),
        (r"\b(which|what) (city|country|place|state|park|address|venue|building|street|neighborhood|hotel|restaurant|cafe|store|airport)\b", "location"),
        // Date/Time
        (r"^when\b", "date"),
        (r"\bwhat (time|day|date|year|month|hour)\b", "date"),
        (r"\bhow long ago\b", "date"),
        // Measurement / Numeric
        (r"^how (many|much|long|old|far|tall|big|fast|heavy|deep|wide)\b", "measurement"),
        (r"\bhow many (days|years|months|weeks|hours|minutes|seconds|miles|km|kilometers)\b", "measurement"),
        // Organization
        (r"\b(which|what) (company|organization|firm|university|college|school|team|brand|agency)\b", "organization"),
        // Work of art (currently bucketed under `other` in our schema)
        (r"\b(which|what) (movie|book|song|album|game|show|film|novel|poem|painting)\b", "other"),
    ];

    use regex::Regex;
    use std::sync::OnceLock;
    static COMPILED: OnceLock<Vec<(Regex, &'static str)>> = OnceLock::new();
    let compiled = COMPILED.get_or_init(|| {
        RULES
            .iter()
            .map(|(pat, label)| (Regex::new(pat).expect("rule regex"), *label))
            .collect()
    });
    for (re, label) in compiled {
        if re.is_match(&q) {
            return Some(label);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predicts_basic_wh() {
        assert_eq!(predict_answer_type("Who attended the wedding?"), Some("person"));
        assert_eq!(
            predict_answer_type("Where did I attend my cousin's wedding?"),
            Some("location"),
        );
        assert_eq!(
            predict_answer_type("When did I book the Airbnb?"),
            Some("date"),
        );
        assert_eq!(
            predict_answer_type("How many items of clothing do I need?"),
            Some("measurement"),
        );
        assert_eq!(
            predict_answer_type("What time do I wake up on Tuesdays?"),
            Some("date"),
        );
        assert_eq!(
            predict_answer_type("What book am I currently reading?"),
            Some("other"),
        );
        assert_eq!(
            predict_answer_type("What university did I attend?"),
            Some("organization"),
        );
    }

    #[test]
    fn unknowns_return_none() {
        assert_eq!(predict_answer_type("What did I buy for my sister?"), None);
        assert_eq!(predict_answer_type("Why did I quit?"), None);
        assert_eq!(
            predict_answer_type("Tell me what was the rotation for Admon."),
            None,
        );
    }
}

/// Analyze query with NLP pipeline to extract entities and content keywords.
///
/// Returns `(entity_refs, keywords)`. Keywords are content words (NOUN,
/// VERB, PROPN, ADJ) joined by spaces — suitable for BM25 fulltext search.
async fn analyze_query(kb: &KnowledgeBase, query: &str) -> (Vec<String>, String) {
    #[cfg(feature = "onnx")]
    {
        let pipeline_opt = uniko_extract::nlp::NlpPipeline::try_new(kb).await;
        if pipeline_opt.is_none() {
            tracing::warn!("NLP pipeline unavailable for query analysis");
        }
        if let Some(pipeline) = pipeline_opt
            && let Ok(result) = pipeline.analyze(query).await
        {
            let labels = uniko_extract::nlp::assets::label_maps();

            // Extract entity names. Normalize for matching against
            // node `name` properties: strip trailing punctuation and
            // possessive `'s`/`'`, drop empty tokens. Without this,
            // `MATCH (:Participant {name: "Caroline's"})` silently
            // returns nothing.
            let entity_refs: Vec<String> = result
                .entities
                .iter()
                .filter_map(|e| normalize_entity_text(&e.text))
                .collect();

            // Extract content keywords by POS tag.
            let keywords: Vec<&str> = result
                .words
                .iter()
                .zip(result.pos_indices.iter())
                .filter_map(|(word, &pos_idx)| {
                    let pos = labels
                        .pos_labels
                        .get(pos_idx)
                        .map(String::as_str)
                        .unwrap_or("");
                    if CONTENT_POS.contains(&pos) {
                        Some(word.as_str())
                    } else {
                        None
                    }
                })
                .collect();

            if !keywords.is_empty() {
                let kw_str = keywords.join(" ");
                tracing::info!(
                    entities = ?entity_refs,
                    keywords = %kw_str,
                    "NLP query analysis",
                );
                return (entity_refs, kw_str);
            }
        }
    }

    // Fallback: rule-based NER, raw query as keywords.
    let raw = extract_entities_rule_based(query);
    let entity_refs: Vec<String> = raw
        .into_iter()
        .filter_map(|e| normalize_entity_text(&e.canonical_name))
        .collect();
    (entity_refs, query.to_string())
}

/// Normalize an entity span captured from a question for matching
/// against stored node `name` properties.
///
/// Strips trailing punctuation (`?`, `.`, `,`, `!`) and possessive
/// suffixes (`'s`, `\u{2019}s`, trailing `'`). Returns `None` for
/// empty / whitespace-only / single-char results (those are almost
/// always extraction noise, not real entities).
fn normalize_entity_text(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    let stripped = trimmed.trim_end_matches(['?', '.', ',', '!', ';', ':', '"']);
    let no_poss = stripped
        .strip_suffix("\u{2019}s")
        .or_else(|| stripped.strip_suffix("'s"))
        .or_else(|| stripped.strip_suffix('\u{2019}'))
        .or_else(|| stripped.strip_suffix('\''))
        .unwrap_or(stripped);
    let cleaned = no_poss.trim().to_string();
    if cleaned.chars().count() < 2 {
        None
    } else {
        Some(cleaned)
    }
}
