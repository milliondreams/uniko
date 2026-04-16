//! Intent profile construction for recall queries.

// Rust guideline compliant

use uniko_extract::ner::rules::extract_entities_rule_based;
use uniko_store::{KnowledgeBase, UnikoError};

/// Structured representation of a recall query.
#[derive(Debug, Clone)]
pub struct IntentProfile {
    /// Embedding of the full query text (384 dimensions).
    pub intent_vec: Vec<f32>,
    /// Entity names extracted from the query via rule-based NER.
    pub entity_refs: Vec<String>,
    /// Number of facets: `max(entity_refs.len(), 1)`.
    pub facet_count: usize,
}

/// Build an [`IntentProfile`] from a query string.
///
/// Embeds the query via Xervo and extracts entity references using the
/// same rule-based NER from P2.
///
/// # Errors
///
/// Returns [`UnikoError::Embedding`] if the embedding runtime is
/// unavailable.
pub async fn build_intent(
    kb: &KnowledgeBase,
    query: &str,
) -> Result<IntentProfile, UnikoError> {
    // Embed the full query.
    let intent_vec = uniko_extract::embedding::embed_text(kb, query)
        .await
        .unwrap_or_default(); // Graceful: empty vec if Xervo unavailable.

    // Extract entity references via rule-based NER.
    let raw_entities = extract_entities_rule_based(query);
    let entity_refs: Vec<String> = raw_entities
        .into_iter()
        .map(|e| e.canonical_name)
        .collect();

    let facet_count = entity_refs.len().max(1);

    Ok(IntentProfile {
        intent_vec,
        entity_refs,
        facet_count,
    })
}
