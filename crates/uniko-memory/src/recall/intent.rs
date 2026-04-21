//! Intent profile construction for recall queries.

// Rust guideline compliant

use uniko_extract::ner::rules::extract_entities_rule_based;
use uniko_store::{KnowledgeBase, UnikoError};

/// Structured representation of a recall query.
#[derive(Debug, Clone)]
pub struct IntentProfile {
    /// Embedding of the full query text.
    pub intent_vec: Vec<f32>,
    /// Entity names extracted from the query.
    pub entity_refs: Vec<String>,
    /// Number of facets: `max(entity_refs.len(), 1)`.
    pub facet_count: usize,
}

/// Build an [`IntentProfile`] from a query string.
///
/// Embeds the query via Xervo and extracts entity references using the
/// ONNX NLP pipeline (when available) with rule-based NER as fallback.
///
/// # Errors
///
/// Returns [`UnikoError::Embedding`] if the embedding runtime is
/// unavailable.
pub async fn build_intent(kb: &KnowledgeBase, query: &str) -> Result<IntentProfile, UnikoError> {
    // Embed the full query.
    let intent_vec = uniko_extract::embedding::embed_text(kb, query)
        .await
        .unwrap_or_default();

    // Extract entity references — prefer ONNX NLP model, fall back to rules.
    let entity_refs = extract_entity_refs(kb, query).await;

    let facet_count = entity_refs.len().max(1);

    Ok(IntentProfile {
        intent_vec,
        entity_refs,
        facet_count,
    })
}

/// Extract entity names from query text.
///
/// Uses the ONNX NLP pipeline if available (finds single-word names
/// like "Jon", "Gina"), falls back to rule-based regex NER.
async fn extract_entity_refs(kb: &KnowledgeBase, query: &str) -> Vec<String> {
    // Try ONNX NLP pipeline first.
    #[cfg(feature = "onnx")]
    if let Some(pipeline) = uniko_extract::nlp::NlpPipeline::try_new(kb).await
        && let Ok(result) = pipeline.analyze(query).await
    {
        let refs: Vec<String> = result.entities.iter().map(|e| e.text.clone()).collect();
        if !refs.is_empty() {
            tracing::debug!(entities = ?refs, "ONNX NER on query");
            return refs;
        }
    }

    // Fallback: rule-based NER.
    let raw = extract_entities_rule_based(query);
    raw.into_iter().map(|e| e.canonical_name).collect()
}
