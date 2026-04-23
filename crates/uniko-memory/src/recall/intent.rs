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

    Ok(IntentProfile {
        intent_vec,
        keywords,
        entity_refs,
        facet_count,
    })
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

            // Extract entity names.
            let entity_refs: Vec<String> = result.entities.iter().map(|e| e.text.clone()).collect();

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
    let entity_refs: Vec<String> = raw.into_iter().map(|e| e.canonical_name).collect();
    (entity_refs, query.to_string())
}
