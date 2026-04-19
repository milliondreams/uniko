//! Embedding computation helpers via uni-db's Xervo runtime.
//!
//! Auto-embed (Message, Chunk, Observation, Summary) is handled
//! automatically by uni-db when the schema configures
//! [`EmbeddingCfg`](uni_db::api::schema::EmbeddingCfg) on the vector
//! index.  No application code is needed for those nodes.
//!
//! This module provides helpers for *computed* embeddings — nodes
//! whose embedding text is synthesized from multiple properties
//! (e.g. Entity: `"name (type)"`).

// Rust guideline compliant

use std::collections::HashMap;

use uni_db::Value;

use uniko_store::schema::EMBED_ALIAS;
use uniko_store::{KnowledgeBase, NodeId, UnikoError};

/// Embed a single text string via uni-db's Xervo runtime.
///
/// Uses the default fastembed model (`"embed/default"`,
/// all-MiniLM-L6-v2, 384 dimensions).
///
/// # Errors
///
/// Returns [`UnikoError::Embedding`] if the Xervo runtime is not
/// available or the model fails.
pub async fn embed_text(kb: &KnowledgeBase, text: &str) -> Result<Vec<f32>, UnikoError> {
    let xervo = kb.db().xervo();
    if !xervo.is_available() {
        return Err(UnikoError::Embedding(
            "Xervo embedding runtime not available".into(),
        ));
    }
    let results = xervo
        .embed(EMBED_ALIAS, &[text])
        .await
        .map_err(|e| UnikoError::Embedding(e.to_string()))?;
    results
        .into_iter()
        .next()
        .ok_or_else(|| UnikoError::Embedding("empty embedding result".into()))
}

/// Embed multiple texts in a single batch call.
///
/// # Errors
///
/// Returns [`UnikoError::Embedding`] if the runtime is unavailable.
pub async fn embed_batch(kb: &KnowledgeBase, texts: &[&str]) -> Result<Vec<Vec<f32>>, UnikoError> {
    let xervo = kb.db().xervo();
    if !xervo.is_available() {
        return Err(UnikoError::Embedding(
            "Xervo embedding runtime not available".into(),
        ));
    }
    xervo
        .embed(EMBED_ALIAS, texts)
        .await
        .map_err(|e| UnikoError::Embedding(e.to_string()))
}

/// Compute and store an embedding for an Entity node.
///
/// Formula: `"name (entity_type)"` or just `"name"` if the type is
/// unknown.  The computed vector is written to the node's `embedding`
/// property.
///
/// # Errors
///
/// Returns [`UnikoError::Embedding`] if embedding fails, or
/// [`UnikoError::Storage`] if the node update fails.
pub async fn embed_entity(
    kb: &KnowledgeBase,
    node_id: NodeId,
    name: &str,
    entity_type: Option<&str>,
) -> Result<Vec<f32>, UnikoError> {
    let embed_text_str = match entity_type {
        Some(t) if !t.is_empty() => format!("{name} ({t})"),
        _ => name.to_string(),
    };

    let vec = embed_text(kb, &embed_text_str).await?;

    // Store the computed embedding on the node.
    let mut props = HashMap::new();
    props.insert("embedding".into(), Value::Vector(vec.clone()));
    kb.update_node(node_id, &props).await?;

    Ok(vec)
}
