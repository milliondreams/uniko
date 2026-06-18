//! Schema registration for the uniko cognitive memory system.
//!
//! Provides [`register_schema`] which registers all 20 node types, 47 edge
//! types, and all indexes with the database.  The call is idempotent: running
//! it multiple times on the same database has no visible effect beyond the
//! first.

pub mod btic;
pub mod constants;

mod actions;
mod artifact_content;
mod artifacts;
mod blocks;
mod chunks;
mod consolidation;
mod entities;
mod episodes;
mod facts;
mod goals;
mod kb_stats;
mod messages;
mod observations;
mod organization;
mod pages;
mod participants;
mod procedures;
mod rules;
mod sessions;
mod summaries;
mod topics;

use uni_db::api::schema::EmbeddingCfg;
use uni_db::{Uni, VectorIndexCfg};

use crate::config::UnikoConfig;
use crate::error::UnikoError;

pub use constants::{edges, labels};

/// Embedding model alias used by uni-db's Xervo runtime (ONNX `local/onnx`).
pub const EMBED_ALIAS: &str = "embed/default";

/// NLP ONNX model alias for multi-task inference (NER, POS, dep, CLS).
pub const NLP_ALIAS: &str = "nlp/default";

/// Cross-encoder reranker alias (ONNX `local/onnx`, registered only when enabled).
pub const RERANK_ALIAS: &str = "rerank/default";

/// Pipeline-OCR alias (ONNX `local/onnx`, registered only when enabled).
///
/// Drives the `Ocr` tier of tiered PDF extraction (`uni-xervo-pdf`).
pub const OCR_ALIAS: &str = "ocr/default";

/// Build a vector index from config (no auto-embed).
///
/// Used for node types whose embeddings are computed in application
/// code (Entity, Fact, Episode, etc.).
pub(crate) fn vector_index(config: &UnikoConfig) -> VectorIndexCfg {
    VectorIndexCfg {
        algorithm: config.vector_algorithm.to_uni_algo(),
        metric: config.vector_metric.to_uni_metric(),
        embedding: None,
    }
}

/// Build a vector index with auto-embed from a source property.
///
/// uni-db automatically computes and stores the embedding when a node
/// is created or the source property is updated.  Uses the ONNX
/// embedding provider configured in [`UnikoConfig::embedding`].
pub(crate) fn auto_embed_vector_index(
    source_property: &str,
    config: &UnikoConfig,
) -> VectorIndexCfg {
    VectorIndexCfg {
        algorithm: config.vector_algorithm.to_uni_algo(),
        metric: config.vector_metric.to_uni_metric(),
        embedding: Some(EmbeddingCfg {
            alias: EMBED_ALIAS.to_string(),
            source_properties: vec![source_property.to_string()],
            batch_size: config.embedding.batch_size,
            document_prefix: config.embedding.document_prefix.clone(),
            query_prefix: config.embedding.query_prefix.clone(),
        }),
    }
}

/// Register the complete uniko schema with the database.
///
/// Idempotent: calling multiple times is safe.  Existing labels, properties,
/// edge types, and indexes are silently skipped by uni-db.
///
/// # Errors
///
/// Returns [`UnikoError::Schema`] if any registration step fails for a reason
/// other than "already exists".
pub async fn register_schema(db: &Uni, config: &UnikoConfig) -> crate::Result<()> {
    let builder = db.schema();

    // ── Phase 1: labels (properties + indexes) ──
    let builder = participants::register_labels(builder);
    let builder = goals::register_labels(builder, config);
    let builder = sessions::register_labels(builder, config);
    let builder = messages::register_labels(builder, config);
    let builder = actions::register_labels(builder, config);
    let builder = episodes::register_labels(builder, config);
    let builder = artifacts::register_labels(builder, config);
    let builder = artifact_content::register_labels(builder);
    let builder = chunks::register_labels(builder, config);
    let builder = pages::register_labels(builder, config);
    let builder = blocks::register_labels(builder, config);
    let builder = entities::register_labels(builder, config);
    let builder = observations::register_labels(builder, config);
    let builder = facts::register_labels(builder, config);
    let builder = topics::register_labels(builder, config);
    let builder = summaries::register_labels(builder, config);
    let builder = procedures::register_labels(builder, config);
    let builder = rules::register_labels(builder);
    let builder = consolidation::register_labels(builder);
    let builder = organization::register_labels(builder);
    let builder = kb_stats::register_labels(builder);

    // ── Phase 2: edge types ──
    let builder = goals::register_edges(builder);
    let builder = sessions::register_edges(builder);
    let builder = messages::register_edges(builder);
    let builder = actions::register_edges(builder);
    let builder = episodes::register_edges(builder);
    let builder = artifacts::register_edges(builder);
    let builder = artifact_content::register_edges(builder);
    let builder = chunks::register_edges(builder);
    let builder = pages::register_edges(builder);
    let builder = blocks::register_edges(builder);
    let builder = entities::register_edges(builder);
    let builder = observations::register_edges(builder);
    let builder = facts::register_edges(builder);
    let builder = topics::register_edges(builder);
    let builder = summaries::register_edges(builder);
    let builder = procedures::register_edges(builder);
    let builder = rules::register_edges(builder);
    let builder = consolidation::register_edges(builder);
    let builder = organization::register_edges(builder);

    builder
        .apply()
        .await
        .map_err(|e| UnikoError::Schema(e.to_string()))
}
