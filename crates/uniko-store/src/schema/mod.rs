//! Schema registration for the uniko cognitive memory system.
//!
//! Provides [`register_schema`] which registers all 20 node types, 47 edge
//! types, and all indexes with the database.  The call is idempotent: running
//! it multiple times on the same database has no visible effect beyond the
//! first.

pub mod btic;
pub mod constants;

mod actions;
mod artifacts;
mod chunks;
mod consolidation;
mod entities;
mod episodes;
mod facts;
mod goals;
mod messages;
mod observations;
mod organization;
mod participants;
mod procedures;
mod rules;
mod sessions;
mod summaries;
mod topics;

use uni_db::api::schema::EmbeddingCfg;
use uni_db::{Uni, VectorAlgo, VectorIndexCfg, VectorMetric};

use crate::error::UnikoError;

pub use constants::{edges, labels};

/// Embedding model alias used by uni-db's Xervo runtime (fastembed).
pub const EMBED_ALIAS: &str = "embed/default";

/// Build a HNSW vector index without auto-embed (for computed embeddings).
pub(crate) fn hnsw_index() -> VectorIndexCfg {
    VectorIndexCfg {
        algorithm: VectorAlgo::Hnsw {
            m: constants::HNSW_M,
            ef_construction: constants::HNSW_EF_CONSTRUCTION,
            partitions: None,
        },
        metric: VectorMetric::Cosine,
        embedding: None,
    }
}

/// Build a HNSW vector index with auto-embed from a source property.
///
/// uni-db automatically computes and stores the embedding when a node
/// is created or the source property is updated.  Uses the fastembed
/// provider (`"embed/default"` alias, all-MiniLM-L6-v2, 384d).
pub(crate) fn hnsw_auto_embed_index(source_property: &str) -> VectorIndexCfg {
    VectorIndexCfg {
        algorithm: VectorAlgo::Hnsw {
            m: constants::HNSW_M,
            ef_construction: constants::HNSW_EF_CONSTRUCTION,
            partitions: None,
        },
        metric: VectorMetric::Cosine,
        embedding: Some(EmbeddingCfg {
            alias: EMBED_ALIAS.to_string(),
            source_properties: vec![source_property.to_string()],
            batch_size: 32,
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
pub async fn register_schema(db: &Uni) -> crate::Result<()> {
    let builder = db.schema();

    // ── Phase 1: labels (properties + indexes) ──
    let builder = participants::register_labels(builder);
    let builder = goals::register_labels(builder);
    let builder = sessions::register_labels(builder);
    let builder = messages::register_labels(builder);
    let builder = actions::register_labels(builder);
    let builder = episodes::register_labels(builder);
    let builder = artifacts::register_labels(builder);
    let builder = chunks::register_labels(builder);
    let builder = entities::register_labels(builder);
    let builder = observations::register_labels(builder);
    let builder = facts::register_labels(builder);
    let builder = topics::register_labels(builder);
    let builder = summaries::register_labels(builder);
    let builder = procedures::register_labels(builder);
    let builder = rules::register_labels(builder);
    let builder = consolidation::register_labels(builder);
    let builder = organization::register_labels(builder);

    // ── Phase 2: edge types ──
    let builder = goals::register_edges(builder);
    let builder = sessions::register_edges(builder);
    let builder = messages::register_edges(builder);
    let builder = actions::register_edges(builder);
    let builder = episodes::register_edges(builder);
    let builder = artifacts::register_edges(builder);
    let builder = chunks::register_edges(builder);
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
