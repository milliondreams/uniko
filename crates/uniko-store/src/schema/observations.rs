//! Layer 4: Observation node type.

use uni_db::{DataType, IndexType, ScalarType, SchemaBuilder};

use super::constants::{edges, labels};
use crate::config::UnikoConfig;

pub(crate) fn register_labels<'a>(
    builder: SchemaBuilder<'a>,
    config: &UnikoConfig,
) -> SchemaBuilder<'a> {
    // Observations are now first-class retrieval targets: each
    // extracted fact (e.g. "Caroline tough breakup", "Melanie play
    // clarinet") gets its own auto-embedded vector + fulltext index.
    // Recall benefits because observations are already in the
    // claim-form a question's gold answer is closest to.
    builder
        .label(labels::OBSERVATION)
        .property("observation_id", DataType::String)
        .property("content", DataType::String)
        .property_nullable("subject", DataType::String)
        // Structured triple surfaced from the P3 matcher (DEP/SRL paths
        // already produce `(subject, anchor=verb, object, modifiers)` —
        // we now persist the verb and object so P4 Consolidation can
        // group observations by `(subject, predicate)` without
        // re-parsing `content`).  Nullable because rule fallbacks and
        // CLS-only pathways cannot always produce a triple.
        .property_nullable("predicate", DataType::String)
        .property_nullable("object", DataType::String)
        // Phase A (RFE rfe-p4-recall-evolution): surface ARGM-TMP from
        // SRL captures as a structured slot.  `temporal_phrase` keeps
        // the surface form ("Last Fri", "two weekends ago") for
        // traceability; `temporal_anchor` is its absolute resolution
        // via `resolve_temporal()` so downstream consumers can compare
        // dates without re-parsing.  Both are nullable: the DEP path
        // and rule-based fallbacks do not yet populate them.
        .property_nullable("temporal_phrase", DataType::String)
        .property_nullable("temporal_anchor", DataType::DateTime)
        .property_nullable("observed_at", DataType::DateTime)
        .property_nullable("confidence", DataType::Float64)
        .property_nullable(
            "embedding",
            DataType::Vector {
                dimensions: config.embedding.dimensions,
            },
        )
        .index("observation_id", IndexType::Scalar(ScalarType::Hash))
        .index("subject", IndexType::Scalar(ScalarType::Hash))
        .index("predicate", IndexType::Scalar(ScalarType::Hash))
        // BTree on temporal_anchor enables date-range filtering for
        // F39 drift detection (>4 invalidations in 30d) and for
        // temporal-aware recall once Phase C lands.
        .index("temporal_anchor", IndexType::Scalar(ScalarType::BTree))
        .index("content", IndexType::FullText)
        .index(
            "embedding",
            IndexType::Vector(super::auto_embed_vector_index("content", config)),
        )
        .done()
}

pub(crate) fn register_edges(builder: SchemaBuilder<'_>) -> SchemaBuilder<'_> {
    builder
        // OBSERVED_IN: multi-target (Observation → Message, Chunk)
        .edge_type(
            edges::OBSERVED_IN,
            &[labels::OBSERVATION],
            &[labels::MESSAGE, labels::CHUNK],
        )
        .done()
        .edge_type(
            edges::OBSERVED_DURING,
            &[labels::OBSERVATION],
            &[labels::EPISODE],
        )
        .done()
        // ABOUT: multi-source (Observation, Fact, Chunk) → Entity or
        // Participant. Chunk included for observation chunks
        // (chunk_type="observation"). Participant target enables the
        // entity-anchored recall pattern
        // `(c:Chunk)-[:ABOUT]->(:Participant{name})`.
        .edge_type(
            edges::ABOUT,
            &[labels::OBSERVATION, labels::FACT, labels::CHUNK],
            &[labels::ENTITY, labels::PARTICIPANT],
        )
        .done()
}
