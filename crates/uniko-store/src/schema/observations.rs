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
