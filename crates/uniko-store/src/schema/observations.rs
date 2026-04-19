//! Layer 4: Observation node type.

use uni_db::{DataType, IndexType, ScalarType, SchemaBuilder};

use super::constants::{edges, labels};
use crate::config::UnikoConfig;

pub(crate) fn register_labels<'a>(
    builder: SchemaBuilder<'a>,
    config: &UnikoConfig,
) -> SchemaBuilder<'a> {
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
        .index("content", IndexType::FullText)
        .index("subject", IndexType::Scalar(ScalarType::Hash))
        .index("subject", IndexType::FullText)
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
        // ABOUT: multi-source (Observation, Fact → Entity)
        .edge_type(
            edges::ABOUT,
            &[labels::OBSERVATION, labels::FACT],
            &[labels::ENTITY],
        )
        .done()
}
