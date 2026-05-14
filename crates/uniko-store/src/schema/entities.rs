//! Layer 4: Entity node type.

use uni_db::{DataType, IndexType, ScalarType, SchemaBuilder};

use super::constants::{edges, labels};
use crate::config::UnikoConfig;

pub(crate) fn register_labels<'a>(
    builder: SchemaBuilder<'a>,
    config: &UnikoConfig,
) -> SchemaBuilder<'a> {
    builder
        .label(labels::ENTITY)
        .property("entity_id", DataType::String)
        .property("name", DataType::String)
        .property_nullable("entity_type", DataType::String)
        .property_nullable("first_seen", DataType::DateTime)
        .property_nullable("last_seen", DataType::DateTime)
        .property_nullable("frequency", DataType::Int64)
        .property_nullable("confidence", DataType::Float64)
        // F39 drift tracking.  `invalidation_count` is the total number
        // of times any Fact with this Entity as subject has been
        // invalidated (BTIC closed) by consolidation.  When the count
        // crosses the configured drift threshold within a 30-day window
        // the `unstable` flag flips to true so recall can force Phase 2+
        // for queries that reference the Entity (drift override).
        .property_nullable("invalidation_count", DataType::Int64)
        .property_nullable("last_invalidation_at", DataType::DateTime)
        .property_nullable("unstable", DataType::Bool)
        .property_nullable(
            "embedding",
            DataType::Vector {
                dimensions: config.embedding.dimensions,
            },
        )
        .index("entity_id", IndexType::Scalar(ScalarType::Hash))
        .index("name", IndexType::Scalar(ScalarType::Hash))
        .index("name", IndexType::FullText)
        .index("entity_type", IndexType::Scalar(ScalarType::Hash))
        .index("unstable", IndexType::Scalar(ScalarType::Hash))
        .index("embedding", IndexType::Vector(super::vector_index(config)))
        .done()
}

pub(crate) fn register_edges(builder: SchemaBuilder<'_>) -> SchemaBuilder<'_> {
    builder
        // MENTIONS: multi-source (Message, Chunk, Action, Artifact, Episode → Entity)
        .edge_type(
            edges::MENTIONS,
            &[
                labels::MESSAGE,
                labels::CHUNK,
                labels::ACTION,
                labels::ARTIFACT,
                labels::EPISODE,
            ],
            &[labels::ENTITY],
        )
        .property_nullable("count", DataType::Int64)
        .done()
}
