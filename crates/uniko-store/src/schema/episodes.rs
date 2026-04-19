//! Layer 2: Episode node type.

use uni_db::{DataType, IndexType, ScalarType, SchemaBuilder};

use super::constants::{edges, labels};
use crate::config::UnikoConfig;

pub(crate) fn register_labels<'a>(
    builder: SchemaBuilder<'a>,
    config: &UnikoConfig,
) -> SchemaBuilder<'a> {
    builder
        .label(labels::EPISODE)
        .property("episode_id", DataType::String)
        .property("action_type", DataType::String)
        .property_nullable("outcome", DataType::String)
        .property_nullable("state", DataType::CypherValue)
        .property_nullable("delta", DataType::CypherValue)
        .property_nullable("importance", DataType::Float64)
        .property("timestamp", DataType::DateTime)
        .property_nullable(
            "embedding",
            DataType::Vector {
                dimensions: config.embedding.dimensions,
            },
        )
        .index("episode_id", IndexType::Scalar(ScalarType::Hash))
        .index("action_type", IndexType::Scalar(ScalarType::Hash))
        .index("outcome", IndexType::Scalar(ScalarType::Hash))
        .index("timestamp", IndexType::Scalar(ScalarType::BTree))
        .index("embedding", IndexType::Vector(super::vector_index(config)))
        .done()
}

pub(crate) fn register_edges(builder: SchemaBuilder<'_>) -> SchemaBuilder<'_> {
    builder
        .edge_type(
            edges::RECORDED_BY,
            &[labels::EPISODE],
            &[labels::PARTICIPANT],
        )
        .done()
        .edge_type(edges::INVOLVES, &[labels::EPISODE], &[labels::ACTION])
        .done()
        .edge_type(edges::FOLLOWED_BY, &[labels::EPISODE], &[labels::EPISODE])
        .property_nullable("gap_ms", DataType::Int64)
        .done()
}
