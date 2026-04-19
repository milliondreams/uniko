//! Layer 4: Topic node type.

use uni_db::{DataType, IndexType, ScalarType, SchemaBuilder};

use super::constants::{edges, labels};
use crate::config::UnikoConfig;

pub(crate) fn register_labels<'a>(
    builder: SchemaBuilder<'a>,
    config: &UnikoConfig,
) -> SchemaBuilder<'a> {
    builder
        .label(labels::TOPIC)
        .property("topic_id", DataType::String)
        .property("name", DataType::String)
        .property_nullable("summary", DataType::String)
        .property_nullable("entity_count", DataType::Int64)
        .property_nullable("fact_count", DataType::Int64)
        .property_nullable(
            "embedding",
            DataType::Vector {
                dimensions: config.embedding.dimensions,
            },
        )
        .index("topic_id", IndexType::Scalar(ScalarType::Hash))
        .index("name", IndexType::FullText)
        .index("embedding", IndexType::Vector(super::vector_index(config)))
        .done()
}

pub(crate) fn register_edges(builder: SchemaBuilder<'_>) -> SchemaBuilder<'_> {
    builder
        // BELONGS_TO: multi-source (Entity, Fact → Topic)
        .edge_type(
            edges::BELONGS_TO,
            &[labels::ENTITY, labels::FACT],
            &[labels::TOPIC],
        )
        .done()
}
