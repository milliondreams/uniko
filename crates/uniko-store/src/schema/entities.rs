//! Layer 4: Entity node type.

use uni_db::{DataType, IndexType, ScalarType, SchemaBuilder};

use super::constants::{edges, labels, DEFAULT_VECTOR_DIM};
use super::hnsw_index;

pub(crate) fn register_labels(builder: SchemaBuilder<'_>) -> SchemaBuilder<'_> {
    builder
        .label(labels::ENTITY)
            .property("entity_id", DataType::String)
            .property("name", DataType::String)
            .property_nullable("entity_type", DataType::String)
            .property_nullable("first_seen", DataType::DateTime)
            .property_nullable("last_seen", DataType::DateTime)
            .property_nullable("frequency", DataType::Int64)
            .property_nullable("confidence", DataType::Float64)
            .property_nullable("embedding", DataType::Vector { dimensions: DEFAULT_VECTOR_DIM })
            .index("entity_id", IndexType::Scalar(ScalarType::Hash))
            .index("name", IndexType::Scalar(ScalarType::Hash))
            .index("name", IndexType::FullText)
            .index("entity_type", IndexType::Scalar(ScalarType::Hash))
            .index("embedding", IndexType::Vector(hnsw_index()))
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
