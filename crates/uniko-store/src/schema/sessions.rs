//! Layer 1: Session node type.

use uni_db::{DataType, IndexType, ScalarType, SchemaBuilder};

use super::constants::{edges, labels, DEFAULT_VECTOR_DIM};
use super::hnsw_index;

pub(crate) fn register_labels(builder: SchemaBuilder<'_>) -> SchemaBuilder<'_> {
    builder
        .label(labels::SESSION)
            .property("session_id", DataType::String)
            .property_nullable("topic", DataType::String)
            .property_nullable("summary", DataType::String)
            .property("started_at", DataType::DateTime)
            .property_nullable("ended_at", DataType::DateTime)
            .property_nullable("embedding", DataType::Vector { dimensions: DEFAULT_VECTOR_DIM })
            .index("session_id", IndexType::Scalar(ScalarType::Hash))
            .index("topic", IndexType::FullText)
            .index("started_at", IndexType::Scalar(ScalarType::BTree))
            .index("embedding", IndexType::Vector(hnsw_index()))
        .done()
}

pub(crate) fn register_edges(builder: SchemaBuilder<'_>) -> SchemaBuilder<'_> {
    builder
        // Session → Task / Goal
        .edge_type(edges::FOR_TASK, &[labels::SESSION, labels::EPISODE], &[labels::TASK])
        .done()
        .edge_type(edges::FOR_GOAL, &[labels::SESSION], &[labels::GOAL])
        .done()
        // Participant → Session (with role)
        .edge_type(edges::PARTICIPATED_IN, &[labels::PARTICIPANT], &[labels::SESSION])
            .property_nullable("role", DataType::String)
        .done()
}
