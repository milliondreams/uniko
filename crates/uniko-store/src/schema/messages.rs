//! Layer 2: Message node type.

use uni_db::{DataType, IndexType, ScalarType, SchemaBuilder};

use super::constants::{edges, labels, DEFAULT_VECTOR_DIM};
use super::hnsw_auto_embed_index;

pub(crate) fn register_labels(builder: SchemaBuilder<'_>) -> SchemaBuilder<'_> {
    builder
        .label(labels::MESSAGE)
            .property("message_id", DataType::String)
            .property("content", DataType::String)
            .property_nullable("content_type", DataType::String)
            .property("timestamp", DataType::DateTime)
            .property_nullable("embedding", DataType::Vector { dimensions: DEFAULT_VECTOR_DIM })
            .index("message_id", IndexType::Scalar(ScalarType::Hash))
            .index("timestamp", IndexType::Scalar(ScalarType::BTree))
            .index("content", IndexType::FullText)
            .index("embedding", IndexType::Vector(hnsw_auto_embed_index("content")))
        .done()
}

pub(crate) fn register_edges(builder: SchemaBuilder<'_>) -> SchemaBuilder<'_> {
    builder
        .edge_type(edges::SENT_BY, &[labels::MESSAGE], &[labels::PARTICIPANT])
            .property_nullable("role", DataType::String)
        .done()
        .edge_type(edges::ADDRESSED_TO, &[labels::MESSAGE], &[labels::PARTICIPANT])
        .done()
        // IN_SESSION: multi-source (Message, Action, Episode → Session)
        .edge_type(
            edges::IN_SESSION,
            &[labels::MESSAGE, labels::ACTION, labels::EPISODE],
            &[labels::SESSION],
        )
        .done()
        .edge_type(edges::NEXT, &[labels::MESSAGE], &[labels::MESSAGE])
            .property_nullable("gap_ms", DataType::Int64)
        .done()
}
