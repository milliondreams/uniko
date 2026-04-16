//! Layer 4: Summary node type.

use uni_db::{DataType, IndexType, SchemaBuilder};

use super::constants::{edges, labels, DEFAULT_VECTOR_DIM};
use super::hnsw_auto_embed_index;

pub(crate) fn register_labels(builder: SchemaBuilder<'_>) -> SchemaBuilder<'_> {
    builder
        .label(labels::SUMMARY)
            .property("summary_id", DataType::String)
            .property("text", DataType::String)
            .property_nullable("level", DataType::String)
            .property_nullable("generated_at", DataType::DateTime)
            .property_nullable("embedding", DataType::Vector { dimensions: DEFAULT_VECTOR_DIM })
            .index("embedding", IndexType::Vector(hnsw_auto_embed_index("text")))
        .done()
}

pub(crate) fn register_edges(builder: SchemaBuilder<'_>) -> SchemaBuilder<'_> {
    builder
        // SUMMARIZES: multi-target (Summary → Session, Task, Goal, Artifact, Entity, Topic)
        .edge_type(
            edges::SUMMARIZES,
            &[labels::SUMMARY],
            &[
                labels::SESSION,
                labels::TASK,
                labels::GOAL,
                labels::ARTIFACT,
                labels::ENTITY,
                labels::TOPIC,
            ],
        )
        .done()
}
