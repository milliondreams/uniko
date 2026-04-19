//! Layer 4: Summary node type.

use uni_db::{DataType, IndexType, SchemaBuilder};

use super::constants::{edges, labels};
use crate::config::UnikoConfig;

pub(crate) fn register_labels<'a>(
    builder: SchemaBuilder<'a>,
    config: &UnikoConfig,
) -> SchemaBuilder<'a> {
    builder
        .label(labels::SUMMARY)
        .property("summary_id", DataType::String)
        .property("text", DataType::String)
        .property_nullable("level", DataType::String)
        .property_nullable("generated_at", DataType::DateTime)
        .property_nullable(
            "embedding",
            DataType::Vector {
                dimensions: config.embedding.dimensions,
            },
        )
        .index(
            "embedding",
            IndexType::Vector(super::auto_embed_vector_index("text", config)),
        )
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
