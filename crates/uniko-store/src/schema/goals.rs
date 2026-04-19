//! Layer 1: Goal and Task node types.

use uni_db::{DataType, IndexType, ScalarType, SchemaBuilder};

use super::constants::{edges, labels};
use crate::config::UnikoConfig;

pub(crate) fn register_labels<'a>(
    builder: SchemaBuilder<'a>,
    config: &UnikoConfig,
) -> SchemaBuilder<'a> {
    builder
        // ── Goal ──
        .label(labels::GOAL)
        .property("goal_id", DataType::String)
        .property("title", DataType::String)
        .property_nullable("description", DataType::String)
        .property_nullable("status", DataType::String)
        .property_nullable("metrics", DataType::CypherValue)
        .property_nullable("guardrails", DataType::CypherValue)
        .property_nullable("owner_id", DataType::String)
        .property_nullable("created_at", DataType::DateTime)
        .property_nullable("deadline", DataType::DateTime)
        .property_nullable("completed_at", DataType::DateTime)
        .property_nullable(
            "embedding",
            DataType::Vector {
                dimensions: config.embedding.dimensions,
            },
        )
        .index("goal_id", IndexType::Scalar(ScalarType::Hash))
        .index("status", IndexType::Scalar(ScalarType::Hash))
        .index("title", IndexType::FullText)
        .index("embedding", IndexType::Vector(super::vector_index(config)))
        // ── Task ──
        .label(labels::TASK)
        .property("task_id", DataType::String)
        .property("title", DataType::String)
        .property_nullable("description", DataType::String)
        .property_nullable("status", DataType::String)
        .property_nullable("priority", DataType::Float64)
        .property_nullable("created_at", DataType::DateTime)
        .property_nullable("completed_at", DataType::DateTime)
        .property_nullable(
            "embedding",
            DataType::Vector {
                dimensions: config.embedding.dimensions,
            },
        )
        .index("task_id", IndexType::Scalar(ScalarType::Hash))
        .index("status", IndexType::Scalar(ScalarType::Hash))
        .index("title", IndexType::FullText)
        .index("embedding", IndexType::Vector(super::vector_index(config)))
        .done()
}

pub(crate) fn register_edges(builder: SchemaBuilder<'_>) -> SchemaBuilder<'_> {
    builder
        .edge_type(edges::OWNED_BY, &[labels::GOAL], &[labels::PARTICIPANT])
        .done()
        .edge_type(edges::PARENT_GOAL, &[labels::GOAL], &[labels::GOAL])
        .done()
        .edge_type(edges::PART_OF, &[labels::TASK], &[labels::GOAL])
        .done()
        .edge_type(edges::ASSIGNED_TO, &[labels::TASK], &[labels::PARTICIPANT])
        .done()
        .edge_type(edges::DEPENDS_ON, &[labels::TASK], &[labels::TASK])
        .done()
        .edge_type(edges::SUBTASK_OF, &[labels::TASK], &[labels::TASK])
        .done()
}
