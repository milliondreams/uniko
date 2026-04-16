//! Layer 5: Procedure node type.

use uni_db::{DataType, IndexType, ScalarType, SchemaBuilder};

use super::constants::{edges, labels, DEFAULT_VECTOR_DIM};
use super::hnsw_index;

pub(crate) fn register_labels(builder: SchemaBuilder<'_>) -> SchemaBuilder<'_> {
    builder
        .label(labels::PROCEDURE)
            .property("procedure_id", DataType::String)
            .property("name", DataType::String)
            .property_nullable("description", DataType::String)
            .property_nullable("steps", DataType::CypherValue)
            .property_nullable("preconditions", DataType::CypherValue)
            .property_nullable("precondition_rule", DataType::String)
            .property_nullable("parameters", DataType::CypherValue)
            .property_nullable("effectiveness", DataType::Float64)
            .property_nullable("use_count", DataType::Int64)
            .property_nullable("success_count", DataType::Int64)
            .property_nullable("failure_count", DataType::Int64)
            .property_nullable("avg_outcome_delta", DataType::CypherValue)
            .property_nullable("status", DataType::String)
            .property_nullable("created_at", DataType::DateTime)
            .property_nullable("last_used_at", DataType::DateTime)
            .property_nullable("embedding", DataType::Vector { dimensions: DEFAULT_VECTOR_DIM })
            .index("procedure_id", IndexType::Scalar(ScalarType::Hash))
            .index("name", IndexType::FullText)
            .index("status", IndexType::Scalar(ScalarType::Hash))
            .index("embedding", IndexType::Vector(hnsw_index()))
        .done()
}

pub(crate) fn register_edges(builder: SchemaBuilder<'_>) -> SchemaBuilder<'_> {
    builder
        .edge_type(edges::OPERATES_ON, &[labels::PROCEDURE], &[labels::ENTITY])
        .done()
        .edge_type(edges::USED_IN, &[labels::PROCEDURE], &[labels::TASK])
        .done()
}
