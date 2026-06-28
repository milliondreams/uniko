//! Layer 5: Pattern node type.
//!
//! A `:Pattern` records a recurring episode pattern surfaced by the
//! `episode_pattern_detector` stdlib Locy rule — grouped by `(action_type,
//! outcome)`, carrying the occurrence count (`support`) and mean importance.
//! The consumer upserts these by a deterministic `pattern_id`.

use uni_db::{DataType, IndexType, ScalarType, SchemaBuilder};

use super::constants::labels;

pub(crate) fn register_labels(builder: SchemaBuilder<'_>) -> SchemaBuilder<'_> {
    builder
        .label(labels::PATTERN)
        .property("pattern_id", DataType::String)
        .property_nullable("action_type", DataType::String)
        .property_nullable("outcome", DataType::String)
        .property_nullable("support", DataType::Int64)
        .property_nullable("mean_importance", DataType::Float64)
        .property_nullable("status", DataType::String)
        .property_nullable("created_at", DataType::DateTime)
        .property_nullable("last_seen_at", DataType::DateTime)
        .index("pattern_id", IndexType::Scalar(ScalarType::Hash))
        .done()
}
