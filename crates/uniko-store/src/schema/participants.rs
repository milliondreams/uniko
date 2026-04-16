//! Layer 0: Participant node type.

use uni_db::{DataType, IndexType, ScalarType, SchemaBuilder};

use super::constants::labels;

pub(crate) fn register_labels(builder: SchemaBuilder<'_>) -> SchemaBuilder<'_> {
    builder
        .label(labels::PARTICIPANT)
            .property("participant_id", DataType::String)
            .property("kind", DataType::String)
            .property_nullable("name", DataType::String)
            .property_nullable("capabilities", DataType::CypherValue)
            .property_nullable("metadata", DataType::CypherValue)
            .property_nullable("first_seen", DataType::DateTime)
            .property_nullable("last_seen", DataType::DateTime)
            .index("participant_id", IndexType::Scalar(ScalarType::Hash))
            .index("kind", IndexType::Scalar(ScalarType::Hash))
            .index("name", IndexType::FullText)
        .done()
}
