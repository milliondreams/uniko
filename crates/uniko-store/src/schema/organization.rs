//! Layer 7: Organization and Team node types.

use uni_db::{DataType, IndexType, ScalarType, SchemaBuilder};

use super::constants::{edges, labels};

pub(crate) fn register_labels(builder: SchemaBuilder<'_>) -> SchemaBuilder<'_> {
    builder
        // ── Organization ──
        .label(labels::ORGANIZATION)
            .property("org_id", DataType::String)
            .property_nullable("name", DataType::String)
            .index("org_id", IndexType::Scalar(ScalarType::Hash))
        // ── Team ──
        .label(labels::TEAM)
            .property("team_id", DataType::String)
            .property_nullable("name", DataType::String)
            .property_nullable("purpose", DataType::String)
            .index("team_id", IndexType::Scalar(ScalarType::Hash))
        .done()
}

pub(crate) fn register_edges(builder: SchemaBuilder<'_>) -> SchemaBuilder<'_> {
    builder
        .edge_type(edges::MEMBER_OF, &[labels::PARTICIPANT], &[labels::ORGANIZATION])
            .property_nullable("role", DataType::String)
            .property_nullable("joined_at", DataType::DateTime)
        .done()
        .edge_type(edges::PART_OF_TEAM, &[labels::PARTICIPANT], &[labels::TEAM])
        .done()
        .edge_type(edges::TEAM_IN_ORG, &[labels::TEAM], &[labels::ORGANIZATION])
        .done()
}
