//! Layer 3: Chunk node type.

use uni_db::{DataType, IndexType, ScalarType, SchemaBuilder};

use super::constants::{edges, labels};
use crate::config::UnikoConfig;

pub(crate) fn register_labels<'a>(
    builder: SchemaBuilder<'a>,
    config: &UnikoConfig,
) -> SchemaBuilder<'a> {
    builder
        .label(labels::CHUNK)
        .property("chunk_id", DataType::String)
        .property("text", DataType::String)
        .property_nullable("index", DataType::Int64)
        .property_nullable("start", DataType::Int64)
        .property_nullable("end", DataType::Int64)
        .property_nullable("token_count", DataType::Int64)
        .property_nullable("chunk_type", DataType::String)
        .property_nullable("language", DataType::String)
        .property_nullable("symbol_name", DataType::String)
        .property_nullable("speaker", DataType::String)
        .property_nullable("heading", DataType::String)
        .property_nullable("mime_type", DataType::String)
        .property_nullable(
            "embedding",
            DataType::Vector {
                dimensions: config.embedding.dimensions,
            },
        )
        .index("text", IndexType::FullText)
        .index("chunk_type", IndexType::Scalar(ScalarType::Hash))
        .index("language", IndexType::Scalar(ScalarType::Hash))
        .index("symbol_name", IndexType::Scalar(ScalarType::Hash))
        .index("speaker", IndexType::Scalar(ScalarType::Hash))
        .index(
            "embedding",
            IndexType::Vector(super::auto_embed_vector_index("text", config)),
        )
        .done()
}

pub(crate) fn register_edges(builder: SchemaBuilder<'_>) -> SchemaBuilder<'_> {
    builder
        // HAS_CHUNK: multi-source (Artifact, Message → Chunk)
        .edge_type(
            edges::HAS_CHUNK,
            &[labels::ARTIFACT, labels::MESSAGE],
            &[labels::CHUNK],
        )
        .property_nullable("index", DataType::Int64)
        .done()
}
