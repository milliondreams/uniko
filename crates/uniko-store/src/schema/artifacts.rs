//! Layer 3: Artifact node type (5 multimodal embedding fields).

use uni_db::{DataType, IndexType, ScalarType, SchemaBuilder};

use super::constants::{edges, labels, DEFAULT_VECTOR_DIM};
use super::hnsw_index;

pub(crate) fn register_labels(builder: SchemaBuilder<'_>) -> SchemaBuilder<'_> {
    builder
        .label(labels::ARTIFACT)
            .property("artifact_id", DataType::String)
            .property("kind", DataType::String)
            .property_nullable("path", DataType::String)
            .property_nullable("content", DataType::String)
            .property_nullable("mime_type", DataType::String)
            .property_nullable("hash", DataType::String)
            .property_nullable("size", DataType::Int64)
            .property_nullable("language", DataType::String)
            .property_nullable("created_at", DataType::DateTime)
            .property_nullable("updated_at", DataType::DateTime)
            // 5 multimodal embedding fields — all nullable
            .property_nullable("text_embedding", DataType::Vector { dimensions: DEFAULT_VECTOR_DIM })
            .property_nullable("image_embedding", DataType::Vector { dimensions: DEFAULT_VECTOR_DIM })
            .property_nullable("audio_embedding", DataType::Vector { dimensions: DEFAULT_VECTOR_DIM })
            .property_nullable("video_embedding", DataType::Vector { dimensions: DEFAULT_VECTOR_DIM })
            .property_nullable("multimodal_embedding", DataType::Vector { dimensions: DEFAULT_VECTOR_DIM })
            .index("artifact_id", IndexType::Scalar(ScalarType::Hash))
            .index("path", IndexType::Scalar(ScalarType::BTree))
            .index("kind", IndexType::Scalar(ScalarType::Hash))
            .index("content", IndexType::FullText)
            .index("language", IndexType::Scalar(ScalarType::Hash))
            .index("mime_type", IndexType::Scalar(ScalarType::Hash))
            .index("text_embedding", IndexType::Vector(hnsw_index()))
            .index("image_embedding", IndexType::Vector(hnsw_index()))
            .index("audio_embedding", IndexType::Vector(hnsw_index()))
            .index("video_embedding", IndexType::Vector(hnsw_index()))
            .index("multimodal_embedding", IndexType::Vector(hnsw_index()))
        .done()
}

pub(crate) fn register_edges(builder: SchemaBuilder<'_>) -> SchemaBuilder<'_> {
    builder
        .edge_type(edges::CREATED_BY, &[labels::ARTIFACT], &[labels::ACTION])
        .done()
        .edge_type(edges::MODIFIED_BY, &[labels::ARTIFACT], &[labels::ACTION])
            .property_nullable("diff_summary", DataType::String)
        .done()
}
