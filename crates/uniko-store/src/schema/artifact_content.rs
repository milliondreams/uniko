//! Layer 3: `:ArtifactContent` — content-addressed metadata/index node
//! for every unique blob.
//
use uni_db::{DataType, IndexType, ScalarType, SchemaBuilder};

use super::constants::{edges, labels};

/// Register the `:ArtifactContent` label and its indexes.
///
/// `:ArtifactContent` is the graph-side metadata node for every unique
/// blob, deduplicated by `content_id` (SHA-256 hex). Bytes live either
/// inline (`bytes` populated, `uri` NULL) when the KB's blob backend is
/// Lance, or externally (`bytes` NULL, `uri` populated) for `Fs` / `S3`
/// backends. The shape is identical across backends — only the path to
/// the bytes differs.
pub(crate) fn register_labels(builder: SchemaBuilder<'_>) -> SchemaBuilder<'_> {
    builder
        .label(labels::ARTIFACT_CONTENT)
        .property("content_id", DataType::String)
        .property_nullable("bytes", DataType::Bytes)
        .property_nullable("uri", DataType::String)
        .property("mime", DataType::String)
        .property("size", DataType::Int64)
        .property_nullable("perceptual_hash", DataType::Int64)
        .property_nullable("audio_fingerprint", DataType::Bytes)
        .property("created_at", DataType::DateTime)
        .index("content_id", IndexType::Scalar(ScalarType::Hash))
        .index("perceptual_hash", IndexType::Scalar(ScalarType::BTree))
        .index("mime", IndexType::Scalar(ScalarType::Hash))
        .index("size", IndexType::Scalar(ScalarType::BTree))
        .done()
}

/// Register the `HAS_CONTENT` edge: `Artifact → ArtifactContent`.
pub(crate) fn register_edges(builder: SchemaBuilder<'_>) -> SchemaBuilder<'_> {
    builder
        .edge_type(
            edges::HAS_CONTENT,
            &[labels::ARTIFACT],
            &[labels::ARTIFACT_CONTENT],
        )
        .property_nullable("role", DataType::String)
        .done()
}
