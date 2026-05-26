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
        // Modality + positioning. `modality` is nullable for migration —
        // existing rows are backfilled to `"text"` by the migration; new
        // ingests always set it explicitly. `bbox` is `[x0, y0, x1, y1]`.
        .property_nullable("modality", DataType::String)
        .property_nullable("bbox", DataType::List(Box::new(DataType::Float32)))
        .property_nullable("time_start_ms", DataType::Int64)
        .property_nullable("time_end_ms", DataType::Int64)
        .property_nullable("page_number", DataType::Int32)
        .property_nullable("reading_order", DataType::Int32)
        // Tracks which derivation model produced this chunk. NULL for
        // non-derived chunks (e.g., direct text chunking).
        .property_nullable("source_model_version", DataType::String)
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
        .index("modality", IndexType::Scalar(ScalarType::Hash))
        .index("time_start_ms", IndexType::Scalar(ScalarType::BTree))
        .index("page_number", IndexType::Scalar(ScalarType::BTree))
        .index(
            "embedding",
            IndexType::Vector(super::auto_embed_vector_index("text", config)),
        )
        .done()
}

pub(crate) fn register_edges(builder: SchemaBuilder<'_>) -> SchemaBuilder<'_> {
    builder
        // HAS_CHUNK: multi-source (Artifact, Message, Session → Chunk)
        .edge_type(
            edges::HAS_CHUNK,
            &[labels::ARTIFACT, labels::MESSAGE, labels::SESSION],
            &[labels::CHUNK],
        )
        .property_nullable("index", DataType::Int64)
        .done()
}
