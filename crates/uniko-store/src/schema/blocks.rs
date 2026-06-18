//! Layer 3: Block node type — an atomic content block within a `:Page`.
//!
//! A `:Block` is one structural unit (paragraph, heading, table, formula, …)
//! produced by tiered PDF extraction, carrying its kind, verbatim text, page
//! geometry (`bbox`), reading order, and source-aware provenance/confidence.
//! Blocks are **atomic** — the embeddable/recallable text rides on a child
//! `:Chunk` (`chunk_type = "block"`), so `:Block` itself has no vector index.
//!
//! `bbox` is stored as four scalar `Float32` columns rather than a list: the
//! arity is fixed at four, scalar columns are directly filterable/orderable in
//! Cypher (e.g. "blocks in the top third of the page"), and this avoids the
//! `List<T>` Arrow type-inference history entirely.

use uni_db::{DataType, IndexType, ScalarType, SchemaBuilder};

use super::constants::{edges, labels};
use crate::config::UnikoConfig;

/// Register the `:Block` label, its properties, and indexes.
pub(crate) fn register_labels<'a>(
    builder: SchemaBuilder<'a>,
    _config: &UnikoConfig,
) -> SchemaBuilder<'a> {
    builder
        .label(labels::BLOCK)
        .property("block_id", DataType::String)
        // DocBlockKind, lowercased: "text"/"heading"/"list"/"table"/"figure"/
        // "formula"/"caption"/"footer"/"header".
        .property("kind", DataType::String)
        .property("text", DataType::String)
        .property_nullable("reading_order", DataType::Int64)
        // Block-level tier ("native" / "ocr" / "vlm").
        .property_nullable("produced_by", DataType::String)
        // Source of the trust score: "deterministic" / "measured" / "derived".
        .property_nullable("confidence_kind", DataType::String)
        // Comparable scalar in [0, 1].
        .property_nullable("confidence_score", DataType::Float32)
        // Page-space bounding box [x0, y0, x1, y1]; NULL when unknown.
        .property_nullable("bbox_x0", DataType::Float32)
        .property_nullable("bbox_y0", DataType::Float32)
        .property_nullable("bbox_x1", DataType::Float32)
        .property_nullable("bbox_y1", DataType::Float32)
        // JSON of the derived-confidence signals; only set for generative
        // (VLM) blocks. NULL for deterministic/measured tiers.
        .property_nullable("confidence_signals", DataType::CypherValue)
        .index("block_id", IndexType::Scalar(ScalarType::Hash))
        .index("kind", IndexType::Scalar(ScalarType::Hash))
        .index("reading_order", IndexType::Scalar(ScalarType::BTree))
        .index("produced_by", IndexType::Scalar(ScalarType::Hash))
        .index("text", IndexType::FullText)
        .done()
}

/// Register `CONTAINS` (`Page` → `Block`) and `NEXT_IN_READING_ORDER`
/// (`Block` → `Block`).
pub(crate) fn register_edges(builder: SchemaBuilder<'_>) -> SchemaBuilder<'_> {
    builder
        .edge_type(edges::CONTAINS, &[labels::PAGE], &[labels::BLOCK])
        .property_nullable("reading_order", DataType::Int64)
        .done()
        .edge_type(
            edges::NEXT_IN_READING_ORDER,
            &[labels::BLOCK],
            &[labels::BLOCK],
        )
        .done()
}
