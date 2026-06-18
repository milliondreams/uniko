//! Layer 3: Page node type — a rendered/parsed page of a PDF artifact.
//!
//! A `:Page` is the per-page node of the document-IR graph produced by tiered
//! PDF extraction (`uni-xervo-pdf`). It carries the page-level provenance (which
//! tier produced it, the concatenated markdown view, and the escalation audit
//! trail) and fans out to its `:Block` children via `CONTAINS`. The embeddable
//! text lives on the per-block child `:Chunk`s, so `:Page` itself has no vector
//! index.

use uni_db::{DataType, IndexType, ScalarType, SchemaBuilder};

use super::constants::{edges, labels};
use crate::config::UnikoConfig;

/// Register the `:Page` label, its properties, and indexes.
pub(crate) fn register_labels<'a>(
    builder: SchemaBuilder<'a>,
    _config: &UnikoConfig,
) -> SchemaBuilder<'a> {
    builder
        .label(labels::PAGE)
        .property("page_id", DataType::String)
        .property_nullable("page_number", DataType::Int64)
        // Page-level tier that produced this page ("native" / "ocr" / "vlm").
        .property_nullable("produced_by", DataType::String)
        // Concatenated markdown view of the page's blocks, in reading order.
        // FullText-indexed for a page-level BM25 fallback.
        .property_nullable("plain_markdown", DataType::String)
        .property_nullable("block_count", DataType::Int64)
        // JSON audit trail of tier escalations ([{from, to, reason}, …]).
        // Low cardinality, never filtered on — rides in the JSON bag.
        .property_nullable("escalations", DataType::CypherValue)
        .index("page_id", IndexType::Scalar(ScalarType::Hash))
        .index("page_number", IndexType::Scalar(ScalarType::BTree))
        .index("plain_markdown", IndexType::FullText)
        .done()
}

/// Register the `HAS_PAGE` edge (`Artifact` → `Page`).
pub(crate) fn register_edges(builder: SchemaBuilder<'_>) -> SchemaBuilder<'_> {
    builder
        .edge_type(edges::HAS_PAGE, &[labels::ARTIFACT], &[labels::PAGE])
        .property_nullable("index", DataType::Int64)
        .done()
}
