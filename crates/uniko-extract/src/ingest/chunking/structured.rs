//! Structured data chunker (CSV/JSON) — delegates to [`TextChunker`]
//! until schema-aware row grouping is implemented.

// Rust guideline compliant

use super::{ChunkConfig, ChunkData, Chunker};

/// Placeholder structured-data chunker.
pub struct StructuredChunker;

impl Chunker for StructuredChunker {
    fn chunk(&self, content: &str, config: &ChunkConfig) -> Vec<ChunkData> {
        // TODO: CSV row grouping and JSON key splitting (Phase 6).
        super::text::TextChunker.chunk(content, config)
    }
}
