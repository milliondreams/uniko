//! PDF chunker — delegates to [`TextChunker`] until page-aware
//! extraction is implemented.

// Rust guideline compliant

use super::{ChunkConfig, ChunkData, Chunker};

/// Placeholder PDF chunker.
pub struct PdfChunker;

impl Chunker for PdfChunker {
    fn chunk(&self, content: &str, config: &ChunkConfig) -> Vec<ChunkData> {
        // TODO: Page extraction + table detection (Phase 6).
        super::text::TextChunker.chunk(content, config)
    }
}
