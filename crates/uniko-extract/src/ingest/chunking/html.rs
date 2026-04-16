//! HTML chunker — delegates to [`TextChunker`] until DOM-aware
//! splitting is implemented.

// Rust guideline compliant

use super::{ChunkConfig, ChunkData, Chunker};

/// Placeholder HTML chunker.
pub struct HtmlChunker;

impl Chunker for HtmlChunker {
    fn chunk(&self, content: &str, config: &ChunkConfig) -> Vec<ChunkData> {
        // TODO: DOM-aware section extraction (Phase 6).
        super::text::TextChunker.chunk(content, config)
    }
}
