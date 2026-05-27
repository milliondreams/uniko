//! Page-aware chunker: turn [`ExtractedPage`]s into [`ChunkData`].
//!
//! Per page:
//! 1. Skip whitespace-only pages.
//! 2. If the page fits in `max_chunk_tokens`, emit one chunk.
//! 3. Otherwise delegate to [`TextChunker`] for that page's text, then
//!    tag every resulting sub-chunk with the same page metadata.
//!
//! Every emitted chunk carries `chunk_type = "page"` and
//! `metadata = {"page_number": N, "page_count": M}` in the JSON bag.
//! Chunk `index` is global (0..N across all pages), matching the rest
//! of the chunker pipeline.

use serde_json::json;

use super::super::chunking::{ChunkConfig, ChunkData, Chunker, count_tokens, text::TextChunker};
use super::extractor::ExtractedPage;

/// Chunk a slice of extracted PDF pages.
///
/// See module docs for behavior.
pub fn chunk_pages(pages: &[ExtractedPage], config: &ChunkConfig) -> Vec<ChunkData> {
    let page_count = pages.len();
    let mut out: Vec<ChunkData> = Vec::new();
    let mut global_idx: usize = 0;

    for page in pages {
        let trimmed = page.text.trim();
        if trimmed.is_empty() {
            continue;
        }

        let meta = json!({
            "page_number": page.page_number,
            "page_count": page_count,
        });

        let page_tokens = count_tokens(&page.text);
        if page_tokens <= config.max_chunk_tokens {
            out.push(ChunkData {
                text: page.text.clone(),
                index: global_idx,
                start: 0,
                end: page.text.len(),
                token_count: page_tokens,
                chunk_type: "page".into(),
                language: None,
                symbol_name: None,
                heading: None,
                metadata: Some(meta),
            });
            global_idx += 1;
            continue;
        }

        // Long page — split via TextChunker, then tag every sub-chunk.
        for sub in TextChunker.chunk(&page.text, config) {
            out.push(ChunkData {
                text: sub.text,
                index: global_idx,
                start: sub.start,
                end: sub.end,
                token_count: sub.token_count,
                chunk_type: "page".into(),
                language: None,
                symbol_name: None,
                heading: None,
                metadata: Some(meta.clone()),
            });
            global_idx += 1;
        }
    }

    out
}

/// [`Chunker`] adapter — used only when the chunker is selected by
/// `chunk_type` string. PDF ingest normally calls [`chunk_pages`]
/// directly with `Vec<ExtractedPage>`.
pub struct PdfPageChunker;

impl Chunker for PdfPageChunker {
    fn chunk(&self, content: &str, config: &ChunkConfig) -> Vec<ChunkData> {
        // Single-page fallback: caller has only a `&str`, so we can't
        // know page boundaries — emit as one virtual page.
        let pages = vec![ExtractedPage {
            page_number: 1,
            text: content.to_string(),
        }];
        chunk_pages(&pages, config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> ChunkConfig {
        ChunkConfig {
            max_chunk_tokens: 50,
            min_chunk_tokens: 10,
            overlap_tokens: 0,
        }
    }

    #[test]
    fn one_chunk_per_short_page() {
        let pages = vec![
            ExtractedPage {
                page_number: 1,
                text: "Short page one.".into(),
            },
            ExtractedPage {
                page_number: 2,
                text: "Short page two.".into(),
            },
        ];
        let chunks = chunk_pages(&pages, &cfg());
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].chunk_type, "page");
        assert_eq!(chunks[0].index, 0);
        assert_eq!(chunks[1].index, 1);
        let m0 = chunks[0].metadata.as_ref().unwrap();
        assert_eq!(m0["page_number"], 1);
        assert_eq!(m0["page_count"], 2);
        let m1 = chunks[1].metadata.as_ref().unwrap();
        assert_eq!(m1["page_number"], 2);
    }

    #[test]
    fn long_page_splits_within_page() {
        let long = "The quick brown fox jumps over the lazy dog. ".repeat(40);
        let pages = vec![ExtractedPage {
            page_number: 7,
            text: long,
        }];
        let chunks = chunk_pages(&pages, &cfg());
        assert!(
            chunks.len() >= 2,
            "expected long page to split into multiple chunks"
        );
        for c in &chunks {
            assert_eq!(c.chunk_type, "page");
            let m = c.metadata.as_ref().unwrap();
            assert_eq!(m["page_number"], 7);
            assert_eq!(m["page_count"], 1);
        }
        // Global indices are sequential.
        for (i, c) in chunks.iter().enumerate() {
            assert_eq!(c.index, i);
        }
    }

    #[test]
    fn skips_empty_pages() {
        let pages = vec![
            ExtractedPage {
                page_number: 1,
                text: "Real content.".into(),
            },
            ExtractedPage {
                page_number: 2,
                text: "   \n\n  ".into(),
            },
            ExtractedPage {
                page_number: 3,
                text: "More content.".into(),
            },
        ];
        let chunks = chunk_pages(&pages, &cfg());
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].metadata.as_ref().unwrap()["page_number"], 1);
        assert_eq!(chunks[1].metadata.as_ref().unwrap()["page_number"], 3);
    }

    #[test]
    fn page_count_reflects_input_pages_not_emitted() {
        // Even when a page is skipped, page_count is the input length —
        // it tells you the source document's total pages.
        let pages = vec![
            ExtractedPage {
                page_number: 1,
                text: "A".into(),
            },
            ExtractedPage {
                page_number: 2,
                text: " ".into(),
            },
        ];
        let chunks = chunk_pages(&pages, &cfg());
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].metadata.as_ref().unwrap()["page_count"], 2);
    }
}
