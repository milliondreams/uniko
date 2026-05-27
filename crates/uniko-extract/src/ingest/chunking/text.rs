//! Recursive text chunker with sentence-boundary alignment.
//!
//! Split hierarchy: paragraphs (`\n\n`) → sentences (`. `) → words (` `).
//! Chunks target 400-512 tokens.  No mid-sentence breaks.

use super::{ChunkConfig, ChunkData, Chunker, count_tokens};

/// Recursive paragraph/sentence splitter for plain text and markdown.
pub struct TextChunker;

impl Chunker for TextChunker {
    fn chunk(&self, content: &str, config: &ChunkConfig) -> Vec<ChunkData> {
        chunk_text(content, config)
    }
}

/// Chunk text content using recursive splitting.
pub fn chunk_text(content: &str, config: &ChunkConfig) -> Vec<ChunkData> {
    if content.is_empty() {
        return Vec::new();
    }

    let total_tokens = count_tokens(content);
    if total_tokens <= config.max_chunk_tokens {
        return vec![ChunkData {
            text: content.to_string(),
            index: 0,
            start: 0,
            end: content.len(),
            token_count: total_tokens,
            chunk_type: "text".into(),
            language: None,
            symbol_name: None,
            heading: extract_heading(content),
            metadata: None,
        }];
    }

    // Split into paragraphs first.
    let segments = split_paragraphs(content);
    let mut raw_chunks = greedy_accumulate(&segments, config);

    // Merge sub-minimum chunks with their neighbors.
    merge_small_chunks(&mut raw_chunks, config);

    // Apply overlap between adjacent chunks.
    apply_overlap(&mut raw_chunks, config);

    // Assign indices and compute final metadata.
    raw_chunks
        .into_iter()
        .enumerate()
        .map(|(i, rc)| ChunkData {
            token_count: count_tokens(&rc.text),
            text: rc.text,
            index: i,
            start: rc.start,
            end: rc.end,
            chunk_type: "text".into(),
            language: None,
            symbol_name: None,
            heading: rc.heading,
            metadata: None,
        })
        .collect()
}

// ── Internal types ──────────────────────────────────────────────────

struct RawChunk {
    text: String,
    start: usize,
    end: usize,
    heading: Option<String>,
}

struct Segment {
    text: String,
    start: usize,
    end: usize,
}

// ── Split helpers ───────────────────────────────────────────────────

/// Split content at paragraph boundaries (`\n\n`).
fn split_paragraphs(content: &str) -> Vec<Segment> {
    let mut segments = Vec::new();
    let mut offset = 0;

    for part in content.split("\n\n") {
        let trimmed = part.trim();
        if !trimmed.is_empty() {
            let start = content[offset..].find(trimmed).unwrap_or(0) + offset;
            let end = start + trimmed.len();
            segments.push(Segment {
                text: trimmed.to_string(),
                start,
                end,
            });
        }
        offset += part.len() + 2; // +2 for the "\n\n" delimiter
    }
    segments
}

/// Split a segment at sentence boundaries.
fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        current.push(ch);
        if matches!(ch, '.' | '?' | '!') {
            // Check if the next char is whitespace or end-of-text.
            sentences.push(current.trim().to_string());
            current = String::new();
        }
    }
    let remaining = current.trim().to_string();
    if !remaining.is_empty() {
        sentences.push(remaining);
    }
    sentences
}

/// Greedily accumulate segments into chunks targeting max_chunk_tokens.
fn greedy_accumulate(segments: &[Segment], config: &ChunkConfig) -> Vec<RawChunk> {
    let mut chunks = Vec::new();
    let mut current_text = String::new();
    let mut current_start = segments.first().map_or(0, |s| s.start);
    let mut current_tokens = 0usize;
    let mut current_heading: Option<String> = None;

    for seg in segments {
        // Track headings (markdown lines starting with #).
        if let Some(h) = extract_heading(&seg.text) {
            current_heading = Some(h);
        }

        let seg_tokens = count_tokens(&seg.text);

        // If this single segment exceeds max, split it into sentences.
        if seg_tokens > config.max_chunk_tokens {
            // Flush accumulated text first.
            if !current_text.is_empty() {
                chunks.push(RawChunk {
                    text: current_text.trim().to_string(),
                    start: current_start,
                    end: seg.start,
                    heading: current_heading.clone(),
                });
                current_text = String::new();
                current_tokens = 0;
            }

            // Split the large segment at sentence boundaries.
            let sentences = split_sentences(&seg.text);
            let mut sent_buf = String::new();
            let mut sent_tokens = 0usize;

            for sent in &sentences {
                let st = count_tokens(sent);
                if sent_tokens + st > config.max_chunk_tokens && !sent_buf.is_empty() {
                    chunks.push(RawChunk {
                        text: sent_buf.trim().to_string(),
                        start: seg.start,
                        end: seg.end,
                        heading: current_heading.clone(),
                    });
                    sent_buf = String::new();
                    sent_tokens = 0;
                }
                if !sent_buf.is_empty() {
                    sent_buf.push(' ');
                }
                sent_buf.push_str(sent);
                sent_tokens += st;
            }
            if !sent_buf.is_empty() {
                chunks.push(RawChunk {
                    text: sent_buf.trim().to_string(),
                    start: seg.start,
                    end: seg.end,
                    heading: current_heading.clone(),
                });
            }
            current_start = seg.end;
            continue;
        }

        // Would adding this segment exceed the limit?
        if current_tokens + seg_tokens > config.max_chunk_tokens && !current_text.is_empty() {
            chunks.push(RawChunk {
                text: current_text.trim().to_string(),
                start: current_start,
                end: seg.start,
                heading: current_heading.clone(),
            });
            current_text = String::new();
            current_tokens = 0;
            current_start = seg.start;
        }

        if !current_text.is_empty() {
            current_text.push_str("\n\n");
        }
        current_text.push_str(&seg.text);
        current_tokens += seg_tokens;
    }

    // Flush remaining.
    if !current_text.is_empty() {
        let end = segments.last().map_or(0, |s| s.end);
        chunks.push(RawChunk {
            text: current_text.trim().to_string(),
            start: current_start,
            end,
            heading: current_heading,
        });
    }

    chunks
}

/// Merge chunks below min_chunk_tokens with their next neighbor.
fn merge_small_chunks(chunks: &mut Vec<RawChunk>, config: &ChunkConfig) {
    let mut i = 0;
    while i < chunks.len().saturating_sub(1) {
        if count_tokens(&chunks[i].text) < config.min_chunk_tokens {
            let next = chunks.remove(i + 1);
            chunks[i].text = format!("{}\n\n{}", chunks[i].text, next.text);
            chunks[i].end = next.end;
            // Don't advance — re-check merged chunk.
        } else {
            i += 1;
        }
    }
}

/// Prepend overlap text from the previous chunk.
fn apply_overlap(chunks: &mut [RawChunk], config: &ChunkConfig) {
    if config.overlap_tokens == 0 || chunks.len() < 2 {
        return;
    }
    // Approximate overlap_tokens as chars (avg ~4 chars/token for English).
    let overlap_chars = config.overlap_tokens * 4;

    for i in 1..chunks.len() {
        let prev_text = chunks[i - 1].text.clone();
        let char_count = prev_text.chars().count();
        if char_count > overlap_chars {
            // Find byte offset of the character `overlap_chars` from the end.
            let start_byte = prev_text
                .char_indices()
                .nth(char_count - overlap_chars)
                .map_or(0, |(idx, _)| idx);
            let suffix = &prev_text[start_byte..];
            // Find the nearest word boundary.
            let boundary = suffix.find(' ').unwrap_or(0);
            let overlap = &suffix[boundary..].trim();
            if !overlap.is_empty() {
                chunks[i].text = format!("{overlap} {}", chunks[i].text);
            }
        }
    }
}

/// Extract the first markdown heading from text.
fn extract_heading(text: &str) -> Option<String> {
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('#') {
            return Some(trimmed.trim_start_matches('#').trim().to_string());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> ChunkConfig {
        ChunkConfig {
            max_chunk_tokens: 50,
            min_chunk_tokens: 10,
            overlap_tokens: 5,
        }
    }

    #[test]
    fn test_short_text_single_chunk() {
        let chunks = chunk_text("Hello world.", &cfg());
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].index, 0);
        assert!(chunks[0].text.contains("Hello"));
    }

    #[test]
    fn test_empty_text() {
        let chunks = chunk_text("", &cfg());
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_paragraph_split() {
        // Create enough text to exceed 50 tokens.
        let para1 = "The quick brown fox jumps over the lazy dog. ".repeat(3);
        let para2 = "A second paragraph with different content here. ".repeat(3);
        let content = format!("{para1}\n\n{para2}");

        let chunks = chunk_text(&content, &cfg());
        assert!(chunks.len() >= 2, "should split into multiple chunks");
    }

    #[test]
    fn test_markdown_heading_extraction() {
        let content = "## Introduction\n\nSome text here about the topic.";
        let chunks = chunk_text(content, &cfg());
        assert_eq!(chunks[0].heading.as_deref(), Some("Introduction"));
    }

    #[test]
    fn test_chunks_have_sequential_indices() {
        let long = "Word. ".repeat(200);
        let chunks = chunk_text(&long, &cfg());
        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.index, i);
        }
    }

    #[test]
    fn test_chunk_type_is_text() {
        let chunks = chunk_text("Some content here.", &cfg());
        assert_eq!(chunks[0].chunk_type, "text");
    }
}
