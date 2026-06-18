//! Content chunking strategies for text, code, and structured data.
//!
//! [`select_chunker`] picks the right strategy based on content type.
//! [`count_tokens`] provides accurate token counting via tiktoken.

pub mod text;

#[cfg(feature = "code-parse")]
pub mod code;

pub mod html;
pub mod structured;

use std::sync::OnceLock;

/// Output of chunking a single parent document.
#[derive(Debug, Clone)]
pub struct ChunkData {
    /// Chunk text content.
    pub text: String,
    /// Zero-based position within the parent.
    pub index: usize,
    /// Byte offset of the chunk start in the source content.
    pub start: usize,
    /// Byte offset of the chunk end in the source content.
    pub end: usize,
    /// Token count (via tiktoken or approximation).
    pub token_count: usize,
    /// Chunk classification: `"text"`, `"code"`, `"imports"`, etc.
    pub chunk_type: String,
    /// Programming language (for code chunks).
    pub language: Option<String>,
    /// Function/class/struct name (for code chunks).
    pub symbol_name: Option<String>,
    /// Nearest preceding heading (for markdown/HTML).
    pub heading: Option<String>,
    /// Modality-specific scalars (e.g. `{"page_number": 3,
    /// "page_count": 12}` for PDF page chunks). Persisted into the
    /// `:Chunk.metadata` JSON column. `None` for chunkers that have
    /// no extra structured metadata to carry.
    pub metadata: Option<serde_json::Value>,
}

/// Chunking parameters derived from [`uniko_store::config::UnikoConfig`].
#[derive(Debug, Clone)]
pub struct ChunkConfig {
    /// Maximum tokens per chunk (default: 512).
    pub max_chunk_tokens: usize,
    /// Minimum tokens per chunk — fragments below this merge (default: 64).
    pub min_chunk_tokens: usize,
    /// Approximate overlap tokens between adjacent chunks.
    pub overlap_tokens: usize,
}

impl ChunkConfig {
    /// Build from [`uniko_store::config::UnikoConfig`] defaults.
    pub fn from_uniko_config(cfg: &uniko_store::config::UnikoConfig) -> Self {
        let overlap = if cfg.chunk_overlap_tokens > 0 {
            cfg.chunk_overlap_tokens
        } else {
            // Auto: ~10% of max, capped at 50 tokens.
            (cfg.max_chunk_tokens / 10).min(50)
        };
        Self {
            max_chunk_tokens: cfg.max_chunk_tokens,
            min_chunk_tokens: cfg.min_chunk_tokens,
            overlap_tokens: overlap,
        }
    }
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            max_chunk_tokens: 512,
            min_chunk_tokens: 64,
            overlap_tokens: 50,
        }
    }
}

/// A content-type-specific splitting strategy.
pub trait Chunker: Send + Sync {
    /// Split `content` into chunks according to `config`.
    fn chunk(&self, content: &str, config: &ChunkConfig) -> Vec<ChunkData>;
}

/// Count tokens in `text` using the cl100k_base tokenizer (GPT-4).
///
/// Falls back to a word-count approximation (tokens ≈ words × 1.3) if
/// the tokenizer cannot be loaded.
pub fn count_tokens(text: &str) -> usize {
    static BPE: OnceLock<Option<tiktoken_rs::CoreBPE>> = OnceLock::new();
    let bpe = BPE.get_or_init(|| tiktoken_rs::cl100k_base().ok());
    match bpe {
        Some(enc) => enc.encode_with_special_tokens(text).len(),
        None => {
            // Fallback: ~1.3 tokens per word for English text.
            let words = text.split_whitespace().count();
            ((words as f64) * 1.3).ceil() as usize
        }
    }
}

/// Select the appropriate chunker for a legacy content-type token.
///
/// Routes the bare `content_type` string through the
/// [`Modality`](uniko_pipes::content::Modality) taxonomy, then to a
/// chunker. Behavior is identical to the former hand-rolled `match`: only
/// `"code"`, `"html"`, `"csv"`, `"json"`, `"structured"` are special;
/// everything else (incl. `"text"`, `"tool_result"`, full MIME strings)
/// uses the text chunker.
pub fn select_chunker(content_type: &str, language: Option<&str>) -> Box<dyn Chunker> {
    let modality = uniko_pipes::content::modality_for_mime(
        &uniko_pipes::content::legacy_content_type_to_mime(content_type),
    );
    chunker_for(modality, language)
}

/// Select a chunker directly from a resolved [`Modality`].
///
/// The single mapping from modality to chunking strategy, shared by
/// [`select_chunker`] and the unified ingest dispatch.
pub fn chunker_for(
    modality: uniko_pipes::content::Modality,
    #[cfg_attr(not(feature = "code-parse"), allow(unused_variables))] language: Option<&str>,
) -> Box<dyn Chunker> {
    use uniko_pipes::content::Modality;
    match modality {
        Modality::Code => {
            #[cfg(feature = "code-parse")]
            if let Some(lang) = language.filter(|l| code::CodeChunker::supports(l)) {
                return Box::new(code::CodeChunker::new(lang));
            }
            // Unsupported language or feature disabled — fall back to text.
            Box::new(text::TextChunker)
        }
        Modality::Markup => Box::new(html::HtmlChunker),
        Modality::Structured => Box::new(structured::StructuredChunker),
        // Text, Document, Pdf, Image, Audio, Video, and anything else
        // fall back to the text chunker.
        _ => Box::new(text::TextChunker),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_tokens_nonempty() {
        let n = count_tokens("Hello, world!");
        assert!(n > 0 && n < 20);
    }

    #[test]
    fn test_count_tokens_empty() {
        assert_eq!(count_tokens(""), 0);
    }

    #[test]
    fn test_select_chunker_text() {
        let c = select_chunker("text", None);
        let chunks = c.chunk("hello", &ChunkConfig::default());
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn test_select_chunker_unknown_falls_back() {
        let c = select_chunker("application/octet-stream", None);
        let chunks = c.chunk("data", &ChunkConfig::default());
        assert_eq!(chunks.len(), 1);
    }
}
