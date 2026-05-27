//! AST-based code chunking using tree-sitter.
//!
//! Extracts functions, classes, structs, and import blocks as individual
//! chunks with `symbol_name` metadata.  Falls back to [`TextChunker`] if
//! parsing fails.

#![cfg(feature = "code-parse")]

use tree_sitter::Parser;

use super::text::TextChunker;
use super::{ChunkConfig, ChunkData, Chunker, count_tokens};

/// AST-aware chunker for source code.
pub struct CodeChunker {
    language: String,
}

impl CodeChunker {
    /// Create a chunker for the given language.
    pub fn new(language: &str) -> Self {
        Self {
            language: language.to_lowercase(),
        }
    }

    /// Whether this language has a tree-sitter grammar available.
    pub fn supports(language: &str) -> bool {
        matches!(
            language.to_lowercase().as_str(),
            "python" | "py" | "rust" | "rs" | "javascript" | "js" | "typescript" | "ts" | "tsx"
        )
    }
}

impl Chunker for CodeChunker {
    fn chunk(&self, content: &str, config: &ChunkConfig) -> Vec<ChunkData> {
        match chunk_code(content, &self.language, config) {
            Some(chunks) if !chunks.is_empty() => chunks,
            _ => {
                tracing::debug!(lang = %self.language, "tree-sitter parse failed, falling back to text chunker");
                TextChunker.chunk(content, config)
            }
        }
    }
}

/// Parse and chunk source code.  Returns `None` if the language is
/// unsupported or parsing fails.
fn chunk_code(content: &str, language: &str, config: &ChunkConfig) -> Option<Vec<ChunkData>> {
    let mut parser = parser_for_language(language)?;
    let tree = parser.parse(content, None)?;
    let root = tree.root_node();

    if root.has_error() {
        tracing::warn!(lang = %language, "tree-sitter parse produced errors");
        // Continue anyway — partial parses are often usable.
    }

    let mut chunks = Vec::new();
    let mut cursor = root.walk();

    for child in root.children(&mut cursor) {
        let kind = child.kind();
        let start_byte = child.start_byte();
        let end_byte = child.end_byte();
        let text = &content[start_byte..end_byte];
        let tokens = count_tokens(text);

        let (chunk_type, symbol_name) = classify_node(kind, language, child, content);

        if tokens <= config.max_chunk_tokens {
            chunks.push(ChunkData {
                text: text.to_string(),
                index: 0, // Re-indexed below.
                start: start_byte,
                end: end_byte,
                token_count: tokens,
                chunk_type,
                language: Some(language.to_string()),
                symbol_name,
                heading: None,
                metadata: None,
            });
        } else {
            // Large node — split at child boundaries.
            let sub_chunks = split_large_node(child, content, language, config);
            if sub_chunks.is_empty() {
                // Last resort: text chunker on this node.
                let fallback = TextChunker.chunk(text, config);
                for mut fc in fallback {
                    fc.language = Some(language.to_string());
                    fc.chunk_type = "code".into();
                    fc.start += start_byte;
                    fc.end += start_byte;
                    chunks.push(fc);
                }
            } else {
                chunks.extend(sub_chunks);
            }
        }
    }

    // Merge small adjacent chunks (imports, constants).
    merge_small_code_chunks(&mut chunks, config);

    // Re-index.
    for (i, chunk) in chunks.iter_mut().enumerate() {
        chunk.index = i;
    }

    Some(chunks)
}

/// Initialize a tree-sitter parser for the given language.
fn parser_for_language(lang: &str) -> Option<Parser> {
    let language_fn = match lang {
        "python" | "py" => tree_sitter_python::LANGUAGE,
        "rust" | "rs" => tree_sitter_rust::LANGUAGE,
        "javascript" | "js" => tree_sitter_javascript::LANGUAGE,
        "typescript" | "ts" => tree_sitter_typescript::LANGUAGE_TYPESCRIPT,
        "tsx" => tree_sitter_typescript::LANGUAGE_TSX,
        _ => return None,
    };
    let mut parser = Parser::new();
    parser.set_language(&language_fn.into()).ok()?;
    Some(parser)
}

/// Classify a top-level AST node into chunk type and extract symbol name.
fn classify_node(
    kind: &str,
    language: &str,
    node: tree_sitter::Node<'_>,
    source: &str,
) -> (String, Option<String>) {
    let symbol = extract_name(node, source);

    let chunk_type = match (language, kind) {
        // Python
        (_, "function_definition" | "decorated_definition") => "function",
        (_, "class_definition") => "class",
        ("python" | "py", "import_statement" | "import_from_statement") => "imports",
        // Rust
        (_, "function_item") => "function",
        (_, "impl_item") => "impl",
        (_, "struct_item") => "struct",
        (_, "enum_item") => "enum",
        (_, "trait_item") => "trait",
        (_, "mod_item") => "module",
        ("rust" | "rs", "use_declaration") => "imports",
        // JS/TS
        (_, "function_declaration") => "function",
        (_, "class_declaration") => "class",
        (_, "export_statement") => "export",
        ("javascript" | "js" | "typescript" | "ts" | "tsx", "import_statement") => "imports",
        _ => "code",
    };

    (chunk_type.to_string(), symbol)
}

/// Extract the name identifier from a declaration node.
fn extract_name(node: tree_sitter::Node<'_>, source: &str) -> Option<String> {
    // Look for a direct child named "name" or "identifier".
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "identifier" || child.kind() == "name" {
            return Some(source[child.start_byte()..child.end_byte()].to_string());
        }
        // For decorated definitions (Python), look one level deeper.
        if child.kind() == "function_definition" || child.kind() == "class_definition" {
            return extract_name(child, source);
        }
    }
    None
}

/// Split a large AST node at child block boundaries.
fn split_large_node(
    node: tree_sitter::Node<'_>,
    source: &str,
    language: &str,
    config: &ChunkConfig,
) -> Vec<ChunkData> {
    let mut chunks = Vec::new();
    let mut cursor = node.walk();

    for child in node.children(&mut cursor) {
        let start = child.start_byte();
        let end = child.end_byte();
        let text = &source[start..end];
        let tokens = count_tokens(text);

        let (chunk_type, symbol_name) = classify_node(child.kind(), language, child, source);

        if tokens <= config.max_chunk_tokens && tokens >= config.min_chunk_tokens {
            chunks.push(ChunkData {
                text: text.to_string(),
                index: 0,
                start,
                end,
                token_count: tokens,
                chunk_type,
                language: Some(language.to_string()),
                symbol_name,
                heading: None,
                metadata: None,
            });
        }
    }
    chunks
}

/// Merge adjacent small code chunks (imports, constants) below min threshold.
fn merge_small_code_chunks(chunks: &mut Vec<ChunkData>, config: &ChunkConfig) {
    let mut i = 0;
    while i < chunks.len().saturating_sub(1) {
        if chunks[i].token_count < config.min_chunk_tokens {
            let next = chunks.remove(i + 1);
            chunks[i].text = format!("{}\n\n{}", chunks[i].text, next.text);
            chunks[i].end = next.end;
            chunks[i].token_count = count_tokens(&chunks[i].text);
            if chunks[i].chunk_type == "imports" && next.chunk_type == "imports" {
                // Keep as imports.
            } else {
                chunks[i].chunk_type = "code".into();
            }
            chunks[i].symbol_name = None;
        } else {
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_python_functions() {
        let code = r#"
def hello():
    print("hello")

def world():
    print("world")
"#;
        let config = ChunkConfig {
            max_chunk_tokens: 500,
            min_chunk_tokens: 5,
            overlap_tokens: 0,
        };
        let chunks = chunk_code(code, "python", &config).unwrap();
        assert!(chunks.len() >= 2, "should have at least 2 function chunks");
        assert!(
            chunks
                .iter()
                .any(|c| c.symbol_name.as_deref() == Some("hello"))
        );
        assert!(
            chunks
                .iter()
                .any(|c| c.symbol_name.as_deref() == Some("world"))
        );
    }

    #[test]
    fn test_rust_struct_and_fn() {
        let code = r#"
struct Foo {
    x: i32,
}

fn bar() -> i32 {
    42
}
"#;
        let config = ChunkConfig {
            max_chunk_tokens: 500,
            min_chunk_tokens: 5,
            overlap_tokens: 0,
        };
        let chunks = chunk_code(code, "rust", &config).unwrap();
        assert!(chunks.len() >= 2);
        assert!(chunks.iter().any(|c| c.chunk_type == "struct"));
        assert!(chunks.iter().any(|c| c.chunk_type == "function"));
    }

    #[test]
    fn test_unsupported_language() {
        assert!(!CodeChunker::supports("cobol"));
        assert!(CodeChunker::supports("python"));
    }

    #[test]
    fn test_code_chunker_fallback() {
        let chunker = CodeChunker::new("cobol");
        let chunks = chunker.chunk("SOME COBOL CODE", &ChunkConfig::default());
        // Should fall back to text chunker.
        assert!(!chunks.is_empty());
    }
}
