//! HTML chunker — strips tags, decodes a small set of entities, and
//! preserves the most recent heading context for each chunk.
//!
//! Not a full DOM parser (no `dom_smoothie` dep yet) — but the
//! behaviour is materially DOM-aware where it matters for retrieval:
//! `<script>` / `<style>` blocks are dropped wholesale, headings get
//! their own `chunk_type="heading"` chunk, and body text inherits the
//! nearest enclosing heading as `Chunk.heading`.
//!
//! Future work: swap the bespoke stripper for `dom_smoothie` when the
//! dep lands. The trait surface and `chunk_type` taxonomy stay the
//! same.
//
// Rust guideline compliant

use super::text::TextChunker;
use super::{ChunkConfig, ChunkData, Chunker};

/// DOM-aware-ish chunker for `text/html`.
pub struct HtmlChunker;

impl Chunker for HtmlChunker {
    fn chunk(&self, content: &str, config: &ChunkConfig) -> Vec<ChunkData> {
        let sections = extract_sections(content);
        if sections.is_empty() {
            // Degenerate input — fall back to TextChunker on the raw
            // bytes so we never silently drop content.
            return TextChunker.chunk(content, config);
        }

        let mut out: Vec<ChunkData> = Vec::new();
        let mut next_index: usize = 0;
        for section in sections {
            let pieces = TextChunker.chunk(&section.text, config);
            for mut piece in pieces {
                piece.index = next_index;
                next_index += 1;
                piece.chunk_type = section.chunk_type.clone();
                piece.heading = section.heading.clone();
                out.push(piece);
            }
        }
        if out.is_empty() {
            // Sections were all empty after stripping — fall back so
            // downstream tests don't see an empty chunk vector when
            // the input had any text at all.
            return TextChunker.chunk(content, config);
        }
        out
    }
}

#[derive(Debug)]
struct Section {
    text: String,
    chunk_type: String,
    heading: Option<String>,
}

/// Walk the HTML in source order, producing a stream of `Section`s:
/// one `chunk_type="heading"` per `<h1>..<h6>`, then one
/// `chunk_type="text"` for the body that follows it.
fn extract_sections(html: &str) -> Vec<Section> {
    let cleaned = strip_uninteresting(html);
    let bytes = cleaned.as_bytes();
    let mut sections: Vec<Section> = Vec::new();
    let mut buf = String::new();
    let mut current_heading: Option<String> = None;

    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'<' {
            // Tag open. Find the matching `>`.
            let Some(rel) = cleaned[i..].find('>') else {
                // Unmatched `<` — treat as literal.
                buf.push('<');
                i += 1;
                continue;
            };
            let tag_raw = &cleaned[i + 1..i + rel];
            let tag_lower = tag_raw.trim_start_matches('/').trim_end_matches('/');
            let tag_name = tag_lower.split_whitespace().next().unwrap_or("").to_ascii_lowercase();

            // Headings: flush body, then capture inner text, emit
            // heading section.
            if matches!(tag_name.as_str(), "h1" | "h2" | "h3" | "h4" | "h5" | "h6")
                && !tag_raw.starts_with('/')
            {
                flush_text(&mut buf, &mut sections, &current_heading);
                // Scan for matching close tag.
                let close = format!("</{tag_name}");
                let after = i + rel + 1;
                let end = cleaned[after..]
                    .find(&close)
                    .map(|p| p + after)
                    .unwrap_or(cleaned.len());
                let inner = strip_inline_tags(&cleaned[after..end]);
                let heading_text = decode_entities(&inner).trim().to_string();
                if !heading_text.is_empty() {
                    sections.push(Section {
                        text: heading_text.clone(),
                        chunk_type: "heading".into(),
                        heading: Some(heading_text.clone()),
                    });
                    current_heading = Some(heading_text);
                }
                // Advance past `</hN>`.
                let close_full = format!("</{tag_name}>");
                let advance = cleaned[end..]
                    .find('>')
                    .map(|p| end + p + 1)
                    .unwrap_or(cleaned.len());
                i = advance.max(end + close_full.len());
                continue;
            }

            // Block-level boundary: insert a paragraph break so the
            // TextChunker can split on `\n\n`.
            if matches!(
                tag_name.as_str(),
                "p" | "div"
                    | "section"
                    | "article"
                    | "li"
                    | "br"
                    | "tr"
                    | "td"
                    | "th"
                    | "blockquote"
            ) {
                if !buf.ends_with("\n\n") {
                    buf.push_str("\n\n");
                }
            }

            // Skip the tag.
            i += rel + 1;
            continue;
        }
        buf.push(bytes[i] as char);
        i += 1;
    }
    flush_text(&mut buf, &mut sections, &current_heading);
    sections
}

fn flush_text(buf: &mut String, out: &mut Vec<Section>, heading: &Option<String>) {
    let s = decode_entities(buf.as_str());
    let collapsed = collapse_whitespace(&s);
    if !collapsed.is_empty() {
        out.push(Section {
            text: collapsed,
            chunk_type: "text".into(),
            heading: heading.clone(),
        });
    }
    buf.clear();
}

/// Drop the contents of `<script>` / `<style>` / `<noscript>` blocks
/// before any further tag walking.
fn strip_uninteresting(html: &str) -> String {
    let mut out = String::with_capacity(html.len());
    let mut i = 0;
    let lower = html.to_ascii_lowercase();
    let bytes = html.as_bytes();
    while i < bytes.len() {
        let rest = &lower[i..];
        if let Some(skip_tag) = ["<script", "<style", "<noscript"]
            .iter()
            .find(|t| rest.starts_with(*t))
        {
            let close = format!("</{}", &skip_tag[1..]);
            let end_rel = lower[i..].find(&close).unwrap_or(lower.len() - i);
            let next = lower[i + end_rel..]
                .find('>')
                .map(|p| i + end_rel + p + 1)
                .unwrap_or(lower.len());
            i = next;
            continue;
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    out
}

/// Remove HTML tags from `s` while preserving textual content.
fn strip_inline_tags(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_tag = false;
    for ch in s.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(ch),
            _ => {}
        }
    }
    out
}

fn decode_entities(s: &str) -> String {
    s.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&nbsp;", " ")
}

fn collapse_whitespace(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut blank_line = false;
    for line in s.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            if !blank_line && !out.is_empty() {
                out.push_str("\n\n");
                blank_line = true;
            }
        } else {
            out.push_str(trimmed);
            out.push('\n');
            blank_line = false;
        }
    }
    out.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strips_script_and_style() {
        let html = "<html><head><style>.x{color:red}</style><script>alert(1)</script></head>\
                    <body><h1>Title</h1><p>Hello world</p></body></html>";
        let chunks = HtmlChunker.chunk(html, &ChunkConfig::default());
        let combined: String = chunks.iter().map(|c| c.text.clone()).collect::<Vec<_>>().join("\n");
        assert!(!combined.contains("alert"), "script body leaked: {combined}");
        assert!(!combined.contains("color:red"), "style body leaked: {combined}");
        assert!(combined.contains("Hello world"));
    }

    #[test]
    fn test_heading_propagates_to_body() {
        let html = "<h2>Greeting</h2><p>Hi there friend.</p>";
        let chunks = HtmlChunker.chunk(html, &ChunkConfig::default());
        let body = chunks
            .iter()
            .find(|c| c.chunk_type == "text")
            .expect("body text chunk");
        assert_eq!(body.heading.as_deref(), Some("Greeting"));
    }

    #[test]
    fn test_heading_emits_heading_chunk_type() {
        let html = "<h1>Title</h1><p>Body.</p>";
        let chunks = HtmlChunker.chunk(html, &ChunkConfig::default());
        assert!(chunks.iter().any(|c| c.chunk_type == "heading"));
    }

    #[test]
    fn test_degenerate_html_falls_back() {
        // No tags at all — still produces a chunk via TextChunker.
        let chunks = HtmlChunker.chunk("just some plain text", &ChunkConfig::default());
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].text.contains("plain text"));
    }
}
