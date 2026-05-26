//! Structured-data chunker for CSV and JSON.
//!
//! Schema-aware row grouping: each chunk's text begins with the header
//! schema (so retrieval over `Chunk.text` carries column names with
//! every row group) and contains a token-budget-sized batch of rows.
//!
//! Not Arrow/Polars-backed yet — a single-pass line splitter is enough
//! for the per-chunk grouping the spec requires, and avoids pulling
//! Polars into uniko-extract just for this. A future PR can swap the
//! implementation while keeping the same `chunk_type="table_row_group"`
//! taxonomy.
//
// Rust guideline compliant

use super::text::TextChunker;
use super::{ChunkConfig, ChunkData, Chunker, count_tokens};

/// Schema-aware row-group chunker for `text/csv` and `application/json`.
pub struct StructuredChunker;

impl Chunker for StructuredChunker {
    fn chunk(&self, content: &str, config: &ChunkConfig) -> Vec<ChunkData> {
        if let Some(rows) = parse_csv_lines(content) {
            return chunk_rows(&rows, config, "csv");
        }
        if let Some(rows) = parse_json_array(content) {
            return chunk_rows(&rows, config, "json");
        }
        // Unrecognized — fall back so we never silently drop content.
        TextChunker.chunk(content, config)
    }
}

/// Parse CSV into `(header, rows)`. Returns `None` when the input
/// doesn't look like CSV (no newlines or no commas).
fn parse_csv_lines(content: &str) -> Option<Rows> {
    let mut lines = content.lines();
    let header = lines.next()?;
    if !header.contains(',') {
        return None;
    }
    let header_cols: Vec<String> = header.split(',').map(|s| s.trim().to_string()).collect();
    let rows: Vec<Vec<String>> = lines
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split(',').map(|s| s.trim().to_string()).collect())
        .collect();
    if rows.is_empty() {
        return None;
    }
    Some(Rows {
        header: header_cols,
        rows,
    })
}

/// Parse a `[ {...}, {...}, ... ]` JSON array of objects. Returns
/// `None` if the input isn't a top-level array.
fn parse_json_array(content: &str) -> Option<Rows> {
    let value: serde_json::Value = serde_json::from_str(content).ok()?;
    let arr = value.as_array()?;
    if arr.is_empty() {
        return None;
    }
    // Header = union of keys in source order from the first object.
    let header_cols: Vec<String> = arr
        .first()
        .and_then(|v| v.as_object())
        .map(|o| o.keys().cloned().collect())?;
    let rows: Vec<Vec<String>> = arr
        .iter()
        .filter_map(|v| v.as_object())
        .map(|obj| {
            header_cols
                .iter()
                .map(|k| match obj.get(k) {
                    Some(serde_json::Value::String(s)) => s.clone(),
                    Some(v) => v.to_string(),
                    None => String::new(),
                })
                .collect()
        })
        .collect();
    Some(Rows {
        header: header_cols,
        rows,
    })
}

#[derive(Debug)]
struct Rows {
    header: Vec<String>,
    rows: Vec<Vec<String>>,
}

fn chunk_rows(rows: &Rows, config: &ChunkConfig, source: &str) -> Vec<ChunkData> {
    let heading_text = rows.header.join(", ");
    let heading_json = serde_json::to_string(&rows.header).unwrap_or_else(|_| "[]".into());
    let header_md = format_header_row(&rows.header);
    let header_tokens = count_tokens(&header_md);

    let mut out: Vec<ChunkData> = Vec::new();
    let mut buf_rows: Vec<String> = Vec::new();
    let mut buf_tokens = header_tokens;
    let mut chunk_index = 0;

    for r in &rows.rows {
        let row_md = format_data_row(r);
        let row_tokens = count_tokens(&row_md);
        if !buf_rows.is_empty() && buf_tokens + row_tokens > config.max_chunk_tokens {
            push_chunk(
                &mut out,
                &header_md,
                &buf_rows,
                &heading_json,
                &heading_text,
                &mut chunk_index,
            );
            buf_rows.clear();
            buf_tokens = header_tokens;
        }
        buf_rows.push(row_md);
        buf_tokens += row_tokens;
    }
    if !buf_rows.is_empty() {
        push_chunk(
            &mut out,
            &header_md,
            &buf_rows,
            &heading_json,
            &heading_text,
            &mut chunk_index,
        );
    }
    tracing::debug!(
        target: "uniko_extract::chunking",
        source,
        chunks = out.len(),
        rows = rows.rows.len(),
        "structured chunking complete"
    );
    out
}

fn push_chunk(
    out: &mut Vec<ChunkData>,
    header_md: &str,
    buf_rows: &[String],
    heading_json: &str,
    _heading_text: &str,
    chunk_index: &mut usize,
) {
    let mut text = String::with_capacity(header_md.len() + buf_rows.iter().map(|s| s.len()).sum::<usize>());
    text.push_str(header_md);
    for r in buf_rows {
        text.push_str(r);
    }
    let token_count = count_tokens(&text);
    out.push(ChunkData {
        text: text.clone(),
        index: *chunk_index,
        start: 0,
        end: text.len(),
        token_count,
        chunk_type: "table_row_group".into(),
        language: None,
        symbol_name: None,
        heading: Some(heading_json.to_string()),
    });
    *chunk_index += 1;
}

fn format_header_row(cols: &[String]) -> String {
    let mut s = String::new();
    s.push('|');
    for c in cols {
        s.push(' ');
        s.push_str(c);
        s.push_str(" |");
    }
    s.push('\n');
    s.push('|');
    for _ in cols {
        s.push_str(" --- |");
    }
    s.push('\n');
    s
}

fn format_data_row(cols: &[String]) -> String {
    let mut s = String::new();
    s.push('|');
    for c in cols {
        s.push(' ');
        s.push_str(c);
        s.push_str(" |");
    }
    s.push('\n');
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csv_round_trip() {
        let csv = "name,age\nalice,30\nbob,25";
        let chunks = StructuredChunker.chunk(csv, &ChunkConfig::default());
        assert!(!chunks.is_empty());
        assert!(chunks.iter().all(|c| c.chunk_type == "table_row_group"));
        assert!(chunks[0].heading.as_deref() == Some("[\"name\",\"age\"]"));
        assert!(chunks[0].text.contains("alice"));
        assert!(chunks[0].text.contains("bob"));
    }

    #[test]
    fn test_json_array_round_trip() {
        let json = "[{\"name\":\"alice\",\"age\":30},{\"name\":\"bob\",\"age\":25}]";
        let chunks = StructuredChunker.chunk(json, &ChunkConfig::default());
        assert!(!chunks.is_empty());
        assert_eq!(chunks[0].chunk_type, "table_row_group");
        assert!(chunks[0].text.contains("alice"));
    }

    #[test]
    fn test_non_csv_non_json_falls_back() {
        let chunks = StructuredChunker.chunk("just plain text", &ChunkConfig::default());
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].chunk_type, "text");
    }

    #[test]
    fn test_csv_chunks_respect_token_budget() {
        // Generate 1000 rows of a 2-col CSV; default max_chunk_tokens
        // is 512, so we should get multiple chunks.
        let mut csv = String::from("col_a,col_b\n");
        for i in 0..1000 {
            csv.push_str(&format!("value_{i},sample_data_{i}\n"));
        }
        let chunks = StructuredChunker.chunk(&csv, &ChunkConfig::default());
        assert!(chunks.len() > 1, "expected multi-chunk output, got {}", chunks.len());
        // Every chunk should start with the header row.
        for c in &chunks {
            assert!(c.text.starts_with("| col_a | col_b |"), "chunk missing header: {}", c.text);
        }
    }
}
