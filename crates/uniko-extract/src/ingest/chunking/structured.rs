//! Structured-data chunker for CSV (any delimiter) and JSON (top-level
//! array of objects, or NDJSON / JSONL).
//!
//! Each emitted chunk's `text` is a Markdown table — header row, then a
//! batch of data rows fitted to the configured token budget — and the
//! header schema is stashed in `Chunk.heading` as a JSON array so
//! retrieval over `Chunk.text` carries column names with every row
//! group and downstream consumers can recover the schema cheaply.
//!
//! Parsing:
//! - CSV via the `csv` crate (BurntSushi's pure-Rust RFC 4180 reader).
//!   Delimiter is sniffed from the first non-empty line (`,` / `\t` /
//!   `;` / `|`). Handles quoted fields with embedded delimiters /
//!   newlines, escaped quotes (`""`), Windows `\r\n` line endings.
//! - JSON whole-doc array first; if that fails or isn't an array of
//!   objects, fall back to line-by-line NDJSON parse. Header is the
//!   union of keys across all objects (source order from first
//!   appearance), so heterogeneous schemas don't silently drop columns.
//!
//! Markdown cells with `|` or `\n` are escaped so the emitted table is
//! always well-formed.
//
// Rust guideline compliant

use std::collections::BTreeMap;

use serde_json::Value;

use super::text::TextChunker;
use super::{ChunkConfig, ChunkData, Chunker, count_tokens};

/// Schema-aware row-group chunker for `text/csv` and `application/json`.
pub struct StructuredChunker;

impl Chunker for StructuredChunker {
    fn chunk(&self, content: &str, config: &ChunkConfig) -> Vec<ChunkData> {
        // Try JSON first when the content looks like it: leading `{` or
        // `[` after trim. JSON with embedded commas would otherwise
        // trick the CSV delimiter sniffer.
        let trimmed_lead = content.trim_start();
        let looks_like_json =
            trimmed_lead.starts_with('{') || trimmed_lead.starts_with('[');
        if looks_like_json
            && let Some(rows) = parse_json(content)
        {
            return chunk_rows(&rows, config, "json");
        }
        if let Some(rows) = parse_csv(content) {
            return chunk_rows(&rows, config, "csv");
        }
        // Last-resort JSON attempt for content that doesn't start with
        // `{`/`[` (e.g. leading whitespace + valid JSON), then fall
        // back to TextChunker so we never silently drop content.
        if !looks_like_json
            && let Some(rows) = parse_json(content)
        {
            return chunk_rows(&rows, config, "json");
        }
        TextChunker.chunk(content, config)
    }
}

/// Parse CSV / TSV / `;`- or `|`-delimited input via the `csv` crate.
/// Returns `None` if no candidate delimiter appears in the first
/// non-empty line.
fn parse_csv(content: &str) -> Option<Rows> {
    let delimiter = sniff_delimiter(content)?;
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(delimiter)
        .has_headers(true)
        .flexible(true)
        .from_reader(content.as_bytes());

    let header: Vec<String> = reader
        .headers()
        .ok()?
        .iter()
        .map(|s| s.to_string())
        .collect();
    if header.is_empty() {
        return None;
    }

    let mut rows: Vec<Vec<String>> = Vec::new();
    for record in reader.records().flatten() {
        // Pad / truncate to header width so chunk_rows always sees
        // rectangular data (helpful for messy real-world CSVs).
        let mut row: Vec<String> = record.iter().map(|s| s.to_string()).collect();
        if row.len() < header.len() {
            row.resize(header.len(), String::new());
        } else if row.len() > header.len() {
            row.truncate(header.len());
        }
        rows.push(row);
    }
    if rows.is_empty() {
        return None;
    }
    Some(Rows {
        header,
        rows,
    })
}

/// Pick the highest-count delimiter from `,` / `\t` / `;` / `|` based
/// on the first non-blank line. Returns `None` when no candidate
/// appears at least once.
fn sniff_delimiter(content: &str) -> Option<u8> {
    let first = content.lines().find(|l| !l.trim().is_empty())?;
    let counts: [(u8, usize); 4] = [
        (b',', first.matches(',').count()),
        (b'\t', first.matches('\t').count()),
        (b';', first.matches(';').count()),
        (b'|', first.matches('|').count()),
    ];
    let (best, n) = counts.into_iter().max_by_key(|(_, n)| *n)?;
    if n == 0 { None } else { Some(best) }
}

/// Parse a JSON array of objects, or NDJSON (one object per line).
/// Header is the **union** of keys across all objects in source order.
fn parse_json(content: &str) -> Option<Rows> {
    let trimmed = content.trim_start();

    // Whole-document array path. Skip cheaply if the input doesn't
    // even start with `[` — otherwise serde will scan the whole blob.
    if trimmed.starts_with('[')
        && let Ok(Value::Array(arr)) = serde_json::from_str::<Value>(content)
        && let Some(rows) = build_rows_from_objects(arr.iter())
    {
        return Some(rows);
    }

    // NDJSON / JSONL fallback: one JSON value per non-blank line.
    let parsed: Vec<Value> = content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .collect();
    if parsed.is_empty() {
        return None;
    }
    build_rows_from_objects(parsed.iter())
}

/// Build a `Rows` view from an iterator of JSON values, keeping only
/// the objects. Header is the union of keys in first-seen order;
/// missing values become empty strings, non-string scalars get
/// `to_string`'d, nested values are stringified as compact JSON.
fn build_rows_from_objects<'a, I: Iterator<Item = &'a Value> + Clone>(values: I) -> Option<Rows> {
    let mut header: Vec<String> = Vec::new();
    let mut seen: BTreeMap<String, ()> = BTreeMap::new();
    let mut object_count = 0usize;
    for v in values.clone() {
        if let Some(obj) = v.as_object() {
            object_count += 1;
            for k in obj.keys() {
                if seen.insert(k.clone(), ()).is_none() {
                    header.push(k.clone());
                }
            }
        }
    }
    if object_count == 0 || header.is_empty() {
        return None;
    }
    let rows: Vec<Vec<String>> = values
        .filter_map(|v| v.as_object())
        .map(|obj| {
            header
                .iter()
                .map(|k| match obj.get(k) {
                    Some(Value::String(s)) => s.clone(),
                    Some(Value::Null) | None => String::new(),
                    Some(v) => v.to_string(),
                })
                .collect()
        })
        .collect();
    Some(Rows { header, rows })
}

#[derive(Debug)]
struct Rows {
    header: Vec<String>,
    rows: Vec<Vec<String>>,
}

fn chunk_rows(rows: &Rows, config: &ChunkConfig, source: &str) -> Vec<ChunkData> {
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
    chunk_index: &mut usize,
) {
    let mut text =
        String::with_capacity(header_md.len() + buf_rows.iter().map(|s| s.len()).sum::<usize>());
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
        s.push_str(&escape_md_cell(c));
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
        s.push_str(&escape_md_cell(c));
        s.push_str(" |");
    }
    s.push('\n');
    s
}

/// Escape a Markdown table cell. Pipes break the column split; CR/LF
/// break the row split. The Markdown table format has no real escape
/// for newlines — `<br>` is the GFM-supported convention.
fn escape_md_cell(s: &str) -> String {
    let trimmed = s.trim();
    if !trimmed.contains('|') && !trimmed.contains('\n') && !trimmed.contains('\r') {
        return trimmed.to_string();
    }
    let mut out = String::with_capacity(trimmed.len() + 4);
    for ch in trimmed.chars() {
        match ch {
            '|' => out.push_str("\\|"),
            '\n' | '\r' => {
                // Collapse any CRLF / CR / LF run into a single <br>.
                if !out.ends_with("<br>") {
                    out.push_str("<br>");
                }
            }
            _ => out.push(ch),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // ────────────────── Existing tests (must stay green) ──────────────────

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
        let mut csv = String::from("col_a,col_b\n");
        for i in 0..1000 {
            csv.push_str(&format!("value_{i},sample_data_{i}\n"));
        }
        let chunks = StructuredChunker.chunk(&csv, &ChunkConfig::default());
        assert!(
            chunks.len() > 1,
            "expected multi-chunk output, got {}",
            chunks.len()
        );
        for c in &chunks {
            assert!(
                c.text.starts_with("| col_a | col_b |"),
                "chunk missing header: {}",
                c.text
            );
        }
    }

    // ─────────────────────── New correctness tests ────────────────────────

    #[test]
    fn test_csv_quoted_field_with_embedded_comma() {
        // RFC 4180 quoted field carrying a literal comma — the
        // pre-fix split-on-',' chunker mis-parsed this as 3 columns.
        let csv = "name,age\n\"smith, john\",30\n\"doe, jane\",25";
        let chunks = StructuredChunker.chunk(csv, &ChunkConfig::default());
        assert_eq!(chunks[0].heading.as_deref(), Some("[\"name\",\"age\"]"));
        assert!(
            chunks[0].text.contains("smith, john"),
            "quoted comma not preserved: {}",
            chunks[0].text
        );
        assert!(
            chunks[0].text.contains("doe, jane"),
            "second quoted comma not preserved: {}",
            chunks[0].text
        );
    }

    #[test]
    fn test_csv_multi_line_quoted_cell() {
        // Quoted field spanning a newline — the pre-fix chunker's
        // `content.lines()` would shred this into two rows.
        let csv = "name,note\nalice,\"line one\nline two\"\nbob,short";
        let chunks = StructuredChunker.chunk(csv, &ChunkConfig::default());
        assert!(!chunks.is_empty());
        // Exactly two data rows — pre-fix would have produced three.
        let data_row_count = chunks[0]
            .text
            .lines()
            .filter(|l| l.starts_with('|') && !l.contains("---") && !l.contains("name"))
            .count();
        assert_eq!(
            data_row_count, 2,
            "expected 2 data rows, got {data_row_count}: {}",
            chunks[0].text
        );
        // The multi-line value lands in a single cell with the
        // embedded newline escaped to <br>.
        assert!(
            chunks[0].text.contains("line one<br>line two"),
            "multi-line cell not preserved: {}",
            chunks[0].text
        );
    }

    #[test]
    fn test_csv_tab_delimited() {
        let tsv = "name\tage\nalice\t30\nbob\t25";
        let chunks = StructuredChunker.chunk(tsv, &ChunkConfig::default());
        assert!(!chunks.is_empty(), "TSV should not fall back");
        assert_eq!(chunks[0].heading.as_deref(), Some("[\"name\",\"age\"]"));
        assert!(chunks[0].text.contains("alice"));
    }

    #[test]
    fn test_csv_semicolon_delimited() {
        // European-locale CSV.
        let csv = "name;age\nalice;30\nbob;25";
        let chunks = StructuredChunker.chunk(csv, &ChunkConfig::default());
        assert!(!chunks.is_empty());
        assert_eq!(chunks[0].heading.as_deref(), Some("[\"name\",\"age\"]"));
        assert!(chunks[0].text.contains("alice"));
    }

    #[test]
    fn test_csv_windows_line_endings() {
        let csv = "name,age\r\nalice,30\r\nbob,25\r\n";
        let chunks = StructuredChunker.chunk(csv, &ChunkConfig::default());
        assert!(!chunks.is_empty());
        assert!(chunks[0].text.contains("alice"));
        assert!(chunks[0].text.contains("bob"));
    }

    #[test]
    fn test_json_ndjson_detection() {
        // One JSON object per line — the pre-fix chunker required a
        // top-level array and fell back to TextChunker here.
        let ndjson = "{\"name\":\"alice\",\"age\":30}\n{\"name\":\"bob\",\"age\":25}";
        let chunks = StructuredChunker.chunk(ndjson, &ChunkConfig::default());
        assert_eq!(chunks[0].chunk_type, "table_row_group");
        assert_eq!(chunks[0].heading.as_deref(), Some("[\"name\",\"age\"]"));
        assert!(chunks[0].text.contains("alice"));
        assert!(chunks[0].text.contains("bob"));
    }

    #[test]
    fn test_json_union_of_keys() {
        // Second object has an extra key `b` that the pre-fix chunker
        // silently dropped (header was first-object keys only).
        let json = "[{\"a\":1},{\"a\":2,\"b\":3}]";
        let chunks = StructuredChunker.chunk(json, &ChunkConfig::default());
        assert_eq!(chunks[0].heading.as_deref(), Some("[\"a\",\"b\"]"));
        // `b` is present as a column header
        assert!(chunks[0].text.contains("| a | b |"));
        // and its value `3` appears
        assert!(chunks[0].text.contains("| 3 |"));
    }

    #[test]
    fn test_markdown_cell_escapes_pipe_and_newline() {
        // CSV value carrying `|` and `\n` would break the Markdown
        // table syntax pre-fix.
        let csv = "k,v\nx,\"a|b\nc\"";
        let chunks = StructuredChunker.chunk(csv, &ChunkConfig::default());
        let text = &chunks[0].text;
        // The pipe is escaped, the newline becomes <br>.
        assert!(text.contains("a\\|b<br>c"), "expected escaped cell, got: {text}");
        // Every data line must still have the expected number of
        // unescaped `|` separators (= cols + 1 = 3).
        for line in text.lines().filter(|l| l.contains("a\\|b<br>c")) {
            let separators = line.matches('|').count() - line.matches("\\|").count();
            assert_eq!(
                separators, 3,
                "broken Markdown row (separator count) in: {line}"
            );
        }
    }
}
