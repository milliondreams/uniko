//! JSONL per-event writer for ingest and query metrics.
//!
//! One [`BenchEvent`] is appended per ingested turn and per asked
//! question.  The file is JSONL (one JSON object per line), opened
//! append-mode so a resumed run extends the same file rather than
//! truncating it.  This is the in-tree analogue of KTH's per-row CSV
//! that downstream cost / latency analysis consumes.
//!
//! Embedding token counts are *estimated* (chars / 4) because
//! `uni-db`'s `xervo.embed()` discards the provider's `usage` field.
//! See [`feedback_no_edit_uni`] in the user's memory — embed-usage
//! capture requires a uni-db patch that is out of scope here.

// Rust guideline compliant

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde::Serialize;

/// One row emitted to the JSONL stream.
///
/// Tagged on the `kind` field so a downstream reader can dispatch on
/// `kind == "ingest_turn"` vs `kind == "query"` without needing the
/// enum's Rust type.  Fields are flattened into the same object as
/// `kind` for ease of analysis.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BenchEvent {
    /// One ingested LoCoMo turn (message + extraction).
    IngestTurn {
        /// LoCoMo `sample_id` this turn belongs to.
        sample_id: String,
        /// 0-based turn ordinal within the conversation.
        turn_index: usize,
        /// Unix epoch milliseconds when the event was emitted.
        ts_unix_ms: u64,
        /// Wall-clock for the full turn (ingest + NER + observation).
        wall_ms: u64,
        /// Message ingestion sub-phase (node + edges + chunks).
        ingest_ms: u64,
        /// Named-entity extraction sub-phase.
        ner_ms: u64,
        /// Observation extraction sub-phase.
        obs_ms: u64,
        /// Provider-reported input tokens for any remote LLM that
        /// fired during this turn.  Default config (local NLP
        /// cascade) leaves this `None`.
        extraction_input_tokens: Option<u64>,
        /// Provider-reported output tokens for any remote LLM that
        /// fired during this turn.
        extraction_output_tokens: Option<u64>,
        /// Model alias that issued the call, if any.
        extraction_model: Option<String>,
        /// Raw character count handed to the embedder.  Used as a
        /// cheap proxy for embedding tokens (chars / 4 ≈ tokens) when
        /// the embedding model has remote pricing; local embedders
        /// yield 0 cost regardless.
        embedding_chars: u64,
        /// Embedding model alias.
        embedding_model: Option<String>,
        /// Total USD cost for this turn (extraction LLM +
        /// embedding).  `0.0` when every component is local.
        cost_usd: f64,
    },
    /// One asked LoCoMo question (recall + answer + judge).
    Query {
        sample_id: String,
        question_index: usize,
        ts_unix_ms: u64,
        wall_ms: u64,
        recall_ms: u64,
        generation_ms: u64,
        /// Wall-clock for the judge call when one ran (`None` for
        /// adversarial questions and `--no-judge` runs).
        judge_ms: Option<u64>,
        answer_input_tokens: Option<u64>,
        answer_output_tokens: Option<u64>,
        answer_model: String,
        judge_input_tokens: Option<u64>,
        judge_output_tokens: Option<u64>,
        judge_model: Option<String>,
        evidence_found: usize,
        evidence_total: usize,
        f1: f64,
        judge_score: Option<f64>,
        /// Total USD cost for this question (answer LLM + judge LLM).
        cost_usd: f64,
    },
}

/// JSONL writer that serialises [`BenchEvent`]s to a file.
///
/// Append-mode and `Mutex`-guarded so concurrent ingest / query loops
/// can share one writer.
#[derive(Debug)]
pub struct EventWriter {
    file: Mutex<BufWriter<File>>,
}

impl EventWriter {
    /// Open (or create) the JSONL file at `path` for appending.
    ///
    /// Parent directories are created if missing.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be created or the
    /// file cannot be opened.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating events directory {}", parent.display()))?;
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .with_context(|| format!("opening events JSONL at {}", path.display()))?;
        Ok(Self {
            file: Mutex::new(BufWriter::new(file)),
        })
    }

    /// Serialise and append one event as a single JSON line.
    ///
    /// Flushes after every line so a killed process does not lose the
    /// last few events.
    ///
    /// # Errors
    ///
    /// Returns an error if serialisation or write fails.
    pub fn write(&self, event: &BenchEvent) -> Result<()> {
        let line = serde_json::to_string(event).context("serialising bench event")?;
        let mut guard = self
            .file
            .lock()
            .map_err(|_| anyhow::anyhow!("events writer mutex poisoned"))?;
        guard.write_all(line.as_bytes())?;
        guard.write_all(b"\n")?;
        guard.flush()?;
        Ok(())
    }
}

/// Current wall-clock as Unix milliseconds.
///
/// Returns 0 if the clock is somehow before the epoch — the bench
/// has no recovery path for that and a zero timestamp surfaces the
/// anomaly without crashing the run.
#[must_use]
pub fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn writes_jsonl_lines() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("events.jsonl");
        let w = EventWriter::open(&path).expect("open");
        w.write(&BenchEvent::Query {
            sample_id: "conv-26".into(),
            question_index: 0,
            ts_unix_ms: 1_700_000_000_000,
            wall_ms: 50,
            recall_ms: 10,
            generation_ms: 40,
            judge_ms: Some(15),
            answer_input_tokens: Some(200),
            answer_output_tokens: Some(50),
            answer_model: "gpt-4o-mini".into(),
            judge_input_tokens: Some(120),
            judge_output_tokens: Some(20),
            judge_model: Some("gpt-4o-mini".into()),
            evidence_found: 1,
            evidence_total: 1,
            f1: 1.0,
            judge_score: Some(1.0),
            cost_usd: 0.001,
        })
        .expect("write");
        let body = std::fs::read_to_string(&path).expect("read");
        assert!(body.contains("\"kind\":\"query\""), "expected tag, got {body}");
        assert!(body.contains("\"sample_id\":\"conv-26\""));
        assert!(body.ends_with('\n'), "trailing newline");
    }

    #[test]
    fn appends_across_opens() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("events.jsonl");
        for i in 0..3 {
            let w = EventWriter::open(&path).expect("open");
            w.write(&BenchEvent::IngestTurn {
                sample_id: "conv-26".into(),
                turn_index: i,
                ts_unix_ms: 0,
                wall_ms: 1,
                ingest_ms: 1,
                ner_ms: 0,
                obs_ms: 0,
                extraction_input_tokens: None,
                extraction_output_tokens: None,
                extraction_model: None,
                embedding_chars: 100,
                embedding_model: Some("bge-small-en-v1.5".into()),
                cost_usd: 0.0,
            })
            .expect("write");
        }
        let body = std::fs::read_to_string(&path).expect("read");
        assert_eq!(body.lines().count(), 3, "three lines appended");
    }
}
