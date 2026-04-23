//! Shared utilities for uniko benchmark harnesses.
//!
//! Contains code reused across multiple benchmark binaries
//! (LoCoMo, LongMemEval, etc.): KB lifecycle, LLM catalog
//! construction, context formatting, and serde helpers.

pub mod longmemeval;

use std::fmt::Write;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use uni_db::{ModelAliasSpec, ModelTask, WarmupPolicy};
use uniko_memory::recall::ContextBundle;
use uniko_store::KnowledgeBase;
use uniko_store::config::UnikoConfig;

// ── KB Lifecycle ────────────────────────────────────────────────

/// Open an existing persistent KB without ingesting.
pub async fn open_kb(
    ingest_dir: &Path,
    config: UnikoConfig,
    extra_catalog: &[ModelAliasSpec],
) -> Result<Arc<KnowledgeBase>> {
    let kb = KnowledgeBase::open_with_xervo(ingest_dir, config, extra_catalog.to_vec())
        .await
        .context("opening KB")?;
    Ok(Arc::new(kb))
}

// ── LLM Catalog ─────────────────────────────────────────────────

/// Build model alias specs for LLM generation and judging.
///
/// Creates catalog entries for the generation LLM and optionally
/// a separate judge LLM (if `judge_alias` differs from `llm_alias`).
pub fn build_llm_catalog(
    llm_alias: Option<&str>,
    llm_model_id: Option<&str>,
    judge_alias: Option<&str>,
) -> Vec<ModelAliasSpec> {
    let mut catalog = Vec::new();

    if let (Some(alias), Some(model_id)) = (llm_alias, llm_model_id) {
        catalog.push(ModelAliasSpec {
            alias: alias.to_string(),
            task: ModelTask::Generate,
            provider_id: "local/mistralrs".to_string(),
            model_id: model_id.to_string(),
            revision: None,
            warmup: WarmupPolicy::Lazy,
            required: false,
            timeout: None,
            load_timeout: None,
            retry: None,
            options: serde_json::json!({"isq": "Q4K"}),
        });

        // If judge alias is different, add it separately.
        if let Some(judge) = judge_alias {
            if judge != alias {
                catalog.push(ModelAliasSpec {
                    alias: judge.to_string(),
                    task: ModelTask::Generate,
                    provider_id: "local/mistralrs".to_string(),
                    model_id: model_id.to_string(),
                    revision: None,
                    warmup: WarmupPolicy::Lazy,
                    required: false,
                    timeout: None,
                    load_timeout: None,
                    retry: None,
                    options: serde_json::json!({"isq": "Q4K"}),
                });
            }
        }
    }

    catalog
}

// ── Context Formatting ──────────────────────────────────────────

/// Format recall items into a numbered context string for LLM prompts.
pub fn format_context(bundle: &ContextBundle) -> String {
    let mut ctx = String::new();
    for (i, item) in bundle.items.iter().enumerate() {
        let _ = writeln!(
            &mut ctx,
            "[{}] ({}, score={:.3}): {}",
            i + 1,
            item.node_type,
            item.score,
            item.content,
        );
    }
    ctx
}

/// In retrieval-only mode, concatenate top recall items as the answer.
pub fn retrieval_answer(bundle: &ContextBundle, top_k: usize) -> String {
    bundle
        .items
        .iter()
        .take(top_k)
        .map(|item| item.content.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

// ── Serde Helpers ───────────────────────────────────────────────

/// Deserialize a value that may be a string or a number into `Option<String>`.
///
/// Many benchmark datasets have answer fields with mixed types
/// (string for most, int for counting questions). This visitor
/// coerces all numeric types to their string representation.
pub fn string_or_number<'de, D>(deserializer: D) -> std::result::Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de;

    struct StringOrNumber;
    impl<'de> de::Visitor<'de> for StringOrNumber {
        type Value = Option<String>;
        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "a string or number")
        }
        fn visit_str<E: de::Error>(self, v: &str) -> std::result::Result<Self::Value, E> {
            Ok(Some(v.to_string()))
        }
        fn visit_i64<E: de::Error>(self, v: i64) -> std::result::Result<Self::Value, E> {
            Ok(Some(v.to_string()))
        }
        fn visit_u64<E: de::Error>(self, v: u64) -> std::result::Result<Self::Value, E> {
            Ok(Some(v.to_string()))
        }
        fn visit_f64<E: de::Error>(self, v: f64) -> std::result::Result<Self::Value, E> {
            Ok(Some(v.to_string()))
        }
        fn visit_none<E: de::Error>(self) -> std::result::Result<Self::Value, E> {
            Ok(None)
        }
        fn visit_unit<E: de::Error>(self) -> std::result::Result<Self::Value, E> {
            Ok(None)
        }
    }
    deserializer.deserialize_any(StringOrNumber)
}
