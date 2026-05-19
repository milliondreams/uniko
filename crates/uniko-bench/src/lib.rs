//! Shared utilities for uniko benchmark harnesses.
//!
//! Contains code reused across multiple benchmark binaries
//! (LoCoMo, LongMemEval, etc.): KB lifecycle, LLM catalog
//! construction, context formatting, and serde helpers.

pub mod longmemeval;

use std::collections::HashMap;
use std::fmt::Write;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use uni_db::{ModelAliasSpec, ModelTask, WarmupPolicy};
use uniko_memory::recall::ContextBundle;
use uniko_store::config::UnikoConfig;
use uniko_store::{KnowledgeBase, ModelRuntime};

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

/// Open an existing persistent KB using a process-shared `ModelRuntime`.
///
/// Same as [`open_kb`] but reuses an externally-built runtime instead
/// of loading its own ONNX sessions — required for `q ≥ 3` on an 8 GB
/// GPU where per-KB sessions OOM. See
/// [`KnowledgeBase::build_shared_runtime`].
pub async fn open_kb_with_runtime(
    ingest_dir: &Path,
    config: UnikoConfig,
    runtime: Arc<ModelRuntime>,
) -> Result<Arc<KnowledgeBase>> {
    let kb = KnowledgeBase::open_with_runtime(ingest_dir, config, runtime)
        .await
        .context("opening KB with shared runtime")?;
    Ok(Arc::new(kb))
}

// ── LLM Catalog ─────────────────────────────────────────────────

/// Build model alias specs for LLM generation and judging.
///
/// Creates catalog entries for the generation LLM and optionally a
/// separate judge LLM. The judge can use a different provider and
/// model from the generator (e.g. local mistralrs for generation +
/// remote OpenAI for judging).
///
/// `judge_provider` and `judge_model_id` default to the generator's
/// values when not specified. `llm_provider` defaults to
/// `"local/mistralrs"`.
#[allow(clippy::too_many_arguments)]
pub fn build_llm_catalog(
    llm_alias: Option<&str>,
    llm_model_id: Option<&str>,
    llm_provider: Option<&str>,
    llm_base_url: Option<&str>,
    judge_alias: Option<&str>,
    judge_model_id: Option<&str>,
    judge_provider: Option<&str>,
    judge_base_url: Option<&str>,
) -> Vec<ModelAliasSpec> {
    let mut catalog = Vec::new();

    let gen_provider = llm_provider.unwrap_or("local/mistralrs");

    if let (Some(alias), Some(model_id)) = (llm_alias, llm_model_id) {
        catalog.push(ModelAliasSpec {
            alias: alias.to_string(),
            task: ModelTask::Generate,
            provider_id: gen_provider.to_string(),
            model_id: model_id.to_string(),
            revision: None,
            warmup: WarmupPolicy::Lazy,
            required: false,
            timeout: None,
            load_timeout: None,
            retry: None,
            options: provider_options(gen_provider, llm_base_url),
        });
    }

    if let Some(judge) = judge_alias {
        let same_as_gen = llm_alias == Some(judge);
        if !same_as_gen {
            // Fall back to generator's provider/model when the caller
            // doesn't override.
            let provider = judge_provider.unwrap_or(gen_provider);
            let model_id = judge_model_id
                .or(llm_model_id)
                .expect("judge_alias requires either judge_model_id or llm_model_id");
            catalog.push(ModelAliasSpec {
                alias: judge.to_string(),
                task: ModelTask::Generate,
                provider_id: provider.to_string(),
                model_id: model_id.to_string(),
                revision: None,
                warmup: WarmupPolicy::Lazy,
                required: false,
                timeout: None,
                load_timeout: None,
                retry: None,
                options: provider_options(provider, judge_base_url),
            });
        }
    }

    catalog
}

/// Per-provider default options for a generation alias.
///
/// `local/mistralrs` defaults to ISQ Q4K so the user doesn't need a
/// pre-quantized model file. `remote/openai` (and OpenAI-compatible
/// servers like LM Studio / vLLM / llama.cpp) accept an optional
/// `base_url` override; uni-xervo 0.10+ honors it.
fn provider_options(provider: &str, base_url: Option<&str>) -> serde_json::Value {
    match provider {
        "local/mistralrs" => serde_json::json!({"isq": "Q4K"}),
        "remote/openai" => match base_url {
            Some(url) => serde_json::json!({"base_url": url}),
            None => serde_json::json!({}),
        },
        _ => serde_json::json!({}),
    }
}

// ── Context Formatting ──────────────────────────────────────────

/// Format recall items into a numbered context string for LLM prompts.
///
/// `session_dates` maps `node_id` → ISO date of the originating
/// Session.started_at.  `temporal_anchors` maps `node_id` → ISO date of
/// the resolved `Observation.temporal_anchor` (Phase A — surface form
/// like `"Last Fri"` is resolved at ingest and stored as an absolute
/// date so the LLM doesn't need to re-resolve it).  Items missing
/// either lookup emit no corresponding field.
///
/// Pass empty maps to skip the respective injection.
pub fn format_context(
    bundle: &ContextBundle,
    session_dates: &HashMap<i64, String>,
    temporal_anchors: &HashMap<i64, String>,
) -> String {
    let mut ctx = String::new();
    for (i, item) in bundle.items.iter().enumerate() {
        let session_date = session_dates.get(&item.node_id);
        let temporal = temporal_anchors.get(&item.node_id);
        let mut header = format!("[{}] ({}, score={:.3}", i + 1, item.node_type, item.score,);
        if let Some(date) = session_date {
            header.push_str(&format!(", session_date={date}"));
        }
        if let Some(t) = temporal {
            header.push_str(&format!(", temporal={t}"));
        }
        let _ = writeln!(&mut ctx, "{header}): {}", item.content);
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
