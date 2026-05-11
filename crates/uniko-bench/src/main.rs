//! LoCoMo benchmark harness for uniko cognitive memory.
//!
//! Ingests LoCoMo conversations, runs recall queries, and evaluates
//! answer quality using token-level F1 and LLM-as-judge metrics.
//!
//! # Usage
//!
//! ```bash
//! # Retrieval-only (no LLM needed)
//! cargo run -p uniko-bench -- --data locomo10.json --retrieval-only
//!
//! # With local LLM
//! cargo run -p uniko-bench -- --data locomo10.json \
//!     --llm-alias llm/gemma4 --llm-model-id google/gemma-3-4b-it
//! ```

mod data;
mod eval;
mod ingest;
mod query;
mod report;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;

use data::{QuestionCategory, build_evidence_lookup, parse_sessions, resolve_evidence};
use query::QueryResult;

/// LoCoMo benchmark for uniko cognitive memory.
#[derive(Parser)]
#[command(name = "uniko-bench", about = "Run LoCoMo benchmark against uniko")]
struct Cli {
    /// Path to locomo10.json data file.
    #[arg(long)]
    data: PathBuf,

    /// LLM model alias for answer generation.
    #[arg(long)]
    llm_alias: Option<String>,

    /// HuggingFace model ID for the LLM (e.g., "google/gemma-3-4b-it").
    #[arg(long)]
    llm_model_id: Option<String>,

    /// Separate LLM alias for judge (defaults to llm-alias).
    #[arg(long)]
    judge_alias: Option<String>,

    /// HuggingFace model ID (or provider model name) for the judge LLM.
    /// Defaults to llm-model-id.
    #[arg(long)]
    judge_model_id: Option<String>,

    /// Provider for the judge LLM (e.g. "remote/openai" for GPT-5).
    /// Defaults to "local/mistralrs".
    #[arg(long)]
    judge_provider: Option<String>,

    /// Provider for the generation LLM. Defaults to "local/mistralrs".
    #[arg(long)]
    llm_provider: Option<String>,

    /// Base URL of an OpenAI-compatible HTTP server for the generation
    /// LLM (e.g. "http://127.0.0.1:1234/v1"). When set, the gen alias
    /// is registered against `remote/openai` with this base URL —
    /// useful for LM Studio / vLLM / llama.cpp servers exposing
    /// `/v1/chat/completions`. Implies `--llm-provider remote/openai`.
    #[arg(long)]
    llm_base_url: Option<String>,

    /// Base URL of an OpenAI-compatible server for the judge LLM.
    /// Defaults to OpenAI's public API.
    #[arg(long)]
    judge_base_url: Option<String>,

    /// Run only specific conversation IDs (comma-separated).
    #[arg(long)]
    conversations: Option<String>,

    /// Run only specific question categories (comma-separated, 1-5).
    #[arg(long)]
    categories: Option<String>,

    /// Output JSON report file path.
    #[arg(long, default_value = "locomo_results.json")]
    output: PathBuf,

    /// Skip LLM judge evaluation.
    #[arg(long)]
    no_judge: bool,

    /// Retrieval-only mode (no LLM generation).
    #[arg(long)]
    retrieval_only: bool,

    /// Directory for persistent KB storage.
    #[arg(long, default_value = "data/kb")]
    ingest_dir: PathBuf,

    /// Reuse existing KB from --ingest-dir (skip ingestion).
    #[arg(long)]
    reuse: bool,

    /// Path to xervo model catalog JSON file.
    #[arg(long)]
    catalog: Option<PathBuf>,

    /// Path to schema JSON file.
    #[arg(long)]
    schema: Option<PathBuf>,

    /// Override embedding dimensions (default 768 for Nomic, use 384 for MiniLM).
    #[arg(long)]
    embedding_dim: Option<usize>,

    /// Embedding preset to use. One of:
    /// - `nomic` (default, 768d)
    /// - `minilm` (384d, BERT-based)
    /// - `bge-small` (384d, BGE-small-en-v1.5)
    /// - `bge-large` (1024d, BGE-large-en-v1.5)
    /// Selects the full preset (model_id + dimensions + prefixes).
    #[arg(long, default_value = "nomic")]
    embedding: String,

    /// Enable the reranker to re-score the top recall candidates before
    /// truncating to limit. Costs an extra ONNX call per query but
    /// typically helps multi-hop / open-domain where bi-encoder ranking
    /// misses synonym/implication links.
    #[arg(long)]
    reranker: bool,

    /// HuggingFace model id for the reranker. xervo 0.11 auto-detects
    /// `token_type_ids`, so XLM-R-based models (e.g.
    /// `BAAI/bge-reranker-base`) work alongside BERT-based ones
    /// (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`). For
    /// `--reranker-style generative`, use a decoder-LM export such as
    /// `onnx-community/Qwen3-Reranker-0.6B-ONNX`.
    #[arg(long, default_value = "BAAI/bge-reranker-base")]
    reranker_model: String,

    /// Reranker code path. `cross-encoder` (default) for BERT-family
    /// (BGE/MiniLM) cross-encoders that emit a relevance logit;
    /// `generative` for decoder-LM rerankers that score yes/no via
    /// next-token logits (Qwen3-Reranker and compatible exports).
    #[arg(long, default_value = "cross-encoder")]
    reranker_style: String,

    /// Top-N RRF candidates fed to the reranker. Memory cost scales
    /// roughly linearly for cross-encoders and super-linearly for
    /// decoder-LM rerankers (KV-cache). Default 50 works for BGE on a
    /// modern GPU; lower (10-15) for Qwen3-Reranker to stay under VRAM.
    #[arg(long, default_value = "50")]
    reranker_top_n: usize,

    /// Comma-separated list of query-reformulation variants to enable.
    /// Recognised: `keywords`, `original`, `declarative`, `type_anchored`.
    /// Empty / unset uses the default 4-variant configuration. Pass
    /// `keywords` alone for legacy single-query behaviour.
    #[arg(long, default_value = "")]
    variants: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                "uniko_bench=info,uniko_memory=warn,uniko_extract=warn,uniko_store=warn"
                    .parse()
                    .unwrap()
            }),
        )
        .init();

    let cli = Cli::parse();

    // Load dataset.
    tracing::info!(path = %cli.data.display(), "loading LoCoMo dataset");
    let samples = data::load_locomo(&cli.data)?;
    tracing::info!(conversations = samples.len(), "dataset loaded");

    // Filter conversations if specified.
    let samples: Vec<_> = if let Some(ref filter) = cli.conversations {
        let ids: Vec<&str> = filter.split(',').map(|s| s.trim()).collect();
        samples
            .into_iter()
            .filter(|s| ids.contains(&s.sample_id.as_str()))
            .collect()
    } else {
        samples
    };

    // Filter categories if specified.
    let category_filter: Option<Vec<u32>> = cli
        .categories
        .as_ref()
        .map(|c| c.split(',').filter_map(|s| s.trim().parse().ok()).collect());

    // Build config with optional catalog/schema/embedding overrides.
    let mut config = uniko_store::config::UnikoConfig {
        catalog_path: cli.catalog.clone(),
        schema_path: cli.schema.clone(),
        ..Default::default()
    };
    // Apply embedding preset BEFORE explicit `--embedding-dim` override so
    // the latter wins when both are passed.
    config.embedding = match cli.embedding.as_str() {
        "nomic" => uniko_store::config::EmbeddingConfig::nomic_v15(),
        "minilm" => uniko_store::config::EmbeddingConfig::minilm_l6_v2(),
        "bge-small" => uniko_store::config::EmbeddingConfig::bge_small_en_v15(),
        "bge-large" => uniko_store::config::EmbeddingConfig::bge_large_en_v15(),
        other => anyhow::bail!(
            "unknown --embedding preset {other:?}; expected one of: nomic, minilm, bge-small, bge-large"
        ),
    };
    if let Some(dim) = cli.embedding_dim {
        config.embedding.dimensions = dim;
    }
    if cli.reranker {
        config.reranker.enabled = true;
        config.reranker.model_id = cli.reranker_model.clone();
        config.reranker.style = cli.reranker_style.clone();
        config.reranker.top_n = cli.reranker_top_n;
    }
    if !cli.variants.trim().is_empty() {
        config.query_variants = cli
            .variants
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
    }

    // Build extra catalog for LLM.
    // `--llm-base-url` implies `remote/openai` provider unless explicitly
    // overridden — that's how an OpenAI-compatible local server (LM Studio,
    // vLLM, llama.cpp) is reached.
    let llm_provider = cli.llm_provider.as_deref().or_else(|| {
        if cli.llm_base_url.is_some() {
            Some("remote/openai")
        } else {
            None
        }
    });
    let extra_catalog = uniko_bench::build_llm_catalog(
        cli.llm_alias.as_deref(),
        cli.llm_model_id.as_deref(),
        llm_provider,
        cli.llm_base_url.as_deref(),
        cli.judge_alias.as_deref(),
        cli.judge_model_id.as_deref(),
        cli.judge_provider.as_deref(),
        cli.judge_base_url.as_deref(),
    );
    for spec in &extra_catalog {
        tracing::info!(
            alias = %spec.alias,
            provider = %spec.provider_id,
            model = %spec.model_id,
            options = %spec.options,
            "catalog entry"
        );
    }
    let llm_alias = if cli.retrieval_only {
        None
    } else {
        cli.llm_alias.as_deref()
    };
    let judge_alias = if cli.no_judge || cli.retrieval_only {
        None
    } else {
        cli.judge_alias.as_deref().or(cli.llm_alias.as_deref())
    };

    // Run benchmark.
    let mut all_results: Vec<(QueryResult, f64, Option<f64>)> = Vec::new();
    let bench_start = Instant::now();

    for (conv_idx, sample) in samples.iter().enumerate() {
        tracing::info!(
            "[{}/{}] Processing {}",
            conv_idx + 1,
            samples.len(),
            sample.sample_id,
        );

        // Parse sessions.
        let sessions = parse_sessions(&sample.sample_id, &sample.conversation)
            .with_context(|| format!("parsing sessions for {}", sample.sample_id))?;

        tracing::info!(
            sessions = sessions.len(),
            turns = sessions.iter().map(|s| s.turns.len()).sum::<usize>(),
            questions = sample.qa.len(),
            "conversation structure",
        );

        // Ingest or reuse persistent KB.
        let kb_dir = cli.ingest_dir.join(&sample.sample_id);
        let kb = if cli.reuse && kb_dir.exists() {
            tracing::info!(path = %kb_dir.display(), "reusing existing KB");
            uniko_bench::open_kb(&kb_dir, config.clone(), &extra_catalog).await?
        } else {
            let ingest_start = Instant::now();
            let kb = ingest::ingest_conversation(
                sample,
                &sessions,
                &kb_dir,
                config.clone(),
                &extra_catalog,
            )
            .await?;
            tracing::info!(
                elapsed_ms = ingest_start.elapsed().as_millis(),
                "ingestion complete"
            );
            kb
        };

        // P4 Consolidation: derive Facts from the just-ingested
        // Observations so the recall cascade has a Phase 1 (Compact)
        // surface to query.  Idempotent — `--reuse` paths re-enter
        // here and only the freshly-added Observations get processed.
        // Cap is generous because per-conversation observation counts
        // can run into the thousands; spec default 500 would force
        // multiple cycles.
        let cycle_start = Instant::now();
        match uniko_memory::consolidation::run_cycle(&kb, &sample.sample_id, Some(10_000)).await {
            Ok(stats) => tracing::info!(
                sample_id = %sample.sample_id,
                processed = stats.observations_processed,
                facts_created = stats.facts_created,
                facts_reinforced = stats.facts_reinforced,
                duration_ms = cycle_start.elapsed().as_millis(),
                "consolidation cycle complete",
            ),
            Err(e) => tracing::warn!(
                sample_id = %sample.sample_id,
                error = %e,
                "consolidation cycle failed (continuing without Facts)",
            ),
        }

        // Build evidence lookup for retrieval evaluation.
        let evidence_lookup = build_evidence_lookup(&sessions);

        // Query each question.
        let questions: Vec<_> = sample
            .qa
            .iter()
            .filter(|qa| {
                category_filter
                    .as_ref()
                    .is_none_or(|cats| cats.contains(&qa.category))
            })
            .collect();

        for (q_idx, qa) in questions.iter().enumerate() {
            if (q_idx + 1) % 50 == 0 {
                tracing::info!(
                    "[{}/{}] questions processed for {}",
                    q_idx + 1,
                    questions.len(),
                    sample.sample_id,
                );
            }

            let evidence_texts = resolve_evidence(qa, &evidence_lookup);
            let qr = query::run_query(&kb, qa, &evidence_texts, llm_alias).await?;

            // Compute token-level F1.
            let f1 = eval::token_f1(&qr.predicted_answer, &qr.gold_answer, qr.category);

            // Run LLM judge (if enabled and not adversarial).
            let judge_score = if let Some(alias) = judge_alias
                && qr.category != QuestionCategory::Adversarial
            {
                match eval::llm_judge(
                    &kb,
                    &qr.question,
                    &qr.gold_answer,
                    &qr.predicted_answer,
                    alias,
                )
                .await
                {
                    Ok(score) => Some(score),
                    Err(e) => {
                        tracing::warn!(error = %e, "LLM judge failed");
                        None
                    }
                }
            } else {
                None
            };

            all_results.push((qr, f1, judge_score));
        }

        // Cleanly shut down the KB so the WAL is checkpointed into a
        // snapshot manifest. Without this, `--reuse` on a subsequent run
        // hits uni-db's "WAL segments but no snapshot manifest" guard
        // and refuses to open. `try_unwrap` only succeeds when nothing
        // else holds the Arc — after the question loop that's just us.
        match Arc::try_unwrap(kb) {
            Ok(kb_owned) => {
                if let Err(e) = kb_owned.shutdown().await {
                    tracing::warn!(sample_id = %sample.sample_id, error = %e, "kb shutdown failed");
                }
            }
            Err(_arc) => {
                tracing::warn!(
                    sample_id = %sample.sample_id,
                    "skipping kb shutdown: outstanding Arc references prevent unwrap",
                );
            }
        }
        tracing::info!(
            sample_id = %sample.sample_id,
            "conversation complete",
        );

        // Per-conversation checkpoint: write the running results to the
        // output path so a kill mid-run preserves everything completed.
        // `conv_idx + 1` is the count-so-far for averages; the final
        // write below uses `samples.len()` for the full denominator.
        let partial = report::aggregate(&all_results, conv_idx + 1);
        if let Err(e) = report::write_json(&all_results, &partial, &cli.output) {
            tracing::warn!(error = %e, "checkpoint write failed (continuing)");
        }
    }

    let bench_elapsed = bench_start.elapsed();
    tracing::info!(
        total_secs = bench_elapsed.as_secs(),
        total_questions = all_results.len(),
        "benchmark complete",
    );

    // Aggregate and report.
    let report = report::aggregate(&all_results, samples.len());
    report::print_report(&report);
    report::write_json(&all_results, &report, &cli.output)?;
    tracing::info!(path = %cli.output.display(), "results written");

    Ok(())
}
