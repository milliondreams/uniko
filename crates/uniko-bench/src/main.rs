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
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use uni_db::{ModelAliasSpec, ModelTask, WarmupPolicy};

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

    // Build config with optional catalog/schema paths.
    let mut config = uniko_store::config::UnikoConfig::default();
    config.catalog_path = cli.catalog.clone();
    config.schema_path = cli.schema.clone();

    // Build extra catalog for LLM.
    let extra_catalog = build_llm_catalog(&cli);
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
            ingest::open_kb(&kb_dir, config.clone(), &extra_catalog).await?
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

        // KB is dropped here, freeing memory.
        tracing::info!(
            sample_id = %sample.sample_id,
            "conversation complete",
        );
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

/// Build LLM model aliases for the xervo catalog.
fn build_llm_catalog(cli: &Cli) -> Vec<ModelAliasSpec> {
    let mut catalog = Vec::new();

    if let (Some(alias), Some(model_id)) = (&cli.llm_alias, &cli.llm_model_id) {
        catalog.push(ModelAliasSpec {
            alias: alias.clone(),
            task: ModelTask::Generate,
            provider_id: "local/mistralrs".to_string(),
            model_id: model_id.clone(),
            revision: None,
            warmup: WarmupPolicy::Lazy,
            required: false,
            timeout: None,
            load_timeout: None,
            retry: None,
            options: serde_json::json!({"isq": "Q4K"}),
        });

        // If judge alias is different, add it separately.
        if let Some(judge) = &cli.judge_alias
            && judge != alias
        {
            catalog.push(ModelAliasSpec {
                alias: judge.clone(),
                task: ModelTask::Generate,
                provider_id: "local/mistralrs".to_string(),
                model_id: model_id.clone(),
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

    catalog
}
