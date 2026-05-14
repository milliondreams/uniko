//! LongMemEval benchmark harness for uniko cognitive memory.
//!
//! Ingests LongMemEval items (each with its own chat history haystack),
//! runs recall queries, and evaluates retrieval quality and answer
//! correctness.
//!
//! # Usage
//!
//! ```bash
//! # Phase 1 gate (retrieval-only, SSU+SSA+MS categories)
//! cargo run -p uniko-bench --bin longmemeval-bench -- \
//!     --data data/longmemeval_s_cleaned.json --phase1
//!
//! # Full benchmark with LLM
//! cargo run -p uniko-bench --bin longmemeval-bench -- \
//!     --data data/longmemeval_s_cleaned.json \
//!     --llm-alias llm/gemma4 --llm-model-id google/gemma-3-4b-it
//!
//! # Dev iteration (5 questions only)
//! cargo run -p uniko-bench --bin longmemeval-bench -- \
//!     --data data/longmemeval_s_cleaned.json --phase1 --max-questions 5
//! ```

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;

use uniko_bench::longmemeval::data::{self, LmeQuestionType};
use uniko_bench::longmemeval::eval;
use uniko_bench::longmemeval::ingest;
use uniko_bench::longmemeval::query;
use uniko_bench::longmemeval::report;

/// LongMemEval benchmark for uniko cognitive memory.
#[derive(Parser)]
#[command(
    name = "longmemeval-bench",
    about = "Run LongMemEval benchmark against uniko"
)]
struct Cli {
    /// Path to LongMemEval JSON file (e.g., longmemeval_s_cleaned.json).
    #[arg(long)]
    data: PathBuf,

    /// Phase 1 mode: filter to SSU+SSA+MS, retrieval-only, context_contains_answer metric.
    #[arg(long)]
    phase1: bool,

    /// Filter to specific question types (comma-separated: ssu,ssa,ssp,ms,tr,ku).
    #[arg(long)]
    question_types: Option<String>,

    /// Run only specific question IDs (comma-separated).
    #[arg(long)]
    questions: Option<String>,

    /// Max questions to process (for dev iteration).
    #[arg(long)]
    max_questions: Option<usize>,

    /// Token budget for recall (default 8192).
    #[arg(long, default_value = "8192")]
    token_budget: usize,

    /// LLM model alias for answer generation.
    #[arg(long)]
    llm_alias: Option<String>,

    /// HuggingFace model ID for the LLM (e.g., "google/gemma-3-4b-it").
    #[arg(long)]
    llm_model_id: Option<String>,

    /// Separate LLM alias for judge (defaults to llm-alias).
    #[arg(long)]
    judge_alias: Option<String>,

    /// Skip LLM judge evaluation.
    #[arg(long)]
    no_judge: bool,

    /// Output JSON report file path.
    #[arg(long, default_value = "longmemeval_results.json")]
    output: PathBuf,

    /// Directory for persistent KB storage.
    #[arg(long, default_value = "data/lme_kb")]
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

    /// Embedding preset to use. Must match what was used at ingest time.
    /// One of: `nomic` (default 768d), `minilm` (384d), `bge-small`
    /// (384d), `bge-large` (1024d).
    #[arg(long, default_value = "nomic")]
    embedding: String,

    /// Enable cross-encoder reranker on top of RRF candidates.
    #[arg(long)]
    reranker: bool,

    /// HuggingFace reranker model id. xervo 0.11 auto-detects whether
    /// the ONNX graph expects `token_type_ids`, so XLM-R-based models
    /// (e.g. `BAAI/bge-reranker-base`) work alongside BERT-based ones
    /// (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`).
    #[arg(long, default_value = "BAAI/bge-reranker-base")]
    reranker_model: String,

    /// Top-N RRF candidates to send through the reranker. Must be
    /// `>= recall_limit` when reranker is enabled.
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
                "longmemeval_bench=info,uniko_bench=info,uniko_memory=warn,uniko_extract=info,uniko_store=warn"
                    .parse()
                    .unwrap()
            }),
        )
        .init();

    let cli = Cli::parse();

    // Load dataset.
    tracing::info!(path = %cli.data.display(), "loading LongMemEval dataset");
    let mut items = data::load_longmemeval(&cli.data)?;
    tracing::info!(total_items = items.len(), "dataset loaded");

    // Apply question type filter.
    let type_filter: Option<Vec<LmeQuestionType>> = if cli.phase1 {
        // Phase 1 gate: SSU + SSA + MS only.
        Some(vec![
            LmeQuestionType::SingleSessionUser,
            LmeQuestionType::SingleSessionAssistant,
            LmeQuestionType::MultiSession,
        ])
    } else {
        cli.question_types.as_ref().map(|types_str| {
            types_str
                .split(',')
                .filter_map(|s| LmeQuestionType::from_shorthand(s.trim()))
                .collect()
        })
    };

    if let Some(ref filter) = type_filter {
        items.retain(|item| filter.contains(&item.question_type));
        tracing::info!(
            filtered = items.len(),
            types = ?filter.iter().map(|t| t.name()).collect::<Vec<_>>(),
            "filtered by question type",
        );
    }

    // Filter by specific question IDs.
    if let Some(ref ids_str) = cli.questions {
        let ids: Vec<&str> = ids_str.split(',').map(|s| s.trim()).collect();
        items.retain(|item| ids.contains(&item.question_id.as_str()));
        tracing::info!(filtered = items.len(), "filtered by question ID");
    }

    // Apply max questions limit.
    if let Some(max) = cli.max_questions {
        items.truncate(max);
        tracing::info!(truncated = items.len(), "limited to max questions");
    }

    // Determine LLM mode.
    let retrieval_only = cli.phase1 || cli.llm_alias.is_none();
    let llm_alias = if retrieval_only {
        None
    } else {
        cli.llm_alias.as_deref()
    };
    let judge_alias = if cli.no_judge || retrieval_only {
        None
    } else {
        cli.judge_alias.as_deref().or(cli.llm_alias.as_deref())
    };

    // Build config.
    let mut config = uniko_store::config::UnikoConfig {
        catalog_path: cli.catalog.clone(),
        schema_path: cli.schema.clone(),
        ..Default::default()
    };
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

    // Build LLM catalog.
    let extra_catalog = uniko_bench::build_llm_catalog(
        cli.llm_alias.as_deref(),
        cli.llm_model_id.as_deref(),
        None,
        None,
        cli.judge_alias.as_deref(),
        None,
        None,
        None,
    );

    // Run benchmark.
    let mut all_results: Vec<(query::LmeQueryResult, Option<f64>)> = Vec::new();
    let bench_start = Instant::now();
    let total_items = items.len();

    for (item_idx, item) in items.iter().enumerate() {
        tracing::info!(
            "[{}/{}] Processing {} (type={}, sessions={})",
            item_idx + 1,
            total_items,
            item.question_id,
            item.question_type.name(),
            item.haystack_sessions.len(),
        );

        // Ingest or reuse KB.
        let kb_dir = cli.ingest_dir.join(&item.question_id);
        let (kb, evidence_map) = if cli.reuse && kb_dir.exists() {
            tracing::info!(path = %kb_dir.display(), "reusing existing KB");
            let kb = uniko_bench::open_kb(&kb_dir, config.clone(), &extra_catalog).await?;
            // Reconstruct evidence map from item data.
            let evidence_map = ingest::EvidenceMap {
                answer_session_ids: item.answer_session_ids.iter().cloned().collect(),
                answer_message_ids: Vec::new(), // not available when reusing
                session_to_messages: std::collections::HashMap::new(),
            };
            (kb, evidence_map)
        } else {
            let ingest_start = Instant::now();
            let result = ingest::ingest_item(item, &kb_dir, config.clone(), &extra_catalog)
                .await
                .with_context(|| format!("ingesting {}", item.question_id))?;
            tracing::info!(
                elapsed_ms = ingest_start.elapsed().as_millis(),
                "ingestion complete"
            );
            result
        };

        // Ensure a bench-agent Participant exists so the post-query
        // episode recording has somewhere to attach `RECORDED_BY`.
        let bench_agent_id = format!("bench-agent-{}", item.question_id);
        let mut agent_props: std::collections::HashMap<String, uni_db::Value> =
            std::collections::HashMap::new();
        agent_props.insert("kind".into(), uni_db::Value::String("agent".into()));
        agent_props.insert("name".into(), uni_db::Value::String("bench-agent".into()));
        if let Err(e) = kb
            .merge_node(
                "Participant",
                "participant_id",
                &bench_agent_id,
                &agent_props,
            )
            .await
        {
            tracing::warn!(error = %e, "failed to create bench-agent Participant");
        }

        // Query.
        let gold = data::gold_answer(item);
        let qr = query::run_lme_query(
            &kb,
            &item.question_id,
            &item.question,
            item.question_type,
            gold,
            &evidence_map,
            cli.token_budget,
            llm_alias,
        )
        .await
        .with_context(|| format!("querying {}", item.question_id))?;

        // Run judge (if enabled).
        let judge_score = if let Some(alias) = judge_alias {
            let is_abstention = LmeQuestionType::is_abstention(&item.question_id);
            if is_abstention {
                // For abstention questions, use rule-based scoring.
                Some(eval::abstention_score(&qr.predicted_answer))
            } else {
                match eval::lme_judge(
                    &kb,
                    &item.question,
                    gold,
                    &qr.predicted_answer,
                    item.question_type,
                    false,
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
            }
        } else {
            None
        };

        tracing::info!(
            question_id = %item.question_id,
            contains_answer = qr.context_contains_answer,
            recall_at_5 = format!("{:.3}", qr.recall_at_5),
            recall_items = qr.recall_items,
            recall_ms = qr.recall_latency_ms,
            "query complete",
        );

        // Record query-outcome episode (best-effort).
        let outcome = match judge_score {
            Some(s) if s >= 0.5 => "success",
            Some(_) => "failure",
            None => {
                if qr.context_contains_answer {
                    "success"
                } else {
                    "failure"
                }
            }
        };
        let state = serde_json::json!({
            "topic": item.question.clone(),
            "question": item.question.clone(),
            "question_type": item.question_type.name(),
            "question_id": item.question_id.clone(),
        });
        let params = uniko_memory::RecordEpisodeParams {
            action_type: "retrieve".into(),
            outcome: Some(outcome.into()),
            state: Some(state),
            importance: Some(judge_score.unwrap_or(0.5).clamp(0.0, 1.0)),
            ..Default::default()
        };
        if let Err(e) = uniko_memory::record_episode(&kb, &bench_agent_id, params).await {
            tracing::debug!(error = %e, "episode recording failed");
        }

        all_results.push((qr, judge_score));

        // Per-item checkpoint: write the running results so a kill
        // mid-run preserves every item completed so far. `total_items`
        // stays the full denominator for averages — partial runs see
        // an honest "X of N completed" picture.
        let partial = report::aggregate_lme(&all_results, total_items);
        if let Err(e) = report::write_lme_json(&all_results, &partial, &cli.output) {
            tracing::warn!(error = %e, "checkpoint write failed (continuing)");
        }

        // KB is dropped here, freeing memory.
    }

    let bench_elapsed = bench_start.elapsed();
    tracing::info!(
        total_secs = bench_elapsed.as_secs(),
        total_questions = all_results.len(),
        "benchmark complete",
    );

    // Aggregate and report.
    let lme_report = report::aggregate_lme(&all_results, total_items);
    report::print_lme_report(&lme_report);
    report::write_lme_json(&all_results, &lme_report, &cli.output)?;
    tracing::info!(path = %cli.output.display(), "results written");

    // Phase 1 gate check.
    if cli.phase1 {
        let pass = lme_report.overall_context_contains_rate >= 0.90;
        println!(
            "\n  Phase 1 Gate: context_contains_answer@R5 = {:.1}% (threshold: 90%)",
            lme_report.overall_context_contains_rate * 100.0
        );
        if pass {
            println!("  Result: PASS");
        } else {
            println!("  Result: FAIL — Fix P1-P3 or recall cascade before proceeding to Phase 2.");
        }
        println!();
    }

    Ok(())
}
