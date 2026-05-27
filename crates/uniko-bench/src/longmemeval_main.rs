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
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use futures::stream::{self, StreamExt};

// mimalloc as global allocator — measured ~3x throughput on uni-db's
// concurrent_mutations benchmark (uni-db commit 65399a2b). Allocation
// is on the hot path for every Cypher parse/plan/execute cycle and
// every Lance L0 buffer write, so the global allocator choice
// materially shifts ingest throughput. M-MIMALLOC-APPS.
#[global_allocator]
static GLOBAL: uni_db::MiMalloc = uni_db::MiMalloc;

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

    /// Maximum number of LME items (questions) processed concurrently.
    /// Each item has its own isolated KB so there's no shared-state
    /// contention. Memory scales linearly: each in-flight item holds
    /// its KB + loaded NLP models. Set to 1 for fully sequential runs.
    #[arg(long, default_value = "8")]
    question_concurrency: usize,

    /// Maximum number of chat sessions ingested concurrently within a
    /// single LME item. All workers write to the per-item KB, so the
    /// win is capped by uni-db's concurrent-write throughput. Set to 1
    /// to ingest sessions sequentially (matches pre-parallel behavior).
    #[arg(long, default_value = "8")]
    session_concurrency: usize,

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
    /// is registered against `remote/openai` with this base URL. Implies
    /// `--llm-provider remote/openai`.
    #[arg(long)]
    llm_base_url: Option<String>,

    /// Base URL of an OpenAI-compatible server for the judge LLM.
    /// Defaults to OpenAI's public API.
    #[arg(long)]
    judge_base_url: Option<String>,

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

    /// Disable the reranker. Reranker is **on by default** — pass
    /// `--no-reranker` to fall back to pure recall ranking. Matches
    /// the LoCoMo bench so the two harnesses tune to the same defaults.
    #[arg(long)]
    no_reranker: bool,

    /// HuggingFace reranker model id. xervo 0.11 auto-detects whether
    /// the ONNX graph expects `token_type_ids`, so XLM-R-based models
    /// (e.g. `BAAI/bge-reranker-base`) work alongside BERT-based ones
    /// (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`). For
    /// `--reranker-style generative`, use a decoder-LM export such as
    /// `onnx-community/Qwen3-Reranker-0.6B-ONNX`.
    #[arg(long, default_value = "cross-encoder/ms-marco-MiniLM-L-6-v2")]
    reranker_model: String,

    /// Reranker code path. `cross-encoder` (default) for BERT-family
    /// cross-encoders that emit a relevance logit; `generative` for
    /// decoder-LM rerankers that score yes/no via next-token logits.
    #[arg(long, default_value = "cross-encoder")]
    reranker_style: String,

    /// Top-N RRF candidates to send through the reranker. Must be
    /// `>= recall_limit` when reranker is enabled.
    #[arg(long, default_value = "50")]
    reranker_top_n: usize,

    /// Maximum items in the recall bundle (overrides `recall_limit`
    /// in the embedding config).
    #[arg(long)]
    recall_limit: Option<usize>,

    /// Phase 1 (Compact) contribution strategy:
    /// - `boost` (default) — Facts/Observations influence chunk ranking
    ///   via a session-level boost; bundle stays 100% Chunks.
    /// - `merge` — cap=3 interleave Facts by score into the Phase 3 bundle.
    /// - `off` — skip Phase 1 entirely.
    #[arg(long, default_value = "boost")]
    phase1_strategy: String,

    /// Multiplicative weight applied to Fact scores when computing the
    /// session-chunk boost under `--phase1-strategy boost`. α=0.6 is
    /// the validated default.
    #[arg(long, default_value = "0.6")]
    phase1_boost_alpha: f64,

    /// Disable the graph spreading-activation channel in Phase 2 recall.
    /// Default: on. Channel only fires when the query has at least one
    /// resolvable entity seed.
    #[arg(long)]
    no_phase2_graph: bool,

    /// Disable the temporal-interval channel in Phase 2 recall.
    /// Default: on. Channel only fires when the query has a parsed
    /// temporal phrase.
    #[arg(long)]
    no_phase2_temporal: bool,

    /// When set, P4 Consolidation refines each Observation's
    /// `(subject, predicate, object)` triple via the LLM at this
    /// alias before grouping into Facts. Off by default — keeps the
    /// SRL/DEP triple P3 produces.
    #[arg(long)]
    extract_triples_llm_alias: Option<String>,

    /// Disable cosine-similarity clustering of object surface forms in
    /// P4 Consolidation. Default: clustering on.
    #[arg(long)]
    no_consolidation_cluster: bool,

    /// Disable the `"[Month Year] "` date prefix prepended to Fact
    /// embed text in P4 Consolidation. Default: date-augment on.
    #[arg(long)]
    no_date_augment_embedding: bool,

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

    // Determine LLM mode.  Eagerly clone alias strings here so cli can
    // be moved into an Arc later without keeping live borrows.
    let retrieval_only = cli.phase1 || cli.llm_alias.is_none();
    let llm_alias_owned: Option<String> = if retrieval_only {
        None
    } else {
        cli.llm_alias.clone()
    };
    let judge_alias_owned: Option<String> = if cli.no_judge || retrieval_only {
        None
    } else {
        cli.judge_alias.clone().or_else(|| cli.llm_alias.clone())
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
    // Reranker is default-on via `RerankerConfig::default()`. Apply
    // CLI overrides (model/style/top_n) regardless, so the user can
    // swap models without explicitly enabling. `--no-reranker` disables.
    config.reranker.enabled = !cli.no_reranker;
    if config.reranker.enabled {
        config.reranker.model_id = cli.reranker_model.clone();
        config.reranker.style = cli.reranker_style.clone();
        config.reranker.top_n = cli.reranker_top_n;
    }
    if let Some(limit) = cli.recall_limit {
        config.recall_limit = limit;
        if config.reranker.enabled && config.reranker.top_n < limit {
            config.reranker.top_n = limit;
        }
    }
    config.phase1_strategy = cli.phase1_strategy.clone();
    config.phase1_boost_alpha = cli.phase1_boost_alpha;
    config.phase2_graph_enabled = !cli.no_phase2_graph;
    config.phase2_temporal_enabled = !cli.no_phase2_temporal;
    config.consolidation_cluster_objects = !cli.no_consolidation_cluster;
    config.consolidation_date_augment_embedding = !cli.no_date_augment_embedding;
    if !cli.variants.trim().is_empty() {
        config.query_variants = cli
            .variants
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
    }

    // Build LLM catalog. `--llm-base-url` implies `remote/openai`
    // unless `--llm-provider` is explicitly set.
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

    // Run benchmark.  Items are processed concurrently up to
    // `--question-concurrency`; each item owns its own KB so there's
    // no cross-item write contention.  Results land in a shared Vec
    // (order is the completion order, not the input order — the
    // aggregation routines key on question_id, not position).
    type ResultEntry = (query::LmeQueryResult, Option<f64>);
    let all_results: Arc<tokio::sync::Mutex<Vec<ResultEntry>>> =
        Arc::new(tokio::sync::Mutex::new(Vec::with_capacity(items.len())));
    let bench_start = Instant::now();
    let total_items = items.len();
    let question_concurrency = cli.question_concurrency.max(1);

    // Build the ONNX/Xervo runtime **once**, before the per-item
    // loop, and share its `Arc` across every concurrent KB. Without
    // this, q≥3 on an 8 GB GPU OOMs because each KB loads its own
    // model session (~3.7 GB each). See
    // `uniko_store::KnowledgeBase::build_shared_runtime`.
    let shared_runtime = uniko_store::KnowledgeBase::build_shared_runtime(&config, &extra_catalog)
        .await
        .context("building shared Xervo runtime")?;
    tracing::info!(
        question_concurrency,
        "shared Xervo runtime built; all KBs in this process will reuse it"
    );

    // Captures shared by every per-item worker.  All Arc-cloneable or
    // Copy/cheap-Clone.  llm_alias_owned/judge_alias_owned were already
    // built upfront (line ~297) so cli can move into Arc freely.
    let cli = Arc::new(cli);
    let config = Arc::new(config);

    stream::iter(items.into_iter().enumerate())
        .for_each_concurrent(question_concurrency, |(item_idx, item)| {
            let cli = cli.clone();
            let config = config.clone();
            let runtime = shared_runtime.clone();
            let all_results = all_results.clone();
            let llm_alias_owned = llm_alias_owned.clone();
            let judge_alias_owned = judge_alias_owned.clone();
            async move {
                tracing::info!(
                    "[{}/{}] Processing {} (type={}, sessions={})",
                    item_idx + 1,
                    total_items,
                    item.question_id,
                    item.question_type.name(),
                    item.haystack_sessions.len(),
                );

                let kb_dir = cli.ingest_dir.join(&item.question_id);
                let (kb, evidence_map) = if cli.reuse && kb_dir.exists() {
                    tracing::info!(path = %kb_dir.display(), "reusing existing KB");
                    match uniko_bench::open_kb_with_runtime(
                        &kb_dir,
                        (*config).clone(),
                        runtime.clone(),
                    )
                    .await
                    {
                        Ok(kb) => {
                            let evidence_map = ingest::EvidenceMap {
                                answer_session_ids: item
                                    .answer_session_ids
                                    .iter()
                                    .cloned()
                                    .collect(),
                                answer_message_ids: Vec::new(),
                                session_to_messages: std::collections::HashMap::new(),
                            };
                            (kb, evidence_map)
                        }
                        Err(e) => {
                            tracing::warn!(question_id = %item.question_id, error = %e, "open_kb failed; skipping item");
                            return;
                        }
                    }
                } else {
                    let ingest_start = Instant::now();
                    match ingest::ingest_item(
                        &item,
                        &kb_dir,
                        (*config).clone(),
                        runtime.clone(),
                        cli.session_concurrency,
                    )
                    .await
                    {
                        Ok(result) => {
                            tracing::info!(
                                question_id = %item.question_id,
                                elapsed_ms = ingest_start.elapsed().as_millis(),
                                "ingestion complete"
                            );
                            result
                        }
                        Err(e) => {
                            tracing::warn!(question_id = %item.question_id, error = %e, "ingest failed; skipping item");
                            return;
                        }
                    }
                };

                // P4 Consolidation: see longmemeval_main rationale.
                let cycle_start = Instant::now();
                let triple_source = match cli.extract_triples_llm_alias.as_deref() {
                    Some(alias) => uniko_memory::consolidation::TripleSource::Llm {
                        alias: alias.to_string(),
                    },
                    None => uniko_memory::consolidation::TripleSource::SrlDep,
                };
                match uniko_memory::consolidation::run_cycle_with(
                    &kb,
                    &item.question_id,
                    Some(10_000),
                    &triple_source,
                )
                .await
                {
                    Ok(stats) => tracing::info!(
                        question_id = %item.question_id,
                        processed = stats.observations_processed,
                        facts_created = stats.facts_created,
                        facts_reinforced = stats.facts_reinforced,
                        duration_ms = cycle_start.elapsed().as_millis(),
                        "consolidation cycle complete",
                    ),
                    Err(e) => tracing::warn!(
                        question_id = %item.question_id,
                        error = %e,
                        "consolidation cycle failed (continuing without Facts)",
                    ),
                }

                // P5 + P6 — explicit cortex sweep per question so the
                // bench always exercises procedure + topic surfaces.
                run_cortex_sweep(&kb, &item.question_id).await;

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

                let gold = data::gold_answer(&item);
                let qr = match query::run_lme_query(
                    &kb,
                    &item.question_id,
                    &item.question,
                    item.question_type,
                    gold,
                    &evidence_map,
                    cli.token_budget,
                    llm_alias_owned.as_deref(),
                )
                .await
                {
                    Ok(qr) => qr,
                    Err(e) => {
                        tracing::warn!(question_id = %item.question_id, error = %e, "query failed; skipping item");
                        return;
                    }
                };

                let judge_score = if let Some(alias) = judge_alias_owned.as_deref() {
                    let is_abstention = LmeQuestionType::is_abstention(&item.question_id);
                    if is_abstention {
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
                if let Err(e) =
                    uniko_memory::record_episode(&kb, &bench_agent_id, params).await
                {
                    tracing::debug!(error = %e, "episode recording failed");
                }

                // Push result + checkpoint write under the lock.  The
                // lock window is short (Vec push + JSON serialize) and
                // serialized writes preserve a consistent on-disk
                // snapshot after each completed item.
                let mut results = all_results.lock().await;
                results.push((qr, judge_score));
                let partial = report::aggregate_lme(&results, total_items);
                if let Err(e) = report::write_lme_json(&results, &partial, &cli.output) {
                    tracing::warn!(error = %e, "checkpoint write failed (continuing)");
                }
                // KB drops at end of scope.
            }
        })
        .await;

    let all_results = Arc::try_unwrap(all_results)
        .map_err(|_| anyhow::anyhow!("all_results still has outstanding refs"))?
        .into_inner();

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

/// Run P5 (procedure promotion) + P6 (topic detection) for one
/// bench item.  Mirrors the live consolidation worker's cortex
/// sweep but unconditional, so every LME run exercises the full
/// downstream surface regardless of consolidation cadence.
///
/// Failures are logged and dropped — cortex is downstream of
/// consolidation, never a hard requirement for the eval.
async fn run_cortex_sweep(kb: &Arc<uniko_store::KnowledgeBase>, question_id: &str) {
    use std::time::Instant;
    let proc_start = Instant::now();
    match uniko_cortex::promote_procedures_once(
        kb,
        question_id,
        uniko_cortex::LifecycleConfig::default(),
    )
    .await
    {
        Ok(r) => tracing::info!(
            question_id,
            created = r.created,
            reinforced = r.reinforced,
            promoted = r.promoted,
            duration_ms = proc_start.elapsed().as_millis(),
            "P5 procedure sweep complete",
        ),
        Err(e) => tracing::warn!(question_id, error = %e, "P5 procedure sweep failed"),
    }

    let topic_start = Instant::now();
    match uniko_cortex::detect_topics_once(kb, uniko_cortex::TopicConfig::default()).await {
        Ok(r) => tracing::info!(
            question_id,
            created = r.created,
            updated = r.updated,
            entities_assigned = r.entities_assigned,
            duration_ms = topic_start.elapsed().as_millis(),
            "P6 topic sweep complete",
        ),
        Err(e) => tracing::warn!(question_id, error = %e, "P6 topic sweep failed"),
    }
}
