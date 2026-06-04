//! NLP model variant comparison: extraction quality + performance.
//!
//! Loads messages from given LME questions, runs the NLP cascade
//! through each model variant (xsmall/small/base), measures per-call
//! latency, and counts extracted entities/observations.
//!
//! Does NOT write to the graph — bootstraps a scratch KB only so
//! `NlpPipeline::try_new` can resolve the `nlp/default` catalog alias.
//! Each variant runs against a fresh tmp KB so model loads are clean.

// mimalloc as global allocator — measured ~3x throughput on uni-db's
// concurrent_mutations benchmark (uni-db commit 65399a2b).
#[global_allocator]
static GLOBAL: uni_db::MiMalloc = uni_db::MiMalloc;

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use serde::Serialize;

use uni_db::ModelAliasSpec;
use uniko_bench::longmemeval::data;
use uniko_extract::ingest::context::SentenceContext;
use uniko_extract::nlp::NlpPipeline;
use uniko_extract::nlp::assets::label_maps;
use uniko_extract::nlp::decode::extract_dep_observations;
use uniko_store::KnowledgeBase;
use uniko_store::config::UnikoConfig;

#[derive(Parser)]
#[command(
    name = "nlp-compare",
    about = "Compare NLP model variants on LME messages"
)]
struct Cli {
    /// Path to LongMemEval JSON file.
    #[arg(long)]
    data: PathBuf,

    /// Comma-separated question IDs to source messages from.
    #[arg(long)]
    questions: String,

    /// Comma-separated variant suffixes (`xsmall`, `small`, `base`).
    #[arg(long, default_value = "xsmall,small,base")]
    variants: String,

    /// ONNX artifact file name within each variant's repo. Use
    /// `onnx/cascade.onnx` for FP32 (preferred on GPU) or
    /// `onnx/cascade-int8.onnx` for INT8.
    #[arg(long, default_value = "onnx/cascade.onnx")]
    artifact: String,

    /// Max batch size for the NLP model (per ONNX call).
    #[arg(long, default_value = "16")]
    max_batch_size: u32,

    /// Intra-op ONNX threads per worker.
    #[arg(long, default_value = "4")]
    intra_op_num_threads: u32,

    /// Scratch directory for transient per-variant KBs (deleted+recreated).
    #[arg(long, default_value = "/tmp/nlp_compare_scratch")]
    scratch_dir: PathBuf,

    /// Output JSON path.
    #[arg(long, default_value = "/tmp/nlp_compare_results.json")]
    output: PathBuf,

    /// Cap message count (useful for quick tests). Default = all.
    #[arg(long)]
    max_messages: Option<usize>,
}

#[derive(Serialize)]
struct VariantStats {
    variant: String,
    model_id: String,
    artifact: String,
    messages_processed: usize,
    sentences_total: usize,
    entities_total: usize,
    observations_total: usize,
    warmup_ms: u64,
    steady_state_ms: u64,
    per_message_avg_ms: f64,
    per_message_p50_ms: u64,
    per_message_p95_ms: u64,
    per_message_p99_ms: u64,
    cls_label_distribution: serde_json::Value,
    entity_type_distribution: serde_json::Value,
}

#[derive(Serialize)]
struct Comparison {
    questions: Vec<String>,
    total_messages: usize,
    artifact: String,
    variants: Vec<VariantStats>,
}

fn percentile(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                "nlp_compare=info,uniko_bench=info,uniko_extract=warn,uniko_store=warn,ort::ep=warn"
                    .parse()
                    .unwrap()
            }),
        )
        .init();

    let cli = Cli::parse();

    tracing::info!(path = %cli.data.display(), "loading LongMemEval dataset");
    let mut items = data::load_longmemeval(&cli.data)?;
    let want_ids: Vec<String> = cli
        .questions
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();
    items.retain(|i| want_ids.contains(&i.question_id));
    if items.is_empty() {
        anyhow::bail!("no questions matched ids {:?}", want_ids);
    }
    tracing::info!(
        questions = items.len(),
        ids = ?items.iter().map(|i| i.question_id.as_str()).collect::<Vec<_>>(),
        "questions selected",
    );

    // Collect all messages from selected questions (preserve source-order).
    let mut messages: Vec<(String, String)> = Vec::new();
    for item in &items {
        for session_turns in &item.haystack_sessions {
            for turn in session_turns {
                messages.push((turn.role.clone(), turn.content.clone()));
            }
        }
    }
    if let Some(cap) = cli.max_messages {
        messages.truncate(cap);
    }
    tracing::info!(messages = messages.len(), "collected messages");

    let variants: Vec<String> = cli
        .variants
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();

    let labels = label_maps();
    let mut comparison = Comparison {
        questions: want_ids,
        total_messages: messages.len(),
        artifact: cli.artifact.clone(),
        variants: Vec::new(),
    };

    for variant in &variants {
        let model_id = format!("dragonscale-ai/kniv-deberta-nlp-base-en-{variant}");
        tracing::info!(variant = %variant, model_id = %model_id, "starting variant");

        // Per-variant scratch KB — fresh dir for clean state.
        let scratch = cli.scratch_dir.join(variant);
        if scratch.exists() {
            std::fs::remove_dir_all(&scratch).context("remove scratch")?;
        }
        std::fs::create_dir_all(&scratch).context("create scratch")?;

        // Write a full per-variant catalog file. `config.catalog_path`
        // *replaces* the auto-built catalog entirely, so we avoid
        // duplicates with the default `nlp/default` xsmall entry.
        let catalog_file = scratch.join("catalog.json");
        let catalog_json = serde_json::json!([
            {
                "alias": "embed/default",
                "task": "embed",
                "provider_id": "local/onnx",
                "model_id": "BGESmallENV15",
                "warmup": "lazy",
                "required": false
            },
            {
                "alias": "nlp/default",
                "task": "raw",
                "provider_id": "local/onnx",
                "model_id": model_id,
                "warmup": "lazy",
                "required": false,
                "options": {
                    "artifact": cli.artifact,
                    "max_batch_size": cli.max_batch_size,
                    "graph_optimization_level": "all",
                    "intra_op_num_threads": cli.intra_op_num_threads,
                }
            }
        ]);
        std::fs::write(&catalog_file, serde_json::to_string_pretty(&catalog_json)?)
            .context("write catalog")?;

        let mut config = UnikoConfig::default();
        config.embedding = uniko_store::config::EmbeddingConfig::bge_small_en_v15();
        config.catalog_path = Some(catalog_file);
        let kb = KnowledgeBase::open_with_xervo(&scratch, config, Vec::<ModelAliasSpec>::new())
            .await
            .context("open scratch KB")?;
        let pipeline = NlpPipeline::try_new(&kb)
            .await
            .context("NlpPipeline::try_new returned None — check catalog and ort/cuda setup")?;

        // Stats accumulators.
        let mut latencies_ms: Vec<u64> = Vec::with_capacity(messages.len());
        let mut sentences_total = 0usize;
        let mut entities_total = 0usize;
        let mut observations_total = 0usize;
        let mut cls_counts: std::collections::HashMap<String, u64> =
            std::collections::HashMap::new();
        let mut ent_counts: std::collections::HashMap<String, u64> =
            std::collections::HashMap::new();
        let mut warmup_ms = 0u64;

        for (i, (speaker, content)) in messages.iter().enumerate() {
            let t0 = Instant::now();
            let results = match pipeline.analyze_sentences(content).await {
                Ok(r) => r,
                Err(e) => {
                    tracing::warn!(idx = i, error = %e, "analyze_sentences failed");
                    continue;
                }
            };
            let elapsed = t0.elapsed().as_millis() as u64;
            if i == 0 {
                warmup_ms = elapsed;
            }
            latencies_ms.push(elapsed);

            let mut sent_ctx = SentenceContext::default();
            for nlp_result in &results {
                sentences_total += 1;
                entities_total += nlp_result.entities.len();
                for span in &nlp_result.entities {
                    *ent_counts
                        .entry(format!("{:?}", span.entity_type))
                        .or_insert(0) += 1;
                }
                let cls_label = labels
                    .cls_labels
                    .get(nlp_result.cls_index)
                    .cloned()
                    .unwrap_or_else(|| format!("UNK{}", nlp_result.cls_index));
                *cls_counts.entry(cls_label).or_insert(0) += 1;

                let obs = extract_dep_observations(
                    &nlp_result.words,
                    &nlp_result.pos_indices,
                    &nlp_result.dep_arcs,
                    &labels.pos_labels,
                    speaker,
                    &mut sent_ctx,
                );
                observations_total += obs.len();
            }

            if i.is_multiple_of(50) && i > 0 {
                tracing::info!(
                    variant = %variant,
                    processed = i,
                    of = messages.len(),
                    entities_so_far = entities_total,
                    observations_so_far = observations_total,
                    "progress",
                );
            }
        }

        let mut sorted = latencies_ms.clone();
        sorted.sort_unstable();
        let total_ms: u64 = latencies_ms.iter().sum();
        let steady_ms = total_ms.saturating_sub(warmup_ms);
        let n = latencies_ms.len().max(1);

        let stats = VariantStats {
            variant: variant.clone(),
            model_id: model_id.clone(),
            artifact: cli.artifact.clone(),
            messages_processed: latencies_ms.len(),
            sentences_total,
            entities_total,
            observations_total,
            warmup_ms,
            steady_state_ms: steady_ms,
            per_message_avg_ms: total_ms as f64 / n as f64,
            per_message_p50_ms: percentile(&sorted, 0.50),
            per_message_p95_ms: percentile(&sorted, 0.95),
            per_message_p99_ms: percentile(&sorted, 0.99),
            cls_label_distribution: serde_json::to_value(&cls_counts).unwrap(),
            entity_type_distribution: serde_json::to_value(&ent_counts).unwrap(),
        };

        tracing::info!(
            variant = %stats.variant,
            messages = stats.messages_processed,
            sentences = stats.sentences_total,
            entities = stats.entities_total,
            observations = stats.observations_total,
            total_ms = total_ms,
            warmup_ms = stats.warmup_ms,
            avg_ms_per_msg = format!("{:.1}", stats.per_message_avg_ms),
            p50_ms = stats.per_message_p50_ms,
            p95_ms = stats.per_message_p95_ms,
            "variant complete",
        );

        comparison.variants.push(stats);

        // Incremental write after each variant — if a later variant
        // fails (download/OOM/crash), we still have prior variants' data.
        match serde_json::to_string_pretty(&comparison) {
            Ok(json) => {
                if let Err(e) = std::fs::write(&cli.output, &json) {
                    tracing::warn!(error = %e, path = %cli.output.display(), "checkpoint write failed");
                } else {
                    tracing::info!(
                        path = %cli.output.display(),
                        variants_done = comparison.variants.len(),
                        "checkpoint written",
                    );
                }
            }
            Err(e) => tracing::warn!(error = %e, "checkpoint serialize failed"),
        }

        drop(pipeline);
        // KB drops at end of scope; CUDA context torn down before next variant.
        drop(kb);
    }

    let json = serde_json::to_string_pretty(&comparison)?;
    std::fs::write(&cli.output, &json).context("write output")?;
    tracing::info!(path = %cli.output.display(), "results written");

    // Concise stdout summary.
    println!("\n=== NLP variant comparison ===");
    println!(
        "questions={:?}  messages={}  artifact={}",
        comparison.questions, comparison.total_messages, comparison.artifact
    );
    println!();
    println!(
        "{:>8} {:>10} {:>10} {:>12} {:>10} {:>8} {:>8} {:>8}",
        "variant", "msgs", "sentences", "entities", "obs", "avg_ms", "p50", "p95"
    );
    for v in &comparison.variants {
        println!(
            "{:>8} {:>10} {:>10} {:>12} {:>10} {:>8.1} {:>8} {:>8}",
            v.variant,
            v.messages_processed,
            v.sentences_total,
            v.entities_total,
            v.observations_total,
            v.per_message_avg_ms,
            v.per_message_p50_ms,
            v.per_message_p95_ms,
        );
    }

    Ok(())
}
