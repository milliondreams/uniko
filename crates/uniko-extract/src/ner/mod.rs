//! Pipeline 2 — Entity extraction.
//!
//! [`EntityExtractionStep`] implements the [`Step`](uniko_pipes::Step)
//! trait and orchestrates rule-based, code AST, ONNX NLP, and LLM
//! (stub) entity extraction followed by deduplication and graph
//! persistence.

// Rust guideline compliant

#[cfg(feature = "code-parse")]
pub mod code;
pub mod dedup;
pub mod llm;
pub mod onnx;
pub mod rules;
pub mod types;

pub use types::{EntityMatch, EntityType, ExtractionSource, RawEntity};

use async_trait::async_trait;

use uniko_pipes::step::PipelineContext;
use uniko_pipes::types::{StepErrorPolicy, StepOutcome};
use uniko_store::UnikoError;

/// Pipeline step that extracts named entities from content.
///
/// Orchestration:
/// 1. Rule-based regex on `ctx.content` (always).
/// 2. Tree-sitter AST when `content_type == "code"` (feature-gated).
/// 3. ONNX NER stub (logged skip).
/// 4. LLM enhancement stub (no-op).
/// 5. Deduplicate by canonical name.
/// 6. Upsert Entity nodes + MENTIONS edges.
/// 7. Populate `ctx.extracted_entities`.
#[derive(Debug)]
pub struct EntityExtractionStep;

#[async_trait]
impl uniko_pipes::Step for EntityExtractionStep {
    fn name(&self) -> &str {
        "entity_extraction"
    }

    fn should_run(&self, ctx: &PipelineContext) -> bool {
        ctx.node_id != 0 && !ctx.content.is_empty()
    }

    async fn execute(&self, ctx: &mut PipelineContext) -> Result<StepOutcome, UnikoError> {
        let step_start = std::time::Instant::now();
        let mut all_raw: Vec<RawEntity> = Vec::new();

        // 1. Rule-based extraction (always runs).
        let rule_ents = rules::extract_entities_rule_based(&ctx.content);
        all_raw.extend(rule_ents);
        let rules_ms = step_start.elapsed().as_millis();

        // 2. Code AST extraction (conditional on content type + feature).
        #[cfg(feature = "code-parse")]
        if ctx.content_type == "code" {
            let lang = ctx
                .metadata
                .get("language")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let code_ents = code::extract_code_entities(&ctx.content, lang);
            tracing::debug!(count = code_ents.len(), lang = %lang, "code entities");
            all_raw.extend(code_ents);
        }

        // 3. ONNX NER via multi-task NLP pipeline (per-sentence).
        let onnx_start = std::time::Instant::now();
        #[allow(unused_assignments)]
        let mut nlp_ms: u128 = 0;
        #[cfg(feature = "onnx")]
        {
            match crate::nlp::NlpPipeline::try_new(&ctx.kb).await {
                Some(pipeline) => {
                    let analyze_start = std::time::Instant::now();
                    match pipeline.analyze_sentences(&ctx.content).await {
                        Ok(nlp_results) => {
                            nlp_ms = analyze_start.elapsed().as_millis();
                            // Merge NER entities from all sentences.
                            for nlp_result in &nlp_results {
                                let onnx_ents =
                                    onnx::entities_from_nlp_result(nlp_result, &ctx.content);
                                all_raw.extend(onnx_ents);
                            }
                            // Stash per-sentence results for P3 (observations).
                            if let Ok(val) = serde_json::to_value(&nlp_results) {
                                ctx.metadata.insert("nlp_results".into(), val);
                            }
                        }
                        Err(e) => tracing::debug!(error = %e, "NLP analysis failed"),
                    }
                }
                None => tracing::debug!("NLP pipeline unavailable, using rules only"),
            }
        }
        #[cfg(not(feature = "onnx"))]
        {
            match onnx::extract_entities_onnx(&ctx.content) {
                Ok(onnx_ents) => all_raw.extend(onnx_ents),
                Err(e) => tracing::debug!(error = %e, "ONNX NER unavailable"),
            }
        }

        // 4. LLM enhancement stub.
        let llm_ents = llm::enhance_entities_llm(&ctx.content, &all_raw).await;
        all_raw.extend(llm_ents);

        let onnx_total_ms = onnx_start.elapsed().as_millis();

        if all_raw.is_empty() {
            // Export timings even when no entities found.
            ctx.metadata.insert(
                "entity_timings".into(),
                serde_json::json!({
                    "rules_ms": rules_ms as u64,
                    "nlp_ms": nlp_ms as u64,
                    "onnx_total_ms": onnx_total_ms as u64,
                    "upsert_ms": 0u64,
                    "total_ms": step_start.elapsed().as_millis() as u64,
                }),
            );
            return Ok(StepOutcome::Skipped {
                reason: "no entities found".into(),
            });
        }

        // 5. Deduplicate within this batch.
        let deduped = dedup::deduplicate_raw(all_raw);

        // 6. Upsert into graph.
        let upsert_start = std::time::Instant::now();
        let matches = dedup::upsert_entities(&ctx.kb, ctx.node_id, deduped).await?;
        let upsert_ms = upsert_start.elapsed().as_millis();

        // 7. Populate context for downstream steps (P3, P7). The
        // canonical name is included so the observation step can do
        // its substring matching against entity names without a
        // per-entity `get_node` round-trip.
        ctx.extracted_entities = matches
            .iter()
            .map(|m| (m.node_id, m.canonical_name.clone()))
            .collect();

        let total_ms = step_start.elapsed().as_millis();
        tracing::info!(
            count = matches.len(),
            rules_ms,
            nlp_ms,
            onnx_total_ms,
            upsert_ms,
            total_ms,
            "entity step",
        );

        // Export sub-timings for benchmark aggregation.
        ctx.metadata.insert(
            "entity_timings".into(),
            serde_json::json!({
                "rules_ms": rules_ms as u64,
                "nlp_ms": nlp_ms as u64,
                "onnx_total_ms": onnx_total_ms as u64,
                "upsert_ms": upsert_ms as u64,
                "total_ms": total_ms as u64,
            }),
        );

        Ok(StepOutcome::Completed)
    }

    fn error_policy(&self) -> StepErrorPolicy {
        StepErrorPolicy::Skip
    }
}
