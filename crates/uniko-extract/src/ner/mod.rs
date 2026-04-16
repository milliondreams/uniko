//! Pipeline 2 — Entity extraction.
//!
//! [`EntityExtractionStep`] implements the [`Step`](uniko_pipes::Step)
//! trait and orchestrates rule-based, code AST, ONNX (stub), and LLM
//! (stub) entity extraction followed by deduplication and graph
//! persistence.

// Rust guideline compliant

pub mod types;
pub mod rules;
#[cfg(feature = "code-parse")]
pub mod code;
pub mod dedup;
pub mod onnx;
pub mod llm;

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
        let mut all_raw: Vec<RawEntity> = Vec::new();

        // 1. Rule-based extraction (always runs).
        let rule_ents = rules::extract_entities_rule_based(&ctx.content);
        tracing::debug!(count = rule_ents.len(), "rule-based entities");
        all_raw.extend(rule_ents);

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

        // 3. ONNX NER stub.
        match onnx::extract_entities_onnx(&ctx.content) {
            Ok(onnx_ents) => all_raw.extend(onnx_ents),
            Err(e) => tracing::debug!(error = %e, "ONNX NER unavailable"),
        }

        // 4. LLM enhancement stub.
        let llm_ents = llm::enhance_entities_llm(&ctx.content, &all_raw).await;
        all_raw.extend(llm_ents);

        if all_raw.is_empty() {
            return Ok(StepOutcome::Skipped {
                reason: "no entities found".into(),
            });
        }

        // 5. Deduplicate within this batch.
        let deduped = dedup::deduplicate_raw(all_raw);
        tracing::debug!(unique = deduped.len(), "entities deduplicated");

        // 6. Upsert into graph.
        let matches = dedup::upsert_entities(&ctx.kb, ctx.node_id, deduped).await?;

        // 7. Populate context for downstream steps (P3, P7).
        ctx.extracted_entities = matches.iter().map(|m| m.node_id).collect();

        tracing::info!(
            count = matches.len(),
            new = matches.iter().filter(|m| !m.was_existing).count(),
            existing = matches.iter().filter(|m| m.was_existing).count(),
            "entity extraction completed",
        );

        Ok(StepOutcome::Completed)
    }

    fn error_policy(&self) -> StepErrorPolicy {
        StepErrorPolicy::Skip
    }
}
