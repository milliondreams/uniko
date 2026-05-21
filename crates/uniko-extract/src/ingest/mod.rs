//! Pipeline 1 — Ingest for Messages and Artifacts.
//!
//! [`IngestStep`] implements the [`Step`](uniko_pipes::Step) trait and
//! dispatches to [`message::ingest_message`] or
//! [`artifact::ingest_artifact`] based on the `ingest_type` metadata
//! key in the [`PipelineContext`](uniko_pipes::PipelineContext).

// Rust guideline compliant

pub mod artifact;
pub mod atomic;
pub mod chunking;
pub mod context;
pub mod message;
pub mod overflow;
pub mod session;
pub mod session_chunk;

pub use artifact::ArtifactIngestResult;
pub use atomic::{AtomicIngestResult, AtomicTimings, ingest_message_atomic};
pub use chunking::{ChunkConfig, ChunkData, Chunker, count_tokens, select_chunker};
pub use message::MessageIngestResult;

use async_trait::async_trait;

use uniko_pipes::step::PipelineContext;
use uniko_pipes::types::{IngestMessage, StepErrorPolicy, StepOutcome};
use uniko_store::UnikoError;

/// Pipeline step that ingests messages and artifacts into the graph.
///
/// Dispatch is based on `ctx.metadata["ingest_type"]`:
/// - `"message"`: deserializes [`IngestMessage`] from `"ingest_payload"`
///   and calls [`message::ingest_message`].
/// - `"artifact"`: deserializes [`IngestArtifact`](uniko_pipes::IngestArtifact)
///   and calls [`artifact::ingest_artifact`].
#[derive(Debug)]
pub struct IngestStep;

#[async_trait]
impl uniko_pipes::Step for IngestStep {
    fn name(&self) -> &str {
        "ingest"
    }

    fn should_run(&self, _ctx: &PipelineContext) -> bool {
        true
    }

    async fn execute(&self, ctx: &mut PipelineContext) -> Result<StepOutcome, UnikoError> {
        let ingest_type = ctx
            .metadata
            .get("ingest_type")
            .and_then(|v| v.as_str())
            .unwrap_or("message");

        match ingest_type {
            "message" => {
                let msg = deserialize_message(&ctx.metadata)?;
                // Use SessionContext from metadata if available, else create a fresh one.
                let mut session_ctx = ctx
                    .metadata
                    .get("session_context")
                    .and_then(|v| serde_json::from_value::<context::SessionContext>(v.clone()).ok())
                    .unwrap_or_else(|| context::SessionContext::new(msg.session_id.clone(), 0));
                let result = message::ingest_message(&ctx.kb, &msg, &mut session_ctx).await?;
                ctx.node_id = result.message_node_id;
                // Forward chunk IDs to downstream steps (embedding).
                ctx.metadata.insert(
                    "chunk_node_ids".to_string(),
                    serde_json::to_value(&result.chunk_node_ids).unwrap_or_default(),
                );
                Ok(StepOutcome::Completed)
            }
            "artifact" => {
                let art = deserialize_artifact(&ctx.metadata)?;
                let result = artifact::ingest_artifact(&ctx.kb, &art).await?;
                ctx.node_id = result.artifact_node_id;
                if result.was_deduplicated {
                    return Ok(StepOutcome::Skipped {
                        reason: "artifact deduplicated by content hash".into(),
                    });
                }
                ctx.metadata.insert(
                    "chunk_node_ids".to_string(),
                    serde_json::to_value(&result.chunk_node_ids).unwrap_or_default(),
                );
                Ok(StepOutcome::Completed)
            }
            other => Ok(StepOutcome::Skipped {
                reason: format!("unknown ingest type: {other}"),
            }),
        }
    }

    fn error_policy(&self) -> StepErrorPolicy {
        // Ingest failure is critical — dead-letter for later retry.
        StepErrorPolicy::DeadLetter
    }
}

/// Deserialize an [`IngestMessage`] from pipeline metadata.
fn deserialize_message(
    metadata: &std::collections::HashMap<String, serde_json::Value>,
) -> Result<IngestMessage, UnikoError> {
    let payload = metadata
        .get("ingest_payload")
        .ok_or_else(|| UnikoError::Pipeline("missing ingest_payload in metadata".into()))?;
    serde_json::from_value(payload.clone())
        .map_err(|e| UnikoError::Pipeline(format!("invalid IngestMessage: {e}")))
}

/// Deserialize an [`IngestArtifact`] from pipeline metadata.
fn deserialize_artifact(
    metadata: &std::collections::HashMap<String, serde_json::Value>,
) -> Result<uniko_pipes::IngestArtifact, UnikoError> {
    let payload = metadata
        .get("ingest_payload")
        .ok_or_else(|| UnikoError::Pipeline("missing ingest_payload in metadata".into()))?;
    serde_json::from_value(payload.clone())
        .map_err(|e| UnikoError::Pipeline(format!("invalid IngestArtifact: {e}")))
}
