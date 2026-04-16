//! The `Step` trait — cornerstone of all pipeline processing.
//!
//! Every pipeline step (NER, observation extraction, embedding, etc.)
//! implements [`Step`].  Steps are composed into chains and executed
//! per-item with error isolation.

// Rust guideline compliant

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use uniko_store::{KnowledgeBase, NodeId, UnikoError};

use crate::circuit_breaker::CircuitBreaker;
use crate::types::{StepErrorPolicy, StepOutcome};

/// A single processing step in a pipeline.
///
/// Steps are stateless and reusable across items.  Each item gets its
/// own [`PipelineContext`] for per-item isolation.
///
/// # Implementors
///
/// Return [`StepOutcome::Completed`] on success, or
/// [`StepOutcome::Failed`] with the appropriate [`StepErrorPolicy`].
#[async_trait]
pub trait Step: Send + Sync {
    /// Human-readable name for logging and metrics.
    fn name(&self) -> &str;

    /// Whether this step should execute for the given context.
    ///
    /// Returns `false` to skip (e.g. code NER skips non-code content).
    fn should_run(&self, ctx: &PipelineContext) -> bool;

    /// Execute the step, mutating context with results.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] on unrecoverable failure.  For expected
    /// failures, prefer returning [`StepOutcome::Failed`] with the
    /// appropriate policy instead.
    async fn execute(&self, ctx: &mut PipelineContext) -> Result<StepOutcome, UnikoError>;

    /// How to handle a failure from this step.
    fn error_policy(&self) -> StepErrorPolicy;
}

/// Mutable context passed through the step chain for a single item.
///
/// Each item gets its own context — no shared mutable state between
/// items.
#[derive(Debug)]
pub struct PipelineContext {
    /// Internal node ID of the item being processed.
    pub node_id: NodeId,
    /// Raw content of the item.
    pub content: String,
    /// Content type (e.g. `"text"`, `"code"`, `"image"`).
    pub content_type: String,
    /// Per-item cancellation token (child of worker token).
    pub cancel: CancellationToken,
    /// Graph storage access.
    pub kb: Arc<KnowledgeBase>,
    /// LLM circuit breaker — check before making LLM calls.
    pub llm_breaker: Arc<CircuitBreaker>,
    /// Entity node IDs extracted so far (populated by NER step).
    pub extracted_entities: Vec<NodeId>,
    /// Observation node IDs extracted so far.
    pub extracted_observations: Vec<NodeId>,
    /// Arbitrary step-to-step metadata.
    pub metadata: HashMap<String, serde_json::Value>,
}

impl PipelineContext {
    /// Create a new context for processing one item.
    pub fn new(
        node_id: NodeId,
        content: String,
        content_type: String,
        cancel: CancellationToken,
        kb: Arc<KnowledgeBase>,
        llm_breaker: Arc<CircuitBreaker>,
    ) -> Self {
        Self {
            node_id,
            content,
            content_type,
            cancel,
            kb,
            llm_breaker,
            extracted_entities: Vec::new(),
            extracted_observations: Vec::new(),
            metadata: HashMap::new(),
        }
    }
}
