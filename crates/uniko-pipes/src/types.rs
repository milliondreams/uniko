//! Data types for pipeline task routing.
//!
//! These types flow through channels between the pipeline system and its
//! workers.  They carry no logic — only data.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use uniko_store::NodeId;

// ── Ingest tasks ────────────────────────────────────────────────────

/// A message to ingest into the knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestMessage {
    /// Caller-provided or auto-generated UUID v7.
    pub message_id: String,
    /// Raw message text.
    pub content: String,
    /// MIME-like content type: `"text"`, `"code"`, `"tool_result"`, etc.
    pub content_type: String,
    /// Participant who sent this message.
    pub sender_id: String,
    /// Session this message belongs to.
    pub session_id: String,
    /// Recipient participant IDs.  If `None`, inferred from session
    /// participants (all except the sender).
    pub addressed_to: Option<Vec<String>>,
    /// When the message was sent.
    pub timestamp: DateTime<Utc>,
    /// Arbitrary caller metadata forwarded to pipeline steps.
    pub metadata: HashMap<String, serde_json::Value>,
}

/// An artifact (file, document, URL) to ingest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestArtifact {
    /// Caller-provided or auto-generated UUID v7.
    pub artifact_id: String,
    /// Text content (empty for binary artifacts).
    pub content: String,
    /// Artifact kind: `"file"`, `"document"`, `"url"`, `"snippet"`, etc.
    pub kind: String,
    /// Filesystem path, URL, or identifier.
    pub path: Option<String>,
    /// Arbitrary caller metadata.
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Source of PDF bytes — mirrors `uniko_extract::ingest::pdf::PdfInput`.
///
/// Mirrored here (rather than re-exported) because `uniko-pipes` is
/// upstream of `uniko-extract` in the workspace dep graph. The wire
/// types intentionally stay serializable; the runtime
/// `PdfIngestOptions::extractor` knob lives only on the extract side.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PdfInput {
    /// In-memory bytes (base64-decoded by the caller before this point).
    Bytes(Vec<u8>),
    /// Filesystem path; bytes are read at ingest time.
    Path(std::path::PathBuf),
}

/// A PDF document to ingest into the knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestPdf {
    /// Caller-provided artifact identifier.
    pub artifact_id: String,
    /// PDF byte source.
    pub input: PdfInput,
    /// Optional source path / URL recorded on `:Artifact.path`.
    pub source_path: Option<String>,
    /// Arbitrary caller metadata.
    pub metadata: HashMap<String, serde_json::Value>,
}

/// A task submitted to the ingest pipeline.
#[derive(Debug, Clone)]
pub enum IngestTask {
    /// Ingest a conversation message.
    Message(IngestMessage),
    /// Ingest an artifact (file, document, etc.).
    Artifact(IngestArtifact),
    /// Ingest a PDF document.
    Pdf(IngestPdf),
}

// ── Consolidation tasks ─────────────────────────────────────────────

/// Notification that new observations are ready for consolidation.
#[derive(Debug, Clone)]
pub struct ObservationsReady {
    /// Agent whose observations should be consolidated.
    pub agent_id: String,
    /// Number of new observations since last consolidation.
    pub observation_count: u32,
    /// Node IDs of the source observations.
    pub source_node_ids: Vec<NodeId>,
}

/// A task submitted to the consolidation pipeline.
#[derive(Debug, Clone)]
pub enum ConsolidationTask {
    /// New observations are available — may trigger consolidation.
    ObservationsReady(ObservationsReady),
    /// Force an immediate consolidation cycle for the agent.
    ForceConsolidate {
        /// Target agent.
        agent_id: String,
    },
    /// Run a consolidation cycle (internal timer trigger).
    RunCycle {
        /// Target agent.
        agent_id: String,
    },
}

// ── Step outcomes ───────────────────────────────────────────────────

/// What to do when a pipeline step fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StepErrorPolicy {
    /// Log warning, continue to next step.
    Skip,
    /// Store in dead-letter queue for later retry, continue.
    DeadLetter,
    /// Stop remaining steps for this item.
    Abort,
}

/// Outcome of a single pipeline step execution.
#[derive(Debug, Clone)]
pub enum StepOutcome {
    /// Step succeeded normally.
    Completed,
    /// Step was intentionally skipped.
    Skipped {
        /// Why the step was skipped.
        reason: String,
    },
    /// Step failed with a specific error policy.
    Failed {
        /// Error description.
        error: String,
        /// How to handle this failure.
        policy: StepErrorPolicy,
    },
}

/// Aggregated result of running a full step chain on one item.
#[derive(Debug, Clone)]
pub struct ItemResult {
    /// Node ID of the processed item.
    pub node_id: NodeId,
    /// Steps that completed successfully.
    pub steps_succeeded: Vec<String>,
    /// Steps that failed: `(step_name, error_message)`.
    pub steps_failed: Vec<(String, String)>,
    /// Steps that were skipped.
    pub steps_skipped: Vec<String>,
}

impl ItemResult {
    /// Create an empty result for a given node.
    pub fn new(node_id: NodeId) -> Self {
        Self {
            node_id,
            steps_succeeded: Vec::new(),
            steps_failed: Vec::new(),
            steps_skipped: Vec::new(),
        }
    }
}
