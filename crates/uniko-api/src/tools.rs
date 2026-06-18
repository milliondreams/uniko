//! Agent-facing tools.
//!
//! These re-exports surface the subjective-state agent tools defined
//! in `uniko-memory` and downstream crates.  Pipelines handle what can
//! be inferred from messages; tools handle what only the agent can
//! decide to record.
//!
//! Engine internals are intentionally **not** re-exported here. These
//! must not compile:
//!
//! ```compile_fail
//! let _: uniko_api::tools::KnowledgeBase;
//! ```
//! ```compile_fail
//! let _: uniko_api::tools::PipelineSystem;
//! ```
//! ```compile_fail
//! let _: uniko_api::tools::IngestMessage;
//! ```
//!
//! Nor the `&KnowledgeBase`-taking free functions — these are `Agent` /
//! `Session` methods instead:
//!
//! ```compile_fail
//! let _ = uniko_api::tools::add_rule;
//! ```
//! ```compile_fail
//! let _ = uniko_api::tools::generate_session_summary;
//! ```
//! ```compile_fail
//! let _ = uniko_api::tools::create_goal;
//! ```
//! ```compile_fail
//! let _ = uniko_api::tools::working_memory;
//! ```

// Engine internals (`KnowledgeBase`, `PipelineSystem`, `IngestMessage`,
// uni-db types) are deliberately NOT re-exported — see the `compile_fail`
// guards in this module's docs. Subjective-state operations are surfaced
// as `Agent` / `Session` methods rather than as free functions taking a
// `&KnowledgeBase`, so the public surface never exposes the store handle.
pub use uniko_memory::{
    AbductionResult, AddObservationParams, Agent, ArtifactIngestResult, AssertFactParams,
    AssumeBuilder, AtomicIngestResult, ContextBundle, CreateGoalParams, CreateTaskParams,
    DerivationNode, DerivationTree, Document, EmbeddingConfig, FactUpsert, GeneratedAnswer,
    InvalidateFactParams, LlmSpec, NodeId, PdfIngestResult, PdfSource, QueryOutcome, RecallConfig,
    RecallItem, RecallScope, RecallTier, Record, RecordActionParams, RecordActionResult,
    RecordEpisodeParams, Session, Turn, Uniko, UnikoBuilder, UnikoConfig, UnikoError, Value,
    ViewerScope, WorkingMemoryParams,
    nl_to_cypher::{is_safe_read_only, translate as translate_nl_to_cypher},
    policy::{Viewer, visibility_admits},
};
