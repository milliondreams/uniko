//! Data types for the observation extraction pipeline.

// Rust guideline compliant

use chrono::{DateTime, Utc};

use uniko_store::NodeId;

/// A raw observation extracted from text before node creation.
///
/// Observations are direct factual statements from messages — not
/// derived knowledge.  Each must be self-contained: understandable
/// without the surrounding message context.
#[derive(Debug, Clone)]
pub struct RawObservation {
    /// Self-contained factual statement.
    pub content: String,
    /// Primary subject entity name (matches an Entity.name from P2).
    pub subject: String,
    /// Normalized verb predicate (snake_case), when extractable.
    ///
    /// Surfaced so P4 Consolidation can group by `(subject, predicate)`
    /// without re-parsing `content`.  `None` when the matcher pathway
    /// (rule-based fallback, copular-only sentence) could not capture
    /// a clean verb token.
    pub predicate: Option<String>,
    /// Object phrase (direct/indirect/complement), when extractable.
    pub object: Option<String>,
    /// When the observation was true in the real world.
    pub observed_at: DateTime<Utc>,
    /// Extraction confidence in \[0.0, 1.0\].
    pub confidence: f64,
}

/// A potential contradiction between two observations.
///
/// Flagged for P4 (consolidation) — P3 never invalidates anything.
#[derive(Debug, Clone)]
pub struct ContradictionFlag {
    /// The newly created observation.
    pub new_observation_id: NodeId,
    /// The existing observation that may be contradicted.
    pub existing_observation_id: NodeId,
    /// Shared subject entity name.
    pub subject: String,
    /// Cosine similarity between embeddings (low = likely contradiction).
    pub similarity: f64,
}
