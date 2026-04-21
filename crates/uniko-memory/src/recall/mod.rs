//! Basic recall API — searches the memory graph and returns ranked
//! results.
//!
//! Phase 1 implements only Phase 3 (Broaden) of the 3-phase recall
//! cascade.  At cold start (no Facts, no Procedures) this is the
//! expected behavior.  Compact (Phase 1) and Expand (Phase 2)
//! activate in execution Phase 2 when consolidation creates Facts.

// Rust guideline compliant

pub mod intent;

pub use intent::{IntentProfile, build_intent};

use std::collections::HashMap;

use serde::Serialize;

use uniko_store::{KnowledgeBase, NodeId, UnikoError};

// ── Types ───────────────────────────────────────────────────────────

/// Tier classification for recall scoring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum RecallTier {
    /// Facts, Topics (Phase 2+).
    Semantic,
    /// Procedures (Phase 3+).
    Procedural,
    /// Episodes, Observations.
    Episodic,
    /// Chunks, Artifacts.
    KnowledgeBase,
    /// Actions, Messages.
    Provenance,
}

impl RecallTier {
    /// Scoring weight for this tier.
    fn weight(self) -> f64 {
        match self {
            Self::Semantic => 0.9,
            Self::Procedural => 0.8,
            Self::Episodic => 0.7,
            Self::KnowledgeBase => 0.5,
            Self::Provenance => 0.4,
        }
    }

    /// Classify a node type label into a tier.
    fn from_label(label: &str) -> Self {
        match label {
            "Fact" | "Topic" => Self::Semantic,
            "Procedure" => Self::Procedural,
            "Episode" | "Observation" => Self::Episodic,
            "Chunk" | "Artifact" => Self::KnowledgeBase,
            _ => Self::Provenance,
        }
    }
}

/// A single recalled item.
#[derive(Debug, Clone, Serialize)]
pub struct RecallItem {
    /// Node ID.
    pub node_id: NodeId,
    /// Node type label.
    pub node_type: String,
    /// Fused score after RRF and tier weighting.
    pub score: f64,
    /// Display text.
    pub content: String,
    /// Tier classification.
    pub tier: RecallTier,
}

/// Ranked result bundle from a recall query.
#[derive(Debug, Clone, Serialize)]
pub struct ContextBundle {
    /// Ranked items.
    pub items: Vec<RecallItem>,
    /// Estimated total tokens.
    pub total_tokens: usize,
    /// Whether Compact phase was sufficient (always false in Phase 1).
    pub phase1_only: bool,
    /// Coverage score (0.0–1.0).
    pub coverage: f64,
}

/// Recall query configuration.
#[derive(Debug, Clone)]
pub struct RecallConfig {
    /// Maximum items to return.
    pub limit: usize,
    /// Maximum total tokens.
    pub token_budget: usize,
    /// Minimum fused score for inclusion.
    pub min_score: f64,
}

impl Default for RecallConfig {
    fn default() -> Self {
        Self {
            limit: 15,
            token_budget: 8192,
            // RRF scores are naturally low: 1/(k+rank) with k=60.
            // A result in 3 lists scores ~0.05 * tier_weight.
            min_score: 0.001,
        }
    }
}

// ── RRF constant ────────────────────────────────────────────────────

/// Estimated tokens per recall item.
const TOKENS_PER_ITEM: usize = 50;

// ── Main recall function ────────────────────────────────────────────

/// Recall relevant context from the memory graph.
///
/// Phase 1 implementation: executes Phase 3 (Broaden) only.
/// Searches across Messages, Chunks, Observations, and Entities via
/// fulltext BM25, vector similarity, and graph traversal, then fuses
/// results via RRF with tier weights.
///
/// # Errors
///
/// Returns [`UnikoError::Search`] on search failure.
pub async fn recall(
    kb: &KnowledgeBase,
    query: &str,
    config: &RecallConfig,
) -> Result<ContextBundle, UnikoError> {
    if query.is_empty() {
        return Ok(empty_bundle());
    }

    let intent = build_intent(kb, query).await?;
    tracing::debug!(
        intent_vec_len = intent.intent_vec.len(),
        entity_refs = ?intent.entity_refs,
        "recall intent built",
    );

    let session = kb.db().session();
    let mut scored: HashMap<NodeId, (String, String, f64)> = HashMap::new(); // (label, content, combined_score)

    // ── Hybrid similar_to: vector + fulltext per node type ─────
    // For node types with both embedding and fulltext indexes,
    // combine both scores so items matching on both signals rank highest.

    let has_vec = !intent.intent_vec.is_empty();

    // (label, embed_field, fts_field, content_field)
    let hybrid_targets: &[(&str, Option<&str>, Option<&str>, &str)] = &[
        ("Message", Some("embedding"), Some("content"), "content"),
        ("Chunk", Some("embedding"), Some("text"), "text"),
        ("Observation", Some("embedding"), Some("content"), "content"),
    ];

    for &(label, embed_field, fts_field, content_field) in hybrid_targets {
        // Build the multi-source similar_to with proper RRF fusion.
        // similar_to([sources], [queries]) handles normalization and fusion.
        let (sources, queries, params_needed) = match (embed_field.filter(|_| has_vec), fts_field) {
            (Some(ef), Some(ff)) => (
                format!("[m.{ef}, m.{ff}]"),
                "[$qvec, $qtxt]".to_string(),
                (true, true),
            ),
            (Some(ef), None) => (format!("m.{ef}"), "$qvec".to_string(), (true, false)),
            (None, Some(ff)) => (format!("m.{ff}"), "$qtxt".to_string(), (false, true)),
            (None, None) => continue,
        };

        let cypher = format!(
            "MATCH (m:{label}) \
             RETURN id(m) AS nid, labels(m)[0] AS lbl, \
                    m.{content_field} AS content, \
                    similar_to({sources}, {queries}) AS score \
             ORDER BY score DESC LIMIT $lim"
        );

        let mut builder = session.query_with(&cypher);
        builder = builder.param("lim", config.limit as i64);
        if params_needed.0 {
            builder = builder.param("qvec", uni_db::Value::Vector(intent.intent_vec.clone()));
        }
        if params_needed.1 {
            builder = builder.param("qtxt", intent.keywords.as_str());
        }

        match builder.fetch_all().await {
            Ok(result) => {
                for row in result.rows() {
                    let nid: i64 = row.get("nid").unwrap_or(0);
                    let lbl: String = row.get("lbl").unwrap_or_default();
                    let content: String = row.get("content").unwrap_or_default();
                    let score: f64 = row.get("score").unwrap_or(0.0);
                    scored
                        .entry(nid)
                        .and_modify(|(_, _, s)| *s = s.max(score))
                        .or_insert((lbl, content, score));
                }
            }
            Err(e) => tracing::debug!(label, error = %e, "hybrid similar_to failed"),
        }
    }

    // Entity-scoped search disabled pending uni-db#41 (pattern matching
    // in WHERE is ~95x slower than unscoped). Entity refs are available
    // in intent.entity_refs for future use when the performance issue
    // is resolved.
    let _ = &intent.entity_refs;

    tracing::debug!(candidates = scored.len(), "recall candidates");

    if scored.is_empty() {
        return Ok(empty_bundle());
    }

    // ── Build items with tier weights ──────────────────────────
    let mut items: Vec<RecallItem> = scored
        .into_iter()
        .filter(|(_, (_, content, _))| !content.is_empty())
        .map(|(nid, (label, content, score))| {
            let tier = RecallTier::from_label(&label);
            RecallItem {
                node_id: nid,
                node_type: label,
                score: score * tier.weight(),
                content,
                tier,
            }
        })
        .collect();

    items.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    items.truncate(config.limit);

    let mut total_tokens = 0;
    let mut final_items = Vec::new();
    for item in items {
        total_tokens += TOKENS_PER_ITEM;
        if total_tokens > config.token_budget {
            break;
        }
        final_items.push(item);
    }

    // Coverage scoring (simplified for Phase 1).
    let mean_score = if final_items.is_empty() {
        0.0
    } else {
        final_items.iter().map(|i| i.score).sum::<f64>() / final_items.len() as f64
    };
    let distinct_tiers = final_items
        .iter()
        .map(|i| std::mem::discriminant(&i.tier))
        .collect::<std::collections::HashSet<_>>()
        .len();
    let diversity = distinct_tiers as f64 / 5.0;
    let coverage = 0.3 * mean_score + 0.3 * diversity; // facet_coverage = 0 at cold start.

    Ok(ContextBundle {
        total_tokens,
        items: final_items,
        phase1_only: false, // Always false in Phase 1.
        coverage,
    })
}

fn empty_bundle() -> ContextBundle {
    ContextBundle {
        items: Vec::new(),
        total_tokens: 0,
        phase1_only: false,
        coverage: 0.0,
    }
}
