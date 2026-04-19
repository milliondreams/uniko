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

use uniko_store::schema::constants::{edges, labels};
use uniko_store::search::SearchResult;
use uniko_store::storage::edges::Direction;
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
            min_score: 0.1,
        }
    }
}

// ── RRF constant ────────────────────────────────────────────────────

/// Reciprocal Rank Fusion dampening constant.
const RRF_K: f64 = 60.0;

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
    let mut all_results: Vec<(SearchResult, usize)> = Vec::new(); // (result, list_index)
    let mut list_count = 0usize;

    // ── Fulltext BM25 searches ──────────────────────────────────
    let ft_targets = [
        ("Message", "content", 20usize),
        ("Chunk", "text", 20),
        ("Observation", "content", 10),
    ];
    for (label, field, top_k) in ft_targets {
        match kb.fulltext_search(query, label, field, top_k).await {
            Ok(results) => {
                for r in results {
                    all_results.push((r, list_count));
                }
                list_count += 1;
            }
            Err(e) => tracing::debug!(label, error = %e, "fulltext search failed"),
        }
    }

    // ── Vector searches (only if embedding available) ───────────
    if !intent.intent_vec.is_empty() {
        let vec_targets = [
            ("Message", "embedding", 10usize),
            ("Chunk", "embedding", 10),
            ("Observation", "embedding", 10),
            ("Entity", "embedding", 10),
        ];
        for (label, field, top_k) in vec_targets {
            match kb
                .vector_search(&intent.intent_vec, label, field, top_k, None)
                .await
            {
                Ok(results) => {
                    for r in results {
                        all_results.push((r, list_count));
                    }
                    list_count += 1;
                }
                Err(e) => tracing::debug!(label, error = %e, "vector search failed"),
            }
        }
    }

    // ── Graph traversal (entity MENTIONS) ───────────────────────
    for entity_name in &intent.entity_refs {
        if let Ok(Some((entity_nid, _))) = kb
            .get_node_by_ext_id(labels::ENTITY, "name", entity_name)
            .await
        {
            // Follow MENTIONS edges to find connected Messages/Chunks.
            if let Ok(mention_edges) = kb
                .get_edges(entity_nid, edges::MENTIONS, Direction::Incoming)
                .await
            {
                for edge in mention_edges {
                    if let Ok(Some((label, props))) = kb.get_node(edge.from).await {
                        all_results.push((
                            SearchResult {
                                node_id: edge.from,
                                node_type: label,
                                score: 0.5, // Graph traversal base score.
                                properties: props,
                            },
                            list_count,
                        ));
                    }
                }
                list_count += 1;
            }
        }
    }

    if all_results.is_empty() {
        return Ok(empty_bundle());
    }

    // ── RRF fusion ──────────────────────────────────────────────
    // Group results by list and assign ranks.
    let mut ranked_lists: HashMap<usize, Vec<(NodeId, usize)>> = HashMap::new();
    let mut node_info: HashMap<NodeId, (String, String, String)> = HashMap::new(); // (node_type, content, label)

    for (result, list_idx) in &all_results {
        let rank = ranked_lists.get(list_idx).map_or(0, |v| v.len());
        ranked_lists
            .entry(*list_idx)
            .or_default()
            .push((result.node_id, rank));

        node_info.entry(result.node_id).or_insert_with(|| {
            let content = result
                .properties
                .get("content")
                .or_else(|| result.properties.get("text"))
                .or_else(|| result.properties.get("name"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            (result.node_type.clone(), content, result.node_type.clone())
        });
    }

    // Compute RRF scores.
    let mut rrf_scores: HashMap<NodeId, f64> = HashMap::new();
    for entries in ranked_lists.values() {
        for &(nid, rank) in entries {
            *rrf_scores.entry(nid).or_insert(0.0) += 1.0 / (RRF_K + (rank + 1) as f64);
        }
    }

    // Apply tier weights and build items.
    let mut items: Vec<RecallItem> = rrf_scores
        .into_iter()
        .filter_map(|(nid, rrf)| {
            let (node_type, content, _) = node_info.get(&nid)?;
            let tier = RecallTier::from_label(node_type);
            let score = rrf * tier.weight();
            if score < config.min_score {
                return None;
            }
            Some(RecallItem {
                node_id: nid,
                node_type: node_type.clone(),
                score,
                content: content.clone(),
                tier,
            })
        })
        .collect();

    // Sort by score descending.
    items.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Truncate to limit and token budget.
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
