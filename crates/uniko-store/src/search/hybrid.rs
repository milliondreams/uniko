//! Hybrid search combining vector similarity and fulltext BM25 via
//! Reciprocal Rank Fusion (RRF).

use std::collections::HashMap;

use crate::error::Result;
use crate::search::SearchResult;
use crate::storage::KnowledgeBase;
use crate::types::NodeId;

/// RRF smoothing constant (standard value from the literature).
/// Higher values decrease the influence of rank position.
pub const RRF_K: f64 = 60.0;

/// Hybrid-search tier classification.  Each tier carries a fixed
/// weight that scales fused RRF scores so semantically richer nodes
/// (Facts) outrank low-signal provenance hits (raw Messages) at
/// equivalent retrieval rank.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// Facts — distilled semantic knowledge.
    Semantic,
    /// Procedures — promoted procedural patterns.
    Procedural,
    /// Episodes, Observations — first-person experiential records.
    Episodic,
    /// Chunks, Artifacts — knowledge-base content.
    Kb,
    /// Actions, Messages — raw provenance.
    Provenance,
}

impl Tier {
    /// Weight multiplier applied to a hit's fused score.
    #[must_use]
    pub const fn weight(self) -> f64 {
        match self {
            Self::Semantic => 1.0,
            Self::Procedural => 0.9,
            Self::Episodic => 0.7,
            Self::Kb => 0.5,
            Self::Provenance => 0.4,
        }
    }
}

/// Tier weight for Semantic nodes (Facts).
pub const TIER_WEIGHT_SEMANTIC: f64 = Tier::Semantic.weight();
/// Tier weight for Procedural nodes (Procedures).
pub const TIER_WEIGHT_PROCEDURAL: f64 = Tier::Procedural.weight();
/// Tier weight for Episodic nodes (Episodes, Observations).
pub const TIER_WEIGHT_EPISODIC: f64 = Tier::Episodic.weight();
/// Tier weight for KnowledgeBase nodes (Chunks, Artifacts).
pub const TIER_WEIGHT_KB: f64 = Tier::Kb.weight();
/// Tier weight for Provenance nodes (Actions, Messages).
pub const TIER_WEIGHT_PROVENANCE: f64 = Tier::Provenance.weight();

/// A search target specifying what to search and how to weight results.
#[derive(Debug, Clone)]
pub struct SearchTarget {
    /// Node label (e.g. `"Fact"`, `"Message"`).
    pub node_type: String,
    /// Embedding field for vector search (e.g. `"embedding"`).
    pub embedding_field: String,
    /// Fulltext field for BM25 search.  `None` to skip fulltext.
    pub fulltext_field: Option<String>,
    /// Weight multiplier applied after RRF fusion.
    pub tier_weight: f64,
}

impl KnowledgeBase {
    /// Single-type hybrid search combining vector + fulltext via RRF.
    ///
    /// Target: < 50 ms (NF3).
    ///
    /// # Errors
    ///
    /// Returns [`crate::UnikoError::Search`] on database failure.
    pub async fn hybrid_search(
        &self,
        query: &str,
        embedding: &[f32],
        node_type: &str,
        top_k: usize,
    ) -> Result<Vec<SearchResult>> {
        // Fetch more candidates than needed so RRF has enough to fuse.
        let fetch_k = top_k * 2;

        let vector_hits = self
            .vector_search(embedding, node_type, "embedding", fetch_k)
            .await?;

        // Determine the best fulltext field for this node type.
        let ft_field = default_fulltext_field(node_type);
        let text_hits = if let Some(field) = ft_field {
            self.fulltext_search(query, node_type, field, fetch_k)
                .await?
        } else {
            Vec::new()
        };

        let lists = [vector_hits, text_hits];
        let weights = [1.0, 1.0];
        Ok(rrf_fuse(&lists, &weights, RRF_K, top_k))
    }

    /// Multi-type hybrid search with explicit tier weighting.
    ///
    /// Searches each target by vector and (optionally) fulltext, then
    /// fuses all results via RRF with per-target tier weights.
    ///
    /// # Errors
    ///
    /// Returns [`crate::UnikoError::Search`] on database failure.
    pub async fn hybrid_search_weighted(
        &self,
        query: &str,
        embedding: &[f32],
        targets: &[SearchTarget],
        top_k: usize,
    ) -> Result<Vec<SearchResult>> {
        if targets.is_empty() {
            return Ok(Vec::new());
        }

        let fetch_k = top_k * 2;
        let mut all_lists = Vec::new();
        let mut all_weights = Vec::new();

        for target in targets {
            let vector_hits = self
                .vector_search(
                    embedding,
                    &target.node_type,
                    &target.embedding_field,
                    fetch_k,
                )
                .await?;
            all_lists.push(vector_hits);
            all_weights.push(target.tier_weight);

            if let Some(ref ft_field) = target.fulltext_field {
                let text_hits = self
                    .fulltext_search(query, &target.node_type, ft_field, fetch_k)
                    .await?;
                all_lists.push(text_hits);
                all_weights.push(target.tier_weight);
            }
        }

        Ok(rrf_fuse(&all_lists, &all_weights, RRF_K, top_k))
    }
}

/// Fuse multiple ranked lists via weighted Reciprocal Rank Fusion.
///
/// For each unique node across all lists:
///   `score = Σ weight_i / (k + rank_in_list_i)`
///
/// Items appearing in multiple lists accumulate contributions from each.
fn rrf_fuse(
    lists: &[Vec<SearchResult>],
    weights: &[f64],
    k: f64,
    limit: usize,
) -> Vec<SearchResult> {
    let mut scores: HashMap<NodeId, (f64, SearchResult)> = HashMap::new();

    for (list_idx, list) in lists.iter().enumerate() {
        let w = weights.get(list_idx).copied().unwrap_or(1.0);
        for (rank, hit) in list.iter().enumerate() {
            // RRF: 1-indexed rank.
            let contribution = w / (k + (rank + 1) as f64);
            scores
                .entry(hit.node_id)
                .and_modify(|(acc, _)| *acc += contribution)
                .or_insert_with(|| (contribution, hit.clone()));
        }
    }

    let mut results: Vec<SearchResult> = scores
        .into_values()
        .map(|(score, mut hit)| {
            hit.score = score;
            hit
        })
        .collect();

    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(limit);
    results
}

/// Map a node type to its primary fulltext field, if any.
fn default_fulltext_field(node_type: &str) -> Option<&'static str> {
    match node_type {
        "Message" => Some("content"),
        "Chunk" => Some("text"),
        "Observation" => Some("content"),
        "Entity" => Some("name"),
        "Goal" | "Task" => Some("title"),
        "Topic" => Some("name"),
        "Procedure" => Some("name"),
        "Artifact" => Some("content"),
        "Fact" => Some("subject"),
        "Session" => Some("topic"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rrf_single_list() {
        let list = vec![
            SearchResult {
                node_id: 1,
                node_type: "Fact".into(),
                score: 0.9,
                properties: HashMap::new(),
            },
            SearchResult {
                node_id: 2,
                node_type: "Fact".into(),
                score: 0.8,
                properties: HashMap::new(),
            },
        ];
        let results = rrf_fuse(&[list], &[1.0], 60.0, 10);
        assert_eq!(results.len(), 2);
        // First item at rank 1: score = 1/(60+1) ≈ 0.01639
        assert!(results[0].score > results[1].score);
    }

    #[test]
    fn test_rrf_dual_source_boost() {
        let list_a = vec![SearchResult {
            node_id: 1,
            node_type: "Fact".into(),
            score: 0.9,
            properties: HashMap::new(),
        }];
        let list_b = vec![SearchResult {
            node_id: 1,
            node_type: "Fact".into(),
            score: 0.8,
            properties: HashMap::new(),
        }];
        let results = rrf_fuse(&[list_a, list_b], &[1.0, 1.0], 60.0, 10);
        assert_eq!(results.len(), 1);
        // Appearing in both lists: score = 2 * 1/(60+1) ≈ 0.03279
        let expected = 2.0 / 61.0;
        assert!((results[0].score - expected).abs() < 1e-10);
    }

    #[test]
    fn test_rrf_weighted() {
        let list_a = vec![SearchResult {
            node_id: 1,
            node_type: "Fact".into(),
            score: 0.9,
            properties: HashMap::new(),
        }];
        let list_b = vec![SearchResult {
            node_id: 2,
            node_type: "Message".into(),
            score: 0.95,
            properties: HashMap::new(),
        }];
        // Fact weight 1.0, Message weight 0.4
        let results = rrf_fuse(&[list_a, list_b], &[1.0, 0.4], 60.0, 10);
        assert_eq!(results.len(), 2);
        // Fact score = 1.0/61 ≈ 0.01639, Message score = 0.4/61 ≈ 0.00656
        assert!(results[0].node_id == 1);
    }
}
