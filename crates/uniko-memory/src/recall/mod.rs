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

use futures::future::join_all;
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
            "Episode" => Self::Episodic,
            // Observations index `content` directly now and compete with
            // Chunks on raw similarity. Putting them in KnowledgeBase
            // (same weight as Chunk) avoids the tier multiplier
            // crowding Chunks out of the bundle.
            "Observation" => Self::KnowledgeBase,
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
    /// Vector similarity weight in hybrid fusion.
    pub vector_weight: f64,
    /// BM25 fulltext weight in hybrid fusion.
    pub bm25_weight: f64,
    /// When true, re-score the top `reranker_top_n` RRF candidates with
    /// the cross-encoder registered at `rerank/default`.
    pub reranker_enabled: bool,
    /// Number of RRF candidates to send to the reranker.
    pub reranker_top_n: usize,
    /// Apply sigmoid to raw cross-encoder logits.
    pub reranker_apply_sigmoid: bool,
    /// When > 1.0, multiplies the score of any RecallItem whose
    /// connected entities include one of `IntentProfile.expected_answer_type`.
    /// `1.0` is a no-op. Only triggers when `predict_answer_type` returns
    /// a label.
    pub answer_type_boost: f64,
    /// Cap on how many top items are checked for the boost (one Cypher
    /// lookup per item — avoid scaling badly when limit is large).
    pub answer_type_top_n: usize,
    /// Variant labels to enable for multi-query reformulation. Empty
    /// vec means "all default variants" (`keywords`, `original`,
    /// `declarative`, `type_anchored`). To reproduce legacy
    /// single-query behaviour, pass `vec!["keywords".into()]`.
    pub query_variants: Vec<String>,
    /// `k` constant for reciprocal rank fusion across variants. Higher
    /// values flatten the weight given to top ranks (k=60 is the
    /// canonical default).
    pub rrf_k: f64,
    /// LIMIT applied to each per-variant Cypher query. Larger values
    /// keep more candidates per variant in the fusion pool at the
    /// cost of latency. Defaults to `limit` when constructed via
    /// `from_uniko_config`.
    pub per_variant_limit: usize,
}

impl Default for RecallConfig {
    fn default() -> Self {
        Self {
            limit: 15,
            token_budget: 8192,
            min_score: 0.001,
            vector_weight: 0.5,
            bm25_weight: 0.5,
            reranker_enabled: false,
            reranker_top_n: 50,
            reranker_apply_sigmoid: true,
            // Off by default. The naive "any connected entity matches
            // predicted type → boost" rule swamps top-K with off-target
            // hits when the predicted type is common in the corpus
            // (especially `measurement` for "how many" questions).
            // Measured −0.149 R@5 / −0.186 NDCG@5 on a 24-question
            // LongMemEval slice (2026-05-03). Set to a small value like
            // 1.05 if used as a tiebreaker, or leave at 1.0.
            answer_type_boost: 1.0,
            answer_type_top_n: 50,
            query_variants: Vec::new(),
            rrf_k: 60.0,
            per_variant_limit: 15,
        }
    }
}

impl RecallConfig {
    /// Build from [`UnikoConfig`](uniko_store::config::UnikoConfig).
    pub fn from_uniko_config(cfg: &uniko_store::config::UnikoConfig) -> Self {
        Self {
            limit: cfg.recall_limit,
            token_budget: cfg.recall_token_budget,
            min_score: cfg.recall_min_score,
            vector_weight: cfg.recall_vector_weight,
            bm25_weight: cfg.recall_bm25_weight,
            reranker_enabled: cfg.reranker.enabled,
            reranker_top_n: cfg.reranker.top_n,
            reranker_apply_sigmoid: cfg.reranker.apply_sigmoid,
            // Off by default. The naive "any connected entity matches
            // predicted type → boost" rule swamps top-K with off-target
            // hits when the predicted type is common in the corpus
            // (especially `measurement` for "how many" questions).
            // Measured −0.149 R@5 / −0.186 NDCG@5 on a 24-question
            // LongMemEval slice (2026-05-03). Set to a small value like
            // 1.05 if used as a tiebreaker, or leave at 1.0.
            answer_type_boost: 1.0,
            answer_type_top_n: 50,
            query_variants: cfg.query_variants.clone(),
            rrf_k: cfg.rrf_k,
            per_variant_limit: cfg.recall_per_variant_limit.unwrap_or(cfg.recall_limit),
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

    let intent = build_intent(kb, query, &config.query_variants).await?;
    tracing::debug!(
        variant_count = intent.variants.len(),
        entity_refs = ?intent.entity_refs,
        "recall intent built",
    );

    // Per-variant fan-out: each query variant runs the full hybrid +
    // entity-scoped pipeline; the resulting per-variant ranked lists
    // are RRF-fused to produce the final candidate set. Variants run
    // concurrently — each future is independent I/O against lance.
    let variant_futures: Vec<_> = intent
        .variants
        .iter()
        .map(|variant| {
            let entity_refs = intent.entity_refs.clone();
            async move {
                let ranked = run_recall_for_variant(
                    kb,
                    variant.vector.as_slice(),
                    variant.text.as_str(),
                    &entity_refs,
                    config,
                )
                .await;
                (variant.label, ranked)
            }
        })
        .collect();
    let variant_results: Vec<(&'static str, Vec<RankedHit>)> = join_all(variant_futures).await;

    for (label, hits) in &variant_results {
        tracing::info!(variant = label, hits = hits.len(), "variant complete");
    }

    // RRF fusion across variants. Pure function so it can be unit
    // tested with synthetic ranked lists.
    let scored = rrf_fuse(
        variant_results.iter().map(|(_, hits)| hits.as_slice()),
        config.rrf_k,
    );

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

    // ── Optional cross-encoder reranker stage ──────────────────────
    // Re-score the top RRF candidates with a cross-encoder before
    // truncating to the recall limit.  Disabled by default — when off,
    // this is a no-op.
    if config.reranker_enabled && !items.is_empty() {
        let top_n = config.reranker_top_n.min(items.len());
        let docs: Vec<&str> = items[..top_n].iter().map(|i| i.content.as_str()).collect();
        match kb
            .db()
            .xervo()
            .rerank(uniko_store::schema::RERANK_ALIAS, query, &docs)
            .await
        {
            Ok(scored) => {
                let mut head: Vec<RecallItem> = scored
                    .into_iter()
                    .filter_map(|s| {
                        items.get(s.index).cloned().map(|mut it| {
                            it.score = if config.reranker_apply_sigmoid {
                                1.0 / (1.0 + f64::exp(-(s.score as f64)))
                            } else {
                                s.score as f64
                            };
                            it
                        })
                    })
                    .collect();
                let tail = items.split_off(top_n);
                head.extend(tail);
                items = head;
                items.sort_by(|a, b| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            Err(e) => tracing::warn!(error = %e, "reranker call failed, falling back to RRF order"),
        }
    }

    // ── Answer-type boost ────────────────────────────────────────────
    // When the question's surface form predicts an entity_type
    // ("where" → location, "how many" → measurement, …), boost any
    // top-N item whose connected entities include one of that type.
    // No-op when no rule fired or the boost is 1.0.
    if config.answer_type_boost > 1.0
        && let Some(target_type) = intent.expected_answer_type
        && !items.is_empty()
    {
        let n = config.answer_type_top_n.min(items.len());
        let mut matched = 0usize;
        for item in items.iter_mut().take(n) {
            if entity_type_match(kb, item.node_id, target_type).await {
                item.score *= config.answer_type_boost;
                matched += 1;
            }
        }
        if matched > 0 {
            items.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        tracing::info!(
            target_type,
            checked = n,
            matched,
            boost = config.answer_type_boost,
            "answer-type reweight applied",
        );
    }

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

/// Does any Entity reachable from `node_id` (via the relationships
/// the recall path actually walks — `MENTIONS` from Message, `ABOUT`
/// from Observation/Chunk) carry `entity_type = target_type`? Also
/// matches when the node *is* an Entity of that type.
///
/// Single Cypher query per call; intended to be invoked on at most
/// `answer_type_top_n` items (default 50). Returns `false` on any
/// query error so a transient failure can't accidentally re-rank
/// items downward.
async fn entity_type_match(
    kb: &uniko_store::KnowledgeBase,
    node_id: uniko_store::NodeId,
    target_type: &str,
) -> bool {
    let session = kb.db().session();
    let q = format!(
        "MATCH (n) WHERE id(n) = {nid} \
         OPTIONAL MATCH (n)-[:MENTIONS|ABOUT]->(e1:Entity {{entity_type: '{t}'}}) \
         OPTIONAL MATCH (n)<-[:ABOUT]-(:Observation)-[:ABOUT]->(e2:Entity {{entity_type: '{t}'}}) \
         RETURN (n.entity_type = '{t}' OR e1 IS NOT NULL OR e2 IS NOT NULL) AS hit \
         LIMIT 1",
        nid = node_id,
        t = target_type.replace('\'', "\\'"),
    );
    match session.query_with(&q).fetch_all().await {
        Ok(result) => result
            .rows()
            .first()
            .and_then(|row| row.get::<bool>("hit").ok())
            .unwrap_or(false),
        Err(e) => {
            tracing::debug!(error = %e, node_id, "entity_type_match: query failed");
            false
        }
    }
}

/// One scored hit returned by [`run_recall_for_variant`]. The order
/// of the returned `Vec<RankedHit>` is `score`-descending; downstream
/// RRF fusion uses the *index* (= rank) rather than `raw_score`.
#[derive(Debug, Clone)]
struct RankedHit {
    node_id: NodeId,
    label: String,
    content: String,
    raw_score: f64,
}

/// Run the full hybrid + entity-scoped recall pipeline for a single
/// query variant. Returns a deduped, score-descending list ready for
/// RRF fusion against the other variants' lists.
///
/// Per-node deduplication uses `score.max()` *within* a single
/// variant's queries (same semantics as the legacy single-query code).
/// Cross-variant fusion is done by the caller via reciprocal rank.
async fn run_recall_for_variant(
    kb: &KnowledgeBase,
    qvec: &[f32],
    qtxt: &str,
    entity_refs: &[String],
    config: &RecallConfig,
) -> Vec<RankedHit> {
    let session = kb.db().session();
    let mut scored: HashMap<NodeId, (String, String, f64)> = HashMap::new();
    let has_vec = !qvec.is_empty();

    // Hybrid (vector + BM25) over chunk types.
    let hybrid_targets: &[(&str, Option<&str>, Option<&str>, &str, &str)] = &[
        (
            "Chunk",
            Some("embedding"),
            Some("text"),
            "text",
            "m.chunk_type = 'session'",
        ),
        (
            "Chunk",
            Some("embedding"),
            Some("text"),
            "text",
            "m.chunk_type = 'observation'",
        ),
    ];

    for &(label, embed_field, fts_field, content_field, where_clause) in hybrid_targets {
        let (sources, queries, fusion, params_needed) =
            match (embed_field.filter(|_| has_vec), fts_field) {
                (Some(ef), Some(ff)) => (
                    format!("[m.{ef}, m.{ff}]"),
                    "[$qvec, $qtxt]".to_string(),
                    format!(
                        ", {{method: 'weighted', weights: [{}, {}]}}",
                        config.vector_weight, config.bm25_weight
                    ),
                    (true, true),
                ),
                (Some(ef), None) => (
                    format!("m.{ef}"),
                    "$qvec".to_string(),
                    String::new(),
                    (true, false),
                ),
                (None, Some(ff)) => (
                    format!("m.{ff}"),
                    "$qtxt".to_string(),
                    String::new(),
                    (false, true),
                ),
                (None, None) => continue,
            };

        let where_part = if where_clause.is_empty() {
            String::new()
        } else {
            format!(" WHERE {where_clause}")
        };

        let cypher = format!(
            "MATCH (m:{label}){where_part} \
             RETURN id(m) AS nid, labels(m)[0] AS lbl, \
                    m.{content_field} AS content, \
                    similar_to({sources}, {queries}{fusion}) AS score \
             ORDER BY score DESC LIMIT $lim"
        );

        let mut builder = session.query_with(&cypher);
        builder = builder.param("lim", config.per_variant_limit as i64);
        if params_needed.0 {
            builder = builder.param("qvec", uni_db::Value::Vector(qvec.to_vec()));
        }
        if params_needed.1 {
            builder = builder.param("qtxt", qtxt);
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

    // Entity-scoped fan-out.
    let entity_scoped_targets: &[(&str, &str, &str, &str)] = &[
        (
            "Chunk",
            "text",
            "embedding",
            "(m)<-[:HAS_CHUNK]-(:Session)<-[:PARTICIPATED_IN]-(:Participant {name: $ename})",
        ),
        (
            "Chunk",
            "text",
            "embedding",
            "(m)-[:ABOUT]->(:Participant {name: $ename})",
        ),
        (
            "Chunk",
            "text",
            "embedding",
            "(m)-[:ABOUT]->(:Entity {name: $ename})",
        ),
        (
            "Observation",
            "content",
            "embedding",
            "(m)-[:ABOUT]->(:Participant {name: $ename})",
        ),
        (
            "Observation",
            "content",
            "embedding",
            "(m)-[:ABOUT]->(:Entity {name: $ename})",
        ),
        (
            "Observation",
            "content",
            "embedding",
            "m.subject = $ename",
        ),
    ];

    for entity_name in entity_refs {
        for &(label, content_field, embed_field, pattern) in entity_scoped_targets {
            let cypher = if has_vec {
                format!(
                    "MATCH (m:{label}) \
                     WHERE {pattern} \
                     RETURN id(m) AS nid, labels(m)[0] AS lbl, \
                            m.{content_field} AS content, \
                            similar_to([m.{embed_field}, m.{content_field}], [$qvec, $qtxt]) AS score \
                     ORDER BY score DESC LIMIT $lim"
                )
            } else {
                format!(
                    "MATCH (m:{label}) \
                     WHERE {pattern} \
                     RETURN id(m) AS nid, labels(m)[0] AS lbl, \
                            m.{content_field} AS content, \
                            similar_to(m.{content_field}, $qtxt) AS score \
                     ORDER BY score DESC LIMIT $lim"
                )
            };

            let mut builder = session.query_with(&cypher);
            builder = builder
                .param("ename", entity_name.as_str())
                .param("qtxt", qtxt)
                .param("lim", config.per_variant_limit as i64);
            if has_vec {
                builder = builder.param("qvec", uni_db::Value::Vector(qvec.to_vec()));
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
                Err(e) => {
                    tracing::debug!(
                        entity = entity_name,
                        label,
                        error = %e,
                        "entity-scoped search failed"
                    )
                }
            }
        }
    }

    let mut hits: Vec<RankedHit> = scored
        .into_iter()
        .filter(|(_, (_, content, _))| !content.is_empty())
        .map(|(node_id, (label, content, raw_score))| RankedHit {
            node_id,
            label,
            content,
            raw_score,
        })
        .collect();
    hits.sort_by(|a, b| {
        b.raw_score
            .partial_cmp(&a.raw_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    hits
}

/// Reciprocal-rank fusion across a set of per-variant ranked lists.
///
/// Each variant contributes `1 / (k + rank)` to a per-node accumulator.
/// Nodes appearing across multiple variants accumulate score; no
/// single variant dominates because the contribution is rank-based,
/// not raw-score based. The first variant that mentions a node
/// supplies the `(label, content)` returned to the caller.
fn rrf_fuse<'a, I>(per_variant: I, k: f64) -> HashMap<NodeId, (String, String, f64)>
where
    I: IntoIterator<Item = &'a [RankedHit]>,
{
    let mut fused: HashMap<NodeId, (String, String, f64)> = HashMap::new();
    for ranked in per_variant {
        for (rank, hit) in ranked.iter().enumerate() {
            let contribution = 1.0 / (k + rank as f64);
            fused
                .entry(hit.node_id)
                .and_modify(|(_, _, s)| *s += contribution)
                .or_insert((hit.label.clone(), hit.content.clone(), contribution));
        }
    }
    fused
}

fn empty_bundle() -> ContextBundle {
    ContextBundle {
        items: Vec::new(),
        total_tokens: 0,
        phase1_only: false,
        coverage: 0.0,
    }
}

#[cfg(test)]
mod rrf_tests {
    use super::*;

    fn hit(node_id: i64, score: f64) -> RankedHit {
        RankedHit {
            node_id,
            label: "Chunk".to_string(),
            content: format!("doc {node_id}"),
            raw_score: score,
        }
    }

    fn fused_sorted(per_variant: Vec<Vec<RankedHit>>, k: f64) -> Vec<(NodeId, f64)> {
        let slices: Vec<&[RankedHit]> = per_variant.iter().map(|v| v.as_slice()).collect();
        let mut pairs: Vec<(NodeId, f64)> = rrf_fuse(slices.into_iter(), k)
            .into_iter()
            .map(|(nid, (_, _, s))| (nid, s))
            .collect();
        pairs.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        pairs
    }

    #[test]
    fn rrf_one_variant_orders_by_input_rank() {
        // Single variant — fused order should match the input's
        // descending raw_score order (i.e. input rank).
        let v0 = vec![hit(1, 0.9), hit(2, 0.5), hit(3, 0.1)];
        let pairs = fused_sorted(vec![v0], 60.0);
        let nids: Vec<i64> = pairs.iter().map(|(n, _)| *n).collect();
        assert_eq!(nids, vec![1, 2, 3]);
    }

    #[test]
    fn rrf_fuses_three_variants() {
        // Item 1 appears in variants 0 and 2 (rank 0 in both).
        // Item 2 appears only in variant 1 (rank 0).
        // Item 1 should outrank item 2 in fused order because it
        // accumulates two contributions (1/60 + 1/60) vs one (1/60).
        let v0 = vec![hit(1, 0.9), hit(3, 0.5)];
        let v1 = vec![hit(2, 0.9), hit(4, 0.5)];
        let v2 = vec![hit(1, 0.9), hit(5, 0.5)];
        let pairs = fused_sorted(vec![v0, v1, v2], 60.0);
        let nids: Vec<i64> = pairs.iter().map(|(n, _)| *n).collect();
        assert_eq!(nids[0], 1, "item 1 hit by 2 variants must rank first");
        // The remaining four items each got one rank-0 or rank-1
        // contribution; their order between them doesn't matter for
        // this assertion.
    }

    #[test]
    fn rrf_robust_to_noisy_variant() {
        // Two variants put gold (id=42) at ranks 0, 1.
        // A third variant returns junk and gold is absent there.
        // Gold must still win because its accumulated 1/60 + 1/61
        // exceeds any junk item's 1/60 from one variant.
        let v0 = vec![hit(42, 1.0), hit(7, 0.5)];
        let v1 = vec![hit(8, 1.0), hit(42, 0.6)];
        let v2 = vec![hit(99, 1.0), hit(98, 0.5)];
        let pairs = fused_sorted(vec![v0, v1, v2], 60.0);
        assert_eq!(pairs[0].0, 42, "gold should rank first across noisy variants");
    }

    #[test]
    fn rrf_empty_input_returns_empty_map() {
        let empty: Vec<Vec<RankedHit>> = vec![];
        let pairs = fused_sorted(empty, 60.0);
        assert!(pairs.is_empty());
    }
}
