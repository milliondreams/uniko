//! Working memory traversal (F13).
//!
//! Working memory is *not* a stored node — it is computed live by
//! traversing the graph from a Goal outward through its Tasks,
//! Sessions, Messages, Facts, and Entities.  When the goal changes the
//! result recomputes instantly; when the underlying knowledge is
//! updated by consolidation the next call reflects the change.
//!
//! Per spec §IX, Phase IX (Recall Cascade), and NF17, this should
//! complete in < 200 ms on a warm in-memory store with < 10K nodes per
//! label.  The implementation uses a single Cypher query per category
//! (sessions, messages, facts, entities) rather than per-node hops so
//! that the heavy lifting stays in uni-db's planner.
//!
//! Items are ranked by tier weight × per-tier recency/importance
//! score, then truncated to fit the caller's token budget.  When the
//! `budget` is None a default of 8192 tokens is used (matches the
//! recall cascade default).

use std::collections::HashMap;

use serde::Serialize;

use uniko_store::schema::labels;
use uniko_store::{KnowledgeBase, UnikoError};

use crate::recall::{ContextBundle, RecallItem, RecallTier};

/// Default token budget when the caller does not specify one.
///
/// Matches the recall cascade default (spec §IX, "Token Budget
/// Enforcement").
pub const DEFAULT_TOKEN_BUDGET: usize = 8192;

/// Approximate tokens per item — same heuristic the recall cascade
/// uses.  A more accurate estimate would tokenize each item's content
/// but ~50 tokens is close enough for budget enforcement and keeps the
/// traversal latency budget intact.
const APPROX_TOKENS_PER_ITEM: usize = 50;

/// Inputs for [`working_memory`].
#[derive(Debug, Clone, Default)]
pub struct WorkingMemoryParams {
    /// Goal node identified by its `goal_id` external key.
    pub goal_id: String,
    /// Maximum tokens in the returned bundle.  When `None`, defaults
    /// to [`DEFAULT_TOKEN_BUDGET`].
    pub budget: Option<usize>,
    /// Include descendant goals via `PARENT_GOAL*` traversal.  Default
    /// `true`.
    pub include_subgoals: bool,
    /// Maximum number of items per tier before final budget cut.
    /// Defaults to 25.  Lower values reduce candidate-side work; higher
    /// values give the budget more material to choose from.
    pub per_tier_limit: usize,
}

impl WorkingMemoryParams {
    /// Construct params with sensible defaults for the given goal.
    #[must_use]
    pub fn new(goal_id: impl Into<String>) -> Self {
        Self {
            goal_id: goal_id.into(),
            budget: None,
            include_subgoals: true,
            per_tier_limit: 25,
        }
    }
}

/// Per-tier categorisation of working-memory traversal results.
#[derive(Debug, Clone, Copy, Serialize)]
pub enum WorkingMemoryCategory {
    /// The Goal itself plus any descendants pulled in.
    Goal,
    /// Tasks under any covered goal.
    Task,
    /// Sessions linked to any covered goal or task.
    Session,
    /// Messages within those sessions.
    Message,
    /// Facts whose subject is mentioned in messages of this goal scope.
    Fact,
    /// Entities mentioned by messages or facts in scope.
    Entity,
}

/// Assemble the working-memory bundle for a Goal.
///
/// Performs a single-pass Cypher traversal over the
/// Goal → Task → Session → Message → Fact/Entity spine.  Items are
/// scored by per-tier weight (see [`RecallTier`]) times a recency
/// boost on time-bearing nodes (Sessions, Messages) and a confidence
/// boost on Facts.  The combined ranking is truncated to fit
/// `params.budget`.
///
/// The Goal identified by `params.goal_id` must exist; absent goals
/// return an empty bundle with `coverage = 0.0` rather than an error,
/// so callers can poll while a goal is being created.
///
/// # Errors
///
/// Returns [`UnikoError::Storage`] if any of the traversal queries
/// fail to execute against the underlying store.
pub async fn working_memory(
    kb: &KnowledgeBase,
    params: WorkingMemoryParams,
) -> Result<ContextBundle, UnikoError> {
    let budget = params.budget.unwrap_or(DEFAULT_TOKEN_BUDGET);
    let per_tier_limit = params.per_tier_limit.max(1);

    // ── Step 1: Resolve goal scope ──
    //
    // The scope is "this goal + (optionally) all descendant goals via
    // PARENT_GOAL*".  We collect goal_ids as strings so subsequent
    // queries can use a flat IN-list, which uni-db plans efficiently.
    let goal_ids = kb
        .goal_scope_ids(&params.goal_id, params.include_subgoals)
        .await?;
    if goal_ids.is_empty() {
        return Ok(empty_bundle());
    }

    // ── Step 2: Run per-tier candidate queries in parallel ──
    //
    // Each call is independent; running them concurrently keeps the
    // NF17 < 200 ms target achievable when goals expand to many
    // sessions/tasks/messages.
    let (goals_res, tasks_res, sessions_res, messages_res, facts_res, entities_res) = tokio::join!(
        fetch_goal_items(kb, &goal_ids),
        fetch_task_items(kb, &goal_ids),
        fetch_session_items(kb, &goal_ids),
        fetch_message_items(kb, &goal_ids),
        fetch_fact_items(kb, &goal_ids),
        fetch_entity_items(kb, &goal_ids),
    );

    let mut items = Vec::new();
    push_results(&mut items, goals_res?, per_tier_limit);
    push_results(&mut items, tasks_res?, per_tier_limit);
    push_results(&mut items, sessions_res?, per_tier_limit);
    push_results(&mut items, messages_res?, per_tier_limit);
    push_results(&mut items, facts_res?, per_tier_limit);
    push_results(&mut items, entities_res?, per_tier_limit);

    // ── Step 3: Rank, dedup, and budget-trim ──
    crate::sort_by_score_desc(&mut items, |x| x.score);
    items.dedup_by(|a, b| a.node_id == b.node_id);

    let max_items = budget / APPROX_TOKENS_PER_ITEM;
    if items.len() > max_items {
        items.truncate(max_items);
    }

    let total_tokens = items.len() * APPROX_TOKENS_PER_ITEM;
    let coverage = coverage_score(&items);

    Ok(ContextBundle {
        items,
        total_tokens,
        phase1_only: false,
        phase2_only: false,
        coverage,
    })
}

/// Fetch the Goal nodes themselves so callers see the goal title in
/// the bundle.  Always small (≤ depth of PARENT_GOAL hierarchy).
async fn fetch_goal_items(
    kb: &KnowledgeBase,
    goal_ids: &[String],
) -> Result<Vec<RecallItem>, UnikoError> {
    let rows = kb.fetch_wm_goals(goal_ids).await?;
    Ok(rows
        .into_iter()
        .map(|r| {
            let score = if r.status == "active" { 1.0 } else { 0.6 };
            RecallItem {
                node_id: r.node_id,
                node_type: labels::GOAL.to_string(),
                score: score * RecallTier::Semantic.weight(),
                content: r.content,
                tier: RecallTier::Semantic,
            }
        })
        .collect())
}

/// Fetch Tasks that are part of any in-scope goal.
async fn fetch_task_items(
    kb: &KnowledgeBase,
    goal_ids: &[String],
) -> Result<Vec<RecallItem>, UnikoError> {
    let rows = kb.fetch_wm_tasks(goal_ids).await?;
    Ok(rows
        .into_iter()
        .map(|r| {
            // Active/in-progress tasks score higher than completed ones; a
            // tied-priority active task should beat a completed one.
            let status_boost = match r.status.as_str() {
                "active" | "in_progress" | "pending" => 1.0,
                "completed" => 0.5,
                _ => 0.75,
            };
            let score = r.priority.clamp(0.0, 1.0) * status_boost * RecallTier::Procedural.weight();
            RecallItem {
                node_id: r.node_id,
                node_type: labels::TASK.to_string(),
                score,
                content: r.content,
                tier: RecallTier::Procedural,
            }
        })
        .collect())
}

/// Fetch Sessions linked to any in-scope goal or its tasks.
async fn fetch_session_items(
    kb: &KnowledgeBase,
    goal_ids: &[String],
) -> Result<Vec<RecallItem>, UnikoError> {
    // Sessions are time-bearing: the query returns them newest-first, so
    // we map ordinal recency to a [0.4..1.0] boost without a temporal call.
    let rows = kb.fetch_wm_sessions(goal_ids).await?;
    let n = rows.len().max(1) as f64;
    Ok(rows
        .into_iter()
        .enumerate()
        .map(|(idx, r)| {
            let recency = ordinal_recency(idx, n);
            RecallItem {
                node_id: r.node_id,
                node_type: labels::SESSION.to_string(),
                score: recency * RecallTier::Episodic.weight(),
                content: r.content,
                tier: RecallTier::Episodic,
            }
        })
        .collect())
}

/// Fetch Messages from any in-scope session.
async fn fetch_message_items(
    kb: &KnowledgeBase,
    goal_ids: &[String],
) -> Result<Vec<RecallItem>, UnikoError> {
    let rows = kb.fetch_wm_messages(goal_ids).await?;
    let n = rows.len().max(1) as f64;
    Ok(rows
        .into_iter()
        .enumerate()
        .map(|(idx, r)| {
            let recency = ordinal_recency(idx, n);
            RecallItem {
                node_id: r.node_id,
                node_type: labels::MESSAGE.to_string(),
                score: recency * RecallTier::Provenance.weight(),
                content: r.content,
                tier: RecallTier::Provenance,
            }
        })
        .collect())
}

/// Fetch Facts whose subject is mentioned in any in-scope message.
async fn fetch_fact_items(
    kb: &KnowledgeBase,
    goal_ids: &[String],
) -> Result<Vec<RecallItem>, UnikoError> {
    let rows = kb.fetch_wm_facts(goal_ids).await?;
    Ok(rows
        .into_iter()
        .map(|r| RecallItem {
            node_id: r.node_id,
            node_type: labels::FACT.to_string(),
            score: r.confidence.clamp(0.0, 1.0) * RecallTier::Semantic.weight(),
            content: r.content,
            tier: RecallTier::Semantic,
        })
        .collect())
}

/// Fetch Entities mentioned in any in-scope message.
async fn fetch_entity_items(
    kb: &KnowledgeBase,
    goal_ids: &[String],
) -> Result<Vec<RecallItem>, UnikoError> {
    // The query returns entities highest-frequency first; map ordinal
    // salience the same way the time-ordered fetchers map recency.
    let rows = kb.fetch_wm_entities(goal_ids).await?;
    let n = rows.len().max(1) as f64;
    Ok(rows
        .into_iter()
        .enumerate()
        .map(|(idx, r)| {
            let salience = ordinal_recency(idx, n);
            RecallItem {
                node_id: r.node_id,
                node_type: labels::ENTITY.to_string(),
                score: salience * RecallTier::KnowledgeBase.weight(),
                content: r.content,
                tier: RecallTier::KnowledgeBase,
            }
        })
        .collect())
}

/// Linear ordinal-recency boost in `[0.4, 1.0]`: newest row scores 1.0,
/// last row scores 0.4. Shared by the three time-ordered fetchers
/// (sessions, messages, entities) — keeps their relative ranking
/// identical when the underlying list lengths differ.
fn ordinal_recency(idx: usize, n: f64) -> f64 {
    1.0 - 0.6 * (idx as f64 / n)
}

/// Truncate a per-tier result list to the configured limit before
/// merging into the global candidate pool.
fn push_results(out: &mut Vec<RecallItem>, mut tier_items: Vec<RecallItem>, limit: usize) {
    if tier_items.len() > limit {
        tier_items.truncate(limit);
    }
    out.extend(tier_items);
}

/// Compute a simple coverage score for working memory.
///
/// Working memory is goal-anchored, so coverage is meaningful as
/// "fraction of tiers represented" rather than the recall-cascade's
/// 3-component blend.
fn coverage_score(items: &[RecallItem]) -> f64 {
    if items.is_empty() {
        return 0.0;
    }
    let mut tiers = HashMap::new();
    for it in items {
        *tiers.entry(it.tier as u8).or_insert(0_u32) += 1;
    }
    // 5 possible tiers per `RecallTier`; full coverage = all 5 present.
    (tiers.len() as f64 / 5.0).clamp(0.0, 1.0)
}

fn empty_bundle() -> ContextBundle {
    ContextBundle {
        items: Vec::new(),
        total_tokens: 0,
        phase1_only: false,
        phase2_only: false,
        coverage: 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_budget_is_8192() {
        assert_eq!(DEFAULT_TOKEN_BUDGET, 8192);
    }

    #[test]
    fn params_new_sets_defaults() {
        let p = WorkingMemoryParams::new("goal-1");
        assert_eq!(p.goal_id, "goal-1");
        assert!(p.budget.is_none());
        assert!(p.include_subgoals);
        assert_eq!(p.per_tier_limit, 25);
    }

    #[test]
    fn coverage_zero_when_empty() {
        assert!((coverage_score(&[]) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn coverage_scales_with_distinct_tiers() {
        let mk = |tier| RecallItem {
            node_id: 0,
            node_type: String::new(),
            score: 0.0,
            content: String::new(),
            tier,
        };
        let items = vec![
            mk(RecallTier::Semantic),
            mk(RecallTier::Procedural),
            mk(RecallTier::Episodic),
        ];
        let c = coverage_score(&items);
        assert!((c - 0.6).abs() < 1e-9, "expected 3/5 = 0.6, got {c}",);
    }

    #[test]
    fn push_results_truncates_to_limit() {
        let mut out = Vec::new();
        let items: Vec<RecallItem> = (0..10)
            .map(|i| RecallItem {
                node_id: i,
                node_type: "X".into(),
                score: 0.0,
                content: String::new(),
                tier: RecallTier::Semantic,
            })
            .collect();
        push_results(&mut out, items, 3);
        assert_eq!(out.len(), 3);
    }
}
