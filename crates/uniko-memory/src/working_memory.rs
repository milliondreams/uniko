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

use uni_db::Value;

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

/// Maximum candidate rows fetched per category before ranking.
///
/// Wider than the final bundle so the per-tier ranking has room to
/// surface the best candidates.  Latency stays well under NF17 because
/// each category is one indexed query.
const CANDIDATES_PER_CATEGORY: i64 = 50;

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
    let session = kb.db().session();

    // ── Step 1: Resolve goal scope ──
    //
    // The scope is "this goal + (optionally) all descendant goals via
    // PARENT_GOAL*".  We collect goal_ids as strings so subsequent
    // queries can use a flat IN-list, which uni-db plans efficiently.
    let goal_ids = collect_goal_scope(kb, &params).await?;
    if goal_ids.is_empty() {
        return Ok(empty_bundle());
    }

    // ── Step 2: Run per-tier candidate queries in parallel ──
    //
    // Each call is independent; running them concurrently keeps the
    // NF17 < 200 ms target achievable when goals expand to many
    // sessions/tasks/messages.
    let (goals_res, tasks_res, sessions_res, messages_res, facts_res, entities_res) = tokio::join!(
        fetch_goal_items(&session, &goal_ids),
        fetch_task_items(&session, &goal_ids),
        fetch_session_items(&session, &goal_ids),
        fetch_message_items(&session, &goal_ids),
        fetch_fact_items(&session, &goal_ids),
        fetch_entity_items(&session, &goal_ids),
    );

    let mut items = Vec::new();
    push_results(&mut items, goals_res?, per_tier_limit);
    push_results(&mut items, tasks_res?, per_tier_limit);
    push_results(&mut items, sessions_res?, per_tier_limit);
    push_results(&mut items, messages_res?, per_tier_limit);
    push_results(&mut items, facts_res?, per_tier_limit);
    push_results(&mut items, entities_res?, per_tier_limit);

    // ── Step 3: Rank, dedup, and budget-trim ──
    items.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
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

/// Compute the goal scope: the input goal plus, if requested, every
/// descendant reachable via `PARENT_GOAL*`.
async fn collect_goal_scope(
    kb: &KnowledgeBase,
    params: &WorkingMemoryParams,
) -> Result<Vec<String>, UnikoError> {
    // First confirm the goal exists.
    let exists = kb
        .get_node_by_ext_id(labels::GOAL, "goal_id", &params.goal_id)
        .await?
        .is_some();
    if !exists {
        return Ok(Vec::new());
    }

    if !params.include_subgoals {
        return Ok(vec![params.goal_id.clone()]);
    }

    let session = kb.db().session();
    let cypher = "MATCH (g:Goal {goal_id: $gid}) \
                  OPTIONAL MATCH (sub:Goal)-[:PARENT_GOAL*1..5]->(g) \
                  WITH collect(DISTINCT g.goal_id) + collect(DISTINCT sub.goal_id) AS ids \
                  UNWIND ids AS gid \
                  WITH gid WHERE gid IS NOT NULL \
                  RETURN DISTINCT gid AS gid";
    let result = session
        .query_with(cypher)
        .param("gid", params.goal_id.as_str())
        .fetch_all()
        .await?;

    let ids: Vec<String> = result
        .rows()
        .iter()
        .filter_map(|row| row.get::<String>("gid").ok())
        .collect();
    if ids.is_empty() {
        Ok(vec![params.goal_id.clone()])
    } else {
        Ok(ids)
    }
}

/// Fetch the Goal nodes themselves so callers see the goal title in
/// the bundle.  Always small (≤ depth of PARENT_GOAL hierarchy).
async fn fetch_goal_items(
    session: &uni_db::Session,
    goal_ids: &[String],
) -> Result<Vec<RecallItem>, UnikoError> {
    let cypher = "MATCH (g:Goal) WHERE g.goal_id IN $ids \
                  RETURN id(g) AS nid, \
                         coalesce(g.title, g.goal_id) AS content, \
                         coalesce(g.status, 'active') AS status";
    let result = session
        .query_with(cypher)
        .param("ids", goal_ids_value(goal_ids))
        .fetch_all()
        .await?;

    let mut items = Vec::new();
    for row in result.rows() {
        let Ok(nid) = row.get::<i64>("nid") else {
            continue;
        };
        let content: String = row.get("content").unwrap_or_default();
        let status: String = row.get("status").unwrap_or_default();
        let score = if status == "active" { 1.0 } else { 0.6 };
        items.push(RecallItem {
            node_id: nid,
            node_type: labels::GOAL.to_string(),
            score: score * RecallTier::Semantic.weight(),
            content,
            tier: RecallTier::Semantic,
        });
    }
    Ok(items)
}

/// Fetch Tasks that are part of any in-scope goal.
async fn fetch_task_items(
    session: &uni_db::Session,
    goal_ids: &[String],
) -> Result<Vec<RecallItem>, UnikoError> {
    let cypher = "MATCH (t:Task)-[:PART_OF]->(g:Goal) \
                  WHERE g.goal_id IN $ids \
                  RETURN id(t) AS nid, \
                         coalesce(t.title, t.task_id) AS content, \
                         coalesce(t.priority, 0.5) AS prio, \
                         coalesce(t.status, 'pending') AS status \
                  LIMIT $lim";
    let result = session
        .query_with(cypher)
        .param("ids", goal_ids_value(goal_ids))
        .param("lim", CANDIDATES_PER_CATEGORY)
        .fetch_all()
        .await?;

    let mut items = Vec::new();
    for row in result.rows() {
        let Ok(nid) = row.get::<i64>("nid") else {
            continue;
        };
        let content: String = row.get("content").unwrap_or_default();
        let prio: f64 = row.get("prio").unwrap_or(0.5);
        let status: String = row.get("status").unwrap_or_default();
        // Active/in-progress tasks score higher than completed ones; a
        // tied-priority active task should beat a completed one.
        let status_boost = match status.as_str() {
            "active" | "in_progress" | "pending" => 1.0,
            "completed" => 0.5,
            _ => 0.75,
        };
        let score = prio.clamp(0.0, 1.0) * status_boost * RecallTier::Procedural.weight();
        items.push(RecallItem {
            node_id: nid,
            node_type: labels::TASK.to_string(),
            score,
            content,
            tier: RecallTier::Procedural,
        });
    }
    Ok(items)
}

/// Fetch Sessions linked to any in-scope goal or its tasks.
async fn fetch_session_items(
    session: &uni_db::Session,
    goal_ids: &[String],
) -> Result<Vec<RecallItem>, UnikoError> {
    let cypher = "MATCH (s:Session) \
                  WHERE EXISTS { (s)-[:FOR_GOAL]->(g:Goal) WHERE g.goal_id IN $ids } \
                     OR EXISTS { (s)-[:FOR_TASK]->(:Task)-[:PART_OF]->(g:Goal) WHERE g.goal_id IN $ids } \
                  RETURN id(s) AS nid, \
                         coalesce(s.topic, s.session_id) AS content, \
                         s.started_at AS started_at \
                  ORDER BY s.started_at DESC \
                  LIMIT $lim";
    let result = session
        .query_with(cypher)
        .param("ids", goal_ids_value(goal_ids))
        .param("lim", CANDIDATES_PER_CATEGORY)
        .fetch_all()
        .await?;

    // Sessions are time-bearing: rank by recency.  The query already
    // orders by started_at DESC; we map ordinal recency to a [0..1]
    // boost without making another temporal call.
    let mut items = Vec::new();
    let rows = result.rows();
    let n = rows.len().max(1) as f64;
    for (idx, row) in rows.iter().enumerate() {
        let Ok(nid) = row.get::<i64>("nid") else {
            continue;
        };
        let content: String = row.get("content").unwrap_or_default();
        // 1.0 for the newest, decaying linearly toward 0.4 for the oldest.
        let recency = 1.0 - 0.6 * (idx as f64 / n);
        items.push(RecallItem {
            node_id: nid,
            node_type: labels::SESSION.to_string(),
            score: recency * RecallTier::Episodic.weight(),
            content,
            tier: RecallTier::Episodic,
        });
    }
    Ok(items)
}

/// Fetch Messages from any in-scope session.
async fn fetch_message_items(
    session: &uni_db::Session,
    goal_ids: &[String],
) -> Result<Vec<RecallItem>, UnikoError> {
    let cypher = "MATCH (m:Message)-[:IN_SESSION]->(s:Session) \
                  WHERE EXISTS { (s)-[:FOR_GOAL]->(g:Goal) WHERE g.goal_id IN $ids } \
                     OR EXISTS { (s)-[:FOR_TASK]->(:Task)-[:PART_OF]->(g:Goal) WHERE g.goal_id IN $ids } \
                  RETURN id(m) AS nid, \
                         m.content AS content \
                  ORDER BY m.timestamp DESC \
                  LIMIT $lim";
    let result = session
        .query_with(cypher)
        .param("ids", goal_ids_value(goal_ids))
        .param("lim", CANDIDATES_PER_CATEGORY)
        .fetch_all()
        .await?;

    let mut items = Vec::new();
    let rows = result.rows();
    let n = rows.len().max(1) as f64;
    for (idx, row) in rows.iter().enumerate() {
        let Ok(nid) = row.get::<i64>("nid") else {
            continue;
        };
        let content: String = row.get("content").unwrap_or_default();
        let recency = 1.0 - 0.6 * (idx as f64 / n);
        items.push(RecallItem {
            node_id: nid,
            node_type: labels::MESSAGE.to_string(),
            score: recency * RecallTier::Provenance.weight(),
            content,
            tier: RecallTier::Provenance,
        });
    }
    Ok(items)
}

/// Fetch Facts whose subject is mentioned in any in-scope message.
async fn fetch_fact_items(
    session: &uni_db::Session,
    goal_ids: &[String],
) -> Result<Vec<RecallItem>, UnikoError> {
    // Facts attach to the graph via DERIVED_FROM → Episode/Action and
    // via SUPPORTED_BY → Observation → MENTIONS → Entity.  The cheapest
    // path that anchors a fact to a goal is through Entity mentions in
    // any in-scope Message.  We bound the join with an IN-list of
    // goal_ids to keep the planner happy.
    let cypher = "MATCH (m:Message)-[:IN_SESSION]->(s:Session) \
                  WHERE EXISTS { (s)-[:FOR_GOAL]->(g:Goal) WHERE g.goal_id IN $ids } \
                     OR EXISTS { (s)-[:FOR_TASK]->(:Task)-[:PART_OF]->(g:Goal) WHERE g.goal_id IN $ids } \
                  MATCH (m)-[:MENTIONS]->(ent:Entity) \
                  MATCH (f:Fact) WHERE f.subject = ent.name \
                  RETURN DISTINCT id(f) AS nid, \
                         coalesce(f.object, f.subject) AS content, \
                         coalesce(f.confidence, 0.5) AS conf \
                  ORDER BY conf DESC \
                  LIMIT $lim";
    let result = session
        .query_with(cypher)
        .param("ids", goal_ids_value(goal_ids))
        .param("lim", CANDIDATES_PER_CATEGORY)
        .fetch_all()
        .await?;

    let mut items = Vec::new();
    for row in result.rows() {
        let Ok(nid) = row.get::<i64>("nid") else {
            continue;
        };
        let content: String = row.get("content").unwrap_or_default();
        let conf: f64 = row.get("conf").unwrap_or(0.5);
        items.push(RecallItem {
            node_id: nid,
            node_type: labels::FACT.to_string(),
            score: conf.clamp(0.0, 1.0) * RecallTier::Semantic.weight(),
            content,
            tier: RecallTier::Semantic,
        });
    }
    Ok(items)
}

/// Fetch Entities mentioned in any in-scope message.
async fn fetch_entity_items(
    session: &uni_db::Session,
    goal_ids: &[String],
) -> Result<Vec<RecallItem>, UnikoError> {
    let cypher = "MATCH (m:Message)-[:IN_SESSION]->(s:Session) \
                  WHERE EXISTS { (s)-[:FOR_GOAL]->(g:Goal) WHERE g.goal_id IN $ids } \
                     OR EXISTS { (s)-[:FOR_TASK]->(:Task)-[:PART_OF]->(g:Goal) WHERE g.goal_id IN $ids } \
                  MATCH (m)-[r:MENTIONS]->(ent:Entity) \
                  RETURN DISTINCT id(ent) AS nid, \
                         ent.name AS content, \
                         coalesce(ent.frequency, 1) AS freq \
                  ORDER BY freq DESC \
                  LIMIT $lim";
    let result = session
        .query_with(cypher)
        .param("ids", goal_ids_value(goal_ids))
        .param("lim", CANDIDATES_PER_CATEGORY)
        .fetch_all()
        .await?;

    let mut items = Vec::new();
    let rows = result.rows();
    let n = rows.len().max(1) as f64;
    for (idx, row) in rows.iter().enumerate() {
        let Ok(nid) = row.get::<i64>("nid") else {
            continue;
        };
        let content: String = row.get("content").unwrap_or_default();
        // Rank by frequency ordinal: highest-freq entity gets full
        // weight, tapering off to 0.4 at the bottom of the list.
        let salience = 1.0 - 0.6 * (idx as f64 / n);
        items.push(RecallItem {
            node_id: nid,
            node_type: labels::ENTITY.to_string(),
            score: salience * RecallTier::KnowledgeBase.weight(),
            content,
            tier: RecallTier::KnowledgeBase,
        });
    }
    Ok(items)
}

/// Truncate a per-tier result list to the configured limit before
/// merging into the global candidate pool.
fn push_results(out: &mut Vec<RecallItem>, mut tier_items: Vec<RecallItem>, limit: usize) {
    if tier_items.len() > limit {
        tier_items.truncate(limit);
    }
    out.extend(tier_items);
}

/// Build a uni-db `Value::List` of strings for IN-list parameters.
fn goal_ids_value(goal_ids: &[String]) -> Value {
    Value::List(
        goal_ids
            .iter()
            .map(|id| Value::String(id.clone()))
            .collect(),
    )
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
