//! Working-memory traversal reads (F13): goal-scope resolution plus the
//! six per-tier candidate queries. Backs `uniko_memory::working_memory`.
//!
//! These methods return raw decoded rows — the per-tier *scoring* (tier
//! weights, recency/status boosts) is memory-domain logic and stays in
//! the caller, which maps these rows into its `RecallItem`s.

use uni_db::Value;

use crate::error::Result;
use crate::schema::labels;
use crate::storage::KnowledgeBase;
use crate::types::NodeId;

/// Maximum candidate rows fetched per category before the caller ranks.
const CANDIDATES_PER_CATEGORY: i64 = 50;

/// Cypher fragment matching any Session bound to an in-scope Goal,
/// either directly (`FOR_GOAL`) or via a Task (`FOR_TASK → PART_OF`).
///
/// Variable `s` must be bound to the candidate Session and the parameter
/// `$ids` to a `List<String>` of goal ids. Inlined into every Session-
/// rooted fetcher so they share the same scope semantics.
const SESSION_IN_GOAL_SCOPE: &str = "(EXISTS { (s)-[:FOR_GOAL]->(g:Goal) WHERE g.goal_id IN $ids } \
                                    OR EXISTS { (s)-[:FOR_TASK]->(:Task)-[:PART_OF]->(g:Goal) WHERE g.goal_id IN $ids })";

/// A Goal node in working-memory scope.
#[derive(Debug, Clone)]
pub struct WmGoalRow {
    pub node_id: NodeId,
    pub content: String,
    pub status: String,
}

/// A Task under an in-scope goal.
#[derive(Debug, Clone)]
pub struct WmTaskRow {
    pub node_id: NodeId,
    pub content: String,
    pub priority: f64,
    pub status: String,
}

/// A Session in scope. Rows are returned newest-first (`started_at`
/// DESC) so the caller's ordinal-recency boost is positional.
#[derive(Debug, Clone)]
pub struct WmSessionRow {
    pub node_id: NodeId,
    pub content: String,
}

/// A Message in scope. Rows are returned newest-first (`timestamp` DESC).
#[derive(Debug, Clone)]
pub struct WmMessageRow {
    pub node_id: NodeId,
    pub content: String,
}

/// A Fact anchored to in-scope messages via entity mentions.
#[derive(Debug, Clone)]
pub struct WmFactRow {
    pub node_id: NodeId,
    pub content: String,
    pub confidence: f64,
}

/// An Entity mentioned by in-scope messages. Rows are returned highest-
/// frequency first so the caller's ordinal-salience boost is positional.
#[derive(Debug, Clone)]
pub struct WmEntityRow {
    pub node_id: NodeId,
    pub content: String,
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

impl KnowledgeBase {
    /// Resolve the goal scope: the input goal plus, if `include_subgoals`,
    /// every descendant reachable via `PARENT_GOAL*1..5`.
    ///
    /// Returns an empty vec when the goal does not exist (so callers can
    /// poll while a goal is being created), and `[goal_id]` when the goal
    /// exists but has no descendants.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`](crate::UnikoError::Storage) when a
    /// lookup fails.
    pub async fn goal_scope_ids(
        &self,
        goal_id: &str,
        include_subgoals: bool,
    ) -> Result<Vec<String>> {
        let exists = self
            .get_node_by_ext_id(labels::GOAL, "goal_id", goal_id)
            .await?
            .is_some();
        if !exists {
            return Ok(Vec::new());
        }
        if !include_subgoals {
            return Ok(vec![goal_id.to_string()]);
        }

        let session = self.db.session();
        let cypher = "MATCH (g:Goal {goal_id: $gid}) \
                      OPTIONAL MATCH (sub:Goal)-[:PARENT_GOAL*1..5]->(g) \
                      WITH collect(DISTINCT g.goal_id) + collect(DISTINCT sub.goal_id) AS ids \
                      UNWIND ids AS gid \
                      WITH gid WHERE gid IS NOT NULL \
                      RETURN DISTINCT gid AS gid";
        let result = session
            .query_with(cypher)
            .param("gid", goal_id)
            .fetch_all()
            .await?;

        let ids: Vec<String> = result
            .rows()
            .iter()
            .filter_map(|row| row.get::<String>("gid").ok())
            .collect();
        if ids.is_empty() {
            Ok(vec![goal_id.to_string()])
        } else {
            Ok(ids)
        }
    }

    /// Fetch the in-scope Goal nodes (small: ≤ PARENT_GOAL depth).
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`](crate::UnikoError::Storage) on
    /// query failure.
    pub async fn fetch_wm_goals(&self, goal_ids: &[String]) -> Result<Vec<WmGoalRow>> {
        let session = self.db.session();
        let cypher = "MATCH (g:Goal) WHERE g.goal_id IN $ids \
                      RETURN id(g) AS nid, \
                             coalesce(g.title, g.goal_id) AS content, \
                             coalesce(g.status, 'active') AS status";
        let result = session
            .query_with(cypher)
            .param("ids", goal_ids_value(goal_ids))
            .fetch_all()
            .await?;
        let mut rows = Vec::new();
        for row in result.rows() {
            let Ok(nid) = row.get::<i64>("nid") else {
                continue;
            };
            rows.push(WmGoalRow {
                node_id: nid,
                content: row.get("content").unwrap_or_default(),
                status: row.get("status").unwrap_or_default(),
            });
        }
        Ok(rows)
    }

    /// Fetch Tasks that are part of any in-scope goal.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`](crate::UnikoError::Storage) on
    /// query failure.
    pub async fn fetch_wm_tasks(&self, goal_ids: &[String]) -> Result<Vec<WmTaskRow>> {
        let session = self.db.session();
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
        let mut rows = Vec::new();
        for row in result.rows() {
            let Ok(nid) = row.get::<i64>("nid") else {
                continue;
            };
            rows.push(WmTaskRow {
                node_id: nid,
                content: row.get("content").unwrap_or_default(),
                priority: row.get("prio").unwrap_or(0.5),
                status: row.get("status").unwrap_or_default(),
            });
        }
        Ok(rows)
    }

    /// Fetch Sessions linked to any in-scope goal or its tasks, newest
    /// first.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`](crate::UnikoError::Storage) on
    /// query failure.
    pub async fn fetch_wm_sessions(&self, goal_ids: &[String]) -> Result<Vec<WmSessionRow>> {
        let session = self.db.session();
        let cypher = format!(
            "MATCH (s:Session) WHERE {SESSION_IN_GOAL_SCOPE} \
             RETURN id(s) AS nid, \
                    coalesce(s.topic, s.session_id) AS content, \
                    s.started_at AS started_at \
             ORDER BY s.started_at DESC \
             LIMIT $lim"
        );
        let result = session
            .query_with(&cypher)
            .param("ids", goal_ids_value(goal_ids))
            .param("lim", CANDIDATES_PER_CATEGORY)
            .fetch_all()
            .await?;
        let mut rows = Vec::new();
        for row in result.rows() {
            let Ok(nid) = row.get::<i64>("nid") else {
                continue;
            };
            rows.push(WmSessionRow {
                node_id: nid,
                content: row.get("content").unwrap_or_default(),
            });
        }
        Ok(rows)
    }

    /// Fetch Messages from any in-scope session, newest first.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`](crate::UnikoError::Storage) on
    /// query failure.
    pub async fn fetch_wm_messages(&self, goal_ids: &[String]) -> Result<Vec<WmMessageRow>> {
        let session = self.db.session();
        let cypher = format!(
            "MATCH (m:Message)-[:IN_SESSION]->(s:Session) \
             WHERE {SESSION_IN_GOAL_SCOPE} \
             RETURN id(m) AS nid, \
                    m.content AS content \
             ORDER BY m.timestamp DESC \
             LIMIT $lim"
        );
        let result = session
            .query_with(&cypher)
            .param("ids", goal_ids_value(goal_ids))
            .param("lim", CANDIDATES_PER_CATEGORY)
            .fetch_all()
            .await?;
        let mut rows = Vec::new();
        for row in result.rows() {
            let Ok(nid) = row.get::<i64>("nid") else {
                continue;
            };
            rows.push(WmMessageRow {
                node_id: nid,
                content: row.get("content").unwrap_or_default(),
            });
        }
        Ok(rows)
    }

    /// Fetch Facts whose subject is mentioned in any in-scope message.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`](crate::UnikoError::Storage) on
    /// query failure.
    pub async fn fetch_wm_facts(&self, goal_ids: &[String]) -> Result<Vec<WmFactRow>> {
        let session = self.db.session();
        let cypher = format!(
            "MATCH (m:Message)-[:IN_SESSION]->(s:Session) \
             WHERE {SESSION_IN_GOAL_SCOPE} \
             MATCH (m)-[:MENTIONS]->(ent:Entity) \
             MATCH (f:Fact) WHERE f.subject = ent.name \
             RETURN DISTINCT id(f) AS nid, \
                    coalesce(f.object, f.subject) AS content, \
                    coalesce(f.confidence, 0.5) AS conf \
             ORDER BY conf DESC \
             LIMIT $lim"
        );
        let result = session
            .query_with(&cypher)
            .param("ids", goal_ids_value(goal_ids))
            .param("lim", CANDIDATES_PER_CATEGORY)
            .fetch_all()
            .await?;
        let mut rows = Vec::new();
        for row in result.rows() {
            let Ok(nid) = row.get::<i64>("nid") else {
                continue;
            };
            rows.push(WmFactRow {
                node_id: nid,
                content: row.get("content").unwrap_or_default(),
                confidence: row.get("conf").unwrap_or(0.5),
            });
        }
        Ok(rows)
    }

    /// Fetch Entities mentioned in any in-scope message, highest-
    /// frequency first.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`](crate::UnikoError::Storage) on
    /// query failure.
    pub async fn fetch_wm_entities(&self, goal_ids: &[String]) -> Result<Vec<WmEntityRow>> {
        let session = self.db.session();
        let cypher = format!(
            "MATCH (m:Message)-[:IN_SESSION]->(s:Session) \
             WHERE {SESSION_IN_GOAL_SCOPE} \
             MATCH (m)-[r:MENTIONS]->(ent:Entity) \
             RETURN DISTINCT id(ent) AS nid, \
                    ent.name AS content, \
                    coalesce(ent.frequency, 1) AS freq \
             ORDER BY freq DESC \
             LIMIT $lim"
        );
        let result = session
            .query_with(&cypher)
            .param("ids", goal_ids_value(goal_ids))
            .param("lim", CANDIDATES_PER_CATEGORY)
            .fetch_all()
            .await?;
        let mut rows = Vec::new();
        for row in result.rows() {
            let Ok(nid) = row.get::<i64>("nid") else {
                continue;
            };
            rows.push(WmEntityRow {
                node_id: nid,
                content: row.get("content").unwrap_or_default(),
            });
        }
        Ok(rows)
    }
}
