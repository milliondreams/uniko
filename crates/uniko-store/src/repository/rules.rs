//! Rule-lifecycle reads: load Rule snapshots for scoring/decay. Backs
//! `uniko_memory::rules::lifecycle`.

use chrono::{DateTime, Utc};

use crate::error::{Result, UnikoError};
use crate::storage::KnowledgeBase;
use crate::types::{NodeId, optional_datetime_from_row};

/// Snapshot of a `:Rule` node loaded for lifecycle work.
#[derive(Debug, Clone)]
pub struct RuleSnapshot {
    /// uni-db node id of the Rule.
    pub node_id: NodeId,
    /// Rule name (display / dedup key).
    pub name: String,
    /// `authored` | `derived` provenance (defaults to `authored`).
    pub source_type: String,
    /// Lifecycle status (defaults to `candidate`).
    pub status: String,
    /// Decayed confidence in `[0, 1]` (defaults to `0.5`).
    pub confidence: f64,
    /// Consecutive cycles with no match (defaults to `0`).
    pub missed_cycles: i64,
    /// Last time the rule was scored, if ever.
    pub last_scored_at: Option<DateTime<Utc>>,
}

/// Standard `RETURN` clause for Rule snapshots — kept in one place so
/// [`fetch_lifecycle_rules`](KnowledgeBase::fetch_lifecycle_rules) and
/// [`fetch_rule_by_name`](KnowledgeBase::fetch_rule_by_name) always
/// project the same columns and `decode_rule_snapshot` can decode them
/// uniformly.
const RULE_SNAPSHOT_RETURN: &str = "RETURN id(r) AS nid, r.name AS name, \
                                    coalesce(r.source_type, 'authored') AS st, \
                                    coalesce(r.status, 'candidate') AS status, \
                                    coalesce(r.confidence, 0.5) AS conf, \
                                    coalesce(r.missed_cycles, 0) AS mc, \
                                    r.last_scored_at AS lsa";

impl KnowledgeBase {
    /// Load snapshots of every `:Rule` node.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] when the query fails.
    pub async fn fetch_lifecycle_rules(&self) -> Result<Vec<RuleSnapshot>> {
        let session = self.db.session();
        let cypher = format!("MATCH (r:Rule) {RULE_SNAPSHOT_RETURN}");
        let result = session.query_with(&cypher).fetch_all().await?;
        Ok(result
            .rows()
            .iter()
            .filter_map(decode_rule_snapshot)
            .collect())
    }

    /// Load the snapshot of the `:Rule` named `name`.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] when the rule is absent, missing
    /// its `nid`, or the query fails.
    pub async fn fetch_rule_by_name(&self, name: &str) -> Result<RuleSnapshot> {
        let session = self.db.session();
        let cypher = format!("MATCH (r:Rule) WHERE r.name = $n {RULE_SNAPSHOT_RETURN} LIMIT 1");
        let result = session
            .query_with(&cypher)
            .param("n", name)
            .fetch_all()
            .await?;
        let row = result
            .rows()
            .first()
            .ok_or_else(|| UnikoError::Storage(format!("rule '{name}' not found")))?;
        decode_rule_snapshot(row)
            .ok_or_else(|| UnikoError::Storage(format!("rule '{name}' missing nid")))
    }
}

/// Decode a Rule row produced by [`RULE_SNAPSHOT_RETURN`] into a
/// [`RuleSnapshot`]. Returns `None` when `nid` is unreadable — callers
/// either skip or surface a missing-rule error. Defaults mirror the
/// `coalesce(...)` literals in the projection.
fn decode_rule_snapshot(row: &uni_db::Row) -> Option<RuleSnapshot> {
    let nid: i64 = row.get("nid").ok()?;
    Some(RuleSnapshot {
        node_id: nid,
        name: row.get("name").unwrap_or_default(),
        source_type: row.get("st").unwrap_or_else(|_| "authored".into()),
        status: row.get("status").unwrap_or_else(|_| "candidate".into()),
        confidence: row.get("conf").unwrap_or(0.5),
        missed_cycles: row.get("mc").unwrap_or(0),
        last_scored_at: optional_datetime_from_row(row, "lsa"),
    })
}
