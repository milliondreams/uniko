//! Procedure promotion (P5) — F41/F42/F43.
//!
//! Recurring successful action/episode sequences are detected by the
//! stdlib `sequence_detector` Locy rule.  Each surfaced pair becomes
//! (or reinforces) a Procedure node with a precondition rule that
//! recall and planning can match against.
//!
//! Lifecycle (F41):
//!
//! ```text
//! candidate ──(promote_threshold reached)──▶ active
//!     │                                       │
//!     └──(stale)──▶ deprecated ◀──(degrade)───┘
//! ```
//!
//! - `candidate` — first observation; not yet eligible for use.
//! - `active` — matched the promotion threshold; surfaced to recall
//!   and planning.
//! - `deprecated` — effectiveness fell below the demotion threshold;
//!   kept for provenance but excluded from active use.
//!
//! `record_procedure_use` updates counters and applies the lifecycle
//! state machine on every call; `promote_procedures_once` runs the
//! sequence detector and creates/refreshes Procedure nodes.

// Rust guideline compliant

use std::collections::HashMap;

use serde::Serialize;
use uni_db::Value;
use uniko_store::schema::labels;
use uniko_store::types::datetime_value;
use uniko_store::{KnowledgeBase, NodeId, UnikoError};

/// Status string for a Procedure that has been observed once but
/// hasn't yet crossed the promotion threshold.
pub const STATUS_CANDIDATE: &str = "candidate";
/// Status string for a Procedure that is eligible for matching and
/// execution.
pub const STATUS_ACTIVE: &str = "active";
/// Status string for a Procedure that was demoted because its
/// effectiveness fell below the demotion threshold.
pub const STATUS_DEPRECATED: &str = "deprecated";

/// Promotion / demotion thresholds for the Procedure lifecycle.
///
/// Values come from the spec's P5 description (Part VI) and the
/// `sequence_detector` `$promotion_threshold` parameter.  Demotion is
/// asymmetric so a Procedure that briefly failed isn't immediately
/// dropped.
#[derive(Debug, Clone, Copy)]
pub struct LifecycleConfig {
    /// Sequence-count threshold for promoting a candidate to active.
    pub promote_threshold: i64,
    /// Effectiveness threshold (success / (success + failure)) below
    /// which active Procedures are demoted to deprecated.
    pub demote_effectiveness: f64,
    /// Effectiveness threshold above which deprecated Procedures
    /// return to active.  Higher than `demote_effectiveness` for
    /// hysteresis.
    pub repromote_effectiveness: f64,
}

impl Default for LifecycleConfig {
    fn default() -> Self {
        Self {
            promote_threshold: 3,
            demote_effectiveness: 0.4,
            repromote_effectiveness: 0.6,
        }
    }
}

/// Outcome of one [`promote_procedures_once`] call.
#[derive(Debug, Clone, Default, Serialize)]
pub struct PromotionReport {
    /// Number of new candidate Procedure nodes created.
    pub created: usize,
    /// Number of existing Procedures whose support_count was bumped.
    pub reinforced: usize,
    /// Number of Procedures whose status flipped to `active` this run.
    pub promoted: usize,
}

/// Detect repeating successful action/episode sequences for `agent_id`
/// and create or update Procedure nodes accordingly.
///
/// Runs the stdlib `sequence_detector` Locy rule with the configured
/// promotion threshold.  Each pair `(action_a → action_b)` becomes a
/// Procedure with `name = "{a} → {b}"`.  Idempotent: re-running merges
/// into the same Procedure by deterministic id.
///
/// # Errors
///
/// - [`UnikoError::Locy`] when the rule cannot be executed and the
///   Cypher fallback also fails.
/// - [`UnikoError::Storage`] when graph writes fail.
pub async fn promote_procedures_once(
    kb: &KnowledgeBase,
    agent_id: &str,
    cfg: LifecycleConfig,
) -> Result<PromotionReport, UnikoError> {
    // Detect every recurring pair (threshold = 1) and let
    // `upsert_procedure` classify candidates vs active based on the
    // configured promotion threshold.  Running the rule at the
    // promotion threshold would hide sequences that haven't yet
    // qualified, defeating candidate creation.
    let detection_threshold: i64 = 1;
    let mut params = HashMap::new();
    params.insert("agent_id".into(), Value::String(agent_id.to_string()));
    params.insert(
        "promotion_threshold".into(),
        Value::Int(detection_threshold),
    );

    let records = match kb.execute_rule("sequence_detector", &params).await {
        Ok(r) => r,
        Err(UnikoError::Locy(msg)) => {
            tracing::debug!(error = %msg, "sequence_detector unavailable — falling back to direct Cypher");
            fallback_sequence_query(kb, agent_id, detection_threshold).await?
        }
        Err(other) => return Err(other),
    };

    let mut report = PromotionReport::default();
    for rec in records {
        let a = pick_string(&rec, &["e1.action_type", "key_0", "a", "action_a"]);
        let b = pick_string(&rec, &["e2.action_type", "key_1", "b", "action_b"]);
        let count = pick_i64(&rec, &["success_count", "value_0", "n"]);
        let (Some(a), Some(b), Some(count)) = (a, b, count) else {
            continue;
        };

        let outcome = upsert_procedure(kb, agent_id, &a, &b, count, cfg).await?;
        match outcome {
            UpsertOutcome::Created => report.created += 1,
            UpsertOutcome::Reinforced => report.reinforced += 1,
            UpsertOutcome::Promoted => {
                report.reinforced += 1;
                report.promoted += 1;
            }
        }
    }
    Ok(report)
}

/// Bump a Procedure's use counters after a real attempt and apply the
/// lifecycle state machine (F42).
///
/// # Errors
///
/// Returns [`UnikoError::Storage`] when the Procedure is missing or
/// the update fails.
pub async fn record_procedure_use(
    kb: &KnowledgeBase,
    procedure_id: &str,
    succeeded: bool,
    cfg: LifecycleConfig,
) -> Result<(), UnikoError> {
    let proc_node = kb
        .get_node_by_ext_id(labels::PROCEDURE, "procedure_id", procedure_id)
        .await?
        .ok_or_else(|| {
            UnikoError::Storage(format!(
                "record_procedure_use: Procedure '{procedure_id}' not found"
            ))
        })?;

    let snapshot = read_procedure(kb, procedure_id).await?;
    let use_count = snapshot.use_count + 1;
    let success = snapshot.success_count + i64::from(succeeded);
    let failure = snapshot.failure_count + i64::from(!succeeded);
    let denom = (success + failure) as f64;
    let effectiveness = if denom > 0.0 {
        success as f64 / denom
    } else {
        0.0
    };

    let new_status = match snapshot.status.as_str() {
        STATUS_CANDIDATE => snapshot.status.clone(),
        STATUS_ACTIVE if effectiveness < cfg.demote_effectiveness => {
            STATUS_DEPRECATED.to_string()
        }
        STATUS_DEPRECATED if effectiveness >= cfg.repromote_effectiveness => {
            STATUS_ACTIVE.to_string()
        }
        _ => snapshot.status.clone(),
    };

    let mut props = HashMap::new();
    props.insert("use_count".into(), Value::Int(use_count));
    props.insert("success_count".into(), Value::Int(success));
    props.insert("failure_count".into(), Value::Int(failure));
    props.insert("effectiveness".into(), Value::Float(effectiveness));
    props.insert("status".into(), Value::String(new_status));
    props.insert("last_used_at".into(), datetime_value(chrono::Utc::now()));
    kb.update_node(proc_node.0, &props).await?;
    Ok(())
}

/// Find active Procedures whose `precondition_rule` matches the
/// current state (F43).
///
/// Each active Procedure stores its precondition as a Locy WHERE
/// fragment in `precondition_rule`.  For the MVP we evaluate it as a
/// comma-separated list of `key=value` clauses that must all appear in
/// `state` (see [`precondition_matches`]).  The signature is shaped to
/// accept a richer evaluator without breaking callers.
///
/// Procedures with no precondition_rule match any state.
///
/// # Errors
///
/// Returns [`UnikoError::Storage`] if the candidate query fails.
pub async fn match_procedures(
    kb: &KnowledgeBase,
    state: &HashMap<String, String>,
) -> Result<Vec<MatchedProcedure>, UnikoError> {
    let session = kb.db().session();
    let cypher = "MATCH (p:Procedure) WHERE p.status = $st \
                  RETURN p.procedure_id AS pid, p.name AS name, \
                         coalesce(p.precondition_rule, '') AS pre, \
                         coalesce(p.effectiveness, 0.0) AS eff \
                  ORDER BY eff DESC";
    let result = session
        .query_with(cypher)
        .param("st", STATUS_ACTIVE)
        .fetch_all()
        .await
        .map_err(|e| UnikoError::Storage(e.to_string()))?;

    let mut out = Vec::new();
    for row in result.rows() {
        let Ok(pid) = row.get::<String>("pid") else {
            continue;
        };
        let name: String = row.get("name").unwrap_or_default();
        let pre: String = row.get("pre").unwrap_or_default();
        let eff: f64 = row.get("eff").unwrap_or(0.0);
        if precondition_matches(&pre, state) {
            out.push(MatchedProcedure {
                procedure_id: pid,
                name,
                effectiveness: eff,
            });
        }
    }
    Ok(out)
}

/// One Procedure that matched the input state.
#[derive(Debug, Clone, Serialize)]
pub struct MatchedProcedure {
    /// External `procedure_id` for downstream tools.
    pub procedure_id: String,
    /// Human-readable name (e.g. `"investigate → implement"`).
    pub name: String,
    /// Current effectiveness score in `[0.0, 1.0]`.
    pub effectiveness: f64,
}

/// Internal: result classification for [`upsert_procedure`].
enum UpsertOutcome {
    Created,
    Reinforced,
    Promoted,
}

#[derive(Debug, Default)]
struct ProcedureSnapshot {
    use_count: i64,
    success_count: i64,
    failure_count: i64,
    status: String,
}

async fn upsert_procedure(
    kb: &KnowledgeBase,
    agent_id: &str,
    action_a: &str,
    action_b: &str,
    support_count: i64,
    cfg: LifecycleConfig,
) -> Result<UpsertOutcome, UnikoError> {
    let name = format!("{action_a} → {action_b}");
    let procedure_id = stable_procedure_id(agent_id, action_a, action_b);

    let existing = read_procedure(kb, &procedure_id).await.ok();
    let new = existing.is_none();
    let snapshot = existing.unwrap_or_default();

    let next_status = if new {
        if support_count >= cfg.promote_threshold {
            STATUS_ACTIVE
        } else {
            STATUS_CANDIDATE
        }
    } else if snapshot.status == STATUS_CANDIDATE && support_count >= cfg.promote_threshold {
        STATUS_ACTIVE
    } else if snapshot.status.is_empty() {
        STATUS_CANDIDATE
    } else {
        // Keep existing status; demotion/repromotion is owned by
        // `record_procedure_use`.
        snapshot.status.as_str()
    };
    let promoted_this_call =
        !new && snapshot.status == STATUS_CANDIDATE && next_status == STATUS_ACTIVE;

    let now = datetime_value(chrono::Utc::now());
    let success_count = snapshot.success_count.max(support_count);
    let denom = (success_count + snapshot.failure_count) as f64;
    let eff = if denom > 0.0 {
        success_count as f64 / denom
    } else {
        1.0
    };

    let mut props = HashMap::new();
    props.insert("name".into(), Value::String(name.clone()));
    props.insert(
        "description".into(),
        Value::String(format!(
            "Detected sequence: when {action_a} succeeds it is often followed by {action_b}."
        )),
    );
    props.insert(
        "precondition_rule".into(),
        Value::String(format!("last_action_type={action_a}")),
    );
    props.insert("status".into(), Value::String(next_status.to_string()));
    props.insert(
        "use_count".into(),
        Value::Int(snapshot.use_count.max(support_count)),
    );
    props.insert("success_count".into(), Value::Int(success_count));
    props.insert("failure_count".into(), Value::Int(snapshot.failure_count));
    props.insert("effectiveness".into(), Value::Float(eff));
    if new {
        props.insert("created_at".into(), now.clone());
    }
    props.insert("last_used_at".into(), now);

    kb.merge_node(labels::PROCEDURE, "procedure_id", &procedure_id, &props)
        .await?;

    if new {
        Ok(UpsertOutcome::Created)
    } else if promoted_this_call {
        Ok(UpsertOutcome::Promoted)
    } else {
        Ok(UpsertOutcome::Reinforced)
    }
}

async fn read_procedure(
    kb: &KnowledgeBase,
    procedure_id: &str,
) -> Result<ProcedureSnapshot, UnikoError> {
    let session = kb.db().session();
    let cypher = "MATCH (p:Procedure) WHERE p.procedure_id = $pid \
                  RETURN coalesce(p.use_count, 0) AS uc, \
                         coalesce(p.success_count, 0) AS sc, \
                         coalesce(p.failure_count, 0) AS fc, \
                         coalesce(p.status, '') AS st";
    let result = session
        .query_with(cypher)
        .param("pid", procedure_id)
        .fetch_all()
        .await
        .map_err(|e| UnikoError::Storage(e.to_string()))?;
    let row = result.rows().first().ok_or_else(|| {
        UnikoError::Storage(format!("procedure '{procedure_id}' not found"))
    })?;
    Ok(ProcedureSnapshot {
        use_count: row.get("uc").unwrap_or(0),
        success_count: row.get("sc").unwrap_or(0),
        failure_count: row.get("fc").unwrap_or(0),
        status: row.get("st").unwrap_or_default(),
    })
}

/// Fallback when the Locy runtime can't execute the stdlib rule.
async fn fallback_sequence_query(
    kb: &KnowledgeBase,
    agent_id: &str,
    threshold: i64,
) -> Result<Vec<HashMap<String, Value>>, UnikoError> {
    let session = kb.db().session();
    let cypher = "MATCH (e1:Episode)-[:FOLLOWED_BY]->(e2:Episode) \
                  MATCH (e1)-[:RECORDED_BY]->(p:Participant {participant_id: $aid}) \
                  WHERE e1.outcome = 'success' AND e2.outcome = 'success' \
                  WITH e1.action_type AS a, e2.action_type AS b, count(*) AS n \
                  WHERE n >= $thr \
                  RETURN a, b, n";
    let result = session
        .query_with(cypher)
        .param("aid", agent_id)
        .param("thr", threshold)
        .fetch_all()
        .await
        .map_err(|e| UnikoError::Storage(e.to_string()))?;

    let mut out = Vec::new();
    for row in result.rows() {
        let a: String = row.get("a").unwrap_or_default();
        let b: String = row.get("b").unwrap_or_default();
        let n: i64 = row.get("n").unwrap_or(0);
        let mut rec = HashMap::new();
        rec.insert("a".into(), Value::String(a));
        rec.insert("b".into(), Value::String(b));
        rec.insert("n".into(), Value::Int(n));
        out.push(rec);
    }
    Ok(out)
}

fn pick_string(rec: &HashMap<String, Value>, keys: &[&str]) -> Option<String> {
    for k in keys {
        if let Some(Value::String(s)) = rec.get(*k) {
            return Some(s.clone());
        }
    }
    None
}

fn pick_i64(rec: &HashMap<String, Value>, keys: &[&str]) -> Option<i64> {
    for k in keys {
        match rec.get(*k) {
            Some(Value::Int(i)) => return Some(*i),
            Some(Value::Float(f)) => return Some(*f as i64),
            _ => continue,
        }
    }
    None
}

/// Build a deterministic procedure_id for the (agent, action_a, action_b) triple.
fn stable_procedure_id(agent_id: &str, a: &str, b: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    agent_id.hash(&mut h);
    a.hash(&mut h);
    b.hash(&mut h);
    format!("proc_{:016x}", h.finish())
}

/// MVP precondition matcher: each clause is `key=value`, multiple
/// clauses joined by `,`.  Matches when every clause finds its key in
/// `state` with the exact value.  An empty precondition matches any
/// state.
fn precondition_matches(rule: &str, state: &HashMap<String, String>) -> bool {
    let rule = rule.trim();
    if rule.is_empty() {
        return true;
    }
    for clause in rule.split(',') {
        let clause = clause.trim();
        let Some((k, v)) = clause.split_once('=') else {
            return false;
        };
        let k = k.trim();
        let v = v.trim();
        match state.get(k) {
            Some(actual) if actual == v => continue,
            _ => return false,
        }
    }
    true
}

/// Aliases for callers building procedures outside of the promotion
/// path.
pub type ProcedureNodeId = NodeId;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lifecycle_defaults_are_spec_aligned() {
        let c = LifecycleConfig::default();
        assert_eq!(c.promote_threshold, 3);
        assert!((c.demote_effectiveness - 0.4).abs() < 1e-9);
        assert!((c.repromote_effectiveness - 0.6).abs() < 1e-9);
    }

    #[test]
    fn precondition_empty_matches() {
        let s = HashMap::new();
        assert!(precondition_matches("", &s));
        assert!(precondition_matches("   ", &s));
    }

    #[test]
    fn precondition_single_clause() {
        let mut s = HashMap::new();
        s.insert("last_action_type".to_string(), "build".to_string());
        assert!(precondition_matches("last_action_type=build", &s));
        assert!(!precondition_matches("last_action_type=test", &s));
    }

    #[test]
    fn precondition_multi_clause_all_must_match() {
        let mut s = HashMap::new();
        s.insert("a".into(), "1".into());
        s.insert("b".into(), "2".into());
        assert!(precondition_matches("a=1,b=2", &s));
        assert!(!precondition_matches("a=1,b=3", &s));
    }

    #[test]
    fn precondition_malformed_clause_fails_closed() {
        let s = HashMap::new();
        assert!(!precondition_matches("just_a_key", &s));
    }

    #[test]
    fn stable_procedure_id_is_deterministic() {
        let a = stable_procedure_id("agent-1", "build", "test");
        let b = stable_procedure_id("agent-1", "build", "test");
        let c = stable_procedure_id("agent-1", "test", "build");
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
