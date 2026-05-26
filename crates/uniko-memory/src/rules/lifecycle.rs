//! Rule lifecycle management (F44, F52).
//!
//! Authored and induced Locy rules go through a confidence-driven
//! state machine:
//!
//! ```text
//!   add_rule()             validate              decay below 0.40
//!     │                       │                       │
//!     ▼                       ▼                       ▼
//!  candidate ──(success)──▶ active ──(repromote ≥ 0.60)──▶ active
//!                            │                       │
//!                            └──(no match 90d)──▶ pruned
//! ```
//!
//! Stdlib rules (`source_type = "stdlib"`) are exempt from
//! decay, demotion, and pruning per spec §IX-B.
//!
//! Confidence decays exponentially with missed cycles:
//! `confidence' = confidence * 0.95^missed_cycles`.  A cycle is
//! "missed" when the rule didn't bind any matches in that
//! consolidation pass; recording a match resets the counter.

// Rust guideline compliant

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::Serialize;
use uni_db::Value;

use uniko_store::id::new_id;
use uniko_store::schema::labels;
use uniko_store::types::datetime_value;
use uniko_store::{KnowledgeBase, NodeId, UnikoError};

use super::stdlib::is_stdlib_rule;

/// Status: brand-new authored or induced rule, awaiting evaluation.
pub const STATUS_CANDIDATE: &str = "candidate";
/// Status: validated and in active use.
pub const STATUS_ACTIVE: &str = "active";
/// Status: demoted due to low confidence.
pub const STATUS_DEMOTED: &str = "demoted";
/// Status: terminal — removed from the Locy runtime; node kept for
/// provenance.
pub const STATUS_PRUNED: &str = "pruned";
/// Status: terminal — replaced by a successor rule via SUPERSEDES.
pub const STATUS_SUPERSEDED: &str = "superseded";

/// Tunable thresholds for the rule lifecycle.
///
/// Defaults match spec §IX-B.  Per-deployment overrides are
/// constructed by callers and threaded through
/// [`apply_decay_cycle`] / [`record_rule_match`].
#[derive(Debug, Clone, Copy)]
pub struct RuleLifecycleConfig {
    /// Per-cycle decay multiplier (stored_confidence * decay_per_cycle).
    pub decay_per_cycle: f64,
    /// Confidence threshold below which active rules are demoted.
    pub demote_threshold: f64,
    /// Confidence threshold above which demoted rules return to active.
    pub repromote_threshold: f64,
    /// Days without any match before a demoted rule is pruned.
    pub prune_after_days: i64,
}

impl Default for RuleLifecycleConfig {
    fn default() -> Self {
        Self {
            decay_per_cycle: 0.95,
            demote_threshold: 0.40,
            repromote_threshold: 0.60,
            prune_after_days: 90,
        }
    }
}

/// Inputs for [`add_rule`].
#[derive(Debug, Clone, Default)]
pub struct AddRuleParams {
    /// Optional pre-assigned id.  When `None`, UUID v7 is generated.
    pub rule_id: Option<String>,
    /// Display name — must be unique within the agent scope.
    pub name: String,
    /// Locy source code (the body of `CREATE RULE name AS ...`).
    pub source: String,
    /// Optional natural-language description for diagnostics / NL-to-
    /// Cypher prompts.
    pub natural_language: Option<String>,
    /// `"authored"` or `"induced"`.  Stdlib rules are registered via
    /// [`super::register_stdlib_rules`], not [`add_rule`].
    pub source_type: String,
    /// Initial confidence.  Defaults to 0.5 — the rule has to earn
    /// promotion via match success.
    pub initial_confidence: Option<f64>,
}

/// Register a new authored / induced Rule.
///
/// Persists a Rule node in `status=candidate`, registers the Locy
/// source with uni-db's runtime, and returns the new NodeId.  When
/// the Locy runtime rejects the source (syntax error), the call
/// returns [`UnikoError::Locy`] *and* leaves no Rule node behind —
/// callers don't have to clean up failed adds.
///
/// # Errors
///
/// - [`UnikoError::Storage`] when the node merge fails.
/// - [`UnikoError::Locy`] when the source has syntax errors.
pub async fn add_rule(
    kb: &KnowledgeBase,
    params: AddRuleParams,
) -> Result<NodeId, UnikoError> {
    if params.source_type == "stdlib" {
        return Err(UnikoError::Storage(
            "add_rule: source_type='stdlib' is reserved for stdlib registration".into(),
        ));
    }
    if params.name.trim().is_empty() {
        return Err(UnikoError::Storage("add_rule: name is required".into()));
    }
    // Validate Locy source up-front so a bad rule never creates a node.
    kb.create_rule(&params.source)?;

    let rule_id = params.rule_id.unwrap_or_else(new_id);
    let confidence = params.initial_confidence.unwrap_or(0.5);
    let now = datetime_value(Utc::now());

    let mut props = HashMap::new();
    props.insert("name".into(), Value::String(params.name));
    props.insert("source".into(), Value::String(params.source));
    if let Some(nl) = params.natural_language {
        props.insert("natural_language".into(), Value::String(nl));
    }
    props.insert("source_type".into(), Value::String(params.source_type));
    props.insert("status".into(), Value::String(STATUS_CANDIDATE.into()));
    props.insert("version".into(), Value::Int(1));
    props.insert("confidence".into(), Value::Float(confidence));
    props.insert("created_at".into(), now);
    props.insert("missed_cycles".into(), Value::Int(0));

    kb.merge_node(labels::RULE, "rule_id", &rule_id, &props).await
}

/// Apply one decay-cycle pass to every non-stdlib Rule.
///
/// For each non-stdlib Rule: multiplies confidence by
/// `decay_per_cycle`, bumps `missed_cycles`, and applies the
/// demotion / pruning state transitions when thresholds are crossed.
/// Returns a [`DecayReport`] summarising the effect.
///
/// Callers should invoke this once per consolidation cycle.
/// [`record_rule_match`] resets `missed_cycles` for individual rules
/// that bound something during the cycle.
///
/// # Errors
///
/// Returns [`UnikoError::Storage`] when the rule scan or update
/// fails.
pub async fn apply_decay_cycle(
    kb: &KnowledgeBase,
    cfg: RuleLifecycleConfig,
) -> Result<DecayReport, UnikoError> {
    let rules = fetch_lifecycle_rules(kb).await?;
    let mut report = DecayReport::default();
    let now = Utc::now();

    for r in rules {
        if is_stdlib_rule(&r.source_type) {
            continue;
        }
        let new_confidence = (r.confidence * cfg.decay_per_cycle).max(0.0);
        let new_missed_cycles = r.missed_cycles + 1;

        let mut new_status = r.status.clone();
        let demote = r.status == STATUS_ACTIVE && new_confidence < cfg.demote_threshold;
        let repromote = r.status == STATUS_DEMOTED && new_confidence >= cfg.repromote_threshold;
        let promote_candidate =
            r.status == STATUS_CANDIDATE && new_confidence >= cfg.repromote_threshold;
        let prune = (r.status == STATUS_DEMOTED || r.status == STATUS_CANDIDATE)
            && days_since(&r.last_scored_at, now) >= cfg.prune_after_days;

        if prune {
            new_status = STATUS_PRUNED.into();
            // Best-effort remove from the runtime; the node stays for
            // provenance.
            if let Err(e) = kb.delete_rule(&r.name) {
                tracing::debug!(name = %r.name, error = %e, "delete_rule during prune failed");
            }
            report.pruned += 1;
        } else if demote {
            new_status = STATUS_DEMOTED.into();
            report.demoted += 1;
        } else if repromote || promote_candidate {
            new_status = STATUS_ACTIVE.into();
            report.promoted += 1;
        } else {
            report.decayed += 1;
        }

        let mut props = HashMap::new();
        props.insert("confidence".into(), Value::Float(new_confidence));
        props.insert("missed_cycles".into(), Value::Int(new_missed_cycles));
        props.insert("status".into(), Value::String(new_status));
        kb.update_node(r.node_id, &props).await?;
    }
    Ok(report)
}

/// Record that a rule bound at least one match during the current cycle.
///
/// Resets `missed_cycles` to zero, bumps `last_scored_at`, and adds a
/// small confidence reward (`+0.05` capped at 1.0) so repeatedly
/// useful rules don't decay below the demotion threshold.  When the
/// rule is in `candidate` and the boosted confidence crosses the
/// repromote threshold, the status flips to active immediately.
///
/// # Errors
///
/// Returns [`UnikoError::Storage`] when the lookup or update fails.
pub async fn record_rule_match(
    kb: &KnowledgeBase,
    name: &str,
    cfg: RuleLifecycleConfig,
) -> Result<(), UnikoError> {
    let rule = fetch_rule_by_name(kb, name).await?;
    if is_stdlib_rule(&rule.source_type) {
        // Stdlib confidence is fixed at 1.0; no state changes.
        return Ok(());
    }
    let boost = 0.05;
    let new_confidence = (rule.confidence + boost).min(1.0);
    let new_status = if matches!(rule.status.as_str(), STATUS_CANDIDATE | STATUS_DEMOTED)
        && new_confidence >= cfg.repromote_threshold
    {
        STATUS_ACTIVE.into()
    } else {
        rule.status
    };

    let mut props = HashMap::new();
    props.insert("confidence".into(), Value::Float(new_confidence));
    props.insert("missed_cycles".into(), Value::Int(0));
    props.insert("status".into(), Value::String(new_status));
    props.insert("last_scored_at".into(), datetime_value(Utc::now()));
    kb.update_node(rule.node_id, &props).await?;
    Ok(())
}

/// Report from one [`apply_decay_cycle`] call.
#[derive(Debug, Clone, Default, Serialize)]
pub struct DecayReport {
    /// Rules whose confidence decayed but status didn't change.
    pub decayed: usize,
    /// Rules transitioned from `active` to `demoted` this cycle.
    pub demoted: usize,
    /// Rules transitioned from `candidate` or `demoted` to `active`.
    pub promoted: usize,
    /// Rules transitioned to terminal `pruned`.
    pub pruned: usize,
}

/// Snapshot of a Rule node loaded for lifecycle work.
#[derive(Debug, Clone)]
struct RuleSnapshot {
    node_id: NodeId,
    name: String,
    source_type: String,
    status: String,
    confidence: f64,
    missed_cycles: i64,
    last_scored_at: Option<DateTime<Utc>>,
}

async fn fetch_lifecycle_rules(kb: &KnowledgeBase) -> Result<Vec<RuleSnapshot>, UnikoError> {
    let session = kb.db().session();
    let cypher = "MATCH (r:Rule) \
                  RETURN id(r) AS nid, r.name AS name, \
                         coalesce(r.source_type, 'authored') AS st, \
                         coalesce(r.status, 'candidate') AS status, \
                         coalesce(r.confidence, 0.5) AS conf, \
                         coalesce(r.missed_cycles, 0) AS mc, \
                         r.last_scored_at AS lsa";
    let result = session
        .query_with(cypher)
        .fetch_all()
        .await
        .map_err(|e| UnikoError::Storage(e.to_string()))?;

    let mut out = Vec::new();
    for row in result.rows() {
        let Ok(nid) = row.get::<i64>("nid") else {
            continue;
        };
        let name: String = row.get("name").unwrap_or_default();
        let source_type: String = row.get("st").unwrap_or_else(|_| "authored".into());
        let status: String = row.get("status").unwrap_or_else(|_| STATUS_CANDIDATE.into());
        let confidence: f64 = row.get("conf").unwrap_or(0.5);
        let missed_cycles: i64 = row.get("mc").unwrap_or(0);
        let last_scored_at = extract_optional_dt(row, "lsa");
        out.push(RuleSnapshot {
            node_id: nid,
            name,
            source_type,
            status,
            confidence,
            missed_cycles,
            last_scored_at,
        });
    }
    Ok(out)
}

async fn fetch_rule_by_name(
    kb: &KnowledgeBase,
    name: &str,
) -> Result<RuleSnapshot, UnikoError> {
    let session = kb.db().session();
    let cypher = "MATCH (r:Rule) WHERE r.name = $n \
                  RETURN id(r) AS nid, r.name AS name, \
                         coalesce(r.source_type, 'authored') AS st, \
                         coalesce(r.status, 'candidate') AS status, \
                         coalesce(r.confidence, 0.5) AS conf, \
                         coalesce(r.missed_cycles, 0) AS mc, \
                         r.last_scored_at AS lsa \
                  LIMIT 1";
    let result = session
        .query_with(cypher)
        .param("n", name)
        .fetch_all()
        .await
        .map_err(|e| UnikoError::Storage(e.to_string()))?;
    let row = result
        .rows()
        .first()
        .ok_or_else(|| UnikoError::Storage(format!("rule '{name}' not found")))?;
    Ok(RuleSnapshot {
        node_id: row.get("nid").map_err(|e| UnikoError::Storage(e.to_string()))?,
        name: row.get("name").unwrap_or_default(),
        source_type: row.get("st").unwrap_or_else(|_| "authored".into()),
        status: row.get("status").unwrap_or_else(|_| STATUS_CANDIDATE.into()),
        confidence: row.get("conf").unwrap_or(0.5),
        missed_cycles: row.get("mc").unwrap_or(0),
        last_scored_at: extract_optional_dt(row, "lsa"),
    })
}

/// Pull a column out of a uni-db row and parse it as a UTC datetime.
///
/// uni-db serializes DateTime properties as `Value::Temporal`, but
/// `Row::get::<String>` can't reach a Temporal directly and
/// `toString()` is not yet implemented for temporals.  We work around
/// this by indexing the row's values directly and matching on the
/// underlying `Value` variant.
fn extract_optional_dt(row: &uni_db::Row, column: &str) -> Option<DateTime<Utc>> {
    let idx = row.columns().iter().position(|c| c == column)?;
    let value = row.values().get(idx)?;
    match value {
        uni_db::Value::Temporal(t) => {
            let millis = t.epoch_millis()?;
            DateTime::<Utc>::from_timestamp_millis(millis)
        }
        uni_db::Value::String(s) if !s.is_empty() => DateTime::parse_from_rfc3339(s)
            .ok()
            .map(|d| d.with_timezone(&Utc)),
        _ => None,
    }
}

fn days_since(last: &Option<DateTime<Utc>>, now: DateTime<Utc>) -> i64 {
    // No scoring history yet: report 0 so the prune timer only
    // starts after the first match.  A rule that's never matched but
    // never decayed below the demote threshold has zero evidence of
    // staleness; we let `missed_cycles`-driven demotion run its
    // course first, and prune only after the rule has been used at
    // least once.
    let Some(t) = last else {
        return 0;
    };
    now.signed_duration_since(*t).num_days()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_spec() {
        let c = RuleLifecycleConfig::default();
        assert!((c.decay_per_cycle - 0.95).abs() < 1e-9);
        assert!((c.demote_threshold - 0.40).abs() < 1e-9);
        assert!((c.repromote_threshold - 0.60).abs() < 1e-9);
        assert_eq!(c.prune_after_days, 90);
    }

    #[test]
    fn days_since_returns_zero_when_never_scored() {
        let now = Utc::now();
        assert_eq!(days_since(&None, now), 0);
    }

    #[test]
    fn days_since_returns_positive() {
        let now = Utc::now();
        let then = Some(now - chrono::Duration::days(10));
        assert_eq!(days_since(&then, now), 10);
    }
}
