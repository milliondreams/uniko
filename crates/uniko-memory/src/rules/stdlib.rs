//! Registration of the 4 stdlib Locy rules from the spec.
//!
//! These rules are exempt from demotion, pruning, and supersession.
//! They are registered as Rule nodes in the graph AND in uni-db's
//! Locy runtime.

use std::collections::HashMap;

use uniko_cortex::procedures::SEQUENCE_DETECTOR_RULE;
use uniko_store::{KnowledgeBase, UnikoError, Value};

/// Rule definitions: (rule_id, name, natural_language, locy_source).
const STDLIB_RULES: &[(&str, &str, &str, &str)] = &[
    (
        "stdlib_relevance_decay",
        "relevance_decay",
        "Decay episode relevance with age: `importance * exp(-decay_rate * \
         age_days)`, where `decay_rate = ln(2) / half_life_days`. Surfaces the \
         episodes that have decayed to or below the threshold (i.e. prunable); \
         the consumer deletes them.",
        // Locy (not Cypher): no `WITH` projections — the decay expression is
        // inlined in WHERE and YIELD; yields the prunable episode_ids.
        // `duration.inDays(a, b)` returns a Temporal; `.days` extracts the
        // integer day count (Cypher-style). Two ordering subtleties:
        //  - `inDays(datetime(), e.timestamp).days` = (timestamp - now) = a
        //    NEGATIVE age for past episodes, so the formula uses `exp(+rate *
        //    .days)` (mathematically exp(-rate * age)).
        //  - the reverse order `inDays(e.timestamp, datetime())` yields `Null`
        //    inside arithmetic (works only as a bare projection), so we use the
        //    `(datetime(), e.timestamp)` order here.
        "CREATE RULE relevance_decay AS \
         MATCH (e:Episode)-[:RECORDED_BY]->(p:Participant {participant_id: $agent_id}) \
         WHERE e.importance * exp($decay_rate * duration.inDays(datetime(), e.timestamp).days) <= $decay_threshold \
         YIELD KEY e.episode_id AS episode_id, \
               e.importance * exp($decay_rate * duration.inDays(datetime(), e.timestamp).days) AS relevance",
    ),
    (
        "stdlib_episode_pattern_detector",
        "episode_pattern_detector",
        "Detect recurring episode patterns by counting episodes of the same \
         action type and outcome. Patterns with at least 3 occurrences and \
         average importance above 0.3 are surfaced.",
        // Locy (not Cypher): single FOLD, post-FOLD WHERE is HAVING, literal
        // thresholds, `expr AS name` (no `VALUE`). Only `COUNT(*)` is used: a
        // FOLD value-aggregate (SUM/AVG) comes back 0.0 when YIELD projects it
        // under a different alias than the FOLD variable — so
        // `avg_imp = AVG(e.importance) ... YIELD ... avg_imp AS mean_importance`
        // yields 0.0 (uni-db #145, verified by minimal repro; COUNT is
        // unaffected). The mean-importance value/filter is computed in the Rust
        // consumer (`consume_episode_patterns`) instead. A same-name FOLD var
        // (`mean_importance = AVG(...) YIELD ... mean_importance AS mean_importance`)
        // dodges #145 but is fragile to a future rename; revisit once #145 lands.
        "CREATE RULE episode_pattern_detector AS \
         MATCH (e:Episode)-[:RECORDED_BY]->(p:Participant {participant_id: $agent_id}) \
         FOLD n = COUNT(*) \
         WHERE n >= 3 \
         YIELD KEY e.action_type AS action_type, KEY e.outcome AS outcome, \
               n AS support",
    ),
    (
        "stdlib_sequence_detector",
        "sequence_detector",
        "Detect recurring successful action sequences. When two actions \
         consistently follow each other with successful outcomes, surface \
         the pattern for procedural knowledge extraction.",
        // Canonical source lives in uniko-cortex (where its consumer,
        // upsert_procedure, lives) — referenced here so the rule string is
        // defined exactly once and the cortex sweep + this registration cannot
        // drift. The consumer applies the promotion threshold (Locy can't
        // resolve a $param in a post-FOLD HAVING; see RC12).
        SEQUENCE_DETECTOR_RULE,
    ),
    (
        "stdlib_contradiction_detector",
        "contradiction_detector",
        "Find episodes whose outcomes contradict established facts. When an \
         episode's outcome differs from the recorded outcome pattern, flag \
         it for fact revision.",
        // Locy (not Cypher): the two patterns are comma-joined into ONE MATCH (a
        // second MATCH clause is a parse error). Yields one row per (stale Fact,
        // contradicting Episode) pair; the consumer counts pairs per Fact
        // (applying the threshold in Rust) and draws a (:Fact)-[:CONTRADICTED_BY]
        // ->(:Episode) marker edge to each contradicting episode. No `VALUE`.
        // Validity uses `f.invalidated_at IS NULL` (a valid Fact is not yet
        // invalidated) — `btic.contains` is interval-vs-interval (both args must
        // be BTIC), not point-in-interval, so it can't take `datetime()`. The
        // FOLD-free KEY columns require uni-db >= 2.4.1 (rustic-ai/uni-db#112).
        "CREATE RULE contradiction_detector AS \
         MATCH (e:Episode)-[:RECORDED_BY]->(p:Participant {participant_id: $agent_id}), \
               (f:Fact) \
         WHERE f.subject = e.action_type \
           AND f.predicate = 'outcome_pattern' \
           AND f.invalidated_at IS NULL \
           AND e.outcome <> f.object \
         YIELD KEY f.fact_id AS stale_fact, \
               KEY e.episode_id AS episode_id",
    ),
];

/// Register the 4 stdlib Locy rules as Rule nodes in the graph.
///
/// Idempotent — uses `merge_node` with deterministic `rule_id` values.
/// Also registers the rule source in uni-db's Locy runtime for
/// execution.
///
/// # Errors
///
/// Returns [`UnikoError::Storage`] if graph operations fail, or
/// [`UnikoError::Locy`] if rule registration fails.
pub async fn register_stdlib_rules(kb: &KnowledgeBase) -> Result<(), UnikoError> {
    let now = uniko_store::types::datetime_value(chrono::Utc::now());

    for &(rule_id, name, natural_language, source) in STDLIB_RULES {
        // Create Rule node in the graph.
        let mut props = HashMap::new();
        props.insert("name".into(), Value::String(name.to_string()));
        props.insert("source".into(), Value::String(source.to_string()));
        props.insert(
            "natural_language".into(),
            Value::String(natural_language.to_string()),
        );
        props.insert("source_type".into(), Value::String("stdlib".to_string()));
        props.insert("status".into(), Value::String("active".to_string()));
        props.insert("version".into(), Value::Int(1));
        props.insert("confidence".into(), Value::Float(1.0));
        props.insert("created_at".into(), now.clone());

        kb.merge_node("Rule", "rule_id", rule_id, &props).await?;

        // Register in uni-db's Locy runtime. All 4 stdlib rules are guarded by
        // the `stdlib_rules_compile_against_unidb` test, so a failure here means a
        // real regression (uni-db Locy change) — surface it loudly rather than
        // swallow, but don't brick startup: the rule node is still recorded.
        if let Err(e) = kb.create_rule(source).await {
            tracing::warn!(
                rule = name,
                error = %e,
                "stdlib Locy rule failed to register in the runtime — it will not execute",
            );
        } else {
            tracing::info!(rule = name, "stdlib rule registered");
        }
    }

    Ok(())
}

/// Check whether a rule is a protected stdlib rule.
///
/// Stdlib rules are exempt from demotion, pruning, and supersession.
pub fn is_stdlib_rule(source_type: &str) -> bool {
    source_type == "stdlib"
}

/// Build the parameter map for the `relevance_decay` Locy rule.
///
/// Converts a half-life in days into the exponential decay rate the rule
/// expects (`decay_rate = ln(2) / half_life_days`) and packages it
/// alongside the prune threshold and the agent id.  Pass the returned
/// map to [`KnowledgeBase::execute_rule`].
///
/// # Panics
///
/// Panics if `half_life_days <= 0.0`.  Callers must validate config
/// before invoking — [`UnikoConfig::validate`] already rejects non-
/// positive half-lives.
pub fn relevance_decay_params(
    agent_id: &str,
    half_life_days: f64,
    decay_threshold: f64,
) -> HashMap<String, Value> {
    assert!(
        half_life_days > 0.0,
        "half_life_days must be positive (got {half_life_days})"
    );
    let decay_rate = std::f64::consts::LN_2 / half_life_days;
    let mut params = HashMap::new();
    params.insert("agent_id".into(), Value::String(agent_id.to_string()));
    params.insert("decay_rate".into(), Value::Float(decay_rate));
    params.insert("decay_threshold".into(), Value::Float(decay_threshold));
    params
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stdlib_rule_count() {
        // 4 registered: relevance_decay, episode_pattern_detector,
        // sequence_detector, contradiction_detector.
        assert_eq!(STDLIB_RULES.len(), 4);
    }

    #[test]
    fn test_is_stdlib() {
        assert!(is_stdlib_rule("stdlib"));
        assert!(!is_stdlib_rule("authored"));
        assert!(!is_stdlib_rule("induced"));
    }

    #[test]
    fn test_relevance_decay_rate_from_half_life() {
        // 14-day half-life → 14-day-old episode should retain half its importance.
        let params = relevance_decay_params("agent-1", 14.0, 0.05);
        let decay_rate = match params.get("decay_rate") {
            Some(Value::Float(r)) => *r,
            other => panic!("expected Float decay_rate, got {other:?}"),
        };
        // ln(2) / 14 ≈ 0.04951.
        assert!(
            (decay_rate - (std::f64::consts::LN_2 / 14.0)).abs() < 1e-12,
            "decay_rate = {decay_rate}",
        );
        // Sanity: applying the rate to 14 days gives 0.5.
        let retained = (-decay_rate * 14.0_f64).exp();
        assert!((retained - 0.5).abs() < 1e-9, "retained = {retained}");
    }

    #[test]
    fn test_relevance_decay_params_contains_agent_id_and_threshold() {
        let params = relevance_decay_params("agent-xyz", 30.0, 0.1);
        match params.get("agent_id") {
            Some(Value::String(s)) => assert_eq!(s, "agent-xyz"),
            other => panic!("expected String agent_id, got {other:?}"),
        }
        match params.get("decay_threshold") {
            Some(Value::Float(t)) => assert!((t - 0.1).abs() < 1e-12),
            other => panic!("expected Float decay_threshold, got {other:?}"),
        }
    }

    #[test]
    #[should_panic(expected = "half_life_days must be positive")]
    fn test_relevance_decay_params_rejects_zero_half_life() {
        let _ = relevance_decay_params("agent-1", 0.0, 0.05);
    }

    /// Step 0 of the stdlib-Locy migration: report which of the 4 rules actually
    /// compile/register against the live uni-db. `register_stdlib_rules` swallows
    /// `create_rule` errors at debug level, so this is the only place the truth is
    /// surfaced. Reports every rule's result, then fails if any did not compile.
    #[tokio::test]
    async fn stdlib_rules_compile_against_unidb() {
        use uniko_store::KnowledgeBase;
        use uniko_store::config::UnikoConfig;

        let kb = KnowledgeBase::in_memory(UnikoConfig::default())
            .await
            .expect("in-memory KB (schema applied)");

        let mut failures = Vec::new();
        for &(_id, name, _nl, source) in STDLIB_RULES {
            match kb.create_rule(source).await {
                Ok(()) => eprintln!("[compile] OK   {name}"),
                Err(e) => {
                    eprintln!("[compile] FAIL {name}: {e}");
                    failures.push(name);
                }
            }
        }

        assert!(
            failures.is_empty(),
            "stdlib rules failed to compile against uni-db: {failures:?}",
        );
    }
}
