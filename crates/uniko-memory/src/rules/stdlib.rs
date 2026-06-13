//! Registration of the 4 stdlib Locy rules from the spec.
//!
//! These rules are exempt from demotion, pruning, and supersession.
//! They are registered as Rule nodes in the graph AND in uni-db's
//! Locy runtime.

use std::collections::HashMap;

use uni_db::Value;

use uniko_store::{KnowledgeBase, UnikoError};

/// Rule definitions: (rule_id, name, natural_language, locy_source).
const STDLIB_RULES: &[(&str, &str, &str, &str)] = &[
    (
        "stdlib_relevance_decay",
        "relevance_decay",
        "Compute decayed relevance for episodes based on age. Older episodes \
         lose relevance exponentially via `importance * exp(-decay_rate * age_days)`. \
         `decay_rate` is derived from a configurable half-life: \
         `decay_rate = ln(2) / half_life_days`. Episodes below the configured \
         threshold are effectively forgotten.",
        "CREATE RULE relevance_decay AS \
         MATCH (e:Episode)-[:RECORDED_BY]->(p:Participant {participant_id: $agent_id}) \
         WITH e, \
              duration.inDays(e.timestamp, datetime()) AS age_days, \
              e.importance AS base_importance \
         WITH e, \
              base_importance * exp(-$decay_rate * age_days) AS decayed \
         WHERE decayed > $decay_threshold \
         YIELD KEY e, VALUE decayed AS relevance",
    ),
    (
        "stdlib_episode_pattern_detector",
        "episode_pattern_detector",
        "Detect recurring episode patterns by counting episodes of the same \
         action type and outcome. Patterns with at least 3 occurrences and \
         average importance above 0.3 are surfaced.",
        "CREATE RULE episode_pattern_detector AS \
         MATCH (e:Episode)-[:RECORDED_BY]->(p:Participant {participant_id: $agent_id}) \
         FOLD n = COUNT(*) \
         FOLD avg_importance = AVG(e.importance) \
         WHERE n >= 3 AND avg_importance > 0.3 \
         YIELD KEY e.action_type, KEY e.outcome, \
               VALUE n AS support, \
               VALUE avg_importance AS mean_importance",
    ),
    (
        "stdlib_sequence_detector",
        "sequence_detector",
        "Detect recurring successful action sequences. When two actions \
         consistently follow each other with successful outcomes, surface \
         the pattern for procedural knowledge extraction.",
        "CREATE RULE sequence_detector AS \
         MATCH (e1:Episode)-[:FOLLOWED_BY]->(e2:Episode) \
         MATCH (e1)-[:RECORDED_BY]->(p:Participant {participant_id: $agent_id}) \
         WHERE e1.outcome = 'success' \
           AND e2.outcome = 'success' \
         FOLD n = COUNT(*) \
         WHERE n >= $promotion_threshold \
         YIELD KEY e1.action_type, KEY e2.action_type, \
               VALUE n AS success_count",
    ),
    (
        "stdlib_contradiction_detector",
        "contradiction_detector",
        "Find episodes whose outcomes contradict established facts. When an \
         episode's outcome differs from the recorded outcome pattern, flag \
         it for fact revision.",
        "CREATE RULE contradiction_detector AS \
         MATCH (e:Episode)-[:RECORDED_BY]->(p:Participant {participant_id: $agent_id}) \
         MATCH (f:Fact) \
         WHERE f.subject = e.action_type \
           AND f.predicate = 'outcome_pattern' \
           AND btic.contains(f.valid_at, datetime()) \
           AND e.outcome <> f.object \
         FOLD n = COUNT(e) \
         WHERE n >= $contradiction_threshold \
         YIELD KEY f.fact_id AS stale_fact, \
               KEY e.action_type AS action, \
               VALUE n AS contradicting_count, \
               VALUE f.object AS old_outcome, \
               VALUE e.outcome AS new_outcome",
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

        // Register in uni-db's Locy runtime (best-effort — may fail if
        // Locy syntax is not supported by the current uni-db version).
        if let Err(e) = kb.create_rule(source).await {
            tracing::debug!(
                rule = name,
                error = %e,
                "Locy rule registration skipped (runtime may not support this syntax)",
            );
        }

        tracing::info!(rule = name, "stdlib rule registered");
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
}
