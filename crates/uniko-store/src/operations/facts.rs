//! Fact node CRUD for P4 Consolidation.
//!
//! Provides upsert-by-triple semantics for derived Facts plus the
//! supporting edges (`SUPPORTED_BY`, `CREATED`, `INVOLVED`) and the
//! `ConsolidationCycle` audit node.  Embeddings are computed by Layer 2
//! and supplied as an opaque vector — this module never calls Xervo.

// Rust guideline compliant

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use uni_db::Value;
use uni_db::common::{TemporalValue, uni_btic::Btic};

use crate::error::Result;
use crate::id::new_id;
use crate::schema::btic::{btic_active, btic_upgrade_certainty, CERTAINTY_THRESHOLD};
use crate::schema::constants::{edges, labels};
use crate::storage::KnowledgeBase;
use crate::types::NodeId;

/// Outcome of a single Fact upsert call.
#[derive(Debug, Clone, Copy)]
pub struct FactUpsert {
    /// Internal node id of the (created or reinforced) Fact.
    pub node_id: NodeId,
    /// `true` when this call inserted a new Fact node.
    pub was_created: bool,
    /// Total `observation_count` after this upsert.
    pub observation_count: i64,
}

/// Build the deterministic external `fact_id` for a triple.
///
/// Idempotency anchor: callers upserting the same triple on subsequent
/// cycles resolve to the same Fact node via the
/// `Fact.fact_id` hash index.  Lowercased to make grouping
/// case-insensitive (P3 already normalizes `predicate`; the other two
/// fields can vary by surface form).
pub fn fact_id_for(subject: &str, predicate: &str, object: Option<&str>) -> String {
    format!(
        "{}|{}|{}",
        subject.to_ascii_lowercase(),
        predicate.to_ascii_lowercase(),
        object.unwrap_or("").to_ascii_lowercase(),
    )
}

/// Laplace-smoothed confidence: `(n + 1) / (n + 2)`.
///
/// Maps `n=0 → 0.5`, `n=1 → 0.67`, `n=4 → 0.83`, `n=10 → 0.92`.
/// Asymptotes toward 1.0 without ever reaching it — the system never
/// claims absolute certainty from observation counts alone.
pub fn laplace_confidence(observation_count: i64) -> f64 {
    let n = observation_count.max(0) as f64;
    (n + 1.0) / (n + 2.0)
}

impl KnowledgeBase {
    /// Upsert a Fact for the given `(subject, predicate, object)` triple.
    ///
    /// First call creates the Fact with `valid_at = btic_active(observed_at)`,
    /// `observation_count = n`, and Laplace-smoothed `confidence`.
    /// Subsequent calls reinforce: `observation_count += n`, `confidence`
    /// is recomputed, and the BTIC certainty is upgraded to
    /// [`Certainty::Definite`](uni_db::common::uni_btic::Certainty::Definite)
    /// once the cumulative count crosses [`CERTAINTY_THRESHOLD`].
    ///
    /// `embedding` is supplied by the caller (computed in Layer 2) and
    /// is only written on the create path; reinforcement does not
    /// re-embed since the canonical content text is unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on any database failure.
    pub async fn upsert_fact_by_triple(
        &self,
        subject: &str,
        predicate: &str,
        object: Option<&str>,
        observation_count: i64,
        observed_at: DateTime<Utc>,
        embedding: Option<Vec<f32>>,
    ) -> Result<FactUpsert> {
        let fact_id = fact_id_for(subject, predicate, object);

        // Idempotency lookup via the fact_id hash index.
        if let Some((nid, props)) = self
            .get_node_by_ext_id(labels::FACT, "fact_id", &fact_id)
            .await?
        {
            let prior_count = props
                .get("observation_count")
                .and_then(|v| v.as_i64())
                .unwrap_or(0);
            let new_count = prior_count.saturating_add(observation_count);
            let confidence = laplace_confidence(new_count);

            let mut updates: HashMap<String, Value> = HashMap::new();
            updates.insert("observation_count".into(), Value::Int(new_count));
            updates.insert("confidence".into(), Value::Float(confidence));

            // Upgrade certainty once the cumulative count crosses the
            // threshold.  Idempotent: a Fact already at Definite stays
            // there.
            if (prior_count as u64) < CERTAINTY_THRESHOLD
                && (new_count as u64) >= CERTAINTY_THRESHOLD
                && let Some(current) = extract_btic(props.get("valid_at"))
            {
                updates.insert(
                    "valid_at".into(),
                    btic_to_value(&btic_upgrade_certainty(&current)),
                );
            }

            self.update_node(nid, &updates).await?;
            return Ok(FactUpsert {
                node_id: nid,
                was_created: false,
                observation_count: new_count,
            });
        }

        // Create path.
        let valid_at = btic_active(observed_at);
        let confidence = laplace_confidence(observation_count);

        let mut props: HashMap<String, Value> = HashMap::new();
        props.insert("fact_id".into(), Value::String(fact_id));
        props.insert("subject".into(), Value::String(subject.to_string()));
        props.insert("predicate".into(), Value::String(predicate.to_string()));
        if let Some(obj) = object {
            props.insert("object".into(), Value::String(obj.to_string()));
        }
        props.insert("confidence".into(), Value::Float(confidence));
        props.insert("observation_count".into(), Value::Int(observation_count));
        props.insert("valid_at".into(), btic_to_value(&valid_at));
        props.insert(
            "source_rule".into(),
            Value::String("consolidation_v1".into()),
        );
        if let Some(vec) = embedding {
            props.insert("embedding".into(), Value::Vector(vec));
        }

        let nid = self.create_node(labels::FACT, &props).await?;
        Ok(FactUpsert {
            node_id: nid,
            was_created: true,
            observation_count,
        })
    }

    /// Wire `SUPPORTED_BY` edges from a Fact to many Observations.
    ///
    /// All edges share `weight = 1.0`.  Idempotency is the caller's
    /// responsibility — passing the same `(fact_id, observation)` pair
    /// twice creates duplicate edges.  In the consolidation cycle this
    /// is avoided by only processing observations that have no
    /// inbound `PROCESSED` edge from any prior cycle.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on database failure.
    pub async fn attach_supported_by(
        &self,
        fact_node_id: NodeId,
        observation_node_ids: &[NodeId],
    ) -> Result<()> {
        if observation_node_ids.is_empty() {
            return Ok(());
        }
        let mut weight_props: HashMap<String, Value> = HashMap::new();
        weight_props.insert("weight".into(), Value::Float(1.0));
        let edge_specs: Vec<(NodeId, NodeId, HashMap<String, Value>)> = observation_node_ids
            .iter()
            .map(|&oid| (fact_node_id, oid, weight_props.clone()))
            .collect();
        self.batch_create_edges_fast(
            edges::SUPPORTED_BY,
            Some(labels::FACT),
            Some(labels::OBSERVATION),
            &edge_specs,
        )
        .await
    }

    /// Create a `ConsolidationCycle` audit node and wire its edges.
    ///
    /// Wires:
    /// - `PROCESSED` to every observation visited in the cycle
    ///   (idempotency anchor for future cycles)
    /// - `CREATED` to every newly-created Fact
    /// - `INVOLVED` to every reinforced Fact (existing Fact whose
    ///   `observation_count` was incremented)
    /// - `APPLIED_RULE` to every Locy rule executed in the cycle
    ///
    /// `INVOLVED` is reused for "reinforced Fact" rather than "Episode"
    /// because the spec's audit edge for "Fact touched by this cycle
    /// but not created" maps to the same semantic; Episodes get their
    /// own edge type when the procedural-memory pipeline ships.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on database failure.
    #[expect(
        clippy::too_many_arguments,
        reason = "ConsolidationCycle is an audit record with five distinct \
                  edge groupings (processed/created/reinforced/applied \
                  rules) plus two timestamps; grouping into a struct \
                  would obscure the call site without simplifying it."
    )]
    pub async fn write_consolidation_cycle(
        &self,
        agent_id: &str,
        started_at: DateTime<Utc>,
        completed_at: DateTime<Utc>,
        processed_observations: &[NodeId],
        created_facts: &[NodeId],
        reinforced_facts: &[NodeId],
        applied_rules: &[NodeId],
    ) -> Result<NodeId> {
        let mut props: HashMap<String, Value> = HashMap::new();
        props.insert("cycle_id".into(), Value::String(new_id()));
        props.insert("agent_id".into(), Value::String(agent_id.to_string()));
        props.insert(
            "started_at".into(),
            Value::String(started_at.to_rfc3339()),
        );
        props.insert(
            "completed_at".into(),
            Value::String(completed_at.to_rfc3339()),
        );
        props.insert(
            "observations_processed".into(),
            Value::Int(processed_observations.len() as i64),
        );
        props.insert(
            "facts_created".into(),
            Value::Int(created_facts.len() as i64),
        );
        props.insert(
            "facts_reinforced".into(),
            Value::Int(reinforced_facts.len() as i64),
        );

        let cycle_nid = self
            .create_node(labels::CONSOLIDATION_CYCLE, &props)
            .await?;

        let empty: HashMap<String, Value> = HashMap::new();
        let mk_edges = |targets: &[NodeId]| -> Vec<(NodeId, NodeId, HashMap<String, Value>)> {
            targets
                .iter()
                .map(|&t| (cycle_nid, t, empty.clone()))
                .collect()
        };

        if !processed_observations.is_empty() {
            self.batch_create_edges_fast(
                edges::PROCESSED,
                Some(labels::CONSOLIDATION_CYCLE),
                Some(labels::OBSERVATION),
                &mk_edges(processed_observations),
            )
            .await?;
        }
        if !created_facts.is_empty() {
            self.batch_create_edges_fast(
                edges::CREATED,
                Some(labels::CONSOLIDATION_CYCLE),
                Some(labels::FACT),
                &mk_edges(created_facts),
            )
            .await?;
        }
        if !reinforced_facts.is_empty() {
            self.batch_create_edges_fast(
                edges::INVOLVED,
                Some(labels::CONSOLIDATION_CYCLE),
                Some(labels::FACT),
                &mk_edges(reinforced_facts),
            )
            .await?;
        }
        if !applied_rules.is_empty() {
            self.batch_create_edges_fast(
                edges::APPLIED_RULE,
                Some(labels::CONSOLIDATION_CYCLE),
                Some(labels::RULE),
                &mk_edges(applied_rules),
            )
            .await?;
        }

        Ok(cycle_nid)
    }
}

/// Convert an in-memory [`Btic`] to the [`Value::Temporal`] wire form.
fn btic_to_value(b: &Btic) -> Value {
    Value::Temporal(TemporalValue::Btic {
        lo: b.lo(),
        hi: b.hi(),
        meta: b.meta(),
    })
}

/// Decode a stored `valid_at` value back into an in-memory [`Btic`].
///
/// Returns `None` when the property is absent, null, or not a BTIC
/// temporal value (defensive — uni-db should not produce other shapes
/// for a column declared as [`uni_db::DataType::Btic`]).
fn extract_btic(value: Option<&Value>) -> Option<Btic> {
    match value? {
        Value::Temporal(TemporalValue::Btic { lo, hi, meta }) => {
            Btic::new(*lo, *hi, *meta).ok()
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fact_id_is_lowercased_and_pipe_separated() {
        assert_eq!(
            fact_id_for("Caroline", "Researches", Some("adoption agencies")),
            "caroline|researches|adoption agencies",
        );
    }

    #[test]
    fn fact_id_treats_missing_object_as_empty() {
        assert_eq!(fact_id_for("Jon", "is_happy", None), "jon|is_happy|");
    }

    #[test]
    fn laplace_grows_monotonically_and_is_bounded() {
        let c0 = laplace_confidence(0);
        let c4 = laplace_confidence(4);
        let c100 = laplace_confidence(100);
        assert!((c0 - 0.5).abs() < 1e-9);
        assert!(c0 < c4 && c4 < c100);
        assert!(c100 < 1.0);
    }
}

// Rust guideline compliant
