//! Fact node CRUD for P4 Consolidation.
//!
//! Provides upsert-by-triple semantics for derived Facts plus the
//! supporting edges (`SUPPORTED_BY`, `CREATED`, `INVOLVED`) and the
//! `ConsolidationCycle` audit node.  Embeddings are computed by Layer 2
//! and supplied as an opaque vector — this module never calls Xervo.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use uni_db::Value;
use uni_db::common::{TemporalValue, uni_btic::Btic};

use crate::error::Result;
use crate::id::new_id;
use crate::schema::btic::{CERTAINTY_THRESHOLD, btic_active, btic_upgrade_certainty};
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

/// One Fact to upsert in [`KnowledgeBase::batch_upsert_facts`].
///
/// Mirrors the per-Fact inputs of [`KnowledgeBase::upsert_fact_by_triple`].
/// `embedding` may be `None` for individual entries when the Layer 2
/// batch embed succeeded only for a subset — those Facts will be stored
/// without an embedding (same semantic as the single-call helper).
#[derive(Debug, Clone)]
pub struct FactUpsertInput {
    /// Triple subject (lower-cased by `fact_id_for` for the id).
    pub subject: String,
    /// Triple predicate.
    pub predicate: String,
    /// Triple object; `None` collapses to empty string in `fact_id`.
    pub object: Option<String>,
    /// Number of observations contributing to this upsert.
    pub observation_count: i64,
    /// Earliest observed_at among the contributing observations.
    pub observed_at: DateTime<Utc>,
    /// Pre-computed embedding from Layer 2, or `None` to store without.
    pub embedding: Option<Vec<f32>>,
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

    /// Upsert many Facts via the same per-Fact semantics as
    /// [`KnowledgeBase::upsert_fact_by_triple`], in batched operations.
    ///
    /// Issues at most four storage ops per call regardless of input
    /// size, replacing the per-Fact transaction loop. The split mirrors
    /// the entity-dedup batched path:
    ///
    /// 1. Read-only `UNWIND … MATCH (f:Fact {fact_id: fid})` to find
    ///    which `fact_id`s already exist (one indexed lookup per row).
    /// 2. [`KnowledgeBase::batch_create_nodes`] for the not-found subset.
    /// 3. `UNWIND … MATCH WHERE id(n)=u.nid SET …` for the reinforce
    ///    subset, including the conditional BTIC certainty upgrade
    ///    pre-computed per row.
    ///
    /// Returns one [`FactUpsert`] per input row in input order, with
    /// the freshly resolved node id and the post-upsert observation
    /// count.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::UnikoError::Storage`] on any storage
    /// failure. Partial state is possible if the call aborts between
    /// phases — same property as the per-Fact loop it replaces.
    pub async fn batch_upsert_facts(
        &self,
        inputs: Vec<FactUpsertInput>,
    ) -> Result<Vec<FactUpsert>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        // Pre-compute fact_ids once; reused across phases.
        let fact_ids: Vec<String> = inputs
            .iter()
            .map(|i| fact_id_for(&i.subject, &i.predicate, i.object.as_deref()))
            .collect();

        // Within-batch dedup by fact_id. Multiple input rows can map
        // to the same fact_id when the caller's grouping key
        // (subject, predicate) differs only in case from another
        // entry — `fact_id_for` lower-cases both sides. The original
        // per-row `upsert_fact_by_triple` loop folded these via its
        // MATCH-then-create-or-update; the batched path must do the
        // same in Rust because two UNWIND rows over the same MERGE
        // node would otherwise clobber each other's counts (for the
        // existing path) or create duplicate Fact nodes (for the
        // new path — the schema has no unique constraint on
        // `fact_id`).
        //
        // First occurrence wins for subject / predicate / object /
        // embedding; `observation_count` is summed across all rows
        // with the same fact_id, and `observed_at` is reduced to the
        // earliest. `first_idx` records the input index that
        // introduced each fact_id so the output `was_created` flag
        // can mirror the per-row helper (true only for the first
        // appearance of a freshly-created Fact).
        let mut canonical: HashMap<String, FactUpsertInput> =
            HashMap::with_capacity(fact_ids.len());
        let mut first_idx: HashMap<String, usize> = HashMap::with_capacity(fact_ids.len());
        let mut unique_fids: Vec<String> = Vec::with_capacity(fact_ids.len());
        for (i, (input, fid)) in inputs.iter().zip(fact_ids.iter()).enumerate() {
            match canonical.entry(fid.clone()) {
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert(input.clone());
                    first_idx.insert(fid.clone(), i);
                    unique_fids.push(fid.clone());
                }
                std::collections::hash_map::Entry::Occupied(mut e) => {
                    let cur = e.get_mut();
                    cur.observation_count = cur
                        .observation_count
                        .saturating_add(input.observation_count);
                    if input.observed_at < cur.observed_at {
                        cur.observed_at = input.observed_at;
                    }
                }
            }
        }

        // Phase 1: batched MATCH to discover which (deduped) fact_ids
        // exist. Returns one row per match; absent fact_ids simply
        // don't appear.
        let fid_list: Vec<Value> = unique_fids.iter().cloned().map(Value::String).collect();
        let match_cypher = "\
            UNWIND $fids AS fid \
            MATCH (n:Fact {fact_id: fid}) \
            RETURN fid AS fact_id, id(n) AS nid, \
                   n.observation_count AS observation_count, \
                   n.valid_at AS valid_at";

        let session = self.db.session();
        let match_result = session
            .query_with(match_cypher)
            .param("fids", Value::List(fid_list))
            .fetch_all()
            .await?;

        // Build fact_id → (nid, prior_count, prior_btic) map.
        let mut existing: HashMap<String, (NodeId, i64, Option<Btic>)> =
            HashMap::with_capacity(match_result.rows().len());
        for row in match_result.rows() {
            let fid: String = row.get("fact_id")?;
            let nid: i64 = row.get("nid")?;
            let prior_count: i64 = row.get("observation_count")?;
            // valid_at comes back as the BTIC Temporal Value; extract
            // structurally rather than going through a string roundtrip.
            // `Row::value` returns the raw `&Value`; `Row::get<T>` would
            // require `T: FromValue` which `Value` itself does not
            // implement.
            let prior_btic = extract_btic(row.value("valid_at"));
            existing.insert(fid, (nid, prior_count, prior_btic));
        }

        // Partition deduped canonical entries into existing (UPDATE)
        // and new (CREATE). `resolved` maps each unique fact_id to
        // its post-upsert node_id and post-upsert observation_count;
        // it's keyed by fact_id so the final output loop can fan it
        // back out across every original input row (including the
        // collapsed duplicates).
        let mut updates_list: Vec<Value> = Vec::new();
        let mut new_props: Vec<HashMap<String, Value>> = Vec::new();
        let mut new_fids_in_order: Vec<String> = Vec::new();
        let mut resolved: HashMap<String, (NodeId, i64)> =
            HashMap::with_capacity(unique_fids.len());

        for fid in &unique_fids {
            let canon = canonical
                .get(fid)
                .expect("canonical map populated above for every unique_fids entry");

            if let Some((nid, prior_count, prior_btic)) = existing.get(fid) {
                let new_count = prior_count.saturating_add(canon.observation_count);
                let confidence = laplace_confidence(new_count);

                let mut update = HashMap::with_capacity(4);
                update.insert("nid".into(), Value::Int(*nid));
                update.insert("new_count".into(), Value::Int(new_count));
                update.insert("new_confidence".into(), Value::Float(confidence));

                // Conditional certainty upgrade — match the single-row
                // path exactly: only upgrade once we cross the
                // threshold from below, and only when we still have a
                // readable prior BTIC to upgrade.
                let upgrade_valid_at = if (*prior_count as u64) < CERTAINTY_THRESHOLD
                    && (new_count as u64) >= CERTAINTY_THRESHOLD
                {
                    prior_btic
                        .as_ref()
                        .map(|b| btic_to_value(&btic_upgrade_certainty(b)))
                } else {
                    None
                };
                // Always send a `valid_at` key so the SET clause is
                // the same shape across rows; rows that don't need an
                // upgrade get `Value::Null` and the COALESCE in the
                // UPDATE statement keeps the existing column value.
                update.insert("valid_at".into(), upgrade_valid_at.unwrap_or(Value::Null));
                updates_list.push(Value::Map(update));

                resolved.insert(fid.clone(), (*nid, new_count));
            } else {
                let valid_at = btic_active(canon.observed_at);
                let confidence = laplace_confidence(canon.observation_count);

                let mut props: HashMap<String, Value> = HashMap::with_capacity(9);
                props.insert("fact_id".into(), Value::String(fid.clone()));
                props.insert("subject".into(), Value::String(canon.subject.clone()));
                props.insert("predicate".into(), Value::String(canon.predicate.clone()));
                if let Some(obj) = &canon.object {
                    props.insert("object".into(), Value::String(obj.clone()));
                }
                props.insert("confidence".into(), Value::Float(confidence));
                props.insert(
                    "observation_count".into(),
                    Value::Int(canon.observation_count),
                );
                props.insert("valid_at".into(), btic_to_value(&valid_at));
                props.insert(
                    "source_rule".into(),
                    Value::String("consolidation_v1".into()),
                );
                if let Some(vec) = canon.embedding.clone() {
                    props.insert("embedding".into(), Value::Vector(vec));
                }
                new_props.push(props);
                new_fids_in_order.push(fid.clone());
            }
        }

        // Phase 2: batched CREATE for not-found Facts.
        if !new_props.is_empty() {
            let new_nids = self.batch_create_nodes(labels::FACT, &new_props).await?;
            if new_nids.len() != new_fids_in_order.len() {
                return Err(crate::error::UnikoError::Storage(format!(
                    "batch_upsert_facts: batch_create_nodes returned {} nids for {} new inputs",
                    new_nids.len(),
                    new_fids_in_order.len()
                )));
            }
            for (fid, nid) in new_fids_in_order.into_iter().zip(new_nids) {
                let canon_count = canonical
                    .get(&fid)
                    .map(|c| c.observation_count)
                    .unwrap_or(0);
                resolved.insert(fid, (nid, canon_count));
            }
        }

        // Phase 3: batched UPDATE for reinforce. The COALESCE keeps the
        // existing valid_at when u.valid_at is null (no certainty
        // upgrade this round); otherwise it adopts the upgraded BTIC.
        if !updates_list.is_empty() {
            let update_cypher = "\
                UNWIND $updates AS u \
                MATCH (n) WHERE id(n) = u.nid \
                SET n.observation_count = u.new_count, \
                    n.confidence = u.new_confidence, \
                    n.valid_at = COALESCE(u.valid_at, n.valid_at)";
            let tx = session.tx().await?;
            tx.execute_with(update_cypher)
                .param("updates", Value::List(updates_list))
                .run()
                .await?;
            tx.commit().await?;
        }

        // Fan resolved (fact_id → node_id) back out across every
        // original input row, preserving caller-visible order. The
        // `was_created` flag is true only for the input that
        // introduced its fact_id AND only when that fact_id wasn't
        // already in the DB before this call — mirroring the
        // per-row `upsert_fact_by_triple` exactly.
        let mut outputs: Vec<FactUpsert> = Vec::with_capacity(inputs.len());
        for (i, fid) in fact_ids.iter().enumerate() {
            let (nid, post_count) = *resolved
                .get(fid)
                .expect("resolved populated for every fact_id");
            let was_first_input = first_idx.get(fid).map(|&first| first == i).unwrap_or(false);
            let was_newly_created = !existing.contains_key(fid);
            outputs.push(FactUpsert {
                node_id: nid,
                was_created: was_first_input && was_newly_created,
                observation_count: post_count,
            });
        }

        Ok(outputs)
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
        reason = "ConsolidationCycle is an audit record with several \
                  edge groupings (processed/created/reinforced/invalidated/ \
                  applied rules) plus drift alerts and two timestamps; \
                  grouping into a struct would obscure the call site \
                  without simplifying it."
    )]
    pub async fn write_consolidation_cycle(
        &self,
        agent_id: &str,
        started_at: DateTime<Utc>,
        completed_at: DateTime<Utc>,
        processed_observations: &[NodeId],
        created_facts: &[NodeId],
        reinforced_facts: &[NodeId],
        invalidated_facts: &[NodeId],
        drift_alerts: i64,
        applied_rules: &[NodeId],
    ) -> Result<NodeId> {
        let mut props: HashMap<String, Value> = HashMap::new();
        props.insert("cycle_id".into(), Value::String(new_id()));
        props.insert("agent_id".into(), Value::String(agent_id.to_string()));
        props.insert(
            "started_at".into(),
            crate::types::datetime_value(started_at),
        );
        props.insert(
            "completed_at".into(),
            crate::types::datetime_value(completed_at),
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
        props.insert(
            "facts_invalidated".into(),
            Value::Int(invalidated_facts.len() as i64),
        );
        props.insert("drift_alerts".into(), Value::Int(drift_alerts));

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

        // Every outgoing edge from the cycle node shares the source label
        // (CONSOLIDATION_CYCLE) and `mk_edges` shape; the only per-batch
        // variation is the edge type, the target label, and the targets vec.
        let edge_batches: [(&str, &str, &[NodeId]); 5] = [
            (
                edges::PROCESSED,
                labels::OBSERVATION,
                processed_observations,
            ),
            (edges::CREATED, labels::FACT, created_facts),
            (edges::INVOLVED, labels::FACT, reinforced_facts),
            (edges::INVALIDATED, labels::FACT, invalidated_facts),
            (edges::APPLIED_RULE, labels::RULE, applied_rules),
        ];
        for (edge_type, target_label, targets) in edge_batches {
            if targets.is_empty() {
                continue;
            }
            self.batch_create_edges_fast(
                edge_type,
                Some(labels::CONSOLIDATION_CYCLE),
                Some(target_label),
                &mk_edges(targets),
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
        Value::Temporal(TemporalValue::Btic { lo, hi, meta }) => Btic::new(*lo, *hi, *meta).ok(),
        _ => None,
    }
}

/// Extract the `lo` bound (epoch millis) from a BTIC display string.
///
/// uni-db's `toString(btic)` produces `"[<iso>, <hi>) <grain>/ [<cert>/<cert>]"`.
/// We isolate the substring between the opening bracket and the first
/// comma, then parse it as RFC 3339.  Returns `None` on shape mismatch
/// — callers fall back to skipping the row.
fn parse_btic_lo_millis(s: &str) -> Option<i64> {
    let after_bracket = s.strip_prefix('[')?;
    let (lo_iso, _) = after_bracket.split_once(", ")?;
    let dt = chrono::DateTime::parse_from_rfc3339(lo_iso.trim()).ok()?;
    Some(dt.with_timezone(&Utc).timestamp_millis())
}

/// Parse uni-db's named Granularity strings back into the enum.
fn parse_granularity(s: &str) -> Option<uni_db::common::uni_btic::Granularity> {
    uni_db::common::uni_btic::Granularity::from_name(s)
}

/// Parse uni-db's named Certainty strings back into the enum.
fn parse_certainty(s: &str) -> Option<uni_db::common::uni_btic::Certainty> {
    use uni_db::common::uni_btic::Certainty;
    Some(match s {
        "definite" => Certainty::Definite,
        "approximate" => Certainty::Approximate,
        "uncertain" => Certainty::Uncertain,
        "unknown" => Certainty::Unknown,
        _ => return None,
    })
}

impl KnowledgeBase {
    /// Find every open-BTIC Fact for `(subject, predicate)` whose
    /// `object` differs from `current_object`.
    ///
    /// "Open" = `valid_at.hi == POS_INF`, i.e. not previously closed.
    /// Used by F38 contradiction detection: when a consolidation cycle
    /// promotes a new canonical object, prior open Facts with stale
    /// objects must be invalidated.
    ///
    /// Returns each match's `(node_id, valid_at)` pair so the caller
    /// can close the BTIC without a second read.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on database failure.
    pub async fn find_stale_open_facts(
        &self,
        subject: &str,
        predicate: &str,
        current_object: Option<&str>,
    ) -> Result<Vec<(NodeId, Btic)>> {
        use uni_db::common::uni_btic::btic::POS_INF;

        let session = self.db.session();
        // `RETURN f` serializes Temporal columns to their display
        // string when packaged inside a Node, which prevents direct
        // `extract_btic` extraction.  We use `btic_is_unbounded` to
        // filter open intervals server-side and `btic_lo_granularity`
        // / `btic_lo_certainty` to recover meta as named strings; the
        // BTIC display is parsed for the `lo` timestamp.
        let cypher = "MATCH (f:Fact) \
                      WHERE f.subject = $s AND f.predicate = $p \
                        AND btic_is_unbounded(f.valid_at) \
                      RETURN id(f) AS nid, f.object AS obj, \
                             toString(f.valid_at) AS vat_str, \
                             btic_lo_granularity(f.valid_at) AS lo_g, \
                             btic_lo_certainty(f.valid_at) AS lo_c";
        let result = session
            .query_with(cypher)
            .param("s", subject)
            .param("p", predicate)
            .fetch_all()
            .await?;

        let current = current_object.unwrap_or("").to_ascii_lowercase();
        let mut out = Vec::new();
        for row in result.rows() {
            let Ok(nid) = row.get::<i64>("nid") else {
                continue;
            };
            let obj: String = row.get::<String>("obj").unwrap_or_default();
            if !current.is_empty() && obj.to_ascii_lowercase() == current {
                continue;
            }
            let Ok(vat_str) = row.get::<String>("vat_str") else {
                continue;
            };
            let Some(lo_ms) = parse_btic_lo_millis(&vat_str) else {
                continue;
            };
            let lo_g = row
                .get::<String>("lo_g")
                .ok()
                .and_then(|s| parse_granularity(&s))
                .unwrap_or(uni_db::common::uni_btic::Granularity::Day);
            let lo_c = row
                .get::<String>("lo_c")
                .ok()
                .and_then(|s| parse_certainty(&s))
                .unwrap_or(uni_db::common::uni_btic::Certainty::Approximate);
            let meta = Btic::build_meta(
                lo_g,
                uni_db::common::uni_btic::Granularity::Millisecond,
                lo_c,
                uni_db::common::uni_btic::Certainty::Definite,
            );
            if let Ok(btic) = Btic::new(lo_ms, POS_INF, meta) {
                out.push((nid, btic));
            }
        }
        Ok(out)
    }

    /// Batched variant of [`KnowledgeBase::find_stale_open_facts`].
    ///
    /// Replaces N per-key session queries with one `UNWIND $keys ...
    /// MATCH (f:Fact) ...` call. Also returns the prior `object`
    /// surface form per Fact (eliminating the follow-up
    /// `MATCH (f) WHERE id(f) = $vid RETURN f.object` that the
    /// per-group caller would otherwise need).
    ///
    /// Result map: each `(subject, predicate)` key in the input maps
    /// to the open-BTIC Facts matching it. Keys with no open Facts
    /// are absent from the map (caller should treat absence as empty).
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on database failure.
    pub async fn find_stale_open_facts_batched(
        &self,
        keys: &[(String, String)],
    ) -> Result<HashMap<(String, String), Vec<(NodeId, Btic, String)>>> {
        use std::collections::hash_map::Entry;
        use uni_db::common::uni_btic::btic::POS_INF;

        if keys.is_empty() {
            return Ok(HashMap::new());
        }

        // Build the UNWIND list as `[{subject, predicate}, …]`.
        let list: Vec<Value> = keys
            .iter()
            .map(|(s, p)| {
                let mut m = HashMap::with_capacity(2);
                m.insert("subject".to_string(), Value::String(s.clone()));
                m.insert("predicate".to_string(), Value::String(p.clone()));
                Value::Map(m)
            })
            .collect();

        let cypher = "UNWIND $keys AS k \
                      MATCH (f:Fact) \
                      WHERE f.subject = k.subject \
                        AND f.predicate = k.predicate \
                        AND btic_is_unbounded(f.valid_at) \
                      RETURN k.subject AS subject, k.predicate AS predicate, \
                             id(f) AS nid, f.object AS obj, \
                             toString(f.valid_at) AS vat_str, \
                             btic_lo_granularity(f.valid_at) AS lo_g, \
                             btic_lo_certainty(f.valid_at) AS lo_c";

        let session = self.db.session();
        let result = session
            .query_with(cypher)
            .param("keys", Value::List(list))
            .fetch_all()
            .await?;

        type FactRow = (NodeId, Btic, String);
        let mut out: HashMap<(String, String), Vec<FactRow>> = HashMap::new();
        for row in result.rows() {
            let Ok(subject) = row.get::<String>("subject") else {
                continue;
            };
            let Ok(predicate) = row.get::<String>("predicate") else {
                continue;
            };
            let Ok(nid) = row.get::<i64>("nid") else {
                continue;
            };
            let obj: String = row.get::<String>("obj").unwrap_or_default();
            let Ok(vat_str) = row.get::<String>("vat_str") else {
                continue;
            };
            let Some(lo_ms) = parse_btic_lo_millis(&vat_str) else {
                continue;
            };
            let lo_g = row
                .get::<String>("lo_g")
                .ok()
                .and_then(|s| parse_granularity(&s))
                .unwrap_or(uni_db::common::uni_btic::Granularity::Day);
            let lo_c = row
                .get::<String>("lo_c")
                .ok()
                .and_then(|s| parse_certainty(&s))
                .unwrap_or(uni_db::common::uni_btic::Certainty::Approximate);
            let meta = Btic::build_meta(
                lo_g,
                uni_db::common::uni_btic::Granularity::Millisecond,
                lo_c,
                uni_db::common::uni_btic::Certainty::Definite,
            );
            let Ok(btic) = Btic::new(lo_ms, POS_INF, meta) else {
                continue;
            };

            match out.entry((subject, predicate)) {
                Entry::Vacant(e) => {
                    e.insert(vec![(nid, btic, obj)]);
                }
                Entry::Occupied(mut e) => {
                    e.get_mut().push((nid, btic, obj));
                }
            }
        }
        Ok(out)
    }

    /// Close a Fact's BTIC interval as of `now` and (optionally) wire an
    /// `INVALIDATES` edge from `replacement` → this fact.
    ///
    /// Used by F38 when a contradicting canonical object is promoted
    /// in a consolidation cycle.  The fact remains in the graph (for
    /// audit) but its BTIC `hi` is closed so `btic.contains(now)` is
    /// false going forward.
    ///
    /// `reason` populates the `INVALIDATES.reason` property; pass
    /// `None` to skip the edge entirely (useful when the caller wants
    /// to invalidate without recording a successor).
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on database failure.
    pub async fn invalidate_fact(
        &self,
        stale_fact: NodeId,
        old_interval: &Btic,
        now: DateTime<Utc>,
        replacement: Option<NodeId>,
        reason: Option<&str>,
    ) -> Result<()> {
        let closed = crate::schema::btic::btic_invalidate(old_interval, now);
        let mut updates: HashMap<String, Value> = HashMap::new();
        updates.insert("valid_at".into(), btic_to_value(&closed));
        self.update_node(stale_fact, &updates).await?;

        if let Some(new_fact) = replacement {
            let mut edge_props: HashMap<String, Value> = HashMap::new();
            if let Some(r) = reason {
                edge_props.insert("reason".into(), Value::String(r.to_string()));
            }
            self.create_edge(edges::INVALIDATES, new_fact, stale_fact, &edge_props)
                .await?;
        }
        Ok(())
    }

    /// Record that an Entity's subject was invalidated and flip the
    /// `unstable` flag once invalidations within the last 30 days
    /// exceed `drift_threshold`.
    ///
    /// Returns whether the Entity transitioned to unstable on this
    /// call (`false` when it was already unstable or remains stable).
    /// Creates an Entity node implicitly when `subject` has none — F38
    /// must not silently swallow drift when the upstream NER missed
    /// the subject.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on database failure.
    pub async fn record_entity_invalidation(
        &self,
        subject: &str,
        now: DateTime<Utc>,
        drift_threshold: i64,
    ) -> Result<bool> {
        let key = subject.trim();
        if key.is_empty() {
            return Ok(false);
        }

        // Fetch (or implicitly create) the Entity by name.
        let session = self.db.session();
        let lookup = session
            .query_with("MATCH (e:Entity {name: $n}) RETURN id(e) AS nid, e.invalidation_count AS c, e.unstable AS u")
            .param("n", key)
            .fetch_all()
            .await
            ?;

        let (nid, prior_count, prior_unstable) = match lookup.rows().first() {
            Some(row) => {
                let nid: i64 = row.get("nid").unwrap_or(0);
                let c: i64 = row.get::<i64>("c").unwrap_or(0);
                let u: bool = row.get::<bool>("u").unwrap_or(false);
                (nid, c, u)
            }
            None => {
                let mut props: HashMap<String, Value> = HashMap::new();
                props.insert("name".into(), Value::String(key.to_string()));
                let nid = self
                    .merge_node(labels::ENTITY, "entity_id", key, &props)
                    .await?;
                (nid, 0, false)
            }
        };

        let new_count = prior_count.saturating_add(1);
        let mut updates: HashMap<String, Value> = HashMap::new();
        updates.insert("invalidation_count".into(), Value::Int(new_count));
        updates.insert(
            "last_invalidation_at".into(),
            crate::types::datetime_value(now),
        );

        // Drift gate: count invalidations within the last 30 days by
        // matching INVALIDATES edges on Facts whose subject equals this
        // entity.  Falls back to the cumulative count when the windowed
        // query yields nothing (e.g., no INVALIDATES edges yet because
        // the caller hasn't wired one for the current invalidation).
        let window_count = self.count_recent_invalidations(key).await?.max(new_count);
        let now_unstable = window_count > drift_threshold;
        updates.insert("unstable".into(), Value::Bool(now_unstable));
        self.update_node(nid, &updates).await?;

        Ok(now_unstable && !prior_unstable)
    }

    /// Count `INVALIDATES` edges on Facts with `subject = entity_name`.
    ///
    /// The cited "last 30 days" window in the surrounding caller is
    /// implemented by the caller-side `.max(new_count)` fallback; this
    /// function is the pure cumulative count.
    async fn count_recent_invalidations(&self, entity_name: &str) -> Result<i64> {
        let session = self.db.session();
        let cypher = "MATCH (new:Fact)-[r:INVALIDATES]->(old:Fact) \
                      WHERE old.subject = $n \
                      RETURN count(r) AS k";
        let result = session
            .query_with(cypher)
            .param("n", entity_name)
            .fetch_all()
            .await?;
        match result.rows().first() {
            Some(row) => Ok(row.get::<i64>("k").unwrap_or(0)),
            None => Ok(0),
        }
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
