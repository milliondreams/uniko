//! Entity deduplication and graph persistence.
//!
//! Three-tier dedup cascade: (1) exact `entity_id` match, (2) embedding
//! similarity above threshold, (3) create new.  Creates MENTIONS edges
//! from source nodes to Entity nodes.

use std::collections::HashMap;

use uni_db::{Transaction, Value};

use uniko_store::schema::constants::{edges, labels};
use uniko_store::types::datetime_value;
use uniko_store::{KnowledgeBase, NodeId, UnikoError};

use super::types::{EntityMatch, RawEntity};

/// Merge raw entities by canonical name within a single extraction batch.
///
/// Entities with the same `canonical_name` are collapsed: the highest
/// confidence extraction is kept, and mention count reflects total
/// occurrences.
pub fn deduplicate_raw(raw: Vec<RawEntity>) -> Vec<(RawEntity, u32)> {
    let mut map: HashMap<String, (RawEntity, u32)> = HashMap::new();
    for entity in raw {
        let key = entity.canonical_name.clone();
        map.entry(key)
            .and_modify(|(best, count)| {
                *count += 1;
                if entity.confidence > best.confidence {
                    *best = entity.clone();
                }
            })
            .or_insert((entity, 1));
    }
    map.into_values().collect()
}

/// Upsert deduplicated entities into the graph and create MENTIONS edges.
///
/// Split-batch implementation that avoids `MERGE`'s per-row executor
/// loop (rustic-ai/uni-db#69). Issues four batched operations per call:
///
/// 1. Read-only `UNWIND … MATCH (n:Entity {entity_id: eid})` to find
///    which entity ids already exist — one indexed lookup per row, no
///    writer lock.
/// 2. [`KnowledgeBase::batch_create_nodes`] for the not-found subset —
///    one bulk `UNWIND … CREATE` that uni-db optimizes via the
///    HashJoin-friendly fast path.
/// 3. `UNWIND … MATCH WHERE id(n)=u.nid SET …` to bump frequency /
///    last_seen / confidence on the found subset — `id()`-equality
///    triggers the HashJoin rewrite (uni-db #53/#54).
/// 4. [`KnowledgeBase::batch_create_edges_fast`] for MENTIONS edges from
///    `source_node_id` to every resolved entity. Safe to create (not
///    MERGE): `source_node_id` is the freshly-ingested `Message` node,
///    so no prior MENTIONS edges from it can exist.
///
/// `entity_id` is `"{type}:{canonical_name}"` and uniquely identifies
/// an entity. On a match: frequency is incremented by `mention_count`,
/// `last_seen` is refreshed, confidence is raised to the new value if
/// higher. On a create: all properties initialised from the input.
///
/// Returns one [`EntityMatch`] per input row in input order.
///
/// # Errors
///
/// Returns [`UnikoError::Storage`] if any underlying query or
/// transaction fails. Partial state is possible if the run aborts
/// between phases (e.g. nodes created but their MENTIONS edges not
/// wired); the original per-entity loop had the same property.
pub async fn upsert_entities(
    kb: &KnowledgeBase,
    source_node_id: NodeId,
    deduped: Vec<(RawEntity, u32)>,
) -> uniko_store::Result<Vec<EntityMatch>> {
    if deduped.is_empty() {
        return Ok(Vec::new());
    }
    let phase_start = std::time::Instant::now();
    let prep = prepare_entity_upsert(kb, deduped).await?;
    let session = kb.db().session();
    let tx = session.tx().await?;
    let matches = apply_entity_upsert(kb, &tx, source_node_id, prep).await?;
    let commit_start = std::time::Instant::now();
    tx.commit().await?;
    let commit_ms = commit_start.elapsed().as_millis() as u64;
    let total_ms = phase_start.elapsed().as_millis() as u64;
    tracing::info!(commit_ms, "dedup commit");
    tracing::info!(
        target: "tx_perf",
        tx_phase = "entity_upsert",
        total_ms,
        commit_ms,
        entity_count = matches.len() as u64,
        "tx phase",
    );
    Ok(matches)
}

/// Pre-tx prep: assemble `EntityUpsertPrep` containing (a) the existing
/// entity-id → (nid, freq, conf) map from a Phase-1 read and (b) the
/// raw input ready for the apply step. Designed to be called BEFORE
/// the atomic-ingest tx is opened so the read uses a fresh session
/// snapshot and never collides with the writer side.
pub async fn prepare_entity_upsert(
    kb: &KnowledgeBase,
    deduped: Vec<(RawEntity, u32)>,
) -> uniko_store::Result<EntityUpsertPrep> {
    let entity_ids: Vec<String> = deduped
        .iter()
        .map(|(entity, _)| format!("{}:{}", entity.entity_type.as_str(), &entity.canonical_name))
        .collect();

    // Phase 1: batched MATCH (read-only) against a fresh session.
    // Done outside any tx so it doesn't extend the eventual write tx's
    // hold time. The atomic ingest path opens its tx AFTER this read.
    let existing = if entity_ids.is_empty() {
        HashMap::new()
    } else {
        let eids_list: Vec<Value> = entity_ids.iter().cloned().map(Value::String).collect();
        let match_cypher = "\
            UNWIND $eids AS eid \
            MATCH (n:Entity {entity_id: eid}) \
            RETURN eid AS entity_id, id(n) AS nid, \
                   n.frequency AS frequency, n.confidence AS confidence";
        let session = kb.db().session();
        let match_result = session
            .query_with(match_cypher)
            .param("eids", Value::List(eids_list))
            .fetch_all()
            .await?;
        let mut existing: HashMap<String, (NodeId, i64, f64)> =
            HashMap::with_capacity(match_result.rows().len());
        for row in match_result.rows() {
            let entity_id: String = row.get("entity_id")?;
            let nid: i64 = row.get("nid")?;
            let frequency: i64 = row.get("frequency")?;
            let confidence: f64 = row.get("confidence")?;
            existing.insert(entity_id, (nid, frequency, confidence));
        }
        existing
    };

    Ok(EntityUpsertPrep {
        deduped,
        entity_ids,
        existing,
        now_value: datetime_value(chrono::Utc::now()),
    })
}

/// Output of [`prepare_entity_upsert`]. Holds the resolved
/// new-vs-existing split plus the raw inputs needed at apply time.
#[derive(Debug)]
pub struct EntityUpsertPrep {
    /// Raw entities (with mention counts) to upsert, in input order.
    pub deduped: Vec<(RawEntity, u32)>,
    /// Pre-computed `"{type}:{canonical}"` ids, parallel to `deduped`.
    pub entity_ids: Vec<String>,
    /// Entities already in the graph: id → (nid, freq, conf).
    pub existing: HashMap<String, (NodeId, i64, f64)>,
    /// `Utc::now()` snapped at prep time; used as `last_seen`/`first_seen`.
    pub now_value: Value,
}

impl EntityUpsertPrep {
    /// True when there are no entities to write — apply step short-circuits.
    pub fn is_empty(&self) -> bool {
        self.deduped.is_empty()
    }
}

/// Writes-only inside the caller's tx. Issues:
/// (1) batch CREATE for not-found entities,
/// (2) batched UPDATE on existing entities (freq + last_seen + confidence),
/// (3) batch MENTIONS edges from `source_node_id` to every resolved Entity.
///
/// Returns one [`EntityMatch`] per input row in original order. The
/// caller owns the commit.
pub async fn apply_entity_upsert(
    kb: &KnowledgeBase,
    tx: &Transaction,
    source_node_id: NodeId,
    prep: EntityUpsertPrep,
) -> uniko_store::Result<Vec<EntityMatch>> {
    if prep.is_empty() {
        return Ok(Vec::new());
    }
    let EntityUpsertPrep {
        deduped,
        entity_ids,
        existing,
        now_value,
    } = prep;

    // Partition into (existing → batched UPDATE) and (new → batched
    // CREATE). `matches` is pre-filled with the right length so we can
    // fill node_id by index for the new rows once batch_create_nodes
    // returns them.
    let mut matches: Vec<EntityMatch> = Vec::with_capacity(deduped.len());
    let mut updates_list: Vec<Value> = Vec::new();
    let mut new_props: Vec<HashMap<String, Value>> = Vec::new();
    let mut new_indices: Vec<usize> = Vec::new();

    for (i, ((entity, mention_count), entity_id)) in
        deduped.iter().zip(entity_ids.iter()).enumerate()
    {
        if let Some(&(nid, old_freq, old_conf)) = existing.get(entity_id) {
            let new_freq = old_freq + i64::from(*mention_count);
            let new_conf = if entity.confidence > old_conf {
                entity.confidence
            } else {
                old_conf
            };

            let mut update = HashMap::with_capacity(3);
            update.insert("nid".into(), Value::Int(nid));
            update.insert("new_frequency".into(), Value::Int(new_freq));
            update.insert("new_confidence".into(), Value::Float(new_conf));
            updates_list.push(Value::Map(update));

            matches.push(EntityMatch {
                canonical_name: entity.canonical_name.clone(),
                node_id: nid,
                was_existing: true,
                mention_count: *mention_count,
            });
        } else {
            let mut props = HashMap::with_capacity(7);
            props.insert("entity_id".into(), Value::String(entity_id.clone()));
            props.insert("name".into(), Value::String(entity.canonical_name.clone()));
            props.insert(
                "entity_type".into(),
                Value::String(entity.entity_type.as_str().to_string()),
            );
            props.insert("first_seen".into(), now_value.clone());
            props.insert("last_seen".into(), now_value.clone());
            props.insert("frequency".into(), Value::Int(i64::from(*mention_count)));
            props.insert("confidence".into(), Value::Float(entity.confidence));
            new_props.push(props);
            new_indices.push(i);

            // Placeholder node_id; filled in after Phase 2.
            matches.push(EntityMatch {
                canonical_name: entity.canonical_name.clone(),
                node_id: 0,
                was_existing: false,
                mention_count: *mention_count,
            });
        }
    }

    // ── Phase 2: batched CREATE for not-found entities.
    let phase2_start = std::time::Instant::now();
    let new_count = new_props.len();
    if !new_props.is_empty() {
        let new_nids = kb
            .batch_create_nodes_in_tx(tx, labels::ENTITY, &new_props)
            .await?;
        if new_nids.len() != new_indices.len() {
            return Err(UnikoError::Storage(format!(
                "apply_entity_upsert: batch_create_nodes returned {} nids for {} new inputs",
                new_nids.len(),
                new_indices.len()
            )));
        }
        for (&i, &nid) in new_indices.iter().zip(new_nids.iter()) {
            matches[i].node_id = nid;
        }
    }
    let phase2_create_ms = phase2_start.elapsed().as_millis();

    // ── Phase 3: batched UPDATE for found entities.
    // The :Entity label hint is load-bearing: without it the planner
    // falls back to a multi-label scan per row, costing ~18 ms/row even
    // on a small KB. Investigated 2026-05-20.
    let phase3_start = std::time::Instant::now();
    let update_count = updates_list.len();
    if !updates_list.is_empty() {
        let update_cypher = "\
            UNWIND $updates AS u \
            MATCH (n:Entity) WHERE id(n) = u.nid \
            SET n.frequency = u.new_frequency, \
                n.last_seen = $now, \
                n.confidence = u.new_confidence";
        tx.execute_with(update_cypher)
            .param("updates", Value::List(updates_list))
            .param("now", now_value)
            .run()
            .await?;
    }
    let phase3_update_ms = phase3_start.elapsed().as_millis();

    // ── Phase 4: batched MENTIONS edges. Source is always the
    // freshly-created Message node, so no MENTIONS edges from it can
    // exist yet — plain CREATE is safe and skips MERGE's per-row penalty.
    let mentions_edges: Vec<(NodeId, NodeId, HashMap<String, Value>)> = matches
        .iter()
        .map(|m| {
            let mut props = HashMap::with_capacity(1);
            props.insert("count".into(), Value::Int(i64::from(m.mention_count)));
            (source_node_id, m.node_id, props)
        })
        .collect();
    let phase4_start = std::time::Instant::now();
    let mentions_count = mentions_edges.len();
    kb.batch_create_edges_fast_in_tx(
        tx,
        edges::MENTIONS,
        Some(labels::MESSAGE),
        Some(labels::ENTITY),
        &mentions_edges,
    )
    .await?;
    let phase4_mentions_ms = phase4_start.elapsed().as_millis();

    tracing::info!(
        target: "apply_entity_breakdown",
        phase2_create_ms = phase2_create_ms as u64,
        phase3_update_ms = phase3_update_ms as u64,
        phase4_mentions_ms = phase4_mentions_ms as u64,
        new_count = new_count as u64,
        update_count = update_count as u64,
        mentions_count = mentions_count as u64,
        "apply_entity breakdown",
    );

    Ok(matches)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ner::types::{EntityType, ExtractionSource};

    fn make_raw(name: &str, confidence: f64) -> RawEntity {
        RawEntity {
            surface_form: name.to_string(),
            canonical_name: name.to_lowercase(),
            entity_type: EntityType::Person,
            confidence,
            source: ExtractionSource::RuleBased,
            start_byte: 0,
            end_byte: name.len(),
        }
    }

    #[test]
    fn test_dedup_merges_same_name() {
        let raw = vec![
            make_raw("Alice", 0.7),
            make_raw("Alice", 0.9),
            make_raw("Bob", 0.8),
        ];
        let deduped = deduplicate_raw(raw);
        assert_eq!(deduped.len(), 2);

        let alice = deduped.iter().find(|(e, _)| e.canonical_name == "alice");
        assert!(alice.is_some());
        let (alice_ent, alice_count) = alice.unwrap();
        assert_eq!(*alice_count, 2);
        // Highest confidence kept.
        assert!((alice_ent.confidence - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn test_dedup_single_entity() {
        let raw = vec![make_raw("Carol", 0.8)];
        let deduped = deduplicate_raw(raw);
        assert_eq!(deduped.len(), 1);
        assert_eq!(deduped[0].1, 1);
    }

    #[test]
    fn test_dedup_empty() {
        let deduped = deduplicate_raw(Vec::new());
        assert!(deduped.is_empty());
    }
}
