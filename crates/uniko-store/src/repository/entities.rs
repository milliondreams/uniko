//! In-transaction entity-upsert reads/writes. Backs the entity-dedup
//! hot path in `uniko_extract::ner::dedup`, which opens its own
//! transaction (under per-entity RMW locks) and drives it through these
//! validated `*_in_tx` helpers instead of raw Cypher.

use std::collections::HashMap;

use uni_db::{Transaction, Value};

use crate::error::Result;
use crate::storage::KnowledgeBase;
use crate::types::NodeId;

impl KnowledgeBase {
    /// Authoritative existence read for an entity-upsert batch, **inside**
    /// the caller's transaction (and under its RMW locks).
    ///
    /// Returns a map from `entity_id` to `(node_id, frequency, confidence)`
    /// for every `entity_ids` member that already exists. Absent ids are
    /// simply missing from the map.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`](crate::UnikoError::Storage) when the
    /// read fails or a required column is unreadable.
    pub async fn fetch_entities_for_upsert_in_tx(
        &self,
        tx: &Transaction,
        entity_ids: &[String],
    ) -> Result<HashMap<String, (NodeId, i64, f64)>> {
        if entity_ids.is_empty() {
            return Ok(HashMap::new());
        }
        let eids_list: Vec<Value> = entity_ids.iter().cloned().map(Value::String).collect();
        let match_cypher = "\
            UNWIND $eids AS eid \
            MATCH (n:Entity {entity_id: eid}) \
            RETURN eid AS entity_id, id(n) AS nid, \
                   n.frequency AS frequency, n.confidence AS confidence";
        let match_result = tx
            .query_with(match_cypher)
            .param("eids", Value::List(eids_list))
            .fetch_all()
            .await?;
        let mut map: HashMap<String, (NodeId, i64, f64)> =
            HashMap::with_capacity(match_result.rows().len());
        for row in match_result.rows() {
            let entity_id: String = row.get("entity_id")?;
            let nid: i64 = row.get("nid")?;
            let frequency: i64 = row.get("frequency")?;
            let confidence: f64 = row.get("confidence")?;
            map.insert(entity_id, (nid, frequency, confidence));
        }
        Ok(map)
    }

    /// Batched counter UPDATE on existing entities, **inside** the caller's
    /// transaction: sets `frequency`, `confidence`, and `last_seen = now`
    /// for each `(node_id, new_frequency, new_confidence)` in `updates`.
    ///
    /// A no-op when `updates` is empty.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`](crate::UnikoError::Storage) when the
    /// UPDATE fails.
    pub async fn batch_update_entity_counters_in_tx(
        &self,
        tx: &Transaction,
        updates: &[(NodeId, i64, f64)],
        now: Value,
    ) -> Result<()> {
        if updates.is_empty() {
            return Ok(());
        }
        let updates_list: Vec<Value> = updates
            .iter()
            .map(|&(nid, new_freq, new_conf)| {
                let mut update = HashMap::with_capacity(3);
                update.insert("nid".into(), Value::Int(nid));
                update.insert("new_frequency".into(), Value::Int(new_freq));
                update.insert("new_confidence".into(), Value::Float(new_conf));
                Value::Map(update)
            })
            .collect();
        // uni-db 2.0 (#53/#54) rewrites `UNWIND … MATCH WHERE id(n) = col`
        // into a HashJoin, so the id()-equality match no longer needs an
        // :Entity label hint to avoid a per-row multi-label scan.
        let update_cypher = "\
            UNWIND $updates AS u \
            MATCH (n) WHERE id(n) = u.nid \
            SET n.frequency = u.new_frequency, \
                n.last_seen = $now, \
                n.confidence = u.new_confidence";
        tx.execute_with(update_cypher)
            .param("updates", Value::List(updates_list))
            .param("now", now)
            .run()
            .await?;
        Ok(())
    }
}
