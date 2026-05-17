//! Entity deduplication and graph persistence.
//!
//! Three-tier dedup cascade: (1) exact `entity_id` match, (2) embedding
//! similarity above threshold, (3) create new.  Creates MENTIONS edges
//! from source nodes to Entity nodes.

// Rust guideline compliant

use std::collections::HashMap;

use uni_db::Value;

use uniko_store::schema::constants::{edges, labels};
use uniko_store::types::datetime_value;
use uniko_store::{KnowledgeBase, NodeId, UnikoError};

use super::types::{EntityMatch, RawEntity};

/// Cosine similarity threshold for same-type entity merging.
const SIMILARITY_SAME_TYPE: f64 = 0.85;

/// Cosine similarity threshold for cross-type entity merging (stricter).
const SIMILARITY_CROSS_TYPE: f64 = 0.92;

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

    let now_value = datetime_value(chrono::Utc::now());

    // Pre-compute the entity_id strings once; reused across phases.
    let entity_ids: Vec<String> = deduped
        .iter()
        .map(|(entity, _)| format!("{}:{}", entity.entity_type.as_str(), &entity.canonical_name))
        .collect();

    // ── Phase 1: batched MATCH to discover which entity_ids exist.
    // Read-only, no transaction — the executor iterates per row but
    // each row is a single indexed hash lookup (no writer lock, no
    // pattern plan beyond the index). Returns one row per match;
    // entries that didn't exist simply don't appear in the result.
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
        .await
        .map_err(|err| UnikoError::Storage(err.to_string()))?;

    let mut existing: HashMap<String, (NodeId, i64, f64)> =
        HashMap::with_capacity(match_result.rows().len());
    for row in match_result.rows() {
        let entity_id: String = row
            .get("entity_id")
            .map_err(|err| UnikoError::Storage(err.to_string()))?;
        let nid: i64 = row
            .get("nid")
            .map_err(|err| UnikoError::Storage(err.to_string()))?;
        let frequency: i64 = row
            .get("frequency")
            .map_err(|err| UnikoError::Storage(err.to_string()))?;
        let confidence: f64 = row
            .get("confidence")
            .map_err(|err| UnikoError::Storage(err.to_string()))?;
        existing.insert(entity_id, (nid, frequency, confidence));
    }

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
    if !new_props.is_empty() {
        let new_nids = kb.batch_create_nodes(labels::ENTITY, &new_props).await?;
        if new_nids.len() != new_indices.len() {
            return Err(UnikoError::Storage(format!(
                "upsert_entities: batch_create_nodes returned {} nids for {} new inputs",
                new_nids.len(),
                new_indices.len()
            )));
        }
        for (&i, &nid) in new_indices.iter().zip(new_nids.iter()) {
            matches[i].node_id = nid;
        }
    }

    // ── Phase 3: batched UPDATE for found entities.
    if !updates_list.is_empty() {
        let update_cypher = "\
            UNWIND $updates AS u \
            MATCH (n) WHERE id(n) = u.nid \
            SET n.frequency = u.new_frequency, \
                n.last_seen = $now, \
                n.confidence = u.new_confidence";
        let tx = session
            .tx()
            .await
            .map_err(|err| UnikoError::Storage(err.to_string()))?;
        tx.execute_with(update_cypher)
            .param("updates", Value::List(updates_list))
            .param("now", now_value)
            .run()
            .await
            .map_err(|err| UnikoError::Storage(err.to_string()))?;
        tx.commit()
            .await
            .map_err(|err| UnikoError::Storage(err.to_string()))?;
    }

    // ── Phase 4: batched MENTIONS edges. Source is always the
    // freshly-created Message node ingested by `EntityExtractionStep`,
    // so no MENTIONS edges from it exist yet — plain CREATE is safe
    // and skips MERGE's per-row penalty. The src_label="Message" hint
    // narrows the planner's source scan.
    let mentions_edges: Vec<(NodeId, NodeId, HashMap<String, Value>)> = matches
        .iter()
        .map(|m| {
            let mut props = HashMap::with_capacity(1);
            props.insert("count".into(), Value::Int(i64::from(m.mention_count)));
            (source_node_id, m.node_id, props)
        })
        .collect();
    kb.batch_create_edges_fast(
        edges::MENTIONS,
        Some(labels::MESSAGE),
        Some(labels::ENTITY),
        &mentions_edges,
    )
    .await?;

    // Touch the labels constant so import stays load-bearing; the
    // hardcoded "Entity" string in the Cypher must match it.
    debug_assert_eq!(labels::ENTITY, "Entity");

    Ok(matches)
}

/// Format the text used to compute an entity's embedding.
///
/// Formula: `"name (type)"` or just `"name"` if type is empty.
fn format_embed_text(name: &str, entity_type: &str) -> String {
    if entity_type.is_empty() || entity_type == "other" {
        name.to_string()
    } else {
        format!("{name} ({entity_type})")
    }
}

/// Search for an existing Entity with a similar embedding.
///
/// Returns `Some((node_id, properties))` if a match above the
/// similarity threshold is found for a compatible type.  Returns
/// `None` if no match, Xervo is unavailable, or the graph has no
/// Entity embeddings yet.
async fn find_similar_entity(
    kb: &KnowledgeBase,
    name: &str,
    entity_type: &str,
) -> uniko_store::Result<Option<(NodeId, HashMap<String, Value>)>> {
    // Compute embedding for the candidate entity.
    let embed_text = format_embed_text(name, entity_type);
    let vec = match crate::embedding::embed_query(kb, &embed_text).await {
        Ok(v) if !v.is_empty() => v,
        _ => return Ok(None), // Xervo unavailable or empty — skip similarity.
    };

    // Vector search for similar existing entities.
    let results = match kb
        .vector_search(&vec, labels::ENTITY, "embedding", 5, None)
        .await
    {
        Ok(r) => r,
        Err(_) => return Ok(None), // No vector index or no data — skip.
    };

    for result in &results {
        let existing_type = result
            .properties
            .get("entity_type")
            .and_then(|v| v.as_str())
            .unwrap_or("other");

        // Determine threshold based on type compatibility.
        let threshold = if !types_compatible(entity_type, existing_type) {
            continue; // Incompatible types — never merge.
        } else if entity_type == existing_type {
            SIMILARITY_SAME_TYPE
        } else {
            SIMILARITY_CROSS_TYPE
        };

        if result.score >= threshold {
            return Ok(Some((result.node_id, result.properties.clone())));
        }
    }

    Ok(None)
}

/// Whether two entity types are compatible for merging.
///
/// Incompatible pairs (never merge regardless of similarity):
/// - person ↔ org
/// - person ↔ place
/// - code_symbol ↔ code_import
/// - date ↔ anything except date
fn types_compatible(a: &str, b: &str) -> bool {
    // "other" is compatible with everything.
    if a == "other" || b == "other" {
        return true;
    }
    // Same type is always compatible.
    if a == b {
        return true;
    }
    // Date is only compatible with date.
    if a == "date" || b == "date" {
        return false;
    }
    // Incompatible pairs.
    let pair = if a < b { (a, b) } else { (b, a) };
    !matches!(
        pair,
        ("organization", "person")
            | ("location", "person")
            | ("location", "organization")
            | ("code_import", "code_symbol")
    )
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
