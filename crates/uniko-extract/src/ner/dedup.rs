//! Entity deduplication and graph persistence.
//!
//! Three-tier dedup cascade: (1) exact `entity_id` match, (2) embedding
//! similarity above threshold, (3) create new.  Creates MENTIONS edges
//! from source nodes to Entity nodes.

// Rust guideline compliant

use std::collections::HashMap;

use uni_db::Value;

use uniko_store::schema::constants::{edges, labels};
use uniko_store::storage::edges::Direction;
use uniko_store::{KnowledgeBase, NodeId};

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
/// For each entity:
/// 1. Look up by `entity_id` (format: `"{type}:{canonical_name}"`).
/// 2. If exists: increment frequency, update `last_seen` and confidence.
/// 3. If new: create Entity node with all properties.
/// 4. Create or update a `MENTIONS` edge from `source_node_id`.
///
/// # Errors
///
/// Returns a storage error if any graph operation fails.
pub async fn upsert_entities(
    kb: &KnowledgeBase,
    source_node_id: NodeId,
    deduped: Vec<(RawEntity, u32)>,
) -> uniko_store::Result<Vec<EntityMatch>> {
    let mut matches = Vec::with_capacity(deduped.len());
    let now_str = chrono::Utc::now().to_rfc3339();

    for (entity, mention_count) in deduped {
        let entity_id = format!("{}:{}", entity.entity_type.as_str(), &entity.canonical_name);

        // Check for existing entity.
        let existing = kb
            .get_node_by_ext_id(labels::ENTITY, "entity_id", &entity_id)
            .await?;

        let (node_id, was_existing) = match existing {
            Some((nid, props)) => {
                // Update existing entity.
                let old_freq = props.get("frequency").and_then(|v| v.as_i64()).unwrap_or(0);
                let mut update = HashMap::new();
                update.insert(
                    "frequency".into(),
                    Value::Int(old_freq + mention_count as i64),
                );
                update.insert("last_seen".into(), Value::String(now_str.clone()));

                let old_conf = props
                    .get("confidence")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);
                if entity.confidence > old_conf {
                    update.insert("confidence".into(), Value::Float(entity.confidence));
                }

                kb.update_node(nid, &update).await?;
                (nid, true)
            }
            None => {
                // Create new entity. Skip embedding similarity search
                // during ingestion — exact entity_id match handles
                // canonical dedup. Embedding-based dedup can be done
                // as a post-processing batch step if needed.
                let entity_type_str = entity.entity_type.as_str();
                let mut props = HashMap::new();
                props.insert("entity_id".into(), Value::String(entity_id));
                props.insert("name".into(), Value::String(entity.canonical_name.clone()));
                props.insert(
                    "entity_type".into(),
                    Value::String(entity_type_str.to_string()),
                );
                props.insert("first_seen".into(), Value::String(now_str.clone()));
                props.insert("last_seen".into(), Value::String(now_str.clone()));
                props.insert("frequency".into(), Value::Int(mention_count as i64));
                props.insert("confidence".into(), Value::Float(entity.confidence));

                let nid = kb.create_node(labels::ENTITY, &props).await?;
                (nid, false)
            }
        };

        // Create or update MENTIONS edge.
        upsert_mentions_edge(kb, source_node_id, node_id, mention_count).await?;

        matches.push(EntityMatch {
            canonical_name: entity.canonical_name,
            node_id,
            was_existing,
            mention_count,
        });
    }

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
    let vec = match crate::embedding::embed_text(kb, &embed_text).await {
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

/// Create or increment a MENTIONS edge from `source` to `entity`.
async fn upsert_mentions_edge(
    kb: &KnowledgeBase,
    source: NodeId,
    entity: NodeId,
    mention_count: u32,
) -> uniko_store::Result<()> {
    let existing_edges = kb
        .get_edges(source, edges::MENTIONS, Direction::Outgoing)
        .await?;

    if let Some(edge) = existing_edges.iter().find(|e| e.to == entity) {
        // Increment existing count.
        let old_count = edge
            .properties
            .get("count")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let mut props = HashMap::new();
        props.insert("count".into(), Value::Int(old_count + mention_count as i64));
        kb.update_edge(edge.id, &props).await?;
    } else {
        // Create new MENTIONS edge.
        let mut props = HashMap::new();
        props.insert("count".into(), Value::Int(mention_count as i64));
        kb.create_edge(edges::MENTIONS, source, entity, &props)
            .await?;
    }

    Ok(())
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
