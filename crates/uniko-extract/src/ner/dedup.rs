//! Entity deduplication and graph persistence.
//!
//! Three-tier dedup cascade: (1) exact `entity_id` match, (2) embedding
//! similarity above threshold, (3) create new.  Creates MENTIONS edges
//! from source nodes to Entity nodes.

use std::collections::HashMap;

use uniko_store::schema::constants::{edges, labels};
use uniko_store::types::datetime_value;
use uniko_store::{KnowledgeBase, NodeId, Transaction, UnikoError, Value};

use super::types::{EntityMatch, EntityType, ExtractionSource, RawEntity};

/// Drop ONNX-NER entities overlapping a format-structured rule entity.
///
/// Email and URL entities are defined by their surface *format*, so the
/// regex extractor is authoritative for them. When the ONNX NER cascade
/// tags the same characters differently — an email address reads as
/// name-like and gets a `Person` tag, for instance — that guess is
/// spurious: keeping it would mint a second, bogus `:Entity` for one
/// span. This drops any [`ExtractionSource::OnnxModel`] entity whose byte
/// span overlaps a rule-based [`EntityType::Email`] or
/// [`EntityType::Url`], leaving the rule entity as the single canonical
/// one. Cross-source overlap resolution: the in-regex pass already
/// suppresses overlaps within `rules`, but ONNX entities are merged in
/// afterward and never cross that filter.
///
/// A no-op on builds without the `onnx` feature, which emit no
/// `OnnxModel` entities.
///
/// # Examples
///
/// ```ignore
/// let kept = suppress_onnx_over_structured(all_raw);
/// ```
pub fn suppress_onnx_over_structured(mut raw: Vec<RawEntity>) -> Vec<RawEntity> {
    let structured: Vec<(usize, usize)> = raw
        .iter()
        .filter(|e| {
            e.source == ExtractionSource::RuleBased
                && matches!(e.entity_type, EntityType::Email | EntityType::Url)
        })
        .map(|e| (e.start_byte, e.end_byte))
        .collect();
    if structured.is_empty() {
        return raw;
    }
    raw.retain(|e| {
        e.source != ExtractionSource::OnnxModel
            || !structured
                .iter()
                .any(|&(s, t)| spans_overlap(e.start_byte, e.end_byte, s, t))
    });
    raw
}

/// Whether two half-open byte ranges `[s1, e1)` and `[s2, e2)` overlap.
fn spans_overlap(s1: usize, e1: usize, s2: usize, e2: usize) -> bool {
    s1 < e2 && s2 < e1
}

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

/// Pre-tx prep: compute the canonical `entity_id`s for the batch and
/// snapshot `now`. The caller uses the ids to acquire the per-entity RMW
/// locks ([`KnowledgeBase::lock_entity_ids`]) BEFORE opening the write
/// tx; the authoritative existence read then happens in
/// [`apply_entity_upsert`] *inside* that tx and *under* those locks.
///
/// (Previously this did the existence read here, outside any tx — but a
/// pre-tx, pre-lock read is non-authoritative: a concurrent ingest could
/// create the same entity in the gap before the writer locked and opened
/// its tx, so the writer would still CREATE a duplicate. The read now
/// lives in `apply` on a post-lock snapshot. See issue #1 / RC2.)
pub async fn prepare_entity_upsert(
    _kb: &KnowledgeBase,
    deduped: Vec<(RawEntity, u32)>,
) -> uniko_store::Result<EntityUpsertPrep> {
    // Canonical entity_id (issue #1): the single shared derivation, keyed on
    // (lower name, canonical type). `EntityType::as_str()` already yields the
    // shared lowercase vocabulary, so overlapping types unify with the
    // action/consolidation paths.
    let entity_ids: Vec<String> = deduped
        .iter()
        .map(|(entity, _)| {
            uniko_store::id::entity_id(&entity.canonical_name, entity.entity_type.as_str())
        })
        .collect();

    Ok(EntityUpsertPrep {
        deduped,
        entity_ids,
        now_value: datetime_value(chrono::Utc::now()),
    })
}

/// Output of [`prepare_entity_upsert`]. Holds the new-vs-existing split
/// inputs needed at apply time. The existence map is *not* precomputed
/// here — `apply_entity_upsert` reads it authoritatively under lock.
///
/// `Clone` so the atomic ingest retry loop can hand a fresh copy to
/// [`apply_entity_upsert`] (which consumes it by value) on each attempt.
#[derive(Debug, Clone)]
pub struct EntityUpsertPrep {
    /// Raw entities (with mention counts) to upsert, in input order.
    pub deduped: Vec<(RawEntity, u32)>,
    /// Pre-computed canonical entity ids, parallel to `deduped`.
    pub entity_ids: Vec<String>,
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
        now_value,
    } = prep;

    // Authoritative existence read, INSIDE the caller's tx and UNDER the
    // per-entity RMW locks the caller holds (see `lock_entity_ids`). The
    // tx snapshot was taken after the locks were acquired, so this read
    // reflects every committed entity and no concurrent writer can create
    // one of `entity_ids` until we commit and release. This is what makes
    // the check-then-create below race-free (issue #1 / RC2).
    let existing: HashMap<String, (NodeId, i64, f64)> =
        kb.fetch_entities_for_upsert_in_tx(tx, &entity_ids).await?;

    // Partition into (existing → batched UPDATE) and (new → batched
    // CREATE). `matches` is pre-filled with the right length so we can
    // fill node_id by index for the new rows once batch_create_nodes
    // returns them.
    let mut matches: Vec<EntityMatch> = Vec::with_capacity(deduped.len());
    let mut updates: Vec<(NodeId, i64, f64)> = Vec::new();
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

            updates.push((nid, new_freq, new_conf));

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
    // uni-db 2.0 (#53/#54) rewrites `UNWIND … MATCH WHERE id(n) = col`
    // into a HashJoin, so the id()-equality match no longer needs an
    // :Entity label hint to avoid the per-row multi-label scan that
    // previously cost ~18 ms/row (investigated 2026-05-20, fixed upstream).
    let phase3_start = std::time::Instant::now();
    let update_count = updates.len();
    if !updates.is_empty() {
        kb.batch_update_entity_counters_in_tx(tx, &updates, now_value)
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

    fn raw_span(
        name: &str,
        entity_type: EntityType,
        source: ExtractionSource,
        start: usize,
        end: usize,
    ) -> RawEntity {
        RawEntity {
            surface_form: name.to_string(),
            canonical_name: name.to_lowercase(),
            entity_type,
            confidence: 0.9,
            source,
            start_byte: start,
            end_byte: end,
        }
    }

    #[test]
    fn test_suppress_onnx_person_overlapping_email() {
        // The NER cascade tags an email address as a PERSON over the same
        // span the email regex matched; the ONNX guess must be dropped.
        let raw = vec![
            raw_span(
                "dedup@example.com",
                EntityType::Email,
                ExtractionSource::RuleBased,
                12,
                29,
            ),
            raw_span(
                "dedup@example.com",
                EntityType::Person,
                ExtractionSource::OnnxModel,
                12,
                29,
            ),
        ];
        let kept = suppress_onnx_over_structured(raw);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].entity_type, EntityType::Email);
        assert_eq!(kept[0].source, ExtractionSource::RuleBased);
    }

    #[test]
    fn test_suppress_keeps_nonoverlapping_onnx() {
        // A PERSON elsewhere in the text is unrelated to the email span
        // and must survive.
        let raw = vec![
            raw_span(
                "dedup@example.com",
                EntityType::Email,
                ExtractionSource::RuleBased,
                12,
                29,
            ),
            raw_span(
                "Alice",
                EntityType::Person,
                ExtractionSource::OnnxModel,
                0,
                5,
            ),
        ];
        let kept = suppress_onnx_over_structured(raw);
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn test_suppress_noop_without_structured() {
        // No Email/URL entity → nothing is suppressed even on overlap.
        let raw = vec![
            raw_span(
                "Alice",
                EntityType::Person,
                ExtractionSource::RuleBased,
                0,
                5,
            ),
            raw_span(
                "Alice",
                EntityType::Person,
                ExtractionSource::OnnxModel,
                0,
                5,
            ),
        ];
        let kept = suppress_onnx_over_structured(raw);
        assert_eq!(kept.len(), 2);
    }
}
