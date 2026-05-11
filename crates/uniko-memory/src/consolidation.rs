//! P4 Consolidation — derive Facts from Observations.
//!
//! A consolidation cycle:
//! 1. Queries unprocessed Observations carrying a structured triple
//!    (`subject` and `predicate` non-null).
//! 2. Groups by `(subject, predicate)`; picks a canonical object per
//!    group (mode, tie-broken by recency).
//! 3. Upserts a Fact per group via
//!    [`KnowledgeBase::upsert_fact_by_triple`]; first observation in a
//!    cluster becomes the embedding source.
//! 4. Wires `SUPPORTED_BY` edges from every contributing Observation
//!    to its Fact.
//! 5. Writes a `ConsolidationCycle` audit node with `PROCESSED`,
//!    `CREATED`, and `INVOLVED` edges — the `PROCESSED` edges are the
//!    idempotency anchor so future cycles skip the same Observations.
//!
//! Contradiction (F38) and drift (F39) detection are deferred — they
//! require either Observation polarity or an LLM judge at consolidation
//! time, both out of scope for the first ship.

// Rust guideline compliant

use std::collections::HashMap;

use chrono::{DateTime, Utc};

use uniko_extract::embedding::embed_document;
use uniko_store::{KnowledgeBase, NodeId, UnikoError};

/// Number of Facts created and reinforced in one cycle.
#[derive(Debug, Default, Clone, Copy)]
pub struct CycleStats {
    /// Observations processed (PROCESSED edges emitted).
    pub observations_processed: usize,
    /// Facts newly created.
    pub facts_created: usize,
    /// Facts reinforced (existing Fact whose count was incremented).
    pub facts_reinforced: usize,
}

/// Maximum observations to process in a single cycle.
///
/// Caps work per cycle so a long-running ingest doesn't starve other
/// agents.  Spillover is picked up on the next sweep.
const DEFAULT_BATCH_SIZE: i64 = 500;

/// Run one consolidation cycle for `agent_id`.
///
/// Idempotent across runs: Observations already wired via `PROCESSED`
/// from any prior `ConsolidationCycle` are excluded by the query.
///
/// Returns counts useful for metrics and tests.  Caller is responsible
/// for emitting `uniko.consolidation.*` metrics from the returned
/// stats; this function only writes to the graph.
///
/// # Errors
///
/// Returns [`UnikoError::Storage`] on any database failure.
pub async fn run_cycle(
    kb: &KnowledgeBase,
    agent_id: &str,
    batch_size: Option<i64>,
) -> Result<CycleStats, UnikoError> {
    let started_at = Utc::now();
    let batch_size = batch_size.unwrap_or(DEFAULT_BATCH_SIZE).max(1);

    let observations = fetch_unprocessed_observations(kb, batch_size).await?;
    if observations.is_empty() {
        // Empty cycles are still recorded so the audit trail is
        // contiguous and metrics show a heartbeat.
        let completed_at = Utc::now();
        kb.write_consolidation_cycle(
            agent_id,
            started_at,
            completed_at,
            &[],
            &[],
            &[],
            &[],
        )
        .await?;
        return Ok(CycleStats::default());
    }

    // Group by (subject, predicate); collect contributing observations
    // and object surface forms for canonical selection.
    let mut groups: HashMap<(String, String), GroupBuilder> = HashMap::new();
    for obs in &observations {
        let entry = groups
            .entry((obs.subject.clone(), obs.predicate.clone()))
            .or_default();
        entry.contributing.push(obs.node_id);
        entry.first_observed_at = Some(match entry.first_observed_at {
            Some(prev) => prev.min(obs.observed_at),
            None => obs.observed_at,
        });
        entry.object_votes.push(ObjectVote {
            text: obs.object.clone(),
            observed_at: obs.observed_at,
            content: obs.content.clone(),
        });
    }

    let mut created_facts: Vec<NodeId> = Vec::new();
    let mut reinforced_facts: Vec<NodeId> = Vec::new();

    for ((subject, predicate), group) in groups {
        let canonical = canonical_object(&group.object_votes);
        let first_observed = group
            .first_observed_at
            .unwrap_or(started_at);

        // Compute the embedding text from the canonical (subject,
        // predicate, object) triple.  Falls back to the freshest
        // contributing observation's `content` when the object slot is
        // empty — that text is at least a paraphrase of the same claim
        // and yields a usable embedding for Phase 1 retrieval.
        let embed_text = match &canonical {
            Some(obj) => format!("{subject} {predicate} {obj}"),
            None => freshest_content(&group.object_votes)
                .unwrap_or_else(|| format!("{subject} {predicate}")),
        };
        let embedding = match embed_document(kb, &embed_text).await {
            Ok(v) => Some(v),
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    subject = %subject,
                    predicate = %predicate,
                    "fact embedding failed; storing Fact without embedding"
                );
                None
            }
        };

        let upsert = kb
            .upsert_fact_by_triple(
                &subject,
                &predicate,
                canonical.as_deref(),
                group.contributing.len() as i64,
                first_observed,
                embedding,
            )
            .await?;

        kb.attach_supported_by(upsert.node_id, &group.contributing)
            .await?;

        if upsert.was_created {
            created_facts.push(upsert.node_id);
        } else {
            reinforced_facts.push(upsert.node_id);
        }
    }

    let processed_ids: Vec<NodeId> = observations.iter().map(|o| o.node_id).collect();
    let completed_at = Utc::now();
    kb.write_consolidation_cycle(
        agent_id,
        started_at,
        completed_at,
        &processed_ids,
        &created_facts,
        &reinforced_facts,
        &[],
    )
    .await?;

    Ok(CycleStats {
        observations_processed: processed_ids.len(),
        facts_created: created_facts.len(),
        facts_reinforced: reinforced_facts.len(),
    })
}

/// Pull unprocessed Observations carrying a structured triple.
///
/// Filters out Observations from the rule-based fallback path
/// (`predicate IS NULL`) and Observations already wired to a prior
/// `ConsolidationCycle` via `PROCESSED`.  Capped at `limit` per cycle
/// to bound worst-case latency.
async fn fetch_unprocessed_observations(
    kb: &KnowledgeBase,
    limit: i64,
) -> Result<Vec<UnprocessedObs>, UnikoError> {
    let session = kb.db().session();
    let cypher = "MATCH (o:Observation) \
                  WHERE o.subject IS NOT NULL AND o.predicate IS NOT NULL \
                  AND NOT EXISTS { MATCH (:ConsolidationCycle)-[:PROCESSED]->(o) } \
                  RETURN id(o) AS nid, o.subject AS subject, o.predicate AS predicate, \
                         o.object AS object, o.content AS content, o.observed_at AS observed_at \
                  ORDER BY o.observed_at ASC \
                  LIMIT $lim";
    let result = session
        .query_with(cypher)
        .param("lim", limit)
        .fetch_all()
        .await
        .map_err(|e| UnikoError::Storage(e.to_string()))?;

    let mut out: Vec<UnprocessedObs> = Vec::with_capacity(result.rows().len());
    for row in result.rows() {
        let Ok(nid) = row.get::<i64>("nid") else {
            continue;
        };
        let Ok(subject) = row.get::<String>("subject") else {
            continue;
        };
        let Ok(predicate) = row.get::<String>("predicate") else {
            continue;
        };
        let object: Option<String> = row.get::<String>("object").ok();
        let content: String = row.get::<String>("content").unwrap_or_default();
        let observed_at = row
            .get::<String>("observed_at")
            .ok()
            .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(Utc::now);
        out.push(UnprocessedObs {
            node_id: nid,
            subject,
            predicate,
            object,
            content,
            observed_at,
        });
    }
    Ok(out)
}

/// Pick the canonical object text for a `(subject, predicate)` cluster.
///
/// Mode over non-empty object phrases; ties broken by the most recent
/// `observed_at`.  Returns `None` when no contributing Observation had
/// an object slot (the triple is `(subject, predicate, _)`).
fn canonical_object(votes: &[ObjectVote]) -> Option<String> {
    let mut tallies: HashMap<String, (usize, DateTime<Utc>)> = HashMap::new();
    for vote in votes {
        let Some(text) = vote.text.as_ref() else {
            continue;
        };
        let trimmed = text.trim();
        if trimmed.is_empty() {
            continue;
        }
        let entry = tallies
            .entry(trimmed.to_string())
            .or_insert((0, vote.observed_at));
        entry.0 += 1;
        if vote.observed_at > entry.1 {
            entry.1 = vote.observed_at;
        }
    }
    tallies
        .into_iter()
        .max_by(|a, b| {
            a.1.0
                .cmp(&b.1.0)
                .then_with(|| a.1.1.cmp(&b.1.1))
        })
        .map(|(k, _)| k)
}

/// Freshest contributing observation's `content`, used as embedding
/// fallback when no contributor produced an object slot.
fn freshest_content(votes: &[ObjectVote]) -> Option<String> {
    votes
        .iter()
        .max_by_key(|v| v.observed_at)
        .map(|v| v.content.clone())
}

#[derive(Debug)]
struct UnprocessedObs {
    node_id: NodeId,
    subject: String,
    predicate: String,
    object: Option<String>,
    content: String,
    observed_at: DateTime<Utc>,
}

#[derive(Debug, Default)]
struct GroupBuilder {
    contributing: Vec<NodeId>,
    object_votes: Vec<ObjectVote>,
    first_observed_at: Option<DateTime<Utc>>,
}

#[derive(Debug)]
struct ObjectVote {
    text: Option<String>,
    observed_at: DateTime<Utc>,
    content: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn ts(y: i32, m: u32, d: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(y, m, d, 0, 0, 0).unwrap()
    }

    #[test]
    fn canonical_object_picks_mode() {
        let votes = vec![
            ObjectVote {
                text: Some("adoption agencies".into()),
                observed_at: ts(2024, 1, 1),
                content: "Caroline researches adoption agencies".into(),
            },
            ObjectVote {
                text: Some("adoption agencies".into()),
                observed_at: ts(2024, 1, 2),
                content: "Caroline is researching adoption agencies".into(),
            },
            ObjectVote {
                text: Some("foster care".into()),
                observed_at: ts(2024, 1, 3),
                content: "Caroline also looking at foster care".into(),
            },
        ];
        assert_eq!(canonical_object(&votes).as_deref(), Some("adoption agencies"));
    }

    #[test]
    fn canonical_object_breaks_tie_by_recency() {
        let votes = vec![
            ObjectVote {
                text: Some("Rust".into()),
                observed_at: ts(2024, 1, 1),
                content: "".into(),
            },
            ObjectVote {
                text: Some("Go".into()),
                observed_at: ts(2024, 3, 1),
                content: "".into(),
            },
        ];
        assert_eq!(canonical_object(&votes).as_deref(), Some("Go"));
    }

    #[test]
    fn canonical_object_none_when_all_empty() {
        let votes = vec![ObjectVote {
            text: None,
            observed_at: ts(2024, 1, 1),
            content: "Caroline is happy".into(),
        }];
        assert!(canonical_object(&votes).is_none());
    }
}
