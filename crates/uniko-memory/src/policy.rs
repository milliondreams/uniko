//! Access-control policy for Facts and Observations (F66).
//!
//! Visibility format on the node's `visibility` property:
//! - `null` or `""` or `"public"` — visible to every viewer.
//! - `"private:{participant_id}"` — visible only when the viewer's
//!   `participant_id` matches.
//! - `"team:{team_id}"` — visible when the viewer is a `PART_OF_TEAM`
//!   member of `team_id`.
//! - `"org:{org_id}"` — visible when the viewer is a `MEMBER_OF`
//!   member of `org_id` (directly or via team membership).
//!
//! The policy is applied as a *filter* over the recall cascade's
//! [`crate::recall::ContextBundle`] just before it's handed to the
//! caller.  We never query unfiltered visibility from outside this
//! module; consolidation and ingest write the field and the recall
//! filter reads it back at assembly time.

// Rust guideline compliant

use std::collections::HashSet;

use uniko_store::{KnowledgeBase, NodeId, UnikoError};

use crate::recall::{ContextBundle, RecallItem};

/// Viewer identity used for filtering.
///
/// Construct via [`Viewer::new`] from an external `participant_id`.
/// The recall filter resolves the participant's team and org memberships
/// once per call and caches them in this struct so per-item checks
/// stay O(1).
#[derive(Debug, Clone)]
pub struct Viewer {
    /// External `participant_id` for the requesting participant.
    pub participant_id: String,
    /// `team_id` values the viewer belongs to.
    pub teams: HashSet<String>,
    /// `org_id` values the viewer belongs to (directly or via team).
    pub orgs: HashSet<String>,
}

impl Viewer {
    /// Build a viewer for `participant_id` by resolving its
    /// memberships against the store.
    ///
    /// Membership is the union of:
    /// - Direct `(:Participant)-[:MEMBER_OF]->(:Organization)` edges.
    /// - Teams reached via `(:Participant)-[:PART_OF_TEAM]->(:Team)`
    ///   plus the orgs reached from those teams via
    ///   `(:Team)-[:TEAM_IN_ORG]->(:Organization)`.
    ///
    /// An unknown `participant_id` returns a viewer with empty
    /// memberships — they can still see anything `"public"` or
    /// `"private:{their_id}"`.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] when the membership lookup fails.
    pub async fn new(
        kb: &KnowledgeBase,
        participant_id: impl Into<String>,
    ) -> Result<Self, UnikoError> {
        let pid = participant_id.into();
        let session = kb.db().session();
        let cypher = "MATCH (p:Participant) WHERE p.participant_id = $pid \
                      OPTIONAL MATCH (p)-[:PART_OF_TEAM]->(t:Team) \
                      OPTIONAL MATCH (t)-[:TEAM_IN_ORG]->(o1:Organization) \
                      OPTIONAL MATCH (p)-[:MEMBER_OF]->(o2:Organization) \
                      RETURN collect(DISTINCT t.team_id) AS teams, \
                             collect(DISTINCT o1.org_id) AS orgs1, \
                             collect(DISTINCT o2.org_id) AS orgs2";
        let result = session
            .query_with(cypher)
            .param("pid", pid.as_str())
            .fetch_all()
            .await
            .map_err(|e| UnikoError::Storage(e.to_string()))?;

        let mut teams = HashSet::new();
        let mut orgs = HashSet::new();
        if let Some(row) = result.rows().first() {
            let team_ids: Vec<String> = row.get("teams").unwrap_or_default();
            let org_ids_a: Vec<String> = row.get("orgs1").unwrap_or_default();
            let org_ids_b: Vec<String> = row.get("orgs2").unwrap_or_default();
            for t in team_ids {
                if !t.is_empty() {
                    teams.insert(t);
                }
            }
            for o in org_ids_a.into_iter().chain(org_ids_b) {
                if !o.is_empty() {
                    orgs.insert(o);
                }
            }
        }
        Ok(Self {
            participant_id: pid,
            teams,
            orgs,
        })
    }

    /// Construct a viewer from already-resolved memberships.
    ///
    /// Useful for tests and for callers that maintain their own
    /// participant directory and don't want a graph round-trip.
    #[must_use]
    pub fn from_parts(
        participant_id: impl Into<String>,
        teams: impl IntoIterator<Item = String>,
        orgs: impl IntoIterator<Item = String>,
    ) -> Self {
        Self {
            participant_id: participant_id.into(),
            teams: teams.into_iter().collect(),
            orgs: orgs.into_iter().collect(),
        }
    }
}

/// Decide whether a single visibility string admits `viewer`.
///
/// Exposed for callers that want to enforce visibility outside the
/// recall cascade (e.g. on direct Fact lookups).
#[must_use]
pub fn visibility_admits(visibility: Option<&str>, viewer: &Viewer) -> bool {
    let raw = visibility.unwrap_or("").trim();
    if raw.is_empty() || raw == "public" {
        return true;
    }
    if let Some(pid) = raw.strip_prefix("private:") {
        return pid.trim() == viewer.participant_id;
    }
    if let Some(tid) = raw.strip_prefix("team:") {
        return viewer.teams.contains(tid.trim());
    }
    if let Some(oid) = raw.strip_prefix("org:") {
        return viewer.orgs.contains(oid.trim());
    }
    // Unknown scheme: fail closed so a malformed visibility tag
    // doesn't accidentally leak data.
    false
}

/// Filter a [`ContextBundle`] in-place to remove items the viewer
/// cannot see.
///
/// Only Fact and Observation items are policy-checked — other node
/// types have no visibility scope in spec v6.  Items are looked up
/// in batch (one Cypher query per call) so the filter stays O(items).
///
/// # Errors
///
/// Returns [`UnikoError::Storage`] when the visibility lookup fails.
pub async fn filter_bundle(
    kb: &KnowledgeBase,
    bundle: &mut ContextBundle,
    viewer: &Viewer,
) -> Result<(), UnikoError> {
    let policy_node_ids: Vec<NodeId> = bundle
        .items
        .iter()
        .filter(|i| matches!(i.node_type.as_str(), "Fact" | "Observation"))
        .map(|i| i.node_id)
        .collect();
    if policy_node_ids.is_empty() {
        return Ok(());
    }

    let visibilities = fetch_visibilities(kb, &policy_node_ids).await?;
    bundle
        .items
        .retain(|item| visibility_for(item, &visibilities).map_or(true, |v| visibility_admits(Some(v.as_str()), viewer)));
    // Recompute token count after filtering — same heuristic the
    // recall cascade and working memory use.
    bundle.total_tokens = bundle.items.len() * 50;
    Ok(())
}

fn visibility_for<'a>(
    item: &RecallItem,
    visibilities: &'a std::collections::HashMap<NodeId, String>,
) -> Option<&'a String> {
    if !matches!(item.node_type.as_str(), "Fact" | "Observation") {
        return None;
    }
    visibilities.get(&item.node_id)
}

async fn fetch_visibilities(
    kb: &KnowledgeBase,
    node_ids: &[NodeId],
) -> Result<std::collections::HashMap<NodeId, String>, UnikoError> {
    let session = kb.db().session();
    let ids: Vec<uni_db::Value> = node_ids
        .iter()
        .map(|n| uni_db::Value::Int(*n))
        .collect();
    let cypher = "MATCH (n) WHERE id(n) IN $ids \
                  RETURN id(n) AS nid, coalesce(n.visibility, '') AS vis";
    let result = session
        .query_with(cypher)
        .param("ids", uni_db::Value::List(ids))
        .fetch_all()
        .await
        .map_err(|e| UnikoError::Storage(e.to_string()))?;
    let mut out = std::collections::HashMap::new();
    for row in result.rows() {
        let Ok(nid) = row.get::<i64>("nid") else {
            continue;
        };
        let vis: String = row.get("vis").unwrap_or_default();
        out.insert(nid, vis);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn viewer(pid: &str, teams: &[&str], orgs: &[&str]) -> Viewer {
        Viewer::from_parts(
            pid,
            teams.iter().map(|s| s.to_string()),
            orgs.iter().map(|s| s.to_string()),
        )
    }

    #[test]
    fn public_admits_everyone() {
        let v = viewer("alice", &[], &[]);
        assert!(visibility_admits(None, &v));
        assert!(visibility_admits(Some(""), &v));
        assert!(visibility_admits(Some("public"), &v));
    }

    #[test]
    fn private_matches_only_owner() {
        let alice = viewer("alice", &[], &[]);
        let bob = viewer("bob", &[], &[]);
        assert!(visibility_admits(Some("private:alice"), &alice));
        assert!(!visibility_admits(Some("private:alice"), &bob));
    }

    #[test]
    fn team_visibility_checks_membership() {
        let alice = viewer("alice", &["red"], &[]);
        let bob = viewer("bob", &["blue"], &[]);
        assert!(visibility_admits(Some("team:red"), &alice));
        assert!(!visibility_admits(Some("team:red"), &bob));
    }

    #[test]
    fn org_visibility_checks_membership() {
        let alice = viewer("alice", &[], &["acme"]);
        let bob = viewer("bob", &[], &["other"]);
        assert!(visibility_admits(Some("org:acme"), &alice));
        assert!(!visibility_admits(Some("org:acme"), &bob));
    }

    #[test]
    fn unknown_scheme_fails_closed() {
        let v = viewer("alice", &[], &[]);
        assert!(!visibility_admits(Some("secret:42"), &v));
        assert!(!visibility_admits(Some("hunter2"), &v));
    }
}
