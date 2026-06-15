//! Access-control reads: participant membership resolution and batch
//! node-visibility lookup. Backs `uniko_memory::policy`.

use std::collections::HashMap;

use uni_db::Value;

use crate::error::Result;
use crate::storage::KnowledgeBase;
use crate::types::NodeId;

/// A participant's resolved group memberships, used to decide which
/// `team:`/`org:`-scoped items they may see.
///
/// `teams` and `orgs` are non-empty membership ids; `orgs` is the union
/// of direct `MEMBER_OF` orgs and orgs reached transitively through the
/// participant's teams. Neither list is deduplicated across the two org
/// sources — callers typically collect into a set.
#[derive(Debug, Clone, Default)]
pub struct ParticipantMemberships {
    /// Team ids reached via `(:Participant)-[:PART_OF_TEAM]->(:Team)`.
    pub teams: Vec<String>,
    /// Org ids: direct `MEMBER_OF` plus `TEAM_IN_ORG` from the teams.
    pub orgs: Vec<String>,
}

impl KnowledgeBase {
    /// Resolve the group memberships for `participant_id`.
    ///
    /// An unknown participant yields empty membership lists (not an
    /// error) — they can still see `public` and their own `private:`
    /// items.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`](crate::UnikoError::Storage) when
    /// the lookup fails.
    pub async fn resolve_participant_memberships(
        &self,
        participant_id: &str,
    ) -> Result<ParticipantMemberships> {
        let session = self.db.session();
        let cypher = "MATCH (p:Participant) WHERE p.participant_id = $pid \
                      OPTIONAL MATCH (p)-[:PART_OF_TEAM]->(t:Team) \
                      OPTIONAL MATCH (t)-[:TEAM_IN_ORG]->(o1:Organization) \
                      OPTIONAL MATCH (p)-[:MEMBER_OF]->(o2:Organization) \
                      RETURN collect(DISTINCT t.team_id) AS teams, \
                             collect(DISTINCT o1.org_id) AS orgs1, \
                             collect(DISTINCT o2.org_id) AS orgs2";
        let result = session
            .query_with(cypher)
            .param("pid", participant_id)
            .fetch_all()
            .await?;

        let mut memberships = ParticipantMemberships::default();
        if let Some(row) = result.rows().first() {
            let team_ids: Vec<String> = row.get("teams").unwrap_or_default();
            let org_ids_a: Vec<String> = row.get("orgs1").unwrap_or_default();
            let org_ids_b: Vec<String> = row.get("orgs2").unwrap_or_default();
            memberships
                .teams
                .extend(team_ids.into_iter().filter(|t| !t.is_empty()));
            memberships.orgs.extend(
                org_ids_a
                    .into_iter()
                    .chain(org_ids_b)
                    .filter(|o| !o.is_empty()),
            );
        }
        Ok(memberships)
    }

    /// Batch-fetch the `visibility` property for `node_ids`.
    ///
    /// Returns a map from node id to its visibility string (`""` when the
    /// property is absent). Nodes that don't exist are simply absent from
    /// the map.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`](crate::UnikoError::Storage) when
    /// the lookup fails.
    pub async fn fetch_visibilities(&self, node_ids: &[NodeId]) -> Result<HashMap<NodeId, String>> {
        let session = self.db.session();
        let ids: Vec<Value> = node_ids.iter().map(|n| Value::Int(*n)).collect();
        let cypher = "MATCH (n) WHERE id(n) IN $ids \
                      RETURN id(n) AS nid, coalesce(n.visibility, '') AS vis";
        let result = session
            .query_with(cypher)
            .param("ids", Value::List(ids))
            .fetch_all()
            .await?;
        let mut out = HashMap::new();
        for row in result.rows() {
            let Ok(nid) = row.get::<i64>("nid") else {
                continue;
            };
            let vis: String = row.get("vis").unwrap_or_default();
            out.insert(nid, vis);
        }
        Ok(out)
    }
}
