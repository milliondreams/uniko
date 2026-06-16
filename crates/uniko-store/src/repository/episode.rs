//! Episode reads: locate the predecessor Episode for `FOLLOWED_BY`
//! chaining. Backs `uniko_memory::episode`.

use chrono::{DateTime, Utc};

use crate::error::{Result, UnikoError};
use crate::storage::KnowledgeBase;
use crate::types::{NodeId, datetime_from_value, datetime_value};

impl KnowledgeBase {
    /// Find the most recent Episode of `agent_id` whose `timestamp` falls
    /// in `[earliest, now]`.
    ///
    /// Returns `(node_id, timestamp)` of the latest match, or `None` when
    /// no candidate exists. The caller owns the window policy (it derives
    /// `earliest` from its `FOLLOWED_BY` window).
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] when the query fails or the matched
    /// Episode is missing its `timestamp` property.
    pub async fn previous_episode_in_window(
        &self,
        agent_id: &str,
        earliest: DateTime<Utc>,
        now: DateTime<Utc>,
    ) -> Result<Option<(NodeId, DateTime<Utc>)>> {
        let cypher = "MATCH (e:Episode)-[:RECORDED_BY]->(p:Participant {participant_id: $aid}) \
                      WHERE e.timestamp >= $earliest AND e.timestamp <= $now \
                      RETURN e \
                      ORDER BY e.timestamp DESC \
                      LIMIT 1";
        let session = self.db.session();
        let result = session
            .query_with(cypher)
            .param("aid", agent_id)
            .param("earliest", datetime_value(earliest))
            .param("now", datetime_value(now))
            .fetch_all()
            .await?;

        let Some(row) = result.rows().first() else {
            return Ok(None);
        };
        let node: uni_db::Node = row.get("e")?;
        let vid = node.vid.as_u64() as i64;
        let ts_value = node
            .properties
            .get("timestamp")
            .ok_or_else(|| UnikoError::Storage("Episode.timestamp missing".into()))?;
        let ts = datetime_from_value(ts_value, "Episode.timestamp")?;
        Ok(Some((vid, ts)))
    }

    /// Prune an agent's Episodes whose age-decayed importance has fallen
    /// to or below `prune_below` (F50 memory decay).
    ///
    /// Decay is `importance * exp(-ln(2)/half_life_days * age_days)`,
    /// computed from the *immutable* stored `importance` so repeated
    /// cycles never compound. Absent `importance` defaults to `0.5`.
    /// Returns the number of Episodes deleted.
    ///
    /// A non-positive `half_life_days` disables decay and returns `0`.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] when the query or a delete fails.
    pub async fn prune_decayed_episodes(
        &self,
        agent_id: &str,
        half_life_days: f64,
        prune_below: f64,
        now: DateTime<Utc>,
    ) -> Result<usize> {
        if half_life_days <= 0.0 {
            return Ok(0);
        }
        let decay_rate = std::f64::consts::LN_2 / half_life_days;

        let session = self.db.session();
        let result = session
            .query_with(
                "MATCH (e:Episode)-[:RECORDED_BY]->(p:Participant {participant_id: $aid}) \
                 RETURN e",
            )
            .param("aid", agent_id)
            .fetch_all()
            .await?;

        let mut to_prune: Vec<NodeId> = Vec::new();
        for row in result.rows() {
            let node: uni_db::Node = row.get("e")?;
            let importance = match node.properties.get("importance") {
                Some(crate::Value::Float(f)) => *f,
                _ => 0.5,
            };
            let Some(ts_value) = node.properties.get("timestamp") else {
                continue;
            };
            let ts = datetime_from_value(ts_value, "Episode.timestamp")?;
            let age_days = (now - ts).num_seconds() as f64 / 86_400.0;
            let decayed = importance * (-decay_rate * age_days.max(0.0)).exp();
            if decayed <= prune_below {
                to_prune.push(node.vid.as_u64() as NodeId);
            }
        }

        let mut pruned = 0usize;
        for nid in to_prune {
            if self.delete_node(nid).await? {
                pruned += 1;
            }
        }
        Ok(pruned)
    }
}
