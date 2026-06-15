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
}
