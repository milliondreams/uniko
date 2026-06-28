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

    // F50 memory decay (`importance * exp(-ln(2)/half_life * age_days)`) lives as
    // the `relevance_decay` stdlib Locy rule (single decay formula); its consumer
    // `uniko_memory::rules::consume_relevance_decay` runs the rule and deletes
    // the yielded Episodes. The Locy form uses
    // `duration.inDays(datetime(), e.timestamp).days` for the age.

    /// Mean `importance` of an agent's Episodes for one `(action_type, outcome)`
    /// group. Absent `importance` defaults to `0.5`; an empty group returns
    /// `0.0`.
    ///
    /// Computed in Rust because uni-db's Locy `AVG` returns 0.0 (filed
    /// upstream) — used by the `episode_pattern_detector` consumer.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] when the query fails.
    pub async fn mean_episode_importance(
        &self,
        agent_id: &str,
        action_type: &str,
        outcome: &str,
    ) -> Result<f64> {
        let session = self.db.session();
        let result = session
            .query_with(
                "MATCH (e:Episode)-[:RECORDED_BY]->(p:Participant {participant_id: $a}) \
                 WHERE e.action_type = $at AND e.outcome = $o RETURN e.importance AS imp",
            )
            .param("a", agent_id)
            .param("at", action_type)
            .param("o", outcome)
            .fetch_all()
            .await?;
        let rows = result.rows();
        if rows.is_empty() {
            return Ok(0.0);
        }
        let sum: f64 = rows.iter().map(|r| r.get::<f64>("imp").unwrap_or(0.5)).sum();
        Ok(sum / rows.len() as f64)
    }
}
