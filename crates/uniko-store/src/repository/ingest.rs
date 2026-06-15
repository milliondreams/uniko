//! Session-chunking reads: transcript/observation aggregation and the
//! entity/participant fan-out for observation chunks. Backs
//! `uniko_extract::ingest::session_chunk`.

use crate::error::Result;
use crate::storage::KnowledgeBase;
use crate::types::NodeId;

/// One message row for transcript chunking: text + resolved speaker name
/// (`"unknown"` when the `SENT_BY` participant is absent).
#[derive(Debug, Clone)]
pub struct TranscriptRow {
    pub content: String,
    pub speaker: String,
}

/// One observation row for observation chunking: text + its subject.
#[derive(Debug, Clone)]
pub struct ObservationRow {
    pub content: String,
    pub subject: String,
}

impl KnowledgeBase {
    /// All messages in `session_id`, oldest first, with the sender's name.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`](crate::UnikoError::Storage) on
    /// query failure.
    pub async fn session_transcript_rows(&self, session_id: &str) -> Result<Vec<TranscriptRow>> {
        let session = self.db.session();
        let cypher = "\
            MATCH (m:Message)-[:IN_SESSION]->(s:Session {session_id: $sid}) \
            OPTIONAL MATCH (m)-[:SENT_BY]->(p:Participant) \
            RETURN m.content AS content, p.name AS speaker, m.timestamp AS ts \
            ORDER BY m.timestamp";
        let result = session
            .query_with(cypher)
            .param("sid", session_id)
            .fetch_all()
            .await?;
        Ok(result
            .rows()
            .iter()
            .map(|row| TranscriptRow {
                content: row.get("content").unwrap_or_default(),
                speaker: row.get("speaker").unwrap_or_else(|_| "unknown".to_string()),
            })
            .collect())
    }

    /// Existing observation-chunk node ids for `session_id` (idempotency
    /// check for `chunk_type = 'observation'`).
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`](crate::UnikoError::Storage) on
    /// query failure.
    pub async fn session_observation_chunk_ids(&self, session_id: &str) -> Result<Vec<NodeId>> {
        let session = self.db.session();
        let cypher = "\
            MATCH (s:Session {session_id: $sid})-[:HAS_CHUNK]->(c:Chunk {chunk_type: 'observation'}) \
            RETURN id(c) AS cid";
        let result = session
            .query_with(cypher)
            .param("sid", session_id)
            .fetch_all()
            .await?;
        Ok(result
            .rows()
            .iter()
            .filter_map(|r| r.get::<NodeId>("cid").ok())
            .collect())
    }

    /// All Observations linked to messages in `session_id`, oldest first.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`](crate::UnikoError::Storage) on
    /// query failure.
    pub async fn session_observation_rows(&self, session_id: &str) -> Result<Vec<ObservationRow>> {
        let session = self.db.session();
        let cypher = "\
            MATCH (o:Observation)-[:OBSERVED_IN]->(m:Message)-[:IN_SESSION]->(s:Session {session_id: $sid}) \
            RETURN o.content AS content, o.subject AS subject \
            ORDER BY m.timestamp";
        let result = session
            .query_with(cypher)
            .param("sid", session_id)
            .fetch_all()
            .await?;
        Ok(result
            .rows()
            .iter()
            .map(|row| ObservationRow {
                content: row.get("content").unwrap_or_default(),
                subject: row.get("subject").unwrap_or_default(),
            })
            .collect())
    }

    /// Distinct Entity node ids referenced (via `ABOUT`) by observations
    /// in `session_id`.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`](crate::UnikoError::Storage) on
    /// query failure.
    pub async fn session_observation_entity_ids(&self, session_id: &str) -> Result<Vec<NodeId>> {
        let session = self.db.session();
        let cypher = "\
            MATCH (o:Observation)-[:OBSERVED_IN]->(m:Message)-[:IN_SESSION]->(s:Session {session_id: $sid}) \
            MATCH (o)-[:ABOUT]->(e:Entity) \
            RETURN DISTINCT id(e) AS eid";
        let result = session
            .query_with(cypher)
            .param("sid", session_id)
            .fetch_all()
            .await?;
        Ok(result
            .rows()
            .iter()
            .filter_map(|row| row.get::<NodeId>("eid").ok())
            .collect())
    }

    /// Distinct Participant node ids referenced (via `ABOUT`) by
    /// observations in `session_id`.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`](crate::UnikoError::Storage) on
    /// query failure.
    pub async fn session_observation_participant_ids(
        &self,
        session_id: &str,
    ) -> Result<Vec<NodeId>> {
        let session = self.db.session();
        let cypher = "\
            MATCH (o:Observation)-[:OBSERVED_IN]->(m:Message)-[:IN_SESSION]->(s:Session {session_id: $sid}) \
            MATCH (o)-[:ABOUT]->(p:Participant) \
            RETURN DISTINCT id(p) AS pid";
        let result = session
            .query_with(cypher)
            .param("sid", session_id)
            .fetch_all()
            .await?;
        Ok(result
            .rows()
            .iter()
            .filter_map(|row| row.get::<NodeId>("pid").ok())
            .collect())
    }
}
