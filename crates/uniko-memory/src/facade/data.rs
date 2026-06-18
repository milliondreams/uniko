//! Addressed retrieval — dereference recall sources by id.
//!
//! [`crate::recall::RecallItem::sources`] hands back *references*
//! (`message_id` / `artifact_id`); this handle turns them into content.
//! Reached via [`Agent::data`](crate::Agent::data) so the high-frequency
//! recall verbs stay flat while by-id lookup lives in one coherent place.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use uniko_store::{KnowledgeBase, UnikoError, Value};

/// A retrieved conversation message.
#[derive(Debug, Clone)]
pub struct MessageView {
    /// External message id.
    pub message_id: String,
    /// External id of the sending participant.
    pub sender_id: String,
    /// The message text.
    pub content: String,
    /// When the message was recorded.
    pub timestamp: DateTime<Utc>,
    /// External id of the session the message belongs to.
    pub session_id: String,
    /// External ids of the addressed recipients.
    pub addressed_to: Vec<String>,
    /// External ids of artifacts attached to this turn.
    pub attachments: Vec<String>,
}

/// A retrieved artifact (attachment or standalone document).
#[derive(Debug, Clone)]
pub struct ArtifactView {
    /// External artifact id.
    pub artifact_id: String,
    /// Artifact kind (`text` / `markdown` / `pdf` / …).
    pub kind: String,
    /// MIME type, when known.
    pub mime: Option<String>,
    /// Source path, when ingested from a file.
    pub path: Option<String>,
    /// Text reassembled from the artifact's chunks (in order).
    pub text: String,
    /// The message this artifact was attached to, when it was shared in a turn.
    pub attached_to_message: Option<String>,
}

/// Addressed (by-id) retrieval over an agent's store.
///
/// Borrows the agent's [`KnowledgeBase`]; obtain one with
/// [`Agent::data`](crate::Agent::data). All lookups return `Ok(None)` when
/// the id is unknown rather than erroring.
#[derive(Debug)]
pub struct Data<'a> {
    kb: &'a KnowledgeBase,
}

impl<'a> Data<'a> {
    /// Construct a handle over `kb`. Crate-internal; callers use
    /// [`Agent::data`](crate::Agent::data).
    pub(crate) fn new(kb: &'a KnowledgeBase) -> Self {
        Self { kb }
    }

    /// Fetch a message by its external id.
    ///
    /// Returns `Ok(None)` when no message has that id.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] when a graph query fails.
    pub async fn message(&self, message_id: &str) -> Result<Option<MessageView>, UnikoError> {
        let params = str_param("id", message_id);
        let rows = self
            .kb
            .query_cypher(
                "MATCH (m:Message {message_id: $id}) \
                 OPTIONAL MATCH (m)-[:SENT_BY]->(s:Participant) \
                 OPTIONAL MATCH (m)-[:IN_SESSION]->(sess:Session) \
                 RETURN m.content AS content, m.timestamp AS ts, \
                        s.participant_id AS sender, sess.session_id AS session",
                &params,
            )
            .await?;
        let Some(row) = rows.first() else {
            return Ok(None);
        };
        let addressed_to = self
            .ext_id_list(
                "MATCH (m:Message {message_id: $id})-[:ADDRESSED_TO]->(p:Participant) \
                 RETURN p.participant_id AS v",
                message_id,
            )
            .await?;
        let attachments = self
            .ext_id_list(
                "MATCH (a:Artifact)-[:ATTACHED_TO]->(m:Message {message_id: $id}) \
                 RETURN a.artifact_id AS v",
                message_id,
            )
            .await?;
        Ok(Some(MessageView {
            message_id: message_id.to_string(),
            sender_id: cell_str(row, "sender").unwrap_or_default(),
            content: cell_str(row, "content").unwrap_or_default(),
            timestamp: cell_datetime(row, "ts"),
            session_id: cell_str(row, "session").unwrap_or_default(),
            addressed_to,
            attachments,
        }))
    }

    /// Fetch an artifact's metadata + reassembled text by its external id.
    ///
    /// Returns `Ok(None)` when no artifact has that id. The text is
    /// reassembled from the artifact's direct `HAS_CHUNK` chunks in index
    /// order; for tiered-PDF artifacts (text under page→block chains) the
    /// text may be empty — fetch [`artifact_bytes`](Self::artifact_bytes)
    /// for the original.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] when a graph query fails.
    pub async fn artifact(&self, artifact_id: &str) -> Result<Option<ArtifactView>, UnikoError> {
        let params = str_param("id", artifact_id);
        let rows = self
            .kb
            .query_cypher(
                "MATCH (a:Artifact {artifact_id: $id}) \
                 OPTIONAL MATCH (a)-[:ATTACHED_TO]->(m:Message) \
                 RETURN a.kind AS kind, a.mime_type AS mime, a.path AS path, \
                        m.message_id AS attached",
                &params,
            )
            .await?;
        let Some(row) = rows.first() else {
            return Ok(None);
        };
        let chunk_rows = self
            .kb
            .query_cypher(
                "MATCH (a:Artifact {artifact_id: $id})-[:HAS_CHUNK]->(c:Chunk) \
                 RETURN c.text AS text ORDER BY c.index",
                &params,
            )
            .await?;
        let text = chunk_rows
            .iter()
            .filter_map(|r| cell_str(r, "text"))
            .collect::<Vec<_>>()
            .join("\n");
        Ok(Some(ArtifactView {
            artifact_id: artifact_id.to_string(),
            kind: cell_str(row, "kind").unwrap_or_default(),
            mime: cell_str(row, "mime"),
            path: cell_str(row, "path"),
            text,
            attached_to_message: cell_str(row, "attached"),
        }))
    }

    /// Fetch an artifact's original bytes by its external id.
    ///
    /// Resolves the artifact's content blob and reads it (inline or from the
    /// backing blob store). Returns `Ok(None)` when the artifact or its
    /// content is unknown.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] when the lookup or blob read fails.
    pub async fn artifact_bytes(&self, artifact_id: &str) -> Result<Option<Vec<u8>>, UnikoError> {
        let rows = self
            .kb
            .query_cypher(
                "MATCH (a:Artifact {artifact_id: $id})-[:HAS_CONTENT]->(ac:ArtifactContent) \
                 RETURN ac.content_id AS cid",
                &str_param("id", artifact_id),
            )
            .await?;
        let Some(content_id) = rows.first().and_then(|r| cell_str(r, "cid")) else {
            return Ok(None);
        };
        Ok(Some(self.kb.fetch_blob(&content_id).await?))
    }

    /// Collect a single `v`-aliased external-id column into a `Vec`.
    async fn ext_id_list(&self, cypher: &str, id: &str) -> Result<Vec<String>, UnikoError> {
        let rows = self.kb.query_cypher(cypher, &str_param("id", id)).await?;
        Ok(rows.iter().filter_map(|r| cell_str(r, "v")).collect())
    }
}

/// Build a single `$key = string` Cypher parameter map.
fn str_param(key: &str, value: &str) -> HashMap<String, Value> {
    HashMap::from([(key.to_string(), Value::String(value.to_string()))])
}

/// Read a string cell from a decoded row.
fn cell_str(row: &HashMap<String, Value>, key: &str) -> Option<String> {
    match row.get(key) {
        Some(Value::String(s)) => Some(s.clone()),
        _ => None,
    }
}

/// Read a datetime cell, falling back to the Unix epoch when absent/malformed.
fn cell_datetime(row: &HashMap<String, Value>, key: &str) -> DateTime<Utc> {
    row.get(key)
        .and_then(|v| uniko_store::types::datetime_from_value(v, "MessageView.timestamp").ok())
        .unwrap_or_else(|| DateTime::<Utc>::from_timestamp(0, 0).expect("epoch is valid"))
}
