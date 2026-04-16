//! Persistent dead-letter queue backed by the knowledge graph.
//!
//! Failed pipeline items are stored as `DeadLetter` nodes and can be
//! retried automatically or managed manually.

// Rust guideline compliant

use std::collections::HashMap;
use std::sync::Arc;

use uni_db::Value;

use uniko_store::{KnowledgeBase, NodeId, UnikoError};

/// Information about a pending dead-letter item.
#[derive(Debug, Clone)]
pub struct DeadLetterInfo {
    /// Internal node ID of the DeadLetter node.
    pub id: NodeId,
    /// Pipeline step that failed.
    pub step: String,
    /// Error message.
    pub error: String,
    /// Node ID of the item that failed processing.
    pub node_ref: NodeId,
    /// How many times this item has been retried.
    pub retry_count: i64,
    /// Maximum automatic retries allowed.
    pub max_retries: i64,
}

/// Persistent dead-letter queue using `DeadLetter` graph nodes.
#[derive(Debug, Clone)]
pub struct DeadLetterQueue {
    kb: Arc<KnowledgeBase>,
}

impl DeadLetterQueue {
    /// Wrap a knowledge base for DLQ operations.
    pub fn new(kb: Arc<KnowledgeBase>) -> Self {
        Self { kb }
    }

    /// Store a failed item as a `DeadLetter` node.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on graph failure.
    pub async fn store(
        &self,
        step: &str,
        error: &str,
        node_ref: NodeId,
        max_retries: u32,
    ) -> Result<NodeId, UnikoError> {
        let mut props = HashMap::new();
        props.insert("step".into(), Value::String(step.to_string()));
        props.insert("error".into(), Value::String(error.to_string()));
        props.insert("node_ref".into(), Value::Int(node_ref));
        props.insert("retry_count".into(), Value::Int(0));
        props.insert("max_retries".into(), Value::Int(max_retries as i64));
        self.kb.create_node("DeadLetter", &props).await
    }

    /// List all pending dead-letter items eligible for retry.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on graph failure.
    pub async fn list_pending(&self) -> Result<Vec<DeadLetterInfo>, UnikoError> {
        let session = self.kb.db().session();
        let result = session
            .query(
                "MATCH (dl:DeadLetter) \
                 WHERE dl.retry_count < dl.max_retries \
                 RETURN dl, id(dl) AS did",
            )
            .await
            .map_err(|e| UnikoError::Storage(e.to_string()))?;

        let mut items = Vec::with_capacity(result.len());
        for row in result.rows() {
            let did: i64 = row
                .get("did")
                .map_err(|e| UnikoError::Storage(e.to_string()))?;
            let node: uni_db::Node = row
                .get("dl")
                .map_err(|e| UnikoError::Storage(e.to_string()))?;
            items.push(DeadLetterInfo {
                id: did,
                step: node
                    .properties
                    .get("step")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                error: node
                    .properties
                    .get("error")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                node_ref: node
                    .properties
                    .get("node_ref")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0),
                retry_count: node
                    .properties
                    .get("retry_count")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0),
                max_retries: node
                    .properties
                    .get("max_retries")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(3),
            });
        }
        Ok(items)
    }

    /// Increment the retry count on a dead-letter item.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on graph failure.
    pub async fn increment_retry(&self, dead_letter_id: NodeId) -> Result<(), UnikoError> {
        let mut props = HashMap::new();
        // Fetch current count, increment.
        if let Some((_, existing)) = self.kb.get_node(dead_letter_id).await? {
            let current = existing
                .get("retry_count")
                .and_then(|v| v.as_i64())
                .unwrap_or(0);
            props.insert("retry_count".into(), Value::Int(current + 1));
            self.kb.update_node(dead_letter_id, &props).await?;
        }
        Ok(())
    }

    /// Delete a single dead-letter item.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on graph failure.
    pub async fn clear(&self, dead_letter_id: NodeId) -> Result<bool, UnikoError> {
        self.kb.delete_node(dead_letter_id).await
    }

    /// Delete all dead-letter items.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on graph failure.
    pub async fn clear_all(&self) -> Result<u64, UnikoError> {
        let session = self.kb.db().session();
        let tx = session
            .tx()
            .await
            .map_err(|e| UnikoError::Storage(e.to_string()))?;
        let result = tx
            .execute("MATCH (dl:DeadLetter) DETACH DELETE dl")
            .await
            .map_err(|e| UnikoError::Storage(e.to_string()))?;
        tx.commit()
            .await
            .map_err(|e| UnikoError::Storage(e.to_string()))?;
        Ok(result.nodes_deleted() as u64)
    }
}
