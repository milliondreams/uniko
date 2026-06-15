//! Persistent dead-letter queue backed by the knowledge graph.
//!
//! Failed pipeline items are stored as `DeadLetter` nodes for offline
//! triage. Retry/list/clear surfaces have no production callers and are
//! not provided.

use std::collections::HashMap;
use std::sync::Arc;

use uniko_store::{KnowledgeBase, NodeId, UnikoError, Value};

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
}
