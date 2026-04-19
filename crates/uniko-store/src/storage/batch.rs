//! Batch creation operations for bulk ingestion.

// Rust guideline compliant

use std::collections::HashMap;

use uni_db::Value;

use super::{
    KnowledgeBase, build_inline_props, build_set_clause, validate_edge_type, validate_label,
};
use crate::error::{Result, UnikoError};
use crate::types::{EdgeId, NodeId};

impl KnowledgeBase {
    /// Create multiple nodes of the same label in a single transaction.
    ///
    /// Significantly faster than individual [`KnowledgeBase::create_node`]
    /// calls for bulk ingestion (e.g. chunking an artifact into many
    /// chunks).
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Schema`] if `label` is unknown, or
    /// [`UnikoError::Storage`] on database failure.  The transaction is
    /// rolled back on any error.
    pub async fn batch_create_nodes(
        &self,
        label: &str,
        items: &[HashMap<String, Value>],
    ) -> Result<Vec<NodeId>> {
        validate_label(label)?;
        if items.is_empty() {
            return Ok(Vec::new());
        }

        let session = self.db.session();
        let tx = session
            .tx()
            .await
            .map_err(|e| UnikoError::Storage(e.to_string()))?;

        let mut node_ids = Vec::with_capacity(items.len());
        for props in items {
            let (inline_props, params) = build_inline_props(props, 0)?;
            let cypher = format!("CREATE (n:{label} {{{inline_props}}}) RETURN id(n) AS vid");

            let mut qb = tx.query_with(&cypher);
            for (k, v) in &params {
                qb = qb.param(k.as_str(), v.clone());
            }
            let result = qb
                .fetch_all()
                .await
                .map_err(|e| UnikoError::Storage(e.to_string()))?;

            let vid: i64 = result
                .rows()
                .first()
                .ok_or_else(|| UnikoError::Storage("CREATE returned no rows".into()))?
                .get("vid")
                .map_err(|e| UnikoError::Storage(e.to_string()))?;
            node_ids.push(vid);
        }

        tx.commit()
            .await
            .map_err(|e| UnikoError::Storage(e.to_string()))?;
        Ok(node_ids)
    }

    /// Create multiple edges of the same type in a single transaction.
    ///
    /// Each tuple is `(from_node_id, to_node_id, properties)`.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Schema`] if `edge_type` is unknown, or
    /// [`UnikoError::Storage`] on database failure.
    pub async fn batch_create_edges(
        &self,
        edge_type: &str,
        edges: &[(NodeId, NodeId, HashMap<String, Value>)],
    ) -> Result<Vec<EdgeId>> {
        validate_edge_type(edge_type)?;
        if edges.is_empty() {
            return Ok(Vec::new());
        }

        let session = self.db.session();
        let tx = session
            .tx()
            .await
            .map_err(|e| UnikoError::Storage(e.to_string()))?;

        let mut edge_ids = Vec::with_capacity(edges.len());
        for (from, to, props) in edges {
            let (set_clause, params) = build_set_clause("r", props, 0)?;
            let cypher = if set_clause.is_empty() {
                format!(
                    "MATCH (a), (b) WHERE id(a) = $src AND id(b) = $dst \
                     CREATE (a)-[r:{edge_type}]->(b) RETURN id(r) AS eid"
                )
            } else {
                format!(
                    "MATCH (a), (b) WHERE id(a) = $src AND id(b) = $dst \
                     CREATE (a)-[r:{edge_type}]->(b) {set_clause} RETURN id(r) AS eid"
                )
            };

            let mut qb = tx.query_with(&cypher);
            qb = qb.param("src", *from).param("dst", *to);
            for (k, v) in &params {
                qb = qb.param(k.as_str(), v.clone());
            }
            let result = qb
                .fetch_all()
                .await
                .map_err(|e| UnikoError::Storage(e.to_string()))?;

            let eid: i64 = result
                .rows()
                .first()
                .ok_or_else(|| UnikoError::Storage("CREATE edge returned no rows".into()))?
                .get("eid")
                .map_err(|e| UnikoError::Storage(e.to_string()))?;
            edge_ids.push(eid);
        }

        tx.commit()
            .await
            .map_err(|e| UnikoError::Storage(e.to_string()))?;
        Ok(edge_ids)
    }
}
