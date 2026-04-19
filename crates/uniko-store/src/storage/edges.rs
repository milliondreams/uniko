//! Edge CRUD operations on the knowledge graph.

// Rust guideline compliant

use std::collections::HashMap;

use uni_db::Value;

use super::{KnowledgeBase, build_set_clause, validate_edge_type};
use crate::error::{Result, UnikoError};
use crate::types::{EdgeId, NodeId};

/// Direction for edge traversal queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Edges leaving the node.
    Outgoing,
    /// Edges arriving at the node.
    Incoming,
    /// Edges in either direction.
    Both,
}

/// A retrieved edge with endpoints and properties.
#[derive(Debug, Clone)]
pub struct EdgeRecord {
    /// Internal edge identifier.
    pub id: EdgeId,
    /// Relationship type name.
    pub edge_type: String,
    /// Source node identifier.
    pub from: NodeId,
    /// Target node identifier.
    pub to: NodeId,
    /// Edge properties.
    pub properties: HashMap<String, Value>,
}

impl KnowledgeBase {
    /// Create a directed edge between two nodes.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Schema`] if `edge_type` is unknown, or
    /// [`UnikoError::Storage`] on database failure.
    pub async fn create_edge(
        &self,
        edge_type: &str,
        from: NodeId,
        to: NodeId,
        properties: &HashMap<String, Value>,
    ) -> Result<EdgeId> {
        validate_edge_type(edge_type)?;
        let (set_clause, params) = build_set_clause("r", properties, 0)?;

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

        let session = self.db.session();
        let tx = session
            .tx()
            .await
            .map_err(|e| UnikoError::Storage(e.to_string()))?;

        let mut qb = tx.query_with(&cypher);
        qb = qb.param("src", from).param("dst", to);
        for (k, v) in &params {
            qb = qb.param(k.as_str(), v.clone());
        }
        let result = qb
            .fetch_all()
            .await
            .map_err(|e| UnikoError::Storage(e.to_string()))?;

        tx.commit()
            .await
            .map_err(|e| UnikoError::Storage(e.to_string()))?;

        let row = result
            .rows()
            .first()
            .ok_or_else(|| UnikoError::Storage("CREATE edge returned no rows".into()))?;
        let eid: i64 = row
            .get("eid")
            .map_err(|e| UnikoError::Storage(e.to_string()))?;
        Ok(eid)
    }

    /// Retrieve edges of a given type incident to `node_id`.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Schema`] if `edge_type` is unknown, or
    /// [`UnikoError::Storage`] on database failure.
    pub async fn get_edges(
        &self,
        node_id: NodeId,
        edge_type: &str,
        direction: Direction,
    ) -> Result<Vec<EdgeRecord>> {
        validate_edge_type(edge_type)?;
        let cypher = match direction {
            Direction::Outgoing => format!(
                "MATCH (a)-[r:{edge_type}]->(b) WHERE id(a) = $nid \
                 RETURN r, id(r) AS eid, id(a) AS src, id(b) AS dst, type(r) AS rtype"
            ),
            Direction::Incoming => format!(
                "MATCH (a)-[r:{edge_type}]->(b) WHERE id(b) = $nid \
                 RETURN r, id(r) AS eid, id(a) AS src, id(b) AS dst, type(r) AS rtype"
            ),
            Direction::Both => format!(
                "MATCH (a)-[r:{edge_type}]-(b) WHERE id(a) = $nid \
                 RETURN r, id(r) AS eid, id(startNode(r)) AS src, id(endNode(r)) AS dst, type(r) AS rtype"
            ),
        };

        let session = self.db.session();
        let result = session
            .query_with(&cypher)
            .param("nid", node_id)
            .fetch_all()
            .await
            .map_err(|e| UnikoError::Storage(e.to_string()))?;

        rows_to_edge_records(&result)
    }

    /// Retrieve all edges incident to `node_id` regardless of type.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on database failure.
    pub async fn get_all_edges(
        &self,
        node_id: NodeId,
        direction: Direction,
    ) -> Result<Vec<EdgeRecord>> {
        let cypher = match direction {
            Direction::Outgoing => {
                "MATCH (a)-[r]->(b) WHERE id(a) = $nid \
                 RETURN r, id(r) AS eid, id(a) AS src, id(b) AS dst, type(r) AS rtype"
            }
            Direction::Incoming => {
                "MATCH (a)-[r]->(b) WHERE id(b) = $nid \
                 RETURN r, id(r) AS eid, id(a) AS src, id(b) AS dst, type(r) AS rtype"
            }
            Direction::Both => {
                "MATCH (a)-[r]-(b) WHERE id(a) = $nid \
                 RETURN r, id(r) AS eid, id(startNode(r)) AS src, id(endNode(r)) AS dst, type(r) AS rtype"
            }
        };

        let session = self.db.session();
        let result = session
            .query_with(cypher)
            .param("nid", node_id)
            .fetch_all()
            .await
            .map_err(|e| UnikoError::Storage(e.to_string()))?;

        rows_to_edge_records(&result)
    }

    /// Delete a single edge by its internal [`EdgeId`].
    ///
    /// Returns `true` if an edge was deleted.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on database failure.
    pub async fn delete_edge(&self, edge_id: EdgeId) -> Result<bool> {
        let session = self.db.session();
        let tx = session
            .tx()
            .await
            .map_err(|e| UnikoError::Storage(e.to_string()))?;
        let result = tx
            .execute_with("MATCH ()-[r]->() WHERE id(r) = $eid DELETE r")
            .param("eid", edge_id)
            .run()
            .await
            .map_err(|e| UnikoError::Storage(e.to_string()))?;
        tx.commit()
            .await
            .map_err(|e| UnikoError::Storage(e.to_string()))?;
        Ok(result.relationships_deleted() > 0)
    }

    /// Delete all edges of `edge_type` between two specific nodes.
    ///
    /// Returns the number of edges deleted.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Schema`] if `edge_type` is unknown, or
    /// [`UnikoError::Storage`] on database failure.
    pub async fn delete_edges_between(
        &self,
        edge_type: &str,
        from: NodeId,
        to: NodeId,
    ) -> Result<u64> {
        validate_edge_type(edge_type)?;
        let cypher =
            format!("MATCH (a)-[r:{edge_type}]->(b) WHERE id(a) = $src AND id(b) = $dst DELETE r");
        let session = self.db.session();
        let tx = session
            .tx()
            .await
            .map_err(|e| UnikoError::Storage(e.to_string()))?;
        let result = tx
            .execute_with(&cypher)
            .param("src", from)
            .param("dst", to)
            .run()
            .await
            .map_err(|e| UnikoError::Storage(e.to_string()))?;
        tx.commit()
            .await
            .map_err(|e| UnikoError::Storage(e.to_string()))?;
        Ok(result.relationships_deleted() as u64)
    }

    /// Update properties on an existing edge.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on database failure.
    pub async fn update_edge(
        &self,
        edge_id: EdgeId,
        properties: &HashMap<String, Value>,
    ) -> Result<()> {
        if properties.is_empty() {
            return Ok(());
        }
        let (set_clause, params) = build_set_clause("r", properties, 0)?;
        let cypher = format!("MATCH ()-[r]->() WHERE id(r) = $eid {set_clause}");

        let session = self.db.session();
        let tx = session
            .tx()
            .await
            .map_err(|e| UnikoError::Storage(e.to_string()))?;

        let mut qb = tx.execute_with(&cypher);
        qb = qb.param("eid", edge_id);
        for (k, v) in &params {
            qb = qb.param(k.as_str(), v.clone());
        }
        qb.run()
            .await
            .map_err(|e| UnikoError::Storage(e.to_string()))?;
        tx.commit()
            .await
            .map_err(|e| UnikoError::Storage(e.to_string()))?;
        Ok(())
    }
}

/// Extract [`EdgeRecord`]s from a query result.
fn rows_to_edge_records(result: &uni_db::QueryResult) -> Result<Vec<EdgeRecord>> {
    let mut records = Vec::with_capacity(result.len());
    for row in result.rows() {
        let eid: i64 = row
            .get("eid")
            .map_err(|e| UnikoError::Storage(e.to_string()))?;
        let src: i64 = row
            .get("src")
            .map_err(|e| UnikoError::Storage(e.to_string()))?;
        let dst: i64 = row
            .get("dst")
            .map_err(|e| UnikoError::Storage(e.to_string()))?;
        let rtype: String = row
            .get("rtype")
            .map_err(|e| UnikoError::Storage(e.to_string()))?;
        let edge: uni_db::Edge = row
            .get("r")
            .map_err(|e| UnikoError::Storage(e.to_string()))?;
        records.push(EdgeRecord {
            id: eid,
            edge_type: rtype,
            from: src,
            to: dst,
            properties: edge.properties,
        });
    }
    Ok(records)
}
