//! Node CRUD operations on the knowledge graph.
//!
//! All methods use parameterized Cypher to prevent injection.  Labels and
//! property names are validated before interpolation; values are always
//! bound as `$pN` parameters.

// Rust guideline compliant

use std::collections::HashMap;

use uni_db::Value;

use super::{build_inline_props, build_set_clause, validate_label, validate_property_name};
use crate::error::{Result, UnikoError};
use crate::storage::KnowledgeBase;
use crate::storage::filter::Filter;
use crate::types::NodeId;

impl KnowledgeBase {
    /// Create a node with the given label and properties.
    ///
    /// Returns the internal [`NodeId`] of the new node.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Schema`] if `label` is unknown or a property
    /// name is invalid, or [`UnikoError::Storage`] on database failure.
    pub async fn create_node(
        &self,
        label: &str,
        properties: &HashMap<String, Value>,
    ) -> Result<NodeId> {
        validate_label(label)?;
        // Properties must be inline in CREATE to satisfy NOT NULL constraints.
        let (inline_props, params) = build_inline_props(properties, 0)?;
        let cypher = format!("CREATE (n:{label} {{{inline_props}}}) RETURN id(n) AS vid");

        let session = self.db.session();
        let tx = session
            .tx()
            .await
            .map_err(|e| UnikoError::Storage(e.to_string()))?;

        let mut qb = tx.query_with(&cypher);
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
            .ok_or_else(|| UnikoError::Storage("CREATE returned no rows".into()))?;
        let vid: i64 = row
            .get("vid")
            .map_err(|e| UnikoError::Storage(e.to_string()))?;
        Ok(vid)
    }

    /// Retrieve a node by its internal [`NodeId`].
    ///
    /// Returns `None` if the node does not exist.  The tuple contains the
    /// primary label and all stored properties.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on database failure.
    pub async fn get_node(
        &self,
        node_id: NodeId,
    ) -> Result<Option<(String, HashMap<String, Value>)>> {
        let session = self.db.session();
        let result = session
            .query_with("MATCH (n) WHERE id(n) = $vid RETURN n")
            .param("vid", node_id)
            .fetch_all()
            .await
            .map_err(|e| UnikoError::Storage(e.to_string()))?;

        match result.rows().first() {
            None => Ok(None),
            Some(row) => {
                let node: uni_db::Node = row
                    .get("n")
                    .map_err(|e| UnikoError::Storage(e.to_string()))?;
                let label = node.labels.into_iter().next().unwrap_or_default();
                Ok(Some((label, node.properties)))
            }
        }
    }

    /// Look up a node by its external ID field (e.g. `message_id`).
    ///
    /// Uses the Hash index on the ID field for O(1) lookup.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Schema`] if `label` or `id_field` is invalid,
    /// or [`UnikoError::Storage`] on database failure.
    pub async fn get_node_by_ext_id(
        &self,
        label: &str,
        id_field: &str,
        ext_id: &str,
    ) -> Result<Option<(NodeId, HashMap<String, Value>)>> {
        validate_label(label)?;
        validate_property_name(id_field)?;

        let cypher = format!("MATCH (n:{label} {{{id_field}: $eid}}) RETURN n, id(n) AS vid");
        let session = self.db.session();
        let result = session
            .query_with(&cypher)
            .param("eid", ext_id)
            .fetch_all()
            .await
            .map_err(|e| UnikoError::Storage(e.to_string()))?;

        match result.rows().first() {
            None => Ok(None),
            Some(row) => {
                let vid: i64 = row
                    .get("vid")
                    .map_err(|e| UnikoError::Storage(e.to_string()))?;
                let node: uni_db::Node = row
                    .get("n")
                    .map_err(|e| UnikoError::Storage(e.to_string()))?;
                Ok(Some((vid, node.properties)))
            }
        }
    }

    /// Update specific properties on an existing node.
    ///
    /// Only the supplied properties are changed; others are preserved.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on database failure or if the node
    /// does not exist.
    pub async fn update_node(
        &self,
        node_id: NodeId,
        properties: &HashMap<String, Value>,
    ) -> Result<()> {
        if properties.is_empty() {
            return Ok(());
        }
        let (set_clause, params) = build_set_clause("n", properties, 0)?;
        let cypher = format!("MATCH (n) WHERE id(n) = $vid {set_clause}");

        let session = self.db.session();
        let tx = session
            .tx()
            .await
            .map_err(|e| UnikoError::Storage(e.to_string()))?;

        let mut qb = tx.execute_with(&cypher);
        qb = qb.param("vid", node_id);
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

    /// Delete a node and all its edges.
    ///
    /// Returns `true` if a node was deleted, `false` if it did not exist.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on database failure.
    pub async fn delete_node(&self, node_id: NodeId) -> Result<bool> {
        let session = self.db.session();
        let tx = session
            .tx()
            .await
            .map_err(|e| UnikoError::Storage(e.to_string()))?;
        let result = tx
            .execute_with("MATCH (n) WHERE id(n) = $vid DETACH DELETE n")
            .param("vid", node_id)
            .run()
            .await
            .map_err(|e| UnikoError::Storage(e.to_string()))?;
        tx.commit()
            .await
            .map_err(|e| UnikoError::Storage(e.to_string()))?;
        Ok(result.nodes_deleted() > 0)
    }

    /// Upsert a node using its external ID field.
    ///
    /// If no node with the given `ext_id` exists, one is created with
    /// `properties`.  If it already exists, the existing node is updated.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Schema`] if `label` or `ext_id_field` is
    /// invalid, or [`UnikoError::Storage`] on database failure.
    pub async fn merge_node(
        &self,
        label: &str,
        ext_id_field: &str,
        ext_id: &str,
        properties: &HashMap<String, Value>,
    ) -> Result<NodeId> {
        validate_label(label)?;
        validate_property_name(ext_id_field)?;

        // Two-step upsert to avoid NOT NULL constraint violations during
        // MERGE's internal CREATE (uni-db evaluates constraints before
        // ON CREATE SET runs).
        if let Some((existing_id, _)) = self.get_node_by_ext_id(label, ext_id_field, ext_id).await?
        {
            if !properties.is_empty() {
                self.update_node(existing_id, properties).await?;
            }
            return Ok(existing_id);
        }

        // Node doesn't exist — create with all properties plus the ext_id.
        let mut all_props = properties.clone();
        all_props
            .entry(ext_id_field.to_string())
            .or_insert_with(|| Value::String(ext_id.to_string()));
        self.create_node(label, &all_props).await
    }

    /// Query nodes of a given label, optionally filtered and limited.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Schema`] if `label` is unknown, or
    /// [`UnikoError::Storage`] on database failure.
    pub async fn query_nodes(
        &self,
        label: &str,
        filter: Option<&Filter>,
        limit: Option<usize>,
    ) -> Result<Vec<(NodeId, HashMap<String, Value>)>> {
        validate_label(label)?;

        let mut cypher = format!("MATCH (n:{label})");
        let mut all_params: Vec<(String, Value)> = Vec::new();

        if let Some(f) = filter {
            let cf = f.to_cypher("n", 0)?;
            cypher.push_str(&format!(" WHERE {}", cf.fragment));
            all_params.extend(cf.params);
        }

        cypher.push_str(" RETURN n, id(n) AS vid");

        if let Some(lim) = limit {
            cypher.push_str(&format!(" LIMIT {lim}"));
        }

        let session = self.db.session();
        let mut qb = session.query_with(&cypher);
        for (k, v) in &all_params {
            qb = qb.param(k.as_str(), v.clone());
        }
        let result = qb
            .fetch_all()
            .await
            .map_err(|e| UnikoError::Storage(e.to_string()))?;

        let mut nodes = Vec::with_capacity(result.len());
        for row in result.rows() {
            let vid: i64 = row
                .get("vid")
                .map_err(|e| UnikoError::Storage(e.to_string()))?;
            let node: uni_db::Node = row
                .get("n")
                .map_err(|e| UnikoError::Storage(e.to_string()))?;
            nodes.push((vid, node.properties));
        }
        Ok(nodes)
    }
}
