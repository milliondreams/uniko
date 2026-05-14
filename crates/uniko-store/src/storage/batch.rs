//! Batch creation via UNWIND for bulk ingestion.
//!
//! Uses single `UNWIND`-based Cypher statements instead of per-row
//! loops.  This leverages the HashJoin rewrite (uni-db #53/#54) that
//! converts `UNWIND … MATCH WHERE id() = e.col` into an efficient
//! hash join rather than a cross-join + filter.

// Rust guideline compliant

use std::collections::{BTreeSet, HashMap};

use uni_db::Value;

use super::{KnowledgeBase, validate_edge_type, validate_label, validate_property_name};
use crate::error::{Result, UnikoError};
use crate::types::{EdgeId, NodeId};

impl KnowledgeBase {
    /// Create multiple nodes of the same label in a single UNWIND query.
    ///
    /// Significantly faster than individual [`KnowledgeBase::create_node`]
    /// calls for bulk ingestion (e.g. chunking an artifact into many
    /// chunks).
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Schema`] if `label` is unknown or a
    /// property name is invalid, or [`UnikoError::Storage`] on database
    /// failure.  The transaction is rolled back on any error.
    pub async fn batch_create_nodes(
        &self,
        label: &str,
        items: &[HashMap<String, Value>],
    ) -> Result<Vec<NodeId>> {
        validate_label(label)?;
        if items.is_empty() {
            return Ok(Vec::new());
        }

        // Collect the union of all property keys (sorted for determinism).
        let prop_keys = collect_property_keys_from_nodes(items)?;

        // Build UNWIND list: one Map per node.
        let list: Vec<Value> = items
            .iter()
            .map(|props| {
                let mut m = HashMap::with_capacity(prop_keys.len());
                for k in &prop_keys {
                    m.insert(k.clone(), props.get(k).cloned().unwrap_or(Value::Null));
                }
                Value::Map(m)
            })
            .collect();

        // Build inline property syntax: `{prop1: item.prop1, prop2: item.prop2}`.
        let inline = if prop_keys.is_empty() {
            String::new()
        } else {
            let pairs: Vec<String> = prop_keys.iter().map(|k| format!("{k}: item.{k}")).collect();
            format!(" {{{}}}", pairs.join(", "))
        };

        let cypher =
            format!("UNWIND $items AS item CREATE (n:{label}{inline}) RETURN id(n) AS vid");

        let t_start = std::time::Instant::now();
        let session = self.db.session();
        let t_session = t_start.elapsed().as_micros();

        let tx = session
            .tx()
            .await
            .map_err(|e| UnikoError::Storage(e.to_string()))?;
        let t_tx_begin = t_start.elapsed().as_micros() - t_session;

        let result = tx
            .query_with(&cypher)
            .param("items", Value::List(list))
            .fetch_all()
            .await
            .map_err(|e| UnikoError::Storage(e.to_string()))?;
        let t_query = t_start.elapsed().as_micros() - t_session - t_tx_begin;

        let mut node_ids = Vec::with_capacity(items.len());
        for row in result.rows() {
            let vid: i64 = row
                .get("vid")
                .map_err(|e| UnikoError::Storage(e.to_string()))?;
            node_ids.push(vid);
        }

        tx.commit()
            .await
            .map_err(|e| UnikoError::Storage(e.to_string()))?;
        let t_commit = t_start.elapsed().as_micros() - t_session - t_tx_begin - t_query;

        tracing::debug!(
            label,
            count = items.len(),
            session_us = t_session as u64,
            tx_begin_us = t_tx_begin as u64,
            query_us = t_query as u64,
            commit_us = t_commit as u64,
            total_us = t_start.elapsed().as_micros() as u64,
            "batch_create_nodes micro"
        );
        Ok(node_ids)
    }

    /// Create multiple edges of the same type in a single UNWIND query.
    ///
    /// Each tuple is `(from_node_id, to_node_id, properties)`.  Returns
    /// the created edge IDs.  When the IDs are not needed, prefer
    /// [`KnowledgeBase::batch_create_edges_fast`] which skips the
    /// `RETURN` clause and is materially faster on auto-embed graphs.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Schema`] if `edge_type` is unknown or a
    /// property name is invalid, or [`UnikoError::Storage`] on database
    /// failure.
    pub async fn batch_create_edges(
        &self,
        edge_type: &str,
        edges: &[(NodeId, NodeId, HashMap<String, Value>)],
    ) -> Result<Vec<EdgeId>> {
        self.batch_create_edges_inner(edge_type, None, None, edges, true)
            .await
    }

    /// Same as [`KnowledgeBase::batch_create_edges`] without returning
    /// edge IDs.  Optional `src_label` and `dst_label` hints constrain
    /// the `MATCH` to a labelled node set, letting the planner narrow
    /// the scan.
    ///
    /// Pass `Some("Message|Chunk")` for label disjunction.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Schema`] if `edge_type` is unknown or a
    /// property name is invalid, or [`UnikoError::Storage`] on database
    /// failure.
    pub async fn batch_create_edges_fast(
        &self,
        edge_type: &str,
        src_label: Option<&str>,
        dst_label: Option<&str>,
        edges: &[(NodeId, NodeId, HashMap<String, Value>)],
    ) -> Result<()> {
        self.batch_create_edges_inner(edge_type, src_label, dst_label, edges, false)
            .await?;
        Ok(())
    }

    async fn batch_create_edges_inner(
        &self,
        edge_type: &str,
        src_label: Option<&str>,
        dst_label: Option<&str>,
        edges: &[(NodeId, NodeId, HashMap<String, Value>)],
        return_ids: bool,
    ) -> Result<Vec<EdgeId>> {
        validate_edge_type(edge_type)?;
        if edges.is_empty() {
            return Ok(Vec::new());
        }

        // Collect the union of all property keys (sorted for determinism).
        let prop_keys = collect_property_keys_from_edges(edges)?;

        // Build UNWIND list: one Map per edge with src, dst, and properties.
        let list: Vec<Value> = edges
            .iter()
            .map(|(from, to, props)| {
                let mut m = HashMap::with_capacity(2 + prop_keys.len());
                m.insert("src".to_string(), Value::Int(*from));
                m.insert("dst".to_string(), Value::Int(*to));
                for k in &prop_keys {
                    m.insert(k.clone(), props.get(k).cloned().unwrap_or(Value::Null));
                }
                Value::Map(m)
            })
            .collect();

        // Build SET clause for properties: `SET r.p1 = e.p1, r.p2 = e.p2`.
        let set_clause = if prop_keys.is_empty() {
            String::new()
        } else {
            let assigns: Vec<String> = prop_keys.iter().map(|k| format!("r.{k} = e.{k}")).collect();
            format!(" SET {}", assigns.join(", "))
        };

        // Multi-MATCH (avoids implicit cartesian product of MATCH (a),(b)).
        // Optional label hints narrow the scan to known label sets.
        let src_lbl = src_label.map(|l| format!(":{l}")).unwrap_or_default();
        let dst_lbl = dst_label.map(|l| format!(":{l}")).unwrap_or_default();

        let cypher = if return_ids {
            format!(
                "UNWIND $edges AS e \
                 MATCH (a{src_lbl}) WHERE id(a) = e.src \
                 MATCH (b{dst_lbl}) WHERE id(b) = e.dst \
                 CREATE (a)-[r:{edge_type}]->(b){set_clause} \
                 RETURN id(r) AS eid"
            )
        } else {
            format!(
                "UNWIND $edges AS e \
                 MATCH (a{src_lbl}) WHERE id(a) = e.src \
                 MATCH (b{dst_lbl}) WHERE id(b) = e.dst \
                 CREATE (a)-[r:{edge_type}]->(b){set_clause}"
            )
        };

        let t_start = std::time::Instant::now();
        let session = self.db.session();
        let t_session = t_start.elapsed().as_micros();

        let tx = session
            .tx()
            .await
            .map_err(|e| UnikoError::Storage(e.to_string()))?;
        let t_tx_begin = t_start.elapsed().as_micros() - t_session;

        let mut edge_ids = Vec::new();
        if return_ids {
            let result = tx
                .query_with(&cypher)
                .param("edges", Value::List(list))
                .fetch_all()
                .await
                .map_err(|e| UnikoError::Storage(e.to_string()))?;
            edge_ids.reserve(edges.len());
            for row in result.rows() {
                let eid: i64 = row
                    .get("eid")
                    .map_err(|e| UnikoError::Storage(e.to_string()))?;
                edge_ids.push(eid);
            }
        } else {
            tx.execute_with(&cypher)
                .param("edges", Value::List(list))
                .run()
                .await
                .map_err(|e| UnikoError::Storage(e.to_string()))?;
        }
        let t_query = t_start.elapsed().as_micros() - t_session - t_tx_begin;

        tx.commit()
            .await
            .map_err(|e| UnikoError::Storage(e.to_string()))?;
        let t_commit = t_start.elapsed().as_micros() - t_session - t_tx_begin - t_query;

        tracing::debug!(
            edge_type,
            count = edges.len(),
            return_ids,
            session_us = t_session as u64,
            tx_begin_us = t_tx_begin as u64,
            query_us = t_query as u64,
            commit_us = t_commit as u64,
            total_us = t_start.elapsed().as_micros() as u64,
            "batch_create_edges micro"
        );
        Ok(edge_ids)
    }
}

// ── Helpers ────────────────────────────────────────────────────────

/// Collect and validate the union of property keys from node items.
fn collect_property_keys_from_nodes(items: &[HashMap<String, Value>]) -> Result<Vec<String>> {
    let mut keys = BTreeSet::new();
    for props in items {
        for k in props.keys() {
            validate_property_name(k)?;
            keys.insert(k.clone());
        }
    }
    Ok(keys.into_iter().collect())
}

/// Collect and validate the union of property keys from edge tuples.
fn collect_property_keys_from_edges(
    edges: &[(NodeId, NodeId, HashMap<String, Value>)],
) -> Result<Vec<String>> {
    let mut keys = BTreeSet::new();
    for (_, _, props) in edges {
        for k in props.keys() {
            validate_property_name(k)?;
            keys.insert(k.clone());
        }
    }
    Ok(keys.into_iter().collect())
}
