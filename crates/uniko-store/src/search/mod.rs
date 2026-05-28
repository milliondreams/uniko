//! Search operations: vector similarity, fulltext BM25, hybrid RRF, and
//! graph traversal.

pub mod fulltext;
pub mod hybrid;
pub mod traversal;
pub mod vector;

use std::collections::HashMap;

use uni_db::{QueryResult, Value};

use crate::error::{Result, UnikoError};
use crate::types::NodeId;

/// A single result from any search method.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Internal node identifier.
    pub node_id: NodeId,
    /// Primary label of the matched node.
    pub node_type: String,
    /// Relevance score (meaning depends on the search method).
    pub score: f64,
    /// All properties of the matched node.
    pub properties: HashMap<String, Value>,
}

/// Convert the shared `(node, score, vid)` row shape emitted by
/// `uni.vector.query` and `uni.fts.query` into a [`Vec<SearchResult>`].
///
/// Both index types yield the same row layout, so the row-walk is the
/// only common cost worth deduping; the surrounding Cypher build stays
/// per-call.
pub(crate) fn rows_to_search_hits(result: &QueryResult) -> Result<Vec<SearchResult>> {
    let mut hits = Vec::with_capacity(result.len());
    for row in result.rows() {
        let node: uni_db::Node = row
            .get("node")
            .map_err(|e| UnikoError::Search(e.to_string()))?;
        let score: f64 = row
            .get("score")
            .map_err(|e| UnikoError::Search(e.to_string()))?;
        let vid: i64 = row
            .get("vid")
            .map_err(|e| UnikoError::Search(e.to_string()))?;
        let label = node.labels.into_iter().next().unwrap_or_default();
        hits.push(SearchResult {
            node_id: vid,
            node_type: label,
            score,
            properties: node.properties,
        });
    }
    Ok(hits)
}
