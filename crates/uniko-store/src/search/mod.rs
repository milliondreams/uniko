//! Search operations: vector similarity, fulltext BM25, hybrid RRF, and
//! graph traversal.

pub mod fulltext;
pub mod hybrid;
pub mod traversal;
pub mod vector;

use std::collections::HashMap;

use uni_db::Value;

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
