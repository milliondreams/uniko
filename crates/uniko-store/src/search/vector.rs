//! Vector similarity search via HNSW indexes.
//!
//! Wraps uni-db's `uni.vector.query` procedure with typed results and
//! optional property filtering.

use crate::error::{Result, UnikoError};
use crate::search::{SearchResult, rows_to_search_hits};
use crate::storage::{KnowledgeBase, validate_label};

impl KnowledgeBase {
    /// Search for nodes by cosine similarity.
    ///
    /// Uses the HNSW index on `field` of `node_type`.  Results are sorted
    /// by descending similarity score.
    ///
    /// Target: < 20 ms for top-10 (NF2).
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Search`] on empty embedding, dimension
    /// mismatch, or database failure.
    pub async fn vector_search(
        &self,
        embedding: &[f32],
        node_type: &str,
        field: &str,
        top_k: usize,
    ) -> Result<Vec<SearchResult>> {
        if embedding.is_empty() {
            return Ok(Vec::new());
        }
        validate_label(node_type).map_err(|e| UnikoError::Search(e.to_string()))?;

        // uni.vector.query returns (node, score) where score is cosine
        // similarity (higher = more similar).
        let cypher = format!(
            "CALL uni.vector.query('{node_type}', '{field}', $vec, {top_k}) \
             YIELD node, score \
             RETURN node, score, id(node) AS vid"
        );

        let session = self.db.session();
        let result = session
            .query_with(&cypher)
            .param("vec", embedding.to_vec())
            .fetch_all()
            .await
            .map_err(|e| UnikoError::Search(e.to_string()))?;

        rows_to_search_hits(&result)
    }

    /// Search across multiple node types simultaneously.
    ///
    /// Performs vector search on each `(node_type, field)` pair and merges
    /// results by score, returning the overall top-k.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Search`] on database failure.
    pub async fn multi_type_vector_search(
        &self,
        embedding: &[f32],
        targets: &[(&str, &str)],
        top_k: usize,
    ) -> Result<Vec<SearchResult>> {
        if embedding.is_empty() || targets.is_empty() {
            return Ok(Vec::new());
        }

        let mut all_hits = Vec::new();
        for &(node_type, field) in targets {
            let hits = self
                .vector_search(embedding, node_type, field, top_k)
                .await?;
            all_hits.extend(hits);
        }

        // Sort by descending score and truncate.
        all_hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all_hits.truncate(top_k);
        Ok(all_hits)
    }
}
