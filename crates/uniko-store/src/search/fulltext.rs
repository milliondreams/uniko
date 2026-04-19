//! BM25-ranked fulltext search via uni-db's fulltext indexes.

// Rust guideline compliant

use crate::error::{Result, UnikoError};
use crate::search::SearchResult;
use crate::storage::{KnowledgeBase, validate_label};

impl KnowledgeBase {
    /// Search by fulltext (BM25) on a single field.
    ///
    /// Searches the Fulltext index on `field` of `node_type`.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Search`] on database failure.
    pub async fn fulltext_search(
        &self,
        query: &str,
        node_type: &str,
        field: &str,
        top_k: usize,
    ) -> Result<Vec<SearchResult>> {
        if query.is_empty() {
            return Ok(Vec::new());
        }
        validate_label(node_type).map_err(|e| UnikoError::Search(e.to_string()))?;

        let cypher = format!(
            "CALL uni.fts.query('{node_type}', '{field}', $query, {top_k}) \
             YIELD node, score \
             RETURN node, score, id(node) AS vid"
        );

        let session = self.db.session();
        let result = session
            .query_with(&cypher)
            .param("query", query)
            .fetch_all()
            .await
            .map_err(|e| UnikoError::Search(e.to_string()))?;

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

    /// Search across multiple fulltext fields simultaneously.
    ///
    /// Deduplicates by `node_id`, keeping the highest score for each node.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Search`] on database failure.
    pub async fn multi_field_fulltext_search(
        &self,
        query: &str,
        targets: &[(&str, &str)],
        top_k: usize,
    ) -> Result<Vec<SearchResult>> {
        if query.is_empty() || targets.is_empty() {
            return Ok(Vec::new());
        }

        let mut all_hits = Vec::new();
        for &(node_type, field) in targets {
            let hits = self.fulltext_search(query, node_type, field, top_k).await?;
            all_hits.extend(hits);
        }

        // Deduplicate by node_id, keeping the highest score.
        let mut best: std::collections::HashMap<i64, SearchResult> =
            std::collections::HashMap::new();
        for hit in all_hits {
            best.entry(hit.node_id)
                .and_modify(|existing| {
                    if hit.score > existing.score {
                        *existing = hit.clone();
                    }
                })
                .or_insert(hit);
        }

        let mut results: Vec<SearchResult> = best.into_values().collect();
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(top_k);
        Ok(results)
    }
}
