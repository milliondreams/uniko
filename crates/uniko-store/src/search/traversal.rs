//! Graph traversal: BFS, shortest path, and Personalized PageRank.

use std::collections::HashMap;

use crate::error::Result;
use crate::storage::KnowledgeBase;
use crate::storage::edges::Direction;
use crate::types::{EdgeId, NodeId};

/// A step in a traversal path.
#[derive(Debug, Clone)]
pub struct PathStep {
    /// Edge traversed.
    pub edge_id: EdgeId,
    /// Relationship type.
    pub edge_type: String,
    /// Source node of the edge.
    pub from: NodeId,
    /// Target node of the edge.
    pub to: NodeId,
}

/// A complete path between two nodes.
#[derive(Debug, Clone)]
pub struct Path {
    /// Ordered steps from source to destination.
    pub steps: Vec<PathStep>,
    /// Number of hops.
    pub length: usize,
}

/// A single node visited during traversal.
#[derive(Debug, Clone)]
pub struct TraversalResult {
    /// Node identifier.
    pub node_id: NodeId,
    /// Primary label.
    pub node_type: String,
    /// Depth at which this node was reached.
    pub depth: usize,
    /// All node properties.
    pub properties: HashMap<String, uni_db::Value>,
}

/// Result of Personalized PageRank computation.
#[derive(Debug, Clone)]
pub struct PageRankResult {
    /// Nodes ranked by PPR score (descending).
    pub scores: Vec<(NodeId, f64)>,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Whether the algorithm converged within tolerance.
    pub converged: bool,
}

impl KnowledgeBase {
    /// Breadth-first traversal from `start` up to `max_depth` hops.
    ///
    /// Target: < 5 ms for depth 3 (NF4).
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on database failure.
    pub async fn traverse(
        &self,
        start: NodeId,
        max_depth: usize,
        direction: Direction,
        edge_types: Option<&[&str]>,
    ) -> Result<Vec<TraversalResult>> {
        if max_depth == 0 {
            return Ok(Vec::new());
        }

        // Build the variable-length relationship pattern.
        // Cypher syntax: -[:TYPE1|TYPE2*1..N]->
        let rel_pattern = match edge_types {
            Some(types) if !types.is_empty() => {
                let joined = types.join("|");
                format!(":{joined}*1..{max_depth}")
            }
            _ => format!("*1..{max_depth}"),
        };
        let match_clause = match direction {
            Direction::Outgoing => {
                format!("MATCH (s)-[{rel_pattern}]->(t) WHERE id(s) = $sid")
            }
            Direction::Incoming => {
                format!("MATCH (s)<-[{rel_pattern}]-(t) WHERE id(s) = $sid")
            }
            Direction::Both => {
                format!("MATCH (s)-[{rel_pattern}]-(t) WHERE id(s) = $sid")
            }
        };
        let cypher = format!("{match_clause} RETURN DISTINCT t, id(t) AS tid");

        let session = self.db.session();
        let result = session
            .query_with(&cypher)
            .param("sid", start)
            .fetch_all()
            .await?;

        let mut nodes = Vec::with_capacity(result.len());
        for row in result.rows() {
            let tid: i64 = row.get("tid")?;
            let node: uni_db::Node = row.get("t")?;
            let label = node.labels.into_iter().next().unwrap_or_default();
            nodes.push(TraversalResult {
                node_id: tid,
                node_type: label,
                depth: 0, // Depth not computed in simple VLP query.
                properties: node.properties,
            });
        }
        Ok(nodes)
    }

    /// Find the shortest path between two nodes.
    ///
    /// Uses Cypher's built-in `shortestPath` for correctness and speed.
    /// Returns `None` if no path exists.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on database failure.
    pub async fn shortest_path(
        &self,
        from: NodeId,
        to: NodeId,
        edge_types: Option<&[&str]>,
        max_depth: usize,
    ) -> Result<Option<Path>> {
        let rel = match edge_types {
            Some(types) if !types.is_empty() => {
                let joined = types.join("|");
                format!(":{joined}")
            }
            _ => String::new(),
        };

        let cypher = format!(
            "MATCH (a), (b) WHERE id(a) = $src AND id(b) = $dst \
             MATCH p = shortestPath((a)-[{rel}*..{max_depth}]->(b)) \
             RETURN length(p) AS len"
        );

        let session = self.db.session();
        let qr = session
            .query_with(&cypher)
            .param("src", from)
            .param("dst", to)
            .fetch_all()
            .await?;

        // uni-db's `shortestPath` returns one row with a NULL `len` when
        // no path connects the two endpoints; real Cypher failures
        // (syntax, missing node id, …) surface as `Err` above and must
        // not be silently coerced to `Ok(None)`.
        let Some(row) = qr.rows().first() else {
            return Ok(None);
        };
        let Some(len) = row.get::<Option<i64>>("len")? else {
            return Ok(None);
        };
        Ok(Some(Path {
            steps: Vec::new(),
            length: len as usize,
        }))
    }

    /// Personalized PageRank from seed nodes.
    ///
    /// Spreads activation from `seeds` across the graph using power
    /// iteration.  Returns the top-`top_k` nodes by PPR score.
    ///
    /// Convergence: stops when the max absolute score change < 1e-6 or
    /// `max_iter` is reached.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on database failure.
    pub async fn personalized_pagerank(
        &self,
        seeds: &[NodeId],
        damping: f64,
        max_iter: usize,
        top_k: usize,
    ) -> Result<PageRankResult> {
        self.personalized_pagerank_weighted(seeds, damping, max_iter, top_k, None)
            .await
    }

    /// Personalized PageRank with per-edge-type weight multipliers.
    ///
    /// Generalises [`personalized_pagerank`] by letting callers bias
    /// propagation along specific relation types — mirrors Hindsight's
    /// `μ(ℓ)` link-type multiplier (arxiv:2512.12818).  Weight `w` on
    /// edge type `e` scales the activation flowing through any
    /// `(_)-[e]->(_)` relation.  Unmapped edge types default to `1.0`
    /// (same as the vanilla algorithm).
    ///
    /// Pass `edge_weights = None` for the uniform-weight case.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] when the adjacency fetch fails.
    pub async fn personalized_pagerank_weighted(
        &self,
        seeds: &[NodeId],
        damping: f64,
        max_iter: usize,
        top_k: usize,
        edge_weights: Option<&HashMap<String, f64>>,
    ) -> Result<PageRankResult> {
        if seeds.is_empty() {
            return Ok(PageRankResult {
                scores: Vec::new(),
                iterations: 0,
                converged: true,
            });
        }

        // 1. Fetch the adjacency list from the graph, including edge type
        //    so we can apply per-type multipliers.
        let session = self.db.session();
        let result = session
            .query("MATCH (a)-[r]->(b) RETURN id(a) AS src, id(b) AS dst, type(r) AS etype")
            .await?;

        // Build weighted adjacency: src -> [(dst, weight), ...]
        let mut out_edges: HashMap<NodeId, Vec<(NodeId, f64)>> = HashMap::new();
        let mut all_nodes: std::collections::HashSet<NodeId> = std::collections::HashSet::new();

        for row in result.rows() {
            let src: i64 = row.get("src")?;
            let dst: i64 = row.get("dst")?;
            let etype: String = row.get("etype").unwrap_or_default();
            let w = edge_weights
                .and_then(|m| m.get(&etype).copied())
                .unwrap_or(1.0);
            // Drop zero-weight edges entirely — they don't contribute.
            if w == 0.0 {
                all_nodes.insert(src);
                all_nodes.insert(dst);
                continue;
            }
            out_edges.entry(src).or_default().push((dst, w));
            all_nodes.insert(src);
            all_nodes.insert(dst);
        }

        if all_nodes.is_empty() {
            return Ok(PageRankResult {
                scores: Vec::new(),
                iterations: 0,
                converged: true,
            });
        }

        // 2. Initialize scores: uniform teleport to seeds.
        let seed_prob = 1.0 / seeds.len() as f64;
        let teleport: HashMap<NodeId, f64> = seeds.iter().map(|&s| (s, seed_prob)).collect();

        let mut scores: HashMap<NodeId, f64> = all_nodes
            .iter()
            .map(|&n| (n, *teleport.get(&n).unwrap_or(&0.0)))
            .collect();

        // 3. Power iteration with weighted propagation.
        //    Each src divides its score among neighbours proportional to
        //    each edge's weight (sum-of-weights replaces out-degree).
        const EPSILON: f64 = 1e-6;
        let mut converged = false;
        let mut iterations = 0;

        for _iter in 0..max_iter {
            iterations += 1;
            let mut new_scores: HashMap<NodeId, f64> = HashMap::new();

            for &node in &all_nodes {
                let tp = teleport.get(&node).copied().unwrap_or(0.0);
                new_scores.insert(node, (1.0 - damping) * tp);
            }

            for (&src, neighbors) in &out_edges {
                let total_w: f64 = neighbors.iter().map(|(_, w)| *w).sum();
                if total_w <= 0.0 {
                    continue;
                }
                let base = damping * scores[&src] / total_w;
                for &(dst, w) in neighbors {
                    *new_scores.entry(dst).or_insert(0.0) += base * w;
                }
            }

            let max_diff = all_nodes
                .iter()
                .map(|n| {
                    let old = scores.get(n).copied().unwrap_or(0.0);
                    let new = new_scores.get(n).copied().unwrap_or(0.0);
                    (old - new).abs()
                })
                .fold(0.0f64, f64::max);

            scores = new_scores;

            if max_diff < EPSILON {
                converged = true;
                break;
            }
        }

        // 4. Rank and return top-k.
        let mut ranked: Vec<(NodeId, f64)> = scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(top_k);

        Ok(PageRankResult {
            scores: ranked,
            iterations,
            converged,
        })
    }
}
