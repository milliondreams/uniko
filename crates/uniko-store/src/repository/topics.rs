//! Topic-detection reads/writes: entity load, co-occurrence aggregation,
//! and BELONGS_TO membership wiring. Backs `uniko_cortex::topics`.

use uni_db::Value;

use crate::error::Result;
use crate::schema::constants::edges;
use crate::storage::KnowledgeBase;
use crate::types::NodeId;

/// One Entity loaded for topic clustering.
#[derive(Debug, Clone)]
pub struct EntityRow {
    pub node_id: NodeId,
    pub entity_id: String,
    pub name: String,
    /// Kniv-assigned NER type (e.g. `"PERSON"`, `"ORG"`, `"LOC"`); empty
    /// when ingest used the rule-based fallback only.
    pub entity_type: String,
    pub embedding: Option<Vec<f32>>,
}

/// An ordered (a &lt; b) co-occurrence pair with its weight (mention
/// count across shared sources).
#[derive(Debug, Clone)]
pub struct CoPair {
    pub a: NodeId,
    pub b: NodeId,
    pub weight: f64,
}

impl KnowledgeBase {
    /// Load every `:Entity` (id, name, type, embedding) for clustering.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`](crate::UnikoError::Storage) on
    /// query failure.
    pub async fn fetch_all_entities(&self) -> Result<Vec<EntityRow>> {
        let session = self.db.session();
        let cypher = "MATCH (e:Entity) \
                      RETURN id(e) AS nid, e.entity_id AS eid, e.name AS name, \
                             coalesce(e.entity_type, '') AS etype, \
                             e.embedding AS emb";
        let result = session.query_with(cypher).fetch_all().await?;
        let mut out = Vec::new();
        for row in result.rows() {
            let Ok(nid) = row.get::<i64>("nid") else {
                continue;
            };
            out.push(EntityRow {
                node_id: nid,
                entity_id: row.get("eid").unwrap_or_default(),
                name: row.get("name").unwrap_or_default(),
                entity_type: row.get("etype").unwrap_or_default(),
                embedding: row.get::<Vec<f32>>("emb").ok(),
            });
        }
        Ok(out)
    }

    /// Aggregate co-occurrence across `MENTIONS`-source labels, returning
    /// ordered pairs (a &lt; b) to dedupe the undirected relation, top
    /// `max_pairs` by weight.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`](crate::UnikoError::Storage) on
    /// query failure.
    pub async fn fetch_cooccurrence_pairs(&self, max_pairs: i64) -> Result<Vec<CoPair>> {
        let session = self.db.session();
        let cypher = "MATCH (src)-[:MENTIONS]->(a:Entity) \
                      MATCH (src)-[:MENTIONS]->(b:Entity) \
                      WHERE id(a) < id(b) \
                      WITH id(a) AS a, id(b) AS b, count(*) AS w \
                      RETURN a, b, w \
                      ORDER BY w DESC \
                      LIMIT $lim";
        let result = session
            .query_with(cypher)
            .param("lim", max_pairs)
            .fetch_all()
            .await?;
        let mut out = Vec::new();
        for row in result.rows() {
            let Ok(a) = row.get::<i64>("a") else { continue };
            let Ok(b) = row.get::<i64>("b") else { continue };
            let w: i64 = row.get("w").unwrap_or(1);
            out.push(CoPair {
                a,
                b,
                weight: w as f64,
            });
        }
        Ok(out)
    }

    /// MERGE a `BELONGS_TO` edge from each member node to `topic_node`
    /// (idempotent on re-run), retrying transient conflicts.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] / [`UnikoError::Conflict`]
    /// ([`UnikoError`](crate::UnikoError)) if the writes exhaust retries.
    pub async fn merge_belongs_to_edges(
        &self,
        topic_node: NodeId,
        member_node_ids: &[NodeId],
    ) -> Result<()> {
        if member_node_ids.is_empty() {
            return Ok(());
        }
        // uni-db 2.0 supports edge-MERGE, so BELONGS_TO is idempotent on
        // re-run — no duplicate edges.
        let cypher = format!(
            "MATCH (m), (t) WHERE id(m) = $m AND id(t) = $t \
             MERGE (m)-[:{}]->(t)",
            edges::BELONGS_TO
        );
        let member_ids = member_node_ids.to_vec();
        self.transact_with_retry(uni_db::RetryOptions::default(), move |tx| {
            let cypher = cypher.clone();
            let member_ids = member_ids.clone();
            async move {
                let r = async {
                    for mid in member_ids {
                        tx.query_with(&cypher)
                            .param("m", Value::Int(mid))
                            .param("t", Value::Int(topic_node))
                            .fetch_all()
                            .await?;
                    }
                    Ok(())
                }
                .await;
                (tx, r)
            }
        })
        .await
    }
}
