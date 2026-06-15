//! Recall read path: the vector / fulltext / temporal / graph / chunk
//! search shapes the cascade issues. Backs `uniko_memory::recall`.
//!
//! These methods return decoded rows; the cascade's scoring (tier
//! weights, RRF fusion, min-max normalization, coverage) stays in
//! `uniko-memory`. Label/field *identifiers* are interpolated (Cypher
//! cannot parameterize them) — callers pass fixed identifiers from their
//! own target tables, never user input. Every *value* is bound as a
//! parameter.

use chrono::{DateTime, Utc};
use uni_db::Value;

use crate::error::Result;
use crate::storage::KnowledgeBase;
use crate::types::{NodeId, datetime_value};

/// A scored search hit: node id, primary label, a content string, and the
/// `similar_to` score. Shared by the vector, fulltext, and chunk/entity-
/// scoped shapes (all project the same `nid`/`lbl`/`content`/`score`).
#[derive(Debug, Clone)]
pub struct ScoredRow {
    pub node_id: NodeId,
    pub label: String,
    pub content: String,
    pub score: f64,
}

/// A temporal-channel hit, with a `source` discriminator (`fact` /
/// `observation` / `episode`) so the caller can score per arm.
#[derive(Debug, Clone)]
pub struct TemporalRow {
    pub node_id: NodeId,
    pub label: String,
    pub source: String,
    pub content: String,
}

/// A node's label + best-effort content, used to hydrate PPR-activated
/// node ids.
#[derive(Debug, Clone)]
pub struct NodeContentRow {
    pub node_id: NodeId,
    pub label: String,
    pub content: String,
}

impl KnowledgeBase {
    /// Vector search over `label.embed_field`, projecting `content_field`.
    ///
    /// `label`/`embed_field`/`content_field` are trusted identifiers from
    /// the caller's fixed target table; `qvec` and `limit` are bound.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`](crate::UnikoError::Storage) on
    /// query failure (the caller decides whether to skip the tier).
    pub async fn recall_vector_search(
        &self,
        label: &str,
        embed_field: &str,
        content_field: &str,
        qvec: &[f32],
        limit: i64,
    ) -> Result<Vec<ScoredRow>> {
        let session = self.db.session();
        let cypher = format!(
            "MATCH (m:{label}) \
             WHERE m.{embed_field} IS NOT NULL \
             RETURN id(m) AS nid, labels(m)[0] AS lbl, \
                    coalesce(m.{content_field}, '') AS content, \
                    similar_to(m.{embed_field}, $qvec) AS score \
             ORDER BY score DESC LIMIT $lim"
        );
        let result = session
            .query_with(&cypher)
            .param("qvec", Value::Vector(qvec.to_vec()))
            .param("lim", limit)
            .fetch_all()
            .await?;
        Ok(decode_scored_rows(result.rows()))
    }

    /// BM25 fulltext search over `label.content_field`.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`](crate::UnikoError::Storage) on
    /// query failure.
    pub async fn recall_fulltext_search(
        &self,
        label: &str,
        content_field: &str,
        qtxt: &str,
        limit: i64,
    ) -> Result<Vec<ScoredRow>> {
        let session = self.db.session();
        let cypher = format!(
            "MATCH (m:{label}) \
             RETURN id(m) AS nid, labels(m)[0] AS lbl, \
                    coalesce(m.{content_field}, '') AS content, \
                    similar_to(m.{content_field}, $qtxt) AS score \
             ORDER BY score DESC LIMIT $lim"
        );
        let result = session
            .query_with(&cypher)
            .param("qtxt", qtxt)
            .param("lim", limit)
            .fetch_all()
            .await?;
        Ok(decode_scored_rows(result.rows()))
    }

    /// Resolve entity/participant `names` to graph seed node ids for
    /// spreading activation. Deduped; infallible (logs and returns empty
    /// on query error, matching the cascade's best-effort contract).
    #[must_use]
    pub async fn resolve_entity_seeds(&self, names: &[String]) -> Vec<NodeId> {
        if names.is_empty() {
            return Vec::new();
        }
        let session = self.db.session();
        let cypher = "UNWIND $names AS name \
                      MATCH (n) \
                      WHERE (n:Entity AND n.name = name) \
                         OR (n:Participant AND n.name = name) \
                      RETURN DISTINCT id(n) AS nid";
        let names_param: Vec<Value> = names.iter().map(|n| Value::String(n.clone())).collect();
        match session
            .query_with(cypher)
            .param("names", Value::List(names_param))
            .fetch_all()
            .await
        {
            Ok(r) => r
                .rows()
                .iter()
                .filter_map(|row| row.get::<i64>("nid").ok())
                .collect(),
            Err(e) => {
                tracing::debug!(error = %e, "resolve_entity_seeds query failed");
                Vec::new()
            }
        }
    }

    /// Temporal-channel fan-out over `[lo, hi)`: Fact via BTIC overlap,
    /// Observation + Episode via BTree range, with per-arm limits.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`](crate::UnikoError::Storage) on
    /// query failure.
    pub async fn temporal_window_hits(
        &self,
        lo: DateTime<Utc>,
        hi: DateTime<Utc>,
        lim_fact: i64,
        lim_obs: i64,
        lim_eps: i64,
    ) -> Result<Vec<TemporalRow>> {
        let qbtic = temporal_window_to_btic_value(lo, hi);
        let session = self.db.session();
        let cypher = "MATCH (f:Fact) \
                      WHERE f.valid_at IS NOT NULL \
                        AND btic_overlaps(f.valid_at, $qbtic) \
                      WITH f LIMIT $lim_fact \
                      RETURN id(f) AS nid, labels(f)[0] AS lbl, 'fact' AS source, \
                             (coalesce(f.subject, '') + ' ' \
                              + coalesce(f.predicate, '') + ' ' \
                              + coalesce(f.object, '')) AS content \
                      UNION ALL \
                      MATCH (o:Observation) \
                      WHERE o.temporal_anchor >= $lo \
                        AND o.temporal_anchor < $hi \
                      WITH o LIMIT $lim_obs \
                      RETURN id(o) AS nid, labels(o)[0] AS lbl, 'observation' AS source, \
                             coalesce(o.content, '') AS content \
                      UNION ALL \
                      MATCH (e:Episode) \
                      WHERE e.timestamp >= $lo \
                        AND e.timestamp < $hi \
                      WITH e LIMIT $lim_eps \
                      RETURN id(e) AS nid, labels(e)[0] AS lbl, 'episode' AS source, \
                             coalesce(e.action_type, '') AS content";
        let result = session
            .query_with(cypher)
            .param("qbtic", qbtic)
            .param("lo", datetime_value(lo))
            .param("hi", datetime_value(hi))
            .param("lim_fact", lim_fact)
            .param("lim_obs", lim_obs)
            .param("lim_eps", lim_eps)
            .fetch_all()
            .await?;
        let mut out = Vec::with_capacity(result.rows().len());
        for row in result.rows() {
            let Ok(nid) = row.get::<i64>("nid") else {
                continue;
            };
            let source: String = row.get("source").unwrap_or_default();
            let default_label = match source.as_str() {
                "fact" => "Fact",
                "observation" => "Observation",
                "episode" => "Episode",
                _ => "Unknown",
            };
            let label: String = row.get("lbl").unwrap_or_else(|_| default_label.into());
            let content: String = row.get("content").unwrap_or_default();
            out.push(TemporalRow {
                node_id: nid,
                label,
                source,
                content,
            });
        }
        Ok(out)
    }

    /// Hydrate PPR-activated node ids with their label + best-effort
    /// content (one round-trip). `content` falls through the common text-
    /// bearing fields across label types.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`](crate::UnikoError::Storage) on
    /// query failure.
    pub async fn fetch_node_contents(&self, nids: &[NodeId]) -> Result<Vec<NodeContentRow>> {
        let session = self.db.session();
        let cypher = "UNWIND $nids AS nid \
                      MATCH (n) WHERE id(n) = nid \
                      RETURN nid, labels(n)[0] AS lbl, \
                             coalesce(n.content, \
                                      n.action_type, \
                                      (coalesce(n.subject, '') + ' ' \
                                       + coalesce(n.predicate, '') + ' ' \
                                       + coalesce(n.object, '')), \
                                      n.name, \
                                      '') AS content";
        let nids_param: Vec<Value> = nids.iter().map(|&v| Value::Int(v)).collect();
        let result = session
            .query_with(cypher)
            .param("nids", Value::List(nids_param))
            .fetch_all()
            .await?;
        let mut out = Vec::with_capacity(result.rows().len());
        for row in result.rows() {
            let Ok(nid) = row.get::<i64>("nid") else {
                continue;
            };
            out.push(NodeContentRow {
                node_id: nid,
                label: row.get("lbl").unwrap_or_default(),
                content: row.get("content").unwrap_or_default(),
            });
        }
        Ok(out)
    }

    /// Distinct session-Chunk node ids reachable from Fact `fact_node_id`
    /// via `SUPPORTED_BY → OBSERVED_IN → IN_SESSION → HAS_CHUNK`.
    ///
    /// The node id is **bound** as a parameter (issue #2: this walk
    /// previously interpolated the id directly into the Cypher string).
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`](crate::UnikoError::Storage) on
    /// query failure.
    pub async fn fact_session_chunk_ids(&self, fact_node_id: NodeId) -> Result<Vec<NodeId>> {
        let session = self.db.session();
        let cypher = "MATCH (f) WHERE id(f) = $nid \
                      MATCH (f)<-[:SUPPORTED_BY]-(:Observation)-[:OBSERVED_IN]->(:Message) \
                           -[:IN_SESSION]->(:Session)-[:HAS_CHUNK]->(c:Chunk) \
                      RETURN DISTINCT id(c) AS chunk_id";
        let result = session
            .query_with(cypher)
            .param("nid", fact_node_id)
            .fetch_all()
            .await?;
        Ok(result
            .rows()
            .iter()
            .filter_map(|row| row.get::<i64>("chunk_id").ok())
            .collect())
    }

    /// Does any Entity reachable from `node_id` (via `MENTIONS`/`ABOUT`)
    /// carry `entity_type = target_type` — or is the node itself such an
    /// Entity? Both values are bound as parameters.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`](crate::UnikoError::Storage) on
    /// query failure (the caller treats errors as "no match").
    pub async fn entity_type_matches(&self, node_id: NodeId, target_type: &str) -> Result<bool> {
        let session = self.db.session();
        let q = "MATCH (n) WHERE id(n) = $nid \
                 OPTIONAL MATCH (n)-[:MENTIONS|ABOUT]->(e1:Entity {entity_type: $t}) \
                 OPTIONAL MATCH (n)<-[:ABOUT]-(:Observation)-[:ABOUT]->(e2:Entity {entity_type: $t}) \
                 RETURN (n.entity_type = $t OR e1 IS NOT NULL OR e2 IS NOT NULL) AS hit \
                 LIMIT 1";
        let result = session
            .query_with(q)
            .param("nid", node_id)
            .param("t", target_type)
            .fetch_all()
            .await?;
        Ok(result
            .rows()
            .first()
            .and_then(|row| row.get::<bool>("hit").ok())
            .unwrap_or(false))
    }

    /// Per-variant chunk hybrid (vector+BM25 over `session`/`observation`
    /// chunks) plus the entity-scoped fan-out, max-merged per node id.
    ///
    /// Infallible by contract: individual sub-queries that fail are logged
    /// at debug and skipped (the cascade is best-effort). The structural
    /// `similar_to` source/query/fusion lists are interpolated; all values
    /// (`chunk_type`, `ename`, vectors, text, limit) are bound.
    ///
    /// Returns the merged hits including empty-content rows — the caller
    /// applies its own empty-content filter and ranking.
    #[must_use]
    pub async fn recall_chunk_and_entity_scoped(
        &self,
        qvec: &[f32],
        qtxt: &str,
        entity_refs: &[String],
        per_variant_limit: i64,
        vector_weight: f64,
        bm25_weight: f64,
    ) -> Vec<ScoredRow> {
        use std::collections::HashMap;

        let session = self.db.session();
        let mut scored: HashMap<NodeId, (String, String, f64)> = HashMap::new();
        let has_vec = !qvec.is_empty();

        // Hybrid (vector + BM25) over chunk types — same query shape, only
        // the chunk_type filter differs.
        let (sources, queries, fusion, needs_qvec, needs_qtxt) = if has_vec {
            (
                "[m.embedding, m.text]",
                "[$qvec, $qtxt]",
                format!(", {{method: 'weighted', weights: [{vector_weight}, {bm25_weight}]}}"),
                true,
                true,
            )
        } else {
            ("m.text", "$qtxt", String::new(), false, true)
        };

        for chunk_type in ["session", "observation"] {
            let cypher = format!(
                "MATCH (m:Chunk) WHERE m.chunk_type = $ctype \
                 RETURN id(m) AS nid, labels(m)[0] AS lbl, \
                        m.text AS content, \
                        similar_to({sources}, {queries}{fusion}) AS score \
                 ORDER BY score DESC LIMIT $lim"
            );
            let mut builder = session.query_with(&cypher);
            builder = builder.param("ctype", chunk_type);
            builder = builder.param("lim", per_variant_limit);
            if needs_qvec {
                builder = builder.param("qvec", Value::Vector(qvec.to_vec()));
            }
            if needs_qtxt {
                builder = builder.param("qtxt", qtxt);
            }
            match builder.fetch_all().await {
                Ok(result) => merge_scored_rows(result.rows(), &mut scored),
                Err(e) => tracing::debug!(chunk_type, error = %e, "hybrid similar_to failed"),
            }
        }

        // Entity-scoped fan-out. The cypher per target depends only on
        // `has_vec` + the per-target tuple, not on `entity_name`, so build
        // the 6 templates once and bind $ename per entity.
        let entity_scoped_targets: &[(&str, &str, &str, &str)] = &[
            (
                "Chunk",
                "text",
                "embedding",
                "(m)<-[:HAS_CHUNK]-(:Session)<-[:PARTICIPATED_IN]-(:Participant {name: $ename})",
            ),
            (
                "Chunk",
                "text",
                "embedding",
                "(m)-[:ABOUT]->(:Participant {name: $ename})",
            ),
            (
                "Chunk",
                "text",
                "embedding",
                "(m)-[:ABOUT]->(:Entity {name: $ename})",
            ),
            (
                "Observation",
                "content",
                "embedding",
                "(m)-[:ABOUT]->(:Participant {name: $ename})",
            ),
            (
                "Observation",
                "content",
                "embedding",
                "(m)-[:ABOUT]->(:Entity {name: $ename})",
            ),
            ("Observation", "content", "embedding", "m.subject = $ename"),
        ];
        let entity_cyphers: Vec<(String, &str)> = entity_scoped_targets
            .iter()
            .map(|&(label, content_field, embed_field, pattern)| {
                let cypher = if has_vec {
                    format!(
                        "MATCH (m:{label}) WHERE {pattern} \
                         RETURN id(m) AS nid, labels(m)[0] AS lbl, \
                                m.{content_field} AS content, \
                                similar_to([m.{embed_field}, m.{content_field}], [$qvec, $qtxt]) AS score \
                         ORDER BY score DESC LIMIT $lim"
                    )
                } else {
                    format!(
                        "MATCH (m:{label}) WHERE {pattern} \
                         RETURN id(m) AS nid, labels(m)[0] AS lbl, \
                                m.{content_field} AS content, \
                                similar_to(m.{content_field}, $qtxt) AS score \
                         ORDER BY score DESC LIMIT $lim"
                    )
                };
                (cypher, label)
            })
            .collect();

        for entity_name in entity_refs {
            for (cypher, label) in &entity_cyphers {
                let mut builder = session.query_with(cypher);
                builder = builder
                    .param("ename", entity_name.as_str())
                    .param("qtxt", qtxt)
                    .param("lim", per_variant_limit);
                if has_vec {
                    builder = builder.param("qvec", Value::Vector(qvec.to_vec()));
                }
                match builder.fetch_all().await {
                    Ok(result) => merge_scored_rows(result.rows(), &mut scored),
                    Err(e) => tracing::debug!(
                        entity = entity_name,
                        label,
                        error = %e,
                        "entity-scoped search failed"
                    ),
                }
            }
        }

        scored
            .into_iter()
            .map(|(node_id, (label, content, score))| ScoredRow {
                node_id,
                label,
                content,
                score,
            })
            .collect()
    }
}

/// Decode a batch of `nid`/`lbl`/`content`/`score` rows into
/// [`ScoredRow`]s. Rows with an unreadable `nid` are skipped.
fn decode_scored_rows(rows: &[uni_db::Row]) -> Vec<ScoredRow> {
    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        let Ok(nid) = row.get::<i64>("nid") else {
            continue;
        };
        out.push(ScoredRow {
            node_id: nid,
            label: row.get("lbl").unwrap_or_default(),
            content: row.get("content").unwrap_or_default(),
            score: row.get("score").unwrap_or(0.0),
        });
    }
    out
}

/// Merge a row batch into the per-variant `scored` accumulator, keeping
/// the max score per node id (first label/content wins).
fn merge_scored_rows(
    rows: &[uni_db::Row],
    scored: &mut std::collections::HashMap<NodeId, (String, String, f64)>,
) {
    for row in rows {
        let nid: i64 = row.get("nid").unwrap_or(0);
        let lbl: String = row.get("lbl").unwrap_or_default();
        let content: String = row.get("content").unwrap_or_default();
        let score: f64 = row.get("score").unwrap_or(0.0);
        scored
            .entry(nid)
            .and_modify(|(_, _, s)| *s = s.max(score))
            .or_insert((lbl, content, score));
    }
}

/// Build a `Value::Temporal(Btic { ... })` covering `[lo, hi)` for the
/// `btic_overlaps()` UDF.
fn temporal_window_to_btic_value(lo: DateTime<Utc>, hi: DateTime<Utc>) -> Value {
    let btic = crate::schema::btic::btic_query_window(lo, hi);
    Value::Temporal(uni_db::common::TemporalValue::Btic {
        lo: btic.lo(),
        hi: btic.hi(),
        meta: btic.meta(),
    })
}
