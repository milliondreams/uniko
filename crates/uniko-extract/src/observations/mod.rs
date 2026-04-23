//! Pipeline 3 — Observation extraction.
//!
//! [`ObservationExtractionStep`] implements the [`Step`](uniko_pipes::Step)
//! trait. Extracts clean declarative observations from messages using the
//! NLP model's dependency tree, with CLS-based filtering and speaker
//! attribution for first-person pronouns.
//!
//! **Key principle**: observations are reconstructed from the DEP tree,
//! not raw text fragments. "I'm starting a dance studio" → "Jon is
//! starting a dance studio" (clean, declarative, speaker-attributed).

// Rust guideline compliant

pub mod contradiction;
pub mod filter;
pub mod llm;
pub mod rules;
pub mod temporal;
pub mod types;

pub use types::{ContradictionFlag, RawObservation};

use std::collections::HashMap;

use async_trait::async_trait;
use uni_db::Value;

use uniko_pipes::step::PipelineContext;
use uniko_pipes::types::{StepErrorPolicy, StepOutcome};
use uniko_store::schema::constants::{edges, labels};
use uniko_store::{NodeId, UnikoError};

/// Pipeline step that extracts observations from content.
///
/// Orchestration:
/// 1. CLS gate: only "inform" and "plan_commit" proceed.
/// 2. DEP tree extraction: reconstruct subject-verb-object from parse.
/// 3. Speaker substitution: "I"/"we" → sender name.
/// 4. Create Observation nodes + OBSERVED_IN + ABOUT edges.
/// 5. Fallback to rule-based when NLP unavailable.
#[derive(Debug)]
pub struct ObservationExtractionStep;

#[async_trait]
impl uniko_pipes::Step for ObservationExtractionStep {
    fn name(&self) -> &str {
        "observation_extraction"
    }

    fn should_run(&self, ctx: &PipelineContext) -> bool {
        ctx.node_id != 0 && !ctx.content.is_empty()
    }

    async fn execute(&self, ctx: &mut PipelineContext) -> Result<StepOutcome, UnikoError> {
        let step_start = std::time::Instant::now();

        // Resolve sender name early — needed for both CLS check and DEP extraction.
        let sender_ref = load_sender_ref(ctx).await;
        let sender_ms = step_start.elapsed().as_millis();

        #[allow(unused_variables)]
        let sender_name = sender_ref.as_ref().map(|(_, name)| name.as_str());

        // 1. CLS gate.
        //    When per-sentence NLP results are available (step 3a), CLS
        //    filtering happens per-sentence inside the extraction loop.
        //    Otherwise (no ONNX, or NLP unavailable), apply the rule-based
        //    filter upfront to reject greetings, short filler, etc.
        let has_nlp = cfg!(feature = "onnx") && ctx.metadata.contains_key("nlp_results");
        if !has_nlp {
            let content_type = ctx
                .metadata
                .get("content_type")
                .and_then(|v| v.as_str())
                .or(Some(ctx.content_type.as_str()));
            if !filter::is_informative(&ctx.content, content_type) {
                return Ok(StepOutcome::Skipped {
                    reason: "content not informative".into(),
                });
            }
        }

        // 2. Resolve timestamp.
        let timestamp = resolve_timestamp(ctx);

        // 3. Extract observations.
        let extract_start = std::time::Instant::now();
        let mut all_obs = Vec::new();
        #[allow(unused_mut)]
        let mut used_model = false;

        // 3a. Model-driven extraction from per-sentence DEP trees.
        //     Each sentence has its own CLS label — only informative
        //     sentences produce observations.
        #[cfg(feature = "onnx")]
        if let Some(nlp_val) = ctx.metadata.get("nlp_results")
            && let Ok(nlp_results) =
                serde_json::from_value::<Vec<crate::nlp::types::NlpResult>>(nlp_val.clone())
        {
            let labels = crate::nlp::assets::label_maps();
            let speaker = sender_name.unwrap_or("unknown");

            // Load or create sentence context for pronoun resolution.
            let mut sent_ctx = ctx
                .metadata
                .get("session_context")
                .and_then(|v| v.get("sentence_ctx"))
                .and_then(|v| {
                    serde_json::from_value::<crate::ingest::context::SentenceContext>(v.clone())
                        .ok()
                })
                .unwrap_or_else(|| {
                    crate::ingest::context::SentenceContext::new(speaker, Vec::new())
                });

            for nlp_result in &nlp_results {
                // Per-sentence CLS gate.
                if !nlp_result.sentence_class.is_informative() {
                    // Still update context from skipped sentences —
                    // nouns in greetings can be useful antecedents.
                    crate::nlp::decode::update_sentence_context(
                        &mut sent_ctx,
                        &nlp_result.words,
                        &nlp_result.pos_indices,
                        &nlp_result.dep_arcs,
                        &labels.pos_labels,
                    );
                    continue;
                }

                let dep_obs = crate::nlp::decode::extract_dep_observations(
                    &nlp_result.words,
                    &nlp_result.pos_indices,
                    &nlp_result.dep_arcs,
                    &labels.pos_labels,
                    speaker,
                    &mut sent_ctx,
                );

                for obs in dep_obs {
                    all_obs.push(RawObservation {
                        content: obs.content,
                        subject: obs.subject,
                        observed_at: timestamp,
                        confidence: obs.confidence,
                    });
                }
            }

            // Write updated sentence context back to metadata for
            // the caller to persist in SessionContext.
            if let Ok(val) = serde_json::to_value(&sent_ctx) {
                ctx.metadata.insert("sentence_ctx_updated".into(), val);
            }

            used_model = true;
        }

        // 3b. Rule-based fallback (only when model unavailable).
        if !used_model {
            let entity_refs = load_all_entity_refs(ctx, &sender_ref).await;
            if !entity_refs.is_empty() {
                let rule_obs =
                    rules::extract_observations_rule_based(&ctx.content, &entity_refs, timestamp);
                all_obs.extend(rule_obs);
            }
        }
        let extract_ms = extract_start.elapsed().as_millis();

        if all_obs.is_empty() {
            return Ok(StepOutcome::Skipped {
                reason: "no observations extracted".into(),
            });
        }

        // 4. Create nodes and wire edges (batched).
        let persist_start = std::time::Instant::now();
        let entity_refs = load_all_entity_refs(ctx, &sender_ref).await;
        let entity_ref_ms = persist_start.elapsed().as_millis();

        let nodes_start = std::time::Instant::now();

        // 4a. Batch create all Observation nodes.
        let obs_props: Vec<HashMap<String, Value>> = all_obs
            .iter()
            .map(|raw| {
                let obs_id = uniko_store::id::new_id();
                let mut props = HashMap::new();
                props.insert("observation_id".into(), Value::String(obs_id));
                props.insert("content".into(), Value::String(raw.content.clone()));
                props.insert("subject".into(), Value::String(raw.subject.clone()));
                props.insert(
                    "observed_at".into(),
                    Value::String(raw.observed_at.to_rfc3339()),
                );
                props.insert("confidence".into(), Value::Float(raw.confidence));
                props
            })
            .collect();
        let obs_node_ids = ctx
            .kb
            .batch_create_nodes(labels::OBSERVATION, &obs_props)
            .await?;

        // 4b. Batch create OBSERVED_IN edges (Observation → source node).
        let observed_edges: Vec<(NodeId, NodeId, HashMap<String, Value>)> = obs_node_ids
            .iter()
            .map(|&nid| (nid, ctx.node_id, HashMap::new()))
            .collect();
        ctx.kb
            .batch_create_edges(edges::OBSERVED_IN, &observed_edges)
            .await?;

        // 4c. Batch create ABOUT edges.
        let mut about_edges: Vec<(NodeId, NodeId, HashMap<String, Value>)> = Vec::new();
        for (i, raw) in all_obs.iter().enumerate() {
            let obs_nid = obs_node_ids[i];

            // Wire to speaker (always).
            if let Some((sender_nid, _)) = &sender_ref {
                about_edges.push((obs_nid, *sender_nid, HashMap::new()));
            }
            // Wire to entities matching the observation subject.
            for &(entity_nid, ref name) in &entity_refs {
                if sender_ref
                    .as_ref()
                    .is_some_and(|(nid, _)| *nid == entity_nid)
                {
                    continue;
                }
                // Substring match: "dance studio" matches entity "dance studio",
                // and "studio" matches "dance studio" (or vice versa).
                let subj = raw.subject.to_lowercase();
                let ename = name.to_lowercase();
                if subj == ename || subj.contains(&ename) || ename.contains(&subj) {
                    about_edges.push((obs_nid, entity_nid, HashMap::new()));
                }
            }
        }
        if !about_edges.is_empty() {
            ctx.kb
                .batch_create_edges(edges::ABOUT, &about_edges)
                .await?;
        }
        let nodes_ms = nodes_start.elapsed().as_millis();

        // 5. Populate context for downstream steps.
        ctx.extracted_observations = obs_node_ids.clone();

        tracing::info!(
            count = obs_node_ids.len(),
            model = used_model,
            sender_ms,
            extract_ms,
            entity_ref_ms,
            nodes_ms,
            total_ms = step_start.elapsed().as_millis(),
            "observation step",
        );

        Ok(StepOutcome::Completed)
    }

    fn error_policy(&self) -> StepErrorPolicy {
        StepErrorPolicy::Skip
    }
}

/// Load the message sender via SENT_BY edge.
async fn load_sender_ref(ctx: &PipelineContext) -> Option<(NodeId, String)> {
    use uniko_store::storage::edges::Direction;

    let edge_list = ctx
        .kb
        .get_edges(ctx.node_id, edges::SENT_BY, Direction::Outgoing)
        .await
        .ok()?;
    let edge = edge_list.first()?;
    let (_, props) = ctx.kb.get_node(edge.to).await.ok()??;
    let name = props.get("name").and_then(|v| v.as_str())?.to_string();
    Some((edge.to, name))
}

/// Load sender + NER entity refs combined.
async fn load_all_entity_refs(
    ctx: &PipelineContext,
    sender_ref: &Option<(NodeId, String)>,
) -> Vec<(NodeId, String)> {
    let mut refs = Vec::new();
    if let Some(sr) = sender_ref {
        refs.push(sr.clone());
    }
    for &nid in &ctx.extracted_entities {
        if let Ok(Some((_, props))) = ctx.kb.get_node(nid).await {
            if let Some(name) = props.get("name").and_then(|v| v.as_str()) {
                if !refs.iter().any(|(_, n)| n == name) {
                    refs.push((nid, name.to_string()));
                }
            }
        }
    }
    refs
}

/// Extract the message timestamp from context metadata or use now.
fn resolve_timestamp(ctx: &PipelineContext) -> chrono::DateTime<chrono::Utc> {
    ctx.metadata
        .get("timestamp")
        .and_then(|v| v.as_str())
        .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
        .map(|dt| dt.with_timezone(&chrono::Utc))
        .unwrap_or_else(chrono::Utc::now)
}
