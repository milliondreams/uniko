//! Pipeline 3 — Observation extraction.
//!
//! [`ObservationExtractionStep`] implements the [`Step`](uniko_pipes::Step)
//! trait.  It filters non-informative content, extracts factual
//! statements anchored to P2 entities, creates Observation nodes, and
//! wires OBSERVED_IN + ABOUT edges.
//!
//! **Key principle**: observations are direct statements from messages.
//! P3 never creates Facts — that is P4's job.

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
/// 1. Filter non-informative content (greetings, questions, short).
/// 2. Load entity names from `ctx.extracted_entities`.
/// 3. Rule-based extraction anchored to P2 entities.
/// 4. LLM enhancement (stub).
/// 5. Create Observation nodes + OBSERVED_IN + ABOUT edges.
/// 6. Populate `ctx.extracted_observations`.
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
        // 1. Filter non-informative content.
        //    Prefer NLP model's CLS classification when available;
        //    fall back to rule-based heuristics otherwise.
        let content_type = ctx
            .metadata
            .get("content_type")
            .and_then(|v| v.as_str())
            .or(Some(ctx.content_type.as_str()));

        // Use rule-based filtering — the CLS model classifies dialogue
        // acts (inform/agreement/social) which doesn't align with "contains
        // extractable facts." Many informative statements are casual
        // ("Yeah, I started a dance studio") and get classified as
        // "agreement" or "social" by the CLS model.
        let informative = filter::is_informative(&ctx.content, content_type);

        if !informative {
            return Ok(StepOutcome::Skipped {
                reason: "content not informative".into(),
            });
        }

        // 2. Load entity names from context (populated by P2) + sender.
        let mut entity_refs = load_entity_refs(ctx).await;
        // Also add the message sender as an entity ref — many messages
        // contain facts about the speaker without naming them explicitly.
        if let Some(sender_ref) = load_sender_ref(ctx).await {
            if !entity_refs.iter().any(|(_, name)| name == &sender_ref.1) {
                entity_refs.push(sender_ref);
            }
        }
        if entity_refs.is_empty() {
            return Ok(StepOutcome::Skipped {
                reason: "no entities from P2 to anchor observations".into(),
            });
        }

        // 3. Resolve the message timestamp.
        let timestamp = resolve_timestamp(ctx);

        // 4. Dep-tree SVO extraction (when NLP result is available).
        //    Falls back to rule-based extraction otherwise.
        let mut all_obs = Vec::new();

        #[cfg(feature = "onnx")]
        if let Some(nlp_val) = ctx.metadata.get("nlp_result")
            && let Ok(nlp_result) =
                serde_json::from_value::<crate::nlp::types::NlpResult>(nlp_val.clone())
        {
            let dep_obs =
                rules::extract_observations_from_dep_tree(&nlp_result, &entity_refs, timestamp);
            tracing::debug!(count = dep_obs.len(), "dep-tree observations");
            all_obs.extend(dep_obs);
        }

        // Rule-based extraction always runs to catch patterns the
        // model might miss (preferences, temporal expressions).
        let rule_obs =
            rules::extract_observations_rule_based(&ctx.content, &entity_refs, timestamp);
        tracing::debug!(count = rule_obs.len(), "rule-based observations");
        all_obs.extend(rule_obs);

        // 5. LLM enhancement (stub — returns empty).
        let llm_obs = llm::enhance_observations_llm(&ctx.content, &entity_refs).await;
        all_obs.extend(llm_obs);

        if all_obs.is_empty() {
            return Ok(StepOutcome::Skipped {
                reason: "no observations extracted".into(),
            });
        }

        // 6. Create nodes and wire edges.
        let mut obs_node_ids = Vec::with_capacity(all_obs.len());
        for raw in &all_obs {
            let obs_nid = create_observation_node(&ctx.kb, raw).await?;

            // OBSERVED_IN: Observation → source node (Message or Chunk).
            ctx.kb
                .create_edge(edges::OBSERVED_IN, obs_nid, ctx.node_id, &HashMap::new())
                .await?;

            // ABOUT: Observation → each entity whose name appears in content.
            for &(entity_nid, ref entity_name) in &entity_refs {
                if raw
                    .content
                    .to_lowercase()
                    .contains(&entity_name.to_lowercase())
                    || raw.subject.to_lowercase() == entity_name.to_lowercase()
                {
                    ctx.kb
                        .create_edge(edges::ABOUT, obs_nid, entity_nid, &HashMap::new())
                        .await?;
                }
            }

            obs_node_ids.push(obs_nid);
        }

        // 7. Populate context for downstream steps.
        ctx.extracted_observations = obs_node_ids.clone();

        tracing::info!(
            count = obs_node_ids.len(),
            "observation extraction completed",
        );

        Ok(StepOutcome::Completed)
    }

    fn error_policy(&self) -> StepErrorPolicy {
        StepErrorPolicy::Skip
    }
}

/// Create an Observation node in the graph.
async fn create_observation_node(
    kb: &uniko_store::KnowledgeBase,
    raw: &RawObservation,
) -> uniko_store::Result<NodeId> {
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
    kb.create_node(labels::OBSERVATION, &props).await
}

/// Load the message sender as an entity reference via SENT_BY edge.
///
/// Returns the Participant's (node_id, name) if available.
async fn load_sender_ref(ctx: &PipelineContext) -> Option<(NodeId, String)> {
    use uniko_store::storage::edges::Direction;

    let edges = ctx
        .kb
        .get_edges(ctx.node_id, "SENT_BY", Direction::Outgoing)
        .await
        .ok()?;
    let edge = edges.first()?;
    let (_, props) = ctx.kb.get_node(edge.to).await.ok()??;
    let name = props.get("name").and_then(|v| v.as_str())?.to_string();
    Some((edge.to, name))
}

/// Load entity (NodeId, name) pairs from the pipeline context.
///
/// Reads `ctx.extracted_entities` and fetches entity names from the
/// graph for each node ID.
async fn load_entity_refs(ctx: &PipelineContext) -> Vec<(NodeId, String)> {
    let mut refs = Vec::with_capacity(ctx.extracted_entities.len());
    for &nid in &ctx.extracted_entities {
        match ctx.kb.get_node(nid).await {
            Ok(Some((_, props))) => {
                if let Some(name) = props.get("name").and_then(|v| v.as_str()) {
                    refs.push((nid, name.to_string()));
                }
            }
            _ => continue,
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
