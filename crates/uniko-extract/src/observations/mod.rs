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

pub mod cleanup;
pub mod contradiction;
pub mod filter;
pub mod llm;
pub mod rules;
#[cfg(feature = "onnx")]
pub mod rules_engine;
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
/// 1. CLS gate: only `inform`/`request`/`confirm`/`offer`/`status` proceed.
/// 2. DEP tree extraction: reconstruct subject-verb-object from the
///    biaffine parse.
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
        let timestamp = resolve_timestamp(ctx);

        // Reconstruct ObservationInputs from the PipelineContext shape
        // so the prep + apply helpers can be the single source of truth
        // for both the legacy step path and the new atomic ingest path.
        #[cfg(feature = "onnx")]
        let nlp_results_owned: Option<Vec<crate::nlp::types::NlpResult>> = ctx
            .metadata
            .get("nlp_results")
            .and_then(|v| serde_json::from_value(v.clone()).ok());
        let seed_sentence_ctx = ctx
            .metadata
            .get("session_context")
            .and_then(|v| v.get("sentence_ctx"))
            .and_then(|v| {
                serde_json::from_value::<crate::ingest::context::SentenceContext>(v.clone()).ok()
            });

        let rules_path = ctx.kb.config().observation_rules_path.clone();

        let inputs = ObservationInputs {
            kb: &ctx.kb,
            message_node_id: ctx.node_id,
            content: &ctx.content,
            content_type: ctx
                .metadata
                .get("content_type")
                .and_then(|v| v.as_str())
                .unwrap_or(ctx.content_type.as_str()),
            sender: ctx.sender.clone(),
            extracted_entities: &ctx.extracted_entities,
            #[cfg(feature = "onnx")]
            nlp_results: nlp_results_owned.as_deref(),
            seed_sentence_ctx: seed_sentence_ctx.as_ref(),
            timestamp,
            observation_rules_path: rules_path.as_deref(),
        };

        let prep = match prepare_observations(inputs).await? {
            ObservationPrepOutcome::Skip(reason) => {
                return Ok(StepOutcome::Skipped { reason });
            }
            ObservationPrepOutcome::Ready(p) => p,
        };

        // Mirror the prep's sentence_ctx_updated into ctx.metadata so
        // the bench / orchestrator can persist it back into
        // SessionContext (existing behaviour).
        if let Some(sc) = &prep.sentence_ctx_updated
            && let Ok(val) = serde_json::to_value(sc)
        {
            ctx.metadata.insert("sentence_ctx_updated".into(), val);
        }

        let sender_ms = prep.sender_ms;
        let extract_ms = prep.extract_ms;
        let used_model = prep.used_model;
        let obs_count = prep.all_obs.len();

        // Open the tx + apply writes + commit (legacy semantics — own tx).
        let nodes_start = std::time::Instant::now();
        let tx = ctx
            .kb
            .db()
            .session()
            .tx()
            .await
            .map_err(|err| UnikoError::Storage(err.to_string()))?;
        let obs_node_ids = apply_observations(&ctx.kb, &tx, ctx.node_id, prep).await?;
        let commit_start = std::time::Instant::now();
        tx.commit()
            .await
            .map_err(|err| UnikoError::Storage(err.to_string()))?;
        let commit_ms = commit_start.elapsed().as_millis() as u64;
        let nodes_ms = nodes_start.elapsed().as_millis();
        tracing::info!(commit_ms, "obs commit");
        tracing::info!(
            target: "tx_perf",
            tx_phase = "obs_step",
            total_ms = nodes_ms as u64,
            commit_ms,
            obs_count = obs_node_ids.len() as u64,
            "tx phase",
        );

        ctx.extracted_observations = obs_node_ids.clone();
        tracing::info!(
            count = obs_node_ids.len(),
            model = used_model,
            sender_ms,
            extract_ms,
            nodes_ms,
            total_ms = step_start.elapsed().as_millis(),
            "observation step",
        );

        let _ = obs_count;
        Ok(StepOutcome::Completed)
    }

    fn error_policy(&self) -> StepErrorPolicy {
        StepErrorPolicy::Skip
    }
}


/// Load the observation rule set, honouring `UnikoConfig.observation_rules_path`.
///
/// External paths are read once and cached in a process-global map keyed
/// by canonical path. The bundled `english.yml` is also cached (parsed
/// once) — reused across every message in a process.
#[cfg(feature = "onnx")]
fn load_observation_rules_from_path(
    cfg_path: Option<&std::path::Path>,
) -> &'static crate::observations::rules_engine::Rules {
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::sync::{Mutex, OnceLock};

    static EXTERNAL: OnceLock<
        Mutex<HashMap<PathBuf, &'static crate::observations::rules_engine::Rules>>,
    > = OnceLock::new();

    if let Some(path) = cfg_path {
        let key = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        let map = EXTERNAL.get_or_init(|| Mutex::new(HashMap::new()));
        let mut guard = map.lock().expect("observation rules cache poisoned");
        if let Some(r) = guard.get(&key) {
            return r;
        }
        match crate::observations::rules_engine::load_rules_from_path(&key) {
            Ok(rules) => {
                let leaked: &'static _ = Box::leak(Box::new(rules));
                guard.insert(key, leaked);
                return leaked;
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "failed to load observation_rules_path; falling back to bundled rules"
                );
            }
        }
    }
    crate::observations::rules_engine::load_bundled_rules()
}

/// Multi-label CLS gate.
///
/// The CLS head emits a softmax over 8 raw labels. We accept a sentence
/// if **any** label whose prob clears `min_prob` belongs to the
/// configured informative set (default: `inform`, `status`, `request`,
/// `offer`). This handles the common case where the argmax is `social`
/// but the sentence carries propositional content (e.g. self-status
/// reports with affective language).
///
/// Falls back to the legacy `is_informative()` argmax check when
/// `cls_probs` is empty (e.g. data serialised before this field
/// existed).
#[cfg(feature = "onnx")]
fn cls_gate_admits(
    nlp_result: &crate::nlp::types::NlpResult,
    cls_labels: &[String],
    gate: &crate::observations::rules_engine::rules::ClsGate,
) -> bool {
    if nlp_result.cls_probs.is_empty() {
        return nlp_result.sentence_class.is_informative();
    }
    for (i, &p) in nlp_result.cls_probs.iter().enumerate() {
        if p < gate.min_prob {
            continue;
        }
        if let Some(name) = cls_labels.get(i)
            && gate.informative_labels.iter().any(|l| l == name)
        {
            return true;
        }
    }
    false
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

/// Substitute a resolved absolute date for a relative temporal phrase
/// in the observation `content` string, and return the resolved date
/// for the structured `temporal_anchor` slot.
///
/// The SRL rule engine renders observations with the surface form of a
/// temporal expression — `"Caroline went to the LGBTQ support group
/// yesterday"`.  Downstream consumers (LLM responders that don't share
/// the conversation date anchor, fact consolidation, downstream
/// embedders) shouldn't have to redo the relative-date math.  We
/// rewrite the content to embed the absolute date directly:
/// `"Caroline went to the LGBTQ support group on 2023-05-07"`.  The raw
/// surface phrase is preserved separately on
/// `RawObservation.temporal_phrase` for analytics.
///
/// Uses [`temporal::resolve_temporal_with_granularity`] (which returns
/// `None` on unparseable input) rather than [`temporal::resolve_temporal`]
/// (which silently falls back to `reference` on no-match) so that an
/// unrecognised phrase like `"next Whitsun"` doesn't get rewritten to
/// the message timestamp.
///
/// Returns the (possibly rewritten) content and the resolved anchor.
#[cfg_attr(not(feature = "onnx"), allow(dead_code))]
fn resolve_temporal_in_content(
    content: String,
    temporal_phrase: Option<&str>,
    reference: chrono::DateTime<chrono::Utc>,
) -> (String, Option<chrono::DateTime<chrono::Utc>>) {
    let resolved =
        temporal_phrase.and_then(|s| temporal::resolve_temporal_with_granularity(s, reference));
    let content = match (temporal_phrase, &resolved) {
        (Some(phrase), Some(r)) => {
            let date_str = r.point.format("%Y-%m-%d").to_string();
            content.replace(phrase, &date_str)
        }
        _ => content,
    };
    (content, resolved.map(|r| r.point))
}

// ── prep / apply split for the atomic ingest path ─────────────────

/// Inputs to [`prepare_observations`]. Bundles every read-only
/// dependency the prep phase needs. The atomic ingest orchestrator
/// builds one of these from typed inputs (no PipelineContext required).
#[derive(Debug)]
pub struct ObservationInputs<'a> {
    pub kb: &'a uniko_store::KnowledgeBase,
    /// VID of the Message that observations will reference.
    pub message_node_id: NodeId,
    pub content: &'a str,
    pub content_type: &'a str,
    /// Pre-resolved sender from the ingest step. When `None`, prep
    /// falls back to a SENT_BY lookup against `message_node_id`.
    pub sender: Option<(NodeId, String)>,
    /// `(NodeId, name)` pairs already created by the entity step.
    pub extracted_entities: &'a [(NodeId, String)],
    /// Per-sentence NLP results (POS / DEP / SRL / CLS). When `None`,
    /// the rule-based fallback runs.
    #[cfg(feature = "onnx")]
    pub nlp_results: Option<&'a [crate::nlp::types::NlpResult]>,
    /// Seed sentence context for pronoun resolution carried across
    /// messages within a session. `None` starts a fresh context.
    pub seed_sentence_ctx: Option<&'a crate::ingest::context::SentenceContext>,
    /// Effective message timestamp (from metadata or wall-clock).
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Optional path to a custom observation rules YAML; `None` uses
    /// bundled rules.
    pub observation_rules_path: Option<&'a std::path::Path>,
}

/// Output of a successful [`prepare_observations`] call.
#[derive(Debug)]
pub struct ObservationPrep {
    pub all_obs: Vec<RawObservation>,
    pub used_model: bool,
    /// Resolved sender (either passed in or loaded via SENT_BY lookup).
    pub sender_ref: Option<(NodeId, String)>,
    /// Combined sender + entity refs used for ABOUT-edge construction.
    pub entity_refs: Vec<(NodeId, String)>,
    /// Updated sentence context for the session. `None` when the model
    /// path was not taken (or no NLP results were available).
    pub sentence_ctx_updated: Option<crate::ingest::context::SentenceContext>,
    pub sender_ms: u128,
    pub extract_ms: u128,
}

impl ObservationPrep {
    pub fn is_empty(&self) -> bool {
        self.all_obs.is_empty()
    }
}

/// Whether prep produced observations to write, or short-circuited.
#[derive(Debug)]
pub enum ObservationPrepOutcome {
    /// CLS gate / no-entities / empty extraction. Caller skips the write.
    Skip(String),
    Ready(ObservationPrep),
}

/// CPU + optional SENT_BY lookup. Does NOT open a transaction; does
/// NOT write anything. Returns `ObservationPrepOutcome::Skip` when
/// the CLS gate rejects the message or no observations were extracted.
///
/// # Errors
///
/// Returns [`UnikoError::Storage`] on a SENT_BY-lookup failure when
/// `sender` is `None` and the slow path is needed.
pub async fn prepare_observations(
    input: ObservationInputs<'_>,
) -> Result<ObservationPrepOutcome, UnikoError> {
    let step_start = std::time::Instant::now();

    // Slow-path sender resolution: only when caller didn't already
    // resolve it (e.g. legacy path through ObservationExtractionStep
    // without ctx.sender populated). The atomic orchestrator always
    // populates `input.sender` from MessageIngestResult, so this is
    // a no-op there.
    let sender_ref = if let Some(s) = input.sender.clone() {
        Some(s)
    } else {
        load_sender_ref_by_lookup(input.kb, input.message_node_id).await
    };
    let sender_ms = step_start.elapsed().as_millis();

    #[allow(unused_variables)]
    let sender_name = sender_ref.as_ref().map(|(_, name)| name.as_str());

    // 1. CLS gate (only when no per-sentence NLP results — those
    //    handle CLS per-sentence in the extraction loop below).
    #[cfg(feature = "onnx")]
    let has_nlp = input.nlp_results.is_some();
    #[cfg(not(feature = "onnx"))]
    let has_nlp = false;
    if !has_nlp && !filter::is_informative(input.content, Some(input.content_type)) {
        return Ok(ObservationPrepOutcome::Skip(
            "content not informative".into(),
        ));
    }

    // 2. Extract observations.
    let extract_start = std::time::Instant::now();
    let mut all_obs = Vec::new();
    #[allow(unused_mut)]
    let mut used_model = false;
    #[allow(unused_mut)]
    let mut sentence_ctx_updated: Option<crate::ingest::context::SentenceContext> = None;

    // 2a. Model-driven extraction from per-sentence DEP trees.
    #[cfg(feature = "onnx")]
    if let Some(nlp_results) = input.nlp_results {
        let labels = crate::nlp::assets::label_maps();
        let speaker = sender_name.unwrap_or("unknown");

        let mut sent_ctx = input
            .seed_sentence_ctx
            .cloned()
            .unwrap_or_else(|| crate::ingest::context::SentenceContext::new(speaker, Vec::new()));

        let rules = load_observation_rules_from_path(input.observation_rules_path);

        for nlp_result in nlp_results {
            if !cls_gate_admits(nlp_result, &labels.cls_labels, &rules.filters.cls_gate) {
                crate::nlp::decode::update_sentence_context(
                    &mut sent_ctx,
                    &nlp_result.words,
                    &nlp_result.pos_indices,
                    &nlp_result.dep_arcs,
                    &labels.pos_labels,
                );
                continue;
            }

            let dep_obs = crate::observations::rules_engine::extract_with_rules(
                rules,
                &nlp_result.words,
                &nlp_result.pos_indices,
                &nlp_result.dep_arcs,
                &labels.pos_labels,
                &nlp_result.srl_frames,
                speaker,
                &mut sent_ctx,
            );

            for obs in dep_obs {
                let temporal_phrase = obs.temporal.clone();
                let (content, temporal_anchor) = resolve_temporal_in_content(
                    obs.content,
                    temporal_phrase.as_deref(),
                    input.timestamp,
                );
                all_obs.push(RawObservation {
                    content,
                    subject: obs.subject,
                    predicate: obs.predicate,
                    object: obs.object,
                    temporal_phrase,
                    temporal_anchor,
                    observed_at: input.timestamp,
                    confidence: obs.confidence,
                });
            }
        }

        sentence_ctx_updated = Some(sent_ctx);
        used_model = true;
    }

    // 2b. Rule-based fallback (only when model unavailable).
    if !used_model {
        let entity_refs = combine_entity_refs(input.extracted_entities, &sender_ref);
        if !entity_refs.is_empty() {
            let rule_obs = rules::extract_observations_rule_based(
                input.content,
                &entity_refs,
                input.timestamp,
            );
            all_obs.extend(rule_obs);
        }
    }
    let extract_ms = extract_start.elapsed().as_millis();

    if all_obs.is_empty() {
        return Ok(ObservationPrepOutcome::Skip(
            "no observations extracted".into(),
        ));
    }

    let entity_refs = combine_entity_refs(input.extracted_entities, &sender_ref);

    Ok(ObservationPrepOutcome::Ready(ObservationPrep {
        all_obs,
        used_model,
        sender_ref,
        entity_refs,
        sentence_ctx_updated,
        sender_ms,
        extract_ms,
    }))
}

/// Writes-only inside the caller's tx. Creates Observation nodes,
/// OBSERVED_IN edges (Obs → Message), and ABOUT edges (Obs → speaker
/// + matching entities).
///
/// Returns the new Observation node ids in input order. Caller owns
/// the commit.
///
/// # Errors
///
/// Returns [`UnikoError::Storage`] on any batched write failure.
pub async fn apply_observations(
    kb: &uniko_store::KnowledgeBase,
    tx: &uni_db::Transaction,
    message_node_id: NodeId,
    prep: ObservationPrep,
) -> Result<Vec<NodeId>, UnikoError> {
    if prep.is_empty() {
        return Ok(Vec::new());
    }
    let ObservationPrep {
        all_obs,
        sender_ref,
        entity_refs,
        ..
    } = prep;

    // 1. Batch create Observation nodes.
    let obs_props: Vec<HashMap<String, Value>> = all_obs
        .iter()
        .map(|raw| {
            let obs_id = uniko_store::id::new_id();
            let mut props = HashMap::new();
            props.insert("observation_id".into(), Value::String(obs_id));
            props.insert("content".into(), Value::String(raw.content.clone()));
            props.insert("subject".into(), Value::String(raw.subject.clone()));
            if let Some(pred) = &raw.predicate {
                props.insert("predicate".into(), Value::String(pred.clone()));
            }
            if let Some(obj) = &raw.object {
                props.insert("object".into(), Value::String(obj.clone()));
            }
            if let Some(phrase) = &raw.temporal_phrase {
                props.insert("temporal_phrase".into(), Value::String(phrase.clone()));
            }
            if let Some(anchor) = raw.temporal_anchor {
                props.insert(
                    "temporal_anchor".into(),
                    uniko_store::types::datetime_value(anchor),
                );
            }
            props.insert(
                "observed_at".into(),
                uniko_store::types::datetime_value(raw.observed_at),
            );
            props.insert("confidence".into(), Value::Float(raw.confidence));
            props
        })
        .collect();
    let obs_create_start = std::time::Instant::now();
    let obs_count = obs_props.len();
    let obs_node_ids = kb
        .batch_create_nodes_in_tx(tx, labels::OBSERVATION, &obs_props)
        .await?;
    let obs_create_ms = obs_create_start.elapsed().as_millis();

    // 2. OBSERVED_IN edges (Observation → Message).
    let observed_start = std::time::Instant::now();
    let observed_edges: Vec<(NodeId, NodeId, HashMap<String, Value>)> = obs_node_ids
        .iter()
        .map(|&nid| (nid, message_node_id, HashMap::new()))
        .collect();
    kb.batch_create_edges_fast_in_tx(
        tx,
        edges::OBSERVED_IN,
        Some(labels::OBSERVATION),
        Some(labels::MESSAGE),
        &observed_edges,
    )
    .await?;
    let observed_ms = observed_start.elapsed().as_millis();

    // 3. ABOUT edges (Observation → speaker + matching entities).
    let about_build_start = std::time::Instant::now();
    let mut about_edges: Vec<(NodeId, NodeId, HashMap<String, Value>)> = Vec::new();
    for (i, raw) in all_obs.iter().enumerate() {
        let obs_nid = obs_node_ids[i];
        if let Some((sender_nid, _)) = &sender_ref {
            about_edges.push((obs_nid, *sender_nid, HashMap::new()));
        }
        for &(entity_nid, ref name) in &entity_refs {
            if sender_ref
                .as_ref()
                .is_some_and(|(nid, _)| *nid == entity_nid)
            {
                continue;
            }
            let subj = raw.subject.to_lowercase();
            let ename = name.to_lowercase();
            if subj == ename || subj.contains(&ename) || ename.contains(&subj) {
                about_edges.push((obs_nid, entity_nid, HashMap::new()));
            }
        }
    }
    let about_build_ms = about_build_start.elapsed().as_millis();
    let about_count = about_edges.len();
    let about_write_start = std::time::Instant::now();
    if !about_edges.is_empty() {
        kb.batch_create_edges_fast_in_tx(
            tx,
            edges::ABOUT,
            Some(labels::OBSERVATION),
            None,
            &about_edges,
        )
        .await?;
    }
    let about_write_ms = about_write_start.elapsed().as_millis();

    tracing::info!(
        target: "apply_obs_breakdown",
        obs_create_ms = obs_create_ms as u64,
        observed_ms = observed_ms as u64,
        about_build_ms = about_build_ms as u64,
        about_write_ms = about_write_ms as u64,
        obs_count = obs_count as u64,
        about_count = about_count as u64,
        "apply_obs breakdown",
    );

    Ok(obs_node_ids)
}

/// Slow-path SENT_BY → Participant lookup. Only used when the caller
/// did not pre-resolve the sender (e.g. legacy ObservationExtractionStep
/// without ctx.sender).
async fn load_sender_ref_by_lookup(
    kb: &uniko_store::KnowledgeBase,
    message_node_id: NodeId,
) -> Option<(NodeId, String)> {
    use uniko_store::storage::edges::Direction;
    let edge_list = kb
        .get_edges(message_node_id, edges::SENT_BY, Direction::Outgoing)
        .await
        .ok()?;
    let edge = edge_list.first()?;
    let (_, props) = kb.get_node(edge.to).await.ok()??;
    let name = props.get("name").and_then(|v| v.as_str())?.to_string();
    Some((edge.to, name))
}

/// Combine sender ref (if any) with extracted entity refs, dedup by name.
fn combine_entity_refs(
    extracted_entities: &[(NodeId, String)],
    sender_ref: &Option<(NodeId, String)>,
) -> Vec<(NodeId, String)> {
    let mut refs: Vec<(NodeId, String)> =
        Vec::with_capacity(extracted_entities.len() + usize::from(sender_ref.is_some()));
    if let Some(sr) = sender_ref {
        refs.push(sr.clone());
    }
    for (nid, name) in extracted_entities {
        if !refs.iter().any(|(_, n)| n == name) {
            refs.push((*nid, name.clone()));
        }
    }
    refs
}

#[cfg(test)]
mod resolve_temporal_in_content_tests {
    use super::resolve_temporal_in_content;
    use chrono::{TimeZone, Utc};

    /// Conversation date 2023-05-08 (a Monday).  Used as the reference
    /// timestamp so "yesterday" resolves to 2023-05-07 and "last
    /// Friday" resolves to 2023-05-05.
    fn ref_date() -> chrono::DateTime<chrono::Utc> {
        Utc.with_ymd_and_hms(2023, 5, 8, 14, 0, 0).unwrap()
    }

    #[test]
    fn yesterday_substituted_with_iso_date() {
        let (content, anchor) = resolve_temporal_in_content(
            "Caroline went to the LGBTQ support group yesterday".into(),
            Some("yesterday"),
            ref_date(),
        );
        assert_eq!(
            content,
            "Caroline went to the LGBTQ support group 2023-05-07"
        );
        assert_eq!(
            anchor.expect("expected resolved anchor for 'yesterday'")
                .format("%Y-%m-%d")
                .to_string(),
            "2023-05-07"
        );
    }

    #[test]
    fn last_friday_substituted_with_iso_date() {
        // 2023-05-08 is Monday; "last Friday" → 2023-05-05.
        let (content, anchor) = resolve_temporal_in_content(
            "Melanie ran a charity race last Friday".into(),
            Some("last Friday"),
            ref_date(),
        );
        assert_eq!(content, "Melanie ran a charity race 2023-05-05");
        assert_eq!(
            anchor.unwrap().format("%Y-%m-%d").to_string(),
            "2023-05-05"
        );
    }

    #[test]
    fn last_week_substituted_with_iso_date() {
        let (content, anchor) = resolve_temporal_in_content(
            "I went camping last week".into(),
            Some("last week"),
            ref_date(),
        );
        // resolve_temporal_with_granularity returns the start of "last
        // week" — the exact day depends on the implementation, but it
        // must NOT be the literal string "last week" and MUST be a
        // 10-character ISO date.
        assert!(
            !content.contains("last week"),
            "expected 'last week' to be substituted, got: {content:?}",
        );
        assert!(
            content.contains("I went camping "),
            "expected leading text preserved, got: {content:?}",
        );
        let date_str = anchor
            .expect("expected resolved anchor for 'last week'")
            .format("%Y-%m-%d")
            .to_string();
        assert!(
            content.ends_with(&date_str),
            "expected content to end with ISO date {date_str}, got: {content:?}",
        );
    }

    #[test]
    fn no_temporal_phrase_passes_content_through() {
        let (content, anchor) = resolve_temporal_in_content(
            "Caroline bought a yellow dress".into(),
            None,
            ref_date(),
        );
        assert_eq!(content, "Caroline bought a yellow dress");
        assert!(anchor.is_none());
    }

    #[test]
    fn unparseable_phrase_leaves_content_untouched() {
        // "next Whitsun" — contains "sun" as a substring; pre-fix the
        // weekday regex matched it as a Sunday relative date.  After
        // the left-`\b` + hyphen-check tightening in
        // `parse_relative_weekday`, this phrase no longer resolves and
        // the content stays untouched.  Guards against the regex
        // regression returning.
        let (content, anchor) = resolve_temporal_in_content(
            "We celebrated next Whitsun".into(),
            Some("next Whitsun"),
            ref_date(),
        );
        assert_eq!(content, "We celebrated next Whitsun");
        assert!(
            anchor.is_none(),
            "unparseable phrase must not yield a resolved anchor"
        );
    }

    #[test]
    fn phrase_appearing_twice_both_substituted() {
        // Generous string replace — semantically what we want: any
        // occurrence of the temporal surface form refers to the same
        // resolved moment within a single observation.
        let (content, _) = resolve_temporal_in_content(
            "I went yesterday and yesterday it was sunny".into(),
            Some("yesterday"),
            ref_date(),
        );
        assert_eq!(
            content,
            "I went 2023-05-07 and 2023-05-07 it was sunny"
        );
    }
}
