//! Prep-then-commit atomic per-message ingest.
//!
//! Folds the three legacy per-message steps (`ingest_message`,
//! `EntityExtractionStep`, `ObservationExtractionStep`) into one
//! `prep then single-tx` flow. All CPU work (NLP, NER, SRL,
//! dedup) and read-only DB lookups happen BEFORE the tx opens;
//! a single transaction then writes Message + edges + chunks +
//! Entities + MENTIONS + Observations + OBSERVED_IN + ABOUT and
//! commits once.
//!
//! At sess=24 in our LME benchmark this cuts commits per message
//! from 3 → 1 (plus 0-2 first-sight Session/Participant commits)
//! and removes the bulk of the "commit-wait" time we measured.
//!
//! Failure semantics are **all-or-nothing**: any error during
//! extraction or in-tx writes aborts the call; the tx is dropped
//! without commit. Today's "Message persisted with no entities"
//! half-state from the legacy 3-step path cannot occur here.
//!
//! Pre-tx `ensure_session_and_sender` still uses its own commits
//! (one per first-sight); folding those is a follow-up.

use uniko_pipes::types::IngestMessage;
use uniko_store::{KnowledgeBase, NodeId};

use super::context::SessionContext;
use super::message::{
    MessageSetup, apply_message_writes_in_tx, ensure_session_and_sender, resolve_recipients,
};
use crate::ner::dedup::{
    EntityUpsertPrep, apply_entity_upsert, deduplicate_raw, prepare_entity_upsert,
};
use crate::ner::types::RawEntity;
use crate::observations::{
    ObservationInputs, ObservationPrepOutcome, apply_observations, prepare_observations,
};

/// Result of an atomic ingest: all the ids the caller might want.
///
/// On idempotency re-ingest, `extracted_entities`,
/// `extracted_observations`, and `chunk_node_ids` will be empty and
/// `sender` will be `None`. The caller should treat that the same as a
/// legacy `MessageIngestResult` with `sender: None`.
#[derive(Debug)]
pub struct AtomicIngestResult {
    pub message_node_id: NodeId,
    pub chunk_node_ids: Vec<NodeId>,
    pub session_node_id: NodeId,
    pub sender: Option<(NodeId, String)>,
    pub extracted_entities: Vec<(NodeId, String)>,
    pub extracted_observations: Vec<NodeId>,
    /// Internal per-phase timings, ms.
    pub timings: AtomicTimings,
}

/// Per-phase timings for one atomic ingest call.
#[derive(Debug, Default, Clone, Copy)]
pub struct AtomicTimings {
    pub setup_ms: u128,
    pub nlp_ms: u128,
    pub extract_ms: u128,
    pub prep_read_ms: u128,
    pub create_ms: u128,
    pub edges_ms: u128,
    pub chunk_ms: u128,
    pub apply_entity_ms: u128,
    pub apply_obs_ms: u128,
    pub commit_ms: u128,
    pub total_ms: u128,
}

/// Atomic per-message ingest: extract → single tx → commit.
///
/// Mirrors the legacy
/// `ingest_message + EntityExtractionStep + ObservationExtractionStep`
/// chain but with one transaction and all-or-nothing failure.
///
/// # Errors
///
/// Returns [`uniko_store::UnikoError`] on any underlying read, write,
/// extraction, or commit failure. On error the tx (if opened) is
/// dropped — no partial state persists for this message.
pub async fn ingest_message_atomic(
    kb: &KnowledgeBase,
    msg: &IngestMessage,
    session_ctx: &mut SessionContext,
) -> uniko_store::Result<AtomicIngestResult> {
    let total_start = std::time::Instant::now();

    // 1. Idempotency.
    if let Some((existing_id, _)) = kb
        .get_node_by_ext_id("Message", "message_id", &msg.message_id)
        .await?
    {
        return Ok(AtomicIngestResult {
            message_node_id: existing_id,
            chunk_node_ids: Vec::new(),
            session_node_id: session_ctx.session_nid,
            sender: None,
            extracted_entities: Vec::new(),
            extracted_observations: Vec::new(),
            timings: AtomicTimings {
                total_ms: total_start.elapsed().as_millis(),
                ..AtomicTimings::default()
            },
        });
    }

    // 2. Session + sender (own commits; first-sight only).
    let setup_start = std::time::Instant::now();
    let (session_nid, participant_nid) = ensure_session_and_sender(kb, msg, session_ctx).await?;
    let recipient_nids = resolve_recipients(kb, msg, participant_nid, session_ctx).await?;
    let setup_ms = setup_start.elapsed().as_millis();

    // 3. CPU extraction (NER + NLP + observations) — no DB I/O.
    let extract_start = std::time::Instant::now();
    let ext = extract_entities_and_nlp(kb, msg).await;
    let deduped = ext.deduped;
    let nlp_ms = ext.nlp_ms;
    #[cfg(feature = "onnx")]
    let nlp_results = ext.nlp_results;
    let extract_ms = extract_start.elapsed().as_millis();

    // 4. Compute canonical entity ids for this batch.
    let prep_start = std::time::Instant::now();
    let entity_prep: EntityUpsertPrep = prepare_entity_upsert(kb, deduped).await?;
    let prep_read_ms = prep_start.elapsed().as_millis();

    // 5. Acquire the per-entity RMW locks BEFORE opening the tx, and hold
    //    them across the commit (step 9). entity_id is non-unique in uni-db,
    //    so without this two concurrent ingests of the same entity both read
    //    "absent" and both CREATE a duplicate row. Locking before tx-open
    //    means our snapshot (and the authoritative re-read in
    //    `apply_entity_upsert`) reflects any entity a prior holder committed.
    //    Guards drop at function exit, after `tx.commit()`.
    let _entity_guards = kb.lock_entity_ids(&entity_prep.entity_ids).await;

    // 6. Open tx (snapshot taken after the entity locks are held).
    let session = kb.db().session();
    let tx = session
        .tx()
        .await
        .map_err(|e| uniko_store::UnikoError::Storage(e.to_string()))?;

    // 6. apply Message writes.
    let setup = MessageSetup {
        session_nid,
        participant_nid,
        recipient_nids,
        prev_msg_nid: session_ctx.prev_message_nid,
    };
    let writes = apply_message_writes_in_tx(kb, &tx, msg, &setup).await?;
    let message_nid = writes.message_node_id;

    // 7. apply entity upsert (Entity + MENTIONS).
    let apply_entity_start = std::time::Instant::now();
    let entity_matches = apply_entity_upsert(kb, &tx, message_nid, entity_prep).await?;
    let apply_entity_ms = apply_entity_start.elapsed().as_millis();

    // 8. apply observations (Observation + OBSERVED_IN + ABOUT).
    let extracted_entities: Vec<(NodeId, String)> = entity_matches
        .iter()
        .map(|m| (m.node_id, m.canonical_name.clone()))
        .collect();
    let sender_ref = Some((participant_nid, msg.sender_id.clone()));

    let apply_obs_start = std::time::Instant::now();
    let timestamp = msg.timestamp;
    let rules_path = kb.config().observation_rules_path.clone();
    let inputs = ObservationInputs {
        kb,
        message_node_id: message_nid,
        content: &msg.content,
        content_type: &msg.content_type,
        sender: sender_ref.clone(),
        extracted_entities: &extracted_entities,
        #[cfg(feature = "onnx")]
        nlp_results: nlp_results.as_deref(),
        seed_sentence_ctx: Some(&session_ctx.sentence_ctx),
        timestamp,
        observation_rules_path: rules_path.as_deref(),
    };
    let obs_outcome = prepare_observations(inputs).await?;
    let (extracted_observations, sentence_ctx_updated) = match obs_outcome {
        ObservationPrepOutcome::Skip(_) => (Vec::new(), None),
        ObservationPrepOutcome::Ready(prep) => {
            let sc_updated = prep.sentence_ctx_updated.clone();
            let obs_nids = apply_observations(kb, &tx, message_nid, prep).await?;
            (obs_nids, sc_updated)
        }
    };
    let apply_obs_ms = apply_obs_start.elapsed().as_millis();

    // 9. Commit.
    let commit_start = std::time::Instant::now();
    tx.commit()
        .await
        .map_err(|e| uniko_store::UnikoError::Storage(e.to_string()))?;
    let commit_ms = commit_start.elapsed().as_millis();

    // 10. Post-commit: update session_ctx.
    session_ctx.prev_message_nid = Some(message_nid);
    if let Some(sc) = sentence_ctx_updated {
        session_ctx.sentence_ctx = sc;
    }

    let timings = AtomicTimings {
        setup_ms,
        nlp_ms,
        extract_ms,
        prep_read_ms,
        create_ms: writes.create_ms,
        edges_ms: writes.edges_ms,
        chunk_ms: writes.chunk_ms,
        apply_entity_ms,
        apply_obs_ms,
        commit_ms,
        total_ms: total_start.elapsed().as_millis(),
    };
    tracing::info!(
        message_id = %msg.message_id,
        setup_ms = timings.setup_ms as u64,
        nlp_ms = timings.nlp_ms as u64,
        extract_ms = timings.extract_ms as u64,
        prep_read_ms = timings.prep_read_ms as u64,
        create_ms = timings.create_ms as u64,
        edges_ms = timings.edges_ms as u64,
        chunk_ms = timings.chunk_ms as u64,
        apply_entity_ms = timings.apply_entity_ms as u64,
        apply_obs_ms = timings.apply_obs_ms as u64,
        commit_ms = timings.commit_ms as u64,
        total_ms = timings.total_ms as u64,
        entities = extracted_entities.len(),
        observations = extracted_observations.len(),
        "atomic ingest",
    );
    tracing::info!(
        target: "tx_perf",
        tx_phase = "ingest_message_atomic",
        total_ms = timings.total_ms as u64,
        commit_ms = timings.commit_ms as u64,
        "tx phase",
    );

    Ok(AtomicIngestResult {
        message_node_id: message_nid,
        chunk_node_ids: writes.chunk_node_ids,
        session_node_id: session_nid,
        sender: sender_ref,
        extracted_entities,
        extracted_observations,
        timings,
    })
}

/// Output of [`extract_entities_and_nlp`]. `nlp_results` is only
/// populated when the ONNX feature is enabled.
struct EntityExtractionOutput {
    deduped: Vec<(RawEntity, u32)>,
    #[cfg(feature = "onnx")]
    nlp_results: Option<Vec<crate::nlp::types::NlpResult>>,
    nlp_ms: u128,
}

/// Run NER (rule-based + code AST + ONNX cascade + LLM stub) and
/// dedup. Returns the deduped entity batch plus the per-sentence NLP
/// results needed by the observation step.
async fn extract_entities_and_nlp(
    #[cfg_attr(not(feature = "onnx"), allow(unused_variables))] kb: &KnowledgeBase,
    msg: &IngestMessage,
) -> EntityExtractionOutput {
    let mut all_raw: Vec<RawEntity> = Vec::new();

    // 1. Rule-based (always).
    all_raw.extend(crate::ner::rules::extract_entities_rule_based(&msg.content));

    // 2. Code AST (feature-gated).
    #[cfg(feature = "code-parse")]
    if msg.content_type == "code" {
        let lang = msg
            .metadata
            .get("language")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let code_ents = crate::ner::code::extract_code_entities(&msg.content, lang);
        all_raw.extend(code_ents);
    }

    // 3. ONNX NER + per-sentence NLP cascade.
    // On `onnx` builds the cascade block yields `(nlp_results, nlp_ms)`;
    // without the feature `nlp_ms` stays 0.
    #[cfg(not(feature = "onnx"))]
    let nlp_ms: u128 = 0;
    #[cfg(feature = "onnx")]
    let (nlp_results, nlp_ms) = {
        let nlp_start = std::time::Instant::now();
        let result = match crate::nlp::NlpPipeline::try_new(kb).await {
            Some(pipeline) => match pipeline.analyze_sentences(&msg.content).await {
                Ok(results) => {
                    for nlp_result in &results {
                        let onnx_ents =
                            crate::ner::onnx::entities_from_nlp_result(nlp_result, &msg.content);
                        all_raw.extend(onnx_ents);
                    }
                    Some(results)
                }
                Err(e) => {
                    tracing::debug!(error = %e, "NLP analysis failed");
                    None
                }
            },
            None => {
                tracing::debug!("NLP pipeline unavailable, using rules only");
                None
            }
        };
        (result, nlp_start.elapsed().as_millis())
    };
    #[cfg(not(feature = "onnx"))]
    {
        match crate::ner::onnx::extract_entities_onnx(&msg.content) {
            Ok(onnx_ents) => all_raw.extend(onnx_ents),
            Err(e) => tracing::debug!(error = %e, "ONNX NER unavailable"),
        }
    }

    // 4. Dedup.
    let deduped = deduplicate_raw(all_raw);

    EntityExtractionOutput {
        deduped,
        #[cfg(feature = "onnx")]
        nlp_results,
        nlp_ms,
    }
}
