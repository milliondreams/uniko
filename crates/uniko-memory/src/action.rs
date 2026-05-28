//! Action recording — the `record_action` agent tool (F17–F20).
//!
//! Actions are the agent's explicit tool calls and operations: shell
//! commands, file edits, API requests, RPCs.  They differ from
//! Episodes (which are subjective learning experiences) in that they
//! carry concrete input/output payloads and link to the artifacts they
//! produce or the messages they were triggered by.
//!
//! When the output exceeds the configured token threshold
//! ([`UnikoConfig::action_output_artifact_threshold`], default 256
//! tokens — F20), the full payload overflows into an Artifact node
//! linked by `PRODUCED`, and the Action stores a short stub instead.
//! This keeps the hot path's payload small while preserving searchable
//! content in the Artifact node's full-text and embedding indexes.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde_json::Value as JsonValue;
use uni_db::Value;

use uniko_extract::embedding::embed_document;
use uniko_extract::ingest::count_tokens;
#[cfg(feature = "onnx")]
use uniko_extract::nlp::NlpPipeline;
#[cfg(feature = "onnx")]
use uniko_extract::nlp::types::NerEntityType;
use uniko_store::id::new_id;
use uniko_store::schema::{edges, labels};
use uniko_store::types::datetime_value;
use uniko_store::{KnowledgeBase, NodeId, UnikoError};

use crate::value_convert::json_to_value;

/// Inputs for [`record_action`].
///
/// Most fields are optional; only `action_type` and the agent id (via
/// the function argument) are required.  Pre-set `action_id` to
/// integrate with external ID spaces (ADR-1).
#[derive(Debug, Clone, Default)]
pub struct RecordActionParams {
    /// Optional pre-assigned id.  When `None`, a fresh UUID v7 is used.
    pub action_id: Option<String>,
    /// Required: kind of action (e.g. `"shell"`, `"file_edit"`,
    /// `"http_request"`).
    pub action_type: String,
    /// Optional structured input payload — what was sent to the tool.
    pub input: Option<JsonValue>,
    /// Optional output payload — what came back.  Large outputs
    /// overflow to an Artifact (F20).
    pub output: Option<JsonValue>,
    /// Outcome status: `"success"`, `"failure"`, `"timeout"`, …
    pub status: Option<String>,
    /// When the action started.  Defaults to `now()`.
    pub started_at: Option<DateTime<Utc>>,
    /// When the action ended.  Defaults to `started_at`.
    pub ended_at: Option<DateTime<Utc>>,
    /// Error message when status is `"failure"`.
    pub error: Option<String>,
    /// Optional `message_id` (external key) of the Message that
    /// triggered this action.  When set, a `TRIGGERED_BY` edge is
    /// created (F19).
    pub triggered_by_message_id: Option<String>,
    /// Optional `session_id` (external key) of the Session this action
    /// belongs to.  When set, an `IN_SESSION` edge is created.
    pub session_id: Option<String>,
    /// Optional `action_id` of the previous action in this chain.
    /// When set, a `NEXT_ACTION` edge is created from the previous to
    /// the new action.
    pub previous_action_id: Option<String>,
}

/// Outcome of a successful [`record_action`] call.
///
/// Returns both the Action node and (when applicable) the Artifact
/// node created from output overflow.  Callers can chain further
/// metadata onto either.
#[derive(Debug, Clone)]
pub struct RecordActionResult {
    /// The new Action node id.
    pub action_node: NodeId,
    /// Externalised Artifact id when output overflowed; `None` otherwise.
    pub overflow_artifact: Option<NodeId>,
}

/// Record an Action node, wire its edges, and embed it.
///
/// Mandatory wiring: `PERFORMED_BY` from the Action to the agent's
/// Participant.  Optional wiring (when the params provide identifiers
/// that resolve to existing nodes): `TRIGGERED_BY` to the Message,
/// `IN_SESSION` to the Session, `NEXT_ACTION` from the previous
/// Action, `PRODUCED` to an overflow Artifact.  Each optional edge is
/// best-effort — when a referenced node is missing, the edge is
/// skipped with a tracing debug rather than failing the call (so
/// callers don't need to pre-check existence under churn).
///
/// The Participant identified by `agent_id` must exist — callers
/// typically create it once at agent bootstrap (same contract as
/// [`crate::record_episode`]).
///
/// # Errors
///
/// - [`UnikoError::Storage`] when the agent is missing or graph
///   operations fail.
/// - [`UnikoError::Embedding`] when the action embedding fails.  The
///   Action node is still created so it can be back-filled later, but
///   the error surfaces to the caller.
pub async fn record_action(
    kb: &KnowledgeBase,
    agent_id: &str,
    params: RecordActionParams,
) -> Result<RecordActionResult, UnikoError> {
    // Resolve the agent up-front.
    let participant_node = kb
        .get_node_by_ext_id(labels::PARTICIPANT, "participant_id", agent_id)
        .await?
        .ok_or_else(|| {
            UnikoError::Storage(format!(
                "record_action: Participant '{agent_id}' not found — create it before recording"
            ))
        })?;
    let participant_id = participant_node.0;

    let action_id = params.action_id.unwrap_or_else(new_id);
    let started_at = params.started_at.unwrap_or_else(Utc::now);
    let ended_at = params.ended_at.unwrap_or(started_at);
    let duration_ms = ended_at
        .signed_duration_since(started_at)
        .num_milliseconds()
        .max(0);

    // ── Output overflow handling (F20) ──
    let threshold = kb.config().action_output_artifact_threshold;
    let (action_output_value, overflow_text) = split_overflow(params.output.as_ref(), threshold);

    let mut props: HashMap<String, Value> = HashMap::new();
    props.insert("action_id".into(), Value::String(action_id.clone()));
    props.insert(
        "action_type".into(),
        Value::String(params.action_type.clone()),
    );
    if let Some(input) = params.input.as_ref() {
        props.insert("input".into(), json_to_value(input));
    }
    if let Some(out) = action_output_value {
        props.insert("output".into(), out);
    }
    if let Some(status) = params.status.as_deref() {
        props.insert("status".into(), Value::String(status.to_string()));
    }
    if let Some(err) = params.error.as_deref() {
        props.insert("error".into(), Value::String(err.to_string()));
    }
    props.insert("started_at".into(), datetime_value(started_at));
    props.insert("ended_at".into(), datetime_value(ended_at));
    props.insert("duration_ms".into(), Value::Int(duration_ms));

    let action_node = kb
        .merge_node(labels::ACTION, "action_id", &action_id, &props)
        .await?;

    // ── Mandatory PERFORMED_BY edge ──
    kb.create_edge(
        edges::PERFORMED_BY,
        action_node,
        participant_id,
        &HashMap::new(),
    )
    .await?;

    // ── Optional TRIGGERED_BY (Action → Message) ──
    if let Some(msg_id) = params.triggered_by_message_id.as_deref() {
        match kb
            .get_node_by_ext_id(labels::MESSAGE, "message_id", msg_id)
            .await?
        {
            Some(msg) => {
                kb.create_edge(edges::TRIGGERED_BY, action_node, msg.0, &HashMap::new())
                    .await?;
            }
            None => {
                tracing::debug!(
                    msg_id,
                    "record_action: triggered_by Message not found — TRIGGERED_BY skipped"
                );
            }
        }
    }

    // ── Optional IN_SESSION (Action → Session) ──
    if let Some(sid) = params.session_id.as_deref() {
        match kb
            .get_node_by_ext_id(labels::SESSION, "session_id", sid)
            .await?
        {
            Some(sess) => {
                kb.create_edge(edges::IN_SESSION, action_node, sess.0, &HashMap::new())
                    .await?;
            }
            None => {
                tracing::debug!(
                    session_id = sid,
                    "record_action: Session not found — IN_SESSION skipped"
                );
            }
        }
    }

    // ── Optional NEXT_ACTION (previous → this) ──
    if let Some(prev_id) = params.previous_action_id.as_deref() {
        match kb
            .get_node_by_ext_id(labels::ACTION, "action_id", prev_id)
            .await?
        {
            Some(prev) => {
                kb.create_edge(edges::NEXT_ACTION, prev.0, action_node, &HashMap::new())
                    .await?;
            }
            None => {
                tracing::debug!(
                    prev_id,
                    "record_action: previous Action not found — NEXT_ACTION skipped"
                );
            }
        }
    }

    // ── Output overflow into an Artifact (F20) ──
    let mut overflow_artifact: Option<NodeId> = None;
    if let Some(payload) = overflow_text {
        let artifact_node = create_overflow_artifact(kb, action_node, &payload, started_at).await?;
        overflow_artifact = Some(artifact_node);
    }

    // ── Embedding ──
    let embed_text = build_embed_text(
        &params.action_type,
        params.input.as_ref(),
        params.output.as_ref(),
    );
    let vec = embed_document(kb, &embed_text).await?;
    let mut emb_props = HashMap::new();
    emb_props.insert("embedding".into(), Value::Vector(vec));
    kb.update_node(action_node, &emb_props).await?;

    // ── NER via Kniv NLP cascade ──
    //
    // Wire MENTIONS edges from the Action to every Entity the NLP
    // model finds in input/output text.  This makes Actions
    // first-class participants in entity-anchored recall — searches
    // for "calls referencing API foo" can now traverse from Foo →
    // MENTIONS → Action, the same way they traverse from Foo →
    // MENTIONS → Message.  Best-effort: when the NLP runtime is
    // unavailable (no ONNX provider) we log and continue.
    #[cfg(feature = "onnx")]
    link_action_entities(kb, action_node, &embed_text).await;

    Ok(RecordActionResult {
        action_node,
        overflow_artifact,
    })
}

/// Run Kniv NER on the Action's text representation and wire
/// `MENTIONS` edges to existing or freshly-merged Entity nodes.
#[cfg(feature = "onnx")]
///
/// We merge by entity name + type so an Action that mentions
/// "Caroline" gets the same Entity node as a Message that mentioned
/// her — that's the entire point of entity-anchored recall.  Edge
/// `count` defaults to 1; consolidation can roll counts up across
/// MENTIONS edges later.
async fn link_action_entities(kb: &KnowledgeBase, action_node: NodeId, text: &str) {
    let Some(pipeline) = NlpPipeline::try_new(kb).await else {
        tracing::debug!("Kniv NLP unavailable; Action MENTIONS edges skipped");
        return;
    };
    let result = match pipeline.analyze(text).await {
        Ok(r) => r,
        Err(e) => {
            tracing::debug!(error = %e, "Kniv analyze failed; Action MENTIONS skipped");
            return;
        }
    };

    for span in result.entities {
        let entity_id = stable_entity_id(&span.text, span.entity_type);
        let mut props = HashMap::new();
        props.insert("name".into(), Value::String(span.text.clone()));
        props.insert(
            "entity_type".into(),
            Value::String(entity_type_str(span.entity_type).to_string()),
        );
        let entity_node = match kb
            .merge_node(labels::ENTITY, "entity_id", &entity_id, &props)
            .await
        {
            Ok(n) => n,
            Err(e) => {
                tracing::debug!(error = %e, entity = %span.text, "Entity merge failed");
                continue;
            }
        };
        let mut edge_props = HashMap::new();
        edge_props.insert("count".into(), Value::Int(1));
        if let Err(e) = kb
            .create_edge(edges::MENTIONS, action_node, entity_node, &edge_props)
            .await
        {
            tracing::debug!(error = %e, "MENTIONS edge failed");
        }
    }
}

/// Stable `entity_id` derived from `(name, entity_type)`.
///
/// Two Actions mentioning the same entity converge on the same
/// Entity node.  Matches the convention used by the P2 ingest path
/// when it deduplicates entities across messages.
///
/// Uses truncated SHA-256 (first 64 bits as big-endian hex) rather than
/// [`std::collections::hash_map::DefaultHasher`] because the latter is not
/// guaranteed stable across rustc versions, and these IDs are persisted to
/// the store as `Entity.entity_id`.
#[cfg(feature = "onnx")]
fn stable_entity_id(name: &str, entity_type: NerEntityType) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(name.to_lowercase().as_bytes());
    h.update(b"\x00");
    h.update(entity_type_str(entity_type).as_bytes());
    let digest = h.finalize();
    let lead = u64::from_be_bytes(digest[..8].try_into().expect("Sha256 yields >= 8 bytes"));
    format!("ent_{lead:016x}")
}

/// Stable string representation of [`NerEntityType`] for storage on
/// the Entity node's `entity_type` property.
#[cfg(feature = "onnx")]
fn entity_type_str(t: NerEntityType) -> &'static str {
    match t {
        NerEntityType::Person => "PERSON",
        NerEntityType::Organization => "ORG",
        NerEntityType::Location => "LOC",
        NerEntityType::Date => "DATE",
        NerEntityType::Numeric => "NUM",
        NerEntityType::Event => "EVENT",
        NerEntityType::Product => "PRODUCT",
        NerEntityType::WorkOfArt => "WORK_OF_ART",
        NerEntityType::Group => "GROUP",
        NerEntityType::Misc => "MISC",
    }
}

/// Decide whether the output payload should be inlined on the Action
/// node or spilled into an Artifact.
///
/// Returns `(action_property, overflow_text)`:
/// - `action_property = Some(value)` when the payload (or a stub
///   pointer) should be stored on the Action.
/// - `overflow_text = Some(text)` when an Artifact must be created.
///
/// Non-string payloads are inlined regardless of token count because
/// they are typically small structured values — only string payloads
/// risk blowing up token budgets.
fn split_overflow(
    output: Option<&JsonValue>,
    threshold_tokens: usize,
) -> (Option<Value>, Option<String>) {
    let Some(out) = output else {
        return (None, None);
    };
    match out {
        JsonValue::String(s) => {
            let tokens = count_tokens(s);
            if tokens > threshold_tokens {
                // Replace the inlined output with a short stub so the
                // Action still records the kind of payload it produced.
                let stub = format!("<overflow: {tokens} tokens spilled to Artifact>");
                (Some(Value::String(stub)), Some(s.clone()))
            } else {
                (Some(Value::String(s.clone())), None)
            }
        }
        other => (Some(json_to_value(other)), None),
    }
}

/// Create an Artifact node for an overflowed Action output and link it
/// via `PRODUCED`.
async fn create_overflow_artifact(
    kb: &KnowledgeBase,
    action_node: NodeId,
    payload: &str,
    timestamp: DateTime<Utc>,
) -> Result<NodeId, UnikoError> {
    let artifact_id = new_id();
    let hash = simple_hash(payload);
    let mut props = HashMap::new();
    props.insert("kind".into(), Value::String("action_output".into()));
    props.insert("content".into(), Value::String(payload.to_string()));
    props.insert("hash".into(), Value::String(hash));
    props.insert("size".into(), Value::Int(payload.len() as i64));
    props.insert("created_at".into(), datetime_value(timestamp));
    let artifact_node = kb
        .merge_node(labels::ARTIFACT, "artifact_id", &artifact_id, &props)
        .await?;

    // PRODUCED: Action → Artifact (F18).
    kb.create_edge(edges::PRODUCED, action_node, artifact_node, &HashMap::new())
        .await?;

    Ok(artifact_node)
}

/// Build the embedding source string for an Action.
///
/// Concatenates action type with summarised input and output text.
/// Long payloads are truncated to keep the embedding call's input
/// size predictable.
fn build_embed_text(
    action_type: &str,
    input: Option<&JsonValue>,
    output: Option<&JsonValue>,
) -> String {
    const MAX_PAYLOAD_CHARS: usize = 400;
    let mut buf = String::with_capacity(64);
    buf.push_str(action_type);
    if let Some(v) = input {
        buf.push_str(" | input: ");
        buf.push_str(&truncate_chars(&json_text(v), MAX_PAYLOAD_CHARS));
    }
    if let Some(v) = output {
        buf.push_str(" | output: ");
        buf.push_str(&truncate_chars(&json_text(v), MAX_PAYLOAD_CHARS));
    }
    buf
}

/// Render a JSON value as a short embedding-friendly string.
fn json_text(v: &JsonValue) -> String {
    match v {
        JsonValue::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn truncate_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max).collect();
    out.push('…');
    out
}

/// Compact 64-bit hash used as a content fingerprint on overflowed
/// Artifacts.  Not cryptographic — collision-resistance is "good
/// enough" to dedupe identical outputs from the same action.
///
/// Uses truncated SHA-256 (first 64 bits as big-endian hex) rather than
/// [`std::collections::hash_map::DefaultHasher`] because the latter is not
/// guaranteed stable across rustc versions, and this fingerprint is
/// persisted to the store as `Artifact.hash`.
fn simple_hash(s: &str) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(s.as_bytes());
    let lead = u64::from_be_bytes(digest[..8].try_into().expect("Sha256 yields >= 8 bytes"));
    format!("{lead:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn split_overflow_inlines_short_string() {
        let (prop, overflow) = split_overflow(Some(&json!("short text")), 256);
        assert!(matches!(prop, Some(Value::String(_))));
        assert!(overflow.is_none());
    }

    #[test]
    fn split_overflow_spills_long_string() {
        let big = "word ".repeat(2_000);
        let (prop, overflow) = split_overflow(Some(&JsonValue::String(big.clone())), 256);
        match prop {
            Some(Value::String(stub)) => {
                assert!(stub.contains("overflow"), "stub was: {stub}");
            }
            other => panic!("expected stub String, got {other:?}"),
        }
        assert_eq!(overflow.as_deref(), Some(big.as_str()));
    }

    #[test]
    fn split_overflow_passes_through_structured_payloads() {
        // Non-string payloads are inlined regardless of size.
        let (prop, overflow) = split_overflow(Some(&json!({"a": [1, 2, 3]})), 4);
        assert!(matches!(prop, Some(Value::Map(_))));
        assert!(overflow.is_none());
    }

    #[test]
    fn truncate_chars_respects_limit() {
        assert_eq!(truncate_chars("abc", 10), "abc");
        let t = truncate_chars(&"x".repeat(20), 5);
        assert!(t.starts_with("xxxxx"));
        assert!(t.ends_with('…'));
    }

    #[test]
    fn build_embed_text_includes_action_type_and_payloads() {
        let txt = build_embed_text(
            "shell",
            Some(&json!("ls -la")),
            Some(&json!("file1\nfile2")),
        );
        assert!(txt.starts_with("shell"));
        assert!(txt.contains("ls -la"));
        assert!(txt.contains("file1"));
    }
}
