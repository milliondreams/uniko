//! Episode recording — the `record_episode` agent tool.
//!
//! Episodes capture the agent's subjective experience: an action it
//! took, the outcome, the state at that moment, and how state changed.
//! They feed P5 procedure promotion, the relevance-decay rule, and
//! Phase 2 of the recall cascade.
//!
//! The spec (Part VI, agent-tools table) classifies `record_episode` as
//! a tool rather than a pipeline because *the agent decides what's
//! worth recording* — it's an explicit subjective act, not something
//! that can be inferred from message content.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde_json::Value as JsonValue;

use uniko_extract::embedding::embed_episode;
use uniko_store::Value;
use uniko_store::id::new_id;
use uniko_store::schema::{edges, labels};
use uniko_store::types::datetime_value;
use uniko_store::{KnowledgeBase, NodeId, UnikoError};

use crate::value_convert::json_to_value;

/// Inputs for [`record_episode`].
///
/// All fields except `action_type` and `outcome` are optional.  Callers
/// that already manage their own ID space may pre-set `episode_id`;
/// otherwise a UUID v7 is generated per ADR-1.
#[derive(Debug, Clone, Default)]
pub struct RecordEpisodeParams {
    /// Optional pre-assigned id.  When `None`, a fresh UUID v7 is used.
    pub episode_id: Option<String>,
    /// Required: what the agent did (e.g. `"retrieve"`, `"build"`).
    pub action_type: String,
    /// Required: how it went (e.g. `"success"`, `"failure"`).
    pub outcome: Option<String>,
    /// Free-form JSON describing the state at the time of action.
    ///
    /// The first non-empty string at one of `topic`, `question`,
    /// `description`, `summary`, or `input` (truncated) becomes the
    /// embedding text — see
    /// [`uniko_extract::embedding::episode_topic_text`].
    pub state: Option<JsonValue>,
    /// Free-form JSON describing the change between pre- and post-
    /// action state.  Stored for diagnostics; not embedded.
    pub delta: Option<JsonValue>,
    /// Subjective importance in `[0.0, 1.0]`.  Drives relevance decay
    /// and Phase 1 score weighting.  Defaults to `0.5` when `None`.
    pub importance: Option<f64>,
    /// Wall-clock time the episode happened.  Defaults to `now()`.
    pub timestamp: Option<DateTime<Utc>>,
    /// Optional `message_id` of the Message that triggered this episode.
    /// When set and resolvable, a `TRIGGERED_BY` edge is created (F15).
    pub triggered_by_message_id: Option<String>,
    /// `action_id`s of the Actions this episode involved.  Each
    /// resolvable one gets an `INVOLVES` edge (F19, best-effort).
    pub involved_action_ids: Vec<String>,
}

/// Window for linking consecutive episodes via `FOLLOWED_BY`.
///
/// When the most recent episode of the same agent is within this many
/// milliseconds, the new episode gets a `FOLLOWED_BY` edge from it with
/// the actual gap on the edge.  One hour matches the spec's session-
/// continuity heuristic.
const FOLLOWED_BY_WINDOW_MS: i64 = 3_600_000;

/// Record an Episode and embed it from its state-topic content.
///
/// Creates an Episode node, wires `RECORDED_BY` → the Participant, and
/// (when applicable) a `FOLLOWED_BY` edge from the agent's prior
/// recent episode.  Computes and stores the topic embedding using
/// [`uniko_extract::embedding::embed_episode`].
///
/// The Participant identified by `agent_id` must already exist —
/// callers typically create it once at agent bootstrap.
///
/// Returns the new Episode's NodeId.
///
/// # Errors
///
/// - [`UnikoError::Storage`] if the agent Participant is missing or
///   the underlying graph operations fail.
/// - [`UnikoError::Embedding`] if topic embedding fails — the Episode
///   node is still created (so it can be back-filled later) but the
///   error is returned to the caller.
pub async fn record_episode(
    kb: &KnowledgeBase,
    agent_id: &str,
    params: RecordEpisodeParams,
) -> Result<NodeId, UnikoError> {
    // Resolve the Participant up-front so we fail fast on bad agent ids.
    let participant_node = kb
        .get_node_by_ext_id(labels::PARTICIPANT, "participant_id", agent_id)
        .await?
        .ok_or_else(|| {
            UnikoError::Storage(format!(
                "record_episode: Participant '{agent_id}' not found — create it before recording"
            ))
        })?;
    let participant_id = participant_node.0;

    let episode_id = params.episode_id.unwrap_or_else(new_id);
    let timestamp = params.timestamp.unwrap_or_else(Utc::now);

    // Convert the serde_json state/delta into uni-db's Value tree.
    let state_value = params.state.as_ref().map(json_to_value);
    let delta_value = params.delta.as_ref().map(json_to_value);

    let mut props: HashMap<String, Value> = HashMap::new();
    props.insert("episode_id".into(), Value::String(episode_id.clone()));
    props.insert(
        "action_type".into(),
        Value::String(params.action_type.clone()),
    );
    if let Some(outcome) = params.outcome.as_deref() {
        props.insert("outcome".into(), Value::String(outcome.to_string()));
    }
    if let Some(ref s) = state_value {
        props.insert("state".into(), s.clone());
    }
    if let Some(ref d) = delta_value {
        props.insert("delta".into(), d.clone());
    }
    props.insert(
        "importance".into(),
        Value::Float(params.importance.unwrap_or(0.5)),
    );
    props.insert("timestamp".into(), datetime_value(timestamp));

    // Look up the previous episode BEFORE creating the new one so the
    // lookup query doesn't have to filter out the in-flight node.
    let earliest = timestamp - chrono::Duration::milliseconds(FOLLOWED_BY_WINDOW_MS);
    let previous = kb
        .previous_episode_in_window(agent_id, earliest, timestamp)
        .await?;

    let episode_node = kb
        .merge_node(labels::EPISODE, "episode_id", &episode_id, &props)
        .await?;

    // RECORDED_BY edge from Episode to Participant.
    kb.create_edge(
        edges::RECORDED_BY,
        episode_node,
        participant_id,
        &HashMap::new(),
    )
    .await?;

    // Optional FOLLOWED_BY chain from the agent's most recent episode.
    if let Some((prev_id, prev_ts)) = previous {
        let gap_ms = timestamp
            .signed_duration_since(prev_ts)
            .num_milliseconds()
            .max(0);
        let mut edge_props = HashMap::new();
        edge_props.insert("gap_ms".into(), Value::Int(gap_ms));
        kb.create_edge(edges::FOLLOWED_BY, prev_id, episode_node, &edge_props)
            .await?;
    }

    // ── Optional TRIGGERED_BY (Episode → Message) ──
    if let Some(msg_id) = params.triggered_by_message_id.as_deref() {
        match kb
            .get_node_by_ext_id(labels::MESSAGE, "message_id", msg_id)
            .await?
        {
            Some(msg) => {
                kb.create_edge(edges::TRIGGERED_BY, episode_node, msg.0, &HashMap::new())
                    .await?;
            }
            None => {
                tracing::debug!(
                    msg_id,
                    "record_episode: triggered_by Message not found — TRIGGERED_BY skipped"
                );
            }
        }
    }

    // ── Optional INVOLVES (Episode → Action), best-effort ──
    for action_id in &params.involved_action_ids {
        match kb
            .get_node_by_ext_id(labels::ACTION, "action_id", action_id)
            .await?
        {
            Some(action) => {
                kb.create_edge(edges::INVOLVES, episode_node, action.0, &HashMap::new())
                    .await?;
            }
            None => {
                tracing::debug!(
                    action_id,
                    "record_episode: involved Action not found — INVOLVES skipped"
                );
            }
        }
    }

    // Compute and store the topic embedding.
    embed_episode(
        kb,
        episode_node,
        state_value.as_ref(),
        &params.action_type,
        params.outcome.as_deref(),
    )
    .await?;

    Ok(episode_node)
}
