//! Summary generation — the F59 session-summary capability.
//!
//! A Summary is a compact, embedded synopsis of a target (here: a
//! Session) that Phase-3 recall can fall back to when finer-grained
//! Observations/Facts under-cover a query.  Generation is **extractive
//! and deterministic by default** — it selects and concatenates the
//! session's strongest claims — so it works offline with no model.  An
//! optional abstractive path behind the `llm` feature rewrites that
//! material into prose.
//!
//! Summaries are idempotent on a stable `summary_id` derived from the
//! target, so re-summarising a session updates its Summary node in place
//! rather than accumulating duplicates.

use std::collections::HashMap;

use chrono::{DateTime, Utc};

use uniko_extract::embedding::embed_document;
use uniko_store::schema::{edges, labels};
use uniko_store::types::datetime_value;
use uniko_store::{KnowledgeBase, NodeId, UnikoError, Value};

/// Maximum number of claims pulled into an extractive session summary.
const MAX_CLAIMS: usize = 40;

/// Soft character cap on the assembled summary text.
const MAX_SUMMARY_CHARS: usize = 2_000;

/// Generate (or refresh) the Summary for a Session (F59).
///
/// Pulls the session's Observations (claim form) — falling back to the
/// raw transcript when none exist — selects a bounded set, and writes a
/// `Summary` node embedded and wired `SUMMARIZES` → the Session.  When
/// `llm_alias` is `Some` and the `llm` feature is built, the extractive
/// material is rewritten abstractively; otherwise the extractive text is
/// stored verbatim.
///
/// Returns the Summary's NodeId, or `None` when the session has no
/// summarisable content.
///
/// # Errors
///
/// - [`UnikoError::Storage`] when the session lookup or graph writes
///   fail.
/// - [`UnikoError::Embedding`] when embedding the summary text fails.
pub async fn generate_session_summary(
    kb: &KnowledgeBase,
    session_id: &str,
    now: DateTime<Utc>,
    llm_alias: Option<&str>,
) -> Result<Option<NodeId>, UnikoError> {
    // Resolve the Session node first; without it there is nothing to
    // wire SUMMARIZES to.
    let Some(session) = kb
        .get_node_by_ext_id(labels::SESSION, "session_id", session_id)
        .await?
    else {
        tracing::debug!(
            session_id,
            "generate_session_summary: Session not found — skipped"
        );
        return Ok(None);
    };
    let session_node = session.0;

    let extractive = build_extractive_text(kb, session_id).await?;
    let Some(extractive) = extractive else {
        return Ok(None);
    };

    let text = match llm_alias {
        Some(alias) => abstractive_or_extractive(kb, alias, &extractive).await,
        None => extractive,
    };

    // Stable id so re-summarising updates in place.
    let summary_id = format!("summary-session-{session_id}");
    let mut props: HashMap<String, Value> = HashMap::new();
    props.insert("summary_id".into(), Value::String(summary_id.clone()));
    props.insert("text".into(), Value::String(text.clone()));
    props.insert("level".into(), Value::String("session".into()));
    props.insert("generated_at".into(), datetime_value(now));

    let summary_node = kb
        .merge_node(labels::SUMMARY, "summary_id", &summary_id, &props)
        .await?;

    // SUMMARIZES: Summary → Session.
    kb.create_edge(
        edges::SUMMARIZES,
        summary_node,
        session_node,
        &HashMap::new(),
    )
    .await?;

    // Embed the summary text so it is a Phase-3 retrieval target.
    let vec = embed_document(kb, &text).await?;
    let mut emb_props = HashMap::new();
    emb_props.insert("embedding".into(), Value::Vector(vec));
    kb.update_node(summary_node, &emb_props).await?;

    Ok(Some(summary_node))
}

/// Assemble the deterministic extractive source text for a session.
///
/// Prefers Observations (already in claim form, closest to what a
/// summary should capture); falls back to the raw transcript when the
/// session produced no observations.  Returns `None` when neither
/// exists.
async fn build_extractive_text(
    kb: &KnowledgeBase,
    session_id: &str,
) -> Result<Option<String>, UnikoError> {
    let observations = kb.session_observation_rows(session_id).await?;
    if !observations.is_empty() {
        let mut seen = std::collections::HashSet::new();
        let mut claims: Vec<&str> = Vec::new();
        for row in &observations {
            let claim = row.content.trim();
            if claim.is_empty() || !seen.insert(claim) {
                continue;
            }
            claims.push(claim);
            if claims.len() >= MAX_CLAIMS {
                break;
            }
        }
        if !claims.is_empty() {
            return Ok(Some(cap_chars(&claims.join("\n"), MAX_SUMMARY_CHARS)));
        }
    }

    // Fallback: the raw transcript, oldest first.
    let transcript = kb.session_transcript_rows(session_id).await?;
    if transcript.is_empty() {
        return Ok(None);
    }
    let joined = transcript
        .iter()
        .map(|r| format!("{}: {}", r.speaker, r.content.trim()))
        .collect::<Vec<_>>()
        .join("\n");
    Ok(Some(cap_chars(&joined, MAX_SUMMARY_CHARS)))
}

/// Truncate `s` to at most `max` characters on a char boundary,
/// appending an ellipsis when truncated.
fn cap_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max).collect();
    out.push('…');
    out
}

/// Rewrite extractive material into prose via the LLM, or return the
/// extractive text unchanged when the `llm` feature is absent or the
/// call fails.
#[cfg(feature = "llm")]
async fn abstractive_or_extractive(kb: &KnowledgeBase, alias: &str, extractive: &str) -> String {
    use uniko_store::xervo::{GenerationOptions, Message};

    let prompt = format!(
        "Summarise the following conversation notes into a concise paragraph. \
         Keep concrete facts; omit filler.\n\n{extractive}"
    );
    let messages = vec![Message::user(&prompt)];
    let options = GenerationOptions {
        max_tokens: Some(256),
        ..Default::default()
    };
    match kb.generate(alias, &messages, options).await {
        Ok(text) if !text.trim().is_empty() => text.trim().to_string(),
        Ok(_) => extractive.to_string(),
        Err(e) => {
            tracing::debug!(error = %e, "summary LLM call failed — using extractive text");
            extractive.to_string()
        }
    }
}

/// Extractive-only build: the `llm` feature is not enabled, so the
/// abstractive path is unavailable and the extractive text is returned
/// verbatim.
#[cfg(not(feature = "llm"))]
#[expect(
    clippy::unused_async,
    reason = "signature parity with the llm-enabled variant"
)]
async fn abstractive_or_extractive(_kb: &KnowledgeBase, _alias: &str, extractive: &str) -> String {
    extractive.to_string()
}
