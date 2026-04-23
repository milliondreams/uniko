//! Session-level ingestion context.
//!
//! [`SessionContext`] persists across messages within a session,
//! tracking participants, message chain, and the pronoun resolution
//! window ([`SentenceContext`]).
//!
//! Created once per session by the caller (bench or production engine),
//! passed to `ingest_message()` and pipeline steps via metadata.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use uniko_store::NodeId;

/// Session-level state that persists across messages.
///
/// Owns the sentence context window for pronoun resolution,
/// participant cache, and message chain tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionContext {
    /// External session identifier.
    pub session_id: String,
    /// Internal node ID of the Session node.
    pub session_nid: NodeId,
    /// Participant name → node ID cache (avoids re-querying).
    pub participants: HashMap<String, NodeId>,
    /// Current message speaker.
    pub current_speaker: String,
    /// Other participants in the session.
    pub other_speakers: Vec<String>,
    /// Previous message node ID for NEXT edge linking.
    pub prev_message_nid: Option<NodeId>,
    /// Pronoun resolution context window.
    pub sentence_ctx: SentenceContext,
}

impl SessionContext {
    /// Create a new session context.
    pub fn new(session_id: String, session_nid: NodeId) -> Self {
        Self {
            session_id,
            session_nid,
            participants: HashMap::new(),
            current_speaker: String::new(),
            other_speakers: Vec::new(),
            prev_message_nid: None,
            sentence_ctx: SentenceContext::default(),
        }
    }

    /// Set the current speaker and update other_speakers list.
    pub fn set_current_speaker(&mut self, speaker: &str) {
        self.current_speaker = speaker.to_string();
        self.sentence_ctx.speaker = speaker.to_string();

        // Rebuild other_speakers from participants excluding current.
        self.other_speakers = self
            .participants
            .keys()
            .filter(|name| name.as_str() != speaker)
            .cloned()
            .collect();
        self.sentence_ctx.other_speakers = self.other_speakers.clone();
    }

    /// Register a participant (caches name → nid).
    pub fn register_participant(&mut self, name: &str, nid: NodeId) {
        self.participants.insert(name.to_string(), nid);
    }

    /// Get a participant node ID by name.
    pub fn participant_nid(&self, name: &str) -> Option<NodeId> {
        self.participants.get(name).copied()
    }
}

/// Pronoun resolution context window.
///
/// Tracks the most recent noun phrases (subject and object) across
/// sentences within a session. Used to resolve pronouns like "it",
/// "this", "that", "you" to their antecedents.
///
/// Updated after each sentence by [`update_sentence_context`](crate::nlp::decode::update_sentence_context).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SentenceContext {
    /// Current speaker name (for first-person resolution).
    pub speaker: String,
    /// Other participants (for second-person resolution).
    pub other_speakers: Vec<String>,
    /// Last NOUN/PROPN subject seen (for "it"/"this"/"that" resolution).
    pub last_noun_subject: Option<String>,
    /// Last NOUN/PROPN object seen (for "it"/"this"/"that" resolution).
    pub last_noun_object: Option<String>,
}

impl SentenceContext {
    /// Create a context with speaker info.
    pub fn new(speaker: &str, other_speakers: Vec<String>) -> Self {
        Self {
            speaker: speaker.to_string(),
            other_speakers,
            last_noun_subject: None,
            last_noun_object: None,
        }
    }
}
