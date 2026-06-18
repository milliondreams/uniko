//! The [`Session`] conversation handle and its [`Turn`] input.

use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

use chrono::{DateTime, Utc};

use uniko_extract::ingest::artifact::ingest_artifact;
use uniko_extract::ingest::atomic::ingest_message_atomic;
use uniko_extract::ingest::context::SessionContext;
use uniko_extract::ingest::pdf::{PdfIngestOptions, PdfInput, ingest_pdf};
use uniko_extract::ingest::{ArtifactIngestResult, AtomicIngestResult, PdfIngestResult};
use uniko_pipes::IngestMessage;
use uniko_pipes::types::{IngestArtifact, IngestTask};
use uniko_store::{KnowledgeBase, NodeId, UnikoError};

use crate::pipeline::PipelineSystem;
use crate::summary::generate_session_summary;

/// A conversation scope that feeds turns into memory.
///
/// Obtain one with [`Agent::session`](crate::Agent::session). A `Session`
/// owns the per-session [`SessionContext`] (turn chain, speaker window,
/// participant cache), so feeding turns through it preserves cross-turn
/// conversational state. A `Session` is single-threaded: feed its turns
/// in order.
///
/// Use [`observe`](Session::observe) for durable, immediately-recallable
/// ingest; [`submit`](Session::submit) + [`flush`](Session::flush) for
/// streaming throughput when the instance was built with
/// [`streaming(true)`](crate::UnikoBuilder::streaming). Use one path or
/// the other per session, not both.
pub struct Session {
    kb: KnowledgeBase,
    ctx: SessionContext,
    streaming: Option<Arc<PipelineSystem>>,
    llm_alias: Option<String>,
}

impl fmt::Debug for Session {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // KnowledgeBase is not Debug; surface the session identity only.
        f.debug_struct("Session")
            .field("session_id", &self.ctx.session_id)
            .field("streaming", &self.streaming.is_some())
            .finish_non_exhaustive()
    }
}

impl Session {
    /// Create a session bound to `session_id`.
    ///
    /// The Session node is created lazily on the first
    /// [`observe`](Session::observe), so this performs no I/O.
    pub(crate) fn new(
        kb: KnowledgeBase,
        session_id: impl Into<String>,
        streaming: Option<Arc<PipelineSystem>>,
        llm_alias: Option<String>,
    ) -> Self {
        // `session_nid = 0` is the sentinel the ingest path resolves /
        // creates on first sight (see `ensure_session_and_sender`).
        let ctx = SessionContext::new(session_id.into(), 0);
        Self {
            kb,
            ctx,
            streaming,
            llm_alias,
        }
    }

    /// The session's external identifier.
    pub fn session_id(&self) -> &str {
        &self.ctx.session_id
    }

    /// Ingest one turn durably, committing before returning.
    ///
    /// The write is immediately visible to
    /// [`Agent::recall`](crate::Agent::recall) (read-after-write). Runs the
    /// full per-message pipeline — chunking, entity extraction, observation
    /// extraction — in a single transaction.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] on any extraction or write failure; on error
    /// no partial state persists for the turn.
    pub async fn observe(&mut self, turn: Turn) -> Result<AtomicIngestResult, UnikoError> {
        // Update speaker / pronoun window before ingest so recipient
        // inference and pronoun resolution see the right context.
        self.ctx.set_current_speaker(&turn.sender_id);
        let session_id = self.ctx.session_id.clone();
        let msg = turn.into_ingest_message(session_id);
        ingest_message_atomic(&self.kb, &msg, &mut self.ctx).await
    }

    /// Enqueue one turn for asynchronous streaming ingest.
    ///
    /// Returns once the task is accepted by the pipeline (fire-and-forget).
    /// Streamed turns are processed independently and do **not** advance
    /// this session's cross-turn context — use [`observe`](Session::observe)
    /// for conversational fidelity. Await [`flush`](Session::flush) before a
    /// recall that must see streamed turns.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Config`] when the instance was not built with
    /// [`streaming(true)`](crate::UnikoBuilder::streaming), or
    /// [`UnikoError::Pipeline`] when the ingest queue is full.
    pub async fn submit(&self, turn: Turn) -> Result<(), UnikoError> {
        let pipeline = self.require_streaming("submit")?;
        let session_id = self.ctx.session_id.clone();
        let msg = turn.into_ingest_message(session_id);
        pipeline.submit_ingest(IngestTask::Message(msg))
    }

    /// Await full processing of everything [`submit`](Session::submit)ted.
    ///
    /// A true barrier: returns only once the ingest queue is drained and no
    /// in-flight task remains, so a following recall sees all streamed
    /// turns.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Config`] when streaming is not enabled.
    pub async fn flush(&self) -> Result<(), UnikoError> {
        let pipeline = self.require_streaming("flush")?;
        pipeline.quiesce().await;
        Ok(())
    }

    /// Ingest a text document, attaching it to this session.
    ///
    /// Deduplicated by content hash: re-ingesting identical content
    /// returns the existing artifact without re-chunking.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] on any chunking or write failure.
    pub async fn ingest_document(
        &self,
        document: Document,
    ) -> Result<ArtifactIngestResult, UnikoError> {
        let artifact = document.into_ingest_artifact(self.ctx.session_id.clone());
        ingest_artifact(&self.kb, &artifact).await
    }

    /// Ingest a PDF, extracting its text into chunks.
    ///
    /// Uses native (pure-Rust) text extraction; OCR for scanned PDFs
    /// requires the `pdf-ocr` build. Note: the PDF is **not** linked to
    /// this session (the underlying ingest carries no session field) —
    /// it is ingested as a standalone artifact.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] on a read or write failure. Text-extraction
    /// failures are reported in [`PdfIngestResult::extraction_failure`]
    /// rather than erroring.
    pub async fn ingest_pdf(
        &self,
        artifact_id: impl Into<String>,
        source: PdfSource,
    ) -> Result<PdfIngestResult, UnikoError> {
        let options = PdfIngestOptions {
            artifact_id: artifact_id.into(),
            extractor: None,
            source_path: source.source_path(),
        };
        ingest_pdf(&self.kb, source.into_pdf_input(), options).await
    }

    /// Generate (or refresh) a synopsis of this session.
    ///
    /// Extractive (deterministic) when no LLM was configured on the
    /// instance, abstractive (LLM-rewritten) when one was. Idempotent on
    /// the session's summary id. Returns the summary node id, or `None`
    /// when the session has no content to summarize.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] on a read, write, or generation failure.
    pub async fn summarize(&self) -> Result<Option<NodeId>, UnikoError> {
        generate_session_summary(
            &self.kb,
            &self.ctx.session_id,
            Utc::now(),
            self.llm_alias.as_deref(),
        )
        .await
    }

    /// Borrow the streaming pipeline or explain that it is disabled.
    fn require_streaming(&self, method: &str) -> Result<&Arc<PipelineSystem>, UnikoError> {
        self.streaming.as_ref().ok_or_else(|| {
            UnikoError::Config(format!(
                "{method}() requires streaming; build with Uniko::builder().streaming(true)"
            ))
        })
    }
}

/// One conversation turn to feed into a [`Session`].
///
/// Construct with [`Turn::new`] (sender + content) and refine with the
/// chainable setters. Maps to a single message ingest; a UUID v7 message
/// id is generated automatically.
///
/// # Examples
///
/// ```
/// use uniko_memory::Turn;
///
/// let turn = Turn::new("alice", "the deploy finished at noon")
///     .content_type("text")
///     .addressed_to(vec!["bob".to_string()]);
/// # let _ = turn;
/// ```
#[derive(Debug, Clone)]
pub struct Turn {
    message_id: Option<String>,
    sender_id: String,
    content: String,
    content_type: String,
    addressed_to: Option<Vec<String>>,
    timestamp: DateTime<Utc>,
    metadata: HashMap<String, serde_json::Value>,
}

impl Turn {
    /// A text turn from `sender_id` carrying `content`, stamped now.
    pub fn new(sender_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            message_id: None,
            sender_id: sender_id.into(),
            content: content.into(),
            content_type: "text".to_string(),
            addressed_to: None,
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        }
    }

    /// Set an explicit message id for idempotent ingest.
    ///
    /// Ingest is idempotent on `message_id`: re-feeding a turn with the
    /// same id is a no-op rather than a duplicate. When unset, a fresh
    /// UUID v7 is generated per turn.
    #[must_use]
    pub fn id(mut self, message_id: impl Into<String>) -> Self {
        self.message_id = Some(message_id.into());
        self
    }

    /// Override the content type (defaults to `"text"`).
    #[must_use]
    pub fn content_type(mut self, content_type: impl Into<String>) -> Self {
        self.content_type = content_type.into();
        self
    }

    /// Set the send time (defaults to now).
    #[must_use]
    pub fn at(mut self, timestamp: DateTime<Utc>) -> Self {
        self.timestamp = timestamp;
        self
    }

    /// Set explicit recipient participant ids.
    ///
    /// When unset, recipients are inferred from session participants.
    #[must_use]
    pub fn addressed_to(mut self, recipients: Vec<String>) -> Self {
        self.addressed_to = Some(recipients);
        self
    }

    /// Attach one arbitrary metadata entry forwarded to ingest.
    #[must_use]
    pub fn metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Lower into the wire ingest message for `session_id`.
    fn into_ingest_message(self, session_id: String) -> IngestMessage {
        IngestMessage {
            message_id: self.message_id.unwrap_or_else(uniko_store::id::new_id),
            content: self.content,
            content_type: self.content_type,
            sender_id: self.sender_id,
            session_id,
            addressed_to: self.addressed_to,
            timestamp: self.timestamp,
            metadata: self.metadata,
        }
    }
}

/// A text document to ingest via [`Session::ingest_document`].
///
/// Construct with [`Document::new`] (the content) and refine with the
/// chainable setters. Maps to one artifact ingest; a UUID v7 artifact id
/// is generated automatically unless set with [`Document::id`].
///
/// # Examples
///
/// ```
/// use uniko_memory::Document;
///
/// let doc = Document::new("# Release notes\n\n- fixed the deploy")
///     .kind("document")
///     .path("notes.md");
/// # let _ = doc;
/// ```
#[derive(Debug, Clone)]
pub struct Document {
    artifact_id: Option<String>,
    content: String,
    kind: String,
    path: Option<String>,
    metadata: HashMap<String, serde_json::Value>,
}

impl Document {
    /// A `"document"`-kind artifact carrying `content`.
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            artifact_id: None,
            content: content.into(),
            kind: "document".to_string(),
            path: None,
            metadata: HashMap::new(),
        }
    }

    /// Set an explicit artifact id (otherwise a UUID v7 is generated).
    #[must_use]
    pub fn id(mut self, artifact_id: impl Into<String>) -> Self {
        self.artifact_id = Some(artifact_id.into());
        self
    }

    /// Override the artifact kind (defaults to `"document"`).
    #[must_use]
    pub fn kind(mut self, kind: impl Into<String>) -> Self {
        self.kind = kind.into();
        self
    }

    /// Record the source path / URL on the artifact.
    #[must_use]
    pub fn path(mut self, path: impl Into<String>) -> Self {
        self.path = Some(path.into());
        self
    }

    /// Attach one arbitrary metadata entry forwarded to ingest.
    #[must_use]
    pub fn metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Lower into the wire artifact, attached to `session_id`.
    fn into_ingest_artifact(self, session_id: String) -> IngestArtifact {
        IngestArtifact {
            artifact_id: self.artifact_id.unwrap_or_else(uniko_store::id::new_id),
            content: self.content,
            kind: self.kind,
            path: self.path,
            metadata: self.metadata,
            session_id: Some(session_id),
            triggered_by_message_id: None,
            produced_by_action_id: None,
        }
    }
}

/// Where the bytes of a PDF come from, for [`Session::ingest_pdf`].
#[derive(Debug, Clone)]
pub enum PdfSource {
    /// In-memory bytes.
    Bytes(Vec<u8>),
    /// A filesystem path; bytes are read at ingest time.
    Path(PathBuf),
}

impl PdfSource {
    /// The original path recorded on the artifact, when known.
    fn source_path(&self) -> Option<String> {
        match self {
            Self::Bytes(_) => None,
            Self::Path(path) => Some(path.display().to_string()),
        }
    }

    /// Lower into the extractor's input type.
    fn into_pdf_input(self) -> PdfInput {
        match self {
            Self::Bytes(bytes) => PdfInput::Bytes(bytes),
            Self::Path(path) => PdfInput::Path(path),
        }
    }
}
