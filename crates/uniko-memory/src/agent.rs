//! The [`Agent`] facade — one ergonomic handle over the agent-facing
//! tools.
//!
//! Every tool in this crate is a free function taking `(&KnowledgeBase,
//! agent_id, params)`.  That is the right shape for the library core,
//! but a caller driving a single agent repeats `kb` and `agent_id` on
//! every call.  `Agent` binds both once and exposes the same tools as
//! methods; the free functions remain the implementation, so there is no
//! behavioural divergence between the two surfaces.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use tokio::sync::OnceCell;
use uniko_store::locy::{AbductionResult, AssumeBuilder, Record};
use uniko_store::operations::facts::FactUpsert;
use uniko_store::xervo::{GenerationOptions, Message};
use uniko_store::{KnowledgeBase, NodeId, UnikoError, Value};

use crate::action::{RecordActionParams, RecordActionResult, record_action};
use crate::episode::{RecordEpisodeParams, record_episode};
use crate::facade::{RecallScope, Session};
use crate::fact::{AssertFactParams, InvalidateFactParams, assert_fact, invalidate_fact};
use crate::goal::{CreateGoalParams, create_goal};
use crate::nl_to_cypher::is_safe_read_only;
use crate::observation::{AddObservationParams, add_observation};
use crate::pipeline::PipelineSystem;
use crate::policy::Viewer;
use crate::query::{GeneratedAnswer, QueryOutcome, QueryRecordOptions, answer_query};
use crate::recall::{ContextBundle, RecallConfig, ViewerScope, recall};
use crate::rules::{AddRuleParams, add_rule};
use crate::task::{CreateTaskParams, create_task};
use crate::working_memory::{WorkingMemoryParams, working_memory};

/// System prompt used by [`Agent::answer`] when generating a reply.
const ANSWER_SYSTEM_PROMPT: &str = "You are a helpful assistant. Answer the question using only the \
    provided context. If the context does not contain the answer, say so. Answer concisely.";

/// An agent-scoped handle over the cognitive-memory tools.
///
/// Holds a cloneable [`KnowledgeBase`] (cheap — it is `Arc`-backed) and
/// the agent's `participant_id`, so each method delegates to the
/// corresponding free function without the caller re-passing them.
///
/// # Examples
///
/// ```no_run
/// # async fn demo(kb: uniko_store::KnowledgeBase) -> Result<(), uniko_store::UnikoError> {
/// use uniko_memory::Agent;
/// use uniko_memory::CreateGoalParams;
///
/// let agent = Agent::new(kb, "assistant-1");
/// let goal = agent
///     .create_goal(CreateGoalParams {
///         title: "Ship the release".into(),
///         ..Default::default()
///     })
///     .await?;
/// # let _ = goal;
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct Agent {
    kb: KnowledgeBase,
    agent_id: String,
    /// LLM alias enabling [`Agent::answer`], when one was configured.
    llm_alias: Option<String>,
    /// Streaming ingest pipeline, shared from the owning `Uniko`.
    streaming: Option<Arc<PipelineSystem>>,
    /// Read visibility scope resolved at recall time.
    scope: RecallScope,
    /// Memoized [`Viewer`] for [`RecallScope::AsAgent`], resolved once.
    viewer_cache: Arc<OnceCell<Viewer>>,
}

impl fmt::Debug for Agent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // KnowledgeBase is not Debug (it wraps the live store handle);
        // redact it and surface only the agent identity.
        f.debug_struct("Agent")
            .field("agent_id", &self.agent_id)
            .finish_non_exhaustive()
    }
}

impl Agent {
    /// Create a facade bound to `kb` and `agent_id`.
    ///
    /// Reads default to [`RecallScope::Unrestricted`] and no LLM is
    /// configured; for the full facade use [`Uniko::agent`](crate::Uniko::agent).
    pub fn new(kb: KnowledgeBase, agent_id: impl Into<String>) -> Self {
        Self::with_context(kb, agent_id, None, None, RecallScope::Unrestricted)
    }

    /// Create a facade carrying the owning instance's LLM, streaming
    /// pipeline, and visibility scope.
    pub(crate) fn with_context(
        kb: KnowledgeBase,
        agent_id: impl Into<String>,
        llm_alias: Option<String>,
        streaming: Option<Arc<PipelineSystem>>,
        scope: RecallScope,
    ) -> Self {
        Self {
            kb,
            agent_id: agent_id.into(),
            llm_alias,
            streaming,
            scope,
            viewer_cache: Arc::new(OnceCell::new()),
        }
    }

    /// The underlying knowledge base.
    pub fn kb(&self) -> &KnowledgeBase {
        &self.kb
    }

    /// The bound agent's `participant_id`.
    pub fn agent_id(&self) -> &str {
        &self.agent_id
    }

    /// Create a Goal owned by this agent (see [`create_goal`]).
    ///
    /// # Errors
    ///
    /// Propagates [`create_goal`]'s errors.
    pub async fn create_goal(&self, params: CreateGoalParams) -> Result<NodeId, UnikoError> {
        create_goal(&self.kb, &self.agent_id, params).await
    }

    /// Create a Task assigned to this agent (see [`create_task`]).
    ///
    /// # Errors
    ///
    /// Propagates [`create_task`]'s errors.
    pub async fn create_task(&self, params: CreateTaskParams) -> Result<NodeId, UnikoError> {
        create_task(&self.kb, &self.agent_id, params).await
    }

    /// Assert a Fact (see [`assert_fact`]).
    ///
    /// # Errors
    ///
    /// Propagates [`assert_fact`]'s errors.
    pub async fn assert_fact(&self, params: AssertFactParams) -> Result<FactUpsert, UnikoError> {
        assert_fact(&self.kb, params).await
    }

    /// Invalidate a Fact (see [`invalidate_fact`]).
    ///
    /// # Errors
    ///
    /// Propagates [`invalidate_fact`]'s errors.
    pub async fn invalidate_fact(&self, params: InvalidateFactParams) -> Result<(), UnikoError> {
        invalidate_fact(&self.kb, params).await
    }

    /// Add an Observation anchored to a Message (see [`add_observation`]).
    ///
    /// # Errors
    ///
    /// Propagates [`add_observation`]'s errors.
    pub async fn add_observation(
        &self,
        params: AddObservationParams,
    ) -> Result<NodeId, UnikoError> {
        add_observation(&self.kb, &self.agent_id, params).await
    }

    /// Record an Episode for this agent (see [`record_episode`]).
    ///
    /// # Errors
    ///
    /// Propagates [`record_episode`]'s errors.
    pub async fn record_episode(&self, params: RecordEpisodeParams) -> Result<NodeId, UnikoError> {
        record_episode(&self.kb, &self.agent_id, params).await
    }

    /// Record an Action for this agent (see [`record_action`]).
    ///
    /// # Errors
    ///
    /// Propagates [`record_action`]'s errors.
    pub async fn record_action(
        &self,
        params: RecordActionParams,
    ) -> Result<RecordActionResult, UnikoError> {
        record_action(&self.kb, &self.agent_id, params).await
    }

    /// Run a read-only Cypher query and return the result rows.
    ///
    /// Each row is a [`Record`] (column name → [`Value`]). The query is
    /// rejected unless it is read-only, so this can never mutate the graph.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] when the query is not read-only, or
    /// [`UnikoError::Locy`] on an evaluation error.
    pub async fn query(&self, cypher: &str) -> Result<Vec<Record>, UnikoError> {
        if !is_safe_read_only(cypher) {
            return Err(UnikoError::Storage(
                "query() accepts read-only Cypher only".to_string(),
            ));
        }
        self.kb.execute_rule(cypher, &HashMap::new()).await
    }

    /// Define a Locy derivation rule from `source`, tracked with a
    /// confidence lifecycle.
    ///
    /// `source` is Locy code (e.g. `CREATE RULE <name> AS MATCH … YIELD …`).
    /// The rule is registered with the logic runtime and recorded as a
    /// `candidate` `:Rule`; the confidence lifecycle (decay / promotion /
    /// pruning) is then driven automatically by consolidation.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Locy`] on a syntax error (no node is left
    /// behind), or [`UnikoError::Storage`] on a write failure.
    pub async fn define_rule(
        &self,
        name: impl Into<String>,
        source: impl Into<String>,
    ) -> Result<NodeId, UnikoError> {
        add_rule(
            &self.kb,
            AddRuleParams {
                name: name.into(),
                source: source.into(),
                source_type: "authored".to_string(),
                ..AddRuleParams::default()
            },
        )
        .await
    }

    /// Run a registered Locy rule by `name`, returning its `YIELD` rows.
    ///
    /// `return_cols` names the rule's `YIELD` aliases; `params` injects
    /// rule body parameters (e.g. `("agent_id", Value::from(...))`).
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Locy`] if the rule is unregistered or
    /// evaluation fails.
    pub async fn run_rule(
        &self,
        name: &str,
        return_cols: &[&str],
        params: HashMap<String, Value>,
    ) -> Result<Vec<Record>, UnikoError> {
        self.kb.query_rule(name, return_cols, &params).await
    }

    /// Begin a hypothetical (`ASSUME`) query: fork state, apply mutations,
    /// query, then roll back — the real graph is never modified.
    ///
    /// Finish with [`AssumeBuilder::then_query`] and
    /// [`AssumeBuilder::run`].
    pub fn assume(&self, assume_block: &str) -> AssumeBuilder<'_> {
        self.kb.assume(assume_block)
    }

    /// Run an abductive query: given a conclusion, find the minimal set of
    /// facts that support it.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Locy`] on an evaluation failure.
    pub async fn abduce(
        &self,
        program: &str,
        params: HashMap<String, Value>,
    ) -> Result<AbductionResult, UnikoError> {
        self.kb.abduce(program, &params).await
    }

    /// Assemble working-memory context (see [`working_memory`]).
    ///
    /// # Errors
    ///
    /// Propagates [`working_memory`]'s errors.
    pub async fn working_memory(
        &self,
        params: WorkingMemoryParams,
    ) -> Result<ContextBundle, UnikoError> {
        working_memory(&self.kb, params).await
    }

    /// Open a [`Session`] for feeding conversation turns.
    pub fn session(&self, session_id: impl Into<String>) -> Session {
        Session::new(
            self.kb.clone(),
            session_id,
            self.streaming.clone(),
            self.llm_alias.clone(),
        )
    }

    /// Run the recall cascade for `query` with the best-known config.
    ///
    /// Derives the configuration from the instance's [`UnikoConfig`] (the
    /// validated stack) and applies this agent's visibility scope. For a
    /// custom configuration use [`recall_with`](Agent::recall_with).
    ///
    /// # Errors
    ///
    /// Propagates [`recall`]'s errors and any membership-resolution
    /// failure when scoping to the agent.
    pub async fn recall(&self, query: &str) -> Result<ContextBundle, UnikoError> {
        let mut config = RecallConfig::from_uniko_config(self.kb.config());
        config.viewer = self.resolve_viewer_scope().await?;
        recall(&self.kb, query, &config).await
    }

    /// Run the recall cascade with a caller-supplied configuration.
    ///
    /// # Errors
    ///
    /// Propagates [`recall`]'s errors.
    pub async fn recall_with(
        &self,
        query: &str,
        config: &RecallConfig,
    ) -> Result<ContextBundle, UnikoError> {
        recall(&self.kb, query, config).await
    }

    /// Recall context for `question` and generate an answer with the
    /// configured LLM, recording a query Episode.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Config`] when no LLM was configured (build
    /// with [`Uniko::builder`](crate::Uniko::builder)`.llm(...)`).
    /// Propagates recall and generation failures.
    pub async fn answer(&self, question: &str) -> Result<QueryOutcome, UnikoError> {
        let alias = self.llm_alias.as_deref().ok_or_else(|| {
            UnikoError::Config(
                "answer() requires an LLM; build with Uniko::builder().llm(...)".to_string(),
            )
        })?;

        let mut config = RecallConfig::from_uniko_config(self.kb.config());
        config.viewer = self.resolve_viewer_scope().await?;

        let kb = &self.kb;
        // Build the prompt synchronously in the closure body so the
        // returned future captures only owned data (`user`) plus the
        // `self`-lifetime borrows (`kb`, `alias`) — never the closure's
        // `&ContextBundle` / `&str` arguments, which would require an
        // (inexpressible) higher-ranked lifetime on the future.
        let generator = |bundle: &ContextBundle, question: &str| {
            let context = format_context(bundle);
            let user = format!("Context:\n{context}\n\nQuestion: {question}\n\nAnswer:");
            async move {
                let messages = vec![Message::system(ANSWER_SYSTEM_PROMPT), Message::user(&user)];
                let options = GenerationOptions {
                    max_tokens: Some(2048),
                    temperature: Some(0.1),
                    ..Default::default()
                };
                let text = kb.generate(alias, &messages, options).await?;
                Ok(GeneratedAnswer {
                    text,
                    input_tokens: None,
                    output_tokens: None,
                    model: Some(alias.to_string()),
                })
            }
        };

        let record = Some(QueryRecordOptions {
            participant_id: self.agent_id.clone(),
            outcome: Some("success".to_string()),
            ..Default::default()
        });
        answer_query(kb, question, &config, generator, record).await
    }

    /// Resolve this agent's [`RecallScope`] into a [`ViewerScope`].
    ///
    /// For [`RecallScope::AsAgent`] the [`Viewer`] is resolved against the
    /// store once and cached for the lifetime of this `Agent` handle.
    async fn resolve_viewer_scope(&self) -> Result<ViewerScope, UnikoError> {
        match &self.scope {
            RecallScope::Unrestricted => Ok(ViewerScope::Unrestricted),
            RecallScope::As(viewer) => Ok(ViewerScope::As(viewer.clone())),
            RecallScope::AsAgent => {
                let viewer = self
                    .viewer_cache
                    .get_or_try_init(|| Viewer::new(&self.kb, &self.agent_id))
                    .await?;
                Ok(ViewerScope::As(viewer.clone()))
            }
        }
    }
}

/// Render a recall bundle into a numbered context block for the LLM.
fn format_context(bundle: &ContextBundle) -> String {
    bundle
        .items
        .iter()
        .enumerate()
        .map(|(i, item)| format!("[{}] {}", i + 1, item.content))
        .collect::<Vec<_>>()
        .join("\n")
}
