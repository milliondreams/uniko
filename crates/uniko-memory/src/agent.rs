//! The [`Agent`] facade — one ergonomic handle over the agent-facing
//! tools.
//!
//! Every tool in this crate is a free function taking `(&KnowledgeBase,
//! agent_id, params)`.  That is the right shape for the library core,
//! but a caller driving a single agent repeats `kb` and `agent_id` on
//! every call.  `Agent` binds both once and exposes the same tools as
//! methods; the free functions remain the implementation, so there is no
//! behavioural divergence between the two surfaces.

use std::fmt;

use uniko_store::operations::facts::FactUpsert;
use uniko_store::{KnowledgeBase, NodeId, UnikoError};

use crate::action::{RecordActionParams, RecordActionResult, record_action};
use crate::episode::{RecordEpisodeParams, record_episode};
use crate::fact::{AssertFactParams, InvalidateFactParams, assert_fact, invalidate_fact};
use crate::goal::{CreateGoalParams, create_goal};
use crate::observation::{AddObservationParams, add_observation};
use crate::recall::{ContextBundle, RecallConfig, recall};
use crate::task::{CreateTaskParams, create_task};
use crate::working_memory::{WorkingMemoryParams, working_memory};

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
    pub fn new(kb: KnowledgeBase, agent_id: impl Into<String>) -> Self {
        Self {
            kb,
            agent_id: agent_id.into(),
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

    /// Run the recall cascade for `query` (see [`recall`]).
    ///
    /// # Errors
    ///
    /// Propagates [`recall`]'s errors.
    pub async fn recall(
        &self,
        query: &str,
        config: &RecallConfig,
    ) -> Result<ContextBundle, UnikoError> {
        recall(&self.kb, query, config).await
    }
}
