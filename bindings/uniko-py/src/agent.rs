// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Industries Inc.

//! The [`PyAgent`] handle — the agent-scoped surface over recall, answer,
//! query, and conversation sessions.
//!
//! `Agent` is `Clone` and `Arc`-backed, so each async method clones it cheaply
//! and runs inside one `await`. The scoped variants (`recall_in`/`query_in`/
//! `answer_in`) arrive with the `Scope` type in the logic-surface phase.

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use uniko_api::tools::Agent;

use crate::convert::{self, params_to_rust};
use crate::data::PyData;
use crate::errors::to_pyerr;
use crate::goals::PyGoals;
use crate::logic::PyAssumeBuilder;
use crate::macros::bridge;
use crate::outputs::{PyAbductionResult, PyAnswer, PyContextBundle, PyDeletionReport};
use crate::scope::PyScope;
use crate::session::PySession;

/// An agent-scoped handle over the cognitive-memory surface.
///
/// Obtained from [`Uniko.agent`](crate::facade::PyUniko). Binds an agent
/// identity to the store so recall / answer / query / session run without
/// re-passing it.
#[pyclass(name = "Agent", module = "uniko")]
pub struct PyAgent {
    inner: Agent,
}

impl PyAgent {
    /// Wrap an owned facade [`Agent`].
    pub(crate) fn wrap(agent: Agent) -> Self {
        Self { inner: agent }
    }
}

#[pymethods]
impl PyAgent {
    /// The bound agent's participant id.
    #[getter]
    fn agent_id(&self) -> &str {
        self.inner.agent_id()
    }

    /// Run the recall cascade for `query`, returning a ranked context bundle.
    fn recall<'py>(&self, py: Python<'py>, query: String) -> PyResult<Bound<'py, PyAny>> {
        bridge!(py, agent = self.inner.clone(), {
            let bundle = agent.recall(&query).await.map_err(to_pyerr)?;
            Python::attach(|py| PyContextBundle::from_rust(py, &bundle))
        })
    }

    /// Recall context for `question` and generate an answer with the
    /// configured LLM.
    ///
    /// Raises `ConfigError` if the instance was built without an LLM.
    fn answer<'py>(&self, py: Python<'py>, question: String) -> PyResult<Bound<'py, PyAny>> {
        bridge!(py, agent = self.inner.clone(), {
            let answer = agent.answer(&question).await.map_err(to_pyerr)?;
            Python::attach(|py| PyAnswer::from_rust(py, &answer))
        })
    }

    /// Run a read-only Cypher query and return the rows as `list[dict]`.
    ///
    /// Rejects any non-read-only Cypher, so this can never mutate the graph.
    fn query<'py>(&self, py: Python<'py>, cypher: String) -> PyResult<Bound<'py, PyAny>> {
        bridge!(py, agent = self.inner.clone(), {
            let rows = agent.query(&cypher).await.map_err(to_pyerr)?;
            Python::attach(|py| convert::records_to_py(py, &rows))
        })
    }

    /// Recall scoped to `scope` (session / participant / time filters).
    fn recall_in<'py>(
        &self,
        py: Python<'py>,
        query: String,
        scope: &PyScope,
    ) -> PyResult<Bound<'py, PyAny>> {
        let scope = scope.snapshot();
        bridge!(py, agent = self.inner.clone(), {
            let bundle = agent.recall_in(&query, scope).await.map_err(to_pyerr)?;
            Python::attach(|py| PyContextBundle::from_rust(py, &bundle))
        })
    }

    /// Generate an answer with recall scoped to `scope`.
    fn answer_in<'py>(
        &self,
        py: Python<'py>,
        question: String,
        scope: &PyScope,
    ) -> PyResult<Bound<'py, PyAny>> {
        let scope = scope.snapshot();
        bridge!(py, agent = self.inner.clone(), {
            let answer = agent.answer_in(&question, scope).await.map_err(to_pyerr)?;
            Python::attach(|py| PyAnswer::from_rust(py, &answer))
        })
    }

    /// Run read-only Cypher with `scope`'s allow-set bound as `$allow`.
    ///
    /// Returns raw rows with no visibility filtering — use `recall_in` when
    /// results must be policy-scoped.
    fn query_in<'py>(
        &self,
        py: Python<'py>,
        cypher: String,
        scope: &PyScope,
    ) -> PyResult<Bound<'py, PyAny>> {
        let scope = scope.snapshot();
        bridge!(py, agent = self.inner.clone(), {
            let rows = agent.query_in(&cypher, &scope).await.map_err(to_pyerr)?;
            Python::attach(|py| convert::records_to_py(py, &rows))
        })
    }

    /// Define a Locy derivation rule, returning its node id.
    fn define_rule<'py>(
        &self,
        py: Python<'py>,
        name: String,
        source: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        bridge!(py, agent = self.inner.clone(), {
            agent.define_rule(name, source).await.map_err(to_pyerr)
        })
    }

    /// Run a registered Locy rule, returning its `YIELD` rows as `list[dict]`.
    #[pyo3(signature = (name, return_cols, params=None))]
    fn run_rule<'py>(
        &self,
        py: Python<'py>,
        name: String,
        return_cols: Vec<String>,
        params: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let params: HashMap<_, _> = params_to_rust(params.as_ref())?;
        bridge!(py, agent = self.inner.clone(), {
            let cols: Vec<&str> = return_cols.iter().map(String::as_str).collect();
            let rows = agent
                .run_rule(&name, &cols, params)
                .await
                .map_err(to_pyerr)?;
            Python::attach(|py| convert::records_to_py(py, &rows))
        })
    }

    /// Run an abductive query: find the minimal facts supporting a conclusion.
    #[pyo3(signature = (program, params=None))]
    fn abduce<'py>(
        &self,
        py: Python<'py>,
        program: String,
        params: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let params = params_to_rust(params.as_ref())?;
        bridge!(py, agent = self.inner.clone(), {
            let result = agent.abduce(&program, params).await.map_err(to_pyerr)?;
            Python::attach(|py| PyAbductionResult::from_rust(py, &result))
        })
    }

    /// Begin a hypothetical (`ASSUME`) query. Finish with `then_query` + `run`.
    fn assume(&self, assume_block: String) -> PyAssumeBuilder {
        PyAssumeBuilder::new(self.inner.clone(), assume_block)
    }

    /// Open a [`Session`](crate::session::PySession) for feeding turns.
    fn session(&self, session_id: String) -> PySession {
        PySession::wrap(self.inner.session(session_id))
    }

    /// Addressed (by-id) retrieval of messages and artifacts.
    #[getter]
    fn data(&self) -> PyData {
        PyData::wrap(self.inner.clone())
    }

    /// The agent's goal/task lifecycle surface.
    #[getter]
    fn goals(&self) -> PyGoals {
        PyGoals::wrap(self.inner.clone())
    }

    /// Hard-delete an entire conversation and everything it owns.
    fn delete_session<'py>(
        &self,
        py: Python<'py>,
        session_id: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        bridge!(py, agent = self.inner.clone(), {
            let report = agent.delete_session(&session_id).await.map_err(to_pyerr)?;
            Python::attach(|py| PyDeletionReport::from_rust(py, &report))
        })
    }

    /// GDPR erasure: remove a participant and the data they authored.
    fn forget_participant<'py>(
        &self,
        py: Python<'py>,
        participant_id: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        bridge!(py, agent = self.inner.clone(), {
            let report = agent
                .forget_participant(&participant_id)
                .await
                .map_err(to_pyerr)?;
            Python::attach(|py| PyDeletionReport::from_rust(py, &report))
        })
    }

    fn __repr__(&self) -> String {
        format!("Agent(agent_id='{}')", self.inner.agent_id())
    }
}
