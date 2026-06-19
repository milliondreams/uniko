// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Industries Inc.

//! The [`PySession`] conversation handle and its [`PyTurn`] input builder.
//!
//! `Session` is the one facade handle that is neither `Clone` nor `&self` for
//! its primary verb (`observe` takes `&mut self`), and it owns cross-turn
//! state. `PySession` therefore holds it behind an `Arc<tokio::sync::Mutex>`
//! and locks across the `await` — serializing turns, which matches the
//! documented single-threaded contract. The external `session_id` is cached
//! alongside so the synchronous getter needs no lock.

use std::sync::Arc;

use pyo3::prelude::*;
use tokio::sync::Mutex;
use uniko_api::tools::{Session, Turn};

use crate::convert;
use crate::errors::to_pyerr;
use crate::ingest::PyIngestSource;
use crate::macros::bridge;
use crate::outputs::{PyDeletionReport, PyIngestOutcome, PyObserveResult};

/// One conversation turn to feed into a [`PySession`].
///
/// Construct with `Turn(sender_id, content)` and refine with the chainable
/// setters. A turn is reusable: feeding it does not consume the Python object.
#[pyclass(name = "Turn", module = "uniko")]
pub struct PyTurn {
    // `Option` so a consuming builder setter can take-apply-replace; always
    // `Some` between calls. A plain `std::sync::Mutex` is correct — no `.await`
    // is ever held across this lock.
    inner: std::sync::Mutex<Option<Turn>>,
}

impl PyTurn {
    /// Apply a consuming `Turn` builder method in place.
    fn map(&self, f: impl FnOnce(Turn) -> Turn) {
        let mut guard = self.inner.lock().expect("Turn mutex poisoned");
        if let Some(turn) = guard.take() {
            *guard = Some(f(turn));
        }
    }

    /// A cloned snapshot of the current turn, for ingest.
    pub(crate) fn snapshot(&self) -> PyResult<Turn> {
        self.inner
            .lock()
            .expect("Turn mutex poisoned")
            .clone()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Turn is in an invalid state"))
    }
}

#[pymethods]
impl PyTurn {
    /// A text turn from `sender_id` carrying `content`, stamped now.
    #[new]
    fn new(sender_id: String, content: String) -> Self {
        Self {
            inner: std::sync::Mutex::new(Some(Turn::new(sender_id, content))),
        }
    }

    /// Set an explicit message id for idempotent ingest.
    fn id<'py>(slf: PyRef<'py, Self>, message_id: String) -> PyRef<'py, Self> {
        slf.map(|t| t.id(message_id));
        slf
    }

    /// Override the content type (defaults to `"text"`).
    fn content_type<'py>(slf: PyRef<'py, Self>, content_type: String) -> PyRef<'py, Self> {
        slf.map(|t| t.content_type(content_type));
        slf
    }

    /// Set the send time (a Python `datetime`; defaults to now).
    fn at<'py>(slf: PyRef<'py, Self>, timestamp: &Bound<'_, PyAny>) -> PyResult<PyRef<'py, Self>> {
        let ts = convert::py_to_datetime(timestamp)?;
        slf.map(|t| t.at(ts));
        Ok(slf)
    }

    /// Set explicit recipient participant ids.
    fn addressed_to<'py>(slf: PyRef<'py, Self>, recipients: Vec<String>) -> PyRef<'py, Self> {
        slf.map(|t| t.addressed_to(recipients));
        slf
    }

    /// Attach one arbitrary metadata entry forwarded to ingest.
    fn metadata<'py>(
        slf: PyRef<'py, Self>,
        key: String,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<PyRef<'py, Self>> {
        let json = convert::py_to_json(value)?;
        slf.map(move |t| t.metadata(key, json));
        Ok(slf)
    }

    /// Attach a document/file shared in this turn.
    fn attach<'py>(
        slf: PyRef<'py, Self>,
        source: &PyIngestSource,
    ) -> PyResult<PyRef<'py, Self>> {
        let src = source.snapshot()?;
        slf.map(|t| t.attach(src));
        Ok(slf)
    }

    /// Attach several documents/files at once.
    fn attachments<'py>(
        slf: PyRef<'py, Self>,
        sources: Vec<Bound<'_, PyIngestSource>>,
    ) -> PyResult<PyRef<'py, Self>> {
        let srcs = sources
            .iter()
            .map(|s| s.borrow().snapshot())
            .collect::<PyResult<Vec<_>>>()?;
        slf.map(|t| t.attachments(srcs));
        Ok(slf)
    }

    fn __repr__(&self) -> String {
        match self.snapshot() {
            Ok(_) => "Turn(...)".to_string(),
            Err(_) => "Turn(<invalid>)".to_string(),
        }
    }
}

/// A conversation scope that feeds turns into memory.
///
/// Obtained from [`Agent.session`](crate::agent::PyAgent). Turns fed through
/// one session preserve cross-turn conversational state, so feed them in order.
#[pyclass(name = "Session", module = "uniko")]
pub struct PySession {
    inner: Arc<Mutex<Session>>,
    session_id: String,
}

impl PySession {
    /// Wrap an owned facade [`Session`], caching its external id.
    pub(crate) fn wrap(session: Session) -> Self {
        let session_id = session.session_id().to_string();
        Self {
            inner: Arc::new(Mutex::new(session)),
            session_id,
        }
    }
}

#[pymethods]
impl PySession {
    /// The session's external identifier.
    #[getter]
    fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Ingest one turn durably, committing before returning.
    ///
    /// The write is immediately visible to recall (read-after-write).
    fn observe<'py>(&self, py: Python<'py>, turn: &PyTurn) -> PyResult<Bound<'py, PyAny>> {
        let turn = turn.snapshot()?;
        bridge!(py, session = self.inner.clone(), {
            let mut guard = session.lock().await;
            let result = guard.observe(turn).await.map_err(to_pyerr)?;
            Python::attach(|py| PyObserveResult::from_rust(py, &result))
        })
    }

    /// Ingest a standalone document/blob into this session (not a turn).
    fn ingest<'py>(&self, py: Python<'py>, source: &PyIngestSource) -> PyResult<Bound<'py, PyAny>> {
        let src = source.snapshot()?;
        bridge!(py, session = self.inner.clone(), {
            let guard = session.lock().await;
            let outcome = guard.ingest(src).await.map_err(to_pyerr)?;
            Python::attach(|py| PyIngestOutcome::from_rust(py, &outcome))
        })
    }

    /// Submit a turn to the streaming pipeline (fire-and-forget).
    ///
    /// Requires the instance to be built with `streaming(True)`.
    fn submit<'py>(&self, py: Python<'py>, turn: &PyTurn) -> PyResult<Bound<'py, PyAny>> {
        let turn = turn.snapshot()?;
        bridge!(py, session = self.inner.clone(), {
            let guard = session.lock().await;
            guard.submit(turn).await.map_err(to_pyerr)?;
            Ok(())
        })
    }

    /// Submit a standalone document/blob to the streaming pipeline.
    fn submit_source<'py>(
        &self,
        py: Python<'py>,
        source: &PyIngestSource,
    ) -> PyResult<Bound<'py, PyAny>> {
        let src = source.snapshot()?;
        bridge!(py, session = self.inner.clone(), {
            let guard = session.lock().await;
            guard.submit_source(src).await.map_err(to_pyerr)?;
            Ok(())
        })
    }

    /// Wait for all submitted work in this session to finish.
    fn flush<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        bridge!(py, session = self.inner.clone(), {
            let guard = session.lock().await;
            guard.flush().await.map_err(to_pyerr)?;
            Ok(())
        })
    }

    /// Generate and persist a summary of the session so far.
    ///
    /// Resolves to the new summary node id, or `None` when there was nothing
    /// new to summarize.
    fn summarize<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        bridge!(py, session = self.inner.clone(), {
            let guard = session.lock().await;
            let node = guard.summarize().await.map_err(to_pyerr)?;
            Ok(node)
        })
    }

    /// Soft-forget a single turn (redact content, keep the chain intact).
    fn forget_turn<'py>(&self, py: Python<'py>, message_id: String) -> PyResult<Bound<'py, PyAny>> {
        bridge!(py, session = self.inner.clone(), {
            let guard = session.lock().await;
            let report = guard.forget_turn(&message_id).await.map_err(to_pyerr)?;
            Python::attach(|py| PyDeletionReport::from_rust(py, &report))
        })
    }

    /// Hard-delete a single turn and its cascade.
    fn delete_turn<'py>(&self, py: Python<'py>, message_id: String) -> PyResult<Bound<'py, PyAny>> {
        bridge!(py, session = self.inner.clone(), {
            let guard = session.lock().await;
            let report = guard.delete_turn(&message_id).await.map_err(to_pyerr)?;
            Python::attach(|py| PyDeletionReport::from_rust(py, &report))
        })
    }

    /// Hard-delete a document/artifact and its cascade.
    fn delete_document<'py>(
        &self,
        py: Python<'py>,
        artifact_id: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        bridge!(py, session = self.inner.clone(), {
            let guard = session.lock().await;
            let report = guard.delete_document(&artifact_id).await.map_err(to_pyerr)?;
            Python::attach(|py| PyDeletionReport::from_rust(py, &report))
        })
    }

    fn __repr__(&self) -> String {
        format!("Session(session_id='{}')", self.session_id)
    }
}
