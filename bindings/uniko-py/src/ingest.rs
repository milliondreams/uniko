// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Industries Inc.

//! The [`PyIngestSource`] input builder for unified blob ingest.
//!
//! Construct with `IngestSource.from_text` / `from_bytes` / `from_path`, refine
//! with the chainable setters, then feed it to `Session.ingest`,
//! `Session.submit_source`, or `Turn.attach`. The MIME is sniffed unless
//! `with_mime` overrides it.

use pyo3::prelude::*;
use uniko_api::tools::{IngestSource, Mime};

/// A blob to ingest through the unified, MIME-routed dispatch.
#[pyclass(name = "IngestSource", module = "uniko")]
pub struct PyIngestSource {
    // `Option` for the take-apply-replace builder pattern; always `Some`
    // between calls. No `.await` is held across this lock.
    inner: std::sync::Mutex<Option<IngestSource>>,
}

impl PyIngestSource {
    fn from_source(source: IngestSource) -> Self {
        Self {
            inner: std::sync::Mutex::new(Some(source)),
        }
    }

    /// Apply a consuming builder method in place.
    fn map(&self, f: impl FnOnce(IngestSource) -> IngestSource) {
        let mut guard = self.inner.lock().expect("IngestSource mutex poisoned");
        if let Some(source) = guard.take() {
            *guard = Some(f(source));
        }
    }

    /// A cloned snapshot of the current source, for ingest.
    pub(crate) fn snapshot(&self) -> PyResult<IngestSource> {
        self.inner
            .lock()
            .expect("IngestSource mutex poisoned")
            .clone()
            .ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("IngestSource is in an invalid state")
            })
    }
}

#[pymethods]
impl PyIngestSource {
    /// A UTF-8 text payload.
    #[staticmethod]
    fn from_text(content: String) -> Self {
        Self::from_source(IngestSource::text(content))
    }

    /// A raw byte payload (MIME sniffed from magic bytes unless overridden).
    #[staticmethod]
    fn from_bytes(data: Vec<u8>) -> Self {
        Self::from_source(IngestSource::bytes(data))
    }

    /// A filesystem path payload; bytes/text are read at ingest time.
    #[staticmethod]
    fn from_path(path: String) -> Self {
        Self::from_source(IngestSource::path(path))
    }

    /// Set an explicit MIME (e.g. `"application/pdf"`), skipping sniffing.
    fn with_mime<'py>(slf: PyRef<'py, Self>, mime: String) -> PyResult<PyRef<'py, Self>> {
        let parsed = Mime::parse(&mime)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("invalid MIME: {e}")))?;
        slf.map(|s| s.with_mime(parsed));
        Ok(slf)
    }

    /// Set an explicit artifact id.
    fn with_id<'py>(slf: PyRef<'py, Self>, id: String) -> PyRef<'py, Self> {
        slf.map(|s| s.with_id(id));
        slf
    }

    /// Record a source path / URL on the artifact.
    fn with_path<'py>(slf: PyRef<'py, Self>, path: String) -> PyRef<'py, Self> {
        slf.map(|s| s.with_path(path));
        slf
    }

    fn __repr__(&self) -> String {
        "IngestSource(...)".to_string()
    }
}
