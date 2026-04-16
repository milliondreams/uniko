use pyo3::prelude::*;

/// Python module for uniko cognitive memory.
#[pymodule]
fn uniko(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // TODO: register Python classes and functions in Phase 4
    let _ = m;
    Ok(())
}
