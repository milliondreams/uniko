// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Industries Inc.

//! The `bridge!` / `bridge_sync!` macros — collapse the two skins to one line.
//!
//! Every async facade method clones its Arc-backed handle, runs the borrow
//! inside a single async block, and hands the future to
//! `pyo3_async_runtimes::tokio::future_into_py` to become a Python awaitable.
//! `bridge!` captures that shape; the `$body` block is an `async` body that
//! resolves to a `PyResult<T>` where `T: IntoPyObject`.
//!
//! `bridge_sync!` is the second skin: it runs the *same* `$body` to completion
//! on the module's shared runtime, releasing the GIL for the duration so other
//! Python threads progress. The two macros take the same arguments, so a
//! `*_sync` twin is a copy of its async sibling with `bridge!` swapped for
//! `bridge_sync!` and the return type changed from a `Bound<PyAny>` awaitable
//! to the body's resolved `T`.

/// Run `$body` as a Python awaitable, binding `$name` to `$init` first.
///
/// `$init` is evaluated eagerly (outside the future) so cheap Arc clones of
/// the handle happen on the calling thread, not when the future is polled.
macro_rules! bridge {
    ($py:expr, $name:ident = $init:expr, $body:block) => {{
        let $name = $init;
        pyo3_async_runtimes::tokio::future_into_py($py, async move $body)
    }};
}

/// Run `$body` to completion on the shared runtime, blocking the caller.
///
/// `$init` is evaluated with the GIL held (so Arc clones / arg moves happen on
/// the calling thread), then [`Python::detach`] releases the GIL and
/// `block_on` drives the future on the shared multi-thread runtime — the same
/// one [`bridge!`] uses, so submitted/streaming work shares one executor. The
/// body re-acquires the GIL via `Python::attach` only for the final output
/// conversion. `detach` reacquires the GIL before returning, so the resolved
/// `Py<…>` is handed back to Python safely. `block_on` runs on the Python
/// caller's thread (never a runtime worker), so it cannot panic on re-entry.
macro_rules! bridge_sync {
    ($py:expr, $name:ident = $init:expr, $body:block) => {{
        let $name = $init;
        $py.detach(move || {
            pyo3_async_runtimes::tokio::get_runtime().block_on(async move $body)
        })
    }};
}

pub(crate) use bridge;
pub(crate) use bridge_sync;
