// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Industries Inc.

//! The `bridge!` macro — collapse the native-async skin to one line.
//!
//! Every async facade method clones its Arc-backed handle, runs the borrow
//! inside a single async block, and hands the future to
//! `pyo3_async_runtimes::tokio::future_into_py` to become a Python awaitable.
//! `bridge!` captures that shape; the `$body` block is an `async` body that
//! resolves to a `PyResult<T>` where `T: IntoPyObject`.
//!
//! A sync-skin arm is intentionally reserved for Phase 4 (`*_sync` methods
//! that `block_on` the same runtime); only the async arm exists today.

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

pub(crate) use bridge;
