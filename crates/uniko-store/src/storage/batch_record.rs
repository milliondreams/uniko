//! In-process capture of bulk-write batches for benchmarking.
//!
//! Records every node and edge batch handed to the uni-db bulk API
//! ([`Transaction::bulk_insert_vertices`](uni_db::Transaction) and
//! `bulk_insert_edges`) during ingestion, so a benchmark can later
//! replay the exact same batches against an alternative write path
//! (e.g. a Cypher `UNWIND … CREATE` statement) and compare timings.
//!
//! Recording is **opt-in and gated behind the `batch-record` cargo
//! feature**. Without the feature the [`record_node_batch`] /
//! [`record_edge_batch`] hooks compile to empty `#[inline(always)]`
//! no-ops, so production builds carry no global state and no per-call
//! overhead. With the feature, captured batches accumulate in a
//! process-global buffer that the benchmark turns on via
//! [`enable_batch_recording`] before a single ingest run and drains
//! once with [`take_recorded_batches`].
//!
//! This is a diagnostic facility, not part of the storage contract:
//! it captures native [`Value`](uni_db::Value) rows (so `Vector` and
//! `Temporal` types round-trip losslessly, unlike a JSON dump), and is
//! single-process only.

use std::collections::HashMap;

use uni_db::Value;

/// A single bulk-write batch captured during ingestion.
///
/// Each variant mirrors one call into uni-db's bulk API: a `Node`
/// batch came from `bulk_insert_vertices` (all rows share one label),
/// an `Edge` batch from `bulk_insert_edges` (all edges share one type).
#[derive(Debug, Clone)]
pub enum RecordedBatch {
    /// A `bulk_insert_vertices` batch — every row shares `label`.
    Node {
        /// Capture order (monotonic across the recording session).
        seq: u64,
        /// Node label the batch was inserted under.
        label: String,
        /// One property map per vertex, exactly as passed to uni-db.
        rows: Vec<HashMap<String, Value>>,
    },
    /// A `bulk_insert_edges` batch — every edge shares `edge_type`.
    Edge {
        /// Capture order (monotonic across the recording session).
        seq: u64,
        /// Edge type the batch was inserted under.
        edge_type: String,
        /// One property map per edge. Endpoint VIDs are intentionally
        /// not captured — replay synthesizes its own endpoints, since
        /// per-edge write cost is independent of which nodes are joined.
        props: Vec<HashMap<String, Value>>,
    },
}

impl RecordedBatch {
    /// Number of rows (vertices or edges) in this batch.
    pub fn len(&self) -> usize {
        match self {
            RecordedBatch::Node { rows, .. } => rows.len(),
            RecordedBatch::Edge { props, .. } => props.len(),
        }
    }

    /// Returns `true` if the batch carries no rows.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ── Feature-gated global buffer (diagnostic only) ───────────────────
//
// M-AVOID-STATICS: a process-global static is acceptable here because
// recording is an opt-in benchmark facility never consulted by
// correctness-relevant code, the buffer is compiled out entirely
// without the `batch-record` feature, and the workspace links a single
// version of this crate (no secret cross-version duplication).
#[cfg(feature = "batch-record")]
mod imp {
    use std::sync::{Mutex, MutexGuard, OnceLock};

    use super::RecordedBatch;

    static BUFFER: OnceLock<Mutex<Option<Vec<RecordedBatch>>>> = OnceLock::new();

    fn lock() -> MutexGuard<'static, Option<Vec<RecordedBatch>>> {
        // Recover from a poisoned lock: a panic mid-record leaves the
        // buffer intact enough for best-effort diagnostics.
        BUFFER
            .get_or_init(|| Mutex::new(None))
            .lock()
            .unwrap_or_else(|p| p.into_inner())
    }

    pub(super) fn enable() {
        *lock() = Some(Vec::new());
    }

    pub(super) fn take() -> Vec<RecordedBatch> {
        lock().take().unwrap_or_default()
    }

    pub(super) fn push(make: impl FnOnce(u64) -> RecordedBatch) {
        let mut guard = lock();
        if let Some(buf) = guard.as_mut() {
            let seq = buf.len() as u64;
            buf.push(make(seq));
        }
    }
}

/// Begin capturing bulk-write batches into the process-global buffer.
///
/// Resets any previously captured batches. Call once before the ingest
/// run whose batches you want to replay, then collect them with
/// [`take_recorded_batches`]. Only available with the `batch-record`
/// feature.
#[cfg(feature = "batch-record")]
pub fn enable_batch_recording() {
    imp::enable();
}

/// Drain and return every batch captured since [`enable_batch_recording`].
///
/// Recording is disabled afterwards (the buffer is taken, not cloned),
/// so a second call returns an empty vector. Only available with the
/// `batch-record` feature.
#[cfg(feature = "batch-record")]
pub fn take_recorded_batches() -> Vec<RecordedBatch> {
    imp::take()
}

// ── Internal hooks called from the bulk-write sites ─────────────────

/// Record a node batch when recording is enabled; otherwise a no-op.
///
/// `rows` is a thunk so the (cloning) materialization of the batch only
/// runs when the `batch-record` feature is on and recording is active.
#[cfg(feature = "batch-record")]
pub(crate) fn record_node_batch<F>(label: &str, rows: F)
where
    F: FnOnce() -> Vec<HashMap<String, Value>>,
{
    imp::push(|seq| RecordedBatch::Node {
        seq,
        label: label.to_string(),
        rows: rows(),
    });
}

/// No-op stand-in compiled when the `batch-record` feature is off.
#[cfg(not(feature = "batch-record"))]
#[inline(always)]
pub(crate) fn record_node_batch<F>(_label: &str, _rows: F)
where
    F: FnOnce() -> Vec<HashMap<String, Value>>,
{
}

/// Record an edge batch when recording is enabled; otherwise a no-op.
///
/// `props` is a thunk so the (cloning) materialization of the per-edge
/// property maps only runs when recording is active.
#[cfg(feature = "batch-record")]
pub(crate) fn record_edge_batch<F>(edge_type: &str, props: F)
where
    F: FnOnce() -> Vec<HashMap<String, Value>>,
{
    imp::push(|seq| RecordedBatch::Edge {
        seq,
        edge_type: edge_type.to_string(),
        props: props(),
    });
}

/// No-op stand-in compiled when the `batch-record` feature is off.
#[cfg(not(feature = "batch-record"))]
#[inline(always)]
pub(crate) fn record_edge_batch<F>(_edge_type: &str, _props: F)
where
    F: FnOnce() -> Vec<HashMap<String, Value>>,
{
}

#[cfg(all(test, feature = "batch-record"))]
mod tests {
    use super::*;

    // These tests share the process-global buffer, so they must not run
    // concurrently with each other. They are grouped into one test fn.
    #[test]
    fn enable_capture_and_take_roundtrip() {
        // Disabled by default: hooks drop batches on the floor.
        let _ = take_recorded_batches(); // clear any prior state
        record_node_batch("Entity", || vec![HashMap::new()]);
        assert!(
            take_recorded_batches().is_empty(),
            "recording must be off until explicitly enabled",
        );

        // Enabled: hooks capture in call order.
        enable_batch_recording();
        record_node_batch("Chunk", || vec![HashMap::new(), HashMap::new()]);
        record_edge_batch("MENTIONS", || vec![HashMap::new()]);
        let batches = take_recorded_batches();
        assert_eq!(batches.len(), 2);
        match &batches[0] {
            RecordedBatch::Node { seq, label, rows } => {
                assert_eq!(*seq, 0);
                assert_eq!(label, "Chunk");
                assert_eq!(rows.len(), 2);
            }
            other => panic!("expected Node batch, got {other:?}"),
        }
        match &batches[1] {
            RecordedBatch::Edge {
                seq,
                edge_type,
                props,
            } => {
                assert_eq!(*seq, 1);
                assert_eq!(edge_type, "MENTIONS");
                assert_eq!(props.len(), 1);
            }
            other => panic!("expected Edge batch, got {other:?}"),
        }

        // Drained: recording is now disabled again.
        record_node_batch("Fact", || vec![HashMap::new()]);
        assert!(take_recorded_batches().is_empty());
    }
}

// Rust guideline compliant
