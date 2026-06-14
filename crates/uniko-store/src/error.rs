//! Error types for the uniko cognitive memory system.
//!
//! `UnikoError` is the single error type used across all layers. Each variant
//! corresponds to a distinct failure domain. Lower layers (uni-db) are wrapped
//! via `From` conversions to avoid leaking internal types.

/// The unified error type for all uniko operations.
#[derive(Debug, thiserror::Error)]
pub enum UnikoError {
    /// Graph storage operation failed (wraps uni-db errors).
    #[error("storage error: {0}")]
    Storage(String),

    /// Search operation failed (vector, fulltext, or hybrid).
    #[error("search error: {0}")]
    Search(String),

    /// Schema registration or validation failed.
    #[error("schema error: {0}")]
    Schema(String),

    /// Pipeline step execution failed.
    #[error("pipeline error: {0}")]
    Pipeline(String),

    /// Locy rule execution or management failed.
    #[error("locy error: {0}")]
    Locy(String),

    /// Configuration validation failed.
    #[error("config error: {0}")]
    Config(String),

    /// Embedding computation failed.
    #[error("embedding error: {0}")]
    Embedding(String),

    /// LLM provider call failed.
    #[error("LLM error: {0}")]
    Llm(String),

    /// Operation exceeded the configured timeout.
    #[error("timeout after {0}ms")]
    Timeout(u64),

    /// A transient storage-layer concurrency conflict: an optimistic-concurrency
    /// (SSI) abort, a constraint/transaction conflict, or commit/lock-timeout
    /// contention. Re-running the operation from a fresh transaction may succeed.
    /// The original uni-db message is preserved for diagnostics.
    ///
    /// This variant exists so callers can distinguish retriable contention from
    /// permanent failures (which stay [`UnikoError::Storage`]); see
    /// [`UnikoError::is_retriable`] and [`crate::KnowledgeBase::transact_with_retry`].
    #[error("conflict (retriable): {0}")]
    Conflict(String),

    /// Unexpected internal error.
    #[error("internal error: {0}")]
    Internal(String),
}

impl UnikoError {
    /// Returns `true` when this error is a transient storage conflict that a
    /// fresh transaction may win. Mirrors `uni_db::UniError::is_retriable`.
    #[must_use]
    pub fn is_retriable(&self) -> bool {
        matches!(self, UnikoError::Conflict(_))
    }
}

impl From<uni_db::UniError> for UnikoError {
    fn from(err: uni_db::UniError) -> Self {
        // Preserve uni-db's own retriability classification across the boundary
        // instead of flattening every variant to an opaque Storage(String).
        // `UniError` is `#[non_exhaustive]`, so we delegate to its classifier
        // rather than enumerating variants here.
        if err.is_retriable() {
            UnikoError::Conflict(err.to_string())
        } else {
            UnikoError::Storage(err.to_string())
        }
    }
}

/// A `Result` alias using `UnikoError` as the error type.
pub type Result<T> = std::result::Result<T, UnikoError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let cases: Vec<(UnikoError, &str)> = vec![
            (
                UnikoError::Storage("disk full".into()),
                "storage error: disk full",
            ),
            (
                UnikoError::Search("no index".into()),
                "search error: no index",
            ),
            (
                UnikoError::Schema("bad label".into()),
                "schema error: bad label",
            ),
            (
                UnikoError::Pipeline("step failed".into()),
                "pipeline error: step failed",
            ),
            (
                UnikoError::Locy("rule error".into()),
                "locy error: rule error",
            ),
            (
                UnikoError::Config("invalid".into()),
                "config error: invalid",
            ),
            (
                UnikoError::Embedding("model missing".into()),
                "embedding error: model missing",
            ),
            (
                UnikoError::Llm("rate limited".into()),
                "LLM error: rate limited",
            ),
            (UnikoError::Timeout(5000), "timeout after 5000ms"),
            (
                UnikoError::Internal("unexpected".into()),
                "internal error: unexpected",
            ),
        ];

        for (err, expected) in cases {
            assert_eq!(err.to_string(), expected);
        }
    }

    #[test]
    fn test_error_from_unidb() {
        let uni_err = uni_db::UniError::NotFound {
            path: "node/test-123".into(),
        };
        let err: UnikoError = uni_err.into();
        match &err {
            UnikoError::Storage(msg) => {
                assert!(msg.to_lowercase().contains("not found"), "got: {msg}")
            }
            other => panic!("expected Storage, got: {other:?}"),
        }
    }

    #[test]
    fn test_retriable_unidb_errors_map_to_conflict() {
        // Every uni-db variant classified retriable must cross the boundary as a
        // retriable UnikoError::Conflict — not get flattened into opaque Storage.
        let retriable = vec![
            uni_db::UniError::SerializationConflict {
                message: "lost update".into(),
            },
            uni_db::UniError::ConstraintConflict {
                message: "unique".into(),
            },
            uni_db::UniError::TransactionConflict {
                message: "ww".into(),
            },
            uni_db::UniError::CommitTimeout {
                tx_id: "tx-1".into(),
                hint: "contended",
            },
            uni_db::UniError::LockTimeout { timeout_ms: 50 },
        ];
        for uni_err in retriable {
            let err: UnikoError = uni_err.into();
            assert!(
                matches!(err, UnikoError::Conflict(_)),
                "expected Conflict, got: {err:?}"
            );
            assert!(err.is_retriable(), "Conflict must be retriable: {err:?}");
        }
    }

    #[test]
    fn test_non_retriable_unidb_error_maps_to_storage() {
        let err: UnikoError = uni_db::UniError::Schema {
            message: "no such label".into(),
        }
        .into();
        assert!(
            matches!(err, UnikoError::Storage(_)),
            "expected Storage, got: {err:?}"
        );
        assert!(!err.is_retriable());
    }
}
