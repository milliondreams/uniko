//! Striped async locks for serializing read-modify-write critical sections.
//!
//! uni-db's transaction commit is last-writer-wins, not serializable.
//! Two concurrent callers that read a row, mutate the value Rust-side,
//! and write it back can both observe the same pre-image and lose one
//! update.  Sites that perform this RMW pattern need an in-process
//! serializer; [`StripedLocks`] provides one without paying per-key
//! allocation.
//!
//! # Design
//!
//! - Per-key locking, not global: callers compute key bytes (e.g.
//!   `"fact:<subject>\0<predicate>"`) and the table hashes them into a
//!   fixed-arity stripe array.  Collisions serialize unrelated keys
//!   harmlessly; the default 256 stripes give ~0.4% collision
//!   probability at 10 concurrent keys.
//! - Async-safe: holders may `.await` (uni-db tx commit is async), so
//!   stripes are [`tokio::sync::Mutex`] rather than [`std::sync::Mutex`].
//! - No reentrancy: no site currently re-enters with the same key.  If
//!   a future site does, it must drop the outer guard first.
//! - No timeout / no `try_lock`: RMW critical sections complete in
//!   milliseconds; a timeout would introduce partial-success states.
//!
//! # Out of scope
//!
//! - Cross-process locking — uniko is single-process today.
//! - Read locks / shared mode — RMW critical sections are always
//!   exclusive.
//! - CAS-style lock-free RMW — uni-db does not expose CAS server-side.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use tokio::sync::{Mutex, MutexGuard};

/// Environment override for stripe count.  Read once at construction.
const ENV_STRIPES: &str = "UNIKO_RMW_STRIPES";

/// Default stripe count.  Picked for ~0.4% collision probability at 10
/// concurrent keys; tunable via [`ENV_STRIPES`].
const DEFAULT_STRIPES: usize = 256;

/// Fixed-arity striped async lock keyed by arbitrary `&[u8]`.
///
/// See module docs for design rationale.
#[derive(Debug)]
pub struct StripedLocks {
    stripes: Box<[Mutex<()>]>,
}

impl StripedLocks {
    /// Build a [`StripedLocks`] with `n` stripes.
    ///
    /// # Panics
    ///
    /// Panics if `n == 0`.
    #[must_use]
    pub fn new(n: usize) -> Self {
        assert!(n > 0, "StripedLocks must have at least 1 stripe");
        let stripes = (0..n).map(|_| Mutex::new(())).collect::<Vec<_>>();
        Self {
            stripes: stripes.into_boxed_slice(),
        }
    }

    /// Build a [`StripedLocks`] sized from the [`ENV_STRIPES`]
    /// environment variable, falling back to [`DEFAULT_STRIPES`].
    ///
    /// Non-numeric or zero values fall back to the default with a
    /// `tracing::warn!`.
    #[must_use]
    pub fn from_env() -> Self {
        let n = match std::env::var(ENV_STRIPES) {
            Ok(v) => match v.parse::<usize>() {
                Ok(n) if n > 0 => n,
                _ => {
                    tracing::warn!(
                        target: "uniko_store::locks",
                        value = %v,
                        default = DEFAULT_STRIPES,
                        "ignoring invalid {ENV_STRIPES}; using default",
                    );
                    DEFAULT_STRIPES
                }
            },
            Err(_) => DEFAULT_STRIPES,
        };
        Self::new(n)
    }

    /// Number of stripes in this table.
    #[must_use]
    pub fn stripe_count(&self) -> usize {
        self.stripes.len()
    }

    /// Acquire the stripe that hashes from `key`.
    ///
    /// Callers should hold the returned guard for the duration of the
    /// RMW critical section.
    pub async fn lock(&self, key: &[u8]) -> MutexGuard<'_, ()> {
        let idx = self.stripe_index(key);
        self.stripes[idx].lock().await
    }

    fn stripe_index(&self, key: &[u8]) -> usize {
        let mut h = DefaultHasher::new();
        key.hash(&mut h);
        // `stripes.len()` is non-zero (checked in `new`).
        (h.finish() as usize) % self.stripes.len()
    }
}

impl Default for StripedLocks {
    fn default() -> Self {
        Self::from_env()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    #[tokio::test]
    async fn same_key_serializes() {
        let locks = Arc::new(StripedLocks::new(16));
        let counter = Arc::new(AtomicUsize::new(0));
        let observed_max = Arc::new(AtomicUsize::new(0));

        let mut handles = Vec::new();
        for _ in 0..32 {
            let l = locks.clone();
            let c = counter.clone();
            let m = observed_max.clone();
            handles.push(tokio::spawn(async move {
                let _g = l.lock(b"shared-key").await;
                let v = c.fetch_add(1, Ordering::SeqCst) + 1;
                m.fetch_max(v, Ordering::SeqCst);
                tokio::task::yield_now().await;
                c.fetch_sub(1, Ordering::SeqCst);
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
        assert_eq!(
            observed_max.load(Ordering::SeqCst),
            1,
            "more than one holder of the same key was observed concurrently",
        );
    }

    #[tokio::test]
    async fn different_keys_can_overlap() {
        // With 256 stripes and only two distinct keys we expect them to
        // map to different stripes (collision probability ~0.4%); rerun
        // the same key pair if a flake ever shows up.
        let locks = Arc::new(StripedLocks::new(256));
        let g1 = locks.lock(b"key-one").await;
        // Should not block even though `g1` is held.
        let g2 = tokio::time::timeout(std::time::Duration::from_millis(50), locks.lock(b"key-two"))
            .await
            .expect("distinct keys must not block on the same stripe");
        drop(g2);
        drop(g1);
    }

    #[test]
    #[should_panic(expected = "at least 1 stripe")]
    fn zero_stripes_panics() {
        let _ = StripedLocks::new(0);
    }
}
