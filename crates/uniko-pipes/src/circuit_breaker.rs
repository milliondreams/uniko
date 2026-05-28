//! Lock-free circuit breaker for external dependency protection.
//!
//! Wraps all LLM calls.  When the provider fails repeatedly the breaker
//! opens and pipeline steps fall back to local alternatives.

use std::future::Future;
use std::sync::atomic::{AtomicU8, AtomicU32, AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use uniko_store::UnikoError;

const STATE_CLOSED: u8 = 0;
const STATE_OPEN: u8 = 1;
const STATE_HALF_OPEN: u8 = 2;

/// Observable state of the circuit breaker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CircuitState {
    /// Normal operation — calls pass through.
    Closed,
    /// Tripped — calls are rejected immediately.
    Open,
    /// Probing — one call allowed to test recovery.
    HalfOpen,
}

/// Lock-free circuit breaker using atomic operations.
///
/// State machine: `Closed → Open → HalfOpen → Closed` (on success) or
/// `HalfOpen → Open` (on failure).
#[derive(Debug)]
pub struct CircuitBreaker {
    state: AtomicU8,
    failure_count: AtomicU32,
    last_failure_ms: AtomicU64,
    /// Consecutive failures before opening.
    failure_threshold: u32,
    /// Milliseconds to wait before probing after Open.
    recovery_ms: u64,
}

impl CircuitBreaker {
    /// Create a new breaker with the given thresholds.
    pub fn new(failure_threshold: u32, recovery_ms: u64) -> Self {
        Self {
            state: AtomicU8::new(STATE_CLOSED),
            failure_count: AtomicU32::new(0),
            last_failure_ms: AtomicU64::new(0),
            failure_threshold,
            recovery_ms,
        }
    }

    /// Read the current state.
    pub fn state(&self) -> CircuitState {
        match self.state.load(Ordering::Relaxed) {
            STATE_CLOSED => CircuitState::Closed,
            STATE_OPEN => CircuitState::Open,
            STATE_HALF_OPEN => CircuitState::HalfOpen,
            // All writers go through the `STATE_*` consts; any other
            // value would mean memory corruption.
            other => unreachable!("invalid circuit state byte: {other}"),
        }
    }

    /// Whether calls are currently allowed (Closed or HalfOpen).
    pub fn is_available(&self) -> bool {
        let st = self.state.load(Ordering::Relaxed);
        if st == STATE_CLOSED || st == STATE_HALF_OPEN {
            return true;
        }
        // Open — check if recovery period elapsed.
        self.check_recovery()
    }

    /// Execute `f` through the breaker.
    ///
    /// # Errors
    ///
    /// Returns `UnikoError::Llm("circuit breaker open")` if the breaker
    /// is open and recovery time has not elapsed, or propagates the error
    /// from `f`.
    pub async fn call<F, Fut, T>(&self, f: F) -> Result<T, UnikoError>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<T, UnikoError>>,
    {
        let st = self.state.load(Ordering::Acquire);

        if st == STATE_OPEN {
            if !self.check_recovery() {
                return Err(UnikoError::Llm("circuit breaker open".into()));
            }
            // Transition to HalfOpen for probe.
            self.state.store(STATE_HALF_OPEN, Ordering::Release);
            tracing::info!("circuit breaker HALF-OPEN — probing");
        }

        match f().await {
            Ok(val) => {
                self.record_success();
                Ok(val)
            }
            Err(e) => {
                self.record_failure();
                Err(e)
            }
        }
    }

    /// Record a successful call — resets failure count, closes breaker.
    pub fn record_success(&self) {
        let prev = self.state.swap(STATE_CLOSED, Ordering::Release);
        self.failure_count.store(0, Ordering::Relaxed);
        if prev != STATE_CLOSED {
            tracing::info!("circuit breaker CLOSED — normal operation resumed");
        }
    }

    /// Record a failed call — increments count, may open breaker.
    pub fn record_failure(&self) {
        let count = self.failure_count.fetch_add(1, Ordering::Relaxed) + 1;
        self.last_failure_ms.store(now_ms(), Ordering::Relaxed);

        if count >= self.failure_threshold {
            let prev = self.state.swap(STATE_OPEN, Ordering::Release);
            if prev != STATE_OPEN {
                tracing::warn!(
                    failures = count,
                    recovery_ms = self.recovery_ms,
                    "circuit breaker OPEN — calls disabled",
                );
            }
        }
    }

    /// Check whether enough time has elapsed to attempt recovery.
    fn check_recovery(&self) -> bool {
        let last = self.last_failure_ms.load(Ordering::Relaxed);
        let elapsed = now_ms().saturating_sub(last);
        elapsed >= self.recovery_ms
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_state_closed() {
        let cb = CircuitBreaker::new(3, 1000);
        assert_eq!(cb.state(), CircuitState::Closed);
        assert!(cb.is_available());
    }

    #[test]
    fn test_open_after_threshold() {
        let cb = CircuitBreaker::new(3, 60_000);
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed);

        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
        assert!(!cb.is_available());
    }

    #[test]
    fn test_success_resets() {
        let cb = CircuitBreaker::new(3, 60_000);
        cb.record_failure();
        cb.record_failure();
        cb.record_success();
        assert_eq!(cb.state(), CircuitState::Closed);

        // Need 3 fresh failures to open again.
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn test_recovery_after_timeout() {
        let cb = CircuitBreaker::new(2, 0); // 0 ms recovery = immediate
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);

        // With 0 ms recovery, is_available should return true.
        assert!(cb.is_available());
    }

    #[tokio::test]
    async fn test_call_closed_success() {
        let cb = CircuitBreaker::new(3, 1000);
        let result = cb.call(|| async { Ok::<_, UnikoError>(42) }).await;
        assert_eq!(result.unwrap(), 42);
    }

    #[tokio::test]
    async fn test_call_open_rejects() {
        let cb = CircuitBreaker::new(2, 60_000);
        cb.record_failure();
        cb.record_failure();
        let result = cb.call(|| async { Ok::<_, UnikoError>(42) }).await;
        assert!(result.is_err());
    }
}
