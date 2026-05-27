//! Retry with exponential backoff, jitter, and cancellation awareness.

use std::future::Future;
use std::time::Duration;

use rand::Rng;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

/// Configures retry behavior for fallible operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    /// Maximum number of attempts (including the first).
    pub max_attempts: u32,
    /// Base delay in milliseconds before the first retry.
    pub initial_delay_ms: u64,
    /// Maximum delay cap in milliseconds.
    pub max_delay_ms: u64,
    /// Multiplier applied per attempt: `delay = initial * mult^(attempt-1)`.
    pub backoff_multiplier: f64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay_ms: 500,
            max_delay_ms: 30_000,
            backoff_multiplier: 2.0,
        }
    }
}

/// Execute `f` with retries according to `policy`, aborting on cancel.
///
/// The first attempt runs immediately.  On failure, waits with
/// exponential backoff + ±25 % jitter before retrying.  Returns the
/// last error if all attempts are exhausted.
///
/// # Errors
///
/// Returns the last error from `f` if all attempts fail, or
/// a cancellation error if the token fires during a backoff sleep.
pub async fn retry_with_policy<F, Fut, T, E>(
    policy: &RetryPolicy,
    cancel: &CancellationToken,
    mut f: F,
) -> Result<T, E>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, E>>,
    E: std::fmt::Display,
{
    let mut last_err: Option<E> = None;

    for attempt in 0..policy.max_attempts {
        // Backoff sleep (skip on first attempt).
        if attempt > 0 {
            let base_ms = policy.initial_delay_ms as f64
                * policy.backoff_multiplier.powi((attempt - 1) as i32);
            let capped_ms = base_ms.min(policy.max_delay_ms as f64);

            // ±25 % uniform jitter.
            let jitter = rand::rng().random_range(0.75..=1.25);
            let delay_ms = (capped_ms * jitter) as u64;
            let delay = Duration::from_millis(delay_ms);

            tokio::select! {
                biased;
                () = cancel.cancelled() => {
                    tracing::debug!(attempt, "retry cancelled during backoff");
                    // Return the last error we have.
                    if let Some(e) = last_err {
                        return Err(e);
                    }
                    // Should not happen, but satisfy the compiler.
                    return (f)().await;
                }
                () = tokio::time::sleep(delay) => {}
            }
        }

        match (f)().await {
            Ok(val) => return Ok(val),
            Err(e) => {
                tracing::warn!(
                    attempt = attempt + 1,
                    max = policy.max_attempts,
                    error = %e,
                    "retry attempt failed",
                );
                last_err = Some(e);
            }
        }
    }

    // All attempts exhausted.
    Err(last_err.expect("at least one attempt must have run"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    #[tokio::test]
    async fn test_retry_success_first_attempt() {
        let policy = RetryPolicy {
            max_attempts: 3,
            initial_delay_ms: 10,
            max_delay_ms: 100,
            backoff_multiplier: 2.0,
        };
        let cancel = CancellationToken::new();
        let result: Result<i32, String> =
            retry_with_policy(&policy, &cancel, || async { Ok(42) }).await;
        assert_eq!(result.unwrap(), 42);
    }

    #[tokio::test]
    async fn test_retry_success_second_attempt() {
        let counter = AtomicU32::new(0);
        let policy = RetryPolicy {
            max_attempts: 3,
            initial_delay_ms: 10,
            max_delay_ms: 100,
            backoff_multiplier: 2.0,
        };
        let cancel = CancellationToken::new();
        let result: Result<i32, String> = retry_with_policy(&policy, &cancel, || {
            let n = counter.fetch_add(1, Ordering::Relaxed);
            async move {
                if n == 0 {
                    Err("transient".to_string())
                } else {
                    Ok(42)
                }
            }
        })
        .await;
        assert_eq!(result.unwrap(), 42);
        assert_eq!(counter.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn test_retry_exhausted() {
        let policy = RetryPolicy {
            max_attempts: 2,
            initial_delay_ms: 10,
            max_delay_ms: 100,
            backoff_multiplier: 2.0,
        };
        let cancel = CancellationToken::new();
        let result: Result<i32, String> =
            retry_with_policy(&policy, &cancel, || async { Err("fail".to_string()) }).await;
        assert_eq!(result.unwrap_err(), "fail");
    }

    #[tokio::test]
    async fn test_retry_cancellation() {
        let policy = RetryPolicy {
            max_attempts: 5,
            initial_delay_ms: 5_000,
            max_delay_ms: 30_000,
            backoff_multiplier: 2.0,
        };
        let cancel = CancellationToken::new();

        // Cancel after a short delay.
        let cancel_clone = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            cancel_clone.cancel();
        });

        let start = std::time::Instant::now();
        let result: Result<i32, String> =
            retry_with_policy(&policy, &cancel, || async { Err("fail".to_string()) }).await;
        let elapsed = start.elapsed();

        assert!(result.is_err());
        // Should abort quickly (< 1 s), not wait the full 5 s backoff.
        assert!(elapsed < Duration::from_secs(1));
    }
}
