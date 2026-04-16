//! # uniko-pipes — Layer 2: Pipeline Infrastructure
//!
//! Generic pipeline machinery: Step trait, circuit breaker, retry with backoff,
//! dead-letter queue, cancellation tokens, health checks, and metrics.
//!
//! Depends on `uniko-store` only. Content processing steps live in `uniko-extract`.

pub mod cancel;
pub mod circuit_breaker;
pub mod config;
pub mod dead_letter;
pub mod health;
pub mod metrics;
pub mod retry;
pub mod step;
pub mod types;
