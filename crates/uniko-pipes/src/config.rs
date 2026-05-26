//! Pipeline configuration with spec-mandated defaults.

// Rust guideline compliant

use serde::{Deserialize, Serialize};

use crate::retry::RetryPolicy;

/// Configuration for the pipeline execution engine.
///
/// All fields have defaults matching the uniko specification v6.0.
/// Use [`PipelineConfig::default()`] and override individual fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Bounded channel capacity for ingest tasks.
    pub ingest_queue_capacity: usize,
    /// Maximum concurrent ingest item processing.
    pub ingest_concurrency: usize,
    /// Bounded channel capacity for consolidation tasks.
    pub consolidation_queue_capacity: usize,
    /// Maximum concurrent consolidation cycles.
    pub consolidation_concurrency: usize,
    /// Observation count that triggers consolidation for an agent.
    pub consolidation_threshold: u32,
    /// Seconds between periodic consolidation sweeps.
    pub consolidation_interval_secs: u64,
    /// Maximum unprocessed Observations to derive Facts from per cycle.
    ///
    /// Caps work per cycle so a long-running ingest doesn't starve
    /// other agents.  Spillover is picked up on the next sweep.
    pub consolidation_batch_size: u32,
    /// Run P5 (procedure promotion) + P6 (topic detection) every N
    /// successful consolidation cycles for an agent.
    ///
    /// `0` disables the cortex sweep entirely.
    pub cortex_cycle_every_n_consolidations: u32,
    /// Minimum wall-clock gap (seconds) between cortex sweeps.
    ///
    /// Acts as an upper bound on cortex frequency even when many
    /// consolidation cycles fire rapidly.  Applied independently per
    /// agent for procedures and globally for topics.
    pub cortex_min_interval_secs: u64,
    /// Use the LLM (when available) to extract `(subject, predicate,
    /// object)` triples from observation content.
    ///
    /// Default `false` reuses the SRL/DEP fields P3 already populates.
    /// Setting `true` runs an LLM call per observation cluster — higher
    /// precision, materially slower and more expensive.
    pub extract_triples_via_llm: bool,
    /// Retry policy for LLM-dependent operations.
    pub retry: RetryPolicy,
    /// Consecutive LLM failures before the circuit breaker opens.
    pub circuit_failure_threshold: u32,
    /// Milliseconds the circuit breaker stays open before probing.
    pub circuit_recovery_ms: u64,
    /// Maximum automatic retries for dead-letter items.
    pub dead_letter_max_retries: u32,
    /// Seconds between automatic dead-letter retry sweeps.
    pub dead_letter_check_interval_secs: u64,
    /// Total seconds allowed for graceful shutdown.
    pub shutdown_timeout_secs: u64,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            ingest_queue_capacity: 200,
            ingest_concurrency: 8,
            consolidation_queue_capacity: 32,
            consolidation_concurrency: 4,
            consolidation_threshold: 20,
            consolidation_interval_secs: 900,
            consolidation_batch_size: 500,
            cortex_cycle_every_n_consolidations: 4,
            cortex_min_interval_secs: 600,
            extract_triples_via_llm: false,
            retry: RetryPolicy::default(),
            circuit_failure_threshold: 5,
            circuit_recovery_ms: 60_000,
            dead_letter_max_retries: 3,
            dead_letter_check_interval_secs: 300,
            shutdown_timeout_secs: 30,
        }
    }
}
