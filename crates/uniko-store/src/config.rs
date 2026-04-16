/// Runtime configuration for the uniko cognitive memory system.
///
/// All fields have spec-mandated defaults. Use `UnikoConfig::default()` and override
/// individual fields as needed. Call `validate()` before use to catch constraint violations.
use serde::{Deserialize, Serialize};

use crate::error::{Result, UnikoError};

/// Configuration for all uniko runtime parameters.
///
/// Default values match the uniko specification v6.0 exactly.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UnikoConfig {
    // Pipeline capacities
    /// Bounded channel capacity for the ingest worker.
    pub ingest_queue_capacity: usize,
    /// Bounded channel capacity for the consolidation worker.
    pub consolidation_queue_capacity: usize,

    // Consolidation triggers
    /// Number of observations that trigger consolidation.
    pub consolidation_threshold: u32,
    /// Seconds between periodic consolidation runs.
    pub consolidation_interval_secs: u64,

    // Retry policy
    /// Maximum retry attempts for retryable operations.
    pub retry_max_attempts: u32,
    /// Initial delay in milliseconds before first retry (exponential backoff base).
    pub retry_initial_delay_ms: u64,
    /// Maximum delay in milliseconds between retries (backoff cap).
    pub retry_max_delay_ms: u64,

    // Circuit breaker
    /// Number of consecutive failures before the circuit breaker opens.
    pub circuit_failure_threshold: u32,
    /// Milliseconds the circuit breaker stays open before probing.
    pub circuit_recovery_ms: u64,

    // Chunking thresholds
    /// Token count above which messages are chunked.
    pub message_chunk_threshold: usize,
    /// Token count above which action outputs overflow to Artifact nodes.
    pub action_output_artifact_threshold: usize,

    // Chunk sizing
    /// Maximum tokens per chunk.
    pub max_chunk_tokens: usize,
    /// Minimum tokens per chunk (fragments below this are merged).
    pub min_chunk_tokens: usize,

    // Memory decay
    /// Half-life in days for importance decay: `importance * exp(-ln(2) / half_life * age_days)`.
    pub half_life_days: f64,
    /// Importance threshold below which nodes are pruned.
    pub prune_below: f64,

    // Recall cascade thresholds
    /// Coverage threshold for Phase 1 (Compact) early exit.
    pub phase1_coverage_threshold: f64,
    /// Coverage threshold for Phase 2 (Expand) early exit.
    pub phase2_coverage_threshold: f64,
}

impl Default for UnikoConfig {
    fn default() -> Self {
        Self {
            ingest_queue_capacity: 200,
            consolidation_queue_capacity: 32,
            consolidation_threshold: 20,
            consolidation_interval_secs: 900,
            retry_max_attempts: 3,
            retry_initial_delay_ms: 500,
            retry_max_delay_ms: 30_000,
            circuit_failure_threshold: 5,
            circuit_recovery_ms: 60_000,
            message_chunk_threshold: 1024,
            action_output_artifact_threshold: 256,
            max_chunk_tokens: 512,
            min_chunk_tokens: 64,
            half_life_days: 30.0,
            prune_below: 0.05,
            phase1_coverage_threshold: 0.75,
            phase2_coverage_threshold: 0.65,
        }
    }
}

impl UnikoConfig {
    /// Validate configuration constraints.
    ///
    /// Returns `Err(UnikoError::Config)` if any constraint is violated.
    pub fn validate(&self) -> Result<()> {
        if self.min_chunk_tokens >= self.max_chunk_tokens {
            return Err(UnikoError::Config(format!(
                "min_chunk_tokens ({}) must be less than max_chunk_tokens ({})",
                self.min_chunk_tokens, self.max_chunk_tokens,
            )));
        }

        if self.half_life_days <= 0.0 {
            return Err(UnikoError::Config(format!(
                "half_life_days ({}) must be positive",
                self.half_life_days,
            )));
        }

        if self.prune_below < 0.0 || self.prune_below >= 1.0 {
            return Err(UnikoError::Config(format!(
                "prune_below ({}) must be in [0.0, 1.0)",
                self.prune_below,
            )));
        }

        if self.phase1_coverage_threshold <= 0.0 || self.phase1_coverage_threshold > 1.0 {
            return Err(UnikoError::Config(format!(
                "phase1_coverage_threshold ({}) must be in (0.0, 1.0]",
                self.phase1_coverage_threshold,
            )));
        }

        if self.phase2_coverage_threshold <= 0.0 || self.phase2_coverage_threshold > 1.0 {
            return Err(UnikoError::Config(format!(
                "phase2_coverage_threshold ({}) must be in (0.0, 1.0]",
                self.phase2_coverage_threshold,
            )));
        }

        if self.retry_initial_delay_ms > self.retry_max_delay_ms {
            return Err(UnikoError::Config(format!(
                "retry_initial_delay_ms ({}) must not exceed retry_max_delay_ms ({})",
                self.retry_initial_delay_ms, self.retry_max_delay_ms,
            )));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let c = UnikoConfig::default();
        assert_eq!(c.ingest_queue_capacity, 200);
        assert_eq!(c.consolidation_queue_capacity, 32);
        assert_eq!(c.consolidation_threshold, 20);
        assert_eq!(c.consolidation_interval_secs, 900);
        assert_eq!(c.retry_max_attempts, 3);
        assert_eq!(c.retry_initial_delay_ms, 500);
        assert_eq!(c.retry_max_delay_ms, 30_000);
        assert_eq!(c.circuit_failure_threshold, 5);
        assert_eq!(c.circuit_recovery_ms, 60_000);
        assert_eq!(c.message_chunk_threshold, 1024);
        assert_eq!(c.action_output_artifact_threshold, 256);
        assert_eq!(c.max_chunk_tokens, 512);
        assert_eq!(c.min_chunk_tokens, 64);
        assert_eq!(c.half_life_days, 30.0);
        assert_eq!(c.prune_below, 0.05);
        assert_eq!(c.phase1_coverage_threshold, 0.75);
        assert_eq!(c.phase2_coverage_threshold, 0.65);
    }

    #[test]
    fn test_config_validation_ok() {
        UnikoConfig::default()
            .validate()
            .expect("default config must be valid");
    }

    #[test]
    fn test_config_validation_fails() {
        // min >= max chunk tokens
        let mut c = UnikoConfig::default();
        c.min_chunk_tokens = 600;
        assert!(c.validate().is_err());

        // half_life_days <= 0
        let mut c = UnikoConfig::default();
        c.half_life_days = 0.0;
        assert!(c.validate().is_err());

        // prune_below out of range
        let mut c = UnikoConfig::default();
        c.prune_below = 1.0;
        assert!(c.validate().is_err());

        // phase1 threshold out of range
        let mut c = UnikoConfig::default();
        c.phase1_coverage_threshold = 0.0;
        assert!(c.validate().is_err());

        // phase2 threshold out of range
        let mut c = UnikoConfig::default();
        c.phase2_coverage_threshold = 1.5;
        assert!(c.validate().is_err());

        // retry initial > max
        let mut c = UnikoConfig::default();
        c.retry_initial_delay_ms = 50_000;
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_config_serde_roundtrip() {
        let original = UnikoConfig::default();
        let json = serde_json::to_string(&original).expect("serialize");
        let restored: UnikoConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(original, restored);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn proptest_config_roundtrip(
            ingest_cap in 1usize..1000,
            consol_cap in 1usize..100,
            max_chunk in 100usize..2000,
            // Use integer-derived f64 to avoid JSON float precision drift
            half_life_tenths in 1u32..3650,
        ) {
            let half_life = f64::from(half_life_tenths) / 10.0;
            let config = UnikoConfig {
                ingest_queue_capacity: ingest_cap,
                consolidation_queue_capacity: consol_cap,
                max_chunk_tokens: max_chunk,
                min_chunk_tokens: max_chunk / 2,
                half_life_days: half_life,
                ..UnikoConfig::default()
            };
            let json = serde_json::to_string(&config).unwrap();
            let restored: UnikoConfig = serde_json::from_str(&json).unwrap();
            prop_assert_eq!(config, restored);
        }
    }
}
