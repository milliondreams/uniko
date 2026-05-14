//! Consolidation pipeline worker with threshold, timer, and force triggers.

// Rust guideline compliant

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::sync::{Semaphore, mpsc};
use tokio_util::sync::CancellationToken;

use uniko_pipes::config::PipelineConfig;
use uniko_pipes::health::HealthTracker;
use uniko_pipes::metrics;
use uniko_pipes::types::ConsolidationTask;
use uniko_store::KnowledgeBase;

/// Long-running consolidation worker with per-agent observation
/// counters and multi-trigger consolidation logic.
pub(crate) struct ConsolidationWorker {
    rx: mpsc::Receiver<ConsolidationTask>,
    #[expect(
        dead_code,
        reason = "concurrency limiting not yet wired to cycle execution"
    )]
    semaphore: Arc<Semaphore>,
    cancel: CancellationToken,
    kb: Arc<KnowledgeBase>,
    health: Arc<Mutex<HealthTracker>>,
    /// Observation count that triggers consolidation.
    threshold: u32,
    /// Periodic consolidation interval.
    interval: Duration,
    /// Maximum observations to derive Facts from per cycle.
    batch_size: u32,
    /// Per-agent observation counters since last consolidation.
    agent_counters: HashMap<String, u32>,
}

impl ConsolidationWorker {
    pub(crate) fn new(
        rx: mpsc::Receiver<ConsolidationTask>,
        concurrency: usize,
        cancel: CancellationToken,
        kb: Arc<KnowledgeBase>,
        health: Arc<Mutex<HealthTracker>>,
        config: &PipelineConfig,
    ) -> Self {
        Self {
            rx,
            semaphore: Arc::new(Semaphore::new(concurrency)),
            cancel,
            kb,
            health,
            threshold: config.consolidation_threshold,
            interval: Duration::from_secs(config.consolidation_interval_secs),
            batch_size: config.consolidation_batch_size,
            agent_counters: HashMap::new(),
        }
    }

    /// Run the worker loop until cancelled or the channel closes.
    pub(crate) async fn run(mut self) {
        tracing::info!("consolidation worker started");
        let mut timer = tokio::time::interval(self.interval);
        // Consume the initial tick (fires immediately).
        timer.tick().await;

        loop {
            tokio::select! {
                biased;

                // 1. Shutdown (highest priority).
                () = self.cancel.cancelled() => {
                    tracing::info!("consolidation worker shutting down");
                    break;
                }

                // 2. Consolidation tasks.
                task = self.rx.recv() => {
                    match task {
                        Some(ConsolidationTask::ObservationsReady(obs)) => {
                            let counter = self.agent_counters
                                .entry(obs.agent_id.clone())
                                .or_insert(0);
                            *counter += obs.observation_count;
                            tracing::debug!(
                                agent = %obs.agent_id,
                                count = *counter,
                                threshold = self.threshold,
                                "observation counter updated",
                            );
                            if *counter >= self.threshold {
                                let agent_id = obs.agent_id.clone();
                                *counter = 0;
                                self.run_consolidation_cycle(&agent_id).await;
                            }
                        }
                        Some(ConsolidationTask::ForceConsolidate { agent_id }) => {
                            self.agent_counters.insert(agent_id.clone(), 0);
                            self.run_consolidation_cycle(&agent_id).await;
                        }
                        Some(ConsolidationTask::RunCycle { agent_id }) => {
                            self.run_consolidation_cycle(&agent_id).await;
                        }
                        None => {
                            tracing::info!("consolidation channel closed");
                            break;
                        }
                    }
                }

                // 3. Periodic timer (lowest priority — only fires when
                //    nothing else is ready).
                _ = timer.tick() => {
                    let agents: Vec<String> = self.agent_counters
                        .iter()
                        .filter(|(_, count)| **count > 0)
                        .map(|(agent_id, _)| agent_id.clone())
                        .collect();
                    for agent_id in agents {
                        self.agent_counters.insert(agent_id.clone(), 0);
                        self.run_consolidation_cycle(&agent_id).await;
                    }
                }
            }
        }
        tracing::info!("consolidation worker stopped");
    }

    /// Execute one consolidation cycle for `agent_id`.
    ///
    /// Delegates to [`crate::consolidation::run_cycle`], which derives
    /// Facts from unprocessed Observations and records a
    /// `ConsolidationCycle` audit node.  Errors are logged but never
    /// propagated — a failed cycle marks unhealthy and waits for the
    /// next trigger.
    async fn run_consolidation_cycle(&self, agent_id: &str) {
        let start = std::time::Instant::now();
        tracing::info!(agent = %agent_id, "consolidation cycle starting");

        metrics::emit_consolidation_cycle(agent_id);

        match crate::consolidation::run_cycle(&self.kb, agent_id, Some(self.batch_size as i64))
            .await
        {
            Ok(stats) => {
                let elapsed_ms = start.elapsed().as_millis() as f64;
                metrics::emit_consolidation_duration(elapsed_ms);
                self.health.lock().unwrap().record_success(elapsed_ms);
                tracing::info!(
                    agent = %agent_id,
                    processed = stats.observations_processed,
                    created = stats.facts_created,
                    reinforced = stats.facts_reinforced,
                    duration_ms = elapsed_ms,
                    "consolidation cycle complete",
                );
            }
            Err(e) => {
                let elapsed_ms = start.elapsed().as_millis() as f64;
                metrics::emit_consolidation_duration(elapsed_ms);
                self.health.lock().unwrap().record_failure();
                tracing::error!(
                    agent = %agent_id,
                    error = %e,
                    duration_ms = elapsed_ms,
                    "consolidation cycle failed",
                );
            }
        }
    }
}
