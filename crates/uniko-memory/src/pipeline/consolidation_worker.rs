//! Consolidation pipeline worker with threshold, timer, and force triggers.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use uniko_cortex::procedures::{LifecycleConfig, promote_procedures_once};
use uniko_cortex::topics::{TopicConfig, detect_topics_once};
use uniko_pipes::config::PipelineConfig;
use uniko_pipes::health::HealthTracker;
use uniko_pipes::metrics;
use uniko_pipes::types::ConsolidationTask;
use uniko_store::KnowledgeBase;

/// Long-running consolidation worker with per-agent observation
/// counters and multi-trigger consolidation logic.
pub(crate) struct ConsolidationWorker {
    rx: mpsc::Receiver<ConsolidationTask>,
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
    /// Run cortex (P5+P6) every N successful consolidation cycles.
    ///
    /// Zero disables the cortex sweep entirely.
    cortex_every_n: u32,
    /// Minimum wall-clock gap between cortex sweeps.
    cortex_min_interval: Duration,
    /// Per-agent count of consolidation cycles since the last
    /// successful cortex sweep for that agent.
    agent_cycles_since_cortex: HashMap<String, u32>,
    /// Last time procedure promotion ran for each agent.
    agent_last_procedure_at: HashMap<String, Instant>,
    /// Last time topic detection ran globally — P6 is agent-agnostic,
    /// so multiple agents firing in the same window should not re-run
    /// it back-to-back.
    last_topic_sweep_at: Option<Instant>,
}

impl ConsolidationWorker {
    pub(crate) fn new(
        rx: mpsc::Receiver<ConsolidationTask>,
        cancel: CancellationToken,
        kb: Arc<KnowledgeBase>,
        health: Arc<Mutex<HealthTracker>>,
        config: &PipelineConfig,
    ) -> Self {
        Self {
            rx,
            cancel,
            kb,
            health,
            threshold: config.consolidation_threshold,
            interval: Duration::from_secs(config.consolidation_interval_secs),
            batch_size: config.consolidation_batch_size,
            agent_counters: HashMap::new(),
            cortex_every_n: config.cortex_cycle_every_n_consolidations,
            cortex_min_interval: Duration::from_secs(config.cortex_min_interval_secs),
            agent_cycles_since_cortex: HashMap::new(),
            agent_last_procedure_at: HashMap::new(),
            last_topic_sweep_at: None,
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
                            let new_count = {
                                let counter = self.agent_counters
                                    .entry(obs.agent_id.clone())
                                    .or_insert(0);
                                *counter += obs.observation_count;
                                *counter
                            };
                            tracing::debug!(
                                agent = %obs.agent_id,
                                count = new_count,
                                threshold = self.threshold,
                                "observation counter updated",
                            );
                            if new_count >= self.threshold {
                                self.agent_counters.insert(obs.agent_id.clone(), 0);
                                self.run_consolidation_cycle(&obs.agent_id).await;
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
    ///
    /// On success, increments the per-agent cortex counter and
    /// invokes [`Self::maybe_run_cortex_sweep`] which runs P5 + P6
    /// when the cadence gates allow.
    async fn run_consolidation_cycle(&mut self, agent_id: &str) {
        let start = Instant::now();
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
                self.maybe_run_cortex_sweep(agent_id).await;
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

    /// Run the cortex sweep (P5 procedure promotion + P6 topic
    /// detection) when the per-agent counter and the per-sweep
    /// wall-clock gates both allow.
    ///
    /// The cycle counter increments on every successful consolidation
    /// cycle.  When it reaches `cortex_every_n`, the sweep runs and
    /// the counter resets.  `cortex_min_interval` is enforced
    /// independently — separately per-agent for P5 (procedures are
    /// agent-scoped) and globally for P6 (topics are not).
    ///
    /// Cortex failures are logged and dropped: consolidation must
    /// stay healthy even when downstream reasoning misfires.
    async fn maybe_run_cortex_sweep(&mut self, agent_id: &str) {
        if self.cortex_every_n == 0 {
            return;
        }

        let count = self
            .agent_cycles_since_cortex
            .entry(agent_id.to_string())
            .or_insert(0);
        *count += 1;
        if *count < self.cortex_every_n {
            return;
        }
        *count = 0;

        let now = Instant::now();
        self.run_procedure_sweep(agent_id, now).await;
        self.run_topic_sweep(agent_id, now).await;
    }

    /// Run P5 — procedure promotion — for one agent.
    ///
    /// Skips when `cortex_min_interval` has not elapsed since the
    /// last run for this agent.
    async fn run_procedure_sweep(&mut self, agent_id: &str, now: Instant) {
        if let Some(last) = self.agent_last_procedure_at.get(agent_id)
            && now.duration_since(*last) < self.cortex_min_interval
        {
            return;
        }

        let start = Instant::now();
        metrics::emit_procedure_cycle(agent_id);
        match promote_procedures_once(&self.kb, agent_id, LifecycleConfig::default()).await {
            Ok(report) => {
                let elapsed_ms = start.elapsed().as_millis() as f64;
                metrics::emit_procedure_cycle_duration(elapsed_ms);
                metrics::emit_procedures_promoted(report.promoted);
                tracing::info!(
                    agent = %agent_id,
                    created = report.created,
                    reinforced = report.reinforced,
                    promoted = report.promoted,
                    duration_ms = elapsed_ms,
                    "procedure sweep complete",
                );
            }
            Err(e) => {
                let elapsed_ms = start.elapsed().as_millis() as f64;
                metrics::emit_procedure_cycle_duration(elapsed_ms);
                tracing::warn!(
                    agent = %agent_id,
                    error = %e,
                    duration_ms = elapsed_ms,
                    "procedure sweep failed — continuing",
                );
            }
        }
        self.agent_last_procedure_at
            .insert(agent_id.to_string(), now);
    }

    /// Run P6 — topic detection — globally.
    ///
    /// Skips when `cortex_min_interval` has not elapsed since the
    /// last global sweep; the `agent_id` is only used for the metric
    /// label that records who triggered the run.
    async fn run_topic_sweep(&mut self, agent_id: &str, now: Instant) {
        if let Some(last) = self.last_topic_sweep_at
            && now.duration_since(last) < self.cortex_min_interval
        {
            return;
        }

        let start = Instant::now();
        metrics::emit_topic_cycle(agent_id);
        match detect_topics_once(&self.kb, TopicConfig::default()).await {
            Ok(report) => {
                let elapsed_ms = start.elapsed().as_millis() as f64;
                metrics::emit_topic_cycle_duration(elapsed_ms);
                metrics::emit_topics_created(report.created);
                tracing::info!(
                    agent = %agent_id,
                    created = report.created,
                    updated = report.updated,
                    entities_assigned = report.entities_assigned,
                    duration_ms = elapsed_ms,
                    "topic sweep complete",
                );
            }
            Err(e) => {
                let elapsed_ms = start.elapsed().as_millis() as f64;
                metrics::emit_topic_cycle_duration(elapsed_ms);
                tracing::warn!(
                    agent = %agent_id,
                    error = %e,
                    duration_ms = elapsed_ms,
                    "topic sweep failed — continuing",
                );
            }
        }
        self.last_topic_sweep_at = Some(now);
    }
}
