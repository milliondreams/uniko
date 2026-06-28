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
    /// Last time the session-maintenance sweep (F14 auto-close + F59
    /// summarisation) ran globally — sessions are agent-agnostic, so
    /// this is throttled once across all agents like the topic sweep.
    last_session_sweep_at: Option<Instant>,
    /// Last time the Rule confidence-decay sweep ran globally — the
    /// `:Rule` lifecycle is agent-agnostic (it scans every non-stdlib
    /// rule), so it is throttled once across all agents.
    last_rule_decay_at: Option<Instant>,
    /// Whether to execute active Locy rules during the cortex sweep
    /// (`episode_pattern_detector`, `contradiction_detector`, and any
    /// user-defined rules). From `PipelineConfig::run_rules_in_sweep`.
    run_rules_in_sweep: bool,
    /// Last time the rule-execution sweep ran for each agent — rules are
    /// agent-scoped (they filter by `participant_id`), so this throttles
    /// per-agent like the procedure sweep.
    agent_last_rule_exec_at: HashMap<String, Instant>,
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
            last_session_sweep_at: None,
            last_rule_decay_at: None,
            run_rules_in_sweep: config.run_rules_in_sweep,
            agent_last_rule_exec_at: HashMap::new(),
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
                        // Reset the counter on the same owned string we
                        // already have rather than cloning again.
                        if let Some(c) = self.agent_counters.get_mut(&agent_id) {
                            *c = 0;
                        }
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
        tracing::info!(agent = %agent_id, "consolidation cycle starting");
        metrics::emit_consolidation_cycle(agent_id);
        let start = Instant::now();

        match crate::consolidation::run_cycle(&self.kb, agent_id, Some(self.batch_size as i64))
            .await
        {
            Ok(stats) => {
                let elapsed_ms = record_duration(start, metrics::emit_consolidation_duration);
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
                let elapsed_ms = record_duration(start, metrics::emit_consolidation_duration);
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
        self.run_decay_sweep(agent_id).await;
        // Execute active rules BEFORE decay so a rule that bound matches this
        // pass records its match and resets `missed_cycles` before the decay
        // sweep penalizes unmatched rules.
        self.run_rule_execution_sweep(agent_id, now).await;
        self.run_rule_decay_sweep(now).await;
        self.run_session_maintenance_sweep(agent_id, now).await;
    }

    /// Execute every active Locy rule for `agent_id` and record matches into
    /// the confidence lifecycle (P-rules). Handles the consumer-backed stdlib
    /// rules `episode_pattern_detector` and `contradiction_detector` plus any
    /// user-defined rules; `relevance_decay` and `sequence_detector` are run by
    /// their own dedicated sweeps and skipped here.
    ///
    /// Per-agent throttle (rules are agent-scoped). Gated by
    /// `run_rules_in_sweep`. Failures are logged and dropped — rule execution
    /// must never destabilize consolidation.
    async fn run_rule_execution_sweep(&mut self, agent_id: &str, now: Instant) {
        if !self.run_rules_in_sweep {
            return;
        }
        if let Some(last) = self.agent_last_rule_exec_at.get(agent_id)
            && now.duration_since(*last) < self.cortex_min_interval
        {
            return;
        }
        self.agent_last_rule_exec_at
            .insert(agent_id.to_string(), now);

        match crate::rules::run_active_rules(
            &self.kb,
            agent_id,
            crate::rules::RuleLifecycleConfig::default(),
        )
        .await
        {
            Ok(report) => {
                if report.rules_matched > 0 {
                    tracing::info!(
                        agent = %agent_id,
                        rules_run = report.rules_run,
                        rules_matched = report.rules_matched,
                        effects = report.effects,
                        "rule execution sweep applied effects",
                    );
                }
            }
            Err(e) => tracing::warn!(
                agent = %agent_id,
                error = %e,
                "rule execution sweep failed — continuing",
            ),
        }
    }

    /// Apply one Rule confidence-decay pass (decay / demote / promote /
    /// prune) across every non-stdlib `:Rule`.
    ///
    /// Global like the topic and session sweeps, so it is throttled once
    /// across all agents by `cortex_min_interval`. Failures are logged and
    /// dropped — rule maintenance must never destabilize consolidation.
    async fn run_rule_decay_sweep(&mut self, now: Instant) {
        if let Some(last) = self.last_rule_decay_at
            && now.duration_since(last) < self.cortex_min_interval
        {
            return;
        }
        self.last_rule_decay_at = Some(now);

        match crate::rules::apply_decay_cycle(
            &self.kb,
            crate::rules::RuleLifecycleConfig::default(),
        )
        .await
        {
            Ok(report) => {
                if report.demoted + report.promoted + report.pruned > 0 {
                    tracing::info!(
                        decayed = report.decayed,
                        demoted = report.demoted,
                        promoted = report.promoted,
                        pruned = report.pruned,
                        "rule decay sweep applied lifecycle transitions",
                    );
                }
            }
            Err(e) => tracing::warn!(error = %e, "rule decay sweep failed — continuing"),
        }
    }

    /// Auto-close inactive Sessions (F14) and summarise each one (F59).
    ///
    /// Global like the topic sweep: Sessions are agent-agnostic, so the
    /// run is throttled once across all agents by `cortex_min_interval`.
    /// A zero `session_inactivity_secs` disables auto-close, in which
    /// case `close_inactive_sessions` returns nothing and no summaries
    /// are produced.  Failures are logged and dropped — maintenance must
    /// never destabilise consolidation.
    async fn run_session_maintenance_sweep(&mut self, agent_id: &str, now: Instant) {
        if let Some(last) = self.last_session_sweep_at
            && now.duration_since(last) < self.cortex_min_interval
        {
            return;
        }
        self.last_session_sweep_at = Some(now);

        let inactivity_secs = self.kb.config().session_inactivity_secs;
        let wall_now = chrono::Utc::now();
        let closed = match self
            .kb
            .close_inactive_sessions(inactivity_secs, wall_now)
            .await
        {
            Ok(ids) => ids,
            Err(e) => {
                tracing::warn!(
                    agent = %agent_id,
                    error = %e,
                    "session auto-close failed — continuing",
                );
                return;
            }
        };

        for session_id in &closed {
            // Extractive, offline summary (no LLM alias) on the same
            // cadence; idempotent on the stable per-session summary id.
            if let Err(e) =
                crate::summary::generate_session_summary(&self.kb, session_id, wall_now, None).await
            {
                tracing::warn!(
                    session_id,
                    error = %e,
                    "session summary generation failed — continuing",
                );
            }
        }

        if !closed.is_empty() {
            tracing::info!(
                closed = closed.len(),
                "session maintenance: closed and summarised inactive sessions",
            );
        }
    }

    /// Run F50 memory decay for one agent: prune Episodes whose
    /// age-decayed importance has fallen at or below the configured
    /// `prune_below` threshold.
    ///
    /// Shares the cortex cadence gate (runs every `cortex_every_n`
    /// cycles). Decay is computed from the immutable stored `importance`,
    /// so the prune is monotonic and safe to repeat. Failures are logged
    /// and dropped — decay must never destabilize consolidation.
    async fn run_decay_sweep(&mut self, agent_id: &str) {
        let cfg = self.kb.config();
        let half_life_days = cfg.half_life_days;
        let prune_below = cfg.prune_below;
        // Consumer for the `relevance_decay` stdlib rule: runs the rule (single
        // decay formula) and deletes the Episodes it yields.
        match crate::rules::consume_relevance_decay(&self.kb, agent_id, half_life_days, prune_below)
            .await
        {
            Ok(pruned) => {
                if pruned > 0 {
                    tracing::info!(
                        agent = %agent_id,
                        pruned,
                        "memory decay pruned stale episodes",
                    );
                }
            }
            Err(e) => tracing::warn!(
                agent = %agent_id,
                error = %e,
                "memory decay sweep failed — continuing",
            ),
        }
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

        metrics::emit_procedure_cycle(agent_id);
        let start = Instant::now();
        match promote_procedures_once(&self.kb, agent_id, LifecycleConfig::default()).await {
            Ok(report) => {
                let elapsed_ms = record_duration(start, metrics::emit_procedure_cycle_duration);
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
                let elapsed_ms = record_duration(start, metrics::emit_procedure_cycle_duration);
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

        metrics::emit_topic_cycle(agent_id);
        let start = Instant::now();
        match detect_topics_once(&self.kb, TopicConfig::default()).await {
            Ok(report) => {
                let elapsed_ms = record_duration(start, metrics::emit_topic_cycle_duration);
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
                let elapsed_ms = record_duration(start, metrics::emit_topic_cycle_duration);
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

/// Emit a duration metric and return the measured milliseconds.
///
/// Centralises the `start.elapsed().as_millis() as f64` + metric-emit
/// boilerplate used by every cycle in this module.
fn record_duration(start: Instant, emit: fn(f64)) -> f64 {
    let elapsed_ms = start.elapsed().as_millis() as f64;
    emit(elapsed_ms);
    elapsed_ms
}
