//! Pipeline orchestration: system lifecycle, task routing, health.
//!
//! [`PipelineSystem`] owns all workers, channels, and the circuit breaker.
//! It is the single entry point for submitting ingest and consolidation
//! tasks.

// Rust guideline compliant

pub(crate) mod consolidation_worker;
pub(crate) mod ingest_worker;

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use uniko_pipes::cancel::ShutdownCoordinator;
use uniko_pipes::circuit_breaker::{CircuitBreaker, CircuitState};
use uniko_pipes::config::PipelineConfig;
use uniko_pipes::dead_letter::DeadLetterQueue;
use uniko_pipes::health::{HealthTracker, PipelineHealth};
use uniko_pipes::metrics::register_pipeline_metrics;
use uniko_pipes::step::Step;
use uniko_pipes::types::{ConsolidationTask, IngestTask};
use uniko_store::KnowledgeBase;

use self::consolidation_worker::ConsolidationWorker;
use self::ingest_worker::IngestWorker;

/// Root orchestrator for all pipeline workers.
///
/// Owns bounded channels, the LLM circuit breaker, and worker handles.
/// Submitting tasks is non-blocking (bounded channel with backpressure).
///
/// # Examples
///
/// ```no_run
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// use std::sync::Arc;
/// use uniko_pipes::PipelineConfig;
/// use uniko_store::{KnowledgeBase, config::UnikoConfig};
/// use uniko_memory::pipeline::PipelineSystem;
///
/// let kb = Arc::new(KnowledgeBase::in_memory(UnikoConfig::default()).await?);
/// let ps = PipelineSystem::new(PipelineConfig::default(), kb, vec![]);
/// // ... submit tasks ...
/// ps.shutdown().await?;
/// # Ok(())
/// # }
/// ```
pub struct PipelineSystem {
    shutdown: ShutdownCoordinator,
    ingest_tx: mpsc::Sender<IngestTask>,
    consolidation_tx: mpsc::Sender<ConsolidationTask>,
    llm_breaker: Arc<CircuitBreaker>,
    workers: Mutex<Vec<JoinHandle<()>>>,
    ingest_health: Arc<Mutex<HealthTracker>>,
    consolidation_health: Arc<Mutex<HealthTracker>>,
    started_at: Instant,
    config: PipelineConfig,
}

impl PipelineSystem {
    /// Create the system, spawn workers, and begin processing.
    ///
    /// Workers start their `select!` loops immediately.
    pub fn new(
        config: PipelineConfig,
        kb: Arc<KnowledgeBase>,
        ingest_steps: Vec<Box<dyn Step>>,
    ) -> Self {
        register_pipeline_metrics();

        let shutdown = ShutdownCoordinator::new();
        let llm_breaker = Arc::new(CircuitBreaker::new(
            config.circuit_failure_threshold,
            config.circuit_recovery_ms,
        ));
        let dlq = Arc::new(DeadLetterQueue::new(kb.clone()));

        // Bounded channels with backpressure.
        let (ingest_tx, ingest_rx) = mpsc::channel(config.ingest_queue_capacity);
        let (consolidation_tx, consolidation_rx) =
            mpsc::channel(config.consolidation_queue_capacity);

        let ingest_health = Arc::new(Mutex::new(HealthTracker::new()));
        let consolidation_health = Arc::new(Mutex::new(HealthTracker::new()));

        // Spawn ingest worker.
        let ingest_worker = IngestWorker::new(
            ingest_rx,
            ingest_steps,
            config.ingest_concurrency,
            shutdown.ingest_token(),
            kb.clone(),
            llm_breaker.clone(),
            dlq.clone(),
            ingest_health.clone(),
            consolidation_tx.clone(),
            config.dead_letter_max_retries,
        );
        let ingest_handle = tokio::spawn(ingest_worker.run());

        // Spawn consolidation worker.
        let consolidation_worker = ConsolidationWorker::new(
            consolidation_rx,
            config.consolidation_concurrency,
            shutdown.consolidation_token(),
            kb,
            consolidation_health.clone(),
            &config,
        );
        let consolidation_handle = tokio::spawn(consolidation_worker.run());

        Self {
            shutdown,
            ingest_tx,
            consolidation_tx,
            llm_breaker,
            workers: Mutex::new(vec![ingest_handle, consolidation_handle]),
            ingest_health,
            consolidation_health,
            started_at: Instant::now(),
            config,
        }
    }

    /// Submit an ingest task.
    ///
    /// Returns `Err` if the channel is full (backpressure) or the system
    /// is shutting down.
    ///
    /// # Errors
    ///
    /// Returns [`uniko_store::UnikoError::Pipeline`] on channel failure.
    pub fn submit_ingest(&self, task: IngestTask) -> Result<(), uniko_store::UnikoError> {
        self.ingest_tx
            .try_send(task)
            .map_err(|e| uniko_store::UnikoError::Pipeline(format!("ingest channel: {e}")))
    }

    /// Submit a consolidation task.
    ///
    /// # Errors
    ///
    /// Returns [`uniko_store::UnikoError::Pipeline`] on channel failure.
    pub fn submit_consolidation(
        &self,
        task: ConsolidationTask,
    ) -> Result<(), uniko_store::UnikoError> {
        self.consolidation_tx
            .try_send(task)
            .map_err(|e| uniko_store::UnikoError::Pipeline(format!("consolidation channel: {e}")))
    }

    /// Query aggregate pipeline health.
    pub fn health(&self) -> PipelineHealth {
        let circuit_open = self.llm_breaker.state() == CircuitState::Open;

        let ingest_depth = self.ingest_tx.max_capacity() - self.ingest_tx.capacity();
        let ingest = self.ingest_health.lock().unwrap().health(
            ingest_depth,
            self.ingest_tx.max_capacity(),
            circuit_open,
        );

        let consol_depth = self.consolidation_tx.max_capacity() - self.consolidation_tx.capacity();
        let consolidation = self.consolidation_health.lock().unwrap().health(
            consol_depth,
            self.consolidation_tx.max_capacity(),
            circuit_open,
        );

        PipelineHealth {
            ingest,
            consolidation,
            llm_circuit: self.llm_breaker.state(),
            uptime_secs: self.started_at.elapsed().as_secs(),
        }
    }

    /// Graceful shutdown: drain workers within the configured timeout.
    ///
    /// # Errors
    ///
    /// Returns an error string if shutdown times out.
    pub async fn shutdown(self) -> Result<(), String> {
        let timeout = Duration::from_secs(self.config.shutdown_timeout_secs);
        let mut workers = self.workers.into_inner().unwrap();
        self.shutdown
            .shutdown(&mut workers, timeout)
            .await
            .map_err(|e| e.to_string())
    }
}

impl std::fmt::Debug for PipelineSystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PipelineSystem")
            .field("uptime_secs", &self.started_at.elapsed().as_secs())
            .finish_non_exhaustive()
    }
}
