//! Ingest pipeline worker with biased select, semaphore concurrency,
//! and per-item step chain execution.

use std::sync::{Arc, Mutex};

use tokio::sync::{Semaphore, mpsc};
use tokio_util::sync::CancellationToken;

use uniko_pipes::circuit_breaker::CircuitBreaker;
use uniko_pipes::dead_letter::DeadLetterQueue;
use uniko_pipes::health::HealthTracker;
use uniko_pipes::metrics;
use uniko_pipes::step::{PipelineContext, Step};
use uniko_pipes::types::{IngestTask, ItemResult, StepErrorPolicy, StepOutcome};
use uniko_store::KnowledgeBase;

/// Long-running ingest worker receiving tasks via a bounded channel.
pub(crate) struct IngestWorker {
    rx: mpsc::Receiver<IngestTask>,
    steps: Arc<Vec<Box<dyn Step>>>,
    semaphore: Arc<Semaphore>,
    cancel: CancellationToken,
    kb: Arc<KnowledgeBase>,
    llm_breaker: Arc<CircuitBreaker>,
    dlq: Arc<DeadLetterQueue>,
    health: Arc<Mutex<HealthTracker>>,
    dlq_max_retries: u32,
}

impl IngestWorker {
    #[expect(
        clippy::too_many_arguments,
        reason = "constructor assembles all worker dependencies"
    )]
    pub(crate) fn new(
        rx: mpsc::Receiver<IngestTask>,
        steps: Vec<Box<dyn Step>>,
        concurrency: usize,
        cancel: CancellationToken,
        kb: Arc<KnowledgeBase>,
        llm_breaker: Arc<CircuitBreaker>,
        dlq: Arc<DeadLetterQueue>,
        health: Arc<Mutex<HealthTracker>>,
        dlq_max_retries: u32,
    ) -> Self {
        Self {
            rx,
            steps: Arc::new(steps),
            semaphore: Arc::new(Semaphore::new(concurrency)),
            cancel,
            kb,
            llm_breaker,
            dlq,
            health,
            dlq_max_retries,
        }
    }

    /// Run the worker loop until cancelled or the channel closes.
    pub(crate) async fn run(mut self) {
        tracing::info!("ingest worker started");
        loop {
            tokio::select! {
                biased;

                () = self.cancel.cancelled() => {
                    tracing::info!("ingest worker shutting down");
                    break;
                }

                task = self.rx.recv() => {
                    match task {
                        Some(task) => self.process(task),
                        None => {
                            tracing::info!("ingest channel closed");
                            break;
                        }
                    }
                }
            }
        }
        tracing::info!("ingest worker stopped");
    }

    /// Spawn a concurrent task for one ingest item.
    fn process(&self, task: IngestTask) {
        let permit = self.semaphore.clone();
        let steps = self.steps.clone();
        let item_cancel = self.cancel.child_token();
        let kb = self.kb.clone();
        let breaker = self.llm_breaker.clone();
        let dlq = self.dlq.clone();
        let health = self.health.clone();
        let dlq_max = self.dlq_max_retries;

        tokio::spawn(async move {
            // Acquire semaphore permit (bounds concurrency).  An `Err`
            // here means the semaphore was closed during shutdown —
            // nothing to do, drop the task.
            let Ok(_permit) = permit.acquire().await else {
                return;
            };

            let start = std::time::Instant::now();

            // Extract content from the task.
            let (content, content_type) = match &task {
                IngestTask::Message(m) => (m.content.clone(), m.content_type.clone()),
                IngestTask::Artifact(a) => (a.content.clone(), a.kind.clone()),
                // PDF ingest is routed through `uniko_extract::ingest_pdf`
                // directly; it never lands on the generic worker step
                // chain because `content` is binary (not UTF-8 String)
                // and chunker selection is handled internally.
                IngestTask::Pdf(p) => (String::new(), format!("pdf:{}", p.artifact_id)),
            };

            let mut ctx = PipelineContext::new(
                0, // Node ID set by the first step (P1 ingest).
                content,
                content_type,
                item_cancel,
                kb,
                breaker,
            );

            metrics::emit_ingest_item();
            let result = run_step_chain(&steps, &mut ctx, &dlq, dlq_max).await;
            let elapsed_ms = start.elapsed().as_millis() as f64;

            // Update health tracker.
            let mut tracker = health.lock().unwrap();
            if result.steps_failed.is_empty() {
                tracker.record_success(elapsed_ms);
            } else {
                tracker.record_failure();
                for (step_name, _) in &result.steps_failed {
                    metrics::emit_ingest_failure(step_name);
                }
            }
            metrics::emit_ingest_duration(elapsed_ms);
        });
    }
}

/// Execute a chain of steps with per-item error isolation.
///
/// Each step's failure is handled according to its
/// [`StepErrorPolicy`].  The chain continues unless `Abort` is
/// declared or the cancellation token fires.
pub(crate) async fn run_step_chain(
    steps: &[Box<dyn Step>],
    ctx: &mut PipelineContext,
    dlq: &DeadLetterQueue,
    dlq_max_retries: u32,
) -> ItemResult {
    let mut result = ItemResult::new(ctx.node_id);

    for step in steps {
        if ctx.cancel.is_cancelled() {
            break;
        }
        if !step.should_run(ctx) {
            result.steps_skipped.push(step.name().to_owned());
            continue;
        }

        let step_name = step.name().to_owned();
        tracing::debug!(step = %step_name, "executing step");
        let (error, policy) = match step.execute(ctx).await {
            Ok(StepOutcome::Completed) => {
                tracing::debug!("step completed");
                result.steps_succeeded.push(step_name);
                continue;
            }
            Ok(StepOutcome::Skipped { reason }) => {
                tracing::debug!(reason = %reason, "step skipped");
                result.steps_skipped.push(step_name);
                continue;
            }
            Ok(StepOutcome::Failed { error, policy }) => (error, policy),
            // Transport-level failure (Step::execute returned Err). No
            // StepOutcome to consult — default to DeadLetter so the item
            // is preserved for inspection / retry.
            Err(e) => (e.to_string(), StepErrorPolicy::DeadLetter),
        };
        if handle_failure(
            &step_name,
            &error,
            policy,
            ctx,
            dlq,
            dlq_max_retries,
            &mut result,
        )
        .await
        {
            break;
        }
    }
    result
}

/// Handle a step failure. Returns `true` if the chain should abort.
async fn handle_failure(
    step_name: &str,
    error: &str,
    policy: StepErrorPolicy,
    ctx: &PipelineContext,
    dlq: &DeadLetterQueue,
    dlq_max_retries: u32,
    result: &mut ItemResult,
) -> bool {
    tracing::warn!(error = %error, policy = ?policy, "step failed");
    result
        .steps_failed
        .push((step_name.to_owned(), error.to_owned()));

    match policy {
        StepErrorPolicy::Skip => false,
        StepErrorPolicy::DeadLetter => {
            if let Err(e) = dlq
                .store(step_name, error, ctx.node_id, dlq_max_retries)
                .await
            {
                tracing::error!(error = %e, "failed to store dead letter");
            }
            false
        }
        StepErrorPolicy::Abort => true,
    }
}
