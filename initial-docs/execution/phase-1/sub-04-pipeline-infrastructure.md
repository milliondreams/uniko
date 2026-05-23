# Sub-Phase 4: Pipeline Infrastructure

## Context

This phase builds the pipeline management framework that all 8 pipelines (P1-P8) will plug into. The infrastructure is split across two crates: generic pipeline machinery lives in `uniko-pipes` (accessible to all layers), while orchestration -- PipelineSystem, workers, and step chain composition -- lives in `uniko-memory` (where it can wire together extraction steps from `uniko-extract` with consolidation logic). By placing the `Step` trait, circuit breaker, retry policy, cancellation tokens, dead-letter queue, health, and metrics in `uniko-pipes`, any crate in the workspace can depend on pipeline primitives without pulling in memory-layer concerns. The `uniko-memory` crate imports these primitives and composes them into the actual worker loops and pipeline orchestration.

The goal is a fully operational execution engine -- PipelineSystem, IngestWorker, ConsolidationWorker, channels, circuit breaker, retry, cancellation, backpressure, dead-letter queue, and observability -- with zero pipeline logic. By the end of Phase 4, the framework accepts tasks, routes them to workers, handles errors, and shuts down gracefully. Individual pipeline steps (P1-P8) are implemented in subsequent phases and register themselves via the `Step` trait defined here.

The design follows seven principles from `pipeline-management.md`:

1. Channel-based actors, not DAG frameworks. The topology is dynamic and reactive.
2. Per-item error isolation. One bad message never kills a pipeline.
3. Circuit breaker for external dependencies (LLM provider).
4. Interactive queries preempt background work via biased `select!`.
5. Bounded channels everywhere. Backpressure propagates to callers.
6. Cancellation from root to leaf via `CancellationToken` hierarchy.
7. Observability built in. Every pipeline emits metrics and structured traces.

## Prerequisites

| Dependency | Status Required | What It Provides |
|---|---|---|
| Phase 2: Schema Types | Complete | All node/edge type definitions (`Message`, `Entity`, `Fact`, `Observation`, `DeadLetter`, etc.) used in task structs and results |
| Phase 3: KnowledgeBase (L1) | Can proceed in parallel | `Arc<KnowledgeBase>` for graph operations; pipeline infra only needs the type signature, not the implementation |
| `tokio` 1.x | Available | Async runtime, `mpsc`, `select!`, `JoinHandle`, `Semaphore` |
| `tokio_util` | Available | `CancellationToken`, `DropGuard` |
| `metrics` crate | Available | Counter, Gauge, Histogram registration |
| `tracing` crate | Available | `instrument`, `info_span`, structured logging |

## Sub-phases

---

### 4.1 -- PipelineSystem & Worker Architecture

**Objective:** Create the root `PipelineSystem` struct that owns all workers, channels, and the circuit breaker. Implement `IngestWorker` and `ConsolidationWorker` as long-running tokio tasks receiving work via bounded mpsc channels.

#### Files in `uniko-pipes`

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-pipes/src/types.rs` | New | `IngestTask`, `ConsolidationTask`, `ObservationsReady`, `ForceConsolidate`, `StepOutcome` |
| `crates/uniko-pipes/src/config.rs` | New | `PipelineConfig` struct with defaults |

#### Files in `uniko-memory`

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-memory/src/pipeline/mod.rs` | New module root | `PipelineSystem` struct, re-exports |
| `crates/uniko-memory/src/pipeline/ingest_worker.rs` | New | `IngestWorker` struct and run loop |
| `crates/uniko-memory/src/pipeline/consolidation_worker.rs` | New | `ConsolidationWorker` struct and run loop |

#### Structs and Functions

**`PipelineSystem`** (`crates/uniko-memory/src/pipeline/mod.rs`):
```rust
pub struct PipelineSystem {
    cancel: CancellationToken,
    ingest_tx: mpsc::Sender<IngestTask>,
    consolidation_tx: mpsc::Sender<ConsolidationTask>,
    llm_breaker: Arc<CircuitBreaker>,
    workers: Vec<JoinHandle<()>>,
    started_at: Instant,
}
```

- `PipelineSystem::new(config: PipelineConfig, kb: Arc<KnowledgeBase>) -> Self` -- Creates channels, spawns workers, returns the system. Workers begin their select loops immediately.
- `PipelineSystem::submit_ingest(&self, task: IngestTask) -> Result<()>` -- Sends to ingest channel. Returns `Err` if channel full (backpressure) or system shutting down.
- `PipelineSystem::submit_consolidation(&self, task: ConsolidationTask) -> Result<()>` -- Sends to consolidation channel.
- `PipelineSystem::health(&self) -> PipelineHealth` -- Queries both workers for current health status.
- `PipelineSystem::shutdown(&self, timeout: Duration) -> Result<()>` -- Graceful shutdown sequence (see 4.5).

**`PipelineConfig`** (`crates/uniko-pipes/src/config.rs`):
```rust
pub struct PipelineConfig {
    pub ingest_queue_capacity: usize,          // default: 200
    pub ingest_concurrency: usize,             // default: 8
    pub consolidation_queue_capacity: usize,    // default: 32
    pub consolidation_concurrency: usize,       // default: 4
    pub consolidation_threshold: u32,           // default: 20 observations
    pub consolidation_interval_secs: u64,       // default: 900 (15 min)
    pub retry: RetryPolicy,
    pub circuit_failure_threshold: u32,         // default: 5
    pub circuit_recovery_ms: u64,              // default: 60_000
    pub dead_letter_max_retries: u32,          // default: 3
    pub dead_letter_check_interval_secs: u64,  // default: 300 (5 min)
    pub shutdown_timeout_secs: u64,            // default: 30
}
```

**`IngestWorker`** (`crates/uniko-memory/src/pipeline/ingest_worker.rs`):
- Receives `IngestTask` via `mpsc::Receiver<IngestTask>` (capacity 200).
- Uses `Semaphore(8)` for concurrency limiting.
- Main loop uses `tokio::select! { biased; }` with ordering:
  1. `cancel.cancelled()` -- shutdown signal (highest priority)
  2. `interactive_rx.recv()` -- interactive queries preempt background
  3. `rx.recv()` -- ingest tasks (background)
- For each task: acquire semaphore permit, spawn processing through the step chain, release on completion.
- Tracks `WorkerHealth` internally (items_processed, items_failed, queue_depth, last_processed_at, avg_latency_ms via exponential moving average).

**`ConsolidationWorker`** (`crates/uniko-memory/src/pipeline/consolidation_worker.rs`):
- Receives `ConsolidationTask` via `mpsc::Receiver<ConsolidationTask>` (capacity 32).
- Uses `Semaphore(4)` for concurrency limiting.
- Trigger logic: maintains per-agent observation counters. Triggers consolidation when:
  - Counter >= 20 observations for an agent (threshold trigger), OR
  - 15-minute timer fires with counter > 0 (timer trigger), OR
  - `ForceConsolidate { agent_id }` received (manual trigger)
- Main loop uses `tokio::select! { biased; }`:
  1. `cancel.cancelled()` -- shutdown
  2. `rx.recv()` -- consolidation tasks / notifications
  3. `timer.tick()` -- periodic trigger (lowest priority, only fires when nothing else ready)

**Task types** (`crates/uniko-pipes/src/types.rs`):
```rust
pub enum IngestTask {
    Message(IngestMessage),
    Artifact(IngestArtifact),
}

pub enum ConsolidationTask {
    ObservationsReady(ObservationsReady),
    ForceConsolidate { agent_id: String },
    RunCycle { agent_id: String },
}

pub struct ObservationsReady {
    pub agent_id: String,
    pub observation_count: u32,
    pub source_node_ids: Vec<NodeId>,
}

pub enum StepOutcome {
    Completed,
    Skipped { reason: String },
    Failed { error: String, policy: StepErrorPolicy },
}
```

---

### 4.2 -- Step Trait & Error Isolation

**Objective:** Define the `Step` trait that all pipeline steps implement. Establish per-item error isolation so that failure in one item never affects another.

#### Files

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-pipes/src/step.rs` | New | `Step` trait, `StepErrorPolicy`, `ItemResult`, `PipelineContext` |

#### Trait Definition

```rust
#[async_trait]
pub trait Step: Send + Sync {
    /// Human-readable name for logging and metrics.
    fn name(&self) -> &str;

    /// Whether this step should execute for the given context.
    /// Returns false to skip (e.g., code NER skips non-code content).
    fn should_run(&self, ctx: &PipelineContext) -> bool;

    /// Execute the step. Mutates context with results.
    async fn execute(&self, ctx: &mut PipelineContext) -> Result<StepOutcome>;

    /// What happens when this step fails.
    fn error_policy(&self) -> StepErrorPolicy;
}
```

**`StepErrorPolicy`:**
```rust
pub enum StepErrorPolicy {
    /// Log warning, continue to next step. Used by NER (P2), Observations (P3).
    Skip,
    /// Store in dead-letter queue for later retry, continue. Used by Embedding (P7).
    DeadLetter,
    /// Stop remaining steps for this item. Used for critical failures.
    Abort,
}
```

**`ItemResult`:**
```rust
pub struct ItemResult {
    pub node_id: NodeId,
    pub steps_succeeded: Vec<String>,
    pub steps_failed: Vec<(String, String)>,  // (step_name, error_message)
    pub steps_skipped: Vec<String>,
}
```

**`PipelineContext`:**
```rust
pub struct PipelineContext {
    pub node_id: NodeId,
    pub content: String,
    pub content_type: String,
    pub cancel: CancellationToken,
    pub kb: Arc<KnowledgeBase>,
    pub llm_breaker: Arc<CircuitBreaker>,
    pub extracted_entities: Vec<EntityId>,
    pub extracted_observations: Vec<ObservationId>,
    pub metadata: HashMap<String, Value>,
}
```

**Step chain executor** (in `crates/uniko-memory/src/pipeline/ingest_worker.rs`):

The `run_step_chain` function lives in `uniko-memory` alongside the workers. It takes `&[Box<dyn Step>]` where `Step` is the trait from `uniko-pipes`:

```rust
async fn run_step_chain(steps: &[Box<dyn Step>], ctx: &mut PipelineContext) -> ItemResult {
    let mut result = ItemResult::new(ctx.node_id);
    for step in steps {
        if ctx.cancel.is_cancelled() { break; }
        if !step.should_run(ctx) {
            result.steps_skipped.push(step.name().to_string());
            continue;
        }
        match step.execute(ctx).await {
            Ok(StepOutcome::Completed) => result.steps_succeeded.push(step.name().to_string()),
            Ok(StepOutcome::Skipped { reason }) => result.steps_skipped.push(step.name().to_string()),
            Ok(StepOutcome::Failed { error, policy }) | Err(error) => {
                let policy = step.error_policy();
                result.steps_failed.push((step.name().to_string(), error.to_string()));
                match policy {
                    StepErrorPolicy::Skip => continue,
                    StepErrorPolicy::DeadLetter => { /* queue to DLQ, continue */ },
                    StepErrorPolicy::Abort => break,
                }
            }
        }
    }
    result
}
```

Per-item isolation guarantee: each item gets its own `PipelineContext` and its own `run_step_chain` invocation. A panic in one task (caught by tokio's `JoinHandle`) does not propagate to others. The semaphore ensures bounded concurrency, not shared state.

---

### 4.3 -- Circuit Breaker

**Objective:** Implement a lock-free circuit breaker that wraps all LLM calls. When the LLM provider fails repeatedly, the breaker opens and all LLM-dependent steps fall back to local alternatives.

#### Files

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-pipes/src/circuit_breaker.rs` | New | `CircuitBreaker` struct, `CircuitState` enum |

#### Struct Definition

```rust
pub struct CircuitBreaker {
    state: AtomicU8,            // Closed=0, Open=1, HalfOpen=2
    failure_count: AtomicU32,
    last_failure_ms: AtomicU64, // millis since UNIX epoch
    failure_threshold: u32,     // default: 5 consecutive failures
    recovery_ms: u64,           // default: 60_000 (1 minute)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}
```

#### Functions

- `CircuitBreaker::new(failure_threshold: u32, recovery_ms: u64) -> Self`
- `CircuitBreaker::state(&self) -> CircuitState` -- Read current state atomically.
- `CircuitBreaker::is_available(&self) -> bool` -- Returns `true` if Closed or HalfOpen (probe allowed).
- `async fn call<F, Fut, T>(&self, f: F) -> Result<T>` where `F: FnOnce() -> Fut, Fut: Future<Output = Result<T>>` -- Core method:
  - If **Closed**: execute `f()`. On success, reset failure count. On failure, increment. If failure_count >= threshold, transition to Open.
  - If **Open**: check if `recovery_ms` has elapsed since `last_failure_ms`. If yes, transition to HalfOpen and execute probe. If no, return `Err(CircuitOpen)` immediately.
  - If **HalfOpen**: execute `f()` as probe. On success, transition to Closed, reset. On failure, transition back to Open.

#### State Transition Logging

```
WARN  circuit breaker OPEN -- LLM calls disabled for 60s (5 consecutive failures)
INFO  circuit breaker HALF-OPEN -- probing LLM availability
INFO  circuit breaker CLOSED -- LLM available, resuming normal operation
```

#### Fallback Behavior When Open

Each pipeline step checks `ctx.llm_breaker.is_available()` before making LLM calls. When unavailable:

| Step | Fallback |
|---|---|
| P2 NER (LLM enhancement) | Skip LLM, use rule-based + ONNX only |
| P3 Observations (LLM path) | Skip LLM, use rule-based extraction only |
| P7d Summaries | Skip entirely |
| P8 Rule Induction (GENERATE) | Skip entirely |
| NL-to-Cypher | Return error to caller |

---

### 4.4 -- Retry with Exponential Backoff

**Objective:** Implement a reusable retry policy with exponential backoff, jitter, and cancellation awareness for all LLM-dependent operations.

#### Files

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-pipes/src/retry.rs` | New | `RetryPolicy` struct, `retry_with_policy` function |

#### Struct Definition

```rust
pub struct RetryPolicy {
    pub max_attempts: u32,         // default: 3
    pub initial_delay_ms: u64,     // default: 500
    pub max_delay_ms: u64,         // default: 30_000
    pub backoff_multiplier: f64,   // default: 2.0
}
```

#### Functions

- `RetryPolicy::default() -> Self` -- Returns defaults (3 attempts, 500ms, 30s, 2.0x).
- `async fn retry_with_policy<F, Fut, T>(policy: &RetryPolicy, cancel: &CancellationToken, f: F) -> Result<T>` where `F: Fn() -> Fut, Fut: Future<Output = Result<T>>`:
  - Attempts up to `max_attempts` times.
  - On failure: compute delay = `initial_delay_ms * multiplier^(attempt - 1)`, capped at `max_delay_ms`.
  - Apply jitter: random +/-25% on the computed delay (uniform distribution).
  - Wait using `tokio::select!` between `tokio::time::sleep(delay)` and `cancel.cancelled()`. If cancelled, abort immediately with `Err(Cancelled)`.
  - On final failure: return the last error.

#### Delay Schedule (defaults, no jitter)

| Attempt | Delay |
|---|---|
| 1 | 0ms (immediate) |
| 2 | 500ms |
| 3 | 1000ms |

With jitter (+/-25%): attempt 2 = 375-625ms, attempt 3 = 750-1250ms.

---

### 4.5 -- CancellationToken Hierarchy

**Objective:** Establish a tree of `CancellationToken`s that enables orderly shutdown from root to leaf, with timeout stages.

#### Files

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-pipes/src/cancel.rs` | New | Token hierarchy construction, `GracefulShutdown` logic |

#### Token Hierarchy

```
root_cancel (PipelineSystem)
  +-- ingest_cancel (IngestWorker)
  |     +-- item_cancel_1 (per-item, for LLM timeouts)
  |     +-- item_cancel_2
  |     +-- ...
  +-- consolidation_cancel (ConsolidationWorker)
  |     +-- cycle_cancel_1 (per-cycle, for long operations)
  |     +-- cycle_cancel_2
  |     +-- ...
  +-- planning_cancel (future: MCTS tree search)
```

All tokens use `tokio_util::sync::CancellationToken`. Child tokens are created via `parent.child_token()`. Cancelling a parent cancels all children. Cancelling a child only affects that subtree.

#### Graceful Shutdown Sequence

`PipelineSystem::shutdown(timeout: Duration)`:

```
1. Cancel ingest_cancel
   -> IngestWorker stops accepting new tasks
   -> In-flight items continue
   Wait up to 5s for in-flight items to complete

2. Cancel consolidation_cancel
   -> ConsolidationWorker stops accepting new tasks
   -> Current cycle continues (partial results committed)
   Wait up to 10s for current cycle to complete

3. Cancel root_cancel
   -> Force-stop anything remaining

4. Join all worker JoinHandles with 30s total timeout

5. Log final health metrics
   -> items_processed, items_failed, avg_latency for each worker
```

#### Per-Item Cancellation

Each item processed by IngestWorker gets a child token from `ingest_cancel`. This enables:
- LLM call timeouts: `tokio::select!` between LLM future and `item_cancel.cancelled()`.
- Individual item abort without affecting other items.
- Automatic cancellation when the worker shuts down.

---

### 4.6 -- Dead-Letter Queue

**Objective:** Implement persistent storage of failed pipeline tasks in the graph as `DeadLetter` nodes, with automatic retry and manual management operations.

#### Files

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-pipes/src/dead_letter.rs` | New | `DeadLetterQueue` struct, CRUD operations, background retry task |

#### DeadLetter Node Schema (from schema-v3)

```
DeadLetter
  step              String    -- "entity_extractor", "summarizer", etc.
  error             String    -- error message
  node_ref          Int64     -- the node that failed processing
  retry_count       Int64     -- how many times retried so far (starts at 0)
  max_retries       Int64     -- default: 3
  next_retry_at     DateTime  -- computed from backoff
  created_at        DateTime
```

#### Functions

- `DeadLetterQueue::new(kb: Arc<KnowledgeBase>, ingest_tx: mpsc::Sender<IngestTask>) -> Self`
- `async fn store(&self, step: &str, error: &str, node_ref: NodeId, max_retries: u32) -> Result<NodeId>` -- Creates a DeadLetter node in the graph. Sets `next_retry_at = created_at + initial_delay`.
- `async fn retry(&self, dead_letter_id: NodeId) -> Result<()>` -- Resubmits the referenced node to the appropriate pipeline. Increments `retry_count`. Computes `next_retry_at = created_at + initial_delay * multiplier^retry_count`. If `retry_count >= max_retries`, marks as permanently failed (no further auto-retry).
- `async fn retry_all_pending(&self) -> Result<u32>` -- Resubmits all items where `next_retry_at < now()` and `retry_count < max_retries`. Returns count of items resubmitted.
- `async fn clear(&self, dead_letter_id: NodeId) -> Result<()>` -- Deletes the DeadLetter node (permanently give up).
- `async fn clear_all(&self) -> Result<u32>` -- Deletes all DeadLetter nodes. Returns count deleted.
- `async fn list_pending(&self) -> Result<Vec<DeadLetterInfo>>` -- Returns all pending items with their metadata.

#### Automatic Retry Background Task

Spawned by `PipelineSystem::new()`:
- Runs every 5 minutes (configurable via `dead_letter_check_interval_secs`).
- Queries: `MATCH (dl:DeadLetter) WHERE dl.next_retry_at < datetime() AND dl.retry_count < dl.max_retries RETURN dl`.
- For each result: call `retry(dl.id)`.
- Respects cancellation: exits on `cancel.cancelled()`.

#### Backoff Computation

```
next_retry_at = created_at + initial_delay * multiplier^retry_count
```

With defaults (500ms initial, 2.0 multiplier):
- Retry 1: +500ms
- Retry 2: +1000ms
- Retry 3: +2000ms (final attempt, then permanent failure)

---

### 4.7 -- Backpressure & Queue Monitoring

**Objective:** Monitor queue depths, classify worker health status, and expose health information for diagnostics and alerting.

#### Files

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-pipes/src/health.rs` | New | `WorkerHealth`, `WorkerStatus`, `PipelineHealth`, health computation logic |

#### Structs

```rust
#[derive(Debug, Serialize)]
pub struct WorkerHealth {
    pub status: WorkerStatus,
    pub queue_depth: usize,
    pub queue_capacity: usize,
    pub items_processed: u64,
    pub items_failed: u64,
    pub last_processed_at: Option<DateTime<Utc>>,
    pub avg_latency_ms: f64,
}

#[derive(Debug, Serialize, PartialEq, Eq)]
pub enum WorkerStatus {
    /// Processing normally. Queue depth < 60% capacity.
    Healthy,
    /// LLM circuit open, using fallbacks. Queue depth may be normal.
    Degraded,
    /// Queue depth > 80% capacity. Approaching saturation.
    Backpressured,
    /// No items processed in > 5 minutes. Possible deadlock or starvation.
    Stalled,
}

#[derive(Debug, Serialize)]
pub struct PipelineHealth {
    pub ingest: WorkerHealth,
    pub consolidation: WorkerHealth,
    pub llm_circuit: CircuitState,
    pub uptime_secs: u64,
}
```

#### Health Classification Thresholds

| Queue Depth (% of capacity) | Status | Action |
|---|---|---|
| < 60% | `Healthy` | Normal operation |
| 60-80% | `Healthy` (with log warning) | Log warning, continue |
| > 80% | `Backpressured` | Emit metric alert, status visible in health endpoint |
| 100% (full) | `Backpressured` | Sender blocks (backpressure propagates to caller) |

Additional status overrides:
- If LLM circuit is Open -> `Degraded` (regardless of queue depth)
- If `last_processed_at` is > 5 minutes ago and queue_depth > 0 -> `Stalled`

#### Average Latency Tracking

Use exponential moving average (EMA) with alpha = 0.1:
```
avg_latency = alpha * new_sample + (1 - alpha) * avg_latency
```

Tracked per-worker, updated after each item completes.

---

### 4.8 -- Observability (Metrics + Tracing)

**Objective:** Register all 14 pipeline metrics and instrument every step execution with structured tracing spans.

#### Files

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-pipes/src/metrics.rs` | New | Metric registration, `MetricsSnapshot` struct, metric emission helpers |

#### Metrics

All metrics use the `metrics` crate (Counter, Gauge, Histogram).

| Metric Name | Type | Labels | Description |
|---|---|---|---|
| `uniko.ingest.items_total` | Counter | pipeline=ingest | Total items submitted to ingest pipeline |
| `uniko.ingest.items_failed` | Counter | pipeline, step | Items that failed at a specific step |
| `uniko.ingest.duration_ms` | Histogram | pipeline | End-to-end ingest processing time per item |
| `uniko.ingest.queue_depth` | Gauge | pipeline | Current ingest channel occupancy |
| `uniko.consolidation.cycles_total` | Counter | agent_id | Total consolidation cycles executed |
| `uniko.consolidation.facts_derived` | Counter | agent_id | Facts created during consolidation |
| `uniko.consolidation.facts_invalidated` | Counter | agent_id | Facts invalidated during consolidation |
| `uniko.consolidation.duration_ms` | Histogram | agent_id | Consolidation cycle duration |
| `uniko.recall.phase1_only_pct` | Gauge | agent_id | Percentage of recalls satisfied by Phase 1 alone |
| `uniko.recall.assembly_ms` | Histogram | -- | Context bundle assembly latency |
| `uniko.llm.calls_total` | Counter | step | Total LLM API calls |
| `uniko.llm.errors_total` | Counter | step, error_type | LLM call failures |
| `uniko.llm.circuit_state` | Gauge | -- | Circuit breaker state (0=Closed, 1=Open, 2=HalfOpen) |
| `uniko.deadletter.pending` | Gauge | step | Current dead-letter queue depth per step |

#### Metric Registration

```rust
pub fn register_pipeline_metrics() {
    // Called once at PipelineSystem::new()
    // Registers all 14 metrics with the global recorder
    describe_counter!("uniko.ingest.items_total", "Total items ingested");
    // ... etc
}
```

#### Structured Tracing

Every step execution is wrapped in a tracing span:

```rust
#[instrument(
    skip(self, ctx),
    fields(node_id = %ctx.node_id, pipeline = "ingest")
)]
async fn process_item(&self, ctx: &mut PipelineContext) {
    for step in &self.steps {
        let _span = info_span!("pipeline_step", step = step.name()).entered();
        match step.execute(ctx).await {
            Ok(_) => info!("step completed"),
            Err(e) => warn!(error = %e, "step failed"),
        }
    }
}
```

Per-node tracing: all spans include `node_id` so a single message can be traced through all pipeline stages:
```
TRACE[node_id=42] pipeline=ingest step=entity_extractor -> 3 entities extracted
TRACE[node_id=42] pipeline=ingest step=observation_extractor -> 2 observations
TRACE[node_id=42] pipeline=consolidation agent=agent-1 -> fact derived
```

#### MetricsSnapshot

```rust
pub struct MetricsSnapshot {
    pub ingest_items_total: u64,
    pub ingest_items_failed: u64,
    pub ingest_avg_duration_ms: f64,
    pub consolidation_cycles_total: u64,
    pub consolidation_facts_derived: u64,
    pub consolidation_facts_invalidated: u64,
    pub llm_calls_total: u64,
    pub llm_errors_total: u64,
    pub llm_circuit_state: CircuitState,
    pub deadletter_pending: u64,
}
```

Export modes:
- Prometheus endpoint (for server mode)
- Structured logs (for embedded mode)
- `MetricsSnapshot` struct (for programmatic access)

---

## Test Plan

### Unit Tests

| Test | File | What It Validates |
|---|---|---|
| `test_pipeline_system_new` | `crates/uniko-memory/src/pipeline/mod.rs` | PipelineSystem creates successfully with default config |
| `test_pipeline_system_shutdown` | `crates/uniko-memory/src/pipeline/mod.rs` | Graceful shutdown completes within timeout, all workers joined |
| `test_pipeline_system_shutdown_timeout` | `crates/uniko-memory/src/pipeline/mod.rs` | Forced shutdown when workers exceed timeout |
| `test_ingest_worker_receives_task` | `crates/uniko-memory/src/pipeline/ingest_worker.rs` | Submit IngestTask via channel, worker processes it |
| `test_ingest_worker_concurrency_limit` | `crates/uniko-memory/src/pipeline/ingest_worker.rs` | Semaphore(8) limits concurrent processing |
| `test_ingest_worker_cancellation` | `crates/uniko-memory/src/pipeline/ingest_worker.rs` | Worker stops on cancellation token |
| `test_consolidation_worker_threshold_trigger` | `crates/uniko-memory/src/pipeline/consolidation_worker.rs` | Consolidation triggers at 20 observations |
| `test_consolidation_worker_timer_trigger` | `crates/uniko-memory/src/pipeline/consolidation_worker.rs` | Consolidation triggers on 15-min timer with pending observations |
| `test_consolidation_worker_force_trigger` | `crates/uniko-memory/src/pipeline/consolidation_worker.rs` | ForceConsolidate triggers immediate consolidation |
| `test_step_chain_success` | `crates/uniko-pipes/src/step.rs` | All steps succeed, ItemResult reflects all succeeded |
| `test_step_chain_skip_policy` | `crates/uniko-pipes/src/step.rs` | Step with Skip policy fails, next step still runs |
| `test_step_chain_deadletter_policy` | `crates/uniko-pipes/src/step.rs` | Step with DeadLetter policy fails, item queued to DLQ, next step runs |
| `test_step_chain_abort_policy` | `crates/uniko-pipes/src/step.rs` | Step with Abort policy fails, remaining steps skipped |
| `test_step_should_run_false` | `crates/uniko-pipes/src/step.rs` | Step with should_run=false is skipped |
| `test_circuit_breaker_closed` | `crates/uniko-pipes/src/circuit_breaker.rs` | Calls succeed in Closed state, failure count resets on success |
| `test_circuit_breaker_open_transition` | `crates/uniko-pipes/src/circuit_breaker.rs` | 5 consecutive failures transition to Open |
| `test_circuit_breaker_open_rejects` | `crates/uniko-pipes/src/circuit_breaker.rs` | Calls rejected immediately in Open state |
| `test_circuit_breaker_halfopen_probe` | `crates/uniko-pipes/src/circuit_breaker.rs` | After recovery_ms, transitions to HalfOpen, allows probe |
| `test_circuit_breaker_halfopen_success` | `crates/uniko-pipes/src/circuit_breaker.rs` | Successful probe transitions to Closed |
| `test_circuit_breaker_halfopen_failure` | `crates/uniko-pipes/src/circuit_breaker.rs` | Failed probe transitions back to Open |
| `test_circuit_breaker_concurrent_access` | `crates/uniko-pipes/src/circuit_breaker.rs` | No panics under concurrent calls from multiple tasks |
| `test_retry_success_first_attempt` | `crates/uniko-pipes/src/retry.rs` | Succeeds on first try, no delay |
| `test_retry_success_second_attempt` | `crates/uniko-pipes/src/retry.rs` | Fails once, succeeds on retry |
| `test_retry_exhausted` | `crates/uniko-pipes/src/retry.rs` | All attempts fail, returns last error |
| `test_retry_backoff_delays` | `crates/uniko-pipes/src/retry.rs` | Delays increase exponentially (500ms, 1000ms) |
| `test_retry_cancellation` | `crates/uniko-pipes/src/retry.rs` | Cancel token aborts retry loop immediately |
| `test_retry_jitter` | `crates/uniko-pipes/src/retry.rs` | Delays have +/-25% variance |
| `test_cancel_hierarchy_parent_cancels_children` | `crates/uniko-pipes/src/cancel.rs` | Root cancel cancels ingest + consolidation children |
| `test_cancel_hierarchy_child_independent` | `crates/uniko-pipes/src/cancel.rs` | Child cancel does not affect parent or siblings |
| `test_graceful_shutdown_sequence` | `crates/uniko-pipes/src/cancel.rs` | Ingest cancelled first (5s), then consolidation (10s), then root (30s) |
| `test_dead_letter_store` | `crates/uniko-pipes/src/dead_letter.rs` | Failed item creates DeadLetter node in graph |
| `test_dead_letter_retry` | `crates/uniko-pipes/src/dead_letter.rs` | Retry resubmits to pipeline, increments retry_count |
| `test_dead_letter_max_retries` | `crates/uniko-pipes/src/dead_letter.rs` | After max_retries, no further auto-retry |
| `test_dead_letter_retry_all_pending` | `crates/uniko-pipes/src/dead_letter.rs` | Bulk retry processes all eligible items |
| `test_dead_letter_clear` | `crates/uniko-pipes/src/dead_letter.rs` | Clear removes DeadLetter node |
| `test_dead_letter_list_pending` | `crates/uniko-pipes/src/dead_letter.rs` | List returns all pending items with correct metadata |
| `test_dead_letter_auto_retry_task` | `crates/uniko-pipes/src/dead_letter.rs` | Background task fires every 5 min and retries eligible items |
| `test_health_status_healthy` | `crates/uniko-pipes/src/health.rs` | Queue < 60% reports Healthy |
| `test_health_status_backpressured` | `crates/uniko-pipes/src/health.rs` | Queue > 80% reports Backpressured |
| `test_health_status_degraded` | `crates/uniko-pipes/src/health.rs` | Circuit open reports Degraded |
| `test_health_status_stalled` | `crates/uniko-pipes/src/health.rs` | No processing for 5 min with pending items reports Stalled |
| `test_health_avg_latency` | `crates/uniko-pipes/src/health.rs` | EMA updates correctly after processing items |
| `test_metrics_registration` | `crates/uniko-pipes/src/metrics.rs` | All 14 metrics registered without panic |
| `test_metrics_counter_increment` | `crates/uniko-pipes/src/metrics.rs` | Counters increment on task completion/failure |
| `test_metrics_gauge_queue_depth` | `crates/uniko-pipes/src/metrics.rs` | Queue depth gauge reflects actual channel occupancy |

### Integration Tests

| Test | What It Validates |
|---|---|
| `test_full_lifecycle` | Create PipelineSystem -> submit tasks -> verify processing -> shutdown -> verify clean exit |
| `test_backpressure_propagation` | Fill channel to 200, verify sender blocks, drain some, verify sender unblocks |
| `test_circuit_breaker_with_worker` | Worker processes tasks through circuit breaker, breaker opens, fallback executes |
| `test_concurrent_ingest_no_panic` | 100 concurrent ingest submissions, no panics, all processed or properly rejected |
| `test_dead_letter_roundtrip` | Submit task -> step fails with DeadLetter policy -> verify DLQ node exists -> retry -> verify reprocessed |
| `test_shutdown_under_load` | Pipeline processing tasks, initiate shutdown, verify graceful drain within timeout |

### Validation Criteria

- [x] PipelineSystem starts and shuts down cleanly
- [x] No panics under concurrent access (circuit breaker, workers, channels)
- [x] Shutdown completes within configured timeout
- [x] Per-item error isolation: one failing item does not affect others
- [x] Circuit breaker transitions are logged at correct levels (WARN for Open, INFO for HalfOpen/Closed)
- [x] Retry respects cancellation (no delays after cancel)
- [x] Backpressure blocks sender when channel is full
- [x] All 14 metrics are emitted correctly
- [x] Dead-letter items survive system restart (persisted in graph)

---

## Documentation Plan

| Document | Content |
|---|---|
| `pipeline/README.md` | Architecture overview with ASCII diagram from pipeline-management.md |
| Inline rustdoc on `PipelineSystem` | Public API docs with usage examples |
| Inline rustdoc on `Step` trait | How to implement a new pipeline step |
| Inline rustdoc on `CircuitBreaker` | State machine description, fallback table |
| Inline rustdoc on `RetryPolicy` | Defaults, delay schedule, jitter behavior |
| Inline rustdoc on `PipelineConfig` | All fields with defaults and valid ranges |

---

## Review Checklist

- [ ] All channels are bounded (no `unbounded_channel` usage)
- [ ] `select!` uses `biased` ordering everywhere (shutdown > interactive > background > timer)
- [ ] Circuit breaker uses atomics only (no Mutex for hot path)
- [ ] Retry is cancellation-aware (never sleeps through a cancel signal)
- [ ] CancellationToken hierarchy matches the documented tree
- [ ] Shutdown sequence follows the documented order (ingest 5s -> consolidation 10s -> root 30s)
- [ ] Dead-letter nodes are persisted in the graph (survive process restart)
- [ ] All metrics use the `uniko.` prefix
- [ ] All tracing spans include `node_id` for per-item correlation
- [ ] `PipelineConfig` has `Default` impl with documented values
- [ ] No `unwrap()` or `expect()` on channel operations (use `Result`)
- [ ] Worker tasks are `spawn`ed, not `block_on`ed
- [ ] `JoinHandle`s are stored and joined on shutdown
- [ ] Health status classification matches documented thresholds

---

## Definition of Done

1. **All files created in `uniko-pipes`**: `step.rs`, `types.rs`, `config.rs`, `circuit_breaker.rs`, `retry.rs`, `cancel.rs`, `dead_letter.rs`, `health.rs`, `metrics.rs` exist in `crates/uniko-pipes/src/`.
2. **All files created in `uniko-memory`**: `mod.rs`, `ingest_worker.rs`, `consolidation_worker.rs` exist in `crates/uniko-memory/src/pipeline/`.
3. **All unit tests pass**: `cargo nextest run -p uniko-pipes` and `cargo nextest run -p uniko-memory` pass with zero failures.
4. **All integration tests pass**: `cargo nextest run -p uniko-memory --test pipeline_integration` passes.
5. **No panics under load**: `test_concurrent_ingest_no_panic` with 100 concurrent tasks completes cleanly.
6. **Shutdown reliability**: `test_shutdown_under_load` completes within 30s timeout in 10/10 runs.
7. **Circuit breaker correctness**: All 6 state transition tests pass (Closed->Open, Open->HalfOpen, HalfOpen->Closed, HalfOpen->Open, concurrent access, reset on success).
8. **Dead-letter persistence**: DLQ items survive `PipelineSystem` drop and are visible after re-initialization.
9. **Metrics observable**: All 14 metrics registered and emitting values in test harness.
10. **Tracing observable**: Per-item trace spans visible in test subscriber output.
11. **No pipeline logic**: Zero P1-P8 step implementations. Only `MockStep` for testing.
12. **Clippy clean**: `cargo clippy -p uniko-pipes -- -D warnings` and `cargo clippy -p uniko-memory -- -D warnings` pass.
13. **Documented**: All public types and functions have rustdoc with examples.
