# Pipeline Management System

## Design Principles

1. **Channel-based actors, not DAG frameworks.** The pipeline topology is dynamic and reactive (event-driven + periodic + threshold-gated + priority), not a static DAG. Tokio actors with `select!` handle this naturally.
2. **Per-item error isolation.** One bad message never kills a pipeline. Steps can fail independently; partial success is tracked.
3. **Circuit breaker for external dependencies.** When the LLM provider goes down, LLM-dependent steps degrade gracefully to local fallbacks. The breaker recovers automatically.
4. **Interactive queries preempt background work.** A user's `recall()` call should never wait behind a consolidation cycle.
5. **Bounded channels everywhere.** No unbounded queues. Backpressure propagates to callers.
6. **Cancellation from root to leaf.** A single `CancellationToken` controls graceful shutdown of the entire system.
7. **Observability built in.** Every pipeline emits metrics (via `metrics` crate) and structured traces (via `tracing`).


## System Architecture

```
                     ┌──────────────────────────────────┐
                     │          PipelineSystem           │
                     │   cancel: CancellationToken       │
                     │   health() → PipelineHealth       │
                     │   shutdown(timeout)                │
                     └────────┬────────┬────────────────┘
                              │        │
               ┌──────────────┘        └──────────────┐
               │                                      │
    ┌──────────▼──────────┐              ┌────────────▼────────────┐
    │  IngestWorker        │              │  ConsolidationWorker    │
    │                      │              │                        │
    │  rx: mpsc(200)       │              │  rx: mpsc(32)          │
    │  sem: Semaphore(8)   │              │  sem: Semaphore(4)     │
    │  cancel: child_token │              │  cancel: child_token   │
    │                      │              │  trigger: 20 obs OR    │
    │  Steps (per item):   │              │    15 min timer        │
    │   P1: Ingest (sync)  │              │                        │
    │   P2: NER (sync)     │   on_done   │  Steps (per agent):    │
    │   P3: Observe (async)│─────────────▶│   P4: Consolidation   │
    │   P7a: Auto-embed    │              │   P5: Procedure Promo  │
    │                      │              │   P6: Topic Detection  │
    │  circuit_breaker:    │              │   P8: Rule Induction   │
    │    LLM calls         │              │                        │
    │  retry_policy:       │              │  P7b-d: Embedding +    │
    │    3 attempts, exp   │              │    Summary (async)     │
    │    backoff            │              │                        │
    └──────────────────────┘              └────────────────────────┘
```


## Core Types

```rust
/// Root of the pipeline system. Created once at Cortex initialization.
pub struct PipelineSystem {
    /// Root cancellation token. Cancel this to shut down everything.
    cancel: CancellationToken,
    /// Send work to the ingest pipeline.
    ingest_tx: mpsc::Sender<IngestTask>,
    /// Send work to the consolidation pipeline.
    consolidation_tx: mpsc::Sender<ConsolidationTask>,
    /// Circuit breaker for LLM provider.
    llm_breaker: Arc<CircuitBreaker>,
    /// Join handles for spawned workers.
    workers: Vec<JoinHandle<()>>,
}

/// Health status of the pipeline system.
#[derive(Debug, Serialize)]
pub struct PipelineHealth {
    pub ingest: WorkerHealth,
    pub consolidation: WorkerHealth,
    pub llm_circuit: CircuitState,
    pub uptime_secs: u64,
}

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

#[derive(Debug, Serialize)]
pub enum WorkerStatus {
    Healthy,        // processing normally
    Degraded,       // LLM circuit open, using fallbacks
    Backpressured,  // queue > 80% capacity
    Stalled,        // no items processed in > 5 minutes
}
```


## Error Handling

### Per-Item Error Isolation

Every pipeline processes items independently. A failure in one item
never affects another.

```rust
/// Result of processing a single item through the ingest pipeline (P1-P3).
pub struct ItemResult {
    pub node_id: NodeId,
    pub steps_succeeded: Vec<String>,
    pub steps_failed: Vec<(String, String)>,  // (step_name, error)
    pub steps_skipped: Vec<String>,
}

/// How a step handles its own failure.
pub enum StepErrorPolicy {
    Skip,        // log warning, continue to next step
    DeadLetter,  // store in dead-letter queue, continue
    Abort,       // stop remaining steps for this item
}
```

Each step declares its error policy:
- NER (P2): `Skip` — if NER fails, observation extraction gets fewer entities but still works
- Observation extraction (P3): `Skip` — if observations fail, consolidation has less input but functions
- Embedding (P7): `DeadLetter` — retry later when model is available
- Summarization (P7d): `Skip` — summaries are nice-to-have, not critical

### Retry with Exponential Backoff

For LLM-dependent operations (entity extraction via Xervo, NL-to-Cypher,
observation extraction LLM path, summarization):

```rust
pub struct RetryPolicy {
    pub max_attempts: u32,       // default: 3
    pub initial_delay_ms: u64,   // default: 500
    pub max_delay_ms: u64,       // default: 30_000
    pub backoff_multiplier: f64, // default: 2.0
}
```

Retry is cancellation-aware: if the pipeline is shutting down, retries
abort immediately.


### Circuit Breaker for LLM Provider

The LLM circuit breaker protects the system when the provider is down.

```rust
pub struct CircuitBreaker {
    state: AtomicU8,           // Closed=0, Open=1, HalfOpen=2
    failure_count: AtomicU32,
    last_failure_ms: AtomicU64,
    failure_threshold: u32,    // default: 5 consecutive failures
    recovery_ms: u64,          // default: 60_000 (1 minute)
}
```

**States:**
- **Closed**: Normal operation. LLM calls proceed.
- **Open**: After N consecutive failures, all LLM calls skip immediately. Non-LLM fallbacks activate (rule-based NER, rule-based observation extraction). Recovery probe after `recovery_ms`.
- **HalfOpen**: One probe call allowed. Success → Closed. Failure → Open.

**Fallback behavior when circuit is open:**
- P2 NER: spaCy/rule-based extraction only (always available)
- P3 Observations: rule-based statement extraction only
- P7d Summaries: skipped entirely
- P8 Rule Induction: skipped entirely (GENERATE step needs LLM)
- NL-to-Cypher: returns error to caller (can't degrade this)

**Logging:**
```
WARN  circuit breaker OPEN — LLM calls disabled for 60s (5 consecutive failures)
INFO  circuit breaker HALF-OPEN — probing LLM availability
INFO  circuit breaker CLOSED — LLM available, resuming normal operation
```


### Dead-Letter Queue

Failed items that should be retried later are stored as DeadLetter
nodes in the graph:

```
DeadLetter
  step              String    -- "entity_extractor", "summarizer", etc.
  error             String    -- error message
  node_ref          Int64     -- the node that failed
  retry_count       Int64     -- how many times retried so far
  max_retries       Int64     -- default: 3
  next_retry_at     DateTime  -- computed from backoff
  created_at        DateTime
```

Operations:
- `retry(id)` — resubmit a single item to the appropriate pipeline
- `retry_all_pending()` — resubmit all items where `next_retry_at < now()`
- `clear(id)` — mark a single item as permanently failed, stop retrying
- `clear_all()` — mark all items as permanently failed
- `list_pending()` — show all items awaiting retry

Automatic retry: a background task checks for pending dead letters
every 5 minutes and resubmits items whose `next_retry_at` has passed.


## Backpressure and Prioritization

### Bounded Channels

| Pipeline | Channel Capacity | Rationale |
|---|---|---|
| Ingest (P1-P3, P7a) | 200 | Content ingestion can batch; LLM calls in P3 are slow |
| Consolidation (P4-P6, P8) | 32 | Heavyweight operations, no need for deep queue |

When a channel is full, the sender blocks (backpressure to the caller).
For the ingest path, this means `memory.ingest()` pauses until there's
room — this is correct behavior, not a bug.

### Interactive Priority via Biased Select

Interactive operations (`recall`, `ask`, `working_memory`) preempt
background work using Tokio's `biased` select:

```rust
// In the pipeline actor's main loop:
loop {
    tokio::select! {
        biased;  // CRITICAL: check in order

        // 1. Shutdown signal (highest priority)
        _ = cancel.cancelled() => break,

        // 2. Interactive queries (preempt background)
        task = interactive_rx.recv() => { ... }

        // 3. Background work (lowest priority)
        task = background_rx.recv() => { ... }

        // 4. Periodic timer (only fires when nothing else is ready)
        _ = consolidation_timer.tick() => { ... }
    }
}
```

This ensures a user's `recall()` query never waits behind a
consolidation cycle or ingest batch.

### Queue Depth Monitoring

Pipeline health degrades based on queue depth:
- < 60% capacity: `Healthy`
- 60-80% capacity: log warning, continue
- 80%+ capacity: `Backpressured` status, emit metric alert
- 100% (full): sender blocks (backpressure propagates)


## Pipeline Dependencies and Coordination

### Per-Item Sequential: P1 → P2 → P3

For a single message, pipelines run in sequence within the ingest
worker. There is no queue between P1, P2, and P3 — they execute as
steps in a single async task:

```
Message arrives
  → P1: create Message node + edges           (sync, < 10ms)
  → P2: extract entities + MENTIONS edges     (sync, < 100ms)
  → P3: extract observations (async spawn)    (async, < 5s)
  → P7a: auto-embed Message                   (async spawn)
```

P3 and P7a are spawned as independent tasks — the ingest call returns
after P2 completes. P3 runs in the background, and when it finishes,
it notifies the consolidation worker that new observations exist.

### Event Notification: Enrichment → Consolidation

When P3 (observation extraction) completes for a batch of messages,
it sends a notification to the consolidation worker:

```rust
pub struct ObservationsReady {
    pub agent_id: String,
    pub observation_count: u32,
    pub source_node_ids: Vec<NodeId>,
}
```

The consolidation worker tracks per-agent observation counts. When
the count crosses the threshold (20), it triggers a consolidation
cycle for that agent.

### Threshold-OR-Timer: Consolidation Trigger

```
ConsolidationWorker receives:
  1. ObservationsReady { agent_id, count } → increment counter
     IF counter >= 20 → trigger consolidation, reset counter
  2. Timer tick (every 15 min) → consolidate all agents with counter > 0
  3. ForceConsolidate { agent_id } → immediate (from manual API call)
```

This means consolidation runs when either:
- 20 new observations accumulate for an agent, OR
- 15 minutes pass with any pending observations

Whichever comes first.


## Cancellation and Shutdown

### CancellationToken Hierarchy

```
root_cancel (PipelineSystem)
  ├── ingest_cancel (IngestWorker)
  │     ├── item_cancel (per-item, for LLM timeouts)
  │     └── ...
  ├── consolidation_cancel (ConsolidationWorker)
  │     ├── cycle_cancel (per-cycle, for Locy evaluation)
  │     └── ...
  └── planning_cancel (MCTS, for tree search)
```

Cancelling the root token cancels everything. Cancelling a child
only affects that subtree.

### Graceful Shutdown Sequence

```
1. Cancel ingest_cancel → stop accepting new messages
   Wait up to 5s for in-flight items to complete
2. Cancel consolidation_cancel → stop background processing
   Wait up to 10s for current cycle to complete
3. Cancel root_cancel → force-stop anything remaining
4. Join all worker tasks with 30s total timeout
5. Log final health metrics
```

### Cancellation in Long-Running Operations

**MCTS planning:** checks cancellation at each tree expansion.
Returns best plan found so far.

**Consolidation cycle:** checks cancellation between steps
(after pattern detection, after fact derivation, etc.).
Partial results are committed — if facts were derived before
cancellation, they persist.

**LLM calls:** wrapped in `tokio::select!` with the item's
cancellation token. Timeout after 30s even without cancellation.


## Observability

### Metrics (via `metrics` crate)

| Metric | Type | Labels |
|---|---|---|
| `uniko.ingest.items_total` | Counter | pipeline={ingest} |
| `uniko.ingest.items_failed` | Counter | pipeline, step |
| `uniko.ingest.duration_ms` | Histogram | pipeline |
| `uniko.ingest.queue_depth` | Gauge | pipeline |
| `uniko.consolidation.cycles_total` | Counter | agent_id |
| `uniko.consolidation.facts_derived` | Counter | agent_id |
| `uniko.consolidation.facts_invalidated` | Counter | agent_id |
| `uniko.consolidation.duration_ms` | Histogram | agent_id |
| `uniko.recall.phase1_only_pct` | Gauge | agent_id |
| `uniko.recall.assembly_ms` | Histogram | — |
| `uniko.llm.calls_total` | Counter | step |
| `uniko.llm.errors_total` | Counter | step, error_type |
| `uniko.llm.circuit_state` | Gauge | — (0=closed, 1=open) |
| `uniko.deadletter.pending` | Gauge | step |

Export: Prometheus endpoint (for server mode), structured logs (for embedded mode), `MetricsSnapshot` (for programmatic access).

### Structured Tracing

Every pipeline step is instrumented with `tracing::instrument`:

```rust
#[instrument(
    skip(self, ctx),
    fields(node_id = %ctx.node_id, pipeline = "ingest")
)]
async fn process_item(&self, ctx: &mut IngestContext) {  // EnrichmentContext is the legacy name
    for step in &self.steps {
        let _span = info_span!("pipeline_step", step = step.name()).entered();
        match step.execute(ctx).await {
            Ok(()) => info!("step completed"),
            Err(e) => warn!(error = %e, "step failed"),
        }
    }
}
```

A single message can be traced through all pipeline stages by
filtering on `node_id`:

```
TRACE[node_id=42] pipeline=ingest step=entity_extractor → 3 entities extracted
TRACE[node_id=42] pipeline=ingest step=observation_extractor → 2 observations
TRACE[node_id=42] pipeline=consolidation agent=agent-1 → fact derived
```

### Health Endpoint

```rust
impl PipelineSystem {
    pub async fn health(&self) -> PipelineHealth {
        PipelineHealth {
            ingest: self.ingest_worker.health(),
            consolidation: self.consolidation_worker.health(),
            llm_circuit: self.llm_breaker.state(),
            uptime_secs: self.started_at.elapsed().as_secs(),
        }
    }
}
```

Exposed via MCP tool (`uniko_health`) and HTTP endpoint (server mode).


## Configuration

```rust
pub struct PipelineConfig {
    // Ingest
    pub ingest_queue_capacity: usize,         // default: 200
    pub ingest_concurrency: usize,            // default: 8

    // Consolidation
    pub consolidation_queue_capacity: usize,   // default: 32
    pub consolidation_concurrency: usize,      // default: 4
    pub consolidation_threshold: u32,          // default: 20 observations
    pub consolidation_interval_secs: u64,      // default: 900 (15 min)

    // Retry
    pub retry_max_attempts: u32,               // default: 3
    pub retry_initial_delay_ms: u64,           // default: 500
    pub retry_max_delay_ms: u64,               // default: 30_000
    pub retry_backoff_multiplier: f64,         // default: 2.0

    // Circuit breaker
    pub circuit_failure_threshold: u32,        // default: 5
    pub circuit_recovery_ms: u64,              // default: 60_000

    // Dead letter
    pub dead_letter_max_retries: u32,          // default: 3
    pub dead_letter_check_interval_secs: u64,  // default: 300 (5 min)

    // Shutdown
    pub shutdown_timeout_secs: u64,            // default: 30
}
```

All values configurable via `UnikoConfig` or environment variables.
