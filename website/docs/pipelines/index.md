# Pipelines Overview

uniko keeps your agent responsive while its knowledge compounds in the background. Writing a
`Message` returns in milliseconds; the expensive intelligent work — deriving Facts, promoting
Procedures, naming Topics — runs on its own cadence and never blocks the turn the agent is in the
middle of. Every pipeline runs in-process: there is no separate service to deploy and no queue to
operate.

uniko splits that work into three movements, each with a different cost profile and a different
sense of urgency:

1. **Atomic ingest** — synchronous, single-transaction storage of a `Message`,
   its entities, and its structural edges. Returns in milliseconds.
2. **Async post-ingest** — observation extraction and embedding spawned off the
   ingest path, plus a worker-driven *consolidation* cycle (and a *cortex sweep*:
   procedure promotion, topic detection, decay, session maintenance) triggered by
   a threshold or a timer.
3. **Three-phase recall** — a coverage-gated cascade that reads the *compiled*
   knowledge first and only broadens to raw content when it has to.

```mermaid
flowchart TB
    msg([Message / Artifact])

    subgraph sync["Atomic ingest — synchronous, one transaction"]
        direction LR
        P1["P1 Ingest<br/>store node + edges<br/>&lt; 10ms"]
        P2["P2 NER<br/>entities + MENTIONS<br/>&lt; 100ms"]
        P1 --> P2
    end

    subgraph spawned["Spawned off the ingest path — async"]
        direction LR
        P3["P3 Observations<br/>extract statements<br/>&lt; 5s"]
        P7a["P7a Auto-embed<br/>Message embedding"]
    end

    subgraph worker["Consolidation worker — threshold OR timer"]
        direction TB
        P4["P4 Consolidation<br/>Observations → Facts<br/>reinforce / invalidate / drift"]
        sweep["Cortex sweep (gated)<br/>P5 Procedure promotion<br/>P6 Topic detection<br/>F50 decay · session maintenance"]
        P4 --> sweep
    end

    subgraph recall["Three-phase recall — coverage-gated cascade"]
        direction TB
        R1["Phase 1 Compact<br/>Facts · Procedures · Topics"]
        R2["Phase 2 Expand<br/>Episodes · Observations · Sessions"]
        R3["Phase 3 Broaden<br/>Chunks · Messages · graph"]
        R1 -->|coverage gate| R2 -->|coverage gate| R3
    end

    msg --> P1
    P2 -->|"ingest call returns"| P3
    P2 --> P7a
    P3 -->|ObservationsReady| P4

    recall -.reads compiled knowledge.-> worker
```

!!! note "uniko is an embedded Rust library"
    Every pipeline runs in-process inside your application. The orchestration
    layer is [`PipelineSystem`](#orchestration-the-pipelinesystem) in
    `uniko-memory`. There is no separate service to deploy — workers are Tokio
    tasks spawned at construction time and torn down on shutdown.

## What runs when

The data flow spans eight numbered pipeline stages. What matters operationally
is *who drives each one* — the calling thread, a spawned task, or a background
worker reacting to a trigger.

| Stage | Driver | Timing |
|---|---|---|
| **P1 Ingest** — store `Message`/`Artifact`, chunk, create edges | Synchronous (caller's task) | &lt; 10ms (message), &lt; 100ms (artifact) |
| **P2 NER** — extract `Entity` nodes + `MENTIONS` edges | Synchronous (caller's task) | &lt; 100ms |
| **P3 Observations** — extract factual statements | Spawned async task | &lt; 5s |
| **P7a Auto-embed** — embed the `Message` | Spawned async task | continuous |
| **P4 Consolidation** — derive `Fact`s, reinforce, invalidate, detect drift | Consolidation worker | background, threshold/timer |
| **P5 Procedure promotion** — recurring sequences → `Procedure`s | Consolidation worker (cortex sweep) | background, post-consolidation |
| **P6 Topic detection** — entity co-occurrence → `Topic`s | Consolidation worker (cortex sweep) | background, post-consolidation |

The ingest call returns after **P2** completes. **P3** and **P7a** are spawned as
independent tasks — observation extraction and embedding finish in the
background and notify the consolidation worker that new Observations exist. That split is
what keeps your agent responsive: the caller gets a fast, durable write-confirmation while
the expensive extraction work compounds knowledge off the hot path.

!!! tip "Compile once, query forever"
    Consolidation compiles raw Messages and Observations into Facts and Procedures once. The
    recall cascade queries that compiled knowledge first, so extraction is paid once at write
    time and amortized across every read — never re-derived per query.

## Atomic ingest

P1 and P2 run as ordered steps inside the **IngestWorker**, in a single async
task per item, with no queue between them. P1 creates the `Message` (or
`Artifact`) node and its structural edges; P2 attaches the extracted entities.
Because there is no queue between the steps, a message is either fully stored
with its entities or not — there is no half-ingested state visible to a query.

Each item flows through a chain of `Step`s. Steps declare their own error policy
via `StepErrorPolicy`, so a failure is contained to a single item and a single
step:

- `Skip` — log and continue to the next step (e.g. NER failing still leaves a
  stored Message).
- `DeadLetter` — persist a `DeadLetter` node for later retry, then continue.
- `Abort` — stop the remaining steps for this one item only.

Concurrency is bounded by a `Semaphore` (default 8), and the worker pulls from a
bounded channel (capacity 200), so backpressure propagates to the caller rather
than growing an unbounded queue.

→ Read more in **[Ingest](ingest.md)**.

## Async post-ingest: consolidation and the cortex sweep

When P3 finishes extracting Observations, it sends an `ObservationsReady`
notification to the **ConsolidationWorker**. The worker keeps a per-agent
counter and fires a consolidation cycle on a **threshold-OR-timer** rule:

- **20 new Observations** accumulate for an agent (`consolidation_threshold`), or
- the periodic timer ticks (`consolidation_interval_secs`, default 900s / 15 min)
  with any pending Observations,

whichever comes first. A `ForceConsolidate` task triggers a cycle immediately.

A consolidation cycle (P4) derives `Fact`s from unprocessed `Observation`s,
reinforces or invalidates existing Facts, detects drift, and records a
`ConsolidationCycle` audit node. After a *successful* cycle, the worker runs
the **cortex sweep** when both gates allow, governed by two independent throttles:

- a per-agent cycle counter (`cortex_cycle_every_n_consolidations`, default 4), and
- a per-sweep wall-clock minimum (`cortex_min_interval_secs`, default 600s).

When both gates allow, the sweep runs P5 procedure promotion (per agent), P6
topic detection (global), F50 memory decay (prune age-decayed `Episode`s), and
session maintenance (auto-close inactive `Session`s and summarise them).

!!! tip "Background reasoning is isolated"
    Procedure promotion, topic detection, decay, and session maintenance run *after*
    consolidation succeeds. A failure in any of them is logged and isolated — it never breaks
    consolidation or the write path. Background reasoning is enrichment, not a critical path.

→ Read more in **[Consolidation](consolidation.md)**.

## Three-phase recall

`recall()` searches across every memory layer but tries to do the *least* work
that still answers the query. It runs a coverage-gated cascade:

=== "Phase 1 — Compact"

    Searches the **compiled** tiers: `Fact`, `Procedure`, `Topic`. This is the
    cheapest path. If coverage clears the Phase 1 gate (default 0.75), the
    cascade can stop here.

=== "Phase 2 — Expand"

    Episodic recall: `Episode`, `Observation`, `Session`, plus fulltext over
    `Message` and `Observation` content, with MMR deduplication. Gated at a
    lower coverage threshold (default 0.65).

=== "Phase 3 — Broaden"

    Raw content and graph traversal: `Chunk` and `Message` search plus
    entity-link traversal. Phase 3 **always completes** — there is no early exit.

!!! tip "Recall sharpens as you ingest"
    Until consolidation has produced Facts, Procedures, and Topics, queries route through
    Phase 3 over raw content. As you ingest, the system shifts toward compiled-knowledge
    retrieval: `phase1_only_pct` rises and recall gets cheaper and sharper the more the
    knowledge base learns. This learned-index effect is structural and intentional.

The coverage score blends facet coverage, mean result score, and tier diversity:

```
coverage = 0.4 * facet_coverage + 0.3 * mean_score + 0.3 * diversity
```

A **drift override** forces Phase 2+ execution even when Phase 1 coverage is
sufficient, whenever the query references an Entity that consolidation has
flagged unstable (F39), via the F58 drift override — so questions about unstable
entities always re-check recent episodic evidence.

→ Read more in **[Recall](recall.md)**.

## Orchestration: the PipelineSystem

`PipelineSystem` owns the bounded channels, the LLM circuit breaker, the
dead-letter queue, and the worker join handles. It is created once and spawns
both workers immediately:

```rust
use std::sync::Arc;
use uniko_pipes::PipelineConfig;
use uniko_store::{KnowledgeBase, config::UnikoConfig};
use uniko_memory::pipeline::PipelineSystem;

let kb = Arc::new(KnowledgeBase::in_memory(UnikoConfig::default()).await?);
let ps = PipelineSystem::new(PipelineConfig::default(), kb, vec![]);

// ... submit ingest / consolidation tasks ...

ps.shutdown().await?;
```

Both workers run a `tokio::select!` loop with `biased` priority ordering:
shutdown first, then queued tasks. The **consolidation worker** adds a periodic
timer arm at the lowest priority; the ingest worker has no timer arm (only
shutdown + `rx.recv()`). This is what guarantees an interactive `recall()` is
never starved behind a consolidation cycle. Submitting a task is non-blocking:

```rust
ps.submit_ingest(task)?;            // backpressure if the channel is full
ps.submit_consolidation(task)?;
```

Health is observable per worker, including the circuit-breaker state:

```rust
let health = ps.health();
println!("ingest queue depth: {}", health.ingest.queue_depth);
println!("llm circuit: {:?}", health.llm_circuit);
```

!!! tip "Offline operation"
    NER, observation extraction, consolidation, and recall all function without
    an LLM. When the LLM provider fails, the circuit breaker opens and
    LLM-dependent steps degrade to local fallbacks (rule-based NER, rule-based
    observation extraction); the breaker probes and recovers automatically.

## Dive deeper

<div class="feature-grid" markdown>
<div class="feature-card" markdown>
### [Ingest](ingest.md)
Atomic single-transaction storage — P1 store, P2 NER, spawned P3/P7a, step error policies, and the IngestWorker.
</div>
<div class="feature-card" markdown>
### [Consolidation](consolidation.md)
Threshold-or-timer cycles that compile Observations into Facts, plus the gated cortex sweep: procedures, topics, decay, sessions.
</div>
<div class="feature-card" markdown>
### [Recall](recall.md)
The coverage-gated three-phase cascade — Compact, Expand, Broaden — with drift override and the phase1_only scaling signal.
</div>
</div>
