# API Reference

uniko is a Rust library, so its API is the set of crates you link and the functions you call —
there is no service to wire up. This page catalogs the public surface a downstream consumer
actually touches: opening a [`KnowledgeBase`](#knowledgebase-lifecycle), driving an ingest pipeline
through `PipelineSystem`, and reading memory back through `recall` and `answer_query`. Every
signature is reproduced from the source.

Two crates matter for callers:

- **`uniko-api`** — the public facade. It contains no logic; it re-exports the cognitive
  stack (`uniko-cortex` and below) plus a curated [`tools`](#agent-tools) module of
  agent-facing functions. Depend on this crate and you get the whole reachable surface.
- **`uniko-memory`** — where the lifecycle, recall, query, working-memory, summary, and
  agent-tool entry points live. `uniko-api::tools` re-exports the subset you most often
  reach for.

Storage primitives (`KnowledgeBase`, `Value`, `Transaction`, `NodeId`) come from
**`uniko-store`**, which seals uni-db behind a typed surface.

!!! note
    Every signature below is reproduced from the source. Parameter structs use
    `..Default::default()` patterns heavily — most fields are optional and documented inline
    in the crate's rustdoc. Only the load-bearing shape is shown here.

---

## KnowledgeBase lifecycle

A `KnowledgeBase` is the handle every other call takes. It wraps uni-db, registers the full
uniko schema on creation, and (by default) eagerly warms the embedding / model runtime.

=== "In-memory"

    ```rust
    use uniko_store::{KnowledgeBase, config::UnikoConfig};

    let kb = KnowledgeBase::in_memory(UnikoConfig::default()).await?;
    ```

=== "Persistent"

    ```rust
    use uniko_store::{KnowledgeBase, config::UnikoConfig};

    let kb = KnowledgeBase::open("./data/kb", UnikoConfig::default()).await?;
    ```

| Constructor | Signature | Description |
| --- | --- | --- |
| `in_memory` | `async fn in_memory(config: UnikoConfig) -> Result<Self>` | Ephemeral KB; registers schema, prefetches models. |
| `in_memory_with_xervo` | `async fn in_memory_with_xervo(config: UnikoConfig, extra_catalog: Vec<ModelAliasSpec>) -> Result<Self>` | As above, merging extra model aliases (e.g. an LLM for answer synthesis). |
| `open` | `async fn open(path: impl AsRef<Path>, config: UnikoConfig) -> Result<Self>` | Open or create a persistent KB; schema registration is idempotent. |
| `open_with_xervo` | `async fn open_with_xervo(path, config, extra_catalog) -> Result<Self>` | Persistent open with extra model aliases. |
| `open_with_xervo_no_prefetch` | `async fn open_with_xervo_no_prefetch(path, config, extra_catalog) -> Result<Self>` | Persistent open that skips model pre-warming; models load lazily on first use. |
| `open_with_runtime` | `async fn open_with_runtime(...)` | Open against a shared `ModelRuntime` (multi-KB workflows sharing one ONNX session). |

!!! note
    `ModelAliasSpec` (the `extra_catalog` element type for the xervo constructors above) comes
    from `uni_db` directly — the one type you import past the sealed `uniko-store` surface, for
    registering extra model aliases.

`generate` runs a completion through the model runtime registered on the KB:

```rust
use uniko_store::xervo::{GenerationOptions, Message};

let text: String = kb
    .generate("llm/default", &[Message::user("Summarise…")], GenerationOptions::default())
    .await?;
```

---

## PipelineSystem

`PipelineSystem` is the orchestration root: it owns the ingest worker, the consolidation
worker, bounded channels with backpressure, an LLM circuit breaker, and health tracking.
Submitting a task is non-blocking.

```rust
use std::sync::Arc;
use uniko_pipes::PipelineConfig;
use uniko_memory::pipeline::PipelineSystem;

let ps = PipelineSystem::new(PipelineConfig::default(), Arc::new(kb), vec![]);
ps.submit_ingest(task)?;        // enqueue a message / artifact / pdf
// …
ps.shutdown().await?;           // graceful drain
```

| Method | Signature | Description |
| --- | --- | --- |
| `new` | `fn new(config: PipelineConfig, kb: Arc<KnowledgeBase>, ingest_steps: Vec<Box<dyn Step>>) -> Self` | Spawn workers and begin processing immediately. |
| `submit_ingest` | `fn submit_ingest(&self, task: IngestTask) -> Result<(), UnikoError>` | Enqueue an ingest task; `Err` on backpressure / shutdown. |
| `submit_consolidation` | `fn submit_consolidation(&self, task: ConsolidationTask) -> Result<(), UnikoError>` | Enqueue a consolidation task. |
| `health` | `fn health(&self) -> PipelineHealth` | Aggregate queue depth, circuit state, and uptime. |
| `shutdown` | `async fn shutdown(self) -> Result<(), String>` | Drain workers within the configured timeout. |

An `IngestTask` is an enum over the three ingest kinds:

```rust
pub enum IngestTask {
    Message(IngestMessage),   // a conversation Message
    Artifact(IngestArtifact), // a file / document / URL / snippet
    Pdf(IngestPdf),           // a PDF document
}
```

A `ConsolidationTask` requests a consolidation cycle — either reactively
(`ObservationsReady`) or explicitly (`ForceConsolidate { agent_id }`, `RunCycle { agent_id }`).

!!! tip
    The consolidation worker is the heartbeat: after a successful cycle it triggers the cortex
    sweeps (procedure promotion, topic detection) under a per-agent throttle. You enqueue
    work; the system schedules the downstream cognition.

---

## Recall

Recall runs the 3-phase cascade (Compact → Expand → Broaden) with coverage gating and returns
a ranked `ContextBundle`.

```rust
use uniko_memory::recall::{recall, RecallConfig};

let bundle = recall(&kb, "what does the user prefer?", &RecallConfig::default()).await?;
for item in &bundle.items {
    println!("[{:?}] {} — {}", item.tier, item.score, item.content);
}
```

| Symbol | Signature / shape | Description |
| --- | --- | --- |
| `recall` | `async fn recall(kb: &KnowledgeBase, query: &str, config: &RecallConfig) -> Result<ContextBundle, UnikoError>` | Run the cascade; applies the `viewer` access-control gate before returning. |
| `RecallConfig` | struct | Cascade tuning: `limit`, `token_budget`, `min_score`, hybrid `vector_weight`/`bm25_weight`, reranker, query variants, per-phase gates, and `viewer` scope. `Default` ships sensible values (`limit: 15`, `token_budget: 8192`). |
| `ContextBundle` | `{ items: Vec<RecallItem>, total_tokens: usize, coverage: f64, … }` | Ranked result set plus coverage and token accounting. |
| `RecallItem` | `{ node_id: NodeId, node_type: String, score: f64, content: String, tier: RecallTier }` | One recalled node. |
| `RecallTier` | enum | `Semantic` / `Procedural` / `Episodic` / `KnowledgeBase` / `Provenance` — drives the scoring weight via `RecallTier::weight()`. |

!!! warning
    `RecallConfig::viewer` defaults to `ViewerScope::Unrestricted` — recall does **not** filter
    Fact/Observation visibility unless you set `viewer` to a concrete participant. Always set a
    viewer when answering on behalf of a specific user; treat post-filtering with
    [`filter_bundle`](#access-control-policy) as a fallback for direct lookups, not a substitute
    for scoping recall itself.

---

## Query

The query layer pairs recall with answer generation and records the result as an Episode —
the input that procedure promotion learns from. Recording is opt-in.

```rust
use uniko_memory::{answer_query, QueryRecordOptions, GeneratedAnswer};

let outcome = answer_query(
    &kb,
    question,
    &recall_config,
    |bundle, q| async move {
        // your own LLM call; uniko-memory does not own model selection
        let text = kb.generate("llm/default", &messages, opts).await?;
        Ok(GeneratedAnswer { text, input_tokens: None, output_tokens: None, model: None })
    },
    Some(QueryRecordOptions { participant_id: "agent-1".into(), ..Default::default() }),
).await?;
```

| Symbol | Signature / shape | Description |
| --- | --- | --- |
| `answer_query` | `async fn answer_query<G, Fut>(kb, question: &str, recall_config: &RecallConfig, generator: G, record: Option<QueryRecordOptions>) -> Result<QueryOutcome, UnikoError>` | Run recall, hand the bundle to your generator closure, and optionally record an Episode. |
| `record_query_episode` | `async fn record_query_episode(kb, agent_id: &str, params: RecordQueryEpisodeParams<'_>) -> Result<NodeId, UnikoError>` | The primitive: materialise an Episode from a question / answer / recall pair you already own. |
| `QueryOutcome` | `{ bundle: ContextBundle, answer: GeneratedAnswer, episode_id: Option<NodeId> }` | Recall bundle, generated answer, and the new Episode id when recording succeeded. |
| `GeneratedAnswer` | `{ text: String, input_tokens: Option<u64>, output_tokens: Option<u64>, model: Option<String> }` | What a generator closure returns. |
| `QueryRecordOptions` | `{ participant_id: String, action_type, outcome, importance, extra_state }` | Opt-in recording config; the Participant must already exist. |
| `RecordQueryEpisodeParams<'a>` | struct | Inputs for the primitive: `question`, `answer`, `recall_node_ids`, coverage / token / token-usage metadata. |

!!! note
    The generator is a closure, not a trait, so uniko-memory never owns LLM selection or
    system prompts. Recording failure surfaces as `episode_id = None` (logged at debug) — it
    never breaks a user-visible answer.

---

## Working memory

Working memory is **not** a stored node — it is computed live by traversing the graph from a
Goal outward through its Tasks, Sessions, Messages, Facts, and Entities. The result reflects
the current graph on every call.

```rust
use uniko_memory::{working_memory, WorkingMemoryParams};

let bundle = working_memory(&kb, WorkingMemoryParams::new("goal-1")).await?;
```

| Symbol | Signature / shape | Description |
| --- | --- | --- |
| `working_memory` | `async fn working_memory(kb: &KnowledgeBase, params: WorkingMemoryParams) -> Result<ContextBundle, UnikoError>` | Assemble the goal-anchored working-memory bundle. |
| `WorkingMemoryParams` | `{ goal_id: String, budget: Option<usize>, include_subgoals: bool, per_tier_limit: usize }` | `new(goal_id)` defaults to budget `None` (→ 8192 tokens), `include_subgoals: true`, `per_tier_limit: 25`. |

An absent goal returns an empty bundle (`coverage = 0.0`) rather than an error, so callers can
poll while a goal is being created.

---

## Summaries

`generate_session_summary` builds (or refreshes) a Summary for a Session — the F59 capability
Phase-3 recall falls back to when finer-grained Observations/Facts under-cover a query.

```rust
use uniko_memory::generate_session_summary;
use chrono::Utc;

let summary = generate_session_summary(&kb, "session-1", Utc::now(), None).await?;
```

| Symbol | Signature | Description |
| --- | --- | --- |
| `generate_session_summary` | `async fn generate_session_summary(kb: &KnowledgeBase, session_id: &str, now: DateTime<Utc>, llm_alias: Option<&str>) -> Result<Option<NodeId>, UnikoError>` | Extractive by default (deterministic, offline). With `llm_alias` set **and** the `llm` feature built, the material is rewritten abstractively. Idempotent on a stable `summary_id`. Returns `None` when the session has no summarisable content. |

---

## Agent tools

Pipelines handle what can be *inferred* from messages; agent tools handle what only the agent
can *decide* to record. These are re-exported by both `uniko-memory` and `uniko-api::tools`.
Most tools take `agent_id` referring to a Participant that must already exist, and return the
new node id. The exceptions are `assert_fact` and `invalidate_fact`, which take no `agent_id`
and return `FactUpsert` and `()` respectively.

```rust
use uniko_api::tools::{assert_fact, AssertFactParams};

let upsert = assert_fact(&kb, AssertFactParams {
    subject: "user".into(),
    predicate: "prefers".into(),
    object: Some("dark mode".into()),
    ..Default::default()
}).await?;
```

| Function | Signature | Description |
| --- | --- | --- |
| `add_observation` | `async fn add_observation(kb, agent_id: &str, params: AddObservationParams) -> Result<NodeId, UnikoError>` | Record an explicit Observation anchored to a Message. |
| `assert_fact` | `async fn assert_fact(kb, params: AssertFactParams) -> Result<FactUpsert, UnikoError>` | Upsert a `(subject, predicate, object)` Fact, embed it, and wire provenance. |
| `invalidate_fact` | `async fn invalidate_fact(kb, params: InvalidateFactParams) -> Result<(), UnikoError>` | Retract a Fact by closing its bitemporal validity interval (F37). |
| `create_goal` | `async fn create_goal(kb, agent_id: &str, params: CreateGoalParams) -> Result<NodeId, UnikoError>` | Create a Goal, wire `OWNED_BY` (and optional `PARENT_GOAL`), embed. |
| `create_task` | `async fn create_task(kb, agent_id: &str, params: CreateTaskParams) -> Result<NodeId, UnikoError>` | Create a Task, wire `ASSIGNED_TO` and optional `PART_OF`/`DEPENDS_ON`/`SUBTASK_OF`. |
| `record_episode` | `async fn record_episode(kb, agent_id: &str, params: RecordEpisodeParams) -> Result<NodeId, UnikoError>` | Record an Episode (action + outcome + state) and embed it. |
| `record_action` | `async fn record_action(kb, agent_id: &str, params: RecordActionParams) -> Result<RecordActionResult, UnikoError>` | Record an Action node, wire its edges, and overflow large output to an Artifact. |

Parameter structs (selected fields; all derive `Default`):

| Struct | Key fields |
| --- | --- |
| `AddObservationParams` | `message_id`, `content`, `subject`, `predicate?`, `object?`, `confidence?` |
| `AssertFactParams` | `subject`, `predicate`, `object?`, `observation_count?`, `supporting_observation_ids` |
| `InvalidateFactParams` | `fact_id`, `replacement_fact_id?`, `reason?`, `now?` |
| `CreateGoalParams` | `title`, `description?`, `status?`, `metrics?`, `guardrails?`, `deadline?`, `parent_goal_id?` |
| `CreateTaskParams` | `title`, `description?`, `status?`, `priority?`, `goal_id?`, `depends_on_task_id?`, `subtask_of_task_id?` |
| `RecordEpisodeParams` | `action_type`, `outcome?`, `state?`, `delta?`, `importance?`, `involved_action_ids` |
| `RecordActionParams` | `action_type`, `input?`, `output?`, `status?`, `triggered_by_message_id?`, `session_id?`, `previous_action_id?` |
| `RecordActionResult` | `{ action_node: NodeId, overflow_artifact: Option<NodeId> }` |

### Agent facade

`Agent` binds a `KnowledgeBase` and an `agent_id` so you don't repeat the id on every call.

```rust
use uniko_memory::Agent;

let agent = Agent::new(kb, "agent-1");
let goal = agent.create_goal(CreateGoalParams { title: "Ship the release".into(), ..Default::default() }).await?;
```

`Agent` exposes the same operations as methods (`create_goal`, `create_task`, `assert_fact`,
`invalidate_fact`, `add_observation`, `record_episode`, `record_action`, `working_memory`,
`recall`), each delegating to the free function with the bound `agent_id`.

---

## Access-control policy

Facts and Observations carry a `visibility` property (`public` / `private:{id}` /
`team:{id}` / `org:{id}`). The policy module filters a `ContextBundle` against a viewer
(F66). `recall` applies this automatically when `RecallConfig::viewer` is set; you can also
apply it directly.

```rust
use uniko_memory::policy::{Viewer, filter_bundle};

let viewer = Viewer::new(&kb, "participant-1").await?;
filter_bundle(&kb, &mut bundle, &viewer).await?;
```

| Symbol | Signature | Description |
| --- | --- | --- |
| `Viewer::new` | `async fn new(kb: &KnowledgeBase, participant_id: impl Into<String>) -> Result<Self, UnikoError>` | Resolve a viewer's team / org memberships from the graph. |
| `Viewer::from_parts` | `fn from_parts(participant_id, teams, orgs) -> Self` | Build a viewer from already-resolved memberships (no round-trip). |
| `filter_bundle` | `async fn filter_bundle(kb, bundle: &mut ContextBundle, viewer: &Viewer) -> Result<(), UnikoError>` | Remove items the viewer cannot see (Fact / Observation only); recomputes token count. |
| `visibility_admits` | `fn visibility_admits(visibility: Option<&str>, viewer: &Viewer) -> bool` | Decide whether a single visibility tag admits the viewer. Unknown schemes fail closed. |

---

## Rules (lifecycle)

Authored or induced Locy rules carry a confidence that decays each cycle and earns promotion
through match success.

| Symbol | Signature | Description |
| --- | --- | --- |
| `add_rule` | `async fn add_rule(kb, params: AddRuleParams) -> Result<NodeId, UnikoError>` | Register a rule in `candidate` status and load its Locy source; bad syntax returns `UnikoError::Locy` and leaves no node behind. |
| `apply_decay_cycle` | `async fn apply_decay_cycle(kb, cfg: RuleLifecycleConfig) -> Result<DecayReport, UnikoError>` | One decay pass over every non-stdlib rule, applying demote / repromote / prune transitions. |
| `record_rule_match` | `async fn record_rule_match(kb, name: &str, cfg: RuleLifecycleConfig) -> Result<(), UnikoError>` | Reset `missed_cycles` and reward a rule that bound a match this cycle. |
| `AddRuleParams` | `{ rule_id?, name, source, natural_language?, source_type, initial_confidence? }` | Rule registration inputs. |
| `RuleLifecycleConfig` | `{ decay_per_cycle, demote_threshold, repromote_threshold, prune_after_days }` | Defaults: `0.95` / `0.40` / `0.60` / `90`. |
| `DecayReport` | `{ decayed, demoted, promoted, pruned }` | Per-cycle transition counts. |

---

## NL → Cypher

A read-only natural-language-to-Cypher translator, guarded against mutating output.

| Symbol | Signature | Description |
| --- | --- | --- |
| `translate` | `async fn translate(kb, nl_query: &str, llm_alias: &str) -> Result<String, UnikoError>` | Translate a question to read-only Cypher (cached by normalised input). Re-exported as `translate_nl_to_cypher`. |
| `is_safe_read_only` | `fn is_safe_read_only(cypher: &str) -> bool` | Reject any query containing a mutating keyword (`CREATE`, `MERGE`, `DELETE`, `SET`, …). |

!!! warning
    `translate` produces a **read-only Cypher string** and never executes it — the caller runs
    the returned query. `is_safe_read_only` guards the output, rejecting any mutating keyword as
    a defence against hallucinated or injected writes.

---

## Sealed storage types

`uniko-store` re-exports the only uni-db value types higher crates legitimately name, so
consumers write `use uniko_store::…` and never reach into uni-db directly.

| Re-export | From | Use |
| --- | --- | --- |
| `KnowledgeBase` | `uniko_store::storage` | The graph handle every call takes. |
| `Value`, `Transaction`, `RetryOptions` | `uni_db` | Graph value type, write transactions, retry policy. |
| `ModelRuntime` | `uni_xervo::runtime` | Shared ONNX session for multi-KB workflows. |
| `temporal::{Btic, TemporalValue}` | `uni_db::common` | Bitemporal interval types on `valid_at` columns. |
| `xervo::{GenerationOptions, Message}` | `uni_db::xervo` | Prompt / generation types for `KnowledgeBase::generate`. |
| `Result`, `UnikoError` | `uniko_store::error` | The crate-wide error type. |

---

## See also

- [Architecture](../concepts/architecture.md) — how these layers fit together.
- [Recall](../pipelines/recall.md) — the cascade behind `recall` / `working_memory`.
- [Consolidation](../pipelines/consolidation.md) — what the consolidation worker does.
