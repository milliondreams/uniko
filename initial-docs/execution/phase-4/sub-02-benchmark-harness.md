# Phase 16: Benchmark Harness & Performance Validation

## Context

This phase builds the benchmark harness infrastructure and runs all 5 benchmarks to produce published performance numbers. Benchmark results prove uniko's competitive position against existing cognitive memory systems (Mem0, Graphiti, Letta, Zep, MemGPT) and validate that the entire pipeline chain (P1-P8) works at scale. Every architectural decision in the system — BTIC temporal intervals, multi-phase recall cascade, consolidation-derived facts, procedural memory, embedding-based search — is stress-tested here against real-world conversational memory datasets.

uniko is a cognitive memory system for AI agents built in Rust on uni-db (embedded graph database). The benchmark suite covers 5 established benchmarks: LoCoMo (episodic + semantic memory over conversations), LongMemEval (long-context memory abilities), MemoryAgentBench (agent memory competencies), BEAM (extreme-scale memory), and Evo-Memory (learning improvement over time). Together these exercise the full system: ingestion, entity extraction, observation extraction, consolidation, fact derivation, episodic recording, recall cascade, context assembly, and LLM answer generation.

**Key principle:** Benchmarks are not just pass/fail gates — they drive optimization priorities. Phase1_only_pct (percentage of recalls satisfied by Phase 1 graph traversal alone) must trend upward as consolidation runs, proving that the system builds progressively better structured knowledge. Compression_ratio (context_bundle_tokens / total_stored_tokens) must exceed 0.1, proving the recall cascade selects relevant information rather than dumping everything.

## Prerequisites

| Dependency | Status Required | What It Provides |
|---|---|---|
| Phase 13: Procedural Memory (P6) | Complete | Procedure promotion, effectiveness scoring, pattern detection from episodes |
| Phase 14: ASSUME/ABDUCE | Complete | Hypothetical reasoning, counterfactual evaluation, forward chaining |
| Phase 15: MCP Server | Complete | Full external API surface (all 12 agent tools exposed via MCP protocol) |
| Phases 1-12 (all DIF features) | Complete | Full pipeline chain P1-P7, recall cascade, consolidation, context assembly, embeddings, NER, working memory |
| LLM provider configured | Available | For answer generation from context bundles (GPT-4o, Claude, or equivalent) |
| Benchmark datasets downloaded | Available | LoCoMo, LongMemEval, MemoryAgentBench, BEAM, Evo-Memory raw data files |
| `tokio` 1.x | Available | Async runtime for benchmark execution |
| `serde_json` | Available | Parsing benchmark data files (JSON/CSV) |
| `criterion` or `divan` | Available | Micro-benchmark timing (optional, for profiling sub-phase) |

## Sub-phases

---

### 16.1 — Benchmark Harness Infrastructure

**Objective:** Build the shared benchmark framework that all 5 benchmarks plug into: data loading, metric computation, token budget enforcement, and result reporting.

#### Files

| File | Type | Purpose |
|---|---|---|
| `benchmarks/Cargo.toml` | Config | Workspace member, depends on `uniko-api` |
| `benchmarks/src/lib.rs` | Rust | Crate root, re-exports |
| `benchmarks/src/harness.rs` | Rust | `BenchmarkHarness`, `Benchmark` trait, execution orchestration |
| `benchmarks/src/metrics.rs` | Rust | `BenchMetrics`, scoring functions, phase1_only_pct tracking |
| `benchmarks/src/reporter.rs` | Rust | JSON + markdown table output, comparison tables |
| `benchmarks/src/data.rs` | Rust | Dataset loaders (JSON, CSV, JSONL parsing) |

#### Root Cargo.toml Addition

```toml
[workspace]
members = [
    # ... existing members ...
    "benchmarks",
]
```

#### `harness.rs` — BenchmarkHarness & Benchmark Trait

```rust
/// Central benchmark execution engine. Owns a Uniko instance,
/// tracks metrics across all runs, and enforces token budgets.
pub struct BenchmarkHarness {
    /// The Uniko instance under test.
    uniko: Uniko,
    /// Accumulated metrics across all benchmark questions.
    metrics: BenchMetrics,
    /// LLM provider for answer generation from context bundles.
    llm: Arc<LlmProvider>,
    /// Token budget for context assembly (8K, 16K, 32K).
    token_budget: usize,
}
```

Functions:

- `BenchmarkHarness::new(config: UnikoConfig, llm: Arc<LlmProvider>, token_budget: usize) -> Result<Self>` — Creates fresh Uniko instance with given config and budget.
- `BenchmarkHarness::reset(&mut self) -> Result<()>` — Drops and recreates the Uniko instance. Used between benchmark runs for isolation.
- `async fn run_benchmark(&mut self, bench: &dyn Benchmark) -> Result<BenchmarkResult>` — Full execution: load data, ingest, run queries, score, report.
- `async fn run_all(&mut self, benchmarks: &[Box<dyn Benchmark>]) -> Result<Vec<BenchmarkResult>>` — Run all benchmarks sequentially, reset between each.

```rust
/// Trait implemented by each benchmark (LoCoMo, LongMemEval, etc.).
#[async_trait]
pub trait Benchmark: Send + Sync {
    /// Human-readable benchmark name.
    fn name(&self) -> &str;

    /// Load the benchmark dataset from disk.
    fn load_data(&self, data_dir: &Path) -> Result<BenchmarkData>;

    /// Ingest all benchmark content into the Uniko instance.
    /// This is the "memorize" phase — conversations, documents, etc.
    async fn ingest(&self, harness: &mut BenchmarkHarness, data: &BenchmarkData) -> Result<IngestStats>;

    /// Run a single query against the loaded memory.
    /// Returns the system's answer for scoring.
    async fn query(&self, harness: &BenchmarkHarness, question: &Question) -> Result<Answer>;

    /// Score a predicted answer against the expected answer.
    /// Returns a per-question score (0.0 to 1.0).
    fn score(&self, predicted: &Answer, expected: &Answer, question_type: &str) -> f64;

    /// Return all questions for this benchmark.
    fn questions(&self, data: &BenchmarkData) -> Vec<Question>;

    /// Number of token budget configurations to test at.
    fn token_budgets(&self) -> Vec<usize> { vec![8_192, 16_384, 32_768] }
}
```

```rust
pub struct BenchmarkData {
    pub conversations: Vec<Conversation>,
    pub questions: Vec<Question>,
    pub metadata: serde_json::Value,
}

pub struct Question {
    pub id: String,
    pub text: String,
    pub question_type: String,       // "single-hop", "temporal", "multi-hop", etc.
    pub expected_answer: String,
    pub metadata: serde_json::Value,  // benchmark-specific fields
}

pub struct Answer {
    pub text: String,
    pub context_tokens: usize,       // tokens in the context bundle used
    pub assembly_latency_ms: u64,    // time to assemble context
    pub generation_latency_ms: u64,  // time for LLM to generate answer
    pub phase1_only: bool,           // was Phase 1 of recall sufficient?
}

pub struct Conversation {
    pub id: String,
    pub sessions: Vec<Session>,
    pub participants: Vec<String>,
}

pub struct Session {
    pub id: String,
    pub turns: Vec<Turn>,
}

pub struct Turn {
    pub sender: String,
    pub content: String,
    pub timestamp: DateTime<Utc>,
}

pub struct IngestStats {
    pub turns_ingested: u64,
    pub entities_extracted: u64,
    pub observations_created: u64,
    pub facts_derived: u64,
    pub episodes_recorded: u64,
    pub consolidation_cycles: u64,
    pub total_ingest_time_ms: u64,
}

pub struct BenchmarkResult {
    pub benchmark_name: String,
    pub total_questions: usize,
    pub overall_score: f64,
    pub scores_by_type: HashMap<String, TypeScore>,
    pub ingest_stats: IngestStats,
    pub recall_stats: RecallStats,
    pub token_budget: usize,
}

pub struct TypeScore {
    pub question_type: String,
    pub count: usize,
    pub mean_score: f64,
    pub median_score: f64,
    pub min_score: f64,
    pub max_score: f64,
}

pub struct RecallStats {
    pub phase1_only_pct: f64,
    pub mean_assembly_latency_ms: f64,
    pub p95_assembly_latency_ms: f64,
    pub mean_generation_latency_ms: f64,
    pub compression_ratio: f64,
}
```

#### `metrics.rs` — BenchMetrics & Scoring

```rust
pub struct BenchMetrics {
    /// Per-question scores indexed by question ID.
    pub scores: HashMap<String, f64>,
    /// Assembly latency samples (ms) for histogram computation.
    pub assembly_latencies: Vec<u64>,
    /// Generation latency samples (ms).
    pub generation_latencies: Vec<u64>,
    /// Count of Phase 1-only recalls vs total recalls.
    pub phase1_only_count: u64,
    pub total_recall_count: u64,
    /// Total tokens stored in the system.
    pub total_stored_tokens: u64,
    /// Total tokens in assembled context bundles.
    pub total_context_tokens: u64,
}
```

Functions:

- `BenchMetrics::new() -> Self` — Initialize with empty collections.
- `fn record_score(&mut self, question_id: &str, score: f64)` — Record a per-question score.
- `fn record_assembly_latency(&mut self, latency_ms: u64)` — Record an assembly latency sample.
- `fn record_recall(&mut self, phase1_only: bool)` — Record whether Phase 1 was sufficient.
- `fn phase1_only_pct(&self) -> f64` — Compute `phase1_only_count / total_recall_count`.
- `fn compression_ratio(&self) -> f64` — Compute `total_context_tokens / total_stored_tokens`.
- `fn mean_assembly_latency_ms(&self) -> f64` — Mean of all assembly latency samples.
- `fn p95_assembly_latency_ms(&self) -> f64` — 95th percentile assembly latency.

Scoring functions:

```rust
/// Binary context-contains-answer: does the context bundle contain the answer?
/// Checks if any significant token overlap exists between context and expected answer.
pub fn context_contains_answer(context: &str, expected: &str) -> bool;

/// Token-level F1 score.
/// F1 = 2 * (precision * recall) / (precision + recall)
/// precision = |predicted_tokens ∩ expected_tokens| / |predicted_tokens|
/// recall = |predicted_tokens ∩ expected_tokens| / |expected_tokens|
pub fn f1_score(predicted: &str, expected: &str) -> f64;

/// Exact match (normalized).
/// Lowercase, strip punctuation, normalize whitespace, then compare.
pub fn exact_match(predicted: &str, expected: &str) -> bool;

/// Token-level precision.
pub fn precision(predicted: &str, expected: &str) -> f64;

/// Token-level recall.
pub fn recall(predicted: &str, expected: &str) -> f64;
```

Token normalization for scoring:

```rust
/// Normalize text for scoring: lowercase, remove punctuation,
/// collapse whitespace, split into token set.
fn normalize_for_scoring(text: &str) -> HashSet<String>;
```

#### `reporter.rs` — Result Output

```rust
/// Output benchmark results as structured JSON.
pub fn report_json(results: &[BenchmarkResult]) -> Result<String>;

/// Output benchmark results as a markdown table.
pub fn report_markdown(results: &[BenchmarkResult]) -> Result<String>;

/// Output a comparison table: uniko vs competitors.
pub fn report_comparison(
    our_results: &[BenchmarkResult],
    competitor_data: &CompetitorData,
) -> Result<String>;
```

```rust
pub struct CompetitorData {
    pub entries: Vec<CompetitorEntry>,
}

pub struct CompetitorEntry {
    pub system_name: String,       // "Mem0", "Graphiti", "Letta", "Zep", etc.
    pub benchmark_name: String,
    pub score: f64,
    pub notes: String,             // "91.6% LoCoMo single-hop", etc.
}
```

Markdown table format:

```
| Benchmark | uniko | Mem0 | Graphiti | Letta | Zep | MemGPT | Hindsight |
|-----------|-------|------|----------|-------|-----|--------|-----------|
| LoCoMo    | XX.X% | 91.6%| 59.5%   | 58.4% | —   | 49.8%  | —         |
| LongMemEval| XX.X%| —    | —        | —     |~75% | —      | —         |
| BEAM (1M) | XX.X% | —    | —        | —     | —   | —      | 64.1%     |
| ...       |       |      |          |       |     |        |           |
```

#### `data.rs` — Dataset Loaders

```rust
/// Load a JSON benchmark dataset from a file path.
pub fn load_json<T: DeserializeOwned>(path: &Path) -> Result<T>;

/// Load a JSONL (JSON Lines) dataset, one record per line.
pub fn load_jsonl<T: DeserializeOwned>(path: &Path) -> Result<Vec<T>>;

/// Load a CSV benchmark dataset.
pub fn load_csv<T: DeserializeOwned>(path: &Path) -> Result<Vec<T>>;

/// Parse a benchmark-specific conversation format into the shared Conversation struct.
pub fn parse_conversations(raw: &serde_json::Value, format: DataFormat) -> Result<Vec<Conversation>>;

pub enum DataFormat {
    LoCoMo,
    LongMemEval,
    MemoryAgentBench,
    Beam,
    EvoMemory,
}
```

---

### 16.2 — LoCoMo Benchmark

**Objective:** Implement the LoCoMo benchmark — 10 conversations with 19 sessions each, 5882 turns total, 1986 questions across 5 types. Target: 75%+ overall accuracy.

#### Files

| File | Type | Purpose |
|---|---|---|
| `benchmarks/src/locomo.rs` | Rust | `LoComoBenchmark` implementation |
| `benchmarks/data/locomo/` | Data | Downloaded dataset (10 conversations, questions) |

#### Struct

```rust
pub struct LoComoBenchmark {
    data_dir: PathBuf,
}

impl Benchmark for LoComoBenchmark {
    fn name(&self) -> &str { "LoCoMo" }
    // ...
}
```

#### Dataset Structure

- 10 conversations, each with ~19 sessions
- 5882 total turns across all conversations
- 1986 questions across 5 types:
  - **Single-hop** (direct lookup): "What did Caroline do last Thursday?" — requires finding one message
  - **Temporal** (time reasoning): "What happened before Caroline's trip?" — requires temporal ordering via BTIC and session timestamps
  - **Multi-hop** (graph traversal): "Who did X interact with about Y?" — requires multiple edge traversals
  - **Open-ended** (synthesis): "What are Caroline's interests?" — requires aggregating multiple facts
  - **Adversarial** (attribution): "Did Caroline say X?" — tests SENT_BY attribution accuracy, must distinguish who said what

#### Ingestion Flow

```
For each conversation (10 total):
  For each session (19 per conversation):
    For each turn (messages in session):
      1. ingest_message(content, sender_id, session_id, timestamp)
         → P1: creates Message node, chunks if > 1024 tokens
         → P2: NER extracts entities, creates MENTIONS edges
         → P3: Observations extracted, linked to entities
      2. After every session: record_episode(action_type: "conversation", ...)
         → Episode node with INVOLVES → relevant Actions, MENTIONS → Entities
    After every 3 sessions: trigger consolidation
      → P4: Observations → Facts (BTIC intervals set)
      → P4: Contradiction detection (adversarial test prep)
      → P4: Entity deduplication
    After all sessions: run P7 embedding refresh
      → P7b: pooled Artifact embeddings updated
      → P7d: Summaries generated for sessions
```

#### Query Flow

```
For each question (1986 total):
  1. recall(question.text, budget=token_budget)
     → Phase 1: graph traversal (entities, facts, observations)
     → Phase 2: vector search (if Phase 1 coverage < threshold)
     → Phase 3: fulltext search (if Phase 2 coverage < threshold)
     → ContextBundle assembled
  2. Record: phase1_only? assembly_latency?
  3. LLM: generate answer from ContextBundle + question
  4. score(predicted, expected, question_type)
```

#### Scoring per Question Type

| Question Type | Primary Metric | Scoring Method |
|---|---|---|
| Single-hop | F1 | Token-level F1 between predicted and expected |
| Temporal | F1 | Token-level F1 (temporal ordering validated by answer correctness) |
| Multi-hop | F1 | Token-level F1 (requires multi-entity resolution) |
| Open-ended | F1 | Token-level F1 (partial credit for incomplete synthesis) |
| Adversarial | F1 + exact_match on attribution | Must correctly attribute or deny attribution |

#### Special Validations

- **Adversarial questions:** Verify SENT_BY edges are correctly wired. Question asks "Did X say Y?" — system must check Message → SENT_BY → Participant and give correct attribution. False positive (claiming X said something they didn't) counts as 0.0.
- **Temporal questions:** Verify observed_at timestamps and session ordering. Question asks "What happened before/after X?" — system must use BTIC temporal intervals and session boundaries to order events correctly.
- **Session boundaries:** Each session is a distinct conversation window. Facts from session 5 should not bleed into session 3's context unless they represent persistent knowledge.

#### Target

- Overall: 75%+
- Competitor reference: Mem0 91.6% (best), Graphiti 59.5%, Letta 58.4%, MemGPT 49.8%

---

### 16.3 — LongMemEval Benchmark

**Objective:** Implement the LongMemEval benchmark — 500 questions testing 5 memory abilities across contexts of 115K-500K tokens. Target: 70%+ overall accuracy.

#### Files

| File | Type | Purpose |
|---|---|---|
| `benchmarks/src/longmemeval.rs` | Rust | `LongMemEvalBenchmark` implementation |
| `benchmarks/data/longmemeval/` | Data | Downloaded dataset |

#### Struct

```rust
pub struct LongMemEvalBenchmark {
    data_dir: PathBuf,
}

impl Benchmark for LongMemEvalBenchmark {
    fn name(&self) -> &str { "LongMemEval" }
    // ...
}
```

#### Memory Abilities Tested

| Ability | Question Count | What It Tests | uniko Feature Exercised |
|---|---|---|---|
| Information Extraction | ~100 | Find specific facts in long context | Recall cascade Phase 1 (graph traversal), entity lookup |
| Multi-Session Reasoning | ~100 | Synthesize across multiple sessions | Cross-session fact aggregation, Topic clustering |
| Temporal Reasoning | ~100 | Order events, reason about time | BTIC temporal intervals, observed_at timestamps, session ordering |
| Knowledge Updates | ~100 | Handle contradicting information | BTIC invalidation, P4 contradiction resolution, Fact supersession |
| Abstention | ~100 | Correctly report "no information" | Recall cascade returns empty/low-confidence result, system declines to answer |

#### Ingestion Flow

```
For each conversation instance (115K-500K tokens):
  Ingest as a series of sessions, each session containing multiple messages.
  Run full pipeline: P1 → P2 → P3 → P4 (consolidation after each session group) → P7.
  Knowledge Updates instances:
    - Ingest initial information (e.g., "Alice lives in New York")
    - Later ingest contradicting information (e.g., "Alice moved to San Francisco")
    - P4 consolidation: BTIC invalidation closes old Fact interval, creates new Fact
    - Query must return current information ("San Francisco"), not outdated ("New York")
```

#### Key Test: Knowledge Updates

```
1. Ingest: "Alice's phone number is 555-1234" at t=1
   → Observation created → P4 derives Fact(subject="Alice", predicate="phone_number", object="555-1234", valid_at=[t1, ∞))
2. Ingest: "Alice changed her number to 555-5678" at t=2
   → Observation created → P4 detects contradiction (same subject + predicate)
   → Old Fact invalidated: valid_at=[t1, t2)
   → New Fact created: valid_at=[t2, ∞)
3. Query: "What is Alice's phone number?"
   → Recall finds Fact with valid_at containing now → "555-5678"
   → Score: 1.0 if "555-5678" in answer, 0.0 if "555-1234" in answer
```

#### Key Test: Abstention

```
1. Ingest: conversation about various topics (NOT including "quantum physics")
2. Query: "What did Alice say about quantum physics?"
3. Expected: system reports "no information" / "not discussed" / refuses to answer
4. Score: 1.0 if system correctly abstains, 0.0 if system hallucmates an answer
   Scoring: check answer for abstention phrases ("no information", "not mentioned",
            "cannot find", "no record") + absence of fabricated content
```

#### Scoring

- Primary metric: F1 per question type
- Abstention scoring: binary (correct abstention = 1.0, false answer = 0.0)
- Knowledge update scoring: exact match on current value (old value = 0.0)

#### Target

- Overall: 70%+
- Competitor reference: Zep ~75% (best)

---

### 16.4 — MemoryAgentBench Benchmark

**Objective:** Implement the MemoryAgentBench benchmark — 4 competencies, 2071 questions, 100K-1.4M tokens. Target: 60%+ overall accuracy.

#### Files

| File | Type | Purpose |
|---|---|---|
| `benchmarks/src/memoryagentbench.rs` | Rust | `MemoryAgentBenchBenchmark` implementation |
| `benchmarks/data/memoryagentbench/` | Data | Downloaded dataset |

#### Struct

```rust
pub struct MemoryAgentBenchBenchmark {
    data_dir: PathBuf,
}

impl Benchmark for MemoryAgentBenchBenchmark {
    fn name(&self) -> &str { "MemoryAgentBench" }
    // ...
}
```

#### Competencies Tested

| Competency | What It Tests | uniko Feature Exercised |
|---|---|---|
| Accurate Retrieval | Find exact information from memorized content | Recall cascade (all phases), entity/fact lookup |
| Test-Time Learning | Learn from context during evaluation, apply later | Observation extraction → Fact derivation within single evaluation run |
| Long-Range Understanding | Reason across large token windows (100K-1.4M) | Scalable graph storage, efficient recall at scale, compression |
| Conflict Resolution | Handle contradicting information correctly | BTIC invalidation, P4 consolidation, Fact confidence scoring |

#### Two-Phase Evaluation

**Phase A — Memorize:**
```
For each context document (100K-1.4M tokens):
  1. Chunk into Artifacts via P1
  2. Extract entities via P2 (NER)
  3. Extract observations via P3
  4. Run consolidation P4 (derive Facts from Observations)
  5. Generate embeddings via P7
  Record: entities_extracted, observations_created, facts_derived, consolidation_cycles
```

**Phase B — Query:**
```
For each question (2071 total):
  1. recall(question.text, budget=token_budget)
  2. LLM generates answer from ContextBundle
  3. score(predicted, expected)
  Record: phase1_only?, assembly_latency, generation_latency
```

#### Conflict Resolution Path

```
1. Ingest: "The project deadline is March 15"
   → Fact(subject="project", predicate="deadline", object="March 15", valid_at=[t1, ∞))
2. Ingest: "The deadline has been extended to April 1"
   → P3: Observation detected with same subject+predicate but different object
   → P4: Contradiction detected. Old Fact invalidated: valid_at=[t1, t2). New Fact: valid_at=[t2, ∞)
3. Query: "When is the project deadline?"
   → Recall: current Fact (valid_at contains now) = "April 1"
   → Score: 1.0 if "April 1", 0.0 if "March 15"
```

#### Scoring

- Primary metric: F1 per competency
- Conflict resolution: exact match on current value
- Overall: weighted average across competencies

#### Target

- Overall: 60%+
- 2071 questions, 100K-1.4M tokens

---

### 16.5 — BEAM & Evo-Memory Benchmarks

**Objective:** Implement the extreme-scale BEAM benchmark and the learning-improvement Evo-Memory benchmark. These push the system to its limits: BEAM tests memory at 10M tokens; Evo-Memory tests whether the system actually learns over time.

#### Files

| File | Type | Purpose |
|---|---|---|
| `benchmarks/src/beam.rs` | Rust | `BeamBenchmark` implementation |
| `benchmarks/src/evomemory.rs` | Rust | `EvoMemoryBenchmark` implementation |

#### BEAM — Benchmark for Extreme-scale Autobiographical Memory

```rust
pub struct BeamBenchmark {
    data_dir: PathBuf,
}

impl Benchmark for BeamBenchmark {
    fn name(&self) -> &str { "BEAM" }
    fn token_budgets(&self) -> Vec<usize> {
        // BEAM tests at specific token scales, not just budget
        vec![8_192, 16_384, 32_768]
    }
    // ...
}
```

**Dataset:**
- 10 memory abilities tested
- 400 questions total
- Token scales: 128K, 500K, 1M, 10M tokens
- Questions per scale: ~100 per scale

**Memory abilities:**

| Ability | Description | What It Stresses |
|---|---|---|
| Single fact retrieval | Find one specific fact | Index efficiency at scale |
| Multi-fact aggregation | Combine facts about an entity | Fact consolidation quality |
| Temporal ordering | Order events correctly | BTIC interval accuracy |
| Causal reasoning | Why did X happen? | Episode graph traversal |
| Entity tracking | Follow entity state changes | Entity deduplication + Fact updates |
| Contradiction handling | Resolve conflicting info | BTIC invalidation accuracy |
| Implicit reasoning | Derive unstated conclusions | Fact derivation + observation quality |
| Cross-entity reasoning | Relate facts across entities | Graph traversal depth |
| Abstention | "Not enough info" | Recall cascade confidence |
| Summarization | Summarize an entity's history | Summary generation quality |

**Scale stress test (at 1M and 10M tokens):**
```
1. Ingest 33K+ messages (at 1M scale) or 330K+ messages (at 10M scale)
2. Track: entity_count, fact_count, observation_count
3. Verify: entity deduplication scales (same entity mentioned 100+ times = 1 Entity node)
4. Verify: consolidation completes within time budget (< 30s per cycle at 1M)
5. Verify: phase1_only_pct trends upward as consolidation accumulates
6. Verify: recall latency stays within target (assembly < 100ms) even at 10M tokens
```

**Scoring:**
- Primary metric: F1 per ability per scale
- Target: 50%+ at 1M tokens
- Competitor reference: Hindsight 64.1% (best at 1M)

#### Evo-Memory — Evolutionary Memory Benchmark

```rust
pub struct EvoMemoryBenchmark {
    data_dir: PathBuf,
}

impl Benchmark for EvoMemoryBenchmark {
    fn name(&self) -> &str { "Evo-Memory" }
    // ...
}
```

**Dataset:**
- 10 datasets across 8 task types
- Sequential task stream (tasks arrive one at a time)
- Each task: question + ground truth answer

**Task types:**

| Task Type | Description |
|---|---|
| Fact recall | Retrieve previously learned facts |
| Preference tracking | Track and recall user preferences |
| Temporal reasoning | Order events from memory |
| Entity updates | Handle entity state changes |
| Multi-hop reasoning | Traverse multiple relationships |
| Pattern recognition | Identify recurring patterns |
| Contradiction resolution | Handle conflicting information |
| Generalization | Apply learned patterns to new contexts |

**Evaluation flow (sequential, learning-oriented):**

```
For each dataset (10 total):
  Initialize fresh Uniko instance
  tasks = load_tasks(dataset)
  results_early = []
  results_late = []

  For i, task in enumerate(tasks):
    1. recall(task.question) → ContextBundle → LLM generates answer
    2. score(predicted, expected) → append to results_early (if i < len/3) or results_late (if i > 2*len/3)
    3. record_episode(action_type="answer", outcome=score, state={task}, delta={answer})
       → Episode created, linked to entities, observations
    4. If i % 10 == 0: trigger consolidation
       → P4 derives/reinforces/invalidates Facts
       → Procedural memory: detect patterns in success/failure sequences

  improvement_delta = mean(results_late) - mean(results_early)
  → Must be > 0 (system gets better over time)
```

**Key metric: improvement_delta**

```
improvement_delta = accuracy[last_third_of_tasks] - accuracy[first_third_of_tasks]

If delta > 0: system is learning — consolidation and fact derivation are working
If delta <= 0: system is NOT learning — consolidation may be ineffective or recall is not benefiting from derived facts
```

**Target:** improvement_delta > 0 for all 10 datasets

**Phase1_only_pct tracking:**

```
Track phase1_only_pct at task intervals:
  - After task 10: expect low phase1_only_pct (few facts derived yet)
  - After task 50: expect increasing phase1_only_pct (consolidation building facts)
  - After task 100: expect higher phase1_only_pct (rich fact graph)

Trend: must be monotonically increasing (with local noise tolerated)
```

---

### 16.6 — Performance Optimization & Competitive Analysis

**Objective:** Profile all benchmarks, tune system parameters, compare against competitors, and produce final published numbers.

#### Files

| File | Type | Purpose |
|---|---|---|
| `benchmarks/src/profiling.rs` | Rust | Profiling helpers, flamegraph integration |
| `benchmarks/results/` | Output | Generated result files (JSON, markdown) |

#### Profiling

- Generate flamegraphs for each benchmark run using `pprof` or `cargo-flamegraph`
- Identify hot paths in:
  - Recall cascade (graph traversal, vector search, fulltext search)
  - Consolidation (observation → fact derivation, BTIC operations)
  - Context assembly (token counting, relevance ranking, budget enforcement)
  - Embedding computation (model inference latency)

#### Parameter Tuning

| Parameter | Default | Tuning Range | What It Affects |
|---|---|---|---|
| `consolidation_threshold` | 20 | 10-50 | How often facts are derived (lower = more frequent, higher quality but slower) |
| `consolidation_interval_secs` | 900 | 300-3600 | Timer-based consolidation trigger |
| `phase1_coverage_threshold` | 0.75 | 0.5-0.95 | When Phase 1 is "good enough" (lower = faster, higher = more graph-dependent) |
| `phase2_coverage_threshold` | 0.65 | 0.4-0.85 | When to fall through to fulltext |
| `half_life_days` | 30.0 | 7.0-90.0 | Memory decay rate |
| `max_chunk_tokens` | 512 | 256-1024 | Chunk granularity |
| `min_chunk_tokens` | 64 | 32-128 | Minimum chunk size |

Tuning approach:
1. Run each benchmark with default parameters, record baseline
2. Sweep one parameter at a time, record scores
3. Select parameter values that maximize overall score across all benchmarks
4. Document final parameter choices with justification

#### Competitive Comparison Table

```rust
/// Generate the final comparison table.
pub fn generate_comparison_table(results: &[BenchmarkResult]) -> String;
```

Known competitor scores (from published papers/evaluations):

| System | LoCoMo | LongMemEval | MemoryAgentBench | BEAM (1M) | Evo-Memory |
|--------|--------|-------------|------------------|-----------|------------|
| Mem0 | 91.6% | — | — | — | — |
| Graphiti | 59.5% | — | — | — | — |
| Letta | 58.4% | — | — | — | — |
| MemGPT | 49.8% | — | — | — | — |
| Zep | — | ~75% | — | — | — |
| Hindsight | — | — | — | 64.1% | — |
| **uniko** | **target 75%+** | **target 70%+** | **target 60%+** | **target 50%+** | **delta > 0** |

#### NF Latency Targets Validation

| Metric | Target | Source |
|---|---|---|
| assembly_latency (recall) | < 100ms (p95) | NF: Context Assembly |
| ingest_latency (per message) | < 5s end-to-end | NF6 |
| consolidation_cycle | < 30s per cycle | NF16 |
| compression_ratio | > 0.1 | Spec: ratio of context_tokens to total_tokens |
| phase1_only_pct | trending up | Spec: must increase with consolidation |

---

## Test Plan

### Unit Tests

| Test | File | What It Validates |
|---|---|---|
| `test_f1_score_exact` | `metrics.rs` | F1 = 1.0 when predicted == expected |
| `test_f1_score_partial` | `metrics.rs` | F1 correctly computed for partial overlap |
| `test_f1_score_no_overlap` | `metrics.rs` | F1 = 0.0 when no token overlap |
| `test_exact_match_normalized` | `metrics.rs` | Case-insensitive, punctuation-stripped comparison |
| `test_context_contains_answer` | `metrics.rs` | Context string containing expected answer → true |
| `test_context_missing_answer` | `metrics.rs` | Context string missing expected answer → false |
| `test_precision_recall` | `metrics.rs` | Precision and recall individually correct |
| `test_normalize_for_scoring` | `metrics.rs` | Normalization strips punctuation, lowercases, collapses whitespace |
| `test_phase1_only_pct` | `metrics.rs` | Correct percentage after mixed phase1-only and multi-phase recalls |
| `test_compression_ratio` | `metrics.rs` | Ratio correctly computed as context/total |
| `test_p95_latency` | `metrics.rs` | 95th percentile computed correctly from samples |
| `test_load_json` | `data.rs` | JSON dataset file parsed into correct structs |
| `test_load_jsonl` | `data.rs` | JSONL dataset parsed line-by-line |
| `test_load_csv` | `data.rs` | CSV dataset parsed correctly |
| `test_parse_conversations_locomo` | `data.rs` | LoCoMo format → Conversation structs |
| `test_report_json_format` | `reporter.rs` | JSON output has correct structure and all fields |
| `test_report_markdown_table` | `reporter.rs` | Markdown table renders correctly with alignment |
| `test_report_comparison` | `reporter.rs` | Comparison table includes competitor data |
| `test_harness_reset` | `harness.rs` | Reset creates fresh Uniko instance, old data gone |

### Integration Tests

| Test | File | What It Validates |
|---|---|---|
| `test_locomo_small_ingest` | `locomo.rs` | Ingest 1 conversation (subset), verify entities/observations/facts created |
| `test_locomo_single_question` | `locomo.rs` | Single LoCoMo question produces a scored answer |
| `test_locomo_adversarial` | `locomo.rs` | Adversarial question correctly checks SENT_BY attribution |
| `test_locomo_temporal` | `locomo.rs` | Temporal question correctly orders events via BTIC |
| `test_longmemeval_knowledge_update` | `longmemeval.rs` | Old fact invalidated, new fact returned after contradiction |
| `test_longmemeval_abstention` | `longmemeval.rs` | System correctly abstains when no relevant information |
| `test_memoryagentbench_conflict` | `memoryagentbench.rs` | Conflict resolution returns current value, not outdated |
| `test_beam_scale_1m` | `beam.rs` | System handles 33K+ messages, recall latency within target |
| `test_evomemory_improvement` | `evomemory.rs` | improvement_delta > 0 on at least one dataset |
| `test_evomemory_phase1_trend` | `evomemory.rs` | phase1_only_pct increases over sequential tasks |
| `test_full_benchmark_suite` | `lib.rs` | All 5 benchmarks run end-to-end, results produced |

### Performance Tests

| Test | Target | What It Validates |
|---|---|---|
| `bench_recall_assembly_latency` | < 100ms p95 | Context assembly at various system sizes |
| `bench_ingest_throughput` | > 100 messages/sec | Sustained ingestion rate |
| `bench_consolidation_cycle` | < 30s per cycle | Consolidation at scale (1000+ observations) |
| `bench_scoring_throughput` | < 1ms per question | Scoring functions are negligible cost |

### Property-Based Tests

| Test | What It Validates |
|---|---|
| `proptest_f1_symmetric` | F1(a, b) is not necessarily symmetric, but F1(a, a) == 1.0 |
| `proptest_f1_bounded` | F1 is always in [0.0, 1.0] |
| `proptest_precision_recall_f1_consistent` | F1 = 2*P*R/(P+R) holds for all inputs |
| `proptest_compression_ratio_bounded` | Compression ratio is in [0.0, 1.0] for valid inputs |

---

## Documentation Plan

| Artifact | Location | Contents |
|---|---|---|
| Crate-level doc comment | `benchmarks/src/lib.rs` | Overview of benchmark suite, how to run, what each benchmark tests |
| `Benchmark` trait doc | `benchmarks/src/harness.rs` | How to implement a new benchmark, data format, scoring contract |
| `BenchMetrics` doc | `benchmarks/src/metrics.rs` | All metrics tracked, formulas for F1/precision/recall/exact_match |
| Per-benchmark docs | `benchmarks/src/{locomo,longmemeval,...}.rs` | Dataset format, question types, scoring method, target scores |
| Results README | `benchmarks/results/README.md` | How to interpret results, what each column means |
| Competitor data sources | `benchmarks/src/reporter.rs` | Where each competitor score comes from (paper, evaluation, blog) |

---

## Review Checklist

- [ ] `benchmarks/` is a workspace member in root `Cargo.toml`
- [ ] `benchmarks/Cargo.toml` depends on `uniko-api` only (not L1/L2/L3 crates directly)
- [ ] `Benchmark` trait has all 7 methods: name, load_data, ingest, query, score, questions, token_budgets
- [ ] `BenchmarkHarness` creates a fresh Uniko instance and supports reset between benchmarks
- [ ] Token budget enforcement: context assembly never exceeds budget
- [ ] F1 scoring: `f1_score("the cat sat", "the cat") = 2*(2/3)*(2/2)/(2/3+2/2) = 0.8`
- [ ] Exact match: case-insensitive, punctuation-stripped
- [ ] `context_contains_answer` correctly handles substring vs token matching
- [ ] phase1_only_pct tracked per recall and aggregated
- [ ] compression_ratio computed as total_context_tokens / total_stored_tokens
- [ ] LoCoMo: 10 conversations, 5882 turns, 1986 questions across 5 types
- [ ] LoCoMo ingestion: P1 → P2 → P3 → P4 (periodic) → P7
- [ ] LoCoMo adversarial: SENT_BY attribution verified
- [ ] LoCoMo temporal: BTIC intervals and session timestamps used
- [ ] LongMemEval: 500 questions, 5 abilities, 115K-500K tokens
- [ ] LongMemEval knowledge updates: BTIC invalidation produces correct current answer
- [ ] LongMemEval abstention: system correctly reports "no information"
- [ ] MemoryAgentBench: 2071 questions, 4 competencies, 100K-1.4M tokens
- [ ] MemoryAgentBench conflict resolution: contradicting chunks → correct current answer
- [ ] BEAM: 400 questions, 4 scales (128K, 500K, 1M, 10M), 10 abilities
- [ ] BEAM scale stress: 33K+ messages handled, entity dedup at scale
- [ ] BEAM target: 50%+ at 1M tokens
- [ ] Evo-Memory: 10 datasets, 8 task types, sequential stream
- [ ] Evo-Memory: improvement_delta = accuracy[late] - accuracy[early] > 0
- [ ] Evo-Memory: phase1_only_pct trends upward
- [ ] Evo-Memory: consolidation triggered every 10 tasks
- [ ] Reporter outputs valid JSON
- [ ] Reporter outputs valid markdown table
- [ ] Comparison table includes all known competitor scores
- [ ] Flamegraph profiling produces actionable hot-path data
- [ ] Parameter tuning documented with justification
- [ ] All NF latency targets validated: assembly < 100ms, ingest < 5s, consolidation < 30s
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] All property-based tests pass

---

## Definition of Done

1. **Harness functional:** `BenchmarkHarness` creates, resets, and runs benchmarks. `Benchmark` trait implemented by all 5 benchmarks. Data loading works for JSON/JSONL/CSV formats.
2. **Scoring accurate:** F1, precision, recall, exact_match, and context_contains_answer all produce correct values verified by unit tests. Normalization (lowercase, strip punctuation) applied consistently.
3. **LoCoMo complete:** All 10 conversations ingested, 1986 questions scored, results broken down by question type. Adversarial (SENT_BY) and temporal (BTIC) validations pass. Overall score 75%+.
4. **LongMemEval complete:** All 500 questions scored across 5 abilities. Knowledge updates correctly return current values via BTIC invalidation. Abstention correctly reports "no information." Overall score 70%+.
5. **MemoryAgentBench complete:** All 2071 questions scored across 4 competencies. Conflict resolution returns current values. Overall score 60%+.
6. **BEAM complete:** All 400 questions scored at 4 scales. Scale stress test passes at 1M tokens (33K+ messages handled). Overall score 50%+ at 1M. Assembly latency < 100ms even at scale.
7. **Evo-Memory complete:** All 10 datasets evaluated. improvement_delta > 0 for all datasets. phase1_only_pct trends upward. Consolidation improves answer quality over time.
8. **Key metrics validated:** phase1_only_pct trends up across all benchmarks. compression_ratio > 0.1. assembly_latency < 100ms (p95). consolidation_cycle < 30s.
9. **Comparison table published:** uniko vs Mem0, Graphiti, Letta, Zep, MemGPT, Hindsight. All competitor scores sourced from published papers/evaluations.
10. **Parameters tuned and documented:** Consolidation threshold, recall cascade thresholds, memory decay, chunk sizing all tuned with benchmark-driven justification.
11. **All tests pass:** Unit, integration, performance, and property-based tests green. `cargo nextest run -n auto -p benchmarks` passes.
12. **Results reproducible:** Running the same benchmark suite twice produces scores within 2% variance (accounting for LLM non-determinism).
