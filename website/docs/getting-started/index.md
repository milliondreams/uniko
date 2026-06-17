# Getting Started

uniko is an embedded, Rust-native cognitive memory system for AI agents. It links into
your host process like SQLite does — there is no service to run, no network hop, and no
separate vector store to keep consistent. You feed it `Message`s and it compiles them into
a typed knowledge graph (`Entity`, `Observation`, `Fact`, `Procedure`, `Topic`) with full
provenance, then answers queries against that compiled knowledge.

This section gets a `KnowledgeBase` open and ingesting in your own Rust project.

<div class="feature-grid">
<div class="feature-card">
### [Installation](installation.md)
Add uniko to your Cargo workspace and pull in the uni-db engine it builds on.
</div>
<div class="feature-card">
### [Quick Start](quickstart.md)
Open a `KnowledgeBase`, ingest a `Message`, and run your first recall — end to end in Rust.
</div>
</div>

!!! note "uniko is a Rust library"
    You use uniko by depending on its crates and calling its async APIs from Rust. Ingest
    runs entirely locally — entity and observation extraction goes through an ONNX model
    cascade, with **zero LLM tokens per message** by default.

## Recommended path

1. **[Install](installation.md)** — add the `uniko-api` facade (and its uni-db path
   dependency) to your workspace. uniko targets the Rust 2024 edition on the stable
   toolchain.
2. **[Quick Start](quickstart.md)** — open a `KnowledgeBase`, ingest a few `Message`s, and
   issue a recall query to see the compile-once / query-forever flow in action.
3. **Learn the model** — read the [Concepts](../concepts/architecture.md) to understand how
   `Message`s become `Observation`s, `Fact`s, and `Procedure`s, and how the recall cascade
   assembles a `ContextBundle`.

!!! tip "Start small, then scale"
    A single `KnowledgeBase` is enough to follow the Quick Start. When you later run many
    KnowledgeBases in one process, they can share one ONNX `ModelRuntime` via
    `KnowledgeBase::build_shared_runtime` and `open_with_runtime` to keep VRAM use low.

## Where to go next

- **[Concepts: Architecture](../concepts/architecture.md)** — the layered crate stack and
  the P1–P7 pipelines that turn messages into knowledge.
- **[Concepts: Memory Model](../concepts/memory-model.md)** — the five memory types
  (working, episodic, semantic, procedural, meta) and the graph nodes behind them.
