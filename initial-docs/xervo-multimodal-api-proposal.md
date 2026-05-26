# Proposal — Multimodal, NLP, VLM, ASR, OCR API for uni-xervo

| | |
|---|---|
| **Status** | Draft for review |
| **Author** | uniko team (consumer of uni-xervo) |
| **Audience** | uni-xervo / uni-db maintainers |
| **Date** | 2026-05-25 |
| **Scope** | Net-new public API on `UniXervo` + corresponding traits in `uni-xervo::traits` + corresponding resolvers on `ModelRuntime` + `ModelTask` enum extensions. No provider implementations land in this PR — those follow as separate consumer-side work. |

---

## 0. TL;DR

Add seven new capabilities to xervo as **siblings of `embed`, `generate`, `rerank`**, following the same one-method-per-trait pattern xervo already uses. Each capability is alias-dispatched, returns typed output, and is opt-in for providers (a provider implements only the trait(s) it can satisfy).

| Capability | Facade method | Provider trait | Catalog `ModelTask` variant |
|---|---|---|---|
| Image embedding | `embed_image` | `ImageEmbedder` | `ImageEmbed` |
| Audio embedding | `embed_audio` | `AudioEmbedder` | `AudioEmbed` |
| Mixed-modality embedding | `embed_multimodal` | `MultimodalEmbedder` | `MultimodalEmbed` |
| Structured NLP cascade | `nlp_analyze` | `NlpModel` | `Nlp` |
| Document VLM extraction | `vlm_extract` | `VlmExtractor` | `VlmExtract` |
| Speech-to-text | `transcribe` | `Transcriber` | `Transcribe` |
| OCR | `ocr` | `OcrModel` | `Ocr` |

The proposal does **not** propose backend changes (no PDF renderer in xervo, no audio decoder in xervo, no training infrastructure, no schema changes to uni-store, no changes to `Value`/`DataType` since `Bytes` and `Vector` already exist). It is a pure API-surface PR for inference dispatch.

---

## 1. Context & Motivation

### 1.1 The consumer

uniko is a cognitive memory system for AI agents built on uni-db. It consumes xervo for every model call: text generation, text embedding, cross-encoder reranking, and (via the `raw_tensor_model` escape hatch) a multi-head DeBERTa NLP cascade that produces POS / NER / DEP / SRL / CLS labels on every ingested message.

Current xervo usage by uniko:

- `xervo().embed(alias, &[&str])` — chunk and observation embeddings.
- `xervo().generate(alias, messages, options)` — answer generation, judge calls, NL-to-Cypher.
- `xervo().rerank(alias, query, &[&str])` — Phase 1+ recall reranking.
- `xervo().raw_tensor_model(alias)` — kniv-deberta NLP cascade in `uniko-extract/src/nlp/`, called per message on the hot ingest path. Tokenization, forward, label decode all live in uniko caller code.

### 1.2 The emerging gap

uniko's Artifact schema (`uniko-store/src/schema/artifacts.rs`) reserves five embedding columns (`text_embedding`, `image_embedding`, `audio_embedding`, `video_embedding`, `multimodal_embedding`) with full vector indexes registered in uni-db. Only `text_embedding` is reachable today, and only indirectly via `Chunk.text` auto-embed — the parent Artifact's `text_embedding` itself is never populated because we have no pooling path.

The other four are schema-defined, indexed, and zero-populated. We have no path to:

1. Embed image bytes → `image_embedding`.
2. Embed audio (raw or transcript-paired) → `audio_embedding`.
3. Embed mixed-modality blocks (text+image together) → `multimodal_embedding`.
4. Run a VLM-based document parser (MinerU 2.5, Granite-Docling, olmOCR-2) to lift PDF / scan content into structured blocks.
5. Transcribe audio via Whisper for the Message-level ingest path.
6. OCR image-only artifacts.

The 2026 production landscape (SigLIP 2, ColQwen2-VL, MinerU 2.5, Whisper, CLAP, Cohere Embed v4, Gemini Embedding 2) consists of models that produce these outputs and ONNX-export cleanly. The host project's constraint is **no Python in the runtime** — every model must run in-process via `ort` / `candle` / `mistralrs` / `whisper.cpp`. xervo already abstracts over those provider backends for `generate` / `embed` / `rerank`; extending the pattern is the natural fit.

### 1.3 Why now

- **Audit found `Artifact.text_embedding` never gets populated** even though the index exists and queries succeed (returning empty). This is a real retrieval bug, not a future enhancement.
- **uniko's NLP cascade is the existence proof** that non-text, non-vector dispatch is viable through xervo's catalog system — but it uses the `raw_tensor_model` escape hatch because there's no managed trait for structured-output models. Pre/post-processing lives in caller code, which prevents the per-call observability, retry, and circuit-breaker discipline that managed embedders and generators get for free.
- **Competitive landscape**: no embedded multimodal agent-memory system exists today. Mem0, Zep/Graphiti, Letta, LangMem are text-only. Cognee handles multimodal but flattens to text before graph extraction. The uniko + uni-xervo stack is positioned to occupy that gap once the xervo API surface exists.

---

## 2. Requirements

### 2.1 Functional

**R1.** Image embedding. Given an alias and a batch of images, return one float vector per image.

**R2.** Audio embedding. Given an alias and a batch of audio inputs, return one float vector per input. Audio inputs must be acceptable as raw bytes (with MIME), as decoded PCM (sample-rate-tagged), or as a path.

**R3.** Mixed-modality embedding. Given an alias and a batch of inputs where each input is a sequence of `ContentBlock` (text / image / audio together), return one float vector per input. Supports models like Cohere Embed v4, Jina Embeddings v4, Gemini Embedding 2 that produce a single vector spanning multiple modalities per call.

**R4.** Structured NLP analysis. Given an alias and a batch of `(text, requested_tasks)`, return per-text structured output: token-level POS / NER / DEP, sentence boundaries, optional SRL frames, optional speech-act classification. Output shape must be JSON-serializable and version-stable across model swaps.

**R5.** VLM document extraction. Given an alias and a batch of pre-rendered page images, return structured per-page blocks (text / heading / figure / table / formula) with optional bounding boxes and a concatenated Markdown convenience field. Caller renders PDFs to images upstream — xervo does not own PDF rendering.

**R6.** Speech-to-text transcription. Given an alias and one audio input plus options, return language detection, segments with start/end timestamps, optional word-level timestamps, optional speaker diarization.

**R7.** OCR. Given an alias and a batch of images, return per-image blocks (text + bbox + confidence) and a concatenated plain-text convenience field.

### 2.2 Non-functional

**N1. Pattern consistency.** Every new method MUST mirror the existing `embed` / `generate` / `rerank` shape: facade method takes alias + typed inputs (optionally typed options), delegates to a `ModelRuntime` resolver, which returns an `Arc<dyn TraitObject>` from a provider implementation.

**N2. One-method-narrow traits.** Each new trait has **one** business method (plus the `metadata()`-style accessors traits typically need). Mirrors `Embedder::embed`, `Generator::generate`, `Reranker::rerank`. This keeps provider implementations focused.

**N3. Selective provider opt-in.** A provider implementing `local/onnx` MAY implement `ImageEmbedder` (for SigLIP-2) without implementing `Transcriber`. A new `local/whisper-cpp` provider MAY implement only `Transcriber`. Trait dispatch decouples capabilities from provider identity.

**N4. Backwards compatibility.** No changes to `embed` / `generate` / `rerank` / `raw_tensor_model` signatures. The `raw_tensor_model` escape hatch remains available indefinitely — new managed traits add an option, not a replacement.

**N5. Async + Send + Sync.** Every facade method is `async fn` returning `Result<T>`; every trait method is `async fn` returning `Result<T>`. Trait objects must be `Send + Sync` so they can be cached in the runtime and called from any tokio task. Matches existing patterns.

**N6. Feature gating.** Provider implementations are feature-gated (`provider-onnx`, future `provider-whisper-cpp`, etc.). The new traits themselves are unconditional; only their **impl** sites are gated. Existing pattern: `cfg(feature = "provider-onnx")` on `raw_tensor_model`.

**N7. Reuse existing types.** `ImageInput`, `ContentBlock`, `TokenUsage` already exist in `uni_xervo::traits` and are re-exported through the facade. New types are added only where existing types don't fit (e.g., `AudioInput`, `NlpResult`, `DocExtractResult`).

**N8. Per-call cost reporting where applicable.** Methods that hit a remote provider (future `remote/cohere-embed-multimodal`, `remote/gemini-embedding-2`) MUST be able to return token usage. Method signatures must accommodate optional usage data without forcing local providers to fabricate it.

### 2.3 Out of scope

This PR does NOT propose:

- **PDF rendering in xervo.** Callers convert PDFs to images via a renderer they own (uniko uses `pdfium-render`). Pulling PDF parsing into uni-db is rejected: it expands the dependency surface for a concern that doesn't belong in a model-dispatch layer.
- **Audio decoding in xervo.** The `AudioInput::Bytes` variant carries a MIME so providers can route through their preferred decoder (whisper.cpp accepts WAV / FLAC / MP3 natively; PCM variant skips decode entirely). xervo itself doesn't decode.
- **Modality auto-detection.** Callers select alias by intent (`embed/image` vs `embed/text`); xervo doesn't sniff content.
- **Multi-vector / ColBERT late interaction.** ColPali/ColQwen2-style retrieval requires a multi-vector index in uni-db, which is a separate uni-db enhancement. This proposal targets single-vector outputs only.
- **Training, fine-tuning, model conversion.** Inference dispatch only.
- **uni-store / schema changes.** `Value::Bytes` and `Value::Vector` already exist with the right shape.
- **Provider implementations.** Each modality's provider impl is its own follow-up PR. This PR is the contract only.
- **Caller migration.** uniko's switch from `raw_tensor_model` to `nlp_analyze` is a separate uniko PR that depends on this one but doesn't ship with it.

---

## 3. Current state of xervo (reference)

### 3.1 Facade surface (verified — `crates/uni/src/api/xervo.rs`)

```rust
pub struct UniXervo { runtime: Option<Arc<ModelRuntime>> }

impl UniXervo {
    pub fn is_available(&self) -> bool;

    pub async fn embed(&self, alias: &str, texts: &[&str])
        -> Result<Vec<Vec<f32>>>;

    pub async fn generate(&self, alias: &str, messages: &[Message],
        options: GenerationOptions) -> Result<GenerationResult>;

    pub async fn generate_text(&self, alias: &str, messages: &[&str],
        options: GenerationOptions) -> Result<GenerationResult>;

    pub async fn rerank(&self, alias: &str, query: &str, documents: &[&str])
        -> Result<Vec<ScoredDoc>>;

    #[cfg(feature = "provider-onnx")]
    pub async fn raw_tensor_model(&self, alias: &str)
        -> Result<Arc<dyn RawTensorModel>>;

    pub async fn prefetch_all(&self) -> Result<()>;
    pub async fn prefetch(&self, aliases: &[&str]) -> Result<()>;
    pub fn raw_runtime(&self) -> Option<&Arc<ModelRuntime>>;
}
```

### 3.2 Trait pattern (existing)

- `Embedder` — one method `embed(Vec<String>) -> Vec<Vec<f32>>`.
- `Generator` — one method `generate(&[Message], GenerationOptions) -> GenerationResult`.
- `Reranker` — one method `rerank(&str, &[&str]) -> Vec<ScoredDoc>`.
- `RawTensorModel` — escape hatch returning a model handle for tokenize/forward/decode in caller code.

`ModelRuntime` resolves alias → `Arc<dyn TraitObject>`: `runtime.embedding(alias)`, `runtime.generator(alias)`, `runtime.reranker(alias)`, `runtime.raw_tensor_model(alias)`.

### 3.3 What `raw_tensor_model` enables, and what it costs

The escape hatch is **load-bearing** — uniko's kniv-deberta cascade lives in `uniko-extract/src/nlp/mod.rs` and pulls a `RawTensorModel` runner. The full cascade (tokenize → forward → decode 5 label arrays → reconstruct dependency tree → emit SRL frames) is implemented in uniko, not in xervo.

This works but has costs:

- **No managed observability** — `metrics` emission for NLP-cascade latency / token counts / failure rates lives in uniko code, parallel to xervo's metrics for embed/generate.
- **No managed retry / circuit breaker** — kniv OOM on >6k inputs (real recurring issue) requires manual handling in uniko, where embed and generate get xervo's standard recovery infrastructure.
- **Postprocessing is not portable** — another consumer that wanted POS or DEP labels couldn't reuse uniko's decoder without copying the code.
- **Catalog can't express "this alias is a NER model"** — `ModelTask` today is `Generate | Embed | Rerank`. NLP is dispatched purely by alias-name convention.

The proposal preserves `raw_tensor_model` (some models will always need it) but adds managed traits so the common cases stop using it.

---

## 4. Solution design

### 4.1 Design principles

1. **Mirror existing patterns.** Every new capability gets one facade method, one trait with one business method, one runtime resolver, one `ModelTask` variant. No exceptions.
2. **Reuse existing types where they fit.** `ImageInput`, `ContentBlock`, `TokenUsage`, `Message` already cover most multimodal cases. Add new types only where shape genuinely differs.
3. **Trait per modality, not per model.** `ImageEmbedder` is the contract; SigLIP-2-ONNX, Nomic-Embed-Vision-ONNX, remote/cohere-embed-v4 are interchangeable behind it.
4. **Provider drivers opt in selectively.** `local/onnx` implements the embedder/NLP/OCR/VLM traits it can handle; a future `local/whisper-cpp` implements only `Transcriber`. Catalog declares which alias maps to which trait via `ModelTask`.
5. **No new runtime concerns.** Lifecycle (warmup, prefetch), error handling (`Result<T, UniError>`), threading (`Send + Sync` trait objects), and async (`async fn` everywhere) all match existing surfaces verbatim.

### 4.2 `ModelTask` extension

```rust
pub enum ModelTask {
    // Existing
    Generate,
    Embed,
    Rerank,

    // New
    ImageEmbed,
    AudioEmbed,
    MultimodalEmbed,
    Nlp,
    VlmExtract,
    Transcribe,
    Ocr,
}
```

The catalog (`ModelAliasSpec`) gains the ability to declare any of these task variants. Provider drivers route to the appropriate trait based on declared task.

### 4.3 New traits

```rust
// === New: image embedding ============================================

#[async_trait::async_trait]
pub trait ImageEmbedder: Send + Sync {
    async fn embed_images(&self, images: Vec<ImageInput>)
        -> Result<Vec<Vec<f32>>>;
    fn dimensions(&self) -> usize;
}

// === New: audio embedding ============================================

#[async_trait::async_trait]
pub trait AudioEmbedder: Send + Sync {
    async fn embed_audios(&self, audios: Vec<AudioInput>)
        -> Result<Vec<Vec<f32>>>;
    fn dimensions(&self) -> usize;
}

// === New: mixed-modality embedding ===================================

#[async_trait::async_trait]
pub trait MultimodalEmbedder: Send + Sync {
    async fn embed_multimodal(&self, inputs: Vec<MultimodalInput>)
        -> Result<Vec<Vec<f32>>>;
    fn dimensions(&self) -> usize;
    fn supported_modalities(&self) -> &[Modality];
}

// === New: structured NLP cascade =====================================

#[async_trait::async_trait]
pub trait NlpModel: Send + Sync {
    async fn analyze(&self, requests: Vec<NlpRequest<'_>>)
        -> Result<Vec<NlpResult>>;
    fn supported_tasks(&self) -> NlpTasks;
}

// === New: VLM document extraction ====================================

#[async_trait::async_trait]
pub trait VlmExtractor: Send + Sync {
    async fn extract(&self, pages: Vec<ImageInput>,
        options: DocExtractOptions) -> Result<Vec<DocExtractResult>>;
}

// === New: speech-to-text =============================================

#[async_trait::async_trait]
pub trait Transcriber: Send + Sync {
    async fn transcribe(&self, audio: AudioInput,
        options: TranscribeOptions) -> Result<TranscribeResult>;
    fn supported_languages(&self) -> &[String];
}

// === New: OCR ========================================================

#[async_trait::async_trait]
pub trait OcrModel: Send + Sync {
    async fn recognize(&self, images: Vec<ImageInput>)
        -> Result<Vec<OcrResult>>;
}
```

### 4.4 New types

```rust
// --- AudioInput ------------------------------------------------------

/// An audio input to a transcriber or audio embedder.
///
/// Three variants because real callers come in three shapes: HTTP
/// handlers have raw `Vec<u8>` with a MIME, in-process audio capture
/// has decoded PCM, file-based ingest has a path.
pub enum AudioInput {
    /// Raw container bytes (WAV, MP3, FLAC, …). Provider decides how
    /// to decode based on `mime`.
    Bytes { mime: String, data: Vec<u8> },
    /// Pre-decoded PCM samples. Provider skips decode.
    Pcm { sample_rate: u32, channels: u16, samples: Vec<f32> },
    /// Local file path. Provider reads + decodes.
    Path { path: std::path::PathBuf },
}

// --- MultimodalInput -------------------------------------------------

/// A heterogeneous input for `embed_multimodal`. Reuses `ContentBlock`
/// so a single input can carry text + image + audio together — exactly
/// what Cohere Embed v4 / Jina v4 / Gemini Embedding 2 expect.
pub struct MultimodalInput {
    pub blocks: Vec<ContentBlock>,
}

pub enum Modality { Text, Image, Audio, Video }

// --- NLP types -------------------------------------------------------

bitflags::bitflags! {
    pub struct NlpTasks: u32 {
        const POS = 1 << 0;
        const NER = 1 << 1;
        const DEP = 1 << 2;
        const SRL = 1 << 3;
        const CLS = 1 << 4;  // speech-act classification
        const ALL = Self::POS.bits() | Self::NER.bits() | Self::DEP.bits()
                  | Self::SRL.bits() | Self::CLS.bits();
    }
}

pub struct NlpRequest<'a> {
    pub text: &'a str,
    pub tasks: NlpTasks,
}

pub struct NlpResult {
    pub tokens: Vec<NlpToken>,
    pub sentences: Vec<NlpSentence>,
    /// Populated only if `NlpTasks::SRL` was requested AND model supports SRL.
    pub frames: Vec<SrlFrame>,
    /// Populated only if `NlpTasks::CLS` was requested AND model supports CLS.
    pub speech_acts: Vec<SpeechAct>,
}

pub struct NlpToken {
    /// Surface form.
    pub text: String,
    /// UTF-8 byte offset in the original text. Closed-open [start, end).
    pub start: usize,
    pub end: usize,
    /// POS tag (Universal Dependencies tagset). `None` if POS not requested.
    pub pos: Option<String>,
    /// Named-entity type (e.g., "PERSON", "ORG"). `None` outside an entity.
    pub ner: Option<String>,
    /// Head token index in this sentence's token list, plus dep relation.
    /// `None` if DEP not requested.
    pub dep: Option<DepLink>,
}

pub struct NlpSentence {
    /// Inclusive [first, last] token indices into `NlpResult.tokens`.
    pub token_range: (usize, usize),
    pub start: usize,
    pub end: usize,
}

pub struct DepLink { pub head: usize, pub relation: String }

pub struct SrlFrame {
    pub predicate_token: usize,
    pub predicate_sense: Option<String>,
    pub roles: Vec<SrlRole>,
}

pub struct SrlRole {
    /// Inclusive [first, last] token indices.
    pub span: (usize, usize),
    pub label: String,        // e.g., "ARG0", "ARG1", "ARGM-TMP"
}

pub struct SpeechAct {
    pub sentence_index: usize,
    pub label: String,        // e.g., "STATEMENT", "QUESTION", "GREETING"
    pub confidence: f32,
}

// --- VLM extraction types --------------------------------------------

pub struct DocExtractOptions {
    pub output: DocOutputFormat,
    pub include_tables: bool,
    pub include_formulas: bool,
    pub include_bboxes: bool,
}

pub enum DocOutputFormat { Markdown, Json, Html }

pub struct DocExtractResult {
    pub blocks: Vec<DocBlock>,
    /// Concatenated Markdown view (convenience).
    pub plain_markdown: String,
}

pub struct DocBlock {
    pub kind: DocBlockKind,
    pub content: String,                     // MD for text/heading/table; LaTeX for formula
    pub bbox: Option<[f32; 4]>,              // x0, y0, x1, y1 in page coords
    pub reading_order: u32,                  // monotonically increasing within a page
}

pub enum DocBlockKind { Text, Heading, List, Table, Figure, Formula, Caption, Footer, Header }

// --- ASR types -------------------------------------------------------

pub struct TranscribeOptions {
    /// `None` → auto-detect.
    pub language: Option<String>,
    pub word_timestamps: bool,
    pub diarize: bool,
    /// Optional initial prompt / context for biasing decoding.
    pub initial_prompt: Option<String>,
}

pub struct TranscribeResult {
    pub language: String,
    pub segments: Vec<TranscribeSegment>,
}

pub struct TranscribeSegment {
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
    pub speaker: Option<String>,             // populated iff diarize=true
    pub words: Vec<TranscribeWord>,          // populated iff word_timestamps=true
}

pub struct TranscribeWord {
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
    pub confidence: Option<f32>,
}

// --- OCR types -------------------------------------------------------

pub struct OcrResult {
    pub blocks: Vec<OcrBlock>,
    pub plain_text: String,
}

pub struct OcrBlock {
    pub text: String,
    pub bbox: [f32; 4],
    pub confidence: f32,
}
```

### 4.5 New facade methods

```rust
impl UniXervo {
    // Multimodal embedding

    pub async fn embed_image(&self, alias: &str, images: &[ImageInput])
        -> Result<Vec<Vec<f32>>>;

    pub async fn embed_audio(&self, alias: &str, audios: &[AudioInput])
        -> Result<Vec<Vec<f32>>>;

    pub async fn embed_multimodal(&self, alias: &str, inputs: &[MultimodalInput])
        -> Result<Vec<Vec<f32>>>;

    // Structured NLP

    pub async fn nlp_analyze(&self, alias: &str, requests: &[NlpRequest<'_>])
        -> Result<Vec<NlpResult>>;

    // VLM document extraction

    pub async fn vlm_extract(&self, alias: &str, pages: &[ImageInput],
        options: DocExtractOptions) -> Result<Vec<DocExtractResult>>;

    // Speech-to-text

    pub async fn transcribe(&self, alias: &str, audio: &AudioInput,
        options: TranscribeOptions) -> Result<TranscribeResult>;

    // OCR

    pub async fn ocr(&self, alias: &str, images: &[ImageInput])
        -> Result<Vec<OcrResult>>;
}
```

### 4.6 Runtime resolvers

```rust
impl ModelRuntime {
    // Existing
    pub async fn embedding(&self, alias: &str) -> Result<Arc<dyn Embedder>>;
    pub async fn generator(&self, alias: &str) -> Result<Arc<dyn Generator>>;
    pub async fn reranker(&self, alias: &str) -> Result<Arc<dyn Reranker>>;
    pub async fn raw_tensor_model(&self, alias: &str) -> Result<Arc<dyn RawTensorModel>>;

    // New
    pub async fn image_embedder(&self, alias: &str) -> Result<Arc<dyn ImageEmbedder>>;
    pub async fn audio_embedder(&self, alias: &str) -> Result<Arc<dyn AudioEmbedder>>;
    pub async fn multimodal_embedder(&self, alias: &str) -> Result<Arc<dyn MultimodalEmbedder>>;
    pub async fn nlp_model(&self, alias: &str) -> Result<Arc<dyn NlpModel>>;
    pub async fn vlm_extractor(&self, alias: &str) -> Result<Arc<dyn VlmExtractor>>;
    pub async fn transcriber(&self, alias: &str) -> Result<Arc<dyn Transcriber>>;
    pub async fn ocr_model(&self, alias: &str) -> Result<Arc<dyn OcrModel>>;
}
```

---

## 5. API specification — per-method rationale

### 5.1 `embed_image`

```rust
pub async fn embed_image(&self, alias: &str, images: &[ImageInput])
    -> Result<Vec<Vec<f32>>>;
```

- **Inputs**: alias + slice of images. Reuses existing `ImageInput`.
- **Output**: one vector per input. Matches `embed(&[&str])`.
- **Why bare `Vec<Vec<f32>>` not a wrapped result**: matches existing `embed`. Cost reporting deferred to §9.1 open question.
- **Batching**: callers pass a slice; provider decides actual GPU batching (typically configured per-provider).

### 5.2 `embed_audio`

```rust
pub async fn embed_audio(&self, alias: &str, audios: &[AudioInput])
    -> Result<Vec<Vec<f32>>>;
```

- **Inputs**: alias + slice of `AudioInput` (three variants: bytes, PCM, path).
- **Output**: one vector per input.
- **PCM variant matters**: whisper.cpp and CLAP both accept f32 PCM at a target sample rate natively. The PCM variant skips a decode step. The Path variant lets file-based ingest avoid loading bytes into memory before passing to the provider.

### 5.3 `embed_multimodal`

```rust
pub async fn embed_multimodal(&self, alias: &str, inputs: &[MultimodalInput])
    -> Result<Vec<Vec<f32>>>;
```

- **Inputs**: alias + slice of `MultimodalInput`, each carrying a `Vec<ContentBlock>`.
- **Output**: one vector per input.
- **Reuse `ContentBlock`**: it's already xervo's multimodal payload type (used in generation `Message.content`). No new type needed for the heterogeneous block sequence.
- **Use cases**: Cohere Embed v4 takes a mixed payload (text + image) in a single API call. Gemini Embedding 2 maps text/image/audio/video into one 3072-d space. Jina v4 has a unified pathway.

### 5.4 `nlp_analyze`

```rust
pub async fn nlp_analyze(&self, alias: &str, requests: &[NlpRequest<'_>])
    -> Result<Vec<NlpResult>>;
```

- **Inputs**: alias + slice of `(text, tasks_bitflag)`.
- **Output**: per-request structured result. Output has all fields; the bitflag tells the provider which to populate vs leave empty. Unrequested fields are `Vec::new()` / `None`.
- **Bitflag rationale**: kniv-deberta is multi-head — one forward produces all label arrays. The bitflag saves *post-processing* (label decoding, dep-tree reconstruction, SRL frame assembly), which is significant on the hot path (decoding all 5 heads roughly doubles per-call wall time vs decoding NER alone).
- **`supported_tasks()` on the trait**: a model that doesn't support SRL declares so; provider returns an error if `SRL` is requested against such a model. Catalog validation can warn at startup.

### 5.5 `vlm_extract`

```rust
pub async fn vlm_extract(&self, alias: &str, pages: &[ImageInput],
    options: DocExtractOptions) -> Result<Vec<DocExtractResult>>;
```

- **Inputs**: alias + pages (each a rendered image) + options.
- **Output**: one extraction result per page, with structured blocks and a concatenated Markdown convenience field.
- **Caller renders PDFs**: keeps xervo free of `pdfium-render` / `mupdf`. uniko-extract renders PDF pages to `ImageInput::Bytes { mime: "image/png", … }` upstream.
- **`include_bboxes`**: bbox extraction adds work for some VLMs; opt-in keeps fast path fast.

### 5.6 `transcribe`

```rust
pub async fn transcribe(&self, alias: &str, audio: &AudioInput,
    options: TranscribeOptions) -> Result<TranscribeResult>;
```

- **Inputs**: alias + one audio + options (language, word timestamps, diarization, initial prompt).
- **Output**: language + segments.
- **Single audio per call, not batch**: real ASR providers process one stream at a time; pseudo-batching has no GPU benefit. If batch semantics ever matter for a provider, that's a `transcribe_batch` follow-up.
- **`initial_prompt`**: whisper.cpp supports this for biasing — domain-specific terminology, named entity priming.

### 5.7 `ocr`

```rust
pub async fn ocr(&self, alias: &str, images: &[ImageInput])
    -> Result<Vec<OcrResult>>;
```

- **Inputs**: alias + images.
- **Output**: per-image blocks (text + bbox + confidence) + concatenated plain text.
- **Why separate from `vlm_extract`**: OCR is the document-blind case ("read text from this image"); VLM extract is the document-aware case ("understand this page's layout"). Different models (EasyOCR / Tesseract vs MinerU / Granite-Docling), different output shapes.

---

## 6. Provider implementation guidance

### 6.1 Existing providers extend selectively

`local/onnx` is the natural home for most new traits because the dominant 2026 models all ONNX-export:

- SigLIP-2-So400m → `ImageEmbedder`
- CLAP-HTSAT → `AudioEmbedder`
- kniv-deberta multi-head → `NlpModel`
- MinerU-2.5 / Granite-Docling → `VlmExtractor`
- PaddleOCR / EasyOCR → `OcrModel`

`remote/openai` and `remote/cohere` extend as APIs become available:

- Cohere Embed v4 → `MultimodalEmbedder`
- Gemini Embedding 2 → `MultimodalEmbedder`

### 6.2 New provider drivers anticipated

- **`local/whisper-cpp`** — implements only `Transcriber`. Wraps `whisper-rs` / `whisper.cpp`. Justification: whisper.cpp is the dominant high-quality OSS ASR; its native API doesn't fit the ONNX runtime path.
- **`local/candle`** — possible home for VLMs that don't ONNX-export cleanly (some Granite-Docling / MinerU variants). Candle's HF integration makes safetensors-direct loading straightforward.

### 6.3 Trait-impl-per-modality matters

A `local/onnx` provider that loads SigLIP-2 only needs to implement `ImageEmbedder`. It does NOT need to fake-implement `Transcriber`. Catalog validation at alias-registration time enforces "alias declared as `ImageEmbed` task must be served by a provider that implements `ImageEmbedder`".

---

## 7. Migration story

### 7.1 `raw_tensor_model` remains

The escape hatch stays public, stays gated on `provider-onnx`, stays unchanged. Models that need full control over tokenization / forward / decode (research models, models with novel input shapes) continue to use it.

### 7.2 uniko's kniv cascade as the reference `NlpModel`

After this PR merges, uniko's `crates/uniko-extract/src/nlp/mod.rs` (which currently uses `raw_tensor_model`) becomes the **reference implementation** of `NlpModel`. Two paths forward, in order:

1. **Lift the implementation upstream**: kniv cascade becomes a provider-side implementation in `local/onnx`, exposed as `NlpModel`. uniko-extract's NLP module thins down to "construct `NlpRequest`s and consume `NlpResult`s". Pre/post-processing leaves uniko and joins xervo.
2. **Or — leave it in uniko**: uniko-extract implements `NlpModel` itself and registers a synthetic provider. Less invasive but loses the "any consumer can use kniv via xervo" benefit. This path is fine as a transitional state.

Choice between the two is a follow-up discussion once the PR lands.

### 7.3 No breakage to existing callers

- `embed`, `generate`, `generate_text`, `rerank`, `raw_tensor_model`, `prefetch_all`, `prefetch`, `is_available`, `raw_runtime` — all unchanged.
- `ModelTask` gains variants — existing pattern matches against `Generate | Embed | Rerank` must add `_ => …` arms OR be updated. Decision: prefer making `ModelTask` `#[non_exhaustive]` (it's a public type the catalog uses; this matches the existing convention on `DataType`).
- Existing trait names unchanged. New trait names are net-new symbols.

---

## 8. Backwards compatibility

| Change | Breaking? | Mitigation |
|---|---|---|
| New facade methods on `UniXervo` | No | additive |
| New traits in `uni_xervo::traits` | No | additive |
| New `ModelTask` variants | Yes if exhaustive match | mark `#[non_exhaustive]` |
| New `ModelRuntime` resolver methods | No | additive |
| New re-exports through `uni::api::xervo` | No | additive |
| Existing methods' signatures | Unchanged | none needed |

---

## 9. Open design questions

Listed honestly. Each question has a recommended answer but warrants the xervo team's call.

### 9.1 Should embed methods return usage?

**Question**: `embed` today returns `Vec<Vec<f32>>` with no usage information. Remote embedders (OpenAI, Cohere, Gemini) DO report token usage on their embedding endpoints. If we want unified cost tracking parallel to `GenerationResult.usage`, the new embed methods should return a struct, not bare vectors.

**Options**:
- **A.** New methods return `Result<EmbedResult>` with `{ vectors, usage: Option<TokenUsage> }`. Backwards-incompatible widening of existing `embed` — needs a deprecation cycle or a `embed_v2`.
- **B.** New methods return bare `Vec<Vec<f32>>` like existing `embed`. Defer usage tracking to a separate later proposal.
- **C.** Return `EmbedResult` only for the new methods; leave existing `embed` alone. Inconsistent but minimally disruptive.

**Recommendation**: A, with `embed` gaining the wrapped return as a separate breaking change in the same PR. uniko already wants this for cost tracking and the wrap is cheap. If the xervo team prefers stability, C is acceptable.

### 9.2 NLP `tasks` bitflag — keep, or always return all?

**Question**: Should `NlpRequest` carry a bitflag selecting which heads to decode, or should the trait always return all-heads?

**Options**:
- **A.** Keep the bitflag (as drafted). Saves post-processing on hot paths.
- **B.** Always return all heads, callers ignore what they don't want. Simpler API; minor wasted cycles per call.

**Recommendation**: A. The hot ingest path measurably benefits (~30-40% of per-call wall time on kniv-xsmall is label decoding for non-NER heads), and the bitflag pattern is well-established in Rust (`bitflags!`).

### 9.3 PDF rendering — caller-side or xervo-side?

**Question**: `vlm_extract` takes `&[ImageInput]`. Callers render PDF pages upstream. Should xervo grow a PDF renderer for ergonomics?

**Options**:
- **A.** Caller-renders (as drafted). Keeps xervo dependency-light.
- **B.** xervo grows a `pdf_to_images(bytes) -> Vec<ImageInput>` helper as a separate feature-gated module.
- **C.** xervo adds a parallel `vlm_extract_pdf(bytes)` that handles both render and extract.

**Recommendation**: A. The render step is one line of `pdfium-render` for callers; pulling MuPDF or Pdfium into uni-db's dep tree for a marginal ergonomic gain is the wrong trade.

### 9.4 Per-task batching semantics

**Question**: Some traits take a slice (`embed_image`, `embed_audio`, `nlp_analyze`, `ocr`); `transcribe` takes a single item. Is "caller assembles batch, provider decides actual GPU batching" the right contract, or should xervo expose a separate batch-control surface?

**Options**:
- **A.** Provider decides (as drafted). Trait signatures expose batch shape; provider implementation chooses sub-batch size based on GPU memory.
- **B.** Trait carries batch hints — `embed_images_with_hints(images, BatchHints)`.

**Recommendation**: A. Matches existing `embed` and `rerank`. Provider-internal tuning is the right place; cross-provider batch hints are speculative.

### 9.5 Streaming for ASR and generation

**Question**: `transcribe` is one-shot. Streaming ASR (partial results, low-latency captions) is a real use case. Same for streaming generation.

**Recommendation**: **Out of scope for this PR.** Streaming has a different async shape (returns an `impl Stream<Item = Result<…>>`). Worth a separate proposal once one-shot is stable. The streaming variants would be `transcribe_streaming`, `generate_streaming`, etc. — additive, not breaking.

### 9.6 Trait method names

**Question**: Trait methods drafted as `embed_images`, `embed_audios`, `embed_multimodal`, `analyze`, `extract`, `transcribe`, `recognize`. Should they all match the facade method name verbatim (`embed_image` etc.) for symmetry?

**Recommendation**: Trait method names mirror the **modality singular** while facade methods use **modality singular as verb + plural for batch shape**. The existing pattern is `Embedder::embed`, `Generator::generate`, `Reranker::rerank` — verb without modality. The drafted trait names match that pattern. Bikeshed-tier; xervo team's call.

---

## 10. Test strategy

For the PR landing this proposal:

1. **Trait existence + object safety** — `cargo check` confirms each trait is `Send + Sync` and dyn-compatible. One test per trait constructs an `Arc<dyn TraitName>` from a no-op test implementation.
2. **`ModelTask` round-trip** — serialize / deserialize every new variant. Confirms catalog format is stable.
3. **Facade error paths** — each new facade method returns `not_configured()` when runtime is `None`. Mirror the existing `embed` / `generate` tests.
4. **Catalog validation** — alias declared as `ImageEmbed` task must resolve to an `ImageEmbedder` impl; otherwise registration fails with a clear error.
5. **No provider implementations land in this PR** — but each trait should ship with a `tests/` module containing a mock implementation that exercises the type round-trip. Mocks are useful for consumer-side testing too.

The acceptance criterion is "uniko can build against `uni-xervo` from this branch and write a `nlp_analyze` call site that compiles, with the runtime returning a not-implemented error at execution time." Live model dispatch lands in follow-up PRs.

---

## 11. Rollout plan (PR breakdown)

**PR 1 — this proposal** (the contract):
- 7 new traits.
- 7 new facade methods.
- `ModelTask` extension (+ `#[non_exhaustive]`).
- New types (`AudioInput`, `MultimodalInput`, `NlpResult`, `DocExtract*`, `Transcribe*`, `Ocr*`).
- Runtime resolvers (returning not-implemented errors initially).
- No provider implementations.

**PR 2 — `local/onnx` extension** (uni-xervo + consumer-side):
- `local/onnx` implements `ImageEmbedder` (SigLIP-2).
- `local/onnx` implements `OcrModel` if a clean ONNX OCR model is available.
- uniko consumer-side wires `Artifact.text_embedding` mean-pooling + `image_embedding` via the new methods.

**PR 3 — NLP migration** (uni-xervo + uniko):
- `local/onnx` implements `NlpModel` (kniv-deberta) — OR uniko hosts its own `NlpModel` provider as a transitional shim (per §7.2).
- uniko's `uniko-extract/src/nlp/` migrates from `raw_tensor_model` to `nlp_analyze`.

**PR 4 — Whisper** (uni-xervo):
- New `local/whisper-cpp` provider implementing `Transcriber`.

**PR 5 — VLM document parsing** (uni-xervo):
- `local/onnx` implements `VlmExtractor` for MinerU-2.5 / Granite-Docling.

**PR 6 — Multimodal embedding** (uni-xervo):
- `MultimodalEmbedder` impls for `remote/cohere` and `remote/gemini` when their multimodal endpoints stabilize.

Each PR after #1 is independent and can ship in any order.

---

## 12. Risks

| Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|
| New trait shape doesn't fit a future model we haven't seen | Medium | Medium | `raw_tensor_model` escape hatch remains. New trait can be added or extended later. |
| `bitflags` dependency unwelcome in uni-xervo | Low | Low | Replace with a plain enum-set struct if preferred; no behavior change. |
| `async_trait` overhead concerns | Low | Low | Existing traits use it already. If xervo wants to migrate to native async-fn-in-traits later, the migration is mechanical. |
| `#[non_exhaustive]` on `ModelTask` breaks downstream pattern matches | Low | Low | Already the convention on `DataType`. Document in CHANGELOG. |
| Spec ossification — `NlpResult` shape becomes hard to evolve | Medium | Medium | Versioning sub-types (`NlpResult` → `NlpResultV1`) is a routine pattern; `NlpResult` stays the latest. SRL/CLS labels are model-specific strings, not enum variants, so model-side evolution is non-breaking. |
| `MultimodalInput::blocks` overlapping with `Message::content` semantics confuses consumers | Low | Medium | Documentation must call out: `Message` is for generation context; `MultimodalInput` is for embedding payload. They share `ContentBlock` because the block shape is the same. |
| Audio decode quality varies by provider (one provider supports MP3, another doesn't) | Medium | Low | Document MIME-support per provider; `AudioInput::Pcm` is the universal lowest-common-denominator. |

---

## Appendix A — Glossary

- **Alias** — string handle (e.g., `embed/image`) declared in the model catalog and resolved at call time to a concrete model + provider.
- **`ModelTask`** — enum declaring what kind of work an alias performs; gates which trait resolves the alias.
- **`raw_tensor_model`** — existing xervo escape hatch returning a low-level ONNX runner; used by consumers needing full tokenize/forward/decode control.
- **kniv** — IBM's multi-head DeBERTa NLP cascade used by uniko for POS / NER / DEP / SRL / CLS extraction.
- **Provider driver** — implementation of one or more traits backed by a runtime (local/onnx, local/mistralrs, remote/openai, etc.).

## Appendix B — Reference: existing facade verbatim

See `crates/uni/src/api/xervo.rs` lines 31-145 for the existing `UniXervo` struct and impl. This proposal adds methods between the existing `rerank` (line 108) and `prefetch_all` (line 127) — no other reordering needed.

## Appendix C — Consumer-side context

uniko's audit of pending capabilities lists the following as blocked on this xervo work:

- `Artifact.text_embedding` mean-pooling (audit §4, F25/P7c).
- Image / audio / video / multimodal embedding population (audit §4, F25).
- VLM-based PDF parsing (audit §11, P7c follow-on).
- ASR for audio Messages (out of current spec; needed for multimodal LoCoMo / Mem-Gallery benchmarking).
- NLP cascade managed dispatch (audit §3 - currently uses `raw_tensor_model`).

The full uniko audit lives at `/home/rohit/.claude/plans/let-us-proceed-with-tingly-music.md`.
