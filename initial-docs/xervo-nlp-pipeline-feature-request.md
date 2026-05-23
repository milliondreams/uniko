# Feature Request: NLP Pipeline Support in uni-xervo

## Problem Statement

uni-xervo currently provides four model task types: `Embed`, `Rerank`, `Generate`, and `Raw`. The `Raw` task exposes the `OnnxRunner` trait — raw tensor in/out with no pre-processing or post-processing.

For NLP workloads (NER, POS tagging, dependency parsing, sentence classification), callers must build their own pipeline on top of `OnnxRunner`:

1. **Tokenize** text with a HuggingFace tokenizer
2. **Construct input tensors** (input_ids, attention_mask, word_ids)
3. **Call `OnnxRunner::run()`** with raw tensors
4. **Decode output tensors** — argmax, subword-to-word alignment, BIO tag merging, dependency tree decoding, CLS label lookup

This pipeline code currently lives in `uniko-extract/src/nlp/` (~800 lines), but none of it is application-specific. It's generic NLP inference machinery that should be reusable by any consumer of multi-task NLP models.

### Current Pain Points

1. **Repeated `onnx_runner()` acquisition**: Each call to `NlpPipeline::try_new()` calls `kb.db().xervo().onnx_runner("nlp/default").await`. In LongMemEval benchmarks, this added ~285ms overhead per turn across 550 turns = **157 seconds** of pure handle-acquisition overhead (vs 14 seconds of actual inference). The caller has no way to cache the pipeline handle because xervo has no pipeline concept — only raw runners.

2. **Embedded assets in the wrong crate**: The HF tokenizer (8MB `tokenizer.json`) and label maps (31KB `label_maps.json`) are compiled into `uniko-extract` via `include_bytes!`. These are model assets, not application assets — they should ship with the model in xervo.

3. **No reuse across consumers**: Any other project wanting to use `dragonscale-ai/kniv-deberta-v3-nlp-en` must duplicate the entire tokenization + decoding stack from uniko-extract.

4. **Batch inference gap**: `OnnxRunner` supports `run_batch()`, but the caller must manually chunk inputs, align subword outputs back to words per batch element, and reassemble. A pipeline abstraction could handle this transparently.

## Proposed Solution

Add a **`TokenClassification` pipeline** to xervo that handles tokenization, inference, and structured output decoding for multi-task NLP models.

### New Model Task

```rust
pub enum ModelTask {
    Embed,
    Rerank,
    Generate,
    Raw,
    TokenClassification,  // NEW
}
```

### New Trait: `NlpModel`

```rust
#[async_trait]
pub trait NlpModel: Send + Sync {
    /// Analyze text and return structured per-sentence NLP results.
    async fn analyze(&self, text: &str) -> Result<Vec<NlpSentenceResult>>;

    /// Analyze multiple texts in a batch.
    async fn analyze_batch(&self, texts: &[&str]) -> Result<Vec<Vec<NlpSentenceResult>>>;
}
```

### Output Types (New in xervo)

These are the structured outputs that the pipeline produces. They're model-output types, not application types — they describe what the model predicted, not what the application does with it.

```rust
/// Per-sentence NLP analysis result.
pub struct NlpSentenceResult {
    /// Original sentence text.
    pub sentence: String,
    /// Tokenized words (subwords merged back to words).
    pub words: Vec<String>,
    /// Named entity spans (BIO-decoded).
    pub ner_spans: Vec<NerSpan>,
    /// Part-of-speech tag indices (per word).
    pub pos_indices: Vec<usize>,
    /// Dependency parse arcs.
    pub dep_arcs: Vec<DepArc>,
    /// Sentence-level classification.
    pub sentence_class: SentenceClassLabel,
    /// Label vocabularies used for decoding indices.
    pub label_vocab: Arc<LabelVocab>,
}

/// A named entity span from BIO decoding.
pub struct NerSpan {
    pub text: String,
    pub entity_type: NerEntityType,
    pub word_start: usize,
    pub word_end: usize,
    pub confidence: f32,
}

/// Dependency arc (head → dependent).
pub struct DepArc {
    pub dependent: usize,
    pub head: usize,       // usize::MAX = root
    pub relation: String,
}

/// Sentence classification label with confidence.
pub struct SentenceClassLabel {
    pub label: String,
    pub confidence: f32,
}

/// NER entity types (CoNLL + OntoNotes scheme).
pub enum NerEntityType {
    Person, Organization, Location, Date,
    Numeric, Event, Product, WorkOfArt, Group, Misc,
}

/// Label index → string mappings for all output heads.
pub struct LabelVocab {
    pub ner_labels: Vec<String>,
    pub pos_labels: Vec<String>,
    pub dep_labels: Vec<String>,
    pub cls_labels: Vec<String>,
}
```

### New UniXervo Method

```rust
impl UniXervo {
    /// Get an NLP model handle by alias.
    /// The handle is cached — subsequent calls return the same instance.
    pub async fn nlp_model(&self, alias: &str) -> Result<Arc<dyn NlpModel>>
}
```

### Pipeline Internals (What Moves from uniko-extract to xervo)

The following code from `uniko-extract/src/nlp/` would move into xervo's `LocalOnnxNlpProvider`:

| Component | Current location | What it does |
|-----------|-----------------|-------------|
| `split_sentences()` | `nlp/mod.rs:189-235` | Split text on sentence boundaries, filter short fragments |
| `tokenizer.json` | `nlp/assets/tokenizer.json` | HuggingFace BPE tokenizer (8MB), currently `include_bytes!` |
| `label_maps.json` | `nlp/assets/label_maps.json` | NER/POS/DEP/CLS label vocabularies (31KB) |
| Tokenizer loading | `nlp/assets.rs:44-49` | `tokenizers::Tokenizer::from_bytes()` with `OnceLock` |
| Label map loading | `nlp/assets.rs:56-60` | JSON parse into `LabelVocab` with `OnceLock` |
| Input tensor construction | `nlp/mod.rs:75-112` | `input_ids`, `attention_mask` from tokenizer output |
| Output tensor extraction | `nlp/mod.rs:120-123` | Extract 4 logit tensors from model output |
| `argmax_rows()` | `nlp/decode.rs:18-30` | 2D logits → per-row argmax indices |
| `argmax_with_confidence()` | `nlp/decode.rs:36-54` | 1D logits → argmax + softmax confidence |
| `align_to_words()` | `nlp/decode.rs:62-77` | Subword predictions → word-level via first-subword-wins |
| `extract_words()` | `nlp/decode.rs:85-109` | Reconstruct words from subwords, strip RoBERTa/DeBERTa markers |
| `merge_bio_spans()` | `nlp/decode.rs:117-164` | BIO tag sequence → NER entity spans |
| `parse_ner_type()` | `nlp/decode.rs:179-198` | BIO label suffix → NerEntityType enum |
| `decode_dep_tree()` | `nlp/decode.rs:208-258` | dep2label indices → DepArc list with offset resolution |
| `resolve_offset()` | `nlp/decode.rs:270-313` | POS-aware head resolution for dependency labels |
| `decode_cls()` | `nlp/decode.rs:665-670` | CLS index → sentence classification label |

### What Stays in uniko-extract (Application Logic)

These consume `NlpSentenceResult` and apply uniko-specific domain logic:

| Component | Location | Why it stays |
|-----------|----------|-------------|
| `extract_dep_observations()` | `nlp/decode.rs:429-528` | Reconstructs observations from DEP tree with speaker substitution. Requires `SentenceContext`. |
| `resolve_subject()` | `nlp/decode.rs:534-585` | Pronoun → speaker name resolution via session context |
| `update_sentence_context()` | `nlp/decode.rs:623-656` | Tracks nouns across sentences for pronoun resolution |
| `SentenceClass::is_informative()` | `nlp/types.rs:146-148` | Business rule: what's worth extracting |
| `entities_from_nlp_result()` | `ner/onnx.rs:23-77` | NerSpan → uniko's RawEntity mapping |
| `ObservationExtractionStep` | `observations/mod.rs` | Full P3 pipeline step: CLS gating → extraction → graph writes |
| Content filters | `observations/filter.rs` | Greeting/filler/question detection heuristics |
| `SessionContext` / `SentenceContext` | `ingest/context.rs` | Speaker tracking, pronoun resolution window |

### Asset Bundling

Currently, assets are embedded in `uniko-extract` at compile time:

```rust
// uniko-extract/src/nlp/assets.rs
static TOKENIZER_BYTES: &[u8] = include_bytes!("assets/tokenizer.json");
static LABEL_MAPS_JSON: &str = include_str!("assets/label_maps.json");
```

**Proposed**: Assets should be bundled with the model in the HuggingFace repo (`dragonscale-ai/kniv-deberta-v3-nlp-en`) and loaded by xervo's provider alongside the ONNX model file:

```
dragonscale-ai/kniv-deberta-v3-nlp-en/
  ├── model-int8.onnx          # Already there
  ├── tokenizer.json           # Move here
  └── label_maps.json          # Move here
```

The `LocalOnnxNlpProvider` would download and cache all three files together via the existing HuggingFace download mechanism.

## Model Registration

The NLP model registration would change from `Raw` to `TokenClassification`:

```json
{
  "alias": "nlp/default",
  "task": "TokenClassification",
  "provider_id": "local/onnx",
  "model_id": "dragonscale-ai/kniv-deberta-v3-nlp-en",
  "options": {
    "artifact": "model-int8.onnx",
    "max_batch_size": 16,
    "output_heads": {
      "ner": { "index": 0, "scheme": "bio", "labels": "auto" },
      "pos": { "index": 1, "scheme": "argmax", "labels": "auto" },
      "dep": { "index": 2, "scheme": "dep2label", "labels": "auto" },
      "cls": { "index": 3, "scheme": "argmax", "labels": "auto" }
    }
  }
}
```

The `output_heads` config tells the provider how to decode each output tensor, making the pipeline configurable for different multi-task model architectures without code changes.

## Consumer API (After Migration)

uniko-extract would simplify from ~800 lines of NLP pipeline code to:

```rust
// Before: manual pipeline management
let runner = kb.db().xervo().onnx_runner("nlp/default").await?;
let pipeline = NlpPipeline { runner };
let tokenizer = assets::tokenizer();
let labels = assets::label_maps();
// ... 60 lines of tokenize → infer → decode per sentence

// After: single call
let nlp = kb.db().xervo().nlp_model("nlp/default").await?;
let results: Vec<NlpSentenceResult> = nlp.analyze(&text).await?;

// Application logic operates on structured results:
for result in &results {
    if result.sentence_class.label == "statement" {
        let observations = extract_dep_observations(
            &result.words, &result.pos_indices, &result.dep_arcs,
            &labels.pos_labels, speaker, &mut sent_ctx,
        );
        // ... graph writes
    }
}
```

## Performance Implications

### Pipeline Handle Caching

`nlp_model()` returns a cached `Arc<dyn NlpModel>`. The tokenizer and label maps are loaded once inside the pipeline handle. This eliminates the 285ms/call overhead from repeated `onnx_runner()` acquisition — the handle is acquired once and reused across all turns.

**Expected impact**: Entity extraction time drops from ~170s to ~14s (the actual inference cost) for 550-turn LongMemEval questions.

### Batch Inference

The `analyze_batch()` method enables processing multiple texts in a single ONNX call (up to `max_batch_size`). The pipeline handles:

- Tokenizing all texts
- Padding to uniform length within the batch
- Running a single ONNX inference
- Splitting and decoding results per text

This is useful for ingestion scenarios where multiple messages can be processed together.

## Implementation Phases

### Phase 1: Core Pipeline (Minimum Viable)
- Add `TokenClassification` to `ModelTask`
- Add `NlpModel` trait with `analyze()` method
- Implement `LocalOnnxNlpProvider` by moving decode logic from uniko-extract
- Move tokenizer + label_maps assets to HuggingFace repo
- Add `nlp_model()` to `UniXervo` API
- Handle caching (same as embed/rerank/generate handles)

### Phase 2: Configuration & Flexibility
- `output_heads` config for multi-task model architectures
- Support models with different subsets of heads (NER-only, POS-only, etc.)
- Custom label map loading from HF repo

### Phase 3: Batch & Optimization
- `analyze_batch()` with automatic padding and result splitting
- Sentence-level batching within a single `analyze()` call
- Token count estimation for budget-aware inference

## Compatibility

- **Backward compatible**: Existing `Raw` task and `OnnxRunner` remain unchanged. Projects using `onnx_runner()` directly continue to work.
- **Migration path**: uniko-extract can migrate incrementally — use `nlp_model()` when available, fall back to `onnx_runner()` + manual pipeline if not.
- **Model compatibility**: The same ONNX model file works with both `Raw` and `TokenClassification` tasks. The difference is whether xervo handles pre/post-processing.
