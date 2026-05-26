# Multimodal Knowledge Store — Design Spec

| | |
|---|---|
| **Status** | Draft for review |
| **Author** | uniko team |
| **Date** | 2026-05-25 |
| **Scope** | Schema, storage shape, ingest pipeline, recall cascade extensions, dedup strategy, and migration plan to make uniko a true multimodal knowledge store. |
| **Depends on** | [`xervo-multimodal-api-proposal.md`](xervo-multimodal-api-proposal.md) for model-dispatch capabilities (`embed_image`, `embed_audio`, `embed_multimodal`, `vlm_extract`, `transcribe`, `ocr`, `nlp_analyze`). This document does not propose changes to uni-xervo. |
| **Companion to** | [`uniko-spec-v6.md`](uniko-spec-v6.md) — extends and concretises F23–F30 (content / artifacts / chunks) and F25 (multimodal embedding). |

---

## 0. TL;DR

uniko's spec reserves five embedding fields on `Artifact` (text / image / audio / video / multimodal) but only one (`text_embedding`) is reachable today, and even that is unpopulated for the parent Artifact (only chunks embed). HTML / PDF / audio / video paths silently fall back to plain-text chunking. The multimodal capability the spec promises is aspirational, not actual.

This document specifies the work to make it real:

1. **A new `:ArtifactContent` label** in the graph for content-addressed blob storage, deduplicated by SHA-256. Artifacts reference content via a `HAS_CONTENT` edge. Many Artifacts → one Content node.
2. **`Artifact` extended with typed nullable modality metadata columns** (`width`, `height`, `duration_ms`, `sample_rate`, `channels`, `fps`, `frame_count`, `page_count`, + `modality_meta: Map` for the tail).
3. **`Chunk` extended with modality-positioning fields** (`modality`, `bbox`, `time_start_ms`, `time_end_ms`, `page_number`, `reading_order`) and a `source_model_version` field for derivation tracking. `Chunk.text` becomes the lingua franca surface form for cross-modal search.
4. **Per-modality ingest entry points**: `ingest_image`, `ingest_audio`, `ingest_video`, `ingest_pdf`, plus the existing text/code paths. Each follows a uniform shape: hash + dedupe → store content → create artifact → run modality-native embedding via xervo → extract captions/transcripts/OCR → fan out the existing NER + observation pipeline over derived text.
5. **Perceptual hashing** for images (`pHash`) and audio fingerprinting (`chromaprint`) for near-duplicate detection.
6. **Lazy modality activation** in the recall cascade: cross-modal vector channels run only against modalities actually present in the corpus, gated by a per-KB modality-presence cache. No feature flags, no manual configuration.
7. **Migration script** for existing `Artifact.content: String` rows to land in `:ArtifactContent` non-destructively.

No changes to uni-db itself beyond what the companion xervo proposal asks for. No changes to BTIC, Locy, consolidation, P4-P6 worker pipelines, or the existing recall Phase 1 / Phase 2 logic — those work over the same uniform graph shape regardless of modality.

---

## 1. Context and motivation

### 1.1 Current state (verified by audit, 2026-05-25)

| Capability | Today |
|---|---|
| Text artifacts | ✅ Full path: hash dedup, chunking, FullText on `Artifact.content`, Chunk vectors auto-embed |
| Source code artifacts | ✅ Tree-sitter chunking when `code-parse` feature on |
| HTML artifacts | ❌ `HtmlChunker` silently delegates to `TextChunker` |
| PDF artifacts | ❌ `PdfChunker` silently delegates to `TextChunker` |
| CSV / JSON artifacts | ❌ `StructuredChunker` silently delegates to `TextChunker` |
| Image artifacts | ❌ No ingest path. Schema reserves `image_embedding`, never populated |
| Audio artifacts | ❌ No ingest path. Schema reserves `audio_embedding`, never populated |
| Video artifacts | ❌ No ingest path. Schema reserves `video_embedding`, never populated |
| Multimodal embeddings (cross-modal queries) | ❌ Schema reserves `multimodal_embedding`, never populated, no recall path consumes it |
| `Artifact.text_embedding` (pooled chunk embeddings) | ❌ Schema field exists with HNSW index; **zero code populates it** |
| Provenance edges (`CREATED_BY`, `MODIFIED_BY`) | ❌ Schema-defined, **zero production code writes them** |

### 1.2 Why this matters

Two production-grade external pressures:

1. **Mem-Gallery benchmark** (arXiv 2601.03515, early 2026) — first agent-memory benchmark where vision actually matters. LoCoMo's "multimodal" tasks can be solved without visual info; Mem-Gallery is where multimodal agent-memory will be judged. We can't score on it until we ingest non-text content.
2. **Competitive positioning**. Mem0, Zep/Graphiti, Letta, LangMem are aggressively text-only. Cognee handles multimodal but flattens to text before graph extraction (loses original-modality embeddings). The defensible niche — **embedded Rust cognitive memory with retained per-modality artifacts under BTIC + Locy** — is unoccupied, and uniko's schema is already shaped for it.

### 1.3 What changed in 2025-2026 that makes this feasible

The xervo proposal documents this in detail. Three things converged:

- Apache-licensed VLM-based document parsers (MinerU 2.5, Granite-Docling, olmOCR-2) that ONNX-export cleanly, so PDF parsing no longer requires a Python sidecar.
- Production-grade cross-modal embedding models (SigLIP 2, Jina v4) shipping as ONNX weights.
- Native Rust ASR (`whisper-rs` binding to whisper.cpp), Rust perceptual-hashing crates (`image-hasher`), Rust audio-fingerprint binding (`chromaprint-rs`) — all mature.

uniko can now be a fully embedded multimodal store with no Python in the runtime.

### 1.4 Prerequisite — xervo PR 1

This spec assumes the xervo proposal's "PR 1 — the contract" has landed. Specifically:

- `xervo.embed_image(alias, &[ImageInput])` → vectors
- `xervo.embed_audio(alias, &[AudioInput])` → vectors
- `xervo.embed_multimodal(alias, &[MultimodalInput])` → vectors
- `xervo.vlm_extract(alias, &[ImageInput], opts)` → structured doc blocks
- `xervo.transcribe(alias, &AudioInput, opts)` → segments
- `xervo.ocr(alias, &[ImageInput])` → text blocks

Without PR 1 there's nothing to wire the new ingest paths to.

---

## 2. Requirements

### 2.1 Functional

**M1.** Store binary content (images, audio, video, PDFs, any blob) in the graph, content-addressed by SHA-256, with N:1 fan-in from Artifact metadata nodes to deduplicate identical bytes regardless of upload source or timestamp.

**M2.** Ingest images: dedupe (exact + perceptual), store bytes, create Artifact metadata node, compute image-modality embedding, generate caption via VLM, run the existing NER + observation extraction over the caption.

**M3.** Ingest audio: dedupe (exact + acoustic fingerprint), store bytes, create Artifact, transcribe via ASR (chunk-per-segment with timestamps + optional speaker labels), compute audio-modality embedding, run NER + observation extraction over the transcript.

**M4.** Ingest video: dedupe (exact), scene-detect, render keyframes, transcribe audio, store video bytes + keyframe references, create Artifact with `kind="video"`, create Chunks per scene (each carrying its scene's keyframe-via-bbox + audio-segment time range + transcript text), compute video-modality embedding, run NER + observation extraction over transcripts + scene captions.

**M5.** Ingest PDFs: dedupe (exact), store bytes, render each page to an image, run VLM document extraction per page, create one Artifact for the PDF with `page_count`, create one or more Chunks per page (text blocks, tables, formulas, captions, figures) with `page_number`, `bbox`, `reading_order`, run NER + observation extraction over the extracted text.

**M6.** Ingest HTML: dedupe (exact), parse via a real DOM-aware extractor (Rust `dom_smoothie`), produce one or more Chunks with `chunk_type="text"` / `"heading"` / `"caption"`, run NER + observation extraction.

**M7.** Ingest CSV / JSON: dedupe (exact), schema-aware row grouping (header preserved per chunk via `heading` field), produce Chunks with `chunk_type="table_row_group"`, run NER + observation extraction.

**M8.** Populate `Artifact.text_embedding` as a mean-pooled vector of all child Chunk text embeddings. Required to make direct artifact-level vector search functional.

**M9.** Cross-modal recall: a text query whose corpus contains multimodal content automatically fans out additional vector channels against the corresponding modality embedding columns. No feature flag; no user configuration.

**M10.** Track derivation provenance on every model-derived Chunk via `source_model_version` so future re-derivation passes can identify stale chunks when their source model changes.

**M11.** Existing text artifacts must migrate non-destructively: their `content: String` moves to `:ArtifactContent.bytes`, with the same hash, via an idempotent migration script.

### 2.2 Non-functional

**N1. No Python in the runtime.** All ingest paths run in pure Rust via native crates or via xervo's ONNX/whisper-cpp/candle providers.

**N2. Backwards compatibility for text ingest.** Existing `IngestMessage` and `IngestArtifact` paths for text and source code continue to work without changes. New modality entry points are additive.

**N3. Single transaction per artifact.** Hash compute → MERGE content node → create Artifact + HAS_CONTENT edge → fire embedding/captioning/chunking. The graph-write portion is one Cypher transaction; the model calls are async-out-of-band where possible.

**N4. Storage efficiency.** Lance per-label storage means `:ArtifactContent` blobs do not page in when HNSW vector recall scans `:Artifact` rows.

**N5. Streaming-tolerant.** Soft cap of 2 GB per `ArtifactContent.bytes` row; larger blobs require upstream chunking (video segmentation by scene, audiobook chapter splits) before they reach the storage layer.

**N6. Cross-modal model alignment.** Image, text, and (where supported) audio embeddings share a vector space when produced for cross-modal queries — i.e., the model used for image embedding is the image-tower of the same dual-tower model used for cross-modal text embedding. SigLIP-2 is the v1 reference.

### 2.3 Out of scope

This proposal does NOT cover:

- **Streaming ingest of very large blobs** (cinema-scale video > 2 GB). Stream-hash + chunked-write is v2.
- **Server-side blob storage** (S3, object stores). v1 stores blobs in-process in Lance. The `:ArtifactContent` label is the abstraction boundary; v2 can back it with a remote object store transparently.
- **Multi-vector (ColBERT / ColPali) document retrieval.** Requires multi-vector indexing in uni-db, which is a separate uni-db enhancement. v1 uses single-vector embeddings via MinerU's caption + the standard SigLIP-2 channel.
- **Editing / version history on `:ArtifactContent`.** Content is immutable; new bytes always produce a new content node with a new hash. Artifact-level edits (re-captioning, swapping the canonical Content) produce a new `:DERIVED_FROM` chain rather than mutating content nodes.
- **Streaming embeddings or streaming ASR results.** Batch-mode only.
- **Storage of model weights or embedding indexes outside Lance.** uni-db owns all storage; xervo owns model dispatch.
- **End-to-end video understanding via single VLM** (Twelve Labs Marengo, VideoLLaMA2). v1 is keyframe + ASR + scene-caption fusion; end-to-end video models are v2.

---

## 3. Schema additions

### 3.1 New label: `:ArtifactContent`

```
:ArtifactContent
  - content_id: String          // SHA-256 hex, primary key, Hash index
  - bytes: Bytes                // the actual blob
  - mime: String                // canonical mime type (e.g., "image/jpeg")
  - size: Int64                 // byte length
  - perceptual_hash: Optional Int64    // pHash (DCT 8x8), 64-bit. NULL for non-image.
  - audio_fingerprint: Optional Bytes  // chromaprint, ~120 bytes. NULL for non-audio.
  - created_at: DateTime
```

**No embeddings on this label.** Embeddings live on `:Artifact`. The content node is bytes-only.

**Indexes:**
- `content_id` — Hash (primary lookup for dedup-on-MERGE)
- `perceptual_hash` — BTree (near-duplicate range queries on Hamming buckets)
- `mime` — Hash (for "all images" filters)
- `size` — BTree (range queries, statistics)

### 3.2 Extensions to `:Artifact`

Existing fields preserved. New fields added, all nullable:

```
:Artifact
  - artifact_id: String                  // existing
  - kind: String                         // existing — modality discriminator
  - path: Optional String                // existing
  - hash: Optional String                // existing — denormalized cache of HAS_CONTENT target's content_id
  - mime_type: Optional String           // existing
  - size: Optional Int64                 // existing
  - language: Optional String            // existing
  - created_at: DateTime                 // existing
  - updated_at: Optional DateTime        // existing

  // Existing embedding fields (unchanged shape, finally getting populated)
  - text_embedding: Optional Vector
  - image_embedding: Optional Vector
  - audio_embedding: Optional Vector
  - video_embedding: Optional Vector
  - multimodal_embedding: Optional Vector

  // NEW — modality metadata (all nullable, all typed for indexability)
  - width: Optional Int32                // image / video
  - height: Optional Int32               // image / video
  - duration_ms: Optional Int64          // audio / video — BTree-indexed
  - sample_rate: Optional Int32          // audio
  - channels: Optional Int16             // audio
  - fps: Optional Float32                // video
  - frame_count: Optional Int32          // video
  - page_count: Optional Int32           // PDF — BTree-indexed
  - modality_meta: Optional CypherValue  // free-form Map for tail metadata (codec, encoder, color_space, exif…)

  // NEW — provenance
  - origin: Optional CypherValue         // Map: { source_type, source_ref, ingested_at }. NULL when origin is an Action (use CREATED_BY edge instead).
```

**Why typed columns for most modality metadata and a Map for the tail:** Lance/Arrow nullable typed columns are essentially free when unset; Maps are stored as opaque CypherValue, not indexable. Common queries ("audio clips > 5 min", "PDFs with more than 50 pages") deserve indexed typed columns. The long tail (codec strings, color spaces, EXIF blobs) goes in the Map.

**Why `origin` is a Map and not edges:** non-Action provenance (URL, filesystem path with timestamp, upload session id) doesn't deserve its own node type. For agent-produced artifacts, the existing `CREATED_BY` edge to `Action` remains the canonical provenance path. The `origin` Map is for everything else.

**New index:**
- `duration_ms` — BTree
- `page_count` — BTree

### 3.3 New edge: `HAS_CONTENT`

```
HAS_CONTENT
  - From: Artifact
  - To: ArtifactContent
  - Properties:
    - role: Optional String         // "primary" (default) | "preview" | "derived"
```

**Default `role="primary"`** on every artifact-content edge created via standard ingest. The `preview` and `derived` roles are reserved for future use (thumbnails, page renders, transcoded variants) — see §6.

### 3.4 New edge: `DERIVED_FROM`

```
DERIVED_FROM
  - From: Artifact
  - To: Artifact
  - Properties:
    - derivation_kind: String       // "pdf_page_render" | "video_keyframe" | "audio_segment" | "transcoded" | …
    - derived_at: DateTime
```

Used when one Artifact is produced from another by non-Action machinery (PDF page rendering, video keyframe extraction, audio segmentation). Distinct from `CREATED_BY` which goes Artifact→Action and represents agent-driven production.

### 3.5 Extensions to `:Chunk`

```
:Chunk
  // Existing fields (unchanged)
  - chunk_id: String
  - text: String                         // lingua franca — caption / transcript / OCR / extracted markdown / source text
  - chunk_type: String                   // "text" | "code" | "caption" | "transcript_segment" | "ocr_region" | "page_block" | "scene"
  - language: Optional String
  - symbol_name: Optional String         // code
  - speaker: Optional String             // audio
  - heading: Optional String             // text / table_row_group

  // NEW — modality positioning
  - modality: String                     // "text" | "image" | "audio" | "video" — Hash-indexed; defaults to "text" for migration
  - bbox: Optional List<Float32>         // [x0, y0, x1, y1] image / video keyframe / PDF page region
  - time_start_ms: Optional Int64        // audio / video — BTree-indexed
  - time_end_ms: Optional Int64
  - page_number: Optional Int32          // PDF — BTree-indexed
  - reading_order: Optional Int32        // monotonic within a page or scene

  // NEW — derivation tracking
  - source_model_version: Optional String  // "vlm/caption@granite-docling-258m-v1.0"; NULL for non-derived chunks
```

**Why a single `:Chunk` label rather than `:TextChunk`, `:AudioChunk`, `:VideoChunk` subtypes:** same reasoning as Artifact-container in §3.2. Sparse columns are free in Lance; edges (`HAS_CHUNK`, `MENTIONS`, `OBSERVED_IN`) stay simple with one endpoint label.

**Why `chunk_type` exists alongside `modality`:** `modality` is the dimension ("audio"); `chunk_type` is the semantic role ("transcript_segment" vs "caption" vs "speaker_turn"). Different audio chunks can share `modality="audio"` but differ in `chunk_type`.

### 3.6 Index additions summary

| Label | Field | Index | Purpose |
|---|---|---|---|
| ArtifactContent | content_id | Hash | dedup-on-MERGE |
| ArtifactContent | perceptual_hash | BTree | near-dup queries |
| ArtifactContent | mime | Hash | modality-filter scans |
| ArtifactContent | size | BTree | range queries |
| Artifact | duration_ms | BTree | "audio > 5 min" |
| Artifact | page_count | BTree | "PDFs with > N pages" |
| Chunk | modality | Hash | per-modality filters |
| Chunk | time_start_ms | BTree | temporal range queries |
| Chunk | page_number | BTree | "all chunks on page 5" |

---

## 4. Storage model — `:ArtifactContent` as graph-native CAS

The content-addressed-store pattern is implemented as a label rather than a sibling Lance dataset. Six properties of this design:

1. **Graph-native dedup via MERGE.** Ingest computes the hash, runs `MERGE (c:ArtifactContent {content_id: $hash})` in Cypher. Existing content node is reused atomically; new content creates a new node. No separate blob-store API.

2. **N:1 fan-in.** Multiple Artifacts referencing the same bytes (same image uploaded twice with different provenance, same PDF analyzed by two different agents) share one `:ArtifactContent` node. Disk usage is bounded by unique bytes.

3. **Transactional with metadata.** One Cypher write creates Artifact + ArtifactContent + HAS_CONTENT in a single transaction. Rollback semantics inherit.

4. **HNSW recall is bytes-free.** Lance per-label storage means vector queries on `:Artifact` scan only the Artifact label's columnar store. Blob bytes live on `:ArtifactContent`, never paged in during recall.

5. **GC is graph-traversal.** Orphan removal sweep:

   ```cypher
   MATCH (c:ArtifactContent)
   WHERE NOT (c)<-[:HAS_CONTENT]-(:Artifact)
   DETACH DELETE c
   ```

   Runs as a background sweep (post-consolidation cadence, alongside P5/P6). For v1, defer GC — orphans accumulate slowly and disk is cheap.

6. **Storage backend portability.** The `:ArtifactContent` abstraction lets v2 transparently swap Lance-local storage for a remote object store (S3, GCS) without touching ingest or recall code. The label becomes a façade.

### 4.1 Size limits

v1 soft cap: **2 GB per `ArtifactContent.bytes` row.** Larger blobs MUST be chunked upstream (video by scene, audiobooks by chapter, multi-gigabyte logs by line range). Enforcement: warn at 500 MB, error at 2 GB. Hard cap derives from Arrow LargeBinary practical limits in Lance batch fragments.

### 4.2 The denormalized `Artifact.hash` field

Existing field preserved as a write-through cache of the HAS_CONTENT target's `content_id`. Allows "does this artifact already exist?" checks to hit a single label index instead of traversing the edge. Cost: one redundant String column on Artifact. Benefit: removes one hop from the hot ingest path (dedup check fires on every ingest).

---

## 5. Ingest pipeline — per-modality entry points

### 5.1 Shared ingest skeleton

Every modality follows the same shape:

```rust
pub async fn ingest_<modality>(
    kb: &KnowledgeBase,
    input: <Modality>Input,
    options: <Modality>IngestOptions,
) -> Result<<Modality>IngestResult> {
    // 1. Compute exact hash + modality-specific perceptual hash / fingerprint
    let hash = sha256(&input.bytes());
    let phash = compute_perceptual_hash(&input)?;  // image only
    let fingerprint = compute_audio_fingerprint(&input)?;  // audio only

    // 2. Dedup: exact hash
    if let Some(existing) = kb.find_artifact_by_hash(&hash).await? {
        return Ok(<Modality>IngestResult::DeduplicatedExact(existing));
    }

    // 3. Dedup: perceptual (logs near-dup, does NOT silently alias)
    let near_dups = kb.find_near_duplicates(phash, hamming_threshold=8).await?;
    if !near_dups.is_empty() && !options.dedupe_near_visual {
        // Continue, but emit a NearDuplicateDetected event for analytics
        tracing::info!(near_dups = ?near_dups, "near-duplicate visual content");
    } else if !near_dups.is_empty() && options.dedupe_near_visual {
        // Alias to the existing content
        return Ok(<Modality>IngestResult::AliasedToExisting(near_dups[0]));
    }

    // 4. Content node — MERGE handles atomic dedup
    let content_id = kb.merge_artifact_content(MergeContent {
        content_id: hash.clone(),
        bytes: input.bytes(),
        mime: input.mime(),
        size: input.size(),
        perceptual_hash: phash,
        audio_fingerprint: fingerprint,
    }).await?;

    // 5. Artifact metadata
    let artifact_id = kb.create_node("Artifact", &props_for(&input)).await?;
    kb.create_edge("HAS_CONTENT", artifact_id, content_id, &edge_props("primary")).await?;

    // 6. Modality-native embedding via xervo
    let modal_embedding = kb.db().xervo()
        .embed_<modality>(<alias>, &[input.into_xervo_input()]).await?[0];
    kb.update_node(artifact_id, &[("<modality>_embedding", modal_embedding)]).await?;

    // 7. Modality-specific derivation pipeline (caption / transcribe / OCR / VLM-extract)
    let chunks = derive_chunks(kb, &input, artifact_id, &options).await?;

    // 8. Mean-pool chunk embeddings into Artifact.text_embedding
    if !chunks.is_empty() {
        let pooled = mean_pool_chunk_embeddings(kb, &chunks).await?;
        kb.update_node(artifact_id, &[("text_embedding", pooled)]).await?;
    }

    Ok(<Modality>IngestResult::Ingested { artifact_id, content_id, chunks })
}
```

### 5.2 Per-modality entry points

```rust
// Existing
pub async fn ingest_text(kb, text: &str, opts: TextIngestOptions) -> Result<TextIngestResult>;
pub async fn ingest_code(kb, source: &str, lang: &str, opts: CodeIngestOptions) -> Result<CodeIngestResult>;
pub async fn ingest_html(kb, html: &str, opts: HtmlIngestOptions) -> Result<HtmlIngestResult>;  // new — uses dom_smoothie

// New
pub async fn ingest_image(kb, image: ImageInput, opts: ImageIngestOptions) -> Result<ImageIngestResult>;
pub async fn ingest_audio(kb, audio: AudioInput, opts: AudioIngestOptions) -> Result<AudioIngestResult>;
pub async fn ingest_video(kb, video: VideoInput, opts: VideoIngestOptions) -> Result<VideoIngestResult>;
pub async fn ingest_pdf(kb,   pdf:   PdfInput,   opts: PdfIngestOptions)   -> Result<PdfIngestResult>;
pub async fn ingest_structured(kb, data: StructuredInput, opts: StructuredIngestOptions) -> Result<StructuredIngestResult>;
```

### 5.3 Modality-specific derivation pipelines

#### 5.3.1 Image (`ingest_image`)

**Steps after the shared skeleton:**

1. **Caption** via `xervo.vlm_extract(alias, &[image], opts)` with single-image input. Output: one `DocExtractResult` whose `plain_markdown` is the caption text.
2. **OCR** (optional, `opts.ocr`) via `xervo.ocr(alias, &[image])`. Output: `OcrResult` with text blocks + bboxes.
3. Create one `caption` Chunk: `modality="image"`, `chunk_type="caption"`, `text=<caption>`, `source_model_version=<vlm alias + version>`.
4. Create N `ocr_region` Chunks (one per OCR block): `modality="image"`, `chunk_type="ocr_region"`, `text=<block.text>`, `bbox=<block.bbox>`, `source_model_version=<ocr alias + version>`.
5. Each Chunk auto-embeds on `text` and goes through P2 (NER) + P3 (observations) → entity extraction + observation creation → eventual consolidation into Facts.
6. Mean-pool Chunk embeddings → `Artifact.text_embedding`.

#### 5.3.2 Audio (`ingest_audio`)

1. **Transcribe** via `xervo.transcribe(alias, &audio, opts)` with `word_timestamps: true`, `diarize: opts.diarize`. Output: `TranscribeResult` with segments.
2. **Audio embedding** via `xervo.embed_audio(alias, &[audio])`.
3. Create one `transcript_segment` Chunk per `TranscribeSegment`: `modality="audio"`, `chunk_type="transcript_segment"`, `text=<segment.text>`, `time_start_ms=<segment.start_ms>`, `time_end_ms=<segment.end_ms>`, `speaker=<segment.speaker>`, `source_model_version=<transcribe alias + version>`.
4. Chunks auto-embed → P2 + P3 → entities + observations as usual.

#### 5.3.3 Video (`ingest_video`)

1. **Scene detection** (upstream, via Rust crate `ffmpeg-the-third` or `pyscenedetect-rs` analog). Produces scene boundaries (start_ms, end_ms) and a representative keyframe image per scene.
2. **Audio extract + transcribe** via `xervo.transcribe`.
3. **Per-scene keyframe captioning**: each scene's keyframe → `xervo.vlm_extract(caption_alias, &[keyframe], opts)` → caption.
4. **Video embedding** via `xervo.embed_video` (if available; for v1 fall back to mean-pool of per-scene keyframe SigLIP-2 embeddings if no end-to-end video model is registered).
5. Each scene becomes one `:Artifact` of `kind="video_scene"` linked via `:DERIVED_FROM` to the parent video Artifact. Each scene Artifact has its own keyframe ArtifactContent and its own caption Chunk + audio-segment Chunk(s).
6. Mean-pool Chunk embeddings of the scene → scene `Artifact.text_embedding`. Mean-pool all scene `text_embedding`s → parent video `Artifact.text_embedding`.

#### 5.3.4 PDF (`ingest_pdf`)

1. **Render pages to images** via `pdfium-render` (Rust). Each page becomes an `ImageInput` for VLM extraction.
2. **VLM extract per page** via `xervo.vlm_extract(alias, &[page_image], opts)`. Output per page: blocks of `(kind, content, bbox, reading_order)`.
3. **Create one PDF Artifact** with `kind="pdf"`, `page_count=N`. Bytes stored in `:ArtifactContent` (original PDF bytes, NOT the renders).
4. Optionally **store rendered pages as `:DERIVED_FROM` Artifacts** (`kind="image"`, `derivation_kind="pdf_page_render"`) for re-running VLM with a new model. Default: don't store, regenerate from PDF on demand. Cost: re-rendering 100 pages takes ~5 seconds with pdfium.
5. **Create Chunks per page block**: one Chunk per `DocBlock`, with `modality="text"` (the block content is text), `chunk_type` per `DocBlockKind` ("text" | "heading" | "table" | "figure" | "formula" | "caption"), `text=<block.content>`, `page_number=<page>`, `bbox=<block.bbox>`, `reading_order=<block.reading_order>`, `source_model_version=<vlm alias + version>`.
6. Chunks auto-embed → P2 + P3 → entities + observations as usual.
7. Mean-pool all Chunk embeddings → `Artifact.text_embedding`.

#### 5.3.5 HTML (`ingest_html`)

Replaces the silent text-fallback chunker with `dom_smoothie` (Rust-native Readability port):

1. Parse HTML via `dom_smoothie` → cleaned main content + structured headings.
2. Recursive 400-512-token chunking of the cleaned content with sentence-boundary alignment.
3. Each Chunk gets `modality="text"`, `chunk_type` per the section semantic ("text" | "heading" | "caption"), `heading` set from the nearest enclosing heading.

#### 5.3.6 Structured (`ingest_structured`) — CSV / JSON / Parquet

Replaces the silent text-fallback chunker with schema-aware grouping:

1. Parse via Arrow / Polars (native Rust, already in uni-db's dep graph via Lance).
2. Detect schema (column names + types).
3. Group rows by token budget (~400-512 tokens per chunk), preserving header schema at the top of each chunk's text representation.
4. Each Chunk: `modality="text"`, `chunk_type="table_row_group"`, `heading=<header schema as JSON>`, `text=<header + grouped rows as Markdown table>`.

---

## 6. Cross-modal recall — lazy modality activation

### 6.1 Mechanism

The recall cascade extends Phase 2 and Phase 3 with optional per-modality vector channels. Activation is corpus-conditional, not query-conditional or flag-conditional.

Per `KnowledgeBase`, maintain a cheap modality-presence cache:

```rust
pub struct ModalityPresence {
    pub has_image_content: bool,
    pub has_audio_content: bool,
    pub has_video_content: bool,
    pub has_multimodal_indexed: bool,  // multimodal_embedding populated for any artifact
}
```

Updated incrementally on artifact ingest (single bool flip), persisted as a Map on a singleton config node (`:KnowledgeBaseStats`). Read once per recall, cached for the call.

### 6.2 Phase 2 extension

Existing Phase 2 (Expand) searches `Episode.embedding`, `Observation.embedding`, `Message.embedding` (vector) and `Message.content` (fulltext). Extension:

- If `modality_presence.has_image_content`: also search `Artifact.image_embedding` (HNSW), using **the query's intent_vec** as the search key. This works because intent_vec lives in the same shared text+image space as image embeddings (SigLIP-2 dual-tower) — text query directly retrieves image artifacts whose visual content matches.
- If `modality_presence.has_audio_content`: also search `Artifact.audio_embedding`. Same logic — query intent_vec → audio embedding space (CLAP-style joint space when the audio embedder is a shared-space model; falls back to text-only via `Chunk.text` transcript channel otherwise).
- If `modality_presence.has_multimodal_indexed`: also search `Artifact.multimodal_embedding`. Reserved for Cohere Embed v4 / Gemini Embed 2 mixed-modality vectors.

All channels feed the existing RRF fusion (k=60). Tier weights unchanged.

### 6.3 Phase 3 extension

Existing Phase 3 (Broaden): BM25 + Chunk.text vector + Entity→MENTIONS + PPR. Extension:

- Same per-modality channels as Phase 2, but at higher candidate-set sizes (top-30 vs top-20).
- Chunk-level cross-modal: when a Chunk has `modality="image"` and `bbox` populated, surface the parent Artifact alongside the Chunk (so the bytes are reachable).

### 6.4 IntentProfile changes

```rust
pub struct IntentProfile {
    // Existing
    pub intent_vec: Vec<f32>,
    pub facet_vecs: Vec<Vec<f32>>,
    pub entity_refs: Vec<String>,
    pub facet_count: usize,

    // New
    pub query_modalities: QueryModalities,    // modality of the QUERY (not the corpus)
}

pub struct QueryModalities {
    pub has_image_input:  bool,    // user query included an image
    pub has_audio_input:  bool,    // user query included audio
    pub image_vec: Option<Vec<f32>>,  // SigLIP-2 vision embedding of query image, when present
    pub audio_vec: Option<Vec<f32>>,
}
```

When the user query includes an image (image-to-image search): use `image_vec` against `Artifact.image_embedding` directly, in addition to `intent_vec` (text → multimodal). Symmetric for audio.

### 6.5 Recall tier weights — unchanged for v1

Tier weights from spec §IX (Semantic 1.0 / Procedural 0.9 / Episodic 0.7 / Store 0.5 / Provenance 0.4) apply across modalities. An image artifact found by cross-modal vector search lives in the Store tier (it's a chunk-bearing artifact, not a fact or procedure). Adjusting per-modality tier weights is a v2 question once we have data on retrieval quality.

---

## 7. Provenance

### 7.1 `Action.PRODUCED.Artifact` — agent-produced

Existing edge from §uniko-spec-v6.md F18 and F20. Used when an Action's output overflows to an Artifact (>256 token rule). Unchanged.

### 7.2 `Artifact.CREATED_BY.Action` — agent-authored via tool call

Spec F30. Schema-defined today; not populated by production code. This proposal **does NOT** add population in v1 — flagged as a follow-up for the agent-tools surface PR. Agent tools that create artifacts (e.g., a tool that generates an image and writes it to memory) MUST write this edge when they land.

### 7.3 `Artifact.DERIVED_FROM.Artifact` — non-Action derivation chains

New edge (§3.4). Used for:
- PDF → rendered page images
- Video → keyframe images
- Video → audio track
- Video → scene-segment Artifacts

Always written by the relevant `ingest_*` entry point, never by an Action.

### 7.4 `Artifact.origin` — external provenance

New field (§3.2). Populated when the artifact came from outside the agent: URL, filesystem path with mtime, upload-session id. Stored as a Map:

```
origin: {
  source_type: "url" | "filesystem" | "upload" | "import",
  source_ref:  "<url>" | "<absolute path>" | "<session id>" | "<importer name>",
  ingested_at: <DateTime>,
}
```

Null when the artifact is agent-produced (use `CREATED_BY` edge instead).

---

## 8. Migration

### 8.1 Existing `Artifact.content: String` → `:ArtifactContent`

One-shot, idempotent migration:

```cypher
MATCH (a:Artifact)
WHERE a.content IS NOT NULL
WITH a, a.content AS text, coalesce(a.hash, sha256(a.content)) AS h
MERGE (c:ArtifactContent {content_id: h})
  ON CREATE SET c.bytes = text,
                c.mime = coalesce(a.mime_type, 'text/plain'),
                c.size = size(text),
                c.perceptual_hash = NULL,
                c.audio_fingerprint = NULL,
                c.created_at = coalesce(a.created_at, datetime())
MERGE (a)-[:HAS_CONTENT {role: 'primary'}]->(c)
REMOVE a.content
```

After migration:
- `Artifact.content` field is dropped from the schema in a follow-up cleanup.
- Existing dedup-by-hash (`Artifact.hash` BTree index) continues to work as a cache; the canonical hash now lives on the content node.
- HNSW vector search on `Artifact.*_embedding` is leaner — content bytes no longer share row pages with the embedding columns.

### 8.2 Chunk schema migration

Adding new nullable fields to `:Chunk` is a non-breaking schema migration. Existing rows keep `modality=NULL` until backfilled.

Backfill script (one-shot):

```cypher
MATCH (c:Chunk) WHERE c.modality IS NULL
SET c.modality = 'text'
```

Post-backfill, set `modality` as required on new Chunk inserts.

### 8.3 ArtifactContent migration validation

After migration runs:

```cypher
// Every Artifact should have exactly one HAS_CONTENT edge
MATCH (a:Artifact)
WHERE NOT (a)-[:HAS_CONTENT]->(:ArtifactContent)
RETURN count(a) AS orphan_artifacts  // expected: 0

// No Artifact should still have a non-null `content` field after migration
MATCH (a:Artifact) WHERE a.content IS NOT NULL
RETURN count(a) AS unmigrated  // expected: 0

// Hash cache should match the content node's content_id
MATCH (a:Artifact)-[:HAS_CONTENT]->(c:ArtifactContent)
WHERE a.hash IS NOT NULL AND a.hash <> c.content_id
RETURN count(a) AS mismatches  // expected: 0
```

These run as `cargo nextest run --test schema_migration_validation` post-migration.

---

## 9. API specifications

### 9.1 Ingest input types

```rust
pub enum ImageInput {
    Bytes { mime: String, data: Vec<u8> },
    Path  { path: PathBuf },
}

pub enum AudioInput {
    Bytes { mime: String, data: Vec<u8> },
    Pcm   { sample_rate: u32, channels: u16, samples: Vec<f32> },
    Path  { path: PathBuf },
}

pub enum VideoInput {
    Bytes { mime: String, data: Vec<u8> },
    Path  { path: PathBuf },
}

pub enum PdfInput {
    Bytes { data: Vec<u8> },
    Path  { path: PathBuf },
}

pub enum StructuredInput {
    Csv  { data: String, headers_present: bool },
    Json { data: String },
    Parquet { bytes: Vec<u8> },
}
```

`ImageInput` and `AudioInput` mirror the types in the xervo proposal — same enum, reused.

### 9.2 Ingest options

```rust
#[derive(Default)]
pub struct ImageIngestOptions {
    pub caption: bool,              // default true — run VLM caption
    pub ocr:     bool,              // default false
    pub vlm_alias:     Option<String>,  // default: "vlm/caption"
    pub embed_alias:   Option<String>,  // default: "embed/image"
    pub ocr_alias:     Option<String>,  // default: "ocr/default"
    pub dedupe_near_visual: bool,   // default false — log near-dups, don't alias
    pub origin: Option<ArtifactOrigin>,
}

#[derive(Default)]
pub struct AudioIngestOptions {
    pub transcribe:    bool,        // default true
    pub diarize:       bool,        // default false
    pub language:      Option<String>,
    pub transcribe_alias: Option<String>,  // default: "asr/default"
    pub embed_alias:      Option<String>,  // default: "embed/audio"
    pub origin: Option<ArtifactOrigin>,
}

#[derive(Default)]
pub struct VideoIngestOptions {
    pub scene_detect:  bool,        // default true
    pub transcribe:    bool,        // default true
    pub caption_scenes: bool,       // default true
    pub embed_alias:   Option<String>,
    pub origin: Option<ArtifactOrigin>,
}

#[derive(Default)]
pub struct PdfIngestOptions {
    pub render_pages_to_storage: bool,  // default false — render on demand
    pub vlm_alias:     Option<String>,
    pub origin: Option<ArtifactOrigin>,
}

#[derive(Default)]
pub struct HtmlIngestOptions {
    pub strip_navigation: bool,     // default true — dom_smoothie cleans nav/footer/sidebar
    pub origin: Option<ArtifactOrigin>,
}

#[derive(Default)]
pub struct StructuredIngestOptions {
    pub rows_per_chunk: Option<usize>,  // default: token-budget-driven
    pub origin: Option<ArtifactOrigin>,
}

pub struct ArtifactOrigin {
    pub source_type: String,  // "url" | "filesystem" | "upload" | "import"
    pub source_ref:  String,
}
```

### 9.3 Ingest result types

```rust
pub enum ImageIngestResult {
    Ingested {
        artifact_id: NodeId,
        content_id:  NodeId,
        chunks:      Vec<NodeId>,
    },
    DeduplicatedExact(NodeId),       // existing Artifact by hash
    AliasedToExisting(NodeId),       // near-duplicate, used existing content
    NearDuplicateDetected {
        artifact_id: NodeId,
        near_dups:   Vec<NodeId>,
    },
}

// AudioIngestResult, VideoIngestResult, PdfIngestResult, HtmlIngestResult,
// StructuredIngestResult: parallel shape — variants for Ingested, DeduplicatedExact,
// and modality-specific outcomes.
```

---

## 10. Rollout phases

The rollout is split into two tracks that progress independently:

- **Track A (Phases 1-5) — pure uniko.** Schema, storage, text-modality quality wins, recall plumbing, and native-Rust binary preprocessing. Zero dependency on xervo PR 1. Land first, land in parallel with the xervo work happening in the sibling repo.
- **Track B (Phases 6-9) — xervo-dependent.** Per-modality semantic extraction (image/PDF/audio/video) plus cross-modal recall activation and the Mem-Gallery score. Each phase here is a thin drop into already-wired ingest skeletons and recall channels — the heavy lifting is on the xervo provider side.

This ordering means:
- Phase 2 closes three documented spec gaps (F24 HTML/CSV silent fallback, F25 unpopulated multimodal fields, F28 unpopulated `Artifact.text_embedding` HNSW index) — but does **not** move LoCoMo/LME numbers, which are dialog-only benches that don't ingest Artifacts on the hot path. The Phase 2 win is correctness + foundation, not benchmark score.
- When xervo PR 1 lands, Track B is mostly wiring + tests — the schema, dedup, recall channels, and ingest skeletons are already in place from Track A.

---

### Track A — self-contained uniko work (no xervo dependency)

#### 10.1 Phase 1 — schema + storage layer

- New `:ArtifactContent` label with all columns and indexes.
- New `HAS_CONTENT` and `DERIVED_FROM` edges.
- New `:Artifact` columns (typed modality metadata + `origin`).
- New `:Chunk` columns (modality positioning + `source_model_version`).
- Migration script for `Artifact.content` → `:ArtifactContent`.
- `:KnowledgeBaseStats` singleton scaffolding (cache reads/writes; flags all default false).
- Backfill `Chunk.modality='text'` for existing rows.

Verification: `cargo nextest run -p uniko-store --test schema_completeness` passes with new label/edge expectations; migration script is idempotent (run twice → no diff); validation queries in §8.3 return zeros.

#### 10.2 Phase 2 — text-modality quality (no xervo, immediate recall wins)

- Replace `HtmlChunker` stub with `dom_smoothie`-based real DOM extraction.
- Replace `StructuredChunker` stub with Arrow/Polars-driven schema-aware row grouping.
- Implement mean-pool helper for `Artifact.text_embedding`.
- Backfill `Artifact.text_embedding` for all existing text/code artifacts that have child Chunks.
- Populate `Artifact.text_embedding` on every new text-path ingest going forward.

Verification: ingest known HTML pages and CSVs end-to-end, assert non-text-fallback chunk types land; LoCoMo / LME re-run shows non-regression (and ideally a small bump from better chunking + populated artifact-level vectors).

#### 10.3 Phase 3 — recall plumbing (no-op until Track B fills in embeddings)

- `IntentProfile` extension with `query_modalities` field (always-NULL for text-only callers in this phase).
- `ModalityPresence` cache wired into Phase 2 / Phase 3 recall code; lazy-activation branches added but all `has_*_content` flags are false in a text-only corpus → branches stay dormant.
- RRF fusion accepts the new (currently empty) channels without changing rankings.
- Add per-modality channel toggles to `RecallConfig` (default off).

Verification: `recall_modality_lazy_e2e.rs` test: text-only corpus, assert image/audio/video channels do NOT fire (instrument call count); LoCoMo / LME re-run shows zero behavior change.

#### 10.4 Phase 4 — native-Rust binary preprocessing (optional early; pure Rust deps)

This phase lands the deterministic, model-free parts of binary ingest. The resulting Artifacts carry bytes + metadata + dedup signals but **no semantic chunks** until Track B fills them in. Useful for early storage-layer validation and as a foundation Track B drops into.

- `image-hasher` integration: pHash computed on every image ingest; near-dup detection working.
- `chromaprint-rs` integration: audio fingerprint computed on every audio ingest.
- `pdfium-render` integration: PDF page rendering to images (`PdfInput` → `Vec<ImageInput>`); no VLM extraction yet, so per-page Chunks are not created.
- `ffmpeg-the-third` integration: video scene-boundary detection + keyframe extraction; scene Artifacts created via `:DERIVED_FROM`; no per-scene captions yet.
- Skeleton `ingest_image` / `ingest_audio` / `ingest_video` / `ingest_pdf` entry points that perform hash + dedup + `:ArtifactContent` write + Artifact metadata + (where applicable) `:DERIVED_FROM` page/scene Artifacts. The semantic-extraction steps (caption / transcript / OCR / VLM-extract) emit a `todo!()`-equivalent skip with a log warning.

Verification: ingest 100 images / 10 PDFs / 10 audio clips / 5 videos; assert `:ArtifactContent` dedup behaves correctly across duplicates and near-duplicates; assert byte sizes and `:DERIVED_FROM` graph shape; assert recall on the binary corpus returns nothing semantic (expected — Track B fills this in).

**Note:** Phase 4 is optional-early. If you'd rather wait for xervo PR 1 and land binary preprocessing in lockstep with the semantic-extraction work, Phase 4 fuses cleanly into the Track B phases. Splitting it out is justified only if you want to validate storage/dedup on real binary corpora before the xervo work is ready.

#### 10.5 Phase 5 — Mem-Gallery bench harness (scaffolding only)

- New `crates/uniko-bench` binary `mem-gallery-bench` with the Mem-Gallery harness skeleton.
- Wire the bench against Track A ingest paths (text/HTML/structured + binary preprocessing).
- Numbers from Phase 5 reflect Track A capabilities only — expected to be low on Mem-Gallery's vision-heavy tasks until Track B lands. Bench in place to measure progress phase-by-phase through Track B.

Verification: bench runs end-to-end against Mem-Gallery's public split; produces a baseline number (deliberately low on multimodal tasks).

---

### Track B — xervo-dependent phases (require xervo PR 1)

These phases assume `xervo.embed_image`, `xervo.embed_audio`, `xervo.embed_multimodal`, `xervo.vlm_extract`, `xervo.transcribe`, `xervo.ocr`, and `xervo.nlp_analyze` are callable. Each phase is a thin drop into already-wired skeletons from Track A.

#### 10.6 Phase 6 — image ingest end-to-end

- xervo provider impl for `embed/image` (SigLIP-2-so400m ONNX).
- xervo provider impl for `vlm/caption` (Granite-Docling or a small image-caption VLM).
- `ingest_image` body completes: caption Chunk creation, optional OCR Chunks, `image_embedding` population, mean-pool into `text_embedding`.
- `ModalityPresence.has_image_content` starts flipping true on ingest → recall channel activates automatically (Track A plumbing).

Verification: ingest 100 images; query "find me images of X" returns relevant results; both `text_embedding` and `image_embedding` populated; cross-modal channel fires in Phase 2 recall.

#### 10.7 Phase 7 — PDF ingest end-to-end

- xervo provider impl for MinerU-2.5 (or Granite-Docling at higher quality tier) for `vlm_extract`.
- `ingest_pdf` body completes: per-page VLM extraction, per-block Chunks with `page_number` / `bbox` / `reading_order`, `text_embedding` mean-pooled across all page chunks.
- `:DERIVED_FROM` edges for rendered pages when `render_pages_to_storage=true` (uses Track A pdfium scaffolding).

Verification: ingest a 50-page PDF with tables and formulas; chunks land with correct positioning; OmniDocBench-style retrieval queries land relevant pages.

#### 10.8 Phase 8 — audio ingest end-to-end

- New xervo provider `local/whisper-cpp` implementing `TranscriptionModel`.
- Optional CLAP via `local/onnx` for `embed/audio`.
- `ingest_audio` body completes: per-segment `transcript_segment` Chunks with timestamps + speaker labels, `audio_embedding` population.
- `ModalityPresence.has_audio_content` starts flipping true → recall channel activates.

Verification: ingest 10 minutes of multi-speaker audio with `diarize=true`; chunks land per segment with correct timestamps and speaker labels; transcript content surfaces in recall via `Chunk.text` channel.

#### 10.9 Phase 9 — video ingest + cross-modal recall validation + Mem-Gallery score

- `ingest_video` body completes: scene Artifacts (from Track A scene-detect) get per-scene VLM captions + per-scene audio transcripts; video-level `Artifact.text_embedding` mean-pooled from scenes; `Artifact.video_embedding` populated (mean-pool of keyframe SigLIP-2 if no end-to-end video model registered).
- Cross-modal recall validation: `recall_cross_modal_e2e.rs` and `recall_image_query_e2e.rs` go green now that embeddings exist.
- Re-run Mem-Gallery bench against the full ingest surface; publish a real score.

Verification: ingest a 5-minute video; scene Artifacts land with correct `:DERIVED_FROM` edges; recall "find the scene where Alice says X" works; Mem-Gallery score published.

---

### Phase ordering rationale

| If you can only do | Land Phases | Result |
|---|---|---|
| Schema + a small text-quality win | 1, 2 | LoCoMo/LME potentially improves; storage shape ready |
| All of Track A | 1–5 | uniko-side is "done" pending xervo; binary corpora can be ingested + deduped; Mem-Gallery baseline measured |
| Track A + image only | 1–6 | Images findable cross-modally |
| Track A + image + PDF | 1–7 | Most production-shaped multimodal use cases covered |
| Everything | 1–9 | Full multimodal store; Mem-Gallery scoreable |

The dependency graph: **1 → {2, 3, 4} → 5** is Track A, fully internal. **6, 7, 8 depend only on xervo PR 1 + Phase 1 (schema)** — they can land in any order once xervo is ready. **Phase 9 depends on 6 + 8** (video needs both image VLM + audio ASR). Phase 5 (bench scaffolding) can re-run anytime to re-measure.

---

## 11. Test strategy

### 11.1 Unit-level

- **Schema completeness** (existing `uniko-store/tests/schema_completeness.rs` extended): asserts presence of all new fields, edges, indexes.
- **CAS dedup semantics**: ingest same bytes twice via different ingest paths → same `:ArtifactContent` node, two different Artifacts.
- **Migration idempotence**: run migration script twice → no schema or row diff.
- **Perceptual hash correctness**: known image pair → expected Hamming distance.
- **Mean-pool correctness**: known chunk embeddings → expected pooled vector.

### 11.2 Integration tests (per `ingest_*` entry point)

Each gets a `crates/uniko-extract/tests/ingest_<modality>_e2e.rs` file with:
- Happy path: ingest one artifact, assert nodes/edges/chunks/embeddings landed correctly.
- Dedup path: ingest the same bytes twice, assert only one `:ArtifactContent`.
- Near-dup path: ingest two perceptually-similar images, assert detection.
- Origin tracking: `origin` field round-trips.
- Embedding population: all expected modality embeddings non-NULL after ingest.

### 11.3 Recall tests

- `recall_cross_modal_e2e.rs`: ingest mixed corpus, query text-only → assert image artifacts surface via cross-modal channel.
- `recall_modality_lazy_e2e.rs`: ingest text-only corpus, instrument recall → assert image/audio/video channels do NOT fire.
- `recall_image_query_e2e.rs`: query with `ImageInput` → assert similar images surface.

### 11.4 Acceptance criteria for the full rollout

- All Phase 1-7 verifications pass.
- Existing LoCoMo / LongMemEval scores do not regress (text-only paths unchanged in behavior).
- New `mem-gallery-bench` produces a published number.

---

## 12. Risks

| Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|
| Lance row-size limits hit by large blobs | Medium | Medium | 2 GB soft cap; require upstream chunking for video > 2 GB |
| Cross-modal vector channel pollutes recall results when caption matches but visual doesn't | Medium | Low | RRF fusion naturally down-weights single-channel hits; monitor with phase1_only_pct delta on multimodal corpora |
| MinerU 2.5 / Granite-Docling ONNX export quality lower than the Python pipeline | Medium | Medium | Acknowledged in xervo proposal; ~5-10% absolute on hard tables. Acceptable for v1 |
| `pdfium-render` license / runtime concerns | Low | Low | Apache-2.0 wrapper around Apache-2.0 Pdfium; battle-tested |
| Modality-presence cache drift (ingest happens, cache misses the update) | Low | Medium | Cache update wrapped in same transaction as Artifact insert |
| Migration script fails on a subset of existing Artifacts | Low | High | Run in dry-mode first; validation queries in §8.3; staged rollout per KB |
| Caption / transcript / OCR model upgrades silently produce drifted observations | Medium | Medium | `source_model_version` on every derived Chunk; future P9 re-derivation sweep |
| Perceptual-hash false positives alias semantically different but visually similar images | Low | Medium | `dedupe_near_visual=false` by default (log only, don't alias) |
| Cohere v4 / Gemini Embed 2 API changes break `multimodal_embedding` population | Medium | Low | Multimodal embedding is opt-in per artifact; absence doesn't break recall |
| `image-hasher` / `chromaprint-rs` maintenance regression | Low | Low | Both are leaf deps; vendor or replace if needed |

---

## Appendix A — Schema diff summary

### New labels

```
:ArtifactContent
  content_id: String (Hash index)
  bytes: Bytes
  mime: String (Hash index)
  size: Int64 (BTree)
  perceptual_hash: Optional Int64 (BTree)
  audio_fingerprint: Optional Bytes
  created_at: DateTime
```

### New edges

```
:HAS_CONTENT (Artifact → ArtifactContent)
  role: Optional String

:DERIVED_FROM (Artifact → Artifact)
  derivation_kind: String
  derived_at: DateTime
```

### Modified labels

```
:Artifact (existing fields preserved; new fields):
  + width, height: Optional Int32
  + duration_ms: Optional Int64 (BTree)
  + sample_rate: Optional Int32
  + channels: Optional Int16
  + fps: Optional Float32
  + frame_count: Optional Int32
  + page_count: Optional Int32 (BTree)
  + modality_meta: Optional CypherValue
  + origin: Optional CypherValue
  - content: String  (dropped after migration)

:Chunk (existing fields preserved; new fields):
  + modality: String (Hash index)
  + bbox: Optional List<Float32>
  + time_start_ms: Optional Int64 (BTree)
  + time_end_ms: Optional Int64
  + page_number: Optional Int32 (BTree)
  + reading_order: Optional Int32
  + source_model_version: Optional String
```

### Singleton node

```
:KnowledgeBaseStats (singleton)
  modality_presence: Map<String, Bool>   // "image" -> true, "audio" -> false, …
  updated_at: DateTime
```

---

## Appendix B — How this maps to uniko-spec-v6 functional requirements

| Spec ID | Today | After this design |
|---|---|---|
| F23 (ingest artifacts) | text + code only | all modalities via dedicated entry points |
| F24 (chunking by content type) | text + tree-sitter only; rest silently text-fall-back | real HTML / PDF / CSV / audio / video chunkers via xervo VLM / `dom_smoothie` / `Whisper` / scene detection |
| F25 (multimodal embedding fields) | schema-reserved, unpopulated | populated via xervo `embed_image` / `embed_audio` / `embed_multimodal` |
| F27 (FullText on Chunk.text + Message.content) | works | unchanged; Chunk.text now carries captions / transcripts / OCR uniformly |
| F28 (vector search on all embedding fields) | partial (Artifact.text_embedding unpopulated) | populated end-to-end |
| F29 (dedup by content hash) | exact SHA-256 only | exact + perceptual (image) + acoustic fingerprint (audio) |
| F30 (provenance via CREATED_BY / MODIFIED_BY) | schema-only | `origin` field for non-Action provenance; `:DERIVED_FROM` for non-Action derivation; `CREATED_BY` still required from agent-tool surface (separate PR) |

---

## Appendix C — What this does NOT change

For clarity, the following remain identical to current uniko-spec-v6 behavior:

- BTIC temporal validity on Facts (F36).
- Contradiction detection and drift detection (F38 / F39).
- Topic detection via weighted LPA (F40).
- Procedure lifecycle (F41-F44).
- Rule lifecycle + stdlib rules (F44-F45, F50, F52).
- Recall Phase 1 (Compact) — Fact / Procedure / Topic vector search.
- Consolidation pipeline (P4).
- Procedure promotion / topic detection sweeps (P5 / P6).
- Working memory traversal (F13).
- Access control (F66).
- NL-to-Cypher (F61).

A multimodal corpus produces facts, procedures, topics, and observations through the same consolidation pipeline as a text-only corpus — because the modality-native content lands as `Chunk.text` (caption / transcript / OCR / extracted markdown) which feeds P2 (NER) → P3 (observations) → P4 (consolidation) identically.

The multimodal channel is additive: it adds new ingest paths, new embedding columns, new recall channels, and new dedup signals. It does not change the cognitive memory pipeline.
