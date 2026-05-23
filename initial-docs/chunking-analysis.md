# Chunking Plan

## What Gets Chunked

Only two node types produce Chunk nodes:

1. **Artifact** → Chunk (always) — files, documents, code, audio, video
2. **Message** → Chunk (only when content exceeds 1024 tokens)

Everything else is either short enough to embed directly, or is
structured data (Json) that isn't prose-searched. Large Action
outputs should be stored as Artifacts via the PRODUCED edge, then
chunked through the Artifact path.


## Chunking Strategies (by content type)

### Text Documents (text/plain, text/markdown)

```
Strategy: Recursive splitting (primary)
  1. Split text recursively at paragraph > sentence boundaries
  2. Target chunk size: 400-512 tokens
  3. Sentence-boundary aligned (never break mid-sentence)
  4. Overlap: 10-20% of chunk size from previous chunk for context

Optional enhancement: Semantic sentence grouping
  1. Compute embedding per sentence (lightweight, batch)
  2. Group adjacent sentences with cosine similarity > 0.7
  3. Break when group exceeds max_chunk_tokens
  Note: Only used when embedding model is available and
  higher-quality chunking is worth the latency cost.

Chunk metadata:
  chunk_type: "paragraph" | "sentence_group"
  heading: nearest preceding heading (for markdown/html)
```


### Code (text/x-python, text/x-rust, text/javascript, etc.)

```
Strategy: tree-sitter AST-based
  1. Parse with tree-sitter for the detected language
  2. Extract top-level declarations: functions, classes, structs, enums
  3. Each declaration = one chunk
  4. Module-level statements grouped into one chunk
  5. Import/use blocks = one chunk
  6. Large functions (> max_chunk_tokens): split at block boundaries

Chunk metadata:
  chunk_type: "function" | "class" | "struct" | "module" | "imports" | "block"
  language: "rust" | "python" | "javascript" | ...
  symbol_name: "process_order" | "AuthService" | ...
```


### HTML / XML (text/html, application/xml)

```
Strategy: DOM section extraction
  1. Parse DOM
  2. Strip non-content: nav, footer, script, style
  3. Split at <article>, <section>, <h1-h6> boundaries
  4. Tables → one chunk per table (header row in metadata)
  5. Lists → one chunk per list (with context from preceding paragraph)

Chunk metadata:
  chunk_type: "heading_section" | "table" | "list"
  heading: section heading text
```


### PDF (application/pdf)

```
Strategy: Page extraction + text chunking
  1. Extract text per page (preserve reading order)
  2. Detect multi-column layouts → merge columns
  3. Apply text document chunking to extracted text
  4. Tables → extract separately, one chunk per table
  5. Images → extract as separate Artifact nodes (kind: "image")

Chunk metadata:
  chunk_type: "paragraph" | "table"
  heading: detected section heading
```


### CSV / Structured Data (text/csv, application/json)

```
Strategy: Schema-aware row grouping
  1. Header row → stored as chunk metadata, not a separate chunk
  2. Group rows by logical record boundaries
  3. If no record boundaries: group N rows (N = max_chunk_tokens / avg_row_tokens)
  4. Each chunk includes header context for self-contained understanding

Chunk metadata:
  chunk_type: "row_group"
  heading: column headers as comma-separated string
```


### Audio (audio/*)

```
Strategy: Speaker-turn chunking with transcript
  1. Transcribe via Whisper or speech-to-text model
  2. Run speaker diarization (who spoke when)
  3. One chunk per speaker turn
  4. If no diarization available: fixed segments aligned to sentence boundaries
  5. If turn exceeds max_chunk_tokens: split at sentence boundaries within turn

Chunk metadata:
  chunk_type: "speaker_turn" | "audio_segment"
  speaker: "Caroline" | "Speaker_1" | ...
  start: 45000 (ms)
  end: 67000 (ms)
```


### Video (video/*)

```
Strategy: Scene boundary + transcript alignment
  1. Detect scene boundaries (frame-level cosine distance > threshold)
  2. Extract audio track → transcribe + diarize
  3. Align transcript chunks to scene boundaries
  4. Each chunk = one scene with its transcript text
  5. Extract keyframe per scene → store as child Artifact (kind: "image")

Chunk metadata:
  chunk_type: "scene"
  speaker: (from transcript, if diarized)
  start: 120000 (ms)
  end: 185000 (ms)
```


### Images (image/*)

```
Strategy: No chunking
  Images are atomic. One image = one Artifact, no Chunks.
  Embedding computed on the Artifact directly (image_embedding).

  Future consideration: region-based chunking for large images
  (object detection → one chunk per detected object/region).
```


## Message Chunking

Messages below 1024 tokens → no chunking. The Message node embeds
directly from content.

Messages above 1024 tokens → chunked using text document strategy:

```
Message (content: full text, embedding: auto-embed from content)
  ─HAS_CHUNK→ Chunk (text: segment 1, index: 0)
  ─HAS_CHUNK→ Chunk (text: segment 2, index: 1)

Message embedding: auto-embed from full content (model truncates)
Chunk embeddings: auto-embed from chunk text (precise per-segment)
```

Retrieval uses chunk embeddings for precision within long messages.
The Message embedding provides a coarse signal for the whole message.


## Action Output Handling

Large Action outputs are NOT chunked inline. Instead:

```
Action (output: {summary: "Build failed with 3 errors", truncated: true})
  ─PRODUCED→ Artifact (kind: "snippet", content: full 500-line output)
                ─HAS_CHUNK→ Chunk (text: lines 1-50)
                ─HAS_CHUNK→ Chunk (text: lines 51-100)
                ...
```

The Action.output field carries a summary. The full output lives in
an Artifact, which gets chunked and embedded through the normal path.
This keeps Action nodes lightweight while making output searchable.


## Chunking Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| max_chunk_tokens | 512 | Target max tokens per chunk |
| min_chunk_tokens | 64 | Minimum (avoid tiny fragments) |
| overlap_sentences | 1 | Sentences repeated from previous chunk for context |
| semantic_similarity_threshold | 0.7 | Cosine threshold for semantic sentence grouping |
| code_split_level | "function" | tree-sitter split level: "function", "class", "block" |
| audio_segment_ms | 30000 | Default audio segment if no diarization |
| video_scene_threshold | 0.5 | Cosine distance threshold for scene boundaries |
| message_chunk_threshold | 1024 | Token count above which messages get chunked |
| action_output_artifact_threshold | 256 | Token count above which Action output becomes Artifact |


## Chunking Pipeline (Pipeline 1: Ingest, step 2)

```
Content arrives (Artifact or long Message)
  │
  ├─ Detect content type
  │   ├─ mime_type from Artifact
  │   └─ content_type from Message
  │
  ├─ Select chunker:
  │   ├─ text/plain, text/markdown       → recursive splitter (400-512 tokens, sentence-boundary aligned, 10-20% overlap)
  │   ├─ text/x-python, text/x-rust, ... → tree-sitter AST chunker
  │   ├─ text/html, application/xml      → DOM section chunker
  │   ├─ application/pdf                 → page extractor + text chunker
  │   ├─ text/csv, application/json      → schema-aware row grouper
  │   ├─ audio/*                         → transcribe + speaker-turn chunker
  │   ├─ video/*                         → scene detector + transcript aligner
  │   └─ image/*                         → no chunking
  │
  ├─ Create Chunk nodes with metadata (type, language, speaker, heading, symbol)
  │
  ├─ Create HAS_CHUNK edges with index
  │
  ├─ Each Chunk auto-embeds from text (uni-db auto-embed)
  │
  ├─ Queue chunks for Pipeline 2 (NER)
  │
  └─ Queue Artifact for embedding pooling (Pipeline 7)
      ├─ text_embedding: mean-pool chunk embeddings
      ├─ image_embedding: vision model on image content
      ├─ audio_embedding: audio model on audio content
      ├─ video_embedding: video model on video content
      └─ multimodal_embedding: unified model on any content
```


## Schema Impact

Chunk node updated in schema v3 with:
- chunk_type (Hash index) — enables "show me all function chunks"
- language (Hash index) — enables "show me all Rust chunks"
- symbol_name (Hash index) — enables "find the process_order function"
- speaker (Hash index) — enables "what did Caroline say?"
- heading (no index) — context for document sections
- mime_type (no index) — source content type for routing

HAS_CHUNK edge extended:
- Artifact → Chunk (existing)
- Message → Chunk (new — for long messages only)

No other schema changes needed. All other nodes are either short
enough to embed directly or should flow through Artifact.
