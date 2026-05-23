# Embedding Analysis: Schema v3 (Revised)


## Embedding Strategy Per Node

### Auto-Embed (uni-db computes on insert, single source field)

| Node | Source Field | Example | Notes |
|------|-------------|---------|-------|
| **Message** | content | "I went to a LGBTQ support group yesterday..." | Text messages only. Image/code messages may need separate handling. |
| **Chunk** | text | "Researching adoption agencies — it's been a dream..." | Works for prose and code. Future: code-specific model when language is set. |
| **Observation** | content | "Caroline attended LGBTQ support group" | Always natural language. Clean auto-embed. |
| **Summary** | text | "Caroline and Melanie discussed career plans..." | Always natural language. |


### Computed Single Embedding (Pipeline 7 constructs embed string, then sets)

| Node | Computation | Example | Notes |
|------|-------------|---------|-------|
| **Entity** | name + " (" + entity_type + ")" | "Java (programming_language)" | Disambiguates by type. Falls back to just name if no type. |
| **Goal** | title | "Reduce refund cycle time by 40%" | Embed title only. Description may be paragraphs — dilutes signal. |
| **Task** | title | "Investigate auth module dependencies" | Same reasoning as Goal. |
| **Session** | topic (initially), topic + summary (after session ends) | "LGBTQ support group, career plans" | Re-embed when summary is generated. |
| **Topic** | name (initially), name + summary (after generation) | "Caroline's adoption journey" | Re-embed when summary is generated. |
| **Fact** | subject + " " + predicate + " " + object | "Caroline pursuing adoption" | Always short text. Concatenation works. |
| **Procedure** | name + ": " + description | "revenue-dip analysis: 7-step diagnostic playbook..." | Truncate description to ~200 chars for embedding. |
| **Episode** | extracted topic from state Json + action_type | "LGBTQ support group, career plans — conversation" | See detailed Episode section below. |
| **Action** | action_type + key input summary | "file_read /src/auth.rs" | Extract key fields from input Json. See details below. |


### Pooled Embedding (aggregated from child embeddings)

| Node | Source | Method | Notes |
|------|--------|--------|-------|
| **Artifact (text)** | Chunk embeddings | Mean-pool | Cannot auto-embed — content may be 10MB. Pool chunk vectors after chunking. |


### Multi-Modal Embedding (different models per modality)

| Node | Modality | Model Type | Embedding Field | Notes |
|------|----------|-----------|----------------|-------|
| **Artifact (text)** | Text | Text embed model | text_embedding | Mean-pooled from chunks |
| **Artifact (image)** | Vision | CLIP / SigLIP | image_embedding | Embeds the image directly |
| **Artifact (audio)** | Audio | CLAP / Whisper+embed | audio_embedding | Transcribe then embed, or native audio embed |
| **Artifact (video)** | Video | Frame sampling + pooling | video_embedding | Sample frames → vision embed → pool. Plus audio track. |

For Artifact, the schema needs multiple optional embedding fields:

```
Artifact
  ...
  text_embedding       Vector      nullable    -- mean-pooled from chunks (text/code)
  image_embedding      Vector      nullable    -- vision model: CLIP/SigLIP (images, video frames)
  audio_embedding      Vector      nullable    -- audio model: CLAP (audio, video audio track)
  video_embedding      Vector      nullable    -- video model: LanguageBind/InternVideo (full video)
  multimodal_embedding Vector      nullable    -- unified space: ImageBind/ONE-PEACE (any modality)
```

**Five embedding fields, each serving a different purpose:**

| Field | Model Type | What It Captures | When Set |
|-------|-----------|-----------------|----------|
| text_embedding | Text (AllMiniLM, etc.) | Semantic meaning of text content | Text artifacts: pooled from chunks |
| image_embedding | Vision (CLIP, SigLIP) | Visual content of images | Images: direct. Video: pooled from sampled frames. |
| audio_embedding | Audio (CLAP) | Audio content, music, speech patterns | Audio files. Video: extracted audio track. |
| video_embedding | Video (LanguageBind, InternVideo) | Temporal visual dynamics, motion, scenes | Video files: native video understanding |
| multimodal_embedding | Unified (ImageBind, ONE-PEACE) | Cross-modal representation in shared space | Any modality: enables cross-modal search |

Each field has its own HNSW index with potentially different
dimensions (text=384, CLIP=768, CLAP=512, video=768, ImageBind=1024).

**Search patterns:**

- Text search → text_embedding (find documents about X)
- Image search → image_embedding (find visually similar images)
- Audio search → audio_embedding (find similar sounds/speech)
- Video search → video_embedding (find similar video content)
- Cross-modal search → multimodal_embedding (find images from text
  query, find text from image query, find video from audio query —
  all modalities mapped to the same space)


### No Embedding

| Node | Reason |
|------|--------|
| **Participant** | Queried by participant_id (Hash), kind (Hash), name (Fulltext). Semantic search over participant names adds nothing over fulltext. |
| **Rule** | Queried by name (Hash), status (Hash), source_type (Hash). Semantic search over Locy source code is not useful. |
| **ConsolidationCycle** | Queried by cycle_id, agent_id, started_at. Operational metadata, not semantic content. |


## Detailed: Episode Embedding

The hardest embedding to get right because state is domain-specific Json.

**Strategy:** Extract text from state Json using a priority list of keys.

```
Priority order:
  1. state.topic       → "LGBTQ support group, career plans"
  2. state.question    → "What does RAII stand for?"
  3. state.description → "Investigating auth module dependencies"
  4. state.summary     → "User reported login failures on mobile"
  5. state.input       → (first 200 chars of input text)
  6. (none found)      → fall back to action_type alone

Embed string = extracted_text + " — " + action_type + " " + outcome
Example:     = "LGBTQ support group, career plans — conversation complete"
```

This gives each Episode a unique semantic position based on what
was actually discussed/done, not just the action category.


## Detailed: Action Embedding

Actions are operational records. The embedding should capture what
was done and on what target.

**Strategy:** Extract key identifiers from input Json.

```
For action_type = "file_read":
  input = {"path": "/src/auth.rs"}
  embed = "file_read /src/auth.rs"

For action_type = "command_run":
  input = {"command": "cargo test --lib"}
  embed = "command_run cargo test --lib"

For action_type = "search":
  input = {"query": "authentication flow"}
  embed = "search authentication flow"

For action_type = "delegate":
  input = {"agent": "reviewer", "task": "review PR #42"}
  embed = "delegate review PR #42"

General: action_type + " " + first_string_value_from_input
```


## Detailed: Artifact Multi-Modal

**Text artifacts (files, documents, code):**
```
1. Ingest → chunk into Chunk nodes → each Chunk auto-embeds from text
2. Pipeline 7: mean-pool all chunk embeddings → set Artifact.text_embedding
3. If content is short (< 512 tokens): also auto-embed directly
```

**Image artifacts:**
```
1. Ingest → store raw bytes or URL
2. Pipeline 7: run vision model (CLIP/SigLIP) → set Artifact.image_embedding
3. Optionally: generate text description via VLM → store as Summary → auto-embed
```

**Audio artifacts:**
```
1. Ingest → store raw bytes or URL
2. Pipeline 7: transcribe (Whisper) → store transcript as child Artifact or Summary
3. Embed transcript chunks → pool → set Artifact.audio_embedding
4. Or: native audio embedding (CLAP) → set Artifact.audio_embedding
```

**Video artifacts:**
```
1. Ingest → store reference
2. Pipeline 7:
   a. Extract audio track → transcribe → embed (audio_embedding)
   b. Sample N frames → vision embed each → pool (image_embedding)
   c. Native video model → full temporal understanding (video_embedding)
   d. Optionally: generate scene descriptions → embed (text_embedding)
```

**Any artifact (multimodal embedding):**
```
1. After modality-specific embedding is computed
2. Pipeline 7: run unified model (ImageBind/ONE-PEACE)
   - Text → multimodal_embedding
   - Image → multimodal_embedding
   - Audio → multimodal_embedding
   - Video → multimodal_embedding
   All map to the same vector space, enabling cross-modal retrieval.
```


## Vector Index Summary

| Index | Node.Property | Dimensions | Auto? | Search Use |
|-------|--------------|------------|-------|------------|
| idx_Message_embedding | Message.embedding | 384 | Yes (content) | Phase 2/3: find relevant messages |
| idx_Chunk_embedding | Chunk.embedding | 384 | Yes (text) | Phase 3: find relevant content chunks |
| idx_Observation_embedding | Observation.embedding | 384 | Yes (content) | Phase 2: find relevant observations |
| idx_Summary_embedding | Summary.embedding | 384 | Yes (text) | Phase 1/2: find relevant summaries |
| idx_Entity_embedding | Entity.embedding | 384 | Computed | Phase 1: find entities by semantics |
| idx_Fact_embedding | Fact.embedding | 384 | Computed | Phase 1: find facts by semantics |
| idx_Goal_embedding | Goal.embedding | 384 | Computed | Working memory: find relevant goals |
| idx_Task_embedding | Task.embedding | 384 | Computed | Working memory: find relevant tasks |
| idx_Session_embedding | Session.embedding | 384 | Computed | Phase 2: find relevant sessions |
| idx_Topic_embedding | Topic.embedding | 384 | Computed | Phase 1: find relevant topics |
| idx_Procedure_embedding | Procedure.embedding | 384 | Computed | Phase 1: find applicable procedures |
| idx_Episode_embedding | Episode.embedding | 384 | Computed | Phase 2: find similar experiences |
| idx_Action_embedding | Action.embedding | 384 | Computed | Phase 2/3: find similar actions |
| idx_Artifact_text | Artifact.text_embedding | 384 | Pooled | Phase 3: find relevant text artifacts |
| idx_Artifact_image | Artifact.image_embedding | 768 | Computed | Find visually similar content |
| idx_Artifact_audio | Artifact.audio_embedding | 512 | Computed | Find similar audio content |
| idx_Artifact_video | Artifact.video_embedding | 768 | Computed | Find similar video content |
| idx_Artifact_multimodal | Artifact.multimodal_embedding | 1024 | Computed | Cross-modal search (any→any) |

*Dimensions are model-dependent. Typical values:
  Text (AllMiniLM): 384d, Vision (CLIP): 768d, Audio (CLAP): 512d,
  Video (InternVideo): 768d, Unified (ImageBind): 1024d.


## Recall Cascade: Updated Embedding Usage

```
Phase 1 (Compact):
  vector search → Fact.embedding, Procedure.embedding, Topic.embedding
  "What did Caroline research?" → finds Fact("Caroline pursuing adoption")

Phase 2 (Expand):
  vector search → Episode.embedding, Observation.embedding, Session.embedding
  keyword/fulltext → Message.content, Observation.content
  "When did she go to the support group?" → finds Episode(session 1) + Observation

Phase 3 (Broaden):
  fulltext search → Chunk.text, Message.content, Artifact.content
  vector search → Chunk.embedding, Message.embedding, Artifact.text_embedding
  Graph traversal → Entity → MENTIONS → Chunk/Message
  "What activities does Melanie do?" → Entity(Melanie) → all MENTIONS → collect
```
