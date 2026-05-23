# Sub-Phase 6: Entity Extraction Pipeline (P2)

## Context

This phase implements Pipeline 2 -- the entity extraction pipeline that runs synchronously after P1 (ingest) for every Message and Chunk node. P2 is the first content intelligence pipeline: it reads raw text and identifies named entities (people, organizations, places, concepts, code symbols), deduplicates them against the existing graph, and creates MENTIONS edges linking source nodes to Entity nodes.

P2 uses a three-tier extraction strategy:

1. **Rule-based NER** (always runs, < 10ms) -- Regex patterns for proper nouns, dates, numbers, URLs, preference expressions.
2. **ONNX NER model** (always runs when available, < 100ms) -- Lightweight distilled spaCy NER model via `ort` crate for PERSON, ORG, GPE, EVENT, PRODUCT, WORK_OF_ART.
3. **Tree-sitter code entity extraction** (for code content) -- Function names, class/struct names, module names, import targets.

An optional **LLM enhancement path** runs asynchronously after the primary extraction to refine entity types, resolve complex coreferences, and catch entities missed by local NER. The LLM path respects the circuit breaker and is non-blocking -- P3 does not wait for it.

Entity deduplication is critical for graph quality. Without it, "Caroline", "caroline", and "Caroline Smith" would create three separate Entity nodes. The dedup pipeline uses exact match, case-insensitive match, and embedding similarity to merge entities while preventing incorrect merges across incompatible types.

The NER code lives in `uniko-extract` (content processing layer). The P2 pipeline step that orchestrates NER and integrates with the pipeline framework lives in `uniko-extract` as well, implementing the `Step` trait from `uniko-pipes`.

Latency targets: < 100ms local NER (NF5), 1-3s with LLM enhancement (NF14).

## Prerequisites

| Dependency | Status Required | What It Provides |
|---|---|---|
| Phase 5: Ingest Pipeline (P1) | Complete | Message and Chunk nodes exist in the graph with text content for extraction |
| Sub-phase 4: Pipeline Infrastructure | Complete | `Step` trait (from `uniko-pipes`), `PipelineContext`, `StepErrorPolicy`, `CircuitBreaker` for LLM gating |
| Phase 3: KnowledgeBase (L1) | Complete | Node/edge CRUD, Hash index lookups on Entity.name, vector search on Entity.embedding |
| Sub-phase 2: Schema Types | Complete | `Entity` node type with entity_id, name, entity_type, first_seen, last_seen, frequency, confidence, embedding. `MENTIONS` edge type with count. |
| `ort` crate (ONNX Runtime) | Available (feature-gated) | Inference on distilled NER model |
| `tree-sitter` + language grammars | Available (from Phase 5) | AST parsing for code entity extraction |
| Distilled NER ONNX model | Available as bundled asset | Weights for PERSON, ORG, GPE, EVENT, PRODUCT, WORK_OF_ART classification |

## Sub-phases

---

### 6.1 -- Rule-Based NER

**Objective:** Implement a fast, dependency-free named entity extractor using regex patterns and heuristics. This is the baseline that always runs, even when ONNX and LLM are unavailable.

#### Files

| File | Type | Purpose |
|---|---|---|
| `uniko-extract/src/ner/mod.rs` | New module root | Re-exports, orchestration |
| `uniko-extract/src/ner/rules.rs` | New | `extract_entities_rule_based` function, pattern matchers |
| `uniko-extract/src/ner/types.rs` | New | `RawEntity`, `EntityMatch`, shared types |

#### Struct

```rust
pub struct RawEntity {
    pub name: String,
    pub entity_type: Option<String>,  // None if type is unknown
    pub start: usize,                 // byte offset in source text
    pub end: usize,                   // byte offset end
    pub confidence: f64,              // 0.0-1.0
    pub source: ExtractionSource,     // Rules, Onnx, TreeSitter, Llm
}

pub enum ExtractionSource {
    Rules,
    Onnx,
    TreeSitter,
    Llm,
}
```

#### Function

```rust
pub fn extract_entities_rule_based(text: &str) -> Vec<RawEntity>
```

#### Pattern Categories

| Pattern | Regex/Heuristic | Entity Type | Confidence | Examples |
|---|---|---|---|---|
| **Capitalized proper nouns** | 2+ consecutive words starting with uppercase, not at sentence start | "person" / "place" / "org" (ambiguous) | 0.5 | "Caroline Smith", "New York", "Goldman Sachs" |
| **Quoted strings** | Text within `"..."` or `'...'` (single words or short phrases) | "concept" / "reference" | 0.4 | `"adoption"`, `"LGBTQ-inclusive"` |
| **Date/time expressions** | Regex patterns: `yesterday`, `last Monday`, `March 2024`, ISO 8601 dates, `\d+ (days|weeks|months|years) ago` | "date" | 0.8 | "yesterday", "last March", "2023-05-08" |
| **Numbers with units** | `\d+(\.\d+)?\s*(GB|MB|KB|ms|s|min|hr|\$|USD|EUR|%)` | "measurement" | 0.9 | "5 GB", "100ms", "$500" |
| **Email addresses** | Standard email regex | "reference" | 0.95 | "user@example.com" |
| **URLs** | `https?://[^\s]+` | "reference" | 0.95 | "https://example.com/path" |
| **Preference patterns** | `I (prefer|like|love|enjoy|hate|dislike|don't like|don't want)\s+(.+?)[\.\,\!]` | Extract the object as entity | 0.6 | "I prefer VSCode" -> Entity("VSCode") |

#### Implementation Notes

- All patterns run in a single pass over the text where possible. Use a `RegexSet` for efficiency.
- Overlapping matches: prefer the longer match. If two patterns match the same span, prefer the one with higher confidence.
- Sentence-start disambiguation: "The" at sentence start is not a proper noun. Track sentence boundaries to avoid false positives from capitalization.
- Strip common determiners from entity names: "the LGBTQ group" -> "LGBTQ group".
- Normalize whitespace in entity names: collapse multiple spaces to single space, trim.

---

### 6.2 -- ONNX NER Model Integration

**Objective:** Integrate a lightweight pre-trained NER model (distilled spaCy NER exported to ONNX) for higher-accuracy entity extraction on prose text.

#### Files

| File | Type | Purpose |
|---|---|---|
| `uniko-extract/src/ner/onnx.rs` | New | `OnnxNer` struct, `extract_entities_onnx` function, tokenizer, BIO tag alignment |

#### Struct

```rust
pub struct OnnxNer {
    session: OrtSession,
    tokenizer: Tokenizer,  // matching tokenizer for the model
}
```

#### Functions

```rust
impl OnnxNer {
    pub fn load(model_path: &Path) -> Result<Self>
    pub fn extract(&self, text: &str) -> Result<Vec<RawEntity>>
}
```

Standalone function for simpler usage:
```rust
pub fn extract_entities_onnx(
    text: &str,
    model: &OnnxNer,
) -> Result<Vec<RawEntity>>
```

#### Algorithm

1. **Tokenize** -- Use the model's corresponding tokenizer (e.g., WordPiece for BERT-based models). Handle subword tokens correctly.
2. **Inference** -- Run the ONNX model via `ort` crate. Input: token IDs, attention mask. Output: per-token BIO tag logits.
3. **BIO tag decoding** -- Convert logits to BIO labels (B-PER, I-PER, B-ORG, I-ORG, B-GPE, I-GPE, B-EVENT, I-EVENT, B-PRODUCT, I-PRODUCT, B-WORK, I-WORK, O).
4. **Token alignment** -- Map BIO-tagged subword tokens back to source text spans. Merge subword pieces into full entity names.
5. **Entity construction** -- For each B-tag span: create `RawEntity` with:
   - `name`: text from source between start/end offsets
   - `entity_type`: mapped from BIO label ("PER" -> "person", "ORG" -> "org", "GPE" -> "place", etc.)
   - `confidence`: softmax probability of the predicted label
   - `source`: `ExtractionSource::Onnx`

#### Entity Type Mapping

| BIO Label | Entity Type |
|---|---|
| PER / PERSON | "person" |
| ORG | "org" |
| GPE | "place" |
| EVENT | "event" |
| PRODUCT | "product" |
| WORK_OF_ART | "work" |

#### Feature Gate

The ONNX integration is behind a Cargo feature flag `onnx`:
```toml
[features]
default = ["onnx"]
onnx = ["ort"]
```

When `onnx` feature is not enabled, `OnnxNer::load()` returns an error, and the orchestrator (6.6) skips ONNX extraction and falls back to rule-based only.

#### Latency Target

< 100ms per text segment (NF5). The model should be lightweight (< 50MB weights). Inference runs on CPU.

---

### 6.3 -- Tree-sitter Code Entity Extraction

**Objective:** Extract code entities (function names, class names, module names, import targets) from code content using tree-sitter AST parsing.

#### Files

| File | Type | Purpose |
|---|---|---|
| `uniko-extract/src/ner/code.rs` | New | `extract_code_entities` function |

#### Function

```rust
pub fn extract_code_entities(
    text: &str,
    language: &str,
) -> Vec<RawEntity>
```

#### Extracted Entity Types

| AST Node | Entity Type | Example |
|---|---|---|
| Function/method definition name | "function" | `process_order`, `__init__` |
| Class/struct/enum definition name | "type" | `AuthService`, `UserRole` |
| Module/package declaration | "module" | `auth`, `services.payment` |
| Import target (the thing being imported) | "dependency" | `tokio`, `react`, `numpy` |

#### Language-Specific AST Node Types

| Language | Function | Class/Struct | Module | Import Target |
|---|---|---|---|---|
| Python | `function_definition` -> `.name` | `class_definition` -> `.name` | -- (file = module) | `import_statement`, `import_from_statement` -> module name |
| Rust | `function_item` -> `.name` | `struct_item`, `enum_item` -> `.name` | `mod_item` -> `.name` | `use_declaration` -> crate/module path |
| JS/TS | `function_declaration` -> `.name`, `variable_declarator` (arrow fn) -> `.name` | `class_declaration` -> `.name` | -- | `import_statement` -> source string |
| Go | `function_declaration` -> `.name`, `method_declaration` -> `.name` | `type_spec` (struct) -> `.name` | `package_clause` -> `.name` | `import_spec` -> path string |
| Java | `method_declaration` -> `.name` | `class_declaration` -> `.name`, `interface_declaration` -> `.name` | `package_declaration` -> `.name` | `import_declaration` -> path |
| C/C++ | `function_definition` -> declarator `.name` | `struct_specifier` -> `.name`, `class_specifier` -> `.name` | -- | `preproc_include` -> path string |

#### Implementation Notes

- Reuse tree-sitter parsers initialized in Phase 5 (chunking). Share parser instances to avoid redundant initialization.
- For each extracted entity: `start` and `end` are byte offsets from the AST node's position.
- Confidence is high (0.9) since tree-sitter parsing is deterministic.
- Only extract the identifier name, not the full definition text.
- Handle anonymous functions/lambdas: skip (no name to extract).
- Handle nested definitions: extract all levels (inner functions, nested classes).

---

### 6.4 -- Entity Deduplication & Merging

**Objective:** Deduplicate extracted entities against the existing graph to prevent duplicate Entity nodes. Merge entities that refer to the same real-world thing using exact match, case-insensitive match, and embedding similarity.

#### Files

| File | Type | Purpose |
|---|---|---|
| `uniko-extract/src/ner/dedup.rs` | New | `deduplicate_entities` function, merge logic, coreference resolution |

#### Types

```rust
pub enum EntityMatch {
    /// Entity already exists in graph. Update frequency + last_seen.
    Existing(EntityId),
    /// Completely new entity. Create new node.
    New(RawEntity),
    /// Entity matches an existing one but has additional info. Merge.
    Merged(EntityId, RawEntity),
}
```

#### Function

```rust
pub async fn deduplicate_entities(
    kb: &KnowledgeBase,
    raw: Vec<RawEntity>,
) -> Result<Vec<EntityMatch>>
```

#### Deduplication Steps (executed in order, first match wins)

**Step 1: Exact match on name (Hash index lookup)**
- Query: `MATCH (e:Entity {name: $name}) RETURN e`
- If found: return `EntityMatch::Existing(e.entity_id)`
- Complexity: O(1) per entity via Hash index

**Step 2: Case-insensitive match**
- Normalize both names to lowercase and compare.
- Query all entities and compare (or maintain a secondary lowercase index).
- If found: return `EntityMatch::Merged(existing_id, raw)` -- the longer form becomes the canonical name; shorter forms become aliases.
- Example: "caroline" matches "Caroline" -> keep "Caroline" as canonical.

**Step 3: Embedding similarity**
- Compute embedding for the new entity name (using the computed embedding formula: `name + " (" + entity_type + ")"` or just `name` if no type).
- Vector search on Entity.embedding: find top-5 nearest neighbors.
- Merge threshold:
  - Same entity_type: cosine similarity > 0.85
  - Different or unknown entity_type: cosine similarity > 0.92 (stricter to prevent cross-type merges)
- If found above threshold: return `EntityMatch::Merged(existing_id, raw)`
- Merge strategy: longest form = canonical name. Shorter forms can be stored as aliases (in Entity.metadata or a separate property).

**Step 4: Type conflict guard**
- Never merge across incompatible types:
  - "person" and "org" are incompatible
  - "person" and "place" are incompatible
  - "function" and "type" are incompatible
  - "date" and any other type are incompatible
- Compatible: unknown type merges with any type (adopts the known type).
- Compatible: "concept" merges with more specific types (adopts the specific type).

**Step 5: Basic coreference (within session)**
- Pronouns ("he", "she", "they", "it", "this") -> most recently mentioned entity of matching type within the same session.
- "He/she" -> most recent "person" entity.
- "It/this" -> most recent entity of any non-person type.
- This is a heuristic with limited accuracy. Complex coreference chains require the LLM enhancement path (6.5).
- Scope: within the current session only. Cross-session coreference is not attempted locally.

#### Upsert Logic (after dedup)

For `EntityMatch::Existing(id)`:
- Increment `frequency` by mention count.
- Update `last_seen` to current timestamp.
- Do NOT change `name` or `entity_type`.

For `EntityMatch::New(raw)`:
- Create Entity node:
  - `entity_id`: UUID v7
  - `name`: raw.name
  - `entity_type`: raw.entity_type (or None if unknown)
  - `first_seen`: now
  - `last_seen`: now
  - `frequency`: 1
  - `confidence`: raw.confidence
- Compute and set embedding: `name + " (" + entity_type + ")"`.

For `EntityMatch::Merged(id, raw)`:
- If raw.name is longer than existing name: update canonical name.
- If raw.entity_type is more specific than existing: update entity_type.
- Increment frequency.
- Update last_seen.

#### MENTIONS Edge Creation

For every entity (existing, new, or merged):
- Create `source_node -[MENTIONS {count: N}]-> Entity` edge.
- `source_node` is the Message or Chunk being processed.
- `count` is the number of times the entity appears in the source text.
- If the edge already exists (same source + entity): increment count.

---

### 6.5 -- LLM Enhancement Path (Async, Optional)

**Objective:** Use an LLM to refine entity types, resolve complex coreferences, and extract entities missed by local NER. This runs asynchronously after the primary extraction and is gated by the circuit breaker.

#### Files

| File | Type | Purpose |
|---|---|---|
| `uniko-extract/src/ner/llm.rs` | New | `enhance_entities_llm` function, prompt construction, response parsing |

#### Function

```rust
pub async fn enhance_entities_llm(
    text: &str,
    existing: &[RawEntity],
    provider: &LlmProvider,
    breaker: &CircuitBreaker,
) -> Result<Vec<RawEntity>>
```

#### Behavior

1. **Check circuit breaker** -- If `breaker.is_available()` returns false, return `Ok(vec![])` immediately. No LLM call attempted.

2. **Construct prompt** -- Provide the source text and the list of already-extracted entities. Ask the LLM to:
   - Refine entity types where the local NER was ambiguous (e.g., "LGBTQ support group" should be "organization" not "concept").
   - Resolve complex coreferences (e.g., "the agency Caroline chose" -> "LGBTQ-inclusive adoption agency").
   - Extract entities missed by local NER (implicit references, paraphrases, metonymy).
   - Return results as structured JSON.

3. **Parse response** -- Extract `Vec<RawEntity>` from the LLM's JSON response. Set `source: ExtractionSource::Llm`, `confidence: 0.8` (LLM entities get slightly lower confidence than ONNX due to hallucination risk).

4. **Merge results** -- The caller merges LLM entities back through the dedup pipeline (6.4). LLM results can:
   - Update `entity_type` on existing entities (refinement).
   - Create new entities the local NER missed.
   - Create additional MENTIONS edges for coreference resolutions.

#### Prompt Template

```
Given the following text and already-extracted entities, perform these tasks:

1. REFINE: For each entity with ambiguous type (type is null or "concept"), 
   suggest a more specific type.
2. RESOLVE: Identify any coreferences (pronouns, descriptive references) 
   that refer to an existing entity. Map them to the entity name.
3. EXTRACT: Find any entities missed by the initial extraction 
   (implicit references, paraphrases).

Text:
{text}

Already extracted entities:
{entities_json}

Respond with JSON:
{
  "refined": [{"name": "...", "entity_type": "..."}],
  "coreferences": [{"reference": "...", "resolves_to": "..."}],
  "new_entities": [{"name": "...", "entity_type": "...", "start": N, "end": N}]
}
```

#### Non-Blocking Execution

The LLM enhancement is spawned as an independent tokio task. P3 (observation extraction) does not wait for LLM NER to complete. Results are merged back into the graph asynchronously:
- Update entity_type on existing Entity nodes.
- Create additional MENTIONS edges.
- Create new Entity nodes through the standard dedup path.

#### Latency Target

1-3s (NF14). This includes LLM API round-trip. The overall P2 step (local NER) completes in < 100ms; the LLM enhancement runs in the background.

---

### 6.6 -- P2 Pipeline Step Integration

**Objective:** Create the `EntityExtractionStep` that implements the `Step` trait from `uniko-pipes`, orchestrating all NER components (rules, ONNX, tree-sitter, dedup, LLM) into a single pipeline step.

#### Files

| File | Type | Purpose |
|---|---|---|
| `uniko-extract/src/ner/mod.rs` | Updated | `EntityExtractionStep` struct implementing `Step` trait |

#### Struct

```rust
pub struct EntityExtractionStep {
    onnx_ner: Option<OnnxNer>,  // None if ONNX feature disabled
}
```

#### Step Trait Implementation

```rust
impl Step for EntityExtractionStep {
    fn name(&self) -> &str { "entity_extractor" }

    fn should_run(&self, ctx: &PipelineContext) -> bool {
        // Always runs for Messages and Chunks with text content
        !ctx.content.is_empty()
    }

    async fn execute(&self, ctx: &mut PipelineContext) -> Result<StepOutcome> {
        // 1. Determine content type
        let is_code = is_code_content(&ctx.content_type);

        // 2. Extract entities (local, synchronous)
        let mut raw_entities = Vec::new();

        if is_code {
            // Code content: use tree-sitter
            let language = detect_language(&ctx.content_type);
            raw_entities.extend(extract_code_entities(&ctx.content, &language));
        } else {
            // Prose content: rule-based + ONNX
            raw_entities.extend(extract_entities_rule_based(&ctx.content));

            if let Some(ref onnx) = self.onnx_ner {
                match onnx.extract(&ctx.content) {
                    Ok(onnx_entities) => raw_entities.extend(onnx_entities),
                    Err(e) => warn!("ONNX NER failed, continuing with rules only: {}", e),
                }
            }
        }

        // 3. Merge overlapping extractions (prefer higher confidence)
        let merged = merge_overlapping(raw_entities);

        // 4. Deduplicate against existing graph
        let matches = deduplicate_entities(&ctx.kb, merged).await?;

        // 5. Upsert entities and create MENTIONS edges
        let entity_ids = upsert_entities(&ctx.kb, ctx.node_id, &matches).await?;
        ctx.extracted_entities = entity_ids;

        // 6. Queue LLM enhancement (async, non-blocking)
        if ctx.llm_breaker.is_available() {
            let text = ctx.content.clone();
            let raw_for_llm = merged.clone();  // pass existing for context
            let kb = ctx.kb.clone();
            let breaker = ctx.llm_breaker.clone();
            let node_id = ctx.node_id;
            tokio::spawn(async move {
                if let Ok(enhanced) = enhance_entities_llm(&text, &raw_for_llm, &provider, &breaker).await {
                    let _ = merge_llm_results(&kb, node_id, enhanced).await;
                }
            });
        }

        Ok(StepOutcome::Completed)
    }

    fn error_policy(&self) -> StepErrorPolicy {
        // If NER fails entirely, P3 gets fewer entities but still works
        StepErrorPolicy::Skip
    }
}
```

#### Orchestration Flow

```
Input: PipelineContext with text content
  |
  +-- Is code content?
  |     YES -> tree-sitter extraction
  |     NO  -> rule-based extraction + ONNX extraction (if available)
  |
  +-- Merge overlapping extractions (deduplicate within batch)
  |
  +-- Deduplicate against graph (exact -> case-insensitive -> embedding)
  |
  +-- Upsert: create/update Entity nodes, create MENTIONS edges
  |
  +-- Store entity_ids in PipelineContext for P3
  |
  +-- Spawn async LLM enhancement (if circuit breaker allows)
  |
Output: StepOutcome::Completed (or Skip on error)
```

#### Merging Overlapping Extractions

When rule-based and ONNX extract the same entity (overlapping spans):
- Keep the extraction with higher confidence.
- If confidence is equal, prefer ONNX (more accurate typing).
- If spans overlap but are not identical, keep both (they may be different entities).

```rust
fn merge_overlapping(mut entities: Vec<RawEntity>) -> Vec<RawEntity> {
    entities.sort_by_key(|e| (e.start, -(e.end as i64)));
    let mut result = Vec::new();
    for entity in entities {
        if let Some(last) = result.last() {
            if spans_overlap(last, &entity) {
                if entity.confidence > last.confidence {
                    *result.last_mut().unwrap() = entity;
                }
                continue;
            }
        }
        result.push(entity);
    }
    result
}
```

---

## Test Plan

### Unit Tests

| Test | File | What It Validates |
|---|---|---|
| **Rule-Based NER** | | |
| `test_rules_capitalized_proper_nouns` | `ner/rules.rs` | "Caroline Smith" extracted as entity |
| `test_rules_sentence_start_not_entity` | `ner/rules.rs` | "The meeting" at sentence start not extracted |
| `test_rules_quoted_strings` | `ner/rules.rs` | `"adoption"` extracted as concept |
| `test_rules_date_yesterday` | `ner/rules.rs` | "yesterday" extracted with type "date" |
| `test_rules_date_month_year` | `ner/rules.rs` | "March 2024" extracted with type "date" |
| `test_rules_date_iso` | `ner/rules.rs` | "2023-05-08" extracted with type "date" |
| `test_rules_date_relative` | `ner/rules.rs` | "3 days ago", "last Monday" extracted |
| `test_rules_number_with_units` | `ner/rules.rs` | "5 GB", "100ms", "$500" extracted with type "measurement" |
| `test_rules_email` | `ner/rules.rs` | "user@example.com" extracted with type "reference" |
| `test_rules_url` | `ner/rules.rs` | "https://example.com" extracted with type "reference" |
| `test_rules_preference_like` | `ner/rules.rs` | "I like Python" -> Entity("Python") |
| `test_rules_preference_prefer` | `ner/rules.rs` | "I prefer VSCode" -> Entity("VSCode") |
| `test_rules_preference_negative` | `ner/rules.rs` | "I don't like Java" -> Entity("Java") |
| `test_rules_no_false_positives` | `ner/rules.rs` | "the quick brown fox" extracts nothing |
| `test_rules_overlapping_patterns` | `ner/rules.rs` | Longer match preferred over shorter |
| `test_rules_determiners_stripped` | `ner/rules.rs` | "the LGBTQ group" -> "LGBTQ group" |
| **ONNX NER** | | |
| `test_onnx_load_model` | `ner/onnx.rs` | Model loads without error |
| `test_onnx_person` | `ner/onnx.rs` | "Caroline went to the store" -> Entity("Caroline", "person") |
| `test_onnx_organization` | `ner/onnx.rs` | "She works at Goldman Sachs" -> Entity("Goldman Sachs", "org") |
| `test_onnx_location` | `ner/onnx.rs` | "They visited New York" -> Entity("New York", "place") |
| `test_onnx_multiple_entities` | `ner/onnx.rs` | Text with 3 entities extracts all 3 |
| `test_onnx_subword_alignment` | `ner/onnx.rs` | Subword tokens correctly merged back to source spans |
| `test_onnx_confidence_scores` | `ner/onnx.rs` | Confidence is valid float in [0,1] |
| `test_onnx_empty_text` | `ner/onnx.rs` | Empty text returns empty vec |
| `test_onnx_latency` | `ner/onnx.rs` | Extraction < 100ms on 500-word text |
| `test_onnx_feature_gate_disabled` | `ner/onnx.rs` | Without `onnx` feature, load returns error |
| **Tree-sitter Code NER** | | |
| `test_code_python_function_names` | `ner/code.rs` | Python `def process_order():` -> Entity("process_order", "function") |
| `test_code_python_class_names` | `ner/code.rs` | Python `class AuthService:` -> Entity("AuthService", "type") |
| `test_code_python_imports` | `ner/code.rs` | `import numpy` -> Entity("numpy", "dependency") |
| `test_code_python_from_import` | `ner/code.rs` | `from os import path` -> Entity("os", "module"), Entity("path", "dependency") |
| `test_code_rust_function` | `ner/code.rs` | `fn process()` -> Entity("process", "function") |
| `test_code_rust_struct` | `ner/code.rs` | `struct Config {}` -> Entity("Config", "type") |
| `test_code_rust_enum` | `ner/code.rs` | `enum Status {}` -> Entity("Status", "type") |
| `test_code_rust_use` | `ner/code.rs` | `use tokio::sync::mpsc` -> Entity("tokio", "dependency") |
| `test_code_js_function` | `ner/code.rs` | JS function declaration -> Entity with "function" type |
| `test_code_js_arrow` | `ner/code.rs` | `const handler = () => {}` -> Entity("handler", "function") |
| `test_code_js_class` | `ner/code.rs` | `class Component {}` -> Entity("Component", "type") |
| `test_code_js_import` | `ner/code.rs` | `import React from 'react'` -> Entity("react", "dependency") |
| `test_code_go_function` | `ner/code.rs` | Go func declaration -> Entity with "function" type |
| `test_code_nested_functions` | `ner/code.rs` | Inner function definitions also extracted |
| `test_code_anonymous_skipped` | `ner/code.rs` | Lambda/anonymous functions not extracted (no name) |
| **Entity Deduplication** | | |
| `test_dedup_exact_match` | `ner/dedup.rs` | "Caroline" matches existing Entity("Caroline") |
| `test_dedup_case_insensitive` | `ner/dedup.rs` | "caroline" matches existing Entity("Caroline") |
| `test_dedup_case_canonical_longest` | `ner/dedup.rs` | "Caroline Smith" becomes canonical over "Caroline" |
| `test_dedup_embedding_same_type` | `ner/dedup.rs` | Cosine > 0.85 with same type -> merge |
| `test_dedup_embedding_different_type` | `ner/dedup.rs` | Cosine > 0.92 with different type -> merge |
| `test_dedup_embedding_below_threshold` | `ner/dedup.rs` | Cosine < 0.85 -> no merge (separate entities) |
| `test_dedup_type_conflict_person_org` | `ner/dedup.rs` | "person" and "org" never merge, even with high cosine |
| `test_dedup_type_conflict_person_place` | `ner/dedup.rs` | "person" and "place" never merge |
| `test_dedup_type_conflict_function_type` | `ner/dedup.rs` | "function" and "type" never merge |
| `test_dedup_unknown_type_merges` | `ner/dedup.rs` | Unknown type merges with known type, adopts known type |
| `test_dedup_concept_to_specific` | `ner/dedup.rs` | "concept" type updated to more specific type on merge |
| `test_dedup_frequency_increment` | `ner/dedup.rs` | Existing entity frequency incremented on match |
| `test_dedup_last_seen_updated` | `ner/dedup.rs` | Existing entity last_seen updated to now |
| `test_dedup_mentions_edge_created` | `ner/dedup.rs` | MENTIONS edge created with correct count |
| `test_dedup_mentions_edge_incremented` | `ner/dedup.rs` | Existing MENTIONS edge count incremented |
| `test_coreference_he_she` | `ner/dedup.rs` | "He went home" resolves to most recent person entity |
| `test_coreference_it` | `ner/dedup.rs` | "It crashed" resolves to most recent non-person entity |
| `test_coreference_session_scope` | `ner/dedup.rs` | Coreference only resolves within same session |
| **LLM Enhancement** | | |
| `test_llm_enhance_refine_type` | `ner/llm.rs` | LLM refines "LGBTQ support group" from "concept" to "organization" |
| `test_llm_enhance_coreference` | `ner/llm.rs` | LLM resolves "the agency she chose" to existing entity |
| `test_llm_enhance_missed_entities` | `ner/llm.rs` | LLM extracts entities not caught by local NER |
| `test_llm_enhance_circuit_open` | `ner/llm.rs` | Circuit breaker open -> returns empty vec immediately |
| `test_llm_enhance_parse_error` | `ner/llm.rs` | Malformed LLM response -> returns error, no panic |
| `test_llm_enhance_merge_back` | `ner/llm.rs` | LLM results merged through dedup pipeline |
| **Orchestration** | | |
| `test_step_prose_uses_rules_and_onnx` | `ner/mod.rs` | Prose text runs both rule-based and ONNX |
| `test_step_code_uses_treesitter` | `ner/mod.rs` | Code text runs tree-sitter, not ONNX |
| `test_step_empty_content_skipped` | `ner/mod.rs` | Empty content -> should_run returns false |
| `test_step_error_policy_skip` | `ner/mod.rs` | error_policy returns Skip |
| `test_step_stores_entity_ids_in_context` | `ner/mod.rs` | PipelineContext.extracted_entities populated after execution |
| `test_step_overlapping_merge` | `ner/mod.rs` | Overlapping rule + ONNX extractions merged correctly |
| `test_offline_mode` | `ner/mod.rs` | No ONNX, no LLM -> rule-based only, still produces entities |

### Integration Tests

| Test | What It Validates |
|---|---|
| `test_message_to_entities` | Ingest message -> P1 creates Message -> P2 extracts entities -> verify Entity nodes + MENTIONS edges exist |
| `test_chunk_to_entities` | Ingest artifact -> P1 creates Chunks -> P2 extracts entities from each chunk |
| `test_code_artifact_entities` | Ingest Python file -> chunks -> tree-sitter entities -> verify function/class/import Entity nodes |
| `test_entity_dedup_across_messages` | Two messages mentioning "Caroline" -> single Entity node with frequency 2 |
| `test_entity_embedding_dedup` | "New York City" and "NYC" -> single Entity (via embedding similarity) |
| `test_coreference_in_session` | "Caroline went to the store. She bought milk." -> "She" resolves to "Caroline" |
| `test_llm_enhancement_async` | Ingest message -> local NER completes -> LLM enhancement runs async -> additional entities appear |
| `test_offline_full_pipeline` | No ONNX, no LLM -> messages ingested -> entities still extracted via rules -> MENTIONS edges correct |

### Accuracy Benchmarks

| Metric | Offline (rules only) | Local (rules + ONNX) | Online (+ LLM) |
|---|---|---|---|
| Entity recall | > 40% | > 60% | > 90% |
| Entity precision | > 70% | > 80% | > 85% |
| No duplicates | 100% | 100% | 100% |
| MENTIONS edges correct | 100% | 100% | 100% |

Measured against a curated test set of 50 messages from LoCoMo conversations with manually annotated entities.

### Latency Benchmarks

| Operation | Target | Measurement |
|---|---|---|
| Rule-based NER (500 words) | < 10ms | `test_rules_*` timing |
| ONNX NER (500 words) | < 100ms | `test_onnx_latency` |
| Deduplication (10 entities, 1K existing) | < 10ms | `test_dedup_*` timing |
| Full P2 step (local only) | < 100ms | `test_step_*` timing |
| LLM enhancement | 1-3s | `test_llm_enhance_*` timing (async, non-blocking) |

---

## Documentation Plan

| Document | Content |
|---|---|
| Inline rustdoc on `EntityExtractionStep` | Orchestration flow, error policy, offline behavior |
| Inline rustdoc on `extract_entities_rule_based` | All pattern categories with examples |
| Inline rustdoc on `OnnxNer` | Model requirements, supported entity types, feature gate |
| Inline rustdoc on `extract_code_entities` | Supported languages, AST node mapping |
| Inline rustdoc on `deduplicate_entities` | Dedup cascade (exact -> case -> embedding), merge rules, type conflicts |
| Inline rustdoc on `enhance_entities_llm` | Prompt template, circuit breaker behavior, merge-back flow |
| Inline rustdoc on `RawEntity` | Field semantics, confidence ranges per source |
| Module-level doc on `ner/mod.rs` | Architecture overview, three-tier strategy, offline vs online behavior |

---

## Review Checklist

- [ ] Rule-based NER handles sentence-start capitalization (no false positives)
- [ ] Rule-based NER strips determiners ("the LGBTQ group" -> "LGBTQ group")
- [ ] ONNX model is feature-gated behind `onnx` Cargo feature
- [ ] ONNX tokenizer matches the model's expected tokenization
- [ ] BIO tag alignment correctly handles subword tokens (no off-by-one in spans)
- [ ] Tree-sitter reuses parsers from Phase 5 (no redundant initialization)
- [ ] Tree-sitter handles all 6+ languages listed in the spec
- [ ] Dedup cascade runs in correct order: exact -> case-insensitive -> embedding
- [ ] Type conflict guard prevents person/org, person/place, function/type merges
- [ ] Embedding similarity thresholds are correct: 0.85 same-type, 0.92 cross-type
- [ ] MENTIONS edge count is correct (matches actual mention count in text)
- [ ] MENTIONS edge is incremented (not duplicated) on repeat mentions
- [ ] LLM enhancement is non-blocking (P3 does not wait)
- [ ] LLM enhancement respects circuit breaker
- [ ] LLM enhancement results go through dedup pipeline (not inserted directly)
- [ ] Entity IDs are UUID v7
- [ ] Entity embedding uses `name + " (" + entity_type + ")"` formula
- [ ] error_policy is `StepErrorPolicy::Skip` (NER failure does not abort pipeline)
- [ ] PipelineContext.extracted_entities is populated after step execution
- [ ] No `unwrap()` on ONNX inference results
- [ ] No `unwrap()` on LLM response parsing
- [ ] Coreference is session-scoped (no cross-session pronoun resolution)

---

## Definition of Done

1. **All files created**: `ner/mod.rs`, `ner/types.rs`, `ner/rules.rs`, `ner/onnx.rs`, `ner/code.rs`, `ner/dedup.rs`, `ner/llm.rs` exist in `uniko-extract/src/`.
2. **Rule-based NER functional**: All 7 pattern categories extract entities correctly. Zero false positives on the "quick brown fox" test.
3. **ONNX NER functional**: Model loads, infers, and produces correct entity types for PERSON, ORG, GPE. Subword alignment verified.
4. **Tree-sitter NER functional**: Entities extracted from Python, Rust, JS/TS, Go code samples with correct types and symbol names.
5. **Dedup correctness**: Exact match, case-insensitive, and embedding similarity all work. Zero duplicate Entity nodes in the graph after processing LoCoMo session 1 (19 messages).
6. **Type conflict guard works**: person/org merge explicitly prevented and verified by test.
7. **MENTIONS edges correct**: Every entity mention in source text has a corresponding MENTIONS edge with correct count.
8. **LLM enhancement works**: Mock LLM provider test shows type refinement, coreference resolution, and new entity extraction.
9. **Offline mode works**: With `onnx` feature disabled and no LLM, rule-based NER still produces entities and MENTIONS edges.
10. **Latency met**: Full P2 step < 100ms for 500-word prose (release mode, warm store). ONNX inference < 100ms.
11. **Step integration**: `EntityExtractionStep` correctly implements `Step` trait with `should_run`, `execute`, `error_policy`.
12. **All unit tests pass**: `cargo nextest run -p uniko-extract --lib ner` passes with zero failures.
13. **All integration tests pass**: `cargo nextest run -p uniko-extract --test ner_integration` passes.
14. **Clippy clean**: `cargo clippy -p uniko-extract -D warnings` passes.
15. **Documented**: All public types and functions have rustdoc with examples.
