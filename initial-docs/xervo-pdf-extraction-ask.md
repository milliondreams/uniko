# Xervo Requirement Spec / Ask — Tiered PDF Document Extraction

**Status:** Draft for discussion (uniko → xervo)
**Author:** uniko team
**Date:** 2026-06-16
**Consumers:** uniko-extract (PDF ingest into the knowledge graph); any future doc-AI consumer

---

## 1. The ask in one sentence

Give xervo a **single tiered document-extraction API** that takes a PDF (or its
pages) and returns structured text, where the caller asks for an **effort
level** — `plain-text` → `ppocr` → `olmocr` — either **pinned**, **bounded**
(min/max), or **auto**, with xervo escalating per-page only as far as needed.

```
DocumentExtract(pdf, { level: Auto { max: OlmOcr }, want: Structure })
  → per-page blocks, each tagged with the level/model that produced it,
    a confidence, and provenance back to the source page region.
```

## 2. Motivation

uniko ingests PDFs into a knowledge graph used for long-term memory/recall.
Today the path is **plain-text only** (pure-Rust `pdf-extract` behind a
`PdfTextExtractor` trait): born-digital PDFs work; **scanned/image PDFs return
empty and persist with zero chunks.** We want to close that gap *and* unlock
structured extraction (tables/formulas/reading-order), but with two hard
constraints:

- **Reliability first.** A generative model that silently alters a number,
  reading order, or table cell writes a confident falsehood into the graph that
  later poisons recall, with no human in the loop. Cheaper deterministic levels
  must be preferred, and every block must be traceable to its producer +
  confidence so downstream can gate trust.
- **Cost-proportional.** Most pages are easy (text layer present). We must not
  pay 7B-VLM cost on a page a 34M CNN — or a zero-cost native parse — handles.

A tiered, escalate-only-when-needed API serves both.

## 3. The level ladder

| Level | Method | Runtime | Input | Output | Hallucination | Cost |
|---|---|---|---|---|---|---|
| **`plain-text`** | Native digital text extraction (`pdf-extract`/`lopdf`) | none (pure Rust) | PDF text layer | text + char coords | none | ~free |
| **`ppocr`** | PP-OCRv6 pipeline (detect → recognize, CTC) | `ort` (ONNX) | **rasterized page image** | text + line boxes | none (reads) | low (34.5M, CPU-viable) |
| **`olmocr`** | olmOCR-2 doc-VLM (Qwen2.5-VL family) | `mistral.rs` vision | **rasterized page image** | structured markdown: tables, formulas, reading order | **yes** (generates) | high (7B, GPU) |

**Semantics of a "level":** the *minimum* capability needed to get a usable
result for that page. `plain-text` answers "what text is digitally embedded";
`ppocr` answers "what text is visually present" (handles scans); `olmocr`
answers "what is the document *structure*" (tables/formulas/layout). Higher
levels are strictly more capable but more expensive and (for `olmocr`) less
deterministic.

> ⚠️ **The ladder mixes two modalities.** `plain-text` reads the PDF's digital
> text — no image. `ppocr`/`olmocr` read a **rasterized page image**. The jump
> from L1→L2 is therefore not just "more effort" — it requires
> **PDF-page → pixel rasterization** (see §7.1, the gating dependency).

## 4. Request API (proposed)

```rust
struct DocExtractRequest {
    /// One page image OR the whole PDF (see §7.2 ownership decision).
    input: DocInput,                 // PdfBytes | PageImages | { native_text, page_images }
    level: LevelPolicy,
    want: OutputWant,                // Text | Structure  (does the caller need tables/markdown?)
    budget: Option<Budget>,          // max GPU pages, latency ceiling, etc.
}

enum LevelPolicy {
    Fixed(Level),                    // always use exactly this level
    Ceiling(Level),                  // auto, but never exceed this (e.g. cap at Ppocr → no VLM, no hallucination)
    Auto { min: Level, max: Level }, // escalate within [min, max] per page
}

enum Level { PlainText, Ppocr, Olmocr }
```

- **Pinned** (`Fixed`): caller knows what it wants.
- **Bounded** (`Ceiling`/`Auto{max}`): the reliability knob — a memory-critical
  caller sets `Ceiling(Ppocr)` to forbid generative extraction entirely.
- **Auto**: xervo escalates per page using §5 signals.

## 5. Auto-routing intelligence (the "decide which levels are needed" part)

Escalation is **per page**, cheapest-signal-first:

**L1 → L2 trigger (cheap, ~free — native parser already knows):**
- No text layer on the page, **or**
- Text coverage below a threshold (char count vs page area; large raster/vector
  image regions with little extractable text → likely scanned), **or**
- Garbled text signal (CID-keyed fonts without `ToUnicode` → mojibake risk).

**L2 → L3 trigger (not free — needs a layout signal):**
- Caller set `want: Structure` and the page has table/figure/multi-column
  layout, **or**
- A layout signal says the page is structurally complex.
- **v1:** cheap heuristic (image-area ratio, detected column count, ruling-line
  density from the native parse / PP-OCR boxes).
- **v2:** a real layout classifier (PP-DocLayout-class model). *Honest: robust
  L2→L3 auto has a cost floor — detecting "this is a table page" is itself
  model work.*

**Escalation is also driven by failure**, not just pre-detection (ParseFixer
pattern): if L1 yields empty/garbage → L2; if `want: Structure` and L2 gives
flat text where layout was expected → L3. Each escalation is logged so callers
see why a page cost more.

## 6. Result schema (unified, provenance-bearing)

One result type across all levels — richness varies, shape doesn't:

```rust
struct DocExtractResult {
    pages: Vec<PageResult>,
}
struct PageResult {
    page_number: u32,
    blocks: Vec<DocBlock>,           // text/heading/table/figure/formula/caption (xervo already has DocBlockKind)
    plain_markdown: String,
    produced_by: Level,              // which level actually ran for this page
    escalations: Vec<Escalation>,    // audit trail: why we climbed the ladder
}
struct DocBlock {
    kind: DocBlockKind,
    content: String,
    bbox: Option<[f32; 4]>,          // source page region — provenance
    reading_order: u32,
    confidence: f32,                 // REQUIRED — downstream gates trust on this
    produced_by: Level,              // per-block, since a page may mix levels later (regions)
}
```

**Reliability requirement:** every block carries `produced_by` + `confidence` +
`bbox`. A consumer (uniko) must be able to say "don't write olmocr-generated
numeric content into the graph below confidence X without verification."

## 7. Hard dependencies & open decisions

### 7.1 Rasterization — the gating dependency (MUST resolve first)

L2 and L3 require **PDF page → image**. There is **no production-ready
pure-Rust PDF rasterizer**; pdfium/mupdf/poppler are C/C++ FFI. This blocks the
entire image-based half of the ask regardless of model readiness.

Options (pick one — this is the real long pole):
- **(R1)** Find/adopt a pure-Rust rasterizer (e.g. evaluate current state of
  `pdf`/`hayro`/`vello`-based rendering) — uncertain maturity, must benchmark
  fidelity on tables/small fonts.
- **(R2)** Accept an **optional, feature-gated FFI rasterizer** (pdfium) as the
  one sanctioned non-Rust component, sandboxed — pragmatic, breaks the
  pure-Rust rule narrowly and explicitly.
- **(R3)** Make rasterization the **caller's problem**: xervo's API takes
  `PageImages`, uniko owns producing them (and still faces R1/R2).

**Recommendation:** decide R1-vs-R2 up front with a fidelity benchmark; it
determines whether levels 2-3 are reachable at all.

### 7.2 Where does the pipeline live? (xervo scope)

- **Option A — xervo owns end-to-end.** API takes `PdfBytes`; xervo does native
  text extraction (new pure-Rust dep), rasterization (§7.1), routing, and all
  model levels. *Pro:* one clean call, solved once for every consumer. *Con:*
  xervo (an inference layer) grows non-inference deps (PDF parse + rasterizer).
- **Option B — xervo owns models + routing; caller supplies pixels + native
  text.** API takes `{ native_text, page_images }`. *Pro:* xervo stays an
  inference engine. *Con:* every consumer re-solves rasterization + native
  extraction; routing intelligence is split across the boundary.

**Recommendation:** lean **A** for the clean abstraction *if* §7.1 lands a
rasterizer xervo can own; otherwise **B**. This is the central design fork.

### 7.3 Model availability (verified 2026-06-16)

- **`plain-text`:** pure-Rust, exists today in uniko (`pdf-extract`); trivially
  portable.
- **`ppocr` (PP-OCRv6):** **official ONNX ships** (tiny/small/medium, det+rec
  as separate models) → runs on xervo's `ort`. xervo already has the
  **recognition** half (`OcrModel`/`local_onnx/ocr.rs`, CTC greedy decode) but
  **not detection** — needs a DBNet detector + box-merge + reading-order
  post-processing (the substantial, accuracy-sensitive work; `oar-ocr`
  implements all of it on `ort` and is a reference/adopt candidate).
- **`olmocr` (olmOCR-2):** **runs on xervo's `mistral.rs` vision pipeline
  today** as a `Qwen2_5VL` model (clean Qwen2.5-VL fine-tune; xervo's
  `provider/mistralrs.rs::load_vision_generator` loads arbitrary `model_id`,
  exposes `GeneratorModel` + `ContentBlock::Image`). xervo **already ships** the
  olmOCR markdown→blocks parser (`local_onnx/document_extract.rs::
  parse_olmocr_markdown`, tested) and the autoregressive decoder helper. The
  existing `DocumentExtractionModel::extract()` stub (returns `Unavailable`,
  was waiting on ONNX exports that are **"not planned"** upstream) should be
  **revived onto the `mistral.rs` vision path**, not ONNX. One page image per
  request (avoids the known multi-image KV-cache bug). *Note:* model choice is
  constrained by mistral.rs arch support — olmOCR-2 (Qwen2.5-VL) works;
  PaddleOCR-VL / dots.ocr ship custom vision towers and would need per-model
  Candle code, so they are **out of scope** for this ask.

## 8. What xervo already has vs. what's new (so this is incremental)

**Already present (verified in 0.14.0):**
- `OcrModel` + ONNX CTC recognition (`local_onnx/ocr.rs`).
- `DocumentExtractionModel` trait + DocTags/MinerU/olmOCR markdown parsers +
  autoreg greedy decoder (`local_onnx/document_extract.rs`).
- `mistral.rs` vision generation, arbitrary model load, image inputs
  (`provider/mistralrs.rs`).
- Image preprocessing (`local_onnx/image.rs`) — *note square-resize TODO; CRNN
  wants aspect-preserving.*
- `RawTensorModel` escape hatch (could host a DBNet detector).
- `DocBlock`/`DocBlockKind`/`DocExtractResult` types.

**New work this ask requires:**
1. **Rasterization** (§7.1) — the long pole.
2. **PP-OCRv6 detection** stage + box-merge + reading-order (or adopt `oar-ocr`).
3. **Wire `document_extract` to the `mistral.rs` vision path** for olmOCR-2
   (small — generator + existing parser).
4. **The tiered router** — level policy, per-page escalation signals (§5),
   unified result with provenance (§6).
5. **A native-text level** inside the API (if Option A) — pure-Rust parse.
6. Generalize image preprocessing for CRNN aspect ratio.

## 9. Non-goals (v1)

- PaddleOCR-VL / dots.ocr / MinerU2.5 (custom vision towers; need per-model
  Candle ports — separate ask).
- Chart/figure → SVG/code reconstruction (dots.mocr-style).
- Region-level (sub-page) escalation — v1 is per-page; region granularity is a
  later refinement of §6's per-block `produced_by`.
- Cross-page table/section reconstruction.
- Handwriting-specialized models.

## 10. Suggested phasing

1. **Resolve §7.1 (rasterizer) + §7.2 (scope)** — these gate everything.
2. **L1 + L2 path** (plain-text + PP-OCRv6) — closes the concrete scanned-PDF
   gap, fully deterministic/reliability-safe. Ship the router with `[PlainText,
   Ppocr]` range.
3. **L3 path** (olmOCR-2 via mistral.rs) + `want: Structure` routing +
   hallucination/confidence gating. Extend router to `Olmocr`.
4. **Layout-model upgrade** for robust L2→L3 auto (replace v1 heuristic).
```
