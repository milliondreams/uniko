# bench-configs

JSON profiles for `uniko-bench` runs.  Each file is a complete spec
of every model / device / recall / cost knob.  Run with:

```bash
./crates/uniko-bench/run.sh \
    --bench-config crates/uniko-bench/bench-configs/<profile>.json \
    --data data/locomo10.json \
    --conversations conv-26 \
    --output data/results.json
```

CLI carries only what changes per invocation: data file, output path,
conversation/category filters, KB reuse.  All model selection lives
in the JSON.

## Shipped profiles

| File | Generator | Judge | Embedder | Notes |
|---|---|---|---|---|
| `locomo-bge-openai.json` | gpt-4o-mini | gpt-4o-mini | bge-small (384d ONNX-CUDA) | Canonical baseline.  Last measured: 0.849 judge, $0.083/conv. |
| `locomo-bge-gemini31.json` | gemini-3.1-flash-lite | gemini-3.1-pro-preview | bge-small | All-Gemini run via Vertex global endpoint.  Requires `VERTEXAI_PROJECT`/`VERTEXAI_LOCATION` env. |
| `locomo-embeddinggemma-openai.json` | gpt-4o-mini | gpt-4o-mini | embeddinggemma-300m (ONNX-CUDA) | A/B vs bge-small for the embedder swap.  768d vectors — fresh ingest required. |

## Schema reference

See the module-level doc comment on
`crates/uniko-bench/src/bench_config.rs::BenchConfig` for the full
serde schema.  Key fields:

- `models.gen` / `models.judge` — required `LlmAlias` objects: `alias`, `model_id`, `provider`, optional `base_url`, optional `use_default_options`.
- `models.extract_triples` — optional; set to an `LlmAlias` to use LLM triple extraction during P4 consolidation.
- `models.embedder` — either `{ "preset": "<name>" }` or `{ "inline": <EmbeddingConfig> }`.  Presets: `bge-small`, `bge-large`, `nomic`, `minilm`, `embeddinggemma`, `embeddinggemma-mistralrs`.
- `models.reranker` — full `RerankerConfig` shape (`enabled`, `model_id`, `style`, `top_n`, `apply_sigmoid`, optional `execution_providers`).
- `models.nlp` — `model_id`, `artifact`, `max_batch_size`, optional `execution_providers`.  Defaults to the xsmall deberta INT8 cascade.
- `recall.*` — phase strategies, recall_limit, variants, consolidation toggles.
- `cost.pricing_csv` — path to the per-model price table.  `cost.no_events` skips the JSONL event writer.

## Per-component CPU/GPU control

Every component (`models.reranker`, `models.nlp`, and the inline
embedder) accepts an `execution_providers` array, e.g.:

```json
"reranker": {
  "enabled": true,
  "model_id": "cross-encoder/ms-marco-MiniLM-L-6-v2",
  "top_n": 50,
  "apply_sigmoid": false,
  "execution_providers": ["cpu"]
}
```

Leave the field `null` to inherit the build-time default
(`["cuda","cpu"]` on `gpu-cuda` builds, `["cpu"]` otherwise).  The
NLP cascade additionally inherits the embedder's setting when its
own `execution_providers` is `null`.
