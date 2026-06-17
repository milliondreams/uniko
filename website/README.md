# uniko Documentation

Source for the [uniko](https://github.com/rustic-ai/uniko) documentation website,
built with [Zensical](https://zensical.org/) (the static site generator from the makers of
Material for MkDocs).

## Develop

```bash
# Install dependencies
poetry install

# Local preview with hot reload
poetry run zensical serve

# Build the static site into ./site
poetry run zensical build
```

## Structure

```
docs/
  index.md                 # Home
  why-uniko.md             # Positioning
  getting-started/         # Overview, Installation, Quick Start
  concepts/                # Architecture, Memory Model, Data Model, Facts & Drift, Visibility
  pipelines/               # Ingest, Consolidation, Recall
  guides/                  # Agent Tools, Reasoning with Locy, Configuration
  reference/               # API, Schema
  benchmarks/              # LoCoMo, LongMemEval, competitive comparison, perf journey
  assets/stylesheets/      # theme.css (palette + layout)
```

Theme configuration lives in `mkdocs.yml`; visual styling in
`docs/assets/stylesheets/theme.css`.
