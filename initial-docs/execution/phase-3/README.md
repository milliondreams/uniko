# Phase 3: Episodes + Actions + Procedures + P5 + P6 + Working Memory Traversal + NL-to-Cypher + Authored Rules + Basic Access Control

## Objective

Phase 3 adds procedural memory -- the system's ability to learn reusable
playbooks from repeated patterns, organize knowledge into topic clusters,
perform hypothetical reasoning via ASSUME/ABDUCE, translate natural language
queries to Cypher, support authored Locy rules, and enforce basic access
control. This phase extends the validated MVP into a full cognitive memory
system.

## Sub-Phases

| File | Description |
|------|-------------|
| `sub-01-procedural-memory.md` | Pipeline 5 (P5) procedure promotion and Pipeline 6 (P6) topic detection: learn reusable playbooks from experience, organize knowledge into coherent clusters |
| `sub-02-hypothetical-reasoning.md` | ASSUME/ABDUCE hypothetical reasoning, NL-to-Cypher query translation, and advanced reasoning APIs for agents |

## Key Milestone

> Procedural memory; access control; benchmark validation

## Prerequisites

Phase 2 complete -- MVP validated via LoCoMo, consolidation pipeline operational, recall cascade functional, public API shippable.

## Definition of Done

- P5 promotes recurring action patterns into Procedure nodes with step sequences and trigger conditions.
- P6 detects topic clusters and creates Topic nodes linking related Entities, Facts, and Observations.
- ASSUME/ABDUCE hypothetical reasoning operates correctly with nested what-if scenarios.
- NL-to-Cypher translates natural language queries into valid Cypher for direct graph querying.
- Authored Locy rules can be defined and execute within the rule engine.
- Basic access control restricts memory access by agent/user/scope.
- Working memory traversal navigates the graph contextually during agent sessions.
- All sub-phase test suites pass.
