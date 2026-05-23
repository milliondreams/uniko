# Phase 4: Full Benchmark Harness + LoCoMo/LongMemEval/MemoryAgentBench/BEAM/Evo-Memory + Contrastive Retrieval + MCP + Python Binding

## Objective

Phase 4 exposes the system via MCP (Model Context Protocol) for universal agent
compatibility, then builds the full benchmark harness to produce published
performance numbers across five benchmark suites. Results validate uniko's
competitive position against existing cognitive memory systems and stress-test
every architectural decision from P1 through P8.

## Sub-Phases

| File | Description |
|------|-------------|
| `sub-01-mcp-server.md` | MCP server in uniko-mcp: expose all agent tools via JSON-RPC so any MCP-compatible agent can use uniko as its cognitive memory backend |
| `sub-02-benchmark-harness.md` | Full benchmark infrastructure running LoCoMo, LongMemEval, MemoryAgentBench, BEAM, and Evo-Memory with contrastive retrieval validation |

## Key Milestone

> Published benchmark numbers; prove LoCoMo uplift

## Prerequisites

Phase 3 complete -- procedural memory (P5/P6), hypothetical reasoning, NL-to-Cypher, access control all operational.

## Definition of Done

- MCP server exposes all agent tools via the standard Model Context Protocol.
- Any MCP-compatible agent can discover and invoke uniko tools through JSON-RPC.
- Benchmark harness runs all 5 benchmark suites (LoCoMo, LongMemEval, MemoryAgentBench, BEAM, Evo-Memory).
- Published benchmark numbers demonstrate competitive or superior performance against Mem0, Graphiti, Letta, Zep, and MemGPT.
- Contrastive retrieval validation confirms the recall cascade outperforms naive approaches.
- LoCoMo uplift is proven with measurable improvement over Phase 2 baseline.
- All sub-phase test suites pass.
