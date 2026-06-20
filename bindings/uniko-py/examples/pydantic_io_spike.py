# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 Dragonscale Industries Inc.
"""Spike: evaluate a Pydantic IO layer for the uniko Python SDK.

Run::

    cd bindings/uniko-py
    uv run --group dev python examples/pydantic_io_spike.py

Parts 1 & 2 need only ``pydantic`` (no native build). Part 3 exercises the
real ``_uniko`` extension if it is importable; it is skipped otherwise so the
ergonomics of the schema/validation story are visible without a full build.
"""

from __future__ import annotations

import asyncio
import datetime
import json

from pydantic import ValidationError

from uniko.models import (
    ContextBundle as ContextBundleModel,
    GoalSpec,
    RecallItem as RecallItemModel,
    RecallSource as RecallSourceModel,
)


def rule(title: str) -> None:
    print(f"\n{'─' * 4} {title} {'─' * (60 - len(title))}")


# ── Part 1: input validation (the "catch it in Python" story) ──────────────


def demo_input_validation() -> None:
    rule("1. Input validation: GoalSpec")

    good = GoalSpec(
        title="Ship Pydantic IO",
        status="achieved",  # free-form on purpose — facade derives phase
        metrics={"target_latency_ms": 50},
        deadline=datetime.datetime(2026, 7, 1, tzinfo=datetime.timezone.utc),
    )
    print("valid spec  ->", good.to_create_kwargs())

    # Empty title is rejected (min_length=1).
    try:
        GoalSpec(title="")
    except ValidationError as e:
        print("empty title -> rejected:", e.errors()[0]["msg"])

    # A typo'd keyword is caught by extra='forbid' instead of being dropped.
    try:
        GoalSpec(title="x", descriptoin="oops")  # ty: ignore[unknown-argument]
    except ValidationError as e:
        print(
            "typo kwarg  -> rejected:",
            e.errors()[0]["msg"],
            f"({e.errors()[0]['loc']})",
        )


# ── Part 2: serializable outputs (the FastAPI / interop story) ─────────────


class _FakeSource:
    kind, message_id, artifact_id, chunk_id = "message", "m-1", None, None


class _FakeItem:
    node_id, kind, score, content = 7, "observation", 0.91, "User prefers dark mode."
    sources = [_FakeSource()]


class _FakeBundle:
    """Duck-type of a native ContextBundle — same attribute surface."""

    items = [_FakeItem()]
    total_tokens, phase1_only, phase2_only, coverage = 12, True, False, 0.5


def demo_serializable_output() -> None:
    rule("2. Typed output + serialization")

    # from_attributes reads the (fake) native object directly; nested lists recurse.
    bundle = ContextBundleModel.from_native(_FakeBundle())
    print("typed access ->", bundle.items[0].kind, bundle.items[0].score)
    print("model_dump   ->", bundle.model_dump())
    print("as JSON      ->", bundle.model_dump_json())

    # The JSON Schema is what FastAPI emits for response models / docs.
    schema = ContextBundleModel.model_json_schema()
    print("json schema  -> top-level keys:", sorted(schema["properties"]))
    print("              defs:", sorted(schema.get("$defs", {})))
    _ = (RecallItemModel, RecallSourceModel)  # referenced for clarity


# ── Part 3: live round-trip against the native extension (optional) ─────────


async def demo_live_roundtrip() -> None:
    rule("3. Live round-trip (needs built _uniko)")
    try:
        import uniko
    except ImportError as e:
        print("skipped: native extension not importable ->", e)
        return

    uni = await uniko.Uniko.in_memory()
    agent = uni.agent("spike")

    # observe → flush → recall, then wrap the *real* native bundle in the model.
    # (Goal creation is shown structurally in Part 1; live creation needs the
    # agent's participant registered first, which is out of scope for this spike.)
    # Non-streaming in_memory(): observe completes inline, no flush needed.
    session = agent.session("s-1")
    await session.observe(uniko.Turn("alice", "I love hiking in the Dolomites."))

    native_bundle = await agent.recall("hiking")
    bundle = ContextBundleModel.from_native(native_bundle)
    print(
        "native bundle ->", type(native_bundle).__module__, type(native_bundle).__name__
    )
    print("recall items  ->", len(bundle))
    print(
        "round-trip JSON coverage ->", json.loads(bundle.model_dump_json())["coverage"]
    )

    # shutdown() needs every Agent/Session handle dropped first.
    del session, agent
    await uni.shutdown()


def main() -> None:
    demo_input_validation()
    demo_serializable_output()
    asyncio.run(demo_live_roundtrip())


if __name__ == "__main__":
    main()
