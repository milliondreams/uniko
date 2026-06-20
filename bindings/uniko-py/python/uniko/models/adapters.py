# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 Dragonscale Industries Inc.
"""Convenience adapters: spec in → native call → model out, in one call.

These free functions take a *native* handle (``Goals``/``Session``/``Agent``)
plus a spec, do the full round-trip, and return a Pydantic model. They are the
thin building blocks the typed wrapper handles in :mod:`uniko.models.typed` reuse.
Each has a blocking ``*_sync`` twin.
"""

from __future__ import annotations

from typing import Any

from .inputs import IngestSourceSpec, ScopeSpec, TurnSpec
from .outputs import Answer, ContextBundle, IngestOutcome, ObserveResult


def _as_turn(turn: Any) -> Any:
    """Accept a :class:`TurnSpec` or an already-native ``Turn``."""
    return turn.to_native() if isinstance(turn, TurnSpec) else turn


def _as_source(source: Any) -> Any:
    """Accept an :class:`IngestSourceSpec` or an already-native ``IngestSource``."""
    return source.to_native() if isinstance(source, IngestSourceSpec) else source


def _as_scope(scope: Any) -> Any:
    """Accept a :class:`ScopeSpec` or an already-native ``Scope``."""
    return scope.to_native() if isinstance(scope, ScopeSpec) else scope


# ── Goals ──────────────────────────────────────────────────────────────────


async def create_goal(goals: Any, spec: Any) -> int:
    return await goals.create(spec.title, **spec.to_create_kwargs())


def create_goal_sync(goals: Any, spec: Any) -> int:
    return goals.create_sync(spec.title, **spec.to_create_kwargs())


async def create_task(goals: Any, spec: Any) -> int:
    return await goals.create_task(spec.title, **spec.to_create_task_kwargs())


def create_task_sync(goals: Any, spec: Any) -> int:
    return goals.create_task_sync(spec.title, **spec.to_create_task_kwargs())


# ── Session ────────────────────────────────────────────────────────────────


async def observe(session: Any, turn: Any) -> ObserveResult:
    return ObserveResult.from_native(await session.observe(_as_turn(turn)))


def observe_sync(session: Any, turn: Any) -> ObserveResult:
    return ObserveResult.from_native(session.observe_sync(_as_turn(turn)))


async def ingest(session: Any, source: Any) -> IngestOutcome:
    return IngestOutcome.from_native(await session.ingest(_as_source(source)))


def ingest_sync(session: Any, source: Any) -> IngestOutcome:
    return IngestOutcome.from_native(session.ingest_sync(_as_source(source)))


# ── Agent (scoped recall/answer) ───────────────────────────────────────────


async def recall_in(agent: Any, query: str, scope: Any) -> ContextBundle:
    return ContextBundle.from_native(await agent.recall_in(query, _as_scope(scope)))


def recall_in_sync(agent: Any, query: str, scope: Any) -> ContextBundle:
    return ContextBundle.from_native(agent.recall_in_sync(query, _as_scope(scope)))


async def answer_in(agent: Any, question: str, scope: Any) -> Answer:
    return Answer.from_native(await agent.answer_in(question, _as_scope(scope)))


def answer_in_sync(agent: Any, question: str, scope: Any) -> Answer:
    return Answer.from_native(agent.answer_in_sync(question, _as_scope(scope)))


__all__ = [
    "create_goal",
    "create_goal_sync",
    "create_task",
    "create_task_sync",
    "observe",
    "observe_sync",
    "ingest",
    "ingest_sync",
    "recall_in",
    "recall_in_sync",
    "answer_in",
    "answer_in_sync",
]
