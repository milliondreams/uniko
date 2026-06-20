# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 Dragonscale Industries Inc.
"""Pydantic model for the subset of configuration ``Uniko.config()`` exposes."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class UnikoConfigModel(BaseModel):
    """The most relevant config values, as returned by ``Uniko.config()``.

    ``extra="ignore"`` (not ``forbid``) is deliberate: ``config()`` is the one
    surface where the native side may legitimately add keys over time, and the
    overlay should not break when it does.
    """

    model_config = ConfigDict(extra="ignore")

    embedding_model_id: str
    embedding_dimensions: int
    recall_limit: int
    recall_token_budget: int
    recall_min_score: float

    @classmethod
    def from_native(cls, uni: Any) -> "UnikoConfigModel":
        """Build from a native ``Uniko`` (or ``TypedUniko``) handle via ``config()``."""
        return cls.model_validate(uni.config())


__all__ = ["UnikoConfigModel"]
