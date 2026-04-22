# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Rank-aware sampling for distributed training.

Provides a custom OTel Sampler that makes sampling decisions based on
the process rank, allowing controlled telemetry overhead in large-scale
training jobs.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import TYPE_CHECKING

from opentelemetry import trace
from opentelemetry.context import Context

if TYPE_CHECKING:
    from opentelemetry.util.types import Attributes


class RankAwareSampler:
    """OTel-compatible Sampler that filters spans based on rank.

    When ``should_sample`` is called, the decision is based on a
    deterministic hash of the rank — not per-span. This means either
    all spans on a rank are sampled, or none are.

    Implements the OTel ``Sampler`` interface so it can be passed
    directly to ``TracerProvider(sampler=...)``.

    Args:
        rank: Current process rank.
        world_size: Total number of ranks.
        sample_rate: Fraction of ranks to sample (0.0-1.0).
    """

    def __init__(self, rank: int, world_size: int, sample_rate: float = 1.0) -> None:
        self._rank = rank
        self._world_size = world_size
        self._sample_rate = sample_rate
        self._should_sample = self._compute_should_sample()

    def _compute_should_sample(self) -> bool:
        if self._sample_rate >= 1.0:
            return True
        if self._sample_rate <= 0.0:
            return False
        h = hashlib.md5(str(self._rank).encode()).hexdigest()
        return (int(h, 16) % 10000) / 10000.0 < self._sample_rate

    def should_sample(
        self,
        parent_context: Context | None = None,
        trace_id: int = 0,
        name: str = "",
        kind: trace.SpanKind | None = None,
        attributes: Attributes | None = None,
        links: Sequence[trace.Link] | None = None,
    ):
        """Return a SamplingResult based on rank sampling decision.

        Requires the OTel SDK. Falls back to returning a bool if the SDK
        is not installed (preserving the original API for non-SDK callers).
        """
        try:
            from opentelemetry.sdk.trace.sampling import Decision, SamplingResult

            if self._should_sample:
                return SamplingResult(
                    decision=Decision.RECORD_AND_SAMPLE,
                    attributes=dict(attributes) if attributes else None,
                )
            return SamplingResult(decision=Decision.DROP)
        except ImportError:
            return self._should_sample

    def get_description(self) -> str:
        """Return a description of this sampler for OTel diagnostics."""
        return f"RankAwareSampler(rank={self._rank}, rate={self._sample_rate})"
