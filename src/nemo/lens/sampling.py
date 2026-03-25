# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Rank-aware sampling for distributed training.

Provides a custom OTel Sampler that makes sampling decisions based on
the process rank, allowing controlled telemetry overhead in large-scale
training jobs.
"""

from __future__ import annotations

import hashlib
from typing import Optional, Sequence

from opentelemetry import trace
from opentelemetry.context import Context


class RankAwareSampler:
    """Sampler that filters spans based on rank.

    This is a lightweight sampler that can be used alongside the export
    strategy. It's useful when you want all ranks to create spans (for
    local debugging) but only export a subset.

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

    @property
    def should_sample(self) -> bool:
        """Whether this rank should produce spans."""
        return self._should_sample

    def should_sample_span(
        self,
        parent_context: Optional[Context] = None,
        trace_id: int = 0,
        name: str = "",
        kind: Optional[trace.SpanKind] = None,
        attributes: Optional[dict] = None,
        links: Optional[Sequence[trace.Link]] = None,
    ) -> bool:
        """Determine whether a specific span should be sampled."""
        return self._should_sample
