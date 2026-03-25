# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""NeMo-Gym specific span groups."""

from typing import ClassVar, Final

from nemo.lens.groups import SpanGroup


class GymSpanGroup(SpanGroup):
    """Span groups for NeMo-Gym instrumentation."""

    # ------------------------------------------------------------------ #
    # Gym-specific groups
    # ------------------------------------------------------------------ #

    SERVER = "server"
    """FastAPI server request spans."""

    ROLLOUT_COLLECTION = "rollout_collection"
    """Rollout collection orchestration spans."""

    VERIFY = "verify"
    """Reward verification endpoint spans."""

    AGGREGATE_METRICS = "aggregate_metrics"
    """Metric aggregation spans."""

    # ------------------------------------------------------------------ #
    # All groups and presets
    # ------------------------------------------------------------------ #

    ALL_GROUPS: Final[frozenset] = SpanGroup.ALL_GROUPS | frozenset([
        SERVER, ROLLOUT_COLLECTION, VERIFY, AGGREGATE_METRICS,
    ])

    _PRESETS: ClassVar[dict] = {
        "default": frozenset([
            SpanGroup.JOB, SpanGroup.CHECKPOINT, SpanGroup.EVALUATE,
            SERVER,
        ]),
        "per_step": frozenset([
            SpanGroup.JOB, SpanGroup.CHECKPOINT, SpanGroup.EVALUATE,
            SERVER, ROLLOUT_COLLECTION, VERIFY, AGGREGATE_METRICS,
        ]),
        "all": ALL_GROUPS,
    }
