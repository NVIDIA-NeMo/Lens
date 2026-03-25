# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Megatron-LM specific span groups."""

from typing import ClassVar, Final

from nemo.lens.groups import SpanGroup


class MegatronSpanGroup(SpanGroup):
    """Span groups for Megatron-LM instrumentation.

    Extends the shared groups with Megatron-specific fine-grained groups.
    """

    # ------------------------------------------------------------------ #
    # Fine-grained (included in "all" only)
    # ------------------------------------------------------------------ #

    MICROBATCH = "microbatch"
    """Per-microbatch forward/backward spans (highest overhead)."""

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    INFERENCE = "inference"
    """Inference server request spans."""

    # ------------------------------------------------------------------ #
    # All groups and presets
    # ------------------------------------------------------------------ #

    ALL_GROUPS: Final[frozenset] = SpanGroup.ALL_GROUPS | frozenset([
        MICROBATCH, INFERENCE,
    ])

    _PRESETS: ClassVar[dict] = {
        "default": frozenset([
            SpanGroup.JOB, SpanGroup.CHECKPOINT, SpanGroup.EVALUATE, INFERENCE,
        ]),
        "per_step": frozenset([
            SpanGroup.JOB, SpanGroup.CHECKPOINT, SpanGroup.EVALUATE,
            SpanGroup.MODEL_INIT, SpanGroup.LOAD_CHECKPOINT,
            SpanGroup.STEP, SpanGroup.FORWARD_BACKWARD, SpanGroup.OPTIMIZER,
            INFERENCE,
        ]),
        "all": ALL_GROUPS,
    }
