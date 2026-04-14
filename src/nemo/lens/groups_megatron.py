# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Megatron-LM specific span groups."""

from typing import ClassVar, Final

from nemo.lens.groups import SpanGroup


class MegatronSpanGroup(SpanGroup):
    """Span groups for Megatron-LM instrumentation.

    Extends the shared groups with Megatron-specific fine-grained groups.
    """

    # ------------------------------------------------------------------ #
    # Fine-grained (included in "per_step" or "all")
    # ------------------------------------------------------------------ #

    MICROBATCH = "microbatch"
    """Per-microbatch forward/backward spans."""

    LAYER = "layer"
    """Per-transformer-layer forward (attention + MLP breakdown)."""

    COMMUNICATION = "communication"
    """P2P send/recv and gradient AllReduce/ReduceScatter."""

    ACTIVATION_OFFLOAD = "activation_offload"
    """GPU<->CPU activation offload/reload spans."""

    DATA_LOADING = "data_loading"
    """Data loading and batch preparation."""

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    INFERENCE = "inference"
    """Inference server request spans."""

    # ------------------------------------------------------------------ #
    # All groups and presets
    # ------------------------------------------------------------------ #

    ALL_GROUPS: Final[frozenset] = SpanGroup.ALL_GROUPS | frozenset(
        [
            MICROBATCH,
            LAYER,
            COMMUNICATION,
            ACTIVATION_OFFLOAD,
            DATA_LOADING,
            INFERENCE,
        ]
    )

    _PRESETS: ClassVar[dict] = {
        "default": frozenset(
            [
                SpanGroup.JOB,
                SpanGroup.CHECKPOINT,
                SpanGroup.EVALUATE,
                INFERENCE,
            ]
        ),
        "per_step": frozenset(
            [
                SpanGroup.JOB,
                SpanGroup.CHECKPOINT,
                SpanGroup.EVALUATE,
                SpanGroup.MODEL_INIT,
                SpanGroup.LOAD_CHECKPOINT,
                SpanGroup.STEP,
                SpanGroup.FORWARD_BACKWARD,
                SpanGroup.OPTIMIZER,
                COMMUNICATION,
                DATA_LOADING,
                INFERENCE,
            ]
        ),
        "all": ALL_GROUPS,
    }
