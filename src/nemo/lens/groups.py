# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""SpanGroup base class and shared presets for cross-library span granularity control."""

from typing import ClassVar, Final


class SpanGroup:
    """Named constants for span granularity groups (shared across all libraries).

    Presets:
        ``"default"`` — coarse-grained only (job, checkpoint, evaluate).
        ``"per_step"`` — adds per-step boundaries.
        ``"all"`` — every span group including fine-grained.

    Subclasses (e.g. ``MegatronSpanGroup``) add library-specific groups.
    """

    # ------------------------------------------------------------------ #
    # Coarse-grained (included in "default")
    # ------------------------------------------------------------------ #

    JOB = "job"
    CHECKPOINT = "checkpoint"
    EVALUATE = "evaluate"

    # ------------------------------------------------------------------ #
    # Medium-grained (included in "per_step")
    # ------------------------------------------------------------------ #

    MODEL_INIT = "model_init"
    LOAD_CHECKPOINT = "load_checkpoint"
    STEP = "step"
    FORWARD_BACKWARD = "forward_backward"
    OPTIMIZER = "optimizer"

    # ------------------------------------------------------------------ #
    # All groups and presets
    # ------------------------------------------------------------------ #

    ALL_GROUPS: Final[frozenset] = frozenset(
        [
            JOB,
            CHECKPOINT,
            EVALUATE,
            MODEL_INIT,
            LOAD_CHECKPOINT,
            STEP,
            FORWARD_BACKWARD,
            OPTIMIZER,
        ]
    )

    _PRESETS: ClassVar[dict] = {
        "default": frozenset([JOB, CHECKPOINT, EVALUATE]),
        "per_step": frozenset(
            [
                JOB,
                CHECKPOINT,
                EVALUATE,
                MODEL_INIT,
                LOAD_CHECKPOINT,
                STEP,
                FORWARD_BACKWARD,
                OPTIMIZER,
            ]
        ),
        "all": ALL_GROUPS,
    }

    @classmethod
    def resolve(cls, spec: str) -> frozenset:
        """Resolve a span-group spec string to a frozenset of group names.

        The spec may be a preset keyword, comma-separated group names, or a mix.

        Args:
            spec: Spec string (case-insensitive).

        Returns:
            A frozenset of group-name strings.

        Raises:
            ValueError: If an unknown keyword or group name is encountered.
        """
        result: set = set()
        for part in (p.strip().lower() for p in spec.split(",") if p.strip()):
            if part in cls._PRESETS:
                result |= cls._PRESETS[part]
            elif part in cls.ALL_GROUPS:
                result.add(part)
            else:
                valid = sorted(cls.ALL_GROUPS | set(cls._PRESETS))
                raise ValueError(f"Unknown span group or preset: {part!r}. Valid options: {valid}")
        return frozenset(result)
