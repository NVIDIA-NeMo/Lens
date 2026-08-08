# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

    #: Groups whose spans are semantic goodput/resiliency boundaries (as opposed
    #: to fine-grained profiling detail). Single source of truth for BOTH the
    #: ``"goodput"`` runtime preset and each span's ``lens.span_category``
    #: attribute. Subclasses override to add library-specific goodput groups.
    GOODPUT_GROUPS: Final[frozenset] = frozenset(
        [JOB, CHECKPOINT, EVALUATE, MODEL_INIT, LOAD_CHECKPOINT, STEP]
    )

    _PRESETS: ClassVar[dict] = {
        "default": frozenset([JOB, CHECKPOINT, EVALUATE]),
        "goodput": GOODPUT_GROUPS,
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
        "profiling": ALL_GROUPS,
        "all": ALL_GROUPS,
    }

    @classmethod
    def categories(cls) -> dict:
        """Return the ``{group: category}`` map for every group in ``ALL_GROUPS``.

        A group is ``"goodput"`` if it is in :attr:`GOODPUT_GROUPS`, else
        ``"profiling"``. Consumed by :func:`~nemo.lens.state.set_group_categories`
        so every emitted span carries a ``lens.span_category`` attribute.
        """
        return {g: ("goodput" if g in cls.GOODPUT_GROUPS else "profiling") for g in cls.ALL_GROUPS}

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
