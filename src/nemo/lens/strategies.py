# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Export strategy registry: maps strategy names to per-rank decision callables.

A strategy is a callable taking ``(config, rank, world_size)`` and returning
``True`` if this rank should export telemetry data. Lens ships four built-ins
and lets users register their own under a chosen name; the chosen name is
selected via ``NemoLensConfig.export_strategy`` or the
``NEMO_LENS_EXPORT_STRATEGY`` env var.
"""

from __future__ import annotations

import hashlib
import os
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nemo.lens.config import NemoLensConfig


type ExportStrategy = Callable[["NemoLensConfig", int, int], bool]
"""Callable taking ``(config, rank, world_size)``; returns ``True`` if this
rank should export. Implementations may read environment variables for extra
context (e.g. ``LOCAL_RANK``)."""


_REGISTRY: dict[str, ExportStrategy] = {}
_REGISTRY_LOCK = threading.Lock()

#: Names of built-in strategies. Cannot be overridden without ``allow_override=True``,
#: cannot be unregistered.
BUILTIN_STRATEGIES: frozenset[str] = frozenset(
    {"all_ranks", "single_rank", "sampled", "first_rank_per_node"}
)


def _all_ranks(config: NemoLensConfig, rank: int, world_size: int) -> bool:
    """Built-in: every rank exports."""
    return True


def _single_rank(config: NemoLensConfig, rank: int, world_size: int) -> bool:
    """Built-in: only ``config.export_rank`` exports (``-1`` means last rank)."""
    resolved = config.export_rank if config.export_rank >= 0 else (world_size - 1)
    return rank == resolved


def _sampled(config: NemoLensConfig, rank: int, world_size: int) -> bool:
    """Built-in: deterministic hash-based sampling. Same rank + same rate → same outcome."""
    h = hashlib.md5(str(rank).encode()).hexdigest()
    return (int(h, 16) % 10000) / 10000.0 < config.export_sample_rate


def _first_rank_per_node(config: NemoLensConfig, rank: int, world_size: int) -> bool:
    """Built-in: the first local rank on each node exports.

    Reads ``LOCAL_RANK`` (set by torchrun, deepspeed, etc.). Treats a missing
    ``LOCAL_RANK`` as ``"0"``, which gives the expected single-node behaviour.
    """
    return int(os.environ.get("LOCAL_RANK", "0")) == 0


def register_export_strategy(
    name: str,
    strategy: ExportStrategy,
    *,
    allow_override: bool = False,
) -> None:
    """Register a custom export strategy under ``name``.

    Once registered, the strategy is selectable via
    ``NemoLensConfig(export_strategy=name)`` or
    ``NEMO_LENS_EXPORT_STRATEGY=<name>``.

    Args:
        name: Strategy identifier. Must be a non-empty string.
        strategy: Callable ``(config, rank, world_size) -> bool``.
        allow_override: If ``False`` (default), raises ``ValueError`` when
            ``name`` is already registered. Built-in names are protected
            and additionally require ``allow_override=True`` to replace.

    Raises:
        ValueError: If ``name`` is empty, or already registered without
            ``allow_override=True``.
    """
    if not name:
        raise ValueError("Strategy name must be a non-empty string.")
    with _REGISTRY_LOCK:
        if name in _REGISTRY and not allow_override:
            raise ValueError(
                f"Strategy {name!r} is already registered. Pass allow_override=True to replace it."
            )
        _REGISTRY[name] = strategy


def unregister_export_strategy(name: str) -> None:
    """Remove a strategy from the registry.

    Raises:
        ValueError: If ``name`` refers to a built-in strategy, or is unknown.
    """
    if name in BUILTIN_STRATEGIES:
        raise ValueError(f"Cannot unregister built-in strategy {name!r}.")
    with _REGISTRY_LOCK:
        if name not in _REGISTRY:
            raise ValueError(f"Strategy {name!r} is not registered.")
        del _REGISTRY[name]


def get_export_strategy(name: str) -> ExportStrategy:
    """Look up a registered strategy by name.

    Raises:
        ValueError: If ``name`` is not registered. Error message lists the
            currently registered names.
    """
    with _REGISTRY_LOCK:
        if name not in _REGISTRY:
            available = sorted(_REGISTRY)
            raise ValueError(
                f"Unknown export_strategy {name!r}. Registered strategies: {available}."
            )
        return _REGISTRY[name]


def registered_strategies() -> list[str]:
    """Return a sorted list of registered strategy names (built-ins + custom)."""
    with _REGISTRY_LOCK:
        return sorted(_REGISTRY)


# Register built-ins at import time. Use the internal mutation to avoid the
# "already registered" guard during initial population.
with _REGISTRY_LOCK:
    _REGISTRY["all_ranks"] = _all_ranks
    _REGISTRY["single_rank"] = _single_rank
    _REGISTRY["sampled"] = _sampled
    _REGISTRY["first_rank_per_node"] = _first_rank_per_node
