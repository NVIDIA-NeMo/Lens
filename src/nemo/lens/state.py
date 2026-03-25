# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Module-level span group state — importable anywhere without circular deps.

Holds a frozenset of enabled span groups so that any module can call
:func:`is_span_group_enabled` without importing the full nemo.lens package.

Span groups are registered once via :func:`set_enabled_span_groups` (called
from :func:`~nemo.lens.handle.setup_telemetry`).  Before that call every
:func:`is_span_group_enabled` query returns ``False``.
"""

from __future__ import annotations

import threading

_LOCK = threading.Lock()
_ENABLED_GROUPS: frozenset = frozenset()


def set_enabled_span_groups(groups: frozenset) -> None:
    """Register the active span groups.

    Called once from :func:`~nemo.lens.handle.setup_telemetry`.
    Subsequent calls override the previous value (useful for testing).
    """
    global _ENABLED_GROUPS
    with _LOCK:
        _ENABLED_GROUPS = groups


def is_span_group_enabled(group: str) -> bool:
    """Return ``True`` if the named span group is currently enabled.

    This is the primary check at every instrumentation site (~2ns overhead).
    Returns ``False`` before :func:`set_enabled_span_groups` is called.
    """
    return group in _ENABLED_GROUPS
