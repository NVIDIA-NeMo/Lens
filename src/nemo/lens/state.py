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


_GROUP_CATEGORIES: dict = {}


def set_group_categories(mapping: dict) -> None:
    """Register the span-group -> category map (e.g. ``{"checkpoint": "goodput"}``).

    Called once from :func:`~nemo.lens.handle.setup_telemetry`, populated from
    the config's span-group class. Categories let a full-depth ``all`` run be
    sliced offline into ``goodput`` (semantic boundaries) vs ``profiling``
    (fine-grained detail) via the ``lens.span_category`` span attribute.
    """
    global _GROUP_CATEGORIES
    with _LOCK:
        _GROUP_CATEGORIES = dict(mapping)


def category_of(group: str) -> str | None:
    """Return the category (``"goodput"``/``"profiling"``) for *group*, or None.

    Returns None before :func:`set_group_categories` is called or for groups
    with no registered category.
    """
    return _GROUP_CATEGORIES.get(group)


_PP_TRACE_CARRIER: dict | None = None


def set_pp_trace_carrier(carrier: dict | None) -> None:
    """Store the pipeline-parallel trace carrier for cross-stage linking.

    Called from the training loop after :func:`broadcast_trace_context`.
    """
    global _PP_TRACE_CARRIER
    _PP_TRACE_CARRIER = carrier


def get_pp_trace_carrier() -> dict | None:
    """Return the current PP trace carrier, or None."""
    return _PP_TRACE_CARRIER
