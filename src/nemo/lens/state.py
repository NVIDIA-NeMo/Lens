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

Two ways in, and the last one called wins:

* :func:`set_span_group_spec` stores the raw user spec (e.g. ``"default,step"``)
  and resolves it **strictly** against :class:`~nemo.lens.groups.SpanRegistry`,
  raising on an entry that names nothing. Libraries register their groups when
  they are imported, which is before ``setup_telemetry``, so an unrecognised
  name at that point is a typo and gets said out loud.

  The spec is nonetheless retained, so that a library registering late still
  takes effect (:func:`refresh_enabled_span_groups`) instead of silently
  emitting nothing. That path warns — see ``SpanRegistry._warn_if_late``.
* :func:`set_enabled_span_groups` pins an explicit set and drops the spec, so a
  later registration cannot reopen it. This is how a disabled process stays
  disabled, and how a test enables exactly the groups it means to.

Before either is called every :func:`is_span_group_enabled` query returns
``False``.
"""

from __future__ import annotations

import logging
import threading

#: Guards ``_SPEC`` and ``_ENABLED_GROUPS`` together, and is held across
#: resolution so the two never drift apart.
#:
#: Lock order is **state -> registry**: the functions below hold this while
#: calling into :class:`~nemo.lens.groups.SpanRegistry`, which takes its own lock.
#: SpanRegistry's mutators therefore call back here only after releasing theirs
#: (see ``SpanRegistry._notify``). Do not call into this module while holding the
#: registry lock, or the cycle closes.
#:
#: :func:`is_span_group_enabled` deliberately does not take it. Rebinding a name
#: to an immutable frozenset is atomic, so the hot path stays one membership test.
_LOCK = threading.Lock()
_ENABLED_GROUPS: frozenset = frozenset()
_SPEC: str = ""
_PENDING: frozenset = frozenset()
#: Last unresolved set already reported, so re-resolving on each registration
#: does not repeat an identical warning.
_WARNED: frozenset | None = None


def set_enabled_span_groups(groups: frozenset) -> None:
    """Pin the active span groups explicitly, discarding any stored spec.

    Subsequent registrations will not change the set — use
    :func:`set_span_group_spec` if you want it to track the registry.
    """
    global _ENABLED_GROUPS, _SPEC, _PENDING, _WARNED
    with _LOCK:
        _ENABLED_GROUPS = groups
        _SPEC = ""
        _PENDING = frozenset()
        _WARNED = None


def set_span_group_spec(spec: str) -> None:
    """Store the user's span-group spec and resolve it against the registry.

    Called from :func:`~nemo.lens.handle.setup_telemetry`. Never raises: an entry
    that resolves to nothing is warned about and kept pending, because this
    process's registry is not the authority on a job-wide spec.
    """
    global _ENABLED_GROUPS, _SPEC, _PENDING, _WARNED

    from nemo.lens.groups import SpanRegistry  # imported before the lock, never under it

    with _LOCK:
        # Resolve inside the lock so the stored spec and the set derived from it
        # are always the same generation; a concurrent refresh cannot land a
        # resolution from a different one on top.
        enabled, unknown = SpanRegistry.resolve(spec)
        _SPEC = spec
        _ENABLED_GROUPS = enabled
        _PENDING = unknown
        _WARNED = unknown
        registry_empty = not SpanRegistry.groups()

    _report(spec, unknown, registry_empty)


def refresh_enabled_span_groups() -> None:
    """Re-resolve the stored spec. Called whenever the registry changes.

    A no-op when the groups were pinned explicitly rather than from a spec.
    Registration happens at import time, so the cost never lands on the hot
    path -- :func:`is_span_group_enabled` stays one frozenset membership test.
    """
    global _ENABLED_GROUPS, _PENDING, _WARNED

    from nemo.lens.groups import SpanRegistry  # imported before the lock, never under it

    with _LOCK:
        # Read, resolve and store in one hold. Splitting them would let two
        # registrations racing to notify commit out of order, so the enabled set
        # could keep a resolution from an older registry indefinitely -- until
        # some later registration happened to refresh it, or forever if none did.
        spec = _SPEC
        if not spec:
            return
        enabled, unknown = SpanRegistry.resolve(spec)
        _ENABLED_GROUPS = enabled
        _PENDING = unknown
        repeat = unknown == _WARNED
        _WARNED = unknown
        registry_empty = not SpanRegistry.groups()

    if not repeat:
        _report(spec, unknown, registry_empty)


def _report(spec: str, unknown: frozenset, registry_empty: bool) -> None:
    """Warn about spec entries that resolved to nothing.

    Deliberately a warning and not an exception: see the module docstring.
    Carries what *is* registered, because that is what turns "I set SPAN_GROUPS
    and got nothing" into a diagnosis -- either a typo, or a library this process
    never imported.
    """
    if not unknown:
        return

    from nemo.lens.groups import SpanRegistry

    log = logging.getLogger(__name__)
    if registry_empty:
        log.warning(
            "Span groups %r were requested, but no library has registered any span "
            "groups in this process. Nothing group-gated will be emitted. Import the "
            "telemetry module of the library you are instrumenting before "
            "setup_telemetry().",
            spec,
        )
        return

    log.warning(
        "Span group spec %r names %s, which no library registered in this process "
        "provides; nothing will be emitted for those. Registered groups: %s. "
        "Presets: %s. Namespaces: %s. If a name is real, the library that owns it "
        "was not imported here -- which is expected when several processes share "
        "one spec, and is why this is a warning rather than an error.",
        spec,
        sorted(unknown),
        sorted(SpanRegistry.groups()),
        sorted(SpanRegistry.presets()),
        SpanRegistry.namespaces(),
    )


def enabled_span_groups() -> frozenset:
    """The groups currently enabled. Diagnostic; not for the hot path."""
    return _ENABLED_GROUPS


def pending_span_groups() -> frozenset:
    """Spec entries that resolved to nothing. Diagnostic; not for the hot path.

    Non-empty here is the usual explanation for "I set SPAN_GROUPS and got
    nothing": either a typo, or a name belonging to a library this process never
    imported.
    """
    return _PENDING


def is_span_group_enabled(group: str) -> bool:
    """Return ``True`` if the named span group is currently enabled.

    This is the primary check at every instrumentation site (~2ns overhead).
    Returns ``False`` before any group has been enabled.
    """
    return group in _ENABLED_GROUPS


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
