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

"""SpanRegistry: consuming libraries declare the span groups they emit.

Lens ships no span-group names and no preset contents of its own. A consuming
library registers what it emits under its own namespace, and users select from
that with the ``<PREFIX>_SPAN_GROUPS`` spec string::

    SpanRegistry.register(
        "megatron",
        groups={"step", "microbatch", "layer", "communication"},
        presets={"default": {"step"}, "per_step": {"step", "microbatch"}},
    )

Two libraries can register at once -- an RL job driving Megatron has both in one
process -- which is why this is a registry rather than the subclass hook it
replaces. Group names share one flat namespace, which keeps instrumentation call
sites terse. Two libraries claiming one name share it, and both get a
``WARNING``. This cannot be an error: neither library knows it is second, and the
check runs while a module is being imported, where there is no caller to catch
it. Presets **union** across
namespaces, so ``SPAN_GROUPS=default`` means every registered library's idea of
default, rather than whichever one registered last.

**Register before calling** :func:`~nemo.lens.handle.setup_telemetry`. Each
library owns its own telemetry, so importing a library is what registers its
groups. A spec entry naming a group this process has no registration for is
reported as a ``WARNING`` naming what *is* registered -- never an exception.

It cannot be an exception, because a registry is **per process** while the spec
is typically **job-wide**. A launcher agent and a spawned checkpoint worker
inherit one ``NEMO_LENS_SPAN_GROUPS`` from the trainer but import a different set
of libraries, so a value that is perfectly valid in the trainer names nothing in
them. That is not a typo, and telemetry must not take those processes down over
it. A process that wants its own vocabulary gives itself its own env prefix::

    NemoLensConfig.from_env(prefix="NVRX_OTEL", fallback_prefix="NEMO_LENS")

A preset may name any group, not just the ones this call declares. That is how a
library layers on one it depends on -- put Megatron's ``step`` in your own
``default``. A member no one has registered is kept but ignored, because every
read compares presets against the groups currently registered; it starts working
if its owner is imported later. Import order therefore does not matter, and a
library can name a group from an *optional* dependency without the process dying
when that dependency is absent. It warns, because a typo produces the same
situation.

Registering *after* ``setup_telemetry`` still works -- the spec is re-resolved --
but it logs a warning, because it means the library's telemetry module was
imported too late to be selectable at startup.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable, Mapping
from typing import NamedTuple

#: Built-in preset meaning "every group currently registered". This is a
#: wildcard over the registry, not a hard-coded list -- lens names no groups.
ALL = "all"

_LOG = logging.getLogger(__name__)


class _Resolution(NamedTuple):
    """A resolution, plus the registry contents it was computed from.

    ``state`` logs its report after releasing its lock, so the values the message
    needs have to be carried alongside the resolution rather than looked up
    again. When the message was built from a second query, it could contradict
    itself: calling a group unregistered while also listing it as registered.
    """

    enabled: frozenset[str]
    pending: frozenset[str]
    registry_empty: bool
    groups: frozenset[str]
    presets: frozenset[str]
    namespaces: tuple[str, ...]


class SpanRegistry:
    """Process-global registry of span groups, keyed by consuming library."""

    _LOCK: threading.Lock = threading.Lock()
    _GROUPS: dict[str, frozenset[str]] = {}
    _PRESETS: dict[str, dict[str, frozenset[str]]] = {}

    @classmethod
    def register(
        cls,
        namespace: str,
        groups: Iterable[str],
        presets: Mapping[str, Iterable[str]] | None = None,
        *,
        allow_override: bool = False,
    ) -> None:
        """Declare the span groups a library emits.

        Args:
            namespace: Owning library, e.g. ``"megatron"``. Also the key for
                :meth:`unregister`.
            groups: Group names this library emits. Lowercase ``snake_case``.
            presets: Optional named subsets, e.g. ``{"default": {"step"}}``.
                A member may be any group already in the registry, not only the
                ones this call declares -- that is how a library layers a preset
                onto a group it depends on. Presets union across namespaces.
                ``"all"`` is reserved.
            allow_override: Permit re-registering *namespace*, and silence the
                warning for claiming a group name another namespace already
                registered. Re-registering **replaces** the namespace wholesale:
                omit ``presets`` and its previous presets are dropped, not
                merged. Pass the full set every time.

        Raises:
            ValueError: Empty namespace, empty group name, a group or preset
                named ``"all"``, or re-registering a namespace without
                ``allow_override=True``.

        Two conditions that used to raise are now warnings: claiming a group
        name another namespace already registered, and naming a group nothing has
        registered yet. No single library can prevent either one, and both happen
        during import, where there is no caller to catch them.
        """
        if not namespace:
            raise ValueError("Namespace must be a non-empty string.")
        resolved_groups = frozenset(g.strip().lower() for g in groups)
        if not all(resolved_groups):
            raise ValueError(f"Namespace {namespace!r} registered an empty group name.")
        if ALL in resolved_groups:
            # Reserved on both sides of the namespace. resolve() checks presets
            # first, so a group called "all" would be permanently unselectable:
            # asking for it by name would enable every group in the process.
            raise ValueError(
                f"Group name {ALL!r} is reserved; it always means every group. "
                f"Namespace {namespace!r} must pick another name."
            )

        # Everything that touches the registry -- the collision scan, preset
        # normalisation, and the commit -- happens under one lock hold, so a
        # concurrent register/unregister cannot slip between resolving a preset
        # member and storing it. Nothing in here calls into nemo.lens.state, and
        # nothing in here logs. See the lock-ordering note on _notify.
        with cls._LOCK:
            if namespace in cls._GROUPS and not allow_override:
                raise ValueError(
                    f"Namespace {namespace!r} is already registered. "
                    "Pass allow_override=True to replace it."
                )
            collisions: list[tuple[str, list[str]]] = []
            if not allow_override:
                # `namespace` itself cannot be in _GROUPS here -- the check above
                # already raised for that -- so every entry belongs to some other
                # namespace and none needs skipping.
                for other, owned in cls._GROUPS.items():
                    clash = owned & resolved_groups
                    if clash:
                        collisions.append((other, sorted(clash)))

            # A preset may reference groups another namespace already registered,
            # so a dependent library can layer on one it imports. A member that
            # is not referenceable yet is kept anyway and reported below -- see
            # _normalise_presets. This namespace's *previous* groups are excluded:
            # re-registering with allow_override replaces them, so a member that
            # depends on one this call is dropping is reported like any other.
            others = frozenset(
                group for ns, owned in cls._GROUPS.items() if ns != namespace for group in owned
            )
            resolved_presets, deferred = cls._normalise_presets(
                namespace, presets, resolved_groups | others
            )

            cls._GROUPS[namespace] = resolved_groups
            cls._PRESETS[namespace] = resolved_presets

        # Logged after the lock, like _notify: a logging handler is arbitrary
        # user code and must not run with the registry held.
        for other, clash in collisions:
            _LOG.warning(
                "Namespace %r registers span group(s) %s that namespace %r already "
                "registered. Group names are one flat namespace, so both libraries' "
                "spans are now behind that one gate: enabling the name enables both. "
                "Pass allow_override=True if that is intended, or rename.",
                namespace,
                clash,
                other,
            )
        for preset_name, unresolved in deferred:
            _LOG.warning(
                "Preset %r in namespace %r names span group(s) %s that nothing has "
                "registered. They are ignored for now and take effect if the library "
                "that owns them is imported later; if a name is a typo, the preset "
                "will quietly select less than you expect.",
                preset_name,
                namespace,
                unresolved,
            )
        cls._warn_if_late(namespace)
        cls._notify()

    @staticmethod
    def _normalise_presets(
        namespace: str,
        presets: Mapping[str, Iterable[str]] | None,
        referenceable: frozenset[str],
    ) -> tuple[dict[str, frozenset[str]], list[tuple[str, list[str]]]]:
        """Lower-case presets, and report members *referenceable* does not cover.

        Returns the normalised presets plus ``(preset, unresolved members)`` pairs
        for the caller to log once it has released the lock. Unresolved members
        are kept rather than dropped, because layering a preset onto an optional
        dependency is a reasonable thing to do, and refusing it here used to kill
        the importing process.

        Pure string work -- touches no shared state and takes no lock -- so it is
        safe to call with ``_LOCK`` held. That matters, because normalisation and
        commit have to see the same registry snapshot.
        """
        resolved: dict[str, frozenset[str]] = {}
        deferred: list[tuple[str, list[str]]] = []
        for name, members in (presets or {}).items():
            key = name.strip().lower()
            if key == ALL:
                raise ValueError(f"Preset name {ALL!r} is reserved; it always means every group.")
            member_set = frozenset(m.strip().lower() for m in members)
            unknown = member_set - referenceable
            if unknown:
                deferred.append((key, sorted(unknown)))
            # Stored raw, unresolved members included. _snapshot() compares every
            # read against the groups currently registered, so a member no one
            # owns is ignored now and starts working once its owner registers.
            # This is why import order does not matter.
            resolved[key] = member_set
        return resolved, deferred

    @classmethod
    def unregister(cls, namespace: str) -> None:
        """Remove everything *namespace* registered.

        Raises:
            ValueError: If *namespace* was never registered.
        """
        with cls._LOCK:
            if namespace not in cls._GROUPS:
                raise ValueError(f"Namespace {namespace!r} is not registered.")
            del cls._GROUPS[namespace]
            cls._PRESETS.pop(namespace, None)
        cls._notify()

    @classmethod
    def clear(cls) -> None:
        """Drop every registration. Primarily a test fixture hook."""
        with cls._LOCK:
            cls._GROUPS.clear()
            cls._PRESETS.clear()
        cls._notify()

    @classmethod
    def _groups_locked(cls) -> frozenset[str]:
        """Union of every namespace's groups. Caller must already hold ``_LOCK``."""
        return frozenset().union(*cls._GROUPS.values()) if cls._GROUPS else frozenset()

    @classmethod
    def groups(cls) -> frozenset[str]:
        """Every group name currently registered, across all namespaces."""
        with cls._LOCK:
            return cls._groups_locked()

    @classmethod
    def namespaces(cls) -> list[str]:
        """Sorted list of registered namespaces."""
        with cls._LOCK:
            return sorted(cls._GROUPS)

    @classmethod
    def _snapshot(
        cls,
    ) -> tuple[dict[str, frozenset[str]], frozenset[str], tuple[str, ...]]:
        """``(presets, groups, namespaces)`` from a single ``_LOCK`` hold.

        One hold, so they cannot come from different registry generations. That
        includes ``namespaces``, which only the diagnostic needs: a warning built
        from a second hold could name a group as unregistered while listing it as
        registered. Everything after the ``with`` is pure work on local copies.
        """
        with cls._LOCK:
            merged: dict[str, set[str]] = {}
            for by_name in cls._PRESETS.values():
                for name, members in by_name.items():
                    merged.setdefault(name, set()).update(members)
            everything = cls._groups_locked()
            namespaces = tuple(sorted(cls._GROUPS))

        # Intersected with the live group set, so a preset can never name a group
        # that is no longer registered. unregister() and register(allow_override=True)
        # both drop a namespace's groups while another namespace's preset may still
        # reference them -- without this, `default` could enable a group absent from
        # `all`, which contradicts the definition of `all`. It is also what lets
        # registration accept a member no one owns yet: a member that was never
        # registered and one whose group was unregistered later are the same
        # state, and both are handled here.
        result = {name: frozenset(members) & everything for name, members in merged.items()}
        result[ALL] = everything
        return result, everything, namespaces

    @classmethod
    def presets(cls) -> dict[str, frozenset[str]]:
        """Preset name -> members, unioned across every namespace.

        Includes the built-in ``"all"``. Members are always a subset of
        :meth:`groups`.
        """
        return cls._snapshot()[0]

    @classmethod
    def resolve(cls, spec: str) -> tuple[frozenset[str], frozenset[str]]:
        """Resolve a spec string against the registry.

        The spec is a comma-separated, case-insensitive mix of preset names and
        bare group names.

        Unlike the fixed-membership scheme this replaces, an unrecognised entry
        is **not** an error: the library that owns it may not have imported yet.
        It is returned separately so the caller can report it.

        Returns:
            ``(enabled, pending)`` -- the groups that resolved, and the spec
            entries that matched nothing yet.
        """
        resolution = cls._resolve_snapshot(spec)
        return resolution.enabled, resolution.pending

    @classmethod
    def _resolve_snapshot(cls, spec: str) -> _Resolution:
        """:meth:`resolve`, plus everything the caller's diagnostic needs.

        ``nemo.lens.state`` needs all of it to describe one registry generation.
        Asking the registry separately -- ``resolve()`` then ``groups()`` -- lets a
        concurrent registration land between them, so the enabled set and the
        warning built from it disagree about what was registered.
        """
        presets, known, namespaces = cls._snapshot()

        enabled: set[str] = set()
        pending: set[str] = set()
        for part in (p.strip().lower() for p in spec.split(",") if p.strip()):
            if part in presets:
                enabled |= presets[part]
            elif part in known:
                enabled.add(part)
            else:
                pending.add(part)
        return _Resolution(
            enabled=frozenset(enabled),
            pending=frozenset(pending),
            registry_empty=not known,
            groups=known,
            presets=frozenset(presets),
            namespaces=namespaces,
        )

    @staticmethod
    def _warn_if_late(namespace: str) -> None:
        """Warn when a library registers after telemetry is already running.

        Not an error: refusing the registration, or raising, would let a
        telemetry misconfiguration take down a training job. But it does mean the
        group was not selectable when the user's spec was validated, so it is not
        something to swallow either.
        """
        from nemo.lens import handle

        if handle._INITIALIZED:
            _LOG.warning(
                "Span groups for namespace %r were registered after setup_telemetry(); "
                "they were not selectable when the span-group spec was validated. "
                "Import this library's telemetry module before calling setup_telemetry().",
                namespace,
            )

    @staticmethod
    def _notify() -> None:
        """Recompute the enabled set, since the registry just changed.

        **Must be called with ``_LOCK`` released.** The lock order across the two
        modules is state -> registry: ``nemo.lens.state`` holds its own lock while
        resolving against this registry, so calling into state while holding
        ``_LOCK`` would close the cycle and deadlock. Every mutator here notifies
        after its ``with cls._LOCK`` block, never inside it.
        """
        from nemo.lens.state import refresh_enabled_span_groups

        refresh_enabled_span_groups()
