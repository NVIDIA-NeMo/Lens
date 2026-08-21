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
replaces. Group names share one flat namespace; ``allow_override`` guards
collisions. Presets **union** across namespaces, so ``SPAN_GROUPS=default``
means every registered library's idea of default, rather than whichever one
registered last.

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

A preset may name any group already in the registry, not just the ones this call
declares. That is how a library layers on one it depends on -- import Megatron,
then put its ``step`` in your own ``default``. Referencing a name without
importing its owner is an error at registration, which keeps the import-order
requirement self-enforcing rather than silent.

Registering *after* ``setup_telemetry`` still works -- the spec is re-resolved --
but it logs a warning, because it means the library's telemetry module was
imported too late to be selectable at startup.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable, Mapping

#: Built-in preset meaning "every group currently registered". This is a
#: wildcard over the registry, not a hard-coded list -- lens names no groups.
ALL = "all"


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
            allow_override: Permit re-registering *namespace*, or claiming a
                group name another namespace already registered. Re-registering
                **replaces** the namespace wholesale: omit ``presets`` and its
                previous presets are dropped, not merged. Pass the full set every
                time.

        Raises:
            ValueError: Empty namespace, empty group name, a group or preset
                named ``"all"``, a preset naming a group no namespace has
                registered, or a collision without ``allow_override=True``.
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

        # Everything that touches the registry -- both collision checks, preset
        # validation, and the commit -- happens under one lock hold, so a
        # concurrent register/unregister cannot slip between validating a preset
        # reference and storing it. Nothing in here calls into nemo.lens.state;
        # see the lock-ordering note on _notify.
        with cls._LOCK:
            if namespace in cls._GROUPS and not allow_override:
                raise ValueError(
                    f"Namespace {namespace!r} is already registered. "
                    "Pass allow_override=True to replace it."
                )
            if not allow_override:
                # `namespace` itself cannot be in _GROUPS here -- the check above
                # already raised for that -- so every entry belongs to some other
                # namespace and none needs skipping.
                for other, owned in cls._GROUPS.items():
                    clash = owned & resolved_groups
                    if clash:
                        raise ValueError(
                            f"Group(s) {sorted(clash)} are already registered by "
                            f"namespace {other!r}. Pass allow_override=True to share them."
                        )

            # A preset may reference groups another namespace already registered,
            # so a dependent library can layer on one it imports. A name whose
            # owner has not been imported yet fails here rather than silently
            # resolving to nothing later. This namespace's *previous* groups are
            # excluded: re-registering with allow_override replaces them, so a
            # preset must not lean on one this call is dropping.
            others = frozenset(
                group for ns, owned in cls._GROUPS.items() if ns != namespace for group in owned
            )
            resolved_presets = cls._normalise_presets(namespace, presets, resolved_groups | others)

            cls._GROUPS[namespace] = resolved_groups
            cls._PRESETS[namespace] = resolved_presets

        cls._warn_if_late(namespace)
        cls._notify()

    @staticmethod
    def _normalise_presets(
        namespace: str,
        presets: Mapping[str, Iterable[str]] | None,
        referenceable: frozenset[str],
    ) -> dict[str, frozenset[str]]:
        """Lower-case and validate preset members against *referenceable*.

        Pure string work -- touches no shared state and takes no lock -- so it is
        safe to call with ``_LOCK`` held, which is the point: validation and
        commit have to see one registry snapshot.
        """
        resolved: dict[str, frozenset[str]] = {}
        for name, members in (presets or {}).items():
            key = name.strip().lower()
            if key == ALL:
                raise ValueError(f"Preset name {ALL!r} is reserved; it always means every group.")
            member_set = frozenset(m.strip().lower() for m in members)
            unknown = member_set - referenceable
            if unknown:
                raise ValueError(
                    f"Preset {name!r} in namespace {namespace!r} names {sorted(unknown)}, "
                    "which this call does not register and no other namespace has "
                    "registered either. Import the library that owns the group before "
                    "referencing it."
                )
            resolved[key] = member_set
        return resolved

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
    def _snapshot(cls) -> tuple[dict[str, frozenset[str]], frozenset[str]]:
        """``(presets, groups)`` from a single ``_LOCK`` hold.

        One hold, so the two cannot come from different registry generations.
        Everything after the ``with`` is pure string work on local copies.
        """
        with cls._LOCK:
            merged: dict[str, set[str]] = {}
            for by_name in cls._PRESETS.values():
                for name, members in by_name.items():
                    merged.setdefault(name, set()).update(members)
            everything = cls._groups_locked()

        # Intersected with the live group set, so a preset can never name a group
        # that is no longer registered. unregister() and register(allow_override=True)
        # both drop a namespace's groups while another namespace's preset may still
        # reference them -- without this, `default` could enable a group absent from
        # `all`, which contradicts both the definition of `all` and the check in
        # _normalise_presets that refuses such a preset at registration time.
        result = {name: frozenset(members) & everything for name, members in merged.items()}
        result[ALL] = everything
        return result, everything

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
        enabled, pending, _ = cls._resolve_snapshot(spec)
        return enabled, pending

    @classmethod
    def _resolve_snapshot(cls, spec: str) -> tuple[frozenset[str], frozenset[str], bool]:
        """:meth:`resolve`, plus whether the registry was empty in the same hold.

        ``nemo.lens.state`` needs all three to describe one registry generation.
        Asking the registry separately -- ``resolve()`` then ``groups()`` -- let a
        concurrent registration land between them, so the enabled set and the
        diagnostic could disagree about what was registered.
        """
        presets, known = cls._snapshot()

        enabled: set[str] = set()
        pending: set[str] = set()
        for part in (p.strip().lower() for p in spec.split(",") if p.strip()):
            if part in presets:
                enabled |= presets[part]
            elif part in known:
                enabled.add(part)
            else:
                pending.add(part)
        return frozenset(enabled), frozenset(pending), not known

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
            logging.getLogger(__name__).warning(
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
