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

"""Consumer-driven metric registry.

Lens ships instruments only for names it owns (``gen_ai.*`` via
:mod:`nemo.lens.instruments.inference`). Metrics whose *set of names* belongs to
a consuming framework — RL training series in NeMo-RL, environment series in
NeMo-Gym — are defined by that framework and registered here, so lens never has
to carry a per-consumer module that the consumer then has to import, extend, or
override.

A consumer declares its series once as :class:`MetricSpec` entries under a group
name. The names are the consumer's own — keep them in one module in the
consumer's tree rather than scattering literals::

    from nemo.lens.instruments import MetricSpec, record_metrics, register_metric_group

    register_metric_group(
        "rl",
        [
            MetricSpec("reward_mean", "rl.reward.mean", "gauge"),
            MetricSpec("kl_divergence", "rl.kl_divergence", "gauge"),
            MetricSpec(
                "generation_duration_ms",
                "rl.generation.duration_ms",
                "histogram",
                unit="ms",
            ),
        ],
    )

then records against it, passing only the series it has values for::

    record_metrics(meter, "rl", reward_mean=0.85, kl_divergence=0.02)

Instruments are created lazily the first time a group is recorded against a
meter and cached per meter, matching the behaviour of the built-in modules. As
with every instrument path, recording never raises into the caller: failures are
logged and swallowed so instrumentation cannot take down a training loop.

A series whose *attributes* change during a run needs an ``observable_*`` kind
rather than a synchronous one. A synchronous gauge keeps its last value per
attribute set and re-exports it every cycle, with no way to say that a set no
longer applies — so a point first reported under one label and later under
another is exported under *both*, forever, and any aggregation over the label
counts it twice. An observable instrument is rebuilt from its callback's return
each cycle, so a label the callback stops yielding simply ends::

    def _hosts(options):
        return [Observation(used, {"host.name": h}) for h, used in sample().items()]

    register_metric_group(
        "rl",
        [MetricSpec("host_mem", "rl.host.memory.used", "observable_gauge",
                    unit="By", callback=_hosts)],
    )
    create_metric_instruments(meter, "rl")  # observables need this; see below
"""

from __future__ import annotations

import logging
import os
import threading
import weakref
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from opentelemetry import metrics

_logger = logging.getLogger(__name__)

#: How each supported synchronous kind is created on a ``Meter`` and how a value
#: is emitted. Maps ``kind`` -> (meter factory method, instrument emit method).
_KIND_METHODS: dict[str, tuple[str, str]] = {
    "gauge": ("create_gauge", "set"),
    "histogram": ("create_histogram", "record"),
    "counter": ("create_counter", "add"),
    "up_down_counter": ("create_up_down_counter", "add"),
}

#: How each observable (asynchronous) kind is created. These take a callback at
#: creation and are never emitted to, so there is no second method to record.
_OBSERVABLE_KIND_FACTORIES: dict[str, str] = {
    "observable_gauge": "create_observable_gauge",
    "observable_counter": "create_observable_counter",
    "observable_up_down_counter": "create_observable_up_down_counter",
}

#: Kinds a :class:`MetricSpec` may declare.
METRIC_KINDS: frozenset[str] = frozenset(_KIND_METHODS) | frozenset(_OBSERVABLE_KIND_FACTORIES)

#: Kinds whose points come from a callback rather than from :func:`record_metrics`.
OBSERVABLE_METRIC_KINDS: frozenset[str] = frozenset(_OBSERVABLE_KIND_FACTORIES)


@dataclass(frozen=True)
class MetricSpec:
    """Declaration of a single metric series.

    Args:
        key: Name the caller uses when recording (e.g. ``reward_mean`` for
            ``record_metrics(meter, group, reward_mean=...)``). Unique within a
            group.
        name: OTel instrument name actually emitted (e.g. ``rl.reward.mean``).
            Owned by the consumer; keep it in one name module rather than
            inlining literals across call sites.
        kind: One of :data:`METRIC_KINDS`. ``gauge`` for a level that is set
            outright, ``histogram`` for a distribution, ``counter`` /
            ``up_down_counter`` for an additive series. The ``observable_*``
            kinds are reported by ``callback`` instead of by
            :func:`record_metrics` — see :data:`OBSERVABLE_METRIC_KINDS`.
        unit: UCUM unit string (e.g. ``ms``, ``{token}/s``). Empty when unitless.
        description: Human-readable description attached to the instrument.
        callback: Required for an ``observable_*`` kind, rejected otherwise.
            Invoked by the SDK once per collection cycle, and what it returns is
            the complete set of points for that cycle. Keep it cheap and
            non-blocking: it runs on the exporter's collection path, so work
            done here delays every metric the process exports.
    """

    key: str
    name: str
    kind: str = "gauge"
    unit: str = ""
    description: str = ""
    callback: metrics.CallbackT | None = None

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("MetricSpec.key must be a non-empty string.")
        if not self.name:
            raise ValueError(f"MetricSpec.name must be a non-empty string (key={self.key!r}).")
        if self.kind not in METRIC_KINDS:
            raise ValueError(
                f"MetricSpec.kind {self.kind!r} is not one of {sorted(METRIC_KINDS)} "
                f"(key={self.key!r})."
            )
        # Both directions are rejected, because both are silent otherwise: an
        # observable without a callback reports nothing at all, and a callback
        # on a synchronous kind is simply never called.
        if self.kind in _OBSERVABLE_KIND_FACTORIES and self.callback is None:
            raise ValueError(
                f"MetricSpec.kind {self.kind!r} requires a callback, which is the only "
                f"way an observable instrument reports a value (key={self.key!r})."
            )
        if self.kind not in _OBSERVABLE_KIND_FACTORIES and self.callback is not None:
            raise ValueError(
                f"MetricSpec.callback is only used by the {sorted(_OBSERVABLE_KIND_FACTORIES)} "
                f"kinds; {self.kind!r} records through record_metrics (key={self.key!r})."
            )


@dataclass
class _Group:
    specs: dict[str, MetricSpec]
    # Instruments materialised per meter; entries drop when the meter is GC'd.
    instruments: weakref.WeakKeyDictionary = field(default_factory=weakref.WeakKeyDictionary)


_REGISTRY: dict[str, _Group] = {}
_REGISTRY_LOCK = threading.Lock()

# Dedup keys for warnings already emitted, so a misconfigured call in a hot loop
# logs once rather than on every step. Each key is a tuple whose second element
# is the group name (see _warn_once call sites), which lets (re)registration
# clear a group's warnings. Guarded by its own short-held lock, independent of
# _REGISTRY_LOCK.
_WARNED: set[tuple] = set()
_WARN_LOCK = threading.Lock()


def _warn_once(dedup_key: tuple, msg: str, *args: object, exc_info: bool = False) -> None:
    """Emit ``msg`` at most once per ``dedup_key`` for the process lifetime."""
    with _WARN_LOCK:
        if dedup_key in _WARNED:
            return
        _WARNED.add(dedup_key)
    _logger.warning(msg, *args, exc_info=exc_info)


def _clear_group_warnings(group: str) -> None:
    """Forget a group's warnings so a redefinition warns afresh."""
    with _WARN_LOCK:
        stale = {k for k in _WARNED if len(k) >= 2 and k[1] == group}
        _WARNED.difference_update(stale)


def register_metric_group(
    group: str,
    specs: Iterable[MetricSpec],
    *,
    allow_override: bool = False,
) -> None:
    """Register a named collection of metric series.

    Args:
        group: Group identifier passed to :func:`record_metrics`. Non-empty.
        specs: The series in this group. Keys must be unique within the group.
        allow_override: If ``False`` (default), raise :class:`ValueError` when
            ``group`` is already registered. If ``True``, replace it and drop any
            instruments already created for the old definition.

    Raises:
        ValueError: If ``group`` is empty, ``specs`` is empty, a key is
            duplicated, or ``group`` is already registered without
            ``allow_override=True``.
    """
    if not group:
        raise ValueError("Metric group name must be a non-empty string.")

    by_key: dict[str, MetricSpec] = {}
    for spec in specs:
        if spec.key in by_key:
            raise ValueError(f"Duplicate metric key {spec.key!r} in group {group!r}.")
        by_key[spec.key] = spec
    if not by_key:
        raise ValueError(f"Metric group {group!r} must declare at least one MetricSpec.")

    with _REGISTRY_LOCK:
        if group in _REGISTRY and not allow_override:
            raise ValueError(
                f"Metric group {group!r} is already registered. "
                "Pass allow_override=True to replace it."
            )
        _REGISTRY[group] = _Group(specs=by_key)
    # A fresh definition should warn afresh (keys may have changed).
    _clear_group_warnings(group)


def unregister_metric_group(group: str) -> None:
    """Remove a group from the registry.

    Raises:
        ValueError: If ``group`` is not registered.
    """
    with _REGISTRY_LOCK:
        if group not in _REGISTRY:
            raise ValueError(f"Metric group {group!r} is not registered.")
        del _REGISTRY[group]


def registered_metric_groups() -> dict[str, tuple[MetricSpec, ...]]:
    """Return a snapshot mapping each registered group to its specs."""
    with _REGISTRY_LOCK:
        return {name: tuple(g.specs.values()) for name, g in _REGISTRY.items()}


def _instruments_for(meter: metrics.Meter, group: str, entry: _Group) -> dict[str, object] | None:
    """Return this group's instruments for ``meter``, creating them once.

    The cache read and write are inside the guard because it is keyed by a weak
    reference: a caller passing something not weak-referenceable (``None`` being
    the likely slip) would otherwise raise into the training loop.
    """
    try:
        instruments = entry.instruments.get(meter)
        if instruments is None:
            instruments = {}
            for key, spec in entry.specs.items():
                observable_factory = _OBSERVABLE_KIND_FACTORIES.get(spec.kind)
                if observable_factory is not None:
                    # The callback is handed over here and never again: from this
                    # point the SDK pulls, and the group's entry in this cache is
                    # only what keeps the instrument reachable.
                    instruments[key] = getattr(meter, observable_factory)(
                        name=spec.name,
                        callbacks=[spec.callback],
                        unit=spec.unit,
                        description=spec.description,
                    )
                    continue
                factory, _ = _KIND_METHODS[spec.kind]
                instruments[key] = getattr(meter, factory)(
                    name=spec.name,
                    unit=spec.unit,
                    description=spec.description,
                )
            entry.instruments[meter] = instruments
    except Exception:
        _warn_once(
            ("create", group),
            "Failed to resolve instruments for metric group %r with a meter of type %s",
            group,
            type(meter).__name__,
            exc_info=True,
        )
        return None
    return instruments


def create_metric_instruments(meter: metrics.Meter, group: str) -> None:
    """Materialise ``group``'s instruments on ``meter`` without recording anything.

    Required for a group containing any ``observable_*`` spec, and a no-op to
    call twice. Synchronous instruments are created lazily on the first
    :func:`record_metrics`, but an observable one is never recorded against, so
    without this call it would never be created and its callback would never
    run. Call it once, after ``setup_telemetry`` has produced the meter.

    Nothing here raises into the caller, on the same grounds as
    :func:`record_metrics`: an unregistered group and a failed instrument
    creation are logged once and skipped.

    Note:
        OTel offers no way to remove an observable instrument from a meter, so
        :func:`unregister_metric_group` cannot stop a callback that is already
        installed. A consumer that needs a callback to go quiet should return no
        observations from it rather than unregistering the group.

    Args:
        meter: The meter to create the instruments on (e.g. ``handle.meter``).
        group: A group previously passed to :func:`register_metric_group`.
    """
    with _REGISTRY_LOCK:
        entry = _REGISTRY.get(group)
    if entry is None:
        _warn_once(
            ("unregistered", group),
            "create_metric_instruments called for unregistered group %r; call "
            "register_metric_group first.",
            group,
        )
        return
    _instruments_for(meter, group, entry)


def record_metrics(
    meter: metrics.Meter,
    group: str,
    values: Mapping[str, float] | None = None,
    /,
    *,
    attributes: Mapping[str, object] | None = None,
    **kwargs: float | None,
) -> None:
    """Record values against a registered group.

    Values may be passed positionally as a mapping, as keyword arguments, or
    both (keywords win on conflict). Keys must be group ``MetricSpec`` keys;
    ``None`` values are skipped, so a caller records only the series it has.

    Nothing here raises into the caller: an unknown group, an unknown key, a
    malformed ``values`` or ``attributes`` argument, an unusable ``meter``, and a
    failed instrument create or emit are all logged once and skipped. This is an
    instrumentation path and must never break the caller.

    ``meter``, ``group`` and ``values`` are positional-only so a consumer is free
    to declare metric keys named ``meter``, ``group`` or ``values`` and pass them
    as keyword arguments without colliding with these parameters. ``attributes``
    is the one reserved keyword: a consumer whose metric key is literally
    ``attributes`` has to pass it inside the ``values`` mapping, and passing it as
    a keyword instead is logged and dropped rather than raised.

    Args:
        meter: The meter to create/emit instruments on (e.g. ``handle.meter``).
        group: A group previously passed to :func:`register_metric_group`.
        values: Optional ``{key: value}`` mapping.
        attributes: Optional attributes attached to every emitted point.
        **kwargs: ``key=value`` pairs, merged over ``values``.
    """
    try:
        merged: dict[str, float | None] = {}
        if values:
            merged.update(values)
        merged.update(kwargs)
    except (TypeError, ValueError):
        _warn_once(
            ("bad_values", group),
            "record_metrics got a `values` argument of type %s for group %r; it must be "
            "a mapping. Skipping.",
            type(values).__name__,
            group,
        )
        return

    attrs: dict[str, object] | None = None
    if attributes is not None:
        try:
            attrs = dict(attributes) or None
        except (TypeError, ValueError):
            _warn_once(
                ("bad_attributes", group),
                "record_metrics got an `attributes` argument of type %s for group %r; it "
                "must be a mapping, so this call records without attributes. Note that "
                "`attributes` is reserved: a metric key of that name has to be passed "
                "inside the `values` mapping.",
                type(attributes).__name__,
                group,
            )

    if not merged:
        return

    with _REGISTRY_LOCK:
        entry = _REGISTRY.get(group)
    if entry is None:
        _warn_once(
            ("unregistered", group),
            "record_metrics called for unregistered group %r; call register_metric_group first.",
            group,
        )
        return

    instruments = _instruments_for(meter, group, entry)
    if instruments is None:
        return

    for key, value in merged.items():
        if value is None:
            continue
        spec = entry.specs.get(key)
        if spec is None:
            _warn_once(
                ("unknown_key", group, key),
                "Unknown metric key %r for group %r; skipping.",
                key,
                group,
            )
            continue
        if spec.kind in _OBSERVABLE_KIND_FACTORIES:
            # Reported by its callback, so there is nothing to push. Warned
            # about rather than ignored: the value would otherwise vanish with
            # no sign that the caller had the wrong idea of how the series works.
            _warn_once(
                ("observable_key", group, key),
                "Metric key %r in group %r is %s, which reports through its callback; "
                "the value passed to record_metrics is dropped.",
                key,
                group,
                spec.kind,
            )
            continue
        _, emit = _KIND_METHODS[spec.kind]
        try:
            getattr(instruments[key], emit)(value, attributes=attrs)
        except Exception:
            _warn_once(
                ("record", group, key),
                "Failed to record metric %r in group %r",
                key,
                group,
                exc_info=True,
            )


# --------------------------------------------------------------------------- #
# Fork safety
# --------------------------------------------------------------------------- #
# A threading.Lock held at os.fork() is copied into the child in a locked state
# with no thread left to release it, so the next acquire in the child deadlocks.
# Guard both module locks the way CPython's logging module guards its own: hold
# them across the fork (so the child inherits a consistent registry) and replace
# them with fresh, unlocked Locks in the child. Registered once at import.


def _acquire_locks_before_fork() -> None:
    _REGISTRY_LOCK.acquire()
    _WARN_LOCK.acquire()


def _release_locks_after_fork_in_parent() -> None:
    _WARN_LOCK.release()
    _REGISTRY_LOCK.release()


def _reinit_locks_after_fork_in_child() -> None:
    # The inherited locks are locked and unowned; discard them for fresh ones.
    # Registry contents carried over from the parent are intentionally kept.
    global _REGISTRY_LOCK, _WARN_LOCK
    _REGISTRY_LOCK = threading.Lock()
    _WARN_LOCK = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(
        before=_acquire_locks_before_fork,
        after_in_parent=_release_locks_after_fork_in_parent,
        after_in_child=_reinit_locks_after_fork_in_child,
    )
