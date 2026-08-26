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
"""

from __future__ import annotations

import logging
import threading
import weakref
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from opentelemetry import metrics

_logger = logging.getLogger(__name__)

#: How each supported kind is created on a ``Meter`` and how a value is emitted.
#: Maps ``kind`` -> (meter factory method, instrument emit method).
_KIND_METHODS: dict[str, tuple[str, str]] = {
    "gauge": ("create_gauge", "set"),
    "histogram": ("create_histogram", "record"),
    "counter": ("create_counter", "add"),
    "up_down_counter": ("create_up_down_counter", "add"),
}

#: Kinds a :class:`MetricSpec` may declare.
METRIC_KINDS: frozenset[str] = frozenset(_KIND_METHODS)


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
            ``up_down_counter`` for an additive series.
        unit: UCUM unit string (e.g. ``ms``, ``{token}/s``). Empty when unitless.
        description: Human-readable description attached to the instrument.
    """

    key: str
    name: str
    kind: str = "gauge"
    unit: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("MetricSpec.key must be a non-empty string.")
        if not self.name:
            raise ValueError(f"MetricSpec.name must be a non-empty string (key={self.key!r}).")
        if self.kind not in _KIND_METHODS:
            raise ValueError(
                f"MetricSpec.kind {self.kind!r} is not one of {sorted(_KIND_METHODS)} "
                f"(key={self.key!r})."
            )


@dataclass
class _Group:
    specs: dict[str, MetricSpec]
    # Instruments materialised per meter; entries drop when the meter is GC'd.
    instruments: weakref.WeakKeyDictionary = field(default_factory=weakref.WeakKeyDictionary)


_REGISTRY: dict[str, _Group] = {}
_REGISTRY_LOCK = threading.Lock()


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
    """Return this group's instruments for ``meter``, creating them once."""
    instruments = entry.instruments.get(meter)
    if instruments is None:
        try:
            instruments = {}
            for key, spec in entry.specs.items():
                factory, _ = _KIND_METHODS[spec.kind]
                instruments[key] = getattr(meter, factory)(
                    name=spec.name,
                    unit=spec.unit,
                    description=spec.description,
                )
        except Exception:
            _logger.warning(
                "Failed to create instruments for metric group %r", group, exc_info=True
            )
            return None
        entry.instruments[meter] = instruments
    return instruments


def record_metrics(
    meter: metrics.Meter,
    group: str,
    values: Mapping[str, float] | None = None,
    *,
    attributes: Mapping[str, object] | None = None,
    **kwargs: float | None,
) -> None:
    """Record values against a registered group.

    Values may be passed positionally as a mapping, as keyword arguments, or
    both (keywords win on conflict). Keys must be group ``MetricSpec`` keys;
    ``None`` values are skipped, so a caller records only the series it has.

    Unknown groups and unknown keys are logged and skipped rather than raised:
    this is an instrumentation path and must never break the caller.

    Args:
        meter: The meter to create/emit instruments on (e.g. ``handle.meter``).
        group: A group previously passed to :func:`register_metric_group`.
        values: Optional ``{key: value}`` mapping.
        attributes: Optional attributes attached to every emitted point.
        **kwargs: ``key=value`` pairs, merged over ``values``.
    """
    merged: dict[str, float | None] = {}
    if values:
        merged.update(values)
    merged.update(kwargs)
    if not merged:
        return

    with _REGISTRY_LOCK:
        entry = _REGISTRY.get(group)
    if entry is None:
        _logger.warning(
            "record_metrics called for unregistered group %r; call register_metric_group first.",
            group,
        )
        return

    instruments = _instruments_for(meter, group, entry)
    if instruments is None:
        return

    attrs = dict(attributes) if attributes else None
    for key, value in merged.items():
        if value is None:
            continue
        spec = entry.specs.get(key)
        if spec is None:
            _logger.warning("Unknown metric key %r for group %r; skipping.", key, group)
            continue
        _, emit = _KIND_METHODS[spec.kind]
        try:
            getattr(instruments[key], emit)(value, attributes=attrs)
        except Exception:
            _logger.warning("Failed to record metric %r in group %r", key, group, exc_info=True)
