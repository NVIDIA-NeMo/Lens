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

"""TelemetryHandle: lifecycle wrapper for the OTel tracer, meter, and logger."""

from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING

from opentelemetry import metrics, trace

if TYPE_CHECKING:
    from nemo.lens.config import NemoLensConfig

_INSTRUMENTATION_SCOPE = "nemo.lens"
_INITIALIZED = False


class TelemetryHandle:
    """Holds an OTel tracer and meter for the current process.

    When telemetry is disabled these are no-op objects.
    Obtain via :func:`setup_telemetry`.
    """

    def __init__(
        self, tracer: trace.Tracer, meter: metrics.Meter, is_exporting: bool = False
    ) -> None:
        self._tracer = tracer
        self._meter = meter
        self.is_exporting = is_exporting
        self._shutdown_done = False

    @property
    def tracer(self) -> trace.Tracer:
        return self._tracer

    @property
    def meter(self) -> metrics.Meter:
        return self._meter

    def shutdown(self, timeout_ms: int = 5000) -> None:
        """Flush pending spans/metrics and shut down providers.

        Idempotent: callers may invoke this from more than one terminal path.
        """
        if self._shutdown_done:
            return
        self._shutdown_done = True
        tracer_provider = trace.get_tracer_provider()
        if hasattr(tracer_provider, "force_flush"):
            tracer_provider.force_flush(timeout_millis=timeout_ms)
        if hasattr(tracer_provider, "shutdown"):
            tracer_provider.shutdown()

        meter_provider = metrics.get_meter_provider()
        if hasattr(meter_provider, "force_flush"):
            meter_provider.force_flush(timeout_millis=timeout_ms)
        if hasattr(meter_provider, "shutdown"):
            meter_provider.shutdown()


def setup_telemetry(
    config: NemoLensConfig,
    *,
    resource_attributes: dict | None = None,
    span_exporter=None,
    metric_reader=None,
    _allow_reinit: bool = False,
) -> TelemetryHandle:
    """Initialise OTel providers and return a TelemetryHandle.

    Single entry point for telemetry initialisation. Call once per process.

    Logic:
    - If disabled: no-op providers, empty span groups.
    - If enabled: real providers with exporters.

    Lens has no notion of rank. A distributed caller that wants per-rank
    identity passes it as a resource attribute::

        from nemo.lens.semconv import NV_DL_RANK, NV_DL_WORLD_SIZE

        setup_telemetry(cfg, resource_attributes={NV_DL_RANK: rank, NV_DL_WORLD_SIZE: ws})

    Omitting ``nv.dl.rank`` logs a warning: without it this process cannot be
    filtered by rank in the collector. A genuinely single-process caller can
    pass rank ``0`` to silence it.

    Restricting export to a subset of ranks is likewise the caller's decision:
    leave ``config.enabled`` false on ranks that should stay quiet, or filter on
    ``nv.dl.rank`` in the collector.

    Everything after ``config`` is keyword-only, deliberately. The removed
    signature was ``(config, rank, world_size, resource_attributes, ...)``, so a
    stale ``setup_telemetry(cfg, 0, 8)`` would otherwise rebind ``0`` to
    ``resource_attributes`` and ``8`` to ``span_exporter``. Keyword-only turns
    every such call site into an immediate ``TypeError``.

    Everything after ``config`` is keyword-only, deliberately. The removed
    signature was ``(config, rank, world_size, resource_attributes, ...)``, so a
    stale ``setup_telemetry(cfg, 0, 8)`` would otherwise rebind ``0`` to
    ``resource_attributes`` and ``8`` to ``span_exporter`` -- yielding a handle
    that reports ``is_exporting=True``, exports nothing, and still exits zero.
    Keyword-only turns every such call site into an immediate ``TypeError``.

    Args:
        config: Telemetry configuration.
        resource_attributes: Extra resource attributes.
        span_exporter: Optional custom span exporter (bypasses config-based exporter).
        metric_reader: Optional custom metric reader (bypasses config-based reader).

    Returns:
        A TelemetryHandle with ``.tracer`` and ``.meter``.
    """
    global _INITIALIZED

    if _INITIALIZED and config.enabled and not _allow_reinit:
        raise RuntimeError(
            "setup_telemetry() has already been initialised for this process. "
            "Call it once at startup. Pass _allow_reinit=True to override (testing only)."
        )

    from nemo.lens.providers import build_noop_providers, build_providers
    from nemo.lens.state import set_enabled_span_groups, set_span_group_spec

    # Auto-generate run_id if not explicitly set.
    if not config.run_id:
        slurm_id = os.environ.get("SLURM_JOB_ID", "")
        config.run_id = slurm_id if slurm_id else uuid.uuid4().hex[:12]

    if config.enabled:
        build_providers(
            config,
            resource_attributes=resource_attributes,
            span_exporter=span_exporter,
            metric_reader=metric_reader,
        )
        set_span_group_spec(config.span_groups)
        _INITIALIZED = True
    else:
        build_noop_providers()
        set_enabled_span_groups(frozenset())

    tracer = trace.get_tracer(_INSTRUMENTATION_SCOPE)
    meter = metrics.get_meter(_INSTRUMENTATION_SCOPE)
    return TelemetryHandle(tracer=tracer, meter=meter, is_exporting=config.enabled)
