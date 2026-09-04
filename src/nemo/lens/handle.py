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
    from nemo.lens.strategies import ExportStrategy

_INSTRUMENTATION_SCOPE = "nemo.lens"
_INITIALIZED = False


class TelemetryHandle:
    """Holds an OTel tracer and meter for the current process.

    On non-exporting ranks (or when disabled) these are no-op objects.
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


def _should_export(
    config: NemoLensConfig,
    rank: int,
    world_size: int,
    override: ExportStrategy | None = None,
) -> bool:
    """Determine if this rank should export telemetry data.

    If ``override`` is supplied, it is called directly. Otherwise the strategy
    named by ``config.export_strategy`` is looked up in the registry.
    """
    from nemo.lens.strategies import get_export_strategy

    strategy = override if override is not None else get_export_strategy(config.export_strategy)
    return strategy(config, rank, world_size)


def setup_telemetry(
    config: NemoLensConfig,
    rank: int = 0,
    world_size: int = 1,
    resource_attributes: dict | None = None,
    span_exporter=None,
    metric_reader=None,
    export_strategy: ExportStrategy | None = None,
    _allow_reinit: bool = False,
) -> TelemetryHandle:
    """Initialise OTel providers and return a TelemetryHandle.

    Single entry point for telemetry initialisation. Call once per process.

    Logic:
    - If disabled: no-op providers, empty span groups.
    - If enabled + exporting rank: real providers with exporters.
    - If enabled + non-exporting rank: no-op providers, empty span groups.

    Args:
        config: Telemetry configuration.
        rank: This process's global rank.
        world_size: Total number of processes.
        resource_attributes: Extra resource attributes.
        span_exporter: Optional custom span exporter (bypasses config-based exporter).
        metric_reader: Optional custom metric reader (bypasses config-based reader).
        export_strategy: Optional callable ``(config, rank, world_size) -> bool``
            that bypasses the registry-based dispatch. Useful for ad-hoc
            strategies without registering globally.

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

    is_export_rank = _should_export(config, rank, world_size, override=export_strategy)

    if not config.enabled:
        build_noop_providers()
        set_enabled_span_groups(frozenset())
        _is_exporting = False
    elif is_export_rank:
        build_providers(
            config,
            rank,
            world_size,
            resource_attributes,
            span_exporter=span_exporter,
            metric_reader=metric_reader,
        )
        set_span_group_spec(config.span_groups)
        _is_exporting = True
    else:
        build_noop_providers()
        set_enabled_span_groups(frozenset())
        _is_exporting = False

    if config.enabled:
        _INITIALIZED = True

    tracer = trace.get_tracer(_INSTRUMENTATION_SCOPE)
    meter = metrics.get_meter(_INSTRUMENTATION_SCOPE)
    return TelemetryHandle(tracer=tracer, meter=meter, is_exporting=_is_exporting)
