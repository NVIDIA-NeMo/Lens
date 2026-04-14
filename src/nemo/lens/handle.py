# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""TelemetryHandle: lifecycle wrapper for the OTel tracer, meter, and logger."""

from __future__ import annotations

import hashlib
import os
import uuid
from typing import TYPE_CHECKING

from opentelemetry import metrics, trace

if TYPE_CHECKING:
    from nemo.lens.config import NemoLensConfig

_INSTRUMENTATION_SCOPE = "nemo.lens"


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

    @property
    def tracer(self) -> trace.Tracer:
        return self._tracer

    @property
    def meter(self) -> metrics.Meter:
        return self._meter

    def shutdown(self, timeout_ms: int = 5000) -> None:
        """Flush pending spans/metrics and shut down providers."""
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
) -> bool:
    """Determine if this rank should export telemetry data."""
    if config.export_strategy == "all_ranks":
        return True
    elif config.export_strategy == "sampled":
        # Deterministic hash-based sampling
        h = hashlib.md5(str(rank).encode()).hexdigest()
        return (int(h, 16) % 10000) / 10000.0 < config.export_sample_rate
    else:
        # single_rank (default)
        resolved = config.export_rank if config.export_rank >= 0 else (world_size - 1)
        return rank == resolved


def setup_telemetry(
    config: NemoLensConfig,
    rank: int = 0,
    world_size: int = 1,
    resource_attributes: dict | None = None,
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

    Returns:
        A TelemetryHandle with ``.tracer`` and ``.meter``.
    """
    from nemo.lens.providers import build_noop_providers, build_providers
    from nemo.lens.state import set_enabled_span_groups

    # Auto-generate run_id if not explicitly set.
    if not config.run_id:
        slurm_id = os.environ.get("SLURM_JOB_ID", "")
        config.run_id = slurm_id if slurm_id else uuid.uuid4().hex[:12]

    is_export_rank = _should_export(config, rank, world_size)

    if not config.enabled:
        build_noop_providers()
        set_enabled_span_groups(frozenset())
        _is_exporting = False
    elif is_export_rank:
        build_providers(config, rank, world_size, resource_attributes)
        set_enabled_span_groups(config.resolved_span_groups)
        _is_exporting = True
    else:
        build_noop_providers()
        set_enabled_span_groups(frozenset())
        _is_exporting = False

    tracer = trace.get_tracer(_INSTRUMENTATION_SCOPE)
    meter = metrics.get_meter(_INSTRUMENTATION_SCOPE)
    return TelemetryHandle(tracer=tracer, meter=meter, is_exporting=_is_exporting)
