# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Internal: build TracerProvider + MeterProvider + LoggerProvider.

Only imported on exporting ranks when telemetry is enabled. All heavy SDK
imports live here; ``opentelemetry-api`` (no-op) is the only dependency for
code paths that never reach this module.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nemo.lens.config import NemoLensConfig


def build_providers(
    config: NemoLensConfig,
    rank: int = 0,
    world_size: int = 1,
    resource_attributes: dict | None = None,
) -> None:
    """Initialise TracerProvider, MeterProvider, and optionally LoggerProvider.

    Imports the OTel SDK. Raises ImportError if not installed.

    Args:
        config: Telemetry configuration.
        rank: Current process rank.
        world_size: Total number of ranks.
        resource_attributes: Extra resource attributes to merge.
    """
    try:
        from opentelemetry.sdk.resources import Resource
    except ImportError as exc:
        raise ImportError(
            "OpenTelemetry SDK is required for telemetry export but is not installed. "
            "Install with: pip install 'nemo-lens[sdk]'"
        ) from exc

    # ------------------------------------------------------------------
    # Resource
    # ------------------------------------------------------------------
    from nemo.lens._version import __version__

    attrs = {
        "service.name": config.service_name,
        "service.version": __version__,
        "dl.rank": rank,
        "dl.world_size": world_size,
    }
    env_name = os.environ.get("DEPLOYMENT_ENV", os.environ.get("ENVIRONMENT", ""))
    if env_name:
        attrs["deployment.environment"] = env_name
    if resource_attributes:
        attrs.update(resource_attributes)

    # Detect deployment environment
    from nemo.lens.resources import detect_resource

    detected = detect_resource()
    attrs.update(detected)

    resource = Resource.create(attrs)

    # ------------------------------------------------------------------
    # Traces
    # ------------------------------------------------------------------
    if config.traces_enabled:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        span_exporter = _build_span_exporter(config)
        tracer_provider = TracerProvider(resource=resource)
        tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
        trace.set_tracer_provider(tracer_provider)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    if config.metrics_enabled:
        from opentelemetry import metrics
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

        metric_exporter = _build_metric_exporter(config)
        _export_interval = int(os.environ.get("OTEL_METRIC_EXPORT_INTERVAL", "10000"))
        reader = PeriodicExportingMetricReader(
            metric_exporter, export_interval_millis=_export_interval
        )
        meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
        metrics.set_meter_provider(meter_provider)

    # ------------------------------------------------------------------
    # Logs (optional)
    # ------------------------------------------------------------------
    if config.logs_enabled:
        _setup_log_provider(config, resource)

    # ------------------------------------------------------------------
    # Propagator (W3C TraceContext + Baggage)
    # ------------------------------------------------------------------
    _set_propagator()


def build_noop_providers() -> None:
    """Register no-op providers for non-exporting ranks or disabled telemetry."""
    from opentelemetry import metrics, trace
    from opentelemetry.metrics import NoOpMeterProvider
    from opentelemetry.trace import NoOpTracerProvider

    trace.set_tracer_provider(NoOpTracerProvider())
    metrics.set_meter_provider(NoOpMeterProvider())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_VALID_EXPORTERS = ("otlp", "console")


def _build_span_exporter(config: NemoLensConfig):
    if config.exporter not in _VALID_EXPORTERS:
        raise ValueError(
            f"Unknown exporter type: {config.exporter!r}. Expected one of: {_VALID_EXPORTERS}"
        )

    if config.exporter == "console":
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter

        return ConsoleSpanExporter()

    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        return OTLPSpanExporter()
    except ImportError:
        pass
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        return OTLPSpanExporter()
    except ImportError:
        pass
    raise ImportError("No OTLP span exporter found. Install with: pip install 'nemo-lens[sdk]'")


def _build_metric_exporter(config: NemoLensConfig):
    if config.exporter not in _VALID_EXPORTERS:
        raise ValueError(
            f"Unknown exporter type: {config.exporter!r}. Expected one of: {_VALID_EXPORTERS}"
        )

    if config.exporter == "console":
        from opentelemetry.sdk.metrics.export import ConsoleMetricExporter

        return ConsoleMetricExporter()

    try:
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

        return OTLPMetricExporter()
    except ImportError:
        pass
    try:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter

        return OTLPMetricExporter()
    except ImportError:
        pass
    raise ImportError("No OTLP metric exporter found. Install with: pip install 'nemo-lens[sdk]'")


def _set_propagator() -> None:
    """Set W3C TraceContext + Baggage as the global text map propagator."""
    from opentelemetry import propagate
    from opentelemetry.baggage.propagation import W3CBaggagePropagator
    from opentelemetry.propagators.composite import CompositePropagator
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

    propagate.set_global_textmap(
        CompositePropagator([TraceContextTextMapPropagator(), W3CBaggagePropagator()])
    )


def _setup_log_provider(config: NemoLensConfig, resource) -> None:
    """Set up the OTel LoggerProvider for log bridging."""
    try:
        from opentelemetry._logs import set_logger_provider
        from opentelemetry.sdk._logs import LoggerProvider
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor

        if config.exporter == "console":
            from opentelemetry.sdk._logs.export import ConsoleLogExporter

            exporter = ConsoleLogExporter()
        else:
            try:
                from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter

                exporter = OTLPLogExporter()
            except ImportError:
                from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter

                exporter = OTLPLogExporter()

        logger_provider = LoggerProvider(resource=resource)
        logger_provider.add_log_record_processor(BatchLogRecordProcessor(exporter))
        set_logger_provider(logger_provider)
    except ImportError:
        pass
