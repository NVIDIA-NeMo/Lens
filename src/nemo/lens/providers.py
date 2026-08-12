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

"""Internal: build TracerProvider + MeterProvider + LoggerProvider.

Only imported on exporting ranks when telemetry is enabled. All heavy SDK
imports live here; ``opentelemetry-api`` (no-op) is the only dependency for
code paths that never reach this module.
"""

from __future__ import annotations

import logging
import os
import random
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nemo.lens.config import NemoLensConfig


class _OpenSpanCloser:
    """Span processor that ends still-open spans on provider shutdown.

    ``BatchSpanProcessor`` only emits a span on ``on_end``, so a span still open
    when the process exits is never exported -- which silently drops exactly the
    long-lived spans that wrap a whole run. This tracks in-flight spans and ends
    the leftovers on shutdown.

    Must be registered BEFORE the ``BatchSpanProcessor``: processors shut down in
    registration order, so ending a straggler here still reaches a live batch
    processor via ``on_end``, and its own shutdown then flushes the queue.

    ``force_flush`` is deliberately a no-op -- it runs mid-run to push already
    ended spans and must not terminate live ones.
    """

    def __init__(self) -> None:
        # Keyed by span_id, not the span object: on_start receives the live _Span
        # but on_end a ReadableSpan snapshot, so an object-keyed set would never
        # discard.
        #
        # The reference is deliberately strong. A span nothing else refers to is
        # precisely the one the application can no longer end itself, so it is
        # what this exists to close. Holding it keeps it out of the collector
        # until shutdown (~3 KB each), which only accumulates if a caller leaks
        # spans -- and then the leak becomes visible in the trace rather than
        # silently disappearing.
        self._open: dict = {}
        self._lock = threading.Lock()
        self._closed = False

    @staticmethod
    def _span_id(span) -> int | None:
        ctx = span.context
        return ctx.span_id if ctx is not None else None

    def on_start(self, span, parent_context=None) -> None:
        sid = self._span_id(span)
        if sid is None:
            return
        with self._lock:
            self._open[sid] = span

    def on_end(self, span) -> None:
        sid = self._span_id(span)
        if sid is None:
            return
        with self._lock:
            self._open.pop(sid, None)

    def _on_ending(self, span) -> None:
        # Required under duck-typing: SynchronousMultiSpanProcessor._on_ending
        # fans out to every processor without a hasattr guard.
        return

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    def shutdown(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            leftovers = list(self._open.values())
            self._open.clear()
        # Innermost first, so a child never outlives the parent it is nested in.
        for sp in reversed(leftovers):
            try:
                sp.end()
            except Exception:  # telemetry must never break the exit path
                logging.getLogger(__name__).debug("Failed to end span on shutdown", exc_info=True)


class SeedIndependentIdGenerator:
    """OTel-compatible IdGenerator whose IDs survive a global ``random.seed()``.

    OTel's default ``RandomIdGenerator`` draws from the process-global ``random``
    module, which training frameworks (Megatron) seed IDENTICALLY across
    data-parallel ranks -> every rank would emit the SAME span/trace IDs, so a
    backend sees many spans sharing one span ID and parent links resolve to the
    wrong span. A private ``random.Random`` instance is seeded from the OS at
    construction and is unaffected by ``random.seed()`` on the global module.

    Duck-typed rather than subclassing ``opentelemetry.sdk.trace.id_generator.
    IdGenerator`` (same as :class:`~nemo.lens.sampling.RankAwareSampler`) so this
    module stays importable without the SDK installed.
    """

    def __init__(self) -> None:
        self._rng = random.Random()  # seeded from os.urandom, not from random.seed()
        # CPython reseeds only the GLOBAL random module at fork; a private Random()
        # gets no such hook, so forked children (dataloader workers under the default
        # "fork" start method, multiprocessing.Pool) would inherit our state and emit
        # identical IDs -- the same collision this class exists to prevent.
        os.register_at_fork(after_in_child=self._rng.seed)

    def generate_span_id(self) -> int:
        return self._rng.getrandbits(64) or 1

    def generate_trace_id(self) -> int:
        return self._rng.getrandbits(128) or 1

    def is_trace_id_random(self) -> bool:
        """Declares the W3C ``random-trace-id`` trace flag (Trace Context Level 2)."""
        return True


def build_providers(
    config: NemoLensConfig,
    rank: int = 0,
    world_size: int = 1,
    resource_attributes: dict | None = None,
    span_exporter=None,
    metric_reader=None,
) -> None:
    """Initialise TracerProvider, MeterProvider, and optionally LoggerProvider.

    Imports the OTel SDK. Raises ImportError if not installed.

    Args:
        config: Telemetry configuration.
        rank: Current process rank.
        world_size: Total number of ranks.
        resource_attributes: Extra resource attributes to merge.
        span_exporter: Optional custom span exporter (bypasses config-based exporter).
        metric_reader: Optional custom metric reader (bypasses config-based reader).
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
    from nemo.lens.package_info import __version__

    attrs = {
        "service.name": config.service_name,
        "service.version": __version__,
        "dl.rank": rank,
        "dl.world_size": world_size,
    }
    # Run identification — shared across all ranks in a job.
    if config.run_id:
        attrs["nemo.run.id"] = config.run_id
        attrs["service.instance.id"] = f"{config.run_id}-rank{rank}"
    if config.user:
        attrs["nemo.user.id"] = config.user
    # W&B Weave resource attributes (required when exporting to Weave).
    if config.wandb_entity:
        attrs["wandb.entity"] = config.wandb_entity
    if config.wandb_project:
        attrs["wandb.project"] = config.wandb_project
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

        _span_exporter = span_exporter or _build_span_exporter(config)

        # Seed-independent IDs cover EVERY setup_telemetry caller -- trainer, ckpt worker, nvrx --
        # since they all build their TracerProvider here (the worker/nvrx set telemetry up in their
        # own process via from_env, so a caller-side patch would miss them; fixing it here does not).
        kwargs: dict = {"resource": resource, "id_generator": SeedIndependentIdGenerator()}
        if config.sampler_enabled:
            from nemo.lens.sampling import RankAwareSampler

            kwargs["sampler"] = RankAwareSampler(
                rank=rank, world_size=world_size, sample_rate=config.export_sample_rate
            )

        tracer_provider = TracerProvider(**kwargs)
        # Order matters: the closer is registered BEFORE the batch processor so its
        # shutdown() runs first and ends any still-open spans -> they flow into the
        # batch queue via on_end -> the batch shutdown then flushes them out.
        tracer_provider.add_span_processor(_OpenSpanCloser())
        tracer_provider.add_span_processor(BatchSpanProcessor(_span_exporter))
        trace.set_tracer_provider(tracer_provider)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    if config.metrics_enabled:
        from opentelemetry import metrics
        from opentelemetry.sdk.metrics import MeterProvider

        if metric_reader is not None:
            _reader = metric_reader
        else:
            from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

            metric_exporter = _build_metric_exporter(config)
            _export_interval = int(os.environ.get("OTEL_METRIC_EXPORT_INTERVAL", "10000"))
            _reader = PeriodicExportingMetricReader(
                metric_exporter, export_interval_millis=_export_interval
            )
        meter_provider = MeterProvider(resource=resource, metric_readers=[_reader])
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


def _resolve_otlp_protocol(signal: str) -> str:
    """Resolve the OTLP wire protocol for the given signal ('traces' / 'metrics' / 'logs').

    Honours the OTel SDK convention: ``OTEL_EXPORTER_OTLP_<SIGNAL>_PROTOCOL``
    takes precedence over ``OTEL_EXPORTER_OTLP_PROTOCOL``. Defaults to ``"grpc"``
    when neither is set, matching the OTel SDK's default.

    Recognised values: ``grpc``, ``http/protobuf``, ``http/json``.
    """
    signal_specific = os.environ.get(f"OTEL_EXPORTER_OTLP_{signal.upper()}_PROTOCOL")
    general = os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL")
    return (signal_specific or general or "grpc").strip().lower()


def _build_span_exporter(config: NemoLensConfig):
    if config.exporter not in _VALID_EXPORTERS:
        raise ValueError(
            f"Unknown exporter type: {config.exporter!r}. Expected one of: {_VALID_EXPORTERS}"
        )

    if config.exporter == "console":
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter

        return ConsoleSpanExporter()

    protocol = _resolve_otlp_protocol("traces")
    prefer_http = protocol in ("http/protobuf", "http/json")

    if prefer_http:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            return OTLPSpanExporter()
        except ImportError:
            pass  # fall through to gRPC

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

    protocol = _resolve_otlp_protocol("metrics")
    prefer_http = protocol in ("http/protobuf", "http/json")

    if prefer_http:
        try:
            from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter

            return OTLPMetricExporter()
        except ImportError:
            pass  # fall through to gRPC

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
