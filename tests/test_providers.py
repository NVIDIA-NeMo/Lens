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

"""Unit tests for provider construction."""

import io
import json
import logging
import os
import signal
import threading
import time

import pytest
from opentelemetry import metrics, trace
from opentelemetry.metrics import NoOpMeterProvider
from opentelemetry.trace import NoOpTracerProvider

from nemo.lens.config import NemoLensConfig
from nemo.lens.providers import (
    SeedIndependentIdGenerator,
    _OpenSpanCloser,
    build_noop_providers,
    build_providers,
)
from nemo.lens.semconv import NEMO_SPAN_TRUNCATED, NV_DL_RANK, NV_DL_WORLD_SIZE, SLURM_JOB_ID


class TestBuildNoopProviders:
    def test_sets_noop_tracer_provider(self):
        build_noop_providers()
        provider = trace.get_tracer_provider()
        assert isinstance(provider, NoOpTracerProvider)

    def test_sets_noop_meter_provider(self):
        build_noop_providers()
        provider = metrics.get_meter_provider()
        assert isinstance(provider, NoOpMeterProvider)


class TestBuildProviders:
    def test_console_exporter(self):
        cfg = NemoLensConfig(enabled=True, exporter="console")
        build_providers(cfg, rank=0, world_size=1)
        # Should not raise; tracer provider should be SDK type
        tracer = trace.get_tracer("test")
        with tracer.start_as_current_span("test") as span:
            assert span is not None

    def test_invalid_exporter_raises(self):
        cfg = NemoLensConfig(enabled=True, exporter="invalid")
        with pytest.raises(ValueError, match="Unknown exporter"):
            build_providers(cfg, rank=0, world_size=1)

    def test_resource_attributes_merged(self):
        cfg = NemoLensConfig(enabled=True, exporter="console")
        build_providers(cfg, rank=0, world_size=1, resource_attributes={"custom.attr": "value"})
        # Should not raise
        tracer = trace.get_tracer("test")
        assert tracer is not None

    def test_launch_resource_attributes_override_caller_defaults(self, monkeypatch):
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        monkeypatch.setenv("SLURM_JOB_ID", "12345")
        monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", f"{SLURM_JOB_ID}=launch")

        custom_exporter = InMemorySpanExporter()
        cfg = NemoLensConfig(enabled=True, exporter="console")
        build_providers(
            cfg,
            rank=0,
            world_size=1,
            resource_attributes={SLURM_JOB_ID: "default"},
            span_exporter=custom_exporter,
        )

        tracer = trace.get_tracer("test")
        with tracer.start_as_current_span("resource-check"):
            pass
        trace.get_tracer_provider().force_flush()

        spans = custom_exporter.get_finished_spans()
        assert spans[0].resource.attributes[SLURM_JOB_ID] == "launch"

    def test_rank_resource_attributes_use_v01_names(self, monkeypatch):
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        monkeypatch.delenv("OTEL_RESOURCE_ATTRIBUTES", raising=False)

        exporter = InMemorySpanExporter()
        cfg = NemoLensConfig(enabled=True, exporter="console")
        build_providers(cfg, rank=3, world_size=16, span_exporter=exporter)

        tracer = trace.get_tracer("test")
        with tracer.start_as_current_span("ranked"):
            pass
        trace.get_tracer_provider().force_flush()

        (span,) = exporter.get_finished_spans()
        resource_attrs = span.resource.attributes
        assert resource_attrs[NV_DL_RANK] == 3
        assert resource_attrs[NV_DL_WORLD_SIZE] == 16
        assert "dl.rank" not in resource_attrs
        assert "dl.world_size" not in resource_attrs

    def test_traces_disabled(self):
        cfg = NemoLensConfig(enabled=True, exporter="console", traces_enabled=False)
        build_providers(cfg, rank=0, world_size=1)
        # Tracer should be no-op (not set by us)

    def test_metrics_disabled(self):
        cfg = NemoLensConfig(enabled=True, exporter="console", metrics_enabled=False)
        build_providers(cfg, rank=0, world_size=1)
        # Meter should be no-op (not set by us)


class TestCustomExporters:
    def test_custom_span_exporter(self):
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        custom_exporter = InMemorySpanExporter()
        cfg = NemoLensConfig(enabled=True, exporter="console")
        build_providers(cfg, rank=0, world_size=1, span_exporter=custom_exporter)
        tracer = trace.get_tracer("test")
        with tracer.start_as_current_span("custom") as span:
            span.set_attribute("key", "value")
        trace.get_tracer_provider().force_flush()
        spans = custom_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "custom"

    def test_custom_metric_reader(self):
        from opentelemetry.sdk.metrics.export import InMemoryMetricReader

        reader = InMemoryMetricReader()
        cfg = NemoLensConfig(enabled=True, exporter="console")
        build_providers(cfg, rank=0, world_size=1, metric_reader=reader)
        meter = metrics.get_meter("test")
        counter = meter.create_counter("test.counter")
        counter.add(1)
        data = reader.get_metrics_data()
        assert data is not None


class TestOtlpProtocolSelection:
    """OTEL_EXPORTER_OTLP_PROTOCOL must route between gRPC and HTTP exporters."""

    def test_default_is_grpc(self, monkeypatch):
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as Grpc

        from nemo.lens.providers import _build_span_exporter

        monkeypatch.delenv("OTEL_EXPORTER_OTLP_PROTOCOL", raising=False)
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", raising=False)

        cfg = NemoLensConfig(enabled=True, exporter="otlp")
        assert isinstance(_build_span_exporter(cfg), Grpc)

    def test_http_protobuf_picks_http_exporter(self, monkeypatch):
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as Http

        from nemo.lens.providers import _build_span_exporter

        monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", raising=False)

        cfg = NemoLensConfig(enabled=True, exporter="otlp")
        assert isinstance(_build_span_exporter(cfg), Http)

    def test_signal_specific_overrides_general(self, monkeypatch):
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as Http

        from nemo.lens.providers import _build_span_exporter

        monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", "http/protobuf")

        cfg = NemoLensConfig(enabled=True, exporter="otlp")
        assert isinstance(_build_span_exporter(cfg), Http)

    def test_http_protocol_selects_http_metric_exporter(self, monkeypatch):
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
            OTLPMetricExporter as HttpMetric,
        )

        from nemo.lens.providers import _build_metric_exporter

        monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")

        cfg = NemoLensConfig(enabled=True, exporter="otlp")
        assert isinstance(_build_metric_exporter(cfg), HttpMetric)

    def test_grpc_default_picks_grpc_metric_exporter(self, monkeypatch):
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
            OTLPMetricExporter as GrpcMetric,
        )

        from nemo.lens.providers import _build_metric_exporter

        monkeypatch.delenv("OTEL_EXPORTER_OTLP_PROTOCOL", raising=False)
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_METRICS_PROTOCOL", raising=False)

        cfg = NemoLensConfig(enabled=True, exporter="otlp")
        assert isinstance(_build_metric_exporter(cfg), GrpcMetric)


class TestSeedIndependentIds:
    def test_trace_and_span_ids_survive_identical_random_seed(self):
        """Data-parallel ranks seed Python's `random` identically, which makes OTel's
        default RandomIdGenerator emit the SAME span/trace IDs on every rank. The
        provider must use a seed-independent generator so IDs stay unique."""
        import random

        cfg = NemoLensConfig(enabled=True, exporter="console")
        build_providers(cfg, rank=0, world_size=1)
        id_generator = trace.get_tracer_provider().id_generator

        state = random.getstate()  # don't leak a deterministic global RNG into later tests
        try:
            random.seed(1234)  # what training frameworks do identically across DP ranks
            first = (id_generator.generate_trace_id(), id_generator.generate_span_id())
            random.seed(1234)
            second = (id_generator.generate_trace_id(), id_generator.generate_span_id())
        finally:
            random.setstate(state)

        assert first != second

    def test_ids_are_in_range_and_never_invalid(self):
        gen = SeedIndependentIdGenerator()
        for _ in range(100):
            trace_id = gen.generate_trace_id()
            span_id = gen.generate_span_id()
            assert 0 < trace_id < 2**128
            assert 0 < span_id < 2**64

    def test_declares_random_trace_id(self):
        """W3C Trace Context L2 `random-trace-id` flag: OTel's own generator sets it, and
        dropping it forces downstream consistent-probability sampling onto its fallback."""
        assert SeedIndependentIdGenerator().is_trace_id_random() is True

    def test_spans_carry_the_random_trace_id_flag(self):
        from opentelemetry.trace import TraceFlags

        cfg = NemoLensConfig(enabled=True, exporter="console")
        build_providers(cfg, rank=0, world_size=1)

        tracer = trace.get_tracer("test")
        with tracer.start_as_current_span("root") as span:
            assert span.get_span_context().trace_flags & TraceFlags.RANDOM_TRACE_ID

    @pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork()")
    def test_ids_differ_across_forked_children(self):
        """CPython reseeds only the GLOBAL random module at fork, so a private
        random.Random() would hand every forked child (dataloader workers, Pool)
        the same state and the same IDs -- the collision this generator prevents."""
        gen = SeedIndependentIdGenerator()

        read_fds = []
        for _ in range(3):
            read_fd, write_fd = os.pipe()
            if os.fork() == 0:  # child
                try:
                    os.close(read_fd)
                    os.write(write_fd, f"{gen.generate_trace_id():032x}".encode())
                finally:
                    os._exit(0)  # never unwind pytest's stack in the child
            os.close(write_fd)
            read_fds.append(read_fd)

        ids = []
        for read_fd in read_fds:
            ids.append(os.read(read_fd, 32).decode())
            os.close(read_fd)
        for _ in read_fds:
            os.wait()

        assert len(set(ids)) == len(ids), f"forked children shared trace IDs: {ids}"


class TestOpenSpanCloser:
    """A span left open when the process exits must still be exported.

    ``BatchSpanProcessor`` emits only on ``on_end``, so without this processor a
    span that is never ended is never exported at all.
    """

    @staticmethod
    def _provider(exporter, closer=None):
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        provider = TracerProvider(shutdown_on_exit=False)
        provider.add_span_processor(closer if closer is not None else _OpenSpanCloser())
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        return provider

    def test_open_span_is_exported_on_shutdown(self):
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        exporter = InMemorySpanExporter()
        provider = self._provider(exporter)
        span = provider.get_tracer("test").start_span("never_ended")  # still in scope
        assert span.end_time is None

        assert exporter.get_finished_spans() == ()
        provider.shutdown()

        assert [s.name for s in exporter.get_finished_spans()] == ["never_ended"]

    def test_force_flush_does_not_end_open_spans(self):
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        exporter = InMemorySpanExporter()
        provider = self._provider(exporter)
        span = provider.get_tracer("test").start_span("still_running")

        provider.force_flush()

        assert exporter.get_finished_spans() == ()
        assert span.end_time is None

    def test_already_ended_spans_are_not_re_exported(self):
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        exporter = InMemorySpanExporter()
        provider = self._provider(exporter)
        provider.get_tracer("test").start_span("done").end()
        provider.shutdown()

        assert [s.name for s in exporter.get_finished_spans()] == ["done"]

    def test_children_are_ended_before_their_parents(self):
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        provider = self._provider(InMemorySpanExporter())
        tracer = provider.get_tracer("test")
        parent = tracer.start_span("parent")
        child = tracer.start_span("child", context=trace.set_span_in_context(parent))
        provider.shutdown()

        assert child.end_time <= parent.end_time

    def test_abandoned_spans_are_still_closed(self):
        """A span the caller dropped without ending is the main thing this catches.

        Nothing else holds it, so the application can no longer end it; closing it
        at shutdown is what keeps the work it recorded (and the leak itself)
        visible instead of silently dropping both.
        """
        import gc

        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        exporter = InMemorySpanExporter()
        provider = self._provider(exporter)
        tracer = provider.get_tracer("test")

        for i in range(10):
            tracer.start_span(f"abandoned_{i}")  # dropped immediately, never ended
        gc.collect()
        provider.shutdown()

        assert len(exporter.get_finished_spans()) == 10

    def test_ended_spans_are_discarded_from_the_open_set(self):
        """on_end must discard by span_id, not object identity.

        on_start receives the live ``_Span``; on_end receives a ``ReadableSpan``
        snapshot of it. Keyed by ``id(span)`` nothing would ever be discarded, so
        every already-ended span would be swept -- and re-ended -- at shutdown.
        """
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        closer = _OpenSpanCloser()
        provider = self._provider(InMemorySpanExporter(), closer=closer)
        provider.get_tracer("test").start_span("done").end()

        assert closer._open == {}

    def test_force_closed_span_is_marked_truncated(self):
        """Its end time is the shutdown time, not when the work finished.

        For a span abandoned early in a long run the duration is wrong by hours;
        a consumer needs to be able to tell that apart from a real measurement.
        """
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        exporter = InMemorySpanExporter()
        provider = self._provider(exporter)
        provider.get_tracer("test").start_span("never_ended")
        provider.shutdown()

        (span,) = exporter.get_finished_spans()
        assert span.attributes.get(NEMO_SPAN_TRUNCATED) is True
        assert [event.name for event in span.events] == ["nemo.span.truncated"]

    def test_normally_ended_span_is_not_marked_truncated(self):
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        exporter = InMemorySpanExporter()
        provider = self._provider(exporter)
        provider.get_tracer("test").start_span("done").end()
        provider.shutdown()

        (span,) = exporter.get_finished_spans()
        assert NEMO_SPAN_TRUNCATED not in (span.attributes or {})
        assert span.events == ()

    def test_spans_started_after_shutdown_are_not_retained(self):
        """Nothing sweeps ``_open`` after shutdown, so tracking there is a pure leak.

        Reachable through the documented ``finally: handle.shutdown()`` pattern,
        where later work can still start spans.
        """
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        closer = _OpenSpanCloser()
        provider = self._provider(InMemorySpanExporter(), closer=closer)
        provider.shutdown()

        for i in range(1000):
            closer.on_start(_FakeSpan(span_id=i))

        assert closer._open == {}

    def test_second_shutdown_does_not_end_late_spans(self):
        """The ``_closed`` guard, from the other side: a second sweep must be inert."""
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        exporter = InMemorySpanExporter()
        closer = _OpenSpanCloser()
        provider = self._provider(exporter, closer=closer)
        provider.shutdown()

        late = _FakeSpan(span_id=1)
        closer.on_start(late)
        closer.shutdown()

        assert late.ended is False
        assert exporter.get_finished_spans() == ()

    @pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork()")
    def test_forked_child_does_not_re_export_the_parents_open_spans(self):
        """``_open`` crosses fork() carrying the parent's in-flight spans.

        Their span IDs were assigned before the fork, so [[lens-28]]'s ID reseeding
        does not help. The SDK's own batch processor reinstalls itself in the child,
        so its exporter is live: without a fork hook every child would export the
        parent's span again, under one span ID, with a fabricated end time.
        """
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        exporter = InMemorySpanExporter()
        closer = _OpenSpanCloser()
        provider = self._provider(exporter, closer=closer)
        provider.get_tracer("test").start_span("parent_only")
        assert len(closer._open) == 1

        read_fd, write_fd = os.pipe()
        if os.fork() == 0:  # child
            try:
                os.close(read_fd)
                provider.shutdown()
                os.write(write_fd, str(len(exporter.get_finished_spans())).encode())
            finally:
                os._exit(0)  # never unwind pytest's stack in the child
        os.close(write_fd)
        exported_in_child = os.read(read_fd, 8).decode()
        os.close(read_fd)
        os.wait()

        assert exported_in_child == "0", (
            f"forked child re-exported {exported_in_child} of the parent's open spans"
        )

    @pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork()")
    def test_forked_child_is_not_deadlocked_by_an_inherited_lock(self):
        """A lock held by another thread at fork() is inherited HELD, forever.

        The holding thread does not exist in the child, so nothing can release it and
        the child's next on_start blocks for the life of the process. on_start runs on
        whichever thread starts a span, so this is an ordinary race, not a contrived
        one. The SDK recreates ``_export_lock`` in ``_at_fork_reinit`` for this reason.
        """
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        closer = _OpenSpanCloser()
        provider = self._provider(InMemorySpanExporter(), closer=closer)
        tracer = provider.get_tracer("test")

        held = threading.Event()
        release = threading.Event()

        def _hold_the_lock():
            with closer._lock:
                held.set()
                release.wait(30)

        holder = threading.Thread(target=_hold_the_lock, daemon=True)
        holder.start()
        assert held.wait(5), "lock holder thread never started"

        pid = os.fork()
        if pid == 0:  # child
            try:
                tracer.start_span("in_child")  # -> on_start -> acquires the lock
                os._exit(0)
            except BaseException:
                os._exit(1)

        # Watchdog: the child has to exit on its own. A hang here IS the bug.
        deadline = time.monotonic() + 15
        status = None
        while time.monotonic() < deadline:
            reaped, status = os.waitpid(pid, os.WNOHANG)
            if reaped:
                break
            time.sleep(0.05)
        else:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
            release.set()
            holder.join(5)
            pytest.fail("forked child deadlocked on a lock inherited while held")

        release.set()
        holder.join(5)
        assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0


class _FakeSpan:
    """Minimal duck-typed span, for driving the closer directly."""

    def __init__(self, span_id: int):
        self.context = trace.SpanContext(
            trace_id=1, span_id=span_id, is_remote=False, trace_flags=trace.TraceFlags(0x01)
        )
        self.ended = False

    def set_attribute(self, key, value):
        pass

    def add_event(self, name, attributes=None):
        pass

    def end(self):
        self.ended = True


class TestOpenSpanCloserWiring:
    """``build_providers`` must register the closer, and register it FIRST.

    Every other test here builds its own ``TracerProvider`` by hand, so none of them
    assert the wiring that actually ships: deleting the ``add_span_processor`` call,
    or moving it after the ``BatchSpanProcessor``, leaves them all green.
    """

    @staticmethod
    def _build(span_exporter):
        cfg = NemoLensConfig(enabled=True, exporter="console")
        build_providers(cfg, rank=0, world_size=1, span_exporter=span_exporter)
        return trace.get_tracer_provider()

    def test_open_span_reaches_the_exporter_end_to_end(self):
        """The whole fix, through the real wiring: no closer, or a closer registered
        after the batch processor, and this span is never exported."""
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        exporter = InMemorySpanExporter()
        provider = self._build(exporter)
        provider.get_tracer("test").start_span("whole_run")  # never ended
        provider.shutdown()

        assert [s.name for s in exporter.get_finished_spans()] == ["whole_run"]

    def test_closer_is_registered_before_the_batch_processor(self):
        """States the ordering directly, so a reorder fails as a wiring bug rather
        than as a puzzling export miss."""
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        provider = self._build(InMemorySpanExporter())
        processors = provider._active_span_processor._span_processors

        assert isinstance(processors[0], _OpenSpanCloser)
        assert any(isinstance(p, BatchSpanProcessor) for p in processors[1:])


class TestConsoleExporterJsonl:
    """Console exporters must emit real JSONL: one compact JSON object per line.

    The SDK defaults to ``to_json(indent=4)``, which spreads a single record
    over many lines and makes a redirected console export unparseable by
    line-oriented tooling.
    """

    @staticmethod
    def _console_cfg():
        return NemoLensConfig(enabled=True, exporter="console")

    def test_span_export_writes_one_line_per_span(self):
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        from nemo.lens.providers import _build_span_exporter

        exporter = _build_span_exporter(self._console_cfg())
        buf = io.StringIO()
        exporter.out = buf

        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("test")
        for name in ("first", "second"):
            with tracer.start_as_current_span(name) as span:
                span.set_attribute("nested.attr", "value")

        lines = [line for line in buf.getvalue().splitlines() if line.strip()]
        assert len(lines) == 2
        assert [json.loads(line)["name"] for line in lines] == ["first", "second"]

    def test_metric_export_writes_one_line_per_batch(self):
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

        from nemo.lens.providers import _build_metric_exporter

        exporter = _build_metric_exporter(self._console_cfg())
        buf = io.StringIO()
        exporter.out = buf

        # A long interval keeps the background thread from exporting on its own;
        # force_flush() below is the only export we want to observe.
        reader = PeriodicExportingMetricReader(exporter, export_interval_millis=600_000)
        provider = MeterProvider(metric_readers=[reader])
        try:
            provider.get_meter("test").create_counter("test.counter").add(1)
            provider.force_flush()

            lines = [line for line in buf.getvalue().splitlines() if line.strip()]
            assert len(lines) == 1
            json.loads(lines[0])
        finally:
            provider.shutdown()

    def test_log_export_writes_one_line_per_record(self):
        from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
        from opentelemetry.sdk._logs.export import SimpleLogRecordProcessor

        from nemo.lens.providers import _build_log_exporter

        exporter = _build_log_exporter(self._console_cfg())
        buf = io.StringIO()
        exporter.out = buf

        provider = LoggerProvider()
        provider.add_log_record_processor(SimpleLogRecordProcessor(exporter))

        logger = logging.getLogger("nemo.lens.tests.jsonl")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        handler = LoggingHandler(level=logging.INFO, logger_provider=provider)
        logger.addHandler(handler)
        try:
            logger.info("first")
            logger.info("second")
        finally:
            logger.removeHandler(handler)

        lines = [line for line in buf.getvalue().splitlines() if line.strip()]
        assert len(lines) == 2
        assert [json.loads(line)["body"] for line in lines] == ["first", "second"]

    def test_formatter_emits_single_trailing_newline(self):
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        from nemo.lens.providers import _compact_jsonl_formatter

        captured = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(captured))
        with provider.get_tracer("test").start_as_current_span("formatted"):
            pass
        (readable_span,) = captured.get_finished_spans()

        formatted = _compact_jsonl_formatter(readable_span)
        assert formatted.endswith("\n")
        assert "\n" not in formatted[:-1]
        assert json.loads(formatted)["name"] == "formatted"
