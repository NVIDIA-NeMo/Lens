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

import os

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
