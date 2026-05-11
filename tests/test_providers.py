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

import pytest
from opentelemetry import metrics, trace
from opentelemetry.metrics import NoOpMeterProvider
from opentelemetry.trace import NoOpTracerProvider

from nemo.lens.config import NemoLensConfig
from nemo.lens.providers import build_noop_providers, build_providers


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
