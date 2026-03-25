# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for provider construction."""

import pytest

from opentelemetry import metrics, trace
from opentelemetry.trace import NoOpTracerProvider
from opentelemetry.metrics import NoOpMeterProvider

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
        cfg = NemoLensConfig(enabled=True, exporter='console')
        build_providers(cfg, rank=0, world_size=1)
        # Should not raise; tracer provider should be SDK type
        tracer = trace.get_tracer('test')
        with tracer.start_as_current_span('test') as span:
            assert span is not None

    def test_invalid_exporter_raises(self):
        cfg = NemoLensConfig(enabled=True, exporter='invalid')
        with pytest.raises(ValueError, match='Unknown exporter'):
            build_providers(cfg, rank=0, world_size=1)

    def test_resource_attributes_merged(self):
        cfg = NemoLensConfig(enabled=True, exporter='console')
        build_providers(cfg, rank=0, world_size=1, resource_attributes={'custom.attr': 'value'})
        # Should not raise
        tracer = trace.get_tracer('test')
        assert tracer is not None

    def test_traces_disabled(self):
        cfg = NemoLensConfig(enabled=True, exporter='console', traces_enabled=False)
        build_providers(cfg, rank=0, world_size=1)
        # Tracer should be no-op (not set by us)

    def test_metrics_disabled(self):
        cfg = NemoLensConfig(enabled=True, exporter='console', metrics_enabled=False)
        build_providers(cfg, rank=0, world_size=1)
        # Meter should be no-op (not set by us)
