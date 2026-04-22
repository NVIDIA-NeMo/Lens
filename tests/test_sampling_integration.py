# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for RankAwareSampler as an OTel Sampler and its integration with providers."""

from opentelemetry.sdk.trace.sampling import Decision

from nemo.lens.sampling import RankAwareSampler


class TestRankAwareSamplerOTelInterface:
    def test_implements_should_sample_method(self):
        sampler = RankAwareSampler(rank=0, world_size=4, sample_rate=1.0)
        result = sampler.should_sample(
            parent_context=None,
            trace_id=12345,
            name="test.span",
        )
        assert result.decision == Decision.RECORD_AND_SAMPLE

    def test_drop_decision_when_rate_excludes_rank(self):
        sampler = RankAwareSampler(rank=0, world_size=4, sample_rate=0.0)
        result = sampler.should_sample(
            parent_context=None,
            trace_id=12345,
            name="test.span",
        )
        assert result.decision == Decision.DROP

    def test_get_description(self):
        sampler = RankAwareSampler(rank=0, world_size=4, sample_rate=0.5)
        desc = sampler.get_description()
        assert "RankAwareSampler" in desc
        assert "0.5" in desc

    def test_attributes_passed_through_on_sample(self):
        sampler = RankAwareSampler(rank=0, world_size=4, sample_rate=1.0)
        result = sampler.should_sample(
            parent_context=None,
            trace_id=12345,
            name="test.span",
            attributes={"key": "value"},
        )
        assert result.decision == Decision.RECORD_AND_SAMPLE
        assert result.attributes == {"key": "value"}

    def test_no_attributes_on_drop(self):
        sampler = RankAwareSampler(rank=0, world_size=4, sample_rate=0.0)
        result = sampler.should_sample(
            parent_context=None,
            trace_id=12345,
            name="test.span",
            attributes={"key": "value"},
        )
        assert result.decision == Decision.DROP


class TestSamplerInProviders:
    def test_sampler_wired_when_enabled(self):
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider

        from nemo.lens.config import NemoLensConfig
        from nemo.lens.providers import build_providers

        cfg = NemoLensConfig(
            enabled=True,
            exporter="console",
            sampler_enabled=True,
            export_sample_rate=0.5,
        )
        build_providers(cfg, rank=0, world_size=4)

        provider = trace.get_tracer_provider()
        assert isinstance(provider, TracerProvider)
        assert isinstance(provider.sampler, RankAwareSampler)

    def test_no_sampler_when_disabled(self):
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider

        from nemo.lens.config import NemoLensConfig
        from nemo.lens.providers import build_providers

        cfg = NemoLensConfig(
            enabled=True,
            exporter="console",
            sampler_enabled=False,
        )
        build_providers(cfg, rank=0, world_size=1)

        provider = trace.get_tracer_provider()
        assert isinstance(provider, TracerProvider)
        assert not isinstance(provider.sampler, RankAwareSampler)
