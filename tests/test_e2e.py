# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""End-to-end integration tests for nemo-lens."""

from opentelemetry import trace

from nemo.lens import (
    GymSpanGroup,
    MegatronSpanGroup,
    NemoLensConfig,
    RLSpanGroup,
    SpanGroup,
    TelemetryHandle,
    extract_context,
    get_tracer,
    inject_context,
    is_span_group_enabled,
    managed_span,
    setup_telemetry,
)
from nemo.lens.instruments.training import record_training_metrics


class TestE2EConsoleExporter:
    def test_full_lifecycle(self):
        """Test complete setup -> use -> shutdown lifecycle."""
        cfg = NemoLensConfig(enabled=True, exporter="console", span_groups="all")
        handle = setup_telemetry(cfg, rank=0, world_size=1)

        assert isinstance(handle, TelemetryHandle)
        assert handle.is_exporting is True

        # Create spans
        with managed_span(SpanGroup.JOB, "test.job", tracer=handle.tracer) as span:
            assert span is not None
            with managed_span(SpanGroup.STEP, "test.step", tracer=handle.tracer) as step_span:
                assert step_span is not None

        # Record metrics
        record_training_metrics(handle.meter, loss=2.5, step_duration_ms=100.0)

        handle.shutdown(timeout_ms=100)

    def test_disabled_zero_overhead(self):
        """When disabled, no spans should be created."""
        cfg = NemoLensConfig(enabled=False, span_groups="all")
        handle = setup_telemetry(cfg, rank=0, world_size=1)

        assert handle.is_exporting is False

        with managed_span(SpanGroup.JOB, "test.job") as span:
            assert span is None

        handle.shutdown(timeout_ms=100)


class TestE2EExportStrategies:
    def test_all_ranks_strategy(self):
        cfg = NemoLensConfig(
            enabled=True,
            exporter="console",
            export_strategy="all_ranks",
            span_groups="default",
        )
        for rank in range(4):
            handle = setup_telemetry(cfg, rank=rank, world_size=4)
            assert handle.is_exporting is True
            handle.shutdown(timeout_ms=100)

    def test_single_rank_strategy(self):
        cfg = NemoLensConfig(
            enabled=True,
            exporter="console",
            export_strategy="single_rank",
            export_rank=0,
            span_groups="default",
        )
        h0 = setup_telemetry(cfg, rank=0, world_size=4)
        assert h0.is_exporting is True
        h0.shutdown(timeout_ms=100)

        h1 = setup_telemetry(cfg, rank=1, world_size=4)
        assert h1.is_exporting is False
        h1.shutdown(timeout_ms=100)


class TestE2ELibrarySpecificGroups:
    def test_megatron_groups(self):
        cfg = NemoLensConfig(
            enabled=True,
            exporter="console",
            span_groups="all",
            _span_group_cls=MegatronSpanGroup,
        )
        handle = setup_telemetry(cfg, rank=0, world_size=1)
        assert is_span_group_enabled(MegatronSpanGroup.MICROBATCH)
        assert is_span_group_enabled(MegatronSpanGroup.INFERENCE)
        handle.shutdown(timeout_ms=100)

    def test_rl_groups(self):
        cfg = NemoLensConfig(
            enabled=True,
            exporter="console",
            span_groups="per_step",
            _span_group_cls=RLSpanGroup,
        )
        handle = setup_telemetry(cfg, rank=0, world_size=1)
        assert is_span_group_enabled(RLSpanGroup.ROLLOUT)
        assert is_span_group_enabled(RLSpanGroup.GENERATION)
        assert is_span_group_enabled(RLSpanGroup.REWARD)
        handle.shutdown(timeout_ms=100)

    def test_gym_groups(self):
        cfg = NemoLensConfig(
            enabled=True,
            exporter="console",
            span_groups="default",
            _span_group_cls=GymSpanGroup,
        )
        handle = setup_telemetry(cfg, rank=0, world_size=1)
        assert is_span_group_enabled(GymSpanGroup.SERVER)
        assert not is_span_group_enabled(GymSpanGroup.VERIFY)
        handle.shutdown(timeout_ms=100)


class TestE2ESpanHierarchy:
    def test_nested_spans(self):
        cfg = NemoLensConfig(enabled=True, exporter="console", span_groups="all")
        handle = setup_telemetry(cfg, rank=0, world_size=1)

        with managed_span(SpanGroup.JOB, "dl.train", tracer=handle.tracer) as job:
            assert job is not None
            with managed_span(
                SpanGroup.STEP, "dl.train_step", tracer=handle.tracer, iteration=1
            ) as step:
                assert step is not None
                with managed_span(
                    SpanGroup.FORWARD_BACKWARD, "dl.forward_backward", tracer=handle.tracer
                ) as fb:
                    assert fb is not None

        handle.shutdown(timeout_ms=100)


class TestE2EContextPropagation:
    def test_inject_extract_roundtrip(self):
        cfg = NemoLensConfig(enabled=True, exporter="console", span_groups="default")
        handle = setup_telemetry(cfg, rank=0, world_size=1)

        tracer = get_tracer()
        with tracer.start_as_current_span("origin") as span:
            carrier = {}
            inject_context(carrier)
            assert "traceparent" in carrier

        ctx = extract_context(carrier)
        remote_span = trace.get_current_span(ctx)
        assert remote_span.get_span_context().trace_id == span.get_span_context().trace_id

        handle.shutdown(timeout_ms=100)
