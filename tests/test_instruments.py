# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for metric instruments."""

import pytest
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from nemo.lens.instruments.gym import record_gym_metrics
from nemo.lens.instruments.inference import record_inference_metrics
from nemo.lens.instruments.rl import record_rl_metrics
from nemo.lens.instruments.training import record_training_metrics


@pytest.fixture
def meter_and_reader():
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    metrics.set_meter_provider(provider)
    meter = metrics.get_meter("test")
    yield meter, reader
    provider.shutdown()


class TestRecordTrainingMetrics:
    def test_records_loss(self, meter_and_reader):
        meter, reader = meter_and_reader
        record_training_metrics(meter, loss=2.5)
        data = reader.get_metrics_data()
        metric_names = [
            m.name for rm in data.resource_metrics for sm in rm.scope_metrics for m in sm.metrics
        ]
        assert "dl.training.loss" in metric_names

    def test_records_step_duration(self, meter_and_reader):
        meter, reader = meter_and_reader
        record_training_metrics(meter, step_duration_ms=150.0)
        data = reader.get_metrics_data()
        metric_names = [
            m.name for rm in data.resource_metrics for sm in rm.scope_metrics for m in sm.metrics
        ]
        assert "dl.training.step_duration_ms" in metric_names

    def test_none_values_skipped(self, meter_and_reader):
        meter, reader = meter_and_reader
        # Should not raise
        record_training_metrics(meter)

    def test_skipped_iters_counter(self, meter_and_reader):
        meter, reader = meter_and_reader
        record_training_metrics(meter, skipped_iters=3)
        data = reader.get_metrics_data()
        metric_names = [
            m.name for rm in data.resource_metrics for sm in rm.scope_metrics for m in sm.metrics
        ]
        assert "dl.training.skipped_iters" in metric_names


class TestRecordInferenceMetrics:
    def test_records_request_duration(self, meter_and_reader):
        meter, reader = meter_and_reader
        record_inference_metrics(meter, request_duration_s=1.5, model="gpt-3")
        data = reader.get_metrics_data()
        metric_names = [
            m.name for rm in data.resource_metrics for sm in rm.scope_metrics for m in sm.metrics
        ]
        assert "gen_ai.server.request.duration" in metric_names

    def test_records_token_usage(self, meter_and_reader):
        meter, reader = meter_and_reader
        record_inference_metrics(meter, input_tokens=100, output_tokens=50, model="gpt-3")
        data = reader.get_metrics_data()
        metric_names = [
            m.name for rm in data.resource_metrics for sm in rm.scope_metrics for m in sm.metrics
        ]
        assert "gen_ai.client.token.usage" in metric_names


class TestRecordRLMetrics:
    def test_records_reward(self, meter_and_reader):
        meter, reader = meter_and_reader
        record_rl_metrics(meter, reward_mean=0.85)
        data = reader.get_metrics_data()
        metric_names = [
            m.name for rm in data.resource_metrics for sm in rm.scope_metrics for m in sm.metrics
        ]
        assert "rl.reward.mean" in metric_names

    def test_records_kl(self, meter_and_reader):
        meter, reader = meter_and_reader
        record_rl_metrics(meter, kl_divergence=0.02)
        data = reader.get_metrics_data()
        metric_names = [
            m.name for rm in data.resource_metrics for sm in rm.scope_metrics for m in sm.metrics
        ]
        assert "rl.kl_divergence" in metric_names


class TestRecordGymMetrics:
    def test_records_server_duration(self, meter_and_reader):
        meter, reader = meter_and_reader
        record_gym_metrics(meter, server_request_duration_ms=50.0)
        data = reader.get_metrics_data()
        metric_names = [
            m.name for rm in data.resource_metrics for sm in rm.scope_metrics for m in sm.metrics
        ]
        assert "gym.server.request_duration_ms" in metric_names
