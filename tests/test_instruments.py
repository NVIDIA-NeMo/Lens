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

"""Unit tests for metric instruments."""

import pytest
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from nemo.lens.instruments.gym import record_gym_metrics
from nemo.lens.instruments.inference import record_inference_metrics
from nemo.lens.instruments.rl import record_rl_metrics


@pytest.fixture
def meter_and_reader():
    """Create an isolated in-memory metrics pipeline for each test."""
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    metrics.set_meter_provider(provider)
    meter = metrics.get_meter("test")
    yield meter, reader
    provider.shutdown()


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

    def test_records_all_metrics(self, meter_and_reader):
        """Verify all optional RL metric inputs are recorded."""
        meter, reader = meter_and_reader
        record_rl_metrics(
            meter,
            reward_mean=0.85,
            kl_divergence=0.02,
            policy_loss=0.3,
            value_loss=0.4,
            entropy=0.5,
            response_length_mean=128.0,
            generation_duration_ms=50.0,
            rollout_duration_ms=100.0,
        )
        data = reader.get_metrics_data()
        metric_names = [
            m.name for rm in data.resource_metrics for sm in rm.scope_metrics for m in sm.metrics
        ]
        assert "rl.policy_loss" in metric_names
        assert "rl.value_loss" in metric_names
        assert "rl.entropy" in metric_names
        assert "rl.response_length.mean" in metric_names
        assert "rl.generation.duration_ms" in metric_names
        assert "rl.rollout.duration_ms" in metric_names


class TestRecordGymMetrics:
    def test_records_server_duration(self, meter_and_reader):
        meter, reader = meter_and_reader
        record_gym_metrics(meter, server_request_duration_ms=50.0)
        data = reader.get_metrics_data()
        metric_names = [
            m.name for rm in data.resource_metrics for sm in rm.scope_metrics for m in sm.metrics
        ]
        assert "gym.server.request_duration_ms" in metric_names

    def test_records_all_metrics(self, meter_and_reader):
        """Verify all optional Gym metric inputs are recorded."""
        meter, reader = meter_and_reader
        record_gym_metrics(
            meter,
            server_request_duration_ms=50.0,
            rollout_duration_ms=75.0,
            verify_duration_ms=25.0,
            verify_success_rate=0.95,
            active_servers=3,
        )
        data = reader.get_metrics_data()
        metric_names = [
            m.name for rm in data.resource_metrics for sm in rm.scope_metrics for m in sm.metrics
        ]
        assert "gym.rollout.duration_ms" in metric_names
        assert "gym.verify.duration_ms" in metric_names
        assert "gym.verify.success_rate" in metric_names
        assert "gym.servers.active" in metric_names
