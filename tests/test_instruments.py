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

import logging
import os

import pytest
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from nemo.lens.instruments import (
    MetricSpec,
    record_metrics,
    register_metric_group,
    registered_metric_groups,
    unregister_metric_group,
)
from nemo.lens.instruments.gym import record_gym_metrics
from nemo.lens.instruments.inference import record_inference_metrics


def _rl_group_specs():
    """The RL series a consumer such as NeMo-RL would register with lens.

    The ``rl.*`` names are consumer-owned — lens no longer defines them in
    ``semconv`` — so they are spelled here as the consumer would spell them.
    """
    return [
        MetricSpec("reward_mean", "rl.reward.mean", "gauge"),
        MetricSpec("kl_divergence", "rl.kl_divergence", "gauge"),
        MetricSpec("policy_loss", "rl.policy_loss", "gauge"),
        MetricSpec("value_loss", "rl.value_loss", "gauge"),
        MetricSpec("entropy", "rl.entropy", "gauge"),
        MetricSpec("response_length_mean", "rl.response_length.mean", "gauge"),
        MetricSpec("grad_norm", "rl.grad_norm", "gauge"),
        MetricSpec("learning_rate", "rl.learning_rate", "gauge"),
        MetricSpec(
            "throughput_tokens_per_sec",
            "rl.throughput.tokens_per_sec",
            "gauge",
            unit="{token}/s",
        ),
        MetricSpec("generation_duration_ms", "rl.generation.duration_ms", "histogram", unit="ms"),
        MetricSpec("rollout_duration_ms", "rl.rollout.duration_ms", "histogram", unit="ms"),
    ]


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


class TestMetricRegistry:
    def test_register_and_record_single(self, meter_and_reader):
        meter, reader = meter_and_reader
        register_metric_group("rl", _rl_group_specs())
        record_metrics(meter, "rl", reward_mean=0.85)
        metric_names = [
            m.name
            for rm in reader.get_metrics_data().resource_metrics
            for sm in rm.scope_metrics
            for m in sm.metrics
        ]
        assert "rl.reward.mean" in metric_names

    def test_records_all_values(self, meter_and_reader):
        """Every provided series is recorded with the value it was given.

        The recorder routes each key through its spec, so a mis-wired key is the
        failure mode a name-only assertion cannot see.
        """
        meter, reader = meter_and_reader
        register_metric_group("rl", _rl_group_specs())
        record_metrics(
            meter,
            "rl",
            reward_mean=0.85,
            kl_divergence=0.02,
            policy_loss=0.3,
            value_loss=0.4,
            entropy=0.5,
            response_length_mean=128.0,
            grad_norm=1.7,
            learning_rate=3e-6,
            throughput_tokens_per_sec=18500.0,
            generation_duration_ms=50.0,
            rollout_duration_ms=100.0,
        )
        points = {
            m.name: list(m.data.data_points)[-1]
            for rm in reader.get_metrics_data().resource_metrics
            for sm in rm.scope_metrics
            for m in sm.metrics
        }
        assert points["rl.reward.mean"].value == 0.85
        assert points["rl.kl_divergence"].value == 0.02
        assert points["rl.policy_loss"].value == 0.3
        assert points["rl.value_loss"].value == 0.4
        assert points["rl.entropy"].value == 0.5
        assert points["rl.response_length.mean"].value == 128.0
        assert points["rl.grad_norm"].value == 1.7
        assert points["rl.learning_rate"].value == 3e-6
        assert points["rl.throughput.tokens_per_sec"].value == 18500.0
        assert points["rl.generation.duration_ms"].sum == 50.0
        assert points["rl.rollout.duration_ms"].sum == 100.0

    def test_values_accepted_as_mapping(self, meter_and_reader):
        meter, reader = meter_and_reader
        register_metric_group("rl", _rl_group_specs())
        record_metrics(meter, "rl", {"reward_mean": 0.5}, kl_divergence=0.01)
        points = {
            m.name: list(m.data.data_points)[-1]
            for rm in reader.get_metrics_data().resource_metrics
            for sm in rm.scope_metrics
            for m in sm.metrics
        }
        assert points["rl.reward.mean"].value == 0.5
        assert points["rl.kl_divergence"].value == 0.01

    def test_spec_unit_is_applied(self, meter_and_reader):
        """A rate gauge must carry the UCUM unit declared on its spec."""
        meter, reader = meter_and_reader
        register_metric_group("rl", _rl_group_specs())
        record_metrics(meter, "rl", throughput_tokens_per_sec=18500.0)
        units = {
            m.name: m.unit
            for rm in reader.get_metrics_data().resource_metrics
            for sm in rm.scope_metrics
            for m in sm.metrics
        }
        assert units["rl.throughput.tokens_per_sec"] == "{token}/s"

    def test_counter_and_up_down_counter_kinds(self, meter_and_reader):
        """`counter` and `up_down_counter` route to create_*/add, not gauge/histogram."""
        meter, reader = meter_and_reader
        register_metric_group(
            "misc",
            [
                MetricSpec("events", "misc.events", "counter"),
                MetricSpec("inflight", "misc.inflight", "up_down_counter"),
            ],
        )
        record_metrics(meter, "misc", events=3, inflight=-1)
        points = {
            m.name: list(m.data.data_points)[-1]
            for rm in reader.get_metrics_data().resource_metrics
            for sm in rm.scope_metrics
            for m in sm.metrics
        }
        assert points["misc.events"].value == 3
        assert points["misc.inflight"].value == -1

    def test_keys_colliding_with_param_names_are_safe(self, meter_and_reader):
        """A consumer key named meter/group/values must not collide with the params.

        These three are positional-only, so passing them as kwargs records the
        metric instead of raising ``TypeError`` into the caller.
        """
        meter, reader = meter_and_reader
        register_metric_group(
            "edge",
            [
                MetricSpec("meter", "edge.meter", "gauge"),
                MetricSpec("group", "edge.group", "gauge"),
                MetricSpec("values", "edge.values", "gauge"),
            ],
        )
        record_metrics(meter, "edge", meter=1.0, group=2.0, values=3.0)
        points = {
            m.name: list(m.data.data_points)[-1].value
            for rm in reader.get_metrics_data().resource_metrics
            for sm in rm.scope_metrics
            for m in sm.metrics
        }
        assert points == {"edge.meter": 1.0, "edge.group": 2.0, "edge.values": 3.0}

    def test_attributes_are_attached(self, meter_and_reader):
        """The `attributes=` mapping lands on every emitted data point."""
        meter, reader = meter_and_reader
        register_metric_group("rl", _rl_group_specs())
        record_metrics(meter, "rl", {"reward_mean": 0.5}, attributes={"rl.algorithm": "grpo"})
        point = next(
            list(m.data.data_points)[-1]
            for rm in reader.get_metrics_data().resource_metrics
            for sm in rm.scope_metrics
            for m in sm.metrics
            if m.name == "rl.reward.mean"
        )
        assert point.attributes["rl.algorithm"] == "grpo"

    def test_non_mapping_values_is_logged_not_raised(self, meter_and_reader, caplog):
        """A `values` argument that is not a mapping is skipped, never raised."""
        meter, _ = meter_and_reader
        register_metric_group("rl", _rl_group_specs())
        with caplog.at_level(logging.WARNING, logger="nemo.lens.instruments.registry"):
            for _ in range(3):
                record_metrics(meter, "rl", 0.5)
        warnings = [r for r in caplog.records if "`values` argument of type" in r.message]
        assert len(warnings) == 1

    def test_metric_key_named_attributes_is_logged_not_raised(self, meter_and_reader, caplog):
        """`attributes` is reserved, so a metric key of that name cannot be a kwarg.

        Passing it anyway must warn and drop it while still recording the rest of
        the call, rather than raising ``TypeError`` from ``dict(attributes)``.
        """
        meter, reader = meter_and_reader
        register_metric_group(
            "edge",
            [MetricSpec("x", "edge.x"), MetricSpec("attributes", "edge.attributes")],
        )
        with caplog.at_level(logging.WARNING, logger="nemo.lens.instruments.registry"):
            record_metrics(meter, "edge", x=1.0, attributes=0.5)
        assert any("`attributes` argument of type" in r.message for r in caplog.records)
        names = [
            m.name
            for rm in reader.get_metrics_data().resource_metrics
            for sm in rm.scope_metrics
            for m in sm.metrics
        ]
        assert "edge.x" in names
        assert "edge.attributes" not in names

    def test_key_named_attributes_records_through_values_mapping(self, meter_and_reader):
        """The documented escape hatch for a key named `attributes`."""
        meter, reader = meter_and_reader
        register_metric_group("edge", [MetricSpec("attributes", "edge.attributes")])
        record_metrics(meter, "edge", {"attributes": 7.0})
        points = {
            m.name: list(m.data.data_points)[-1].value
            for rm in reader.get_metrics_data().resource_metrics
            for sm in rm.scope_metrics
            for m in sm.metrics
        }
        assert points == {"edge.attributes": 7.0}

    def test_unusable_meter_is_logged_not_raised(self, caplog):
        """A meter that cannot be weak-referenced (``None``) must not raise.

        The instrument cache is a WeakKeyDictionary, so the lookup itself throws
        for such a meter — it has to be caught before it reaches the caller.
        """
        register_metric_group("rl", _rl_group_specs())
        with caplog.at_level(logging.WARNING, logger="nemo.lens.instruments.registry"):
            record_metrics(None, "rl", reward_mean=0.5)
        assert any("Failed to resolve instruments" in r.message for r in caplog.records)

    def test_none_and_unknown_keys_are_skipped(self, meter_and_reader):
        meter, reader = meter_and_reader
        register_metric_group("rl", _rl_group_specs())
        record_metrics(meter, "rl", reward_mean=0.85, kl_divergence=None, not_a_metric=1.0)
        metric_names = [
            m.name
            for rm in reader.get_metrics_data().resource_metrics
            for sm in rm.scope_metrics
            for m in sm.metrics
        ]
        assert "rl.reward.mean" in metric_names
        assert "rl.kl_divergence" not in metric_names

    def test_unknown_key_warns_once_not_per_call(self, meter_and_reader, caplog):
        """A misconfigured key in a hot loop logs once, not on every call."""
        meter, reader = meter_and_reader
        register_metric_group("rl", _rl_group_specs())
        with caplog.at_level(logging.WARNING, logger="nemo.lens.instruments.registry"):
            for _ in range(5):
                record_metrics(meter, "rl", not_a_metric=1.0)
        unknown_key_warnings = [r for r in caplog.records if "Unknown metric key" in r.message]
        assert len(unknown_key_warnings) == 1

    def test_unregistered_group_is_noop(self, meter_and_reader):
        """Recording against an unknown group is swallowed, never raised."""
        meter, reader = meter_and_reader
        record_metrics(meter, "nonexistent", reward_mean=0.85)
        # No instruments were created, so the reader has nothing to collect.
        data = reader.get_metrics_data()
        names = [
            m.name
            for rm in (data.resource_metrics if data else ())
            for sm in rm.scope_metrics
            for m in sm.metrics
        ]
        assert names == []

    def test_register_rejects_duplicate_group(self):
        register_metric_group("rl", _rl_group_specs())
        with pytest.raises(ValueError, match="already registered"):
            register_metric_group("rl", _rl_group_specs())

    def test_register_allow_override_replaces(self):
        register_metric_group("rl", [MetricSpec("reward_mean", "rl.reward.mean")])
        register_metric_group("rl", [MetricSpec("entropy", "rl.entropy")], allow_override=True)
        specs = {s.key for s in registered_metric_groups()["rl"]}
        assert specs == {"entropy"}

    def test_register_rejects_duplicate_key(self):
        with pytest.raises(ValueError, match="Duplicate metric key"):
            register_metric_group(
                "rl",
                [
                    MetricSpec("reward_mean", "rl.reward.mean"),
                    MetricSpec("reward_mean", "rl.kl_divergence"),
                ],
            )

    def test_register_rejects_empty_group(self):
        with pytest.raises(ValueError, match="at least one"):
            register_metric_group("rl", [])

    def test_register_rejects_empty_group_name(self):
        with pytest.raises(ValueError, match="non-empty"):
            register_metric_group("", _rl_group_specs())

    def test_metric_spec_rejects_unknown_kind(self):
        with pytest.raises(ValueError, match="not one of"):
            MetricSpec("x", "rl.x", "summary")

    def test_metric_spec_rejects_empty_key_or_name(self):
        with pytest.raises(ValueError, match="key must be"):
            MetricSpec("", "rl.x")
        with pytest.raises(ValueError, match="name must be"):
            MetricSpec("x", "")

    def test_record_with_no_values_is_noop(self, meter_and_reader):
        meter, reader = meter_and_reader
        register_metric_group("rl", _rl_group_specs())
        record_metrics(meter, "rl")  # no values, no kwargs
        data = reader.get_metrics_data()
        names = [
            m.name
            for rm in (data.resource_metrics if data else ())
            for sm in rm.scope_metrics
            for m in sm.metrics
        ]
        assert names == []

    def test_unregister_removes_group(self):
        register_metric_group("rl", _rl_group_specs())
        unregister_metric_group("rl")
        assert "rl" not in registered_metric_groups()
        with pytest.raises(ValueError, match="not registered"):
            unregister_metric_group("rl")

    def test_emitted_names_match_registered_specs(self, meter_and_reader):
        """The name recorded under each key is the ``MetricSpec.name`` declared for it."""
        meter, reader = meter_and_reader
        specs = _rl_group_specs()
        register_metric_group("rl", specs)
        record_metrics(meter, "rl", **{s.key: 1.0 for s in specs})
        emitted = {
            m.name
            for rm in reader.get_metrics_data().resource_metrics
            for sm in rm.scope_metrics
            for m in sm.metrics
        }
        assert {s.name for s in specs} <= emitted


class TestForkSafety:
    def test_child_reinit_replaces_held_locks(self):
        """The child-side fork handler swaps a lock held at fork for a fresh one.

        A ``threading.Lock`` inherited locked with no owner would deadlock the
        child on the next acquire; the handler must hand back an unlocked lock.
        """
        import nemo.lens.instruments.registry as reg

        old_registry_lock, old_warn_lock = reg._REGISTRY_LOCK, reg._WARN_LOCK
        old_registry_lock.acquire()
        old_warn_lock.acquire()
        try:
            reg._reinit_locks_after_fork_in_child()
            assert reg._REGISTRY_LOCK is not old_registry_lock
            assert reg._WARN_LOCK is not old_warn_lock
            assert reg._REGISTRY_LOCK.acquire(blocking=False)
            reg._REGISTRY_LOCK.release()
            assert reg._WARN_LOCK.acquire(blocking=False)
            reg._WARN_LOCK.release()
        finally:
            old_registry_lock.release()
            old_warn_lock.release()

    @pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork")
    def test_forked_child_can_use_registry_without_deadlock(self):
        """End-to-end: fork, then use lock-guarded registry calls in the child.

        Exercises the registered before / after_in_parent / after_in_child
        handlers together. A regression (e.g. dropping after_in_child) leaves the
        inherited locks held, so the child deadlocks — the alarm turns that into a
        fast non-zero exit instead of a hung suite.
        """
        import signal

        register_metric_group("rl", _rl_group_specs(), allow_override=True)
        pid = os.fork()
        if pid == 0:  # child
            signal.alarm(10)
            try:
                register_metric_group("child", [MetricSpec("x", "child.x")], allow_override=True)
                assert "child" in registered_metric_groups()
                os._exit(0)
            except BaseException:
                os._exit(1)
        _, status = os.waitpid(pid, 0)
        assert os.waitstatus_to_exitcode(status) == 0


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
