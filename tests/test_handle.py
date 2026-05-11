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

"""Unit tests for TelemetryHandle and setup_telemetry."""

import pytest

from nemo.lens.config import NemoLensConfig
from nemo.lens.groups import SpanGroup
from nemo.lens.handle import TelemetryHandle, _should_export, setup_telemetry
from nemo.lens.state import is_span_group_enabled


class TestShouldExport:
    def test_all_ranks_always_exports(self):
        cfg = NemoLensConfig(export_strategy="all_ranks")
        assert _should_export(cfg, rank=0, world_size=4) is True
        assert _should_export(cfg, rank=3, world_size=4) is True

    def test_single_rank_last(self):
        cfg = NemoLensConfig(export_strategy="single_rank", export_rank=-1)
        assert _should_export(cfg, rank=3, world_size=4) is True
        assert _should_export(cfg, rank=0, world_size=4) is False

    def test_single_rank_zero(self):
        cfg = NemoLensConfig(export_strategy="single_rank", export_rank=0)
        assert _should_export(cfg, rank=0, world_size=4) is True
        assert _should_export(cfg, rank=1, world_size=4) is False

    def test_sampled_deterministic(self):
        cfg = NemoLensConfig(export_strategy="sampled", export_sample_rate=0.5)
        # Should be deterministic for same rank
        result1 = _should_export(cfg, rank=0, world_size=100)
        result2 = _should_export(cfg, rank=0, world_size=100)
        assert result1 == result2

    def test_sampled_rate_zero(self):
        cfg = NemoLensConfig(export_strategy="sampled", export_sample_rate=0.0)
        # With rate 0.0, no rank should export (hash % 10000 / 10000 is always >= 0)
        # Actually hash/10000 could be 0 for some ranks, but 0.0 < 0.0 is False
        # so no rank should export
        for r in range(10):
            assert _should_export(cfg, rank=r, world_size=10) is False

    def test_sampled_rate_one(self):
        cfg = NemoLensConfig(export_strategy="sampled", export_sample_rate=1.0)
        for r in range(10):
            assert _should_export(cfg, rank=r, world_size=10) is True

    def test_first_rank_per_node_local_zero(self, monkeypatch):
        monkeypatch.setenv("LOCAL_RANK", "0")
        cfg = NemoLensConfig(export_strategy="first_rank_per_node")
        assert _should_export(cfg, rank=0, world_size=8) is True

    def test_first_rank_per_node_local_nonzero(self, monkeypatch):
        monkeypatch.setenv("LOCAL_RANK", "5")
        cfg = NemoLensConfig(export_strategy="first_rank_per_node")
        assert _should_export(cfg, rank=5, world_size=8) is False

    def test_first_rank_per_node_no_env(self, monkeypatch):
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        cfg = NemoLensConfig(export_strategy="first_rank_per_node")
        assert _should_export(cfg, rank=0, world_size=1) is True

    def test_unknown_strategy_raises(self):
        cfg = NemoLensConfig(export_strategy="bogus_strategy_xyz")
        with pytest.raises(ValueError, match="Unknown export_strategy"):
            _should_export(cfg, rank=0, world_size=4)

    def test_override_callable_takes_precedence(self):
        cfg = NemoLensConfig(export_strategy="all_ranks")
        assert _should_export(cfg, rank=0, world_size=4, override=lambda c, r, ws: False) is False


class TestSetupTelemetryDisabled:
    def test_returns_handle(self):
        cfg = NemoLensConfig(enabled=False)
        handle = setup_telemetry(cfg, rank=0, world_size=4)
        assert isinstance(handle, TelemetryHandle)

    def test_tracer_accessible(self):
        cfg = NemoLensConfig(enabled=False)
        handle = setup_telemetry(cfg, rank=0, world_size=4)
        assert handle.tracer is not None

    def test_meter_accessible(self):
        cfg = NemoLensConfig(enabled=False)
        handle = setup_telemetry(cfg, rank=0, world_size=4)
        assert handle.meter is not None

    def test_noop_span_creation(self):
        cfg = NemoLensConfig(enabled=False)
        handle = setup_telemetry(cfg, rank=0, world_size=4)
        with handle.tracer.start_as_current_span("test") as span:
            assert span is not None

    def test_shutdown_completes(self):
        cfg = NemoLensConfig(enabled=False)
        handle = setup_telemetry(cfg, rank=0, world_size=4)
        handle.shutdown(timeout_ms=100)

    def test_is_not_exporting(self):
        cfg = NemoLensConfig(enabled=False)
        handle = setup_telemetry(cfg, rank=0, world_size=1)
        assert handle.is_exporting is False


class TestSetupTelemetryEnabled:
    def test_export_rank_is_exporting(self):
        cfg = NemoLensConfig(enabled=True, export_rank=0, exporter="console")
        handle = setup_telemetry(cfg, rank=0, world_size=4)
        assert handle.is_exporting is True

    def test_non_export_rank_not_exporting(self):
        cfg = NemoLensConfig(enabled=True, export_rank=-1, exporter="console")
        handle = setup_telemetry(cfg, rank=0, world_size=4)
        assert handle.is_exporting is False

    def test_last_rank_default(self):
        cfg = NemoLensConfig(enabled=True, export_rank=-1, exporter="console")
        handle = setup_telemetry(cfg, rank=3, world_size=4)
        assert handle.is_exporting is True

    def test_all_ranks_strategy(self):
        cfg = NemoLensConfig(enabled=True, export_strategy="all_ranks", exporter="console")
        handle = setup_telemetry(cfg, rank=0, world_size=4)
        assert handle.is_exporting is True

    def test_first_rank_per_node_strategy(self, monkeypatch):
        monkeypatch.setenv("LOCAL_RANK", "0")
        cfg = NemoLensConfig(
            enabled=True, export_strategy="first_rank_per_node", exporter="console"
        )
        handle = setup_telemetry(cfg, rank=0, world_size=4)
        assert handle.is_exporting is True

    def test_first_rank_per_node_non_local_zero(self, monkeypatch):
        monkeypatch.setenv("LOCAL_RANK", "3")
        cfg = NemoLensConfig(
            enabled=True, export_strategy="first_rank_per_node", exporter="console"
        )
        handle = setup_telemetry(cfg, rank=3, world_size=8)
        assert handle.is_exporting is False


class TestSetupTelemetrySpanGroups:
    def test_disabled_clears_all_groups(self):
        cfg = NemoLensConfig(enabled=False, span_groups="all")
        setup_telemetry(cfg, rank=0, world_size=1)
        for group in SpanGroup.ALL_GROUPS:
            assert not is_span_group_enabled(group)

    def test_enabled_registers_default_groups(self):
        cfg = NemoLensConfig(enabled=True, span_groups="default", exporter="console")
        setup_telemetry(cfg, rank=0, world_size=1)
        assert is_span_group_enabled(SpanGroup.JOB) is True
        assert is_span_group_enabled(SpanGroup.CHECKPOINT) is True
        assert is_span_group_enabled(SpanGroup.STEP) is False

    def test_enabled_registers_per_step_groups(self):
        cfg = NemoLensConfig(enabled=True, span_groups="per_step", exporter="console")
        setup_telemetry(cfg, rank=0, world_size=1)
        assert is_span_group_enabled(SpanGroup.STEP) is True
        assert is_span_group_enabled(SpanGroup.FORWARD_BACKWARD) is True

    def test_non_export_rank_clears_span_groups(self):
        cfg = NemoLensConfig(enabled=True, span_groups="all", exporter="console")
        setup_telemetry(cfg, rank=0, world_size=4)
        for group in SpanGroup.ALL_GROUPS:
            assert not is_span_group_enabled(group)


class TestDoubleInitGuard:
    def test_double_init_raises(self):
        cfg = NemoLensConfig(enabled=True, exporter="console")
        setup_telemetry(cfg, rank=0, world_size=1)
        with pytest.raises(RuntimeError, match="already been initialised"):
            setup_telemetry(cfg, rank=0, world_size=1)

    def test_double_init_disabled_is_allowed(self):
        cfg = NemoLensConfig(enabled=False)
        setup_telemetry(cfg, rank=0, world_size=1)
        handle = setup_telemetry(cfg, rank=0, world_size=1)
        assert handle.is_exporting is False

    def test_allow_reinit_flag(self):
        cfg = NemoLensConfig(enabled=True, exporter="console")
        setup_telemetry(cfg, rank=0, world_size=1)
        handle = setup_telemetry(cfg, rank=0, world_size=1, _allow_reinit=True)
        assert handle is not None


class TestTelemetryHandleShutdown:
    def test_shutdown_idempotent(self):
        cfg = NemoLensConfig(enabled=False)
        handle = setup_telemetry(cfg, rank=0, world_size=1)
        handle.shutdown(timeout_ms=100)
        handle.shutdown(timeout_ms=100)
