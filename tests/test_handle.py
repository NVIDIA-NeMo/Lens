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
from nemo.lens.groups import SpanRegistry
from nemo.lens.handle import TelemetryHandle, ensure_run_id, setup_telemetry
from nemo.lens.state import is_span_group_enabled


class TestSetupTelemetryDisabled:
    def test_returns_handle(self):
        cfg = NemoLensConfig(enabled=False)
        handle = setup_telemetry(cfg)
        assert isinstance(handle, TelemetryHandle)

    def test_tracer_accessible(self):
        cfg = NemoLensConfig(enabled=False)
        handle = setup_telemetry(cfg)
        assert handle.tracer is not None

    def test_meter_accessible(self):
        cfg = NemoLensConfig(enabled=False)
        handle = setup_telemetry(cfg)
        assert handle.meter is not None

    def test_noop_span_creation(self):
        cfg = NemoLensConfig(enabled=False)
        handle = setup_telemetry(cfg)
        with handle.tracer.start_as_current_span("test") as span:
            assert span is not None

    def test_shutdown_completes(self):
        cfg = NemoLensConfig(enabled=False)
        handle = setup_telemetry(cfg)
        handle.shutdown(timeout_ms=100)

    def test_is_not_exporting(self):
        cfg = NemoLensConfig(enabled=False)
        handle = setup_telemetry(cfg)
        assert handle.is_exporting is False


class TestSetupTelemetryEnabled:
    def test_ensure_run_id_preserves_explicit_value(self):
        cfg = NemoLensConfig(run_id="explicit-run")

        assert ensure_run_id(cfg, {}) == "explicit-run"
        assert cfg.run_id == "explicit-run"

    def test_ensure_run_id_uses_slurm_job_id(self):
        cfg = NemoLensConfig()

        assert ensure_run_id(cfg, {"SLURM_JOB_ID": "12345"}) == "12345"
        assert cfg.run_id == "12345"

    def test_ensure_run_id_generates_local_value(self):
        cfg = NemoLensConfig()

        run_id = ensure_run_id(cfg, {})

        assert len(run_id) == 12
        assert cfg.run_id == run_id

    def test_enabled_is_exporting(self):
        cfg = NemoLensConfig(enabled=True, exporter="console")
        handle = setup_telemetry(cfg)
        assert handle.is_exporting is True

    def test_every_process_exports(self):
        """Lens no longer elects an exporting rank -- enabled means exporting.

        Restricting export to a subset of processes is the caller's decision now:
        it leaves ``enabled`` false on the ones that should stay quiet.
        """
        cfg = NemoLensConfig(enabled=True, exporter="console")
        for _ in range(4):
            handle = setup_telemetry(cfg, _allow_reinit=True)
            assert handle.is_exporting is True

    def test_rank_identity_travels_as_a_resource_attribute(self):
        """The replacement for the two removed positional parameters.

        Asserts on the built Resource, not on ``is_exporting``: the latter is just
        ``config.enabled`` and would stay green if the parameter were dropped on
        the floor between ``setup_telemetry`` and ``build_providers``, which is the
        whole seam this replaces.
        """
        from opentelemetry import trace

        from nemo.lens.semconv import NV_DL_RANK, NV_DL_WORLD_SIZE

        cfg = NemoLensConfig(enabled=True, exporter="console", run_id="run1")
        handle = setup_telemetry(
            cfg,
            resource_attributes={NV_DL_RANK: 3, NV_DL_WORLD_SIZE: 8},
        )

        attrs = dict(trace.get_tracer_provider().resource.attributes)
        assert attrs[NV_DL_RANK] == 3
        assert attrs[NV_DL_WORLD_SIZE] == 8
        assert attrs["service.instance.id"] == "run1-rank3"
        assert handle.is_exporting is True

    def test_setup_telemetry_rejects_the_removed_positional_arguments(self):
        """A stale ``setup_telemetry(cfg, rank, world_size)`` must fail loudly.

        Before the parameters were made keyword-only these rebound onto
        ``resource_attributes`` and ``span_exporter``, producing a handle that
        claimed to be exporting, dropped every span, and exited zero.
        """
        cfg = NemoLensConfig(enabled=True, exporter="console")
        with pytest.raises(TypeError):
            setup_telemetry(cfg, 0, 8)


class TestSetupTelemetrySpanGroups:
    def test_disabled_clears_all_groups(self, demo_groups):
        cfg = NemoLensConfig(enabled=False, span_groups="all")
        setup_telemetry(cfg)
        for group in SpanRegistry.groups():
            assert not is_span_group_enabled(group)

    def test_enabled_registers_default_groups(self, demo_groups):
        cfg = NemoLensConfig(enabled=True, span_groups="default", exporter="console")
        setup_telemetry(cfg)
        assert is_span_group_enabled("job") is True
        assert is_span_group_enabled("checkpoint") is True
        assert is_span_group_enabled("step") is False

    def test_enabled_registers_per_step_groups(self, demo_groups):
        cfg = NemoLensConfig(enabled=True, span_groups="per_step", exporter="console")
        setup_telemetry(cfg)
        assert is_span_group_enabled("step") is True
        assert is_span_group_enabled("forward_backward") is True

    def test_a_library_registering_after_setup_warns_but_still_works(self, demo_groups, caplog):
        """Registering late is a consumer import-order bug, not a lens feature.

        It is loud rather than fatal: refusing it would drop spans silently, and
        raising would let a telemetry misconfiguration kill a training job.
        """
        cfg = NemoLensConfig(enabled=True, span_groups="per_step", exporter="console")
        setup_telemetry(cfg)
        assert is_span_group_enabled("extra") is False

        with caplog.at_level("WARNING"):
            SpanRegistry.register("late", {"extra"}, {"per_step": {"extra"}})

        assert is_span_group_enabled("extra") is True
        assert "registered after setup_telemetry()" in caplog.text

    def test_a_typo_in_the_spec_warns_but_still_starts(self, demo_groups, caplog):
        cfg = NemoLensConfig(enabled=True, span_groups="per_stpe", exporter="console")
        with caplog.at_level("WARNING"):
            handle = setup_telemetry(cfg)
        assert handle.is_exporting is True
        assert "no library registered in this process provides" in caplog.text

    def test_a_process_missing_the_job_wide_vocabulary_still_gets_telemetry(self, caplog):
        """A launcher agent or spawned worker inherits one NEMO_LENS_SPAN_GROUPS
        from the trainer but imports a different set of libraries. The spec names
        the trainer's groups, which are absent here -- that must not cost this
        process its own telemetry, nor leave it exporting with no handle.
        """
        from opentelemetry import trace

        SpanRegistry.register("sidecar", {"sidecar.ft"}, {"default": {"sidecar.ft"}})
        cfg = NemoLensConfig(enabled=True, span_groups="per_step", exporter="console")
        with caplog.at_level("WARNING"):
            handle = setup_telemetry(cfg)

        assert handle.is_exporting is True
        assert hasattr(trace.get_tracer_provider(), "force_flush"), (
            "the caller must be able to bound its own flush"
        )
        assert "per_step" in caplog.text
        handle.shutdown(timeout_ms=100)

    def test_no_registered_library_does_not_crash_startup(self):
        """The default spec must survive a process with nothing instrumented."""
        cfg = NemoLensConfig(enabled=True, span_groups="default", exporter="console")
        handle = setup_telemetry(cfg)
        assert handle.is_exporting is True

    def test_a_disabled_process_stays_disabled_after_a_late_registration(self):
        cfg = NemoLensConfig(enabled=False, span_groups="all")
        setup_telemetry(cfg)
        SpanRegistry.register("late", {"step"})
        assert is_span_group_enabled("step") is False


class TestDoubleInitGuard:
    def test_double_init_raises(self):
        cfg = NemoLensConfig(enabled=True, exporter="console")
        setup_telemetry(cfg)
        with pytest.raises(RuntimeError, match="already been initialised"):
            setup_telemetry(cfg)

    def test_double_init_disabled_is_allowed(self):
        cfg = NemoLensConfig(enabled=False)
        setup_telemetry(cfg)
        handle = setup_telemetry(cfg)
        assert handle.is_exporting is False

    def test_allow_reinit_flag(self):
        cfg = NemoLensConfig(enabled=True, exporter="console")
        setup_telemetry(cfg)
        handle = setup_telemetry(cfg, _allow_reinit=True)
        assert handle is not None


class TestTelemetryHandleShutdown:
    def test_shutdown_idempotent(self):
        cfg = NemoLensConfig(enabled=False)
        handle = setup_telemetry(cfg)
        handle.shutdown(timeout_ms=100)
        handle.shutdown(timeout_ms=100)

    def test_second_shutdown_does_not_touch_the_providers_again(self, monkeypatch):
        """Idempotence has to be observable, not just "does not raise".

        A second pass would force-flush and shut down providers that are already
        down -- and, with the open-span closer registered, would give a second
        sweep a chance to end spans started after the first shutdown.
        """
        from opentelemetry import metrics, trace

        calls = []

        class _RecordingProvider:
            def __init__(self, label):
                self._label = label

            def force_flush(self, timeout_millis=None):
                calls.append(f"{self._label}.force_flush")
                return True

            def shutdown(self):
                calls.append(f"{self._label}.shutdown")

        cfg = NemoLensConfig(enabled=False)
        handle = setup_telemetry(cfg)
        monkeypatch.setattr(trace, "get_tracer_provider", lambda: _RecordingProvider("tracer"))
        monkeypatch.setattr(metrics, "get_meter_provider", lambda: _RecordingProvider("meter"))

        handle.shutdown(timeout_ms=100)
        assert calls == [
            "tracer.force_flush",
            "tracer.shutdown",
            "meter.force_flush",
            "meter.shutdown",
        ]

        calls.clear()
        handle.shutdown(timeout_ms=100)
        assert calls == []
