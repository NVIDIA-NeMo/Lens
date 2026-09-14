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

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from opentelemetry import trace

from nemo.lens.config import NemoLensConfig
from nemo.lens.groups import SpanRegistry
from nemo.lens.handle import TelemetryHandle, ensure_run_id, setup_telemetry
from nemo.lens.resources.attributes import parse_otel_resource_attributes
from nemo.lens.semconv import NV_DL_RANK, NV_DL_ROLE, NV_DL_WORLD_SIZE
from nemo.lens.state import is_span_group_enabled

REPO_ROOT = Path(__file__).resolve().parents[1]


# Independent inventory: committed schema 89182747, nv.dl.yaml / nv.gpu.yaml.
_CARRIED_INTEGERS = {
    "nv.dl.rank": 7,
    "nv.dl.world_size": 16,
    "nv.dl.local_rank": 3,
    "nv.dl.topology.size.tp": 2,
    "nv.dl.topology.size.pp": 4,
    "nv.dl.topology.size.dp": 2,
    "nv.dl.training.config.global_batch_size": 128,
    "nv.dl.training.config.micro_batch_size": 1,
    "nv.dl.training.config.sequence_length": 8192,
    "nv.dl.training.target.train_iters": 1000,
    "nv.dl.training.target.train_samples": 128000,
    "nv.dl.training.target.train_tokens": 2**53 + 1,
    "nv.gpu.index": 3,
    "nv.gpu.memory_total": 85899345920,
    "slurm.array.count": 1,
    "slurm.nnodes": 2,
    "slurm.ntasks": 16,
    "slurm.restart_count": 0,
}


@pytest.mark.parametrize("value", [True, 3.5, "3.5", "true", "1_000"])
def test_integer_constructor_rejects_non_integer_values(value):
    from nemo.lens.resources.attributes import _resource_integer

    with pytest.raises(ValueError):
        _resource_integer(value)


def test_normalizer_preserves_native_integers_and_string_fields():
    from nemo.lens.resources.attributes import _normalize_resource_attributes

    attrs = {
        **_CARRIED_INTEGERS,
        "nv.dl.job.uuid": "123",
        "nv.dl.run.uuid": "456",
        "nemo.run.id": "007",
        "slurm.job.id": "00123",
        "nv.dl.software.cuda": "12.8",
        "nv.gpu.compute_capability": "9.0",
        "nv.gpu.serial": "001234",
        "app.count": "123",
        "nv.dl.topology.size.cp": "2",
    }
    result = _normalize_resource_attributes(attrs)
    assert result == attrs
    assert {key: type(value) for key, value in result.items()} == {
        key: type(value) for key, value in attrs.items()
    }


def test_exported_resource_types_survive_subprocess(monkeypatch):
    from tests.conftest import InMemorySpanExporter

    previous = "app.code=007,process.pid=999,host.name=stale"
    monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", previous)
    exporter = InMemorySpanExporter()
    config = NemoLensConfig(enabled=True, metrics_enabled=False, service_name="trainer")
    handle = setup_telemetry(
        config,
        resource_attributes=_CARRIED_INTEGERS,
        publish_resource_attributes=True,
        span_exporter=exporter,
    )
    try:
        with handle.tracer.start_as_current_span("parent"):
            pass
        trace.get_tracer_provider().force_flush()
        parent = exporter.get_finished_spans()[0].resource.attributes
        carrier = parse_otel_resource_attributes(os.environ["OTEL_RESOURCE_ATTRIBUTES"])
        for key, value in _CARRIED_INTEGERS.items():
            assert parent[key] == value and type(parent[key]) is int
            assert carrier[key] == str(value) and type(carrier[key]) is str
        child_code = """
import json, os
from opentelemetry import trace
from nemo.lens import NemoLensConfig, setup_telemetry
from nemo.lens.resources.attributes import parse_otel_resource_attributes
from tests.conftest import InMemorySpanExporter
exporter = InMemorySpanExporter()
before = os.environ['OTEL_RESOURCE_ATTRIBUTES']
handle = setup_telemetry(
    NemoLensConfig(enabled=True, metrics_enabled=False, service_name='nvrx.ckpt_worker'),
    resource_attribute_defaults={'nv.dl.rank': 99},
    resource_attributes={'nv.dl.role': 'ckpt_worker'},
    publish_resource_attributes=True, span_exporter=exporter,
)
with handle.tracer.start_as_current_span('child'):
    pass
trace.get_tracer_provider().force_flush()
attrs = dict(exporter.get_finished_spans()[0].resource.attributes)
carrier = parse_otel_resource_attributes(os.environ['OTEL_RESOURCE_ATTRIBUTES'])
handle.shutdown()
assert os.environ['OTEL_RESOURCE_ATTRIBUTES'] == before
print(json.dumps({'attrs': attrs, 'carrier': carrier, 'pid': os.getpid()}))
"""
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [str(REPO_ROOT / "src"), str(REPO_ROOT), env.get("PYTHONPATH", "")]
        )
        result = subprocess.run(
            [sys.executable, "-c", child_code],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        child = json.loads(result.stdout.strip().splitlines()[-1])
        for key, value in _CARRIED_INTEGERS.items():
            assert child["attrs"][key] == value and type(child["attrs"][key]) is int
            assert child["carrier"][key] == str(value)
        assert child["attrs"]["service.name"] == "nvrx.ckpt_worker"
        assert child["attrs"]["nv.dl.role"] == "ckpt_worker"
        assert child["attrs"]["app.code"] == "007"
        assert child["attrs"]["process.pid"] == child["pid"]
        assert "process.pid" not in child["carrier"]
        assert "host.name" not in child["carrier"]
    finally:
        handle.shutdown()
    assert os.environ["OTEL_RESOURCE_ATTRIBUTES"] == previous


@pytest.mark.parametrize("raw,expected", [("7", 7), (" +007 ", 7), ("-1", -1), ("0", 0)])
@pytest.mark.parametrize("explicit", [None, 11])
def test_inherited_integer_precedence(monkeypatch, raw, expected, explicit):
    from tests.conftest import InMemorySpanExporter

    monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", f"nv.dl.rank={raw}")
    exporter = InMemorySpanExporter()
    handle = setup_telemetry(
        NemoLensConfig(enabled=True, metrics_enabled=False),
        resource_attribute_defaults={"nv.dl.rank": 3},
        resource_attributes={"nv.dl.rank": explicit},
        span_exporter=exporter,
    )
    try:
        with handle.tracer.start_as_current_span("rank"):
            pass
        trace.get_tracer_provider().force_flush()
        rank = exporter.get_finished_spans()[0].resource.attributes["nv.dl.rank"]
        assert rank == (expected if explicit is None else explicit)
        assert type(rank) is int
    finally:
        handle.shutdown()


@pytest.mark.parametrize("raw", ["3.5", "true", "1e3", "1_000", "--1", "seven"])
def test_invalid_inherited_integer_is_visible(monkeypatch, caplog, raw):
    from tests.conftest import InMemorySpanExporter

    monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", f"nv.dl.rank={raw}")
    exporter = InMemorySpanExporter()
    handle = setup_telemetry(
        NemoLensConfig(enabled=True, metrics_enabled=False),
        resource_attribute_defaults={"nv.dl.rank": 3},
        span_exporter=exporter,
    )
    try:
        with handle.tracer.start_as_current_span("invalid"):
            pass
        trace.get_tracer_provider().force_flush()
        assert exporter.get_finished_spans()[0].resource.attributes["nv.dl.rank"] == raw
        assert "nv.dl.rank requires a base-10 integer" in caplog.text
    finally:
        handle.shutdown()


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

    def test_does_not_detect_or_publish_resources(self, monkeypatch):
        import nemo.lens.resources as resources

        def fail(*args, **kwargs):
            raise AssertionError("disabled telemetry must not detect resources")

        monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", "launcher.key=unchanged")
        monkeypatch.setattr(resources, "detect_gpu", fail)
        monkeypatch.setattr(resources, "detect_kubernetes", fail)
        monkeypatch.setattr(resources, "detect_local", fail)
        monkeypatch.setattr(resources, "detect_slurm", fail)

        cfg = NemoLensConfig(enabled=False)
        handle = setup_telemetry(
            cfg,
            resource_attribute_defaults={NV_DL_RANK: 3},
            publish_resource_attributes=True,
        )

        assert handle.resource_attributes == {}
        assert os.environ["OTEL_RESOURCE_ATTRIBUTES"] == "launcher.key=unchanged"


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

    @pytest.mark.parametrize("inherited", [False, True])
    def test_service_name_carrier_agrees_without_expanding_keys(self, monkeypatch, inherited):
        from opentelemetry import trace

        previous = "custom.key=with%20space"
        if inherited:
            previous += ",service.name=pretraining"
        monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", previous)
        config = NemoLensConfig(
            enabled=True, metrics_enabled=False, service_name="worker", exporter="console"
        )
        handle = setup_telemetry(config, publish_resource_attributes=True)
        try:
            assert trace.get_tracer_provider().resource.attributes["service.name"] == "worker"
            carrier = parse_otel_resource_attributes(os.environ["OTEL_RESOURCE_ATTRIBUTES"])
            if inherited:
                assert (
                    carrier["service.name"]
                    == handle.resource_attributes["service.name"]
                    == "worker"
                )
            else:
                assert "service.name" not in carrier
                assert "service.name" not in handle.resource_attributes
        finally:
            handle.shutdown()
        assert os.environ["OTEL_RESOURCE_ATTRIBUTES"] == previous

    def test_publishes_resolved_resource_map_for_handle_lifetime(self, monkeypatch):
        previous = "launcher.key=with%20space,nv.dl.rank=launcher-rank,nv.dl.role=worker"
        monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", previous)
        cfg = NemoLensConfig(
            enabled=True,
            exporter="console",
            traces_enabled=False,
            metrics_enabled=False,
            run_id="run-1",
        )

        handle = setup_telemetry(
            cfg,
            resource_attribute_defaults={NV_DL_RANK: 3, NV_DL_WORLD_SIZE: 8},
            resource_attributes={NV_DL_ROLE: "trainer"},
            local_rank=0,
            publish_resource_attributes=True,
        )

        published = parse_otel_resource_attributes(os.environ["OTEL_RESOURCE_ATTRIBUTES"])
        assert published == {key: str(value) for key, value in handle.resource_attributes.items()}
        assert published[NV_DL_RANK] == "launcher-rank"
        assert published[NV_DL_ROLE] == "trainer"
        assert published["launcher.key"] == "with space"

        handle.shutdown()
        handle.shutdown()
        assert os.environ["OTEL_RESOURCE_ATTRIBUTES"] == previous

    def test_stale_local_identity_is_replaced_and_not_inherited_by_child(self, monkeypatch):
        from opentelemetry import trace

        local_keys = {"host.name", "host.gpu.count", "process.pid"}
        monkeypatch.setenv(
            "OTEL_RESOURCE_ATTRIBUTES",
            "process.pid=999,host.name=ancestor,host.gpu.count=99,nv.dl.rank=0",
        )
        cfg = NemoLensConfig(
            enabled=True,
            exporter="console",
            metrics_enabled=False,
            run_id="parent",
        )

        handle = setup_telemetry(cfg, publish_resource_attributes=True)

        parent_resource = dict(trace.get_tracer_provider().resource.attributes)
        published = parse_otel_resource_attributes(os.environ["OTEL_RESOURCE_ATTRIBUTES"])
        assert parent_resource["process.pid"] == os.getpid()
        assert local_keys.isdisjoint(published)

        child_code = """
import json
import os
from opentelemetry import trace
from nemo.lens import NemoLensConfig, setup_telemetry
from nemo.lens.resources.attributes import parse_otel_resource_attributes

config = NemoLensConfig(
    enabled=True,
    exporter="console",
    metrics_enabled=False,
    run_id="child",
)
handle = setup_telemetry(config, publish_resource_attributes=True)
resource = dict(trace.get_tracer_provider().resource.attributes)
carrier = parse_otel_resource_attributes(os.environ["OTEL_RESOURCE_ATTRIBUTES"])
print(json.dumps({"pid": os.getpid(), "resource_pid": resource["process.pid"], "carrier": carrier}))
handle.shutdown()
"""
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT / "src"), env.get("PYTHONPATH", "")])
        result = subprocess.run(
            [sys.executable, "-c", child_code],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0, result.stderr
        child = json.loads(result.stdout.strip().splitlines()[-1])
        assert child["resource_pid"] == child["pid"]
        assert local_keys.isdisjoint(child["carrier"])

        handle.shutdown()
        assert os.environ["OTEL_RESOURCE_ATTRIBUTES"].startswith("process.pid=999,")

    def test_publication_failure_restores_previous_environment(self, monkeypatch):
        import nemo.lens.resources.attributes as attributes

        previous = "launcher.key=original"
        monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", previous)

        def fail_after_mutation(additions, *, environ=None, overwrite=False, exclude=()):
            os.environ["OTEL_RESOURCE_ATTRIBUTES"] = "partially=changed"
            raise RuntimeError("publication failed")

        monkeypatch.setattr(attributes, "set_otel_resource_attributes", fail_after_mutation)
        cfg = NemoLensConfig(
            enabled=True,
            exporter="console",
            traces_enabled=False,
            metrics_enabled=False,
        )

        with pytest.raises(RuntimeError, match="publication failed"):
            setup_telemetry(
                cfg,
                resource_attribute_defaults={NV_DL_RANK: 0},
                publish_resource_attributes=True,
            )

        assert os.environ["OTEL_RESOURCE_ATTRIBUTES"] == previous


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
