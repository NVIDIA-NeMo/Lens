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

"""Unit tests for nemo.lens.sandbox.observability."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from nemo.lens.sandbox.observability.recorder import (
    SandboxEventRecorder,
    build_recorder_from_env,
    event_context,
    observability_span,
    observability_sync_span,
    observability_suppressed,
    suppress_observability_events,
    use_recorder,
)
from nemo.lens.sandbox.observability.traces import export_trace_artifacts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_recorder(tmp_path: Path, **kwargs) -> SandboxEventRecorder:
    return SandboxEventRecorder(
        output_dir=tmp_path / "obs",
        resource_sample_interval_s=0,
        max_rendered_trajectories=10,
        artifacts={
            "enabled": False,
            "render_html": False,
            "render_png": False,
            "export_otlp_json": False,
        },
        otel={"service_name": "test-service", "export_logs": False},
        wandb={"enabled": False},
        process_trace={"enabled": False},
        privacy={"include_command_text": False},
        run_id="test-run",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Recorder basics
# ---------------------------------------------------------------------------


def test_recorder_appends_to_events_jsonl(tmp_path):
    rec = _make_recorder(tmp_path)
    rec.record_event("test", "my.event", attributes={"key": "value"})
    events_path = tmp_path / "obs" / "events.jsonl"
    assert events_path.exists()
    lines = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
    names = [row["name"] for row in lines]
    assert "run.start" in names
    assert "my.event" in names


def test_span_records_start_and_end_with_duration(tmp_path):
    import asyncio

    rec = _make_recorder(tmp_path)

    async def _run():
        async with rec.span("test.operation", phase="execution"):
            pass

    asyncio.run(_run())
    events_path = tmp_path / "obs" / "events.jsonl"
    lines = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
    by_name = {row["name"]: row for row in lines}
    assert "test.operation" in by_name
    end_event = by_name["test.operation"]
    assert end_event["event_type"] == "span_end"
    assert isinstance((end_event.get("attributes") or {}).get("duration_s"), float)


def test_sync_span_records_start_and_end(tmp_path):
    rec = _make_recorder(tmp_path)
    with rec.sync_span("sync.op", phase="setup"):
        pass
    events_path = tmp_path / "obs" / "events.jsonl"
    lines = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
    types_by_name = {row["name"]: row["event_type"] for row in lines if row["name"] == "sync.op"}
    assert "sync.op" in types_by_name
    assert types_by_name["sync.op"] == "span_end"


def test_suppress_observability_events_prevents_recording(tmp_path):
    rec = _make_recorder(tmp_path)
    with suppress_observability_events():
        assert observability_suppressed()


def test_suppress_observability_events_restores_after(tmp_path):
    with suppress_observability_events():
        pass
    assert not observability_suppressed()


def test_event_context_merges_attributes(tmp_path):
    rec = _make_recorder(tmp_path)
    with use_recorder(rec):
        with event_context(trajectory_id="traj-123"):
            rec.record_event("test", "ctx.event")
    events_path = tmp_path / "obs" / "events.jsonl"
    lines = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
    ctx_events = [row for row in lines if row["name"] == "ctx.event"]
    assert ctx_events
    assert ctx_events[0]["attributes"]["trajectory_id"] == "traj-123"


def test_ingest_jsonl_ingests_external_events(tmp_path):
    rec = _make_recorder(tmp_path)
    external = tmp_path / "external.jsonl"
    external.write_text(
        json.dumps({"event_type": "span_end", "name": "external.op", "attributes": {"phase": "setup"}})
        + "\n",
        encoding="utf-8",
    )
    count = rec.ingest_jsonl(external, source="test-agent", trajectory_id="traj-abc")
    assert count == 1
    events = [
        json.loads(line)
        for line in (tmp_path / "obs" / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    ingested = [e for e in events if e.get("name") == "external.op"]
    assert ingested
    assert ingested[0]["attributes"]["source"] == "test-agent"
    assert ingested[0]["attributes"]["trajectory_id"] == "traj-abc"


# ---------------------------------------------------------------------------
# Trace artifact export
# ---------------------------------------------------------------------------


def test_export_trace_artifacts_otlp_schema(tmp_path):
    rec = _make_recorder(tmp_path)
    with rec.sync_span("llm.request", phase="llm", attributes={"trajectory_id": "t1"}):
        pass
    artifacts = export_trace_artifacts(tmp_path / "obs", service_name="test-svc", run_id="r1")
    assert "otlp_json" in artifacts
    otlp = json.loads(Path(artifacts["otlp_json"]).read_text())
    assert "resourceSpans" in otlp
    resource_spans = otlp["resourceSpans"]
    assert isinstance(resource_spans, list) and resource_spans
    scope_spans = resource_spans[0]["scopeSpans"]
    assert isinstance(scope_spans, list) and scope_spans
    spans = scope_spans[0]["spans"]
    assert isinstance(spans, list) and spans


def test_export_trace_artifacts_chrome_keys(tmp_path):
    rec = _make_recorder(tmp_path)
    with rec.sync_span("trajectory.tool", phase="agent_tool", attributes={"trajectory_id": "t2"}):
        pass
    artifacts = export_trace_artifacts(tmp_path / "obs", service_name="test-svc", run_id="r2")
    assert "chrome_trace" in artifacts
    chrome = json.loads(Path(artifacts["chrome_trace"]).read_text())
    assert "traceEvents" in chrome
    assert chrome.get("displayTimeUnit") == "ms"


def test_export_trace_artifacts_returns_empty_for_no_events(tmp_path):
    (tmp_path / "obs").mkdir(parents=True, exist_ok=True)
    result = export_trace_artifacts(tmp_path / "obs", service_name="test", run_id=None)
    assert result == {}


# ---------------------------------------------------------------------------
# _OtelSink: nemo-lens import guard
# ---------------------------------------------------------------------------


def test_otel_sink_noop_when_nemo_lens_missing(tmp_path):
    """_OtelSink must not raise even when nemo-lens import fails."""
    import sys
    from nemo.lens.sandbox.observability import recorder as recorder_mod

    original = sys.modules.get("nemo.lens")
    # Simulate ImportError by patching setup_telemetry import inside recorder
    with patch.dict(sys.modules, {"nemo.lens": None}):
        sink = recorder_mod._OtelSink({"service_name": "test", "export_logs": False})
    # Sink should be no-op — no exception, no lens handle
    assert sink._lens_handle is None
    assert sink._counter is None


def test_otel_sink_calls_setup_telemetry(tmp_path):
    """When nemo-lens is available, _OtelSink calls setup_telemetry once."""
    from nemo.lens.sandbox.observability import recorder as recorder_mod

    mock_config = MagicMock()
    mock_handle = MagicMock()
    mock_setup = MagicMock(return_value=mock_handle)
    mock_config_cls = MagicMock(return_value=mock_config)
    mock_from_env = MagicMock(return_value=mock_config)
    mock_config_cls.from_env = mock_from_env

    fake_lens = MagicMock()
    fake_lens.NemoLensConfig = mock_config_cls
    fake_lens.setup_telemetry = mock_setup

    with patch.dict("sys.modules", {"nemo.lens": fake_lens}):
        with patch("opentelemetry.metrics.get_meter") as mock_get_meter:
            mock_meter = MagicMock()
            mock_get_meter.return_value = mock_meter
            mock_meter.create_histogram.return_value = MagicMock()
            mock_meter.create_counter.return_value = MagicMock()
            sink = recorder_mod._OtelSink({"service_name": "test", "export_logs": False})

    mock_setup.assert_called_once_with(mock_config, rank=0, world_size=1)
    assert sink._lens_handle is mock_handle


def test_otel_sink_shutdown_calls_lens_handle(tmp_path):
    """shutdown() must call _lens_handle.shutdown()."""
    from nemo.lens.sandbox.observability import recorder as recorder_mod

    mock_handle = MagicMock()
    fake_lens = MagicMock()
    fake_lens.NemoLensConfig.from_env.return_value = MagicMock()
    fake_lens.setup_telemetry.return_value = mock_handle

    with patch.dict("sys.modules", {"nemo.lens": fake_lens}):
        with patch("opentelemetry.metrics.get_meter") as mock_get_meter:
            mock_meter = MagicMock()
            mock_get_meter.return_value = mock_meter
            mock_meter.create_histogram.return_value = MagicMock()
            mock_meter.create_counter.return_value = MagicMock()
            sink = recorder_mod._OtelSink({"service_name": "test", "export_logs": False})

    sink.shutdown()
    mock_handle.shutdown.assert_called_once()


# ---------------------------------------------------------------------------
# build_recorder_from_env
# ---------------------------------------------------------------------------


def test_build_recorder_from_env_returns_none_without_env_var(tmp_path, monkeypatch):
    monkeypatch.delenv("NEMO_RL_SANDBOX_OBSERVABILITY_DIR", raising=False)
    assert build_recorder_from_env() is None


def test_build_recorder_from_env_creates_recorder(tmp_path, monkeypatch):
    obs_dir = tmp_path / "env_obs"
    monkeypatch.setenv("NEMO_RL_SANDBOX_OBSERVABILITY_DIR", str(obs_dir))
    rec = build_recorder_from_env()
    assert rec is not None
    assert rec.output_dir == obs_dir
    rec.finalize()


# ---------------------------------------------------------------------------
# Trajectory trace extraction
# ---------------------------------------------------------------------------


def test_trajectory_trace_extracts_tool_spans(tmp_path):
    from nemo.lens.sandbox.observability.trajectory_trace import extract_agent_tool_spans

    trajectory = {
        "messages": [
            {
                "type": "response",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "c1",
                        "name": "bash",
                        "arguments": '{"command": "ls /"}',
                    }
                ],
                "extra": {"timestamp": 1700000000.0},
            },
            {
                "type": "function_call_output",
                "call_id": "c1",
                "output": "bin  etc  usr\n<returncode>0</returncode>",
                "extra": {"timestamp": 1700000001.5, "returncode": 0},
            },
        ]
    }
    path = tmp_path / "trajectory.json"
    path.write_text(json.dumps(trajectory), encoding="utf-8")
    spans = extract_agent_tool_spans(path, trajectory_id="t1", include_command_text=False)
    assert spans
    attrs = spans[0]["attributes"]
    assert attrs["trajectory_id"] == "t1"
    assert attrs["tool_name"] == "bash"
    assert attrs["return_code"] == 0
    assert "command_hash" in attrs
    assert "command_text" not in attrs
