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

"""Unit tests for span utility helpers."""

import logging
from decimal import Decimal

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from nemo.lens.span_utilities import emit_span, linux_process_create_time
from nemo.lens.state import set_enabled_span_groups
from tests.conftest import InMemorySpanExporter


@pytest.fixture
def tracer_and_exporter():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    yield trace.get_tracer("test"), exporter
    provider.shutdown()


def test_emit_span_uses_explicit_times_context_and_attributes(tracer_and_exporter, caplog):
    tracer, exporter = tracer_and_exporter
    parent = tracer.start_span("parent")

    emit_span(
        tracer,
        "test.explicit_interval",
        1_700_000_000.25,
        1_700_000_001.5,
        context=trace.set_span_in_context(parent),
        attributes={"phase": "startup", "ignored": None},
    )
    parent.end()

    child, _parent = exporter.get_finished_spans()
    assert child.name == "test.explicit_interval"
    assert child.parent.span_id == parent.context.span_id
    assert child.start_time == 1_700_000_000_250_000_000
    assert child.end_time == 1_700_000_001_500_000_000
    assert child.attributes["phase"] == "startup"
    assert "ignored" not in child.attributes
    assert not caplog.records


def test_zero_duration_and_marker_parentage(tracer_and_exporter, caplog):
    tracer, exporter = tracer_and_exporter
    set_enabled_span_groups(frozenset({"test"}))
    with tracer.start_as_current_span("ambient") as ambient:
        marker = emit_span(
            None,
            "marker",
            1700000000.25,
            1700000000.25,
            group="test",
            attributes={"phase": "run", "password": "secret"},
        )
        child = emit_span(
            tracer, "child", 1700000000.25, 1700000001.5, context=trace.set_span_in_context(marker)
        )
        assert trace.get_current_span() is ambient
    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert spans["marker"].start_time == spans["marker"].end_time == 1700000000250000000
    assert spans["marker"].context == marker.get_span_context()
    assert spans["marker"].parent == ambient.get_span_context()
    assert spans["marker"].attributes == {"phase": "run", "password": "[REDACTED]"}
    assert spans["child"].parent == marker.get_span_context()
    assert child.get_span_context().trace_id == marker.get_span_context().trace_id
    assert not marker.is_recording()
    assert not child.is_recording()
    assert not caplog.records


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (1.0000000002, 1.0000000001),
        (1.004, 1.0),
        (1.03, 1.0),
        (1.05, 1.0),
        (2.0, 1.0),
        (1.0300000001, 1.0),
        (3600, 0),
        ("1790038547.061741440", "1790038547.061741439"),
    ],
)
def test_reversed_interval_always_clamps_and_records_skew(tracer_and_exporter, caplog, start, end):
    tracer, exporter = tracer_and_exporter
    completed = emit_span(tracer, "reversed", start, end)
    (span,) = exporter.get_finished_spans()
    assert span.start_time == span.end_time == int(Decimal(str(start)) * 1_000_000_000)
    assert not completed.is_recording()
    assert span.attributes == {
        "inverted.skew": True,
        "inverted.start_epoch_seconds": str(start),
        "inverted.end_epoch_seconds": str(end),
        "inverted.delta_seconds": str(Decimal(str(start)) - Decimal(str(end))),
    }
    (warning,) = caplog.records
    assert warning.name == "nemo.lens.span_utilities"
    assert warning.levelno == logging.WARNING
    assert "reversed" in warning.message
    assert f"start_epoch_seconds={start}" in warning.message
    assert f"end_epoch_seconds={end}" in warning.message
    assert f"inversion={Decimal(str(start)) - Decimal(str(end))} seconds" in warning.message
    assert "clamping end to start (zero duration)" in warning.message
    assert "possible clock skew" in warning.message


def test_inverted_parent_retains_children_and_copies_attributes(tracer_and_exporter):
    tracer, exporter = tracer_and_exporter
    attributes = {"phase": "launch", "inverted.skew": False}
    parent = emit_span(tracer, "parent", 20, 10, attributes=attributes)
    child = emit_span(tracer, "child", 21, 19, context=trace.set_span_in_context(parent))
    emit_span(tracer, "grandchild", 22, 23, context=trace.set_span_in_context(child))

    spans = {span.name: span for span in exporter.get_finished_spans()}
    assert len(spans) == 3
    assert spans["parent"].start_time == spans["parent"].end_time == 20_000_000_000
    assert spans["child"].start_time == spans["child"].end_time == 21_000_000_000
    assert spans["grandchild"].start_time == 22_000_000_000
    assert spans["grandchild"].end_time == 23_000_000_000
    assert spans["child"].parent == parent.get_span_context()
    assert spans["grandchild"].parent == child.get_span_context()
    assert len({span.context.trace_id for span in spans.values()}) == 1
    assert spans["parent"].attributes["phase"] == "launch"
    assert spans["parent"].attributes["inverted.skew"] is True
    assert spans["child"].attributes["inverted.skew"] is True
    assert "inverted.skew" not in spans["grandchild"].attributes
    assert attributes == {"phase": "launch", "inverted.skew": False}


def test_timed_group_gate_precedes_all_work(monkeypatch, caplog):
    def fail(*args, **kwargs):
        raise AssertionError("work before gate")

    monkeypatch.setattr("nemo.lens.span_utilities.trace.get_tracer", fail)
    monkeypatch.setattr("nemo.lens.span_utilities._finite_epoch_seconds", fail)
    assert emit_span(None, "off", object(), object(), group="off") is None
    assert not caplog.records


def test_timed_span_ends_when_attribute_processing_fails(tracer_and_exporter, monkeypatch):
    tracer, exporter = tracer_and_exporter

    def fail(*args):
        raise ValueError("attribute failure")

    monkeypatch.setattr("nemo.lens.span_utilities.safe_set_span_attributes", fail)
    with pytest.raises(ValueError, match="attribute failure"):
        emit_span(tracer, "failure", 1, 2, attributes={"a": 1})
    assert exporter.get_finished_spans()[0].end_time == 2000000000


@pytest.mark.parametrize(
    ("start", "end", "message"),
    [
        (float("nan"), 1.0, "start_epoch_seconds must be finite"),
        (1.0, float("inf"), "end_epoch_seconds must be finite"),
        (1.0, float("-inf"), "end_epoch_seconds must be finite"),
        ("invalid", 1.0, "start_epoch_seconds must be finite"),
        (1.0, None, "end_epoch_seconds must be finite"),
    ],
)
def test_emit_span_rejects_invalid_inputs(tracer_and_exporter, start, end, message):
    tracer, exporter = tracer_and_exporter
    with pytest.raises(ValueError, match=message):
        emit_span(tracer, "test.invalid", start, end)
    assert not exporter.get_finished_spans()


def test_linux_process_create_time_uses_start_ticks():
    stat_fields_after_comm = ["S", *(["0"] * 18), "250"]
    stat_text = "123 (python worker) " + " ".join(stat_fields_after_comm)

    assert (
        linux_process_create_time(
            stat_text=stat_text,
            uptime_text="1000.00 2000.00",
            read_time=1_700_000_000.0,
            clock_ticks_per_second=100,
        )
        == 1_699_999_002.5
    )


@pytest.mark.parametrize(
    ("stat_text", "uptime_text", "clock_ticks_per_second"),
    [
        ("123 (python worker) S", "1000.00 2000.00", 100),
        ("123 (python worker) " + " ".join(["S", *(["0"] * 18), "250"]), "", 100),
        ("123 (python worker) " + " ".join(["S", *(["0"] * 18), "250"]), "not-a-number", 100),
        ("123 (python worker) " + " ".join(["S", *(["0"] * 18), "250"]), "1000.00", 0),
    ],
)
def test_linux_process_create_time_raises_for_malformed_proc_data(
    stat_text,
    uptime_text,
    clock_ticks_per_second,
):
    with pytest.raises(ValueError, match="Malformed Linux process stat or uptime data"):
        linux_process_create_time(
            stat_text=stat_text,
            uptime_text=uptime_text,
            read_time=1_700_000_000.0,
            clock_ticks_per_second=clock_ticks_per_second,
        )


def test_linux_process_create_time_raises_when_proc_data_is_unavailable(tmp_path):
    with pytest.raises(RuntimeError, match="requires readable process stat and uptime data"):
        linux_process_create_time(
            stat_path=tmp_path / "missing-stat",
            uptime_path=tmp_path / "missing-uptime",
        )
