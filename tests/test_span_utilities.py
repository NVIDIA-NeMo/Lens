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

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from nemo.lens.span_utilities import emit_span, linux_process_create_time
from tests.conftest import InMemorySpanExporter


@pytest.fixture
def tracer_and_exporter():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    yield trace.get_tracer("test"), exporter
    provider.shutdown()


def test_emit_span_uses_explicit_times_context_and_attributes(tracer_and_exporter):
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


def test_emit_span_rejects_end_before_start(tracer_and_exporter):
    tracer, _exporter = tracer_and_exporter

    with pytest.raises(ValueError, match="end_epoch_seconds"):
        emit_span(tracer, "test.invalid", 2.0, 1.0)


@pytest.mark.parametrize(
    ("start_epoch_seconds", "end_epoch_seconds"),
    [
        (float("nan"), 1.0),
        (1.0, float("inf")),
    ],
)
def test_emit_span_rejects_non_finite_timestamps(
    tracer_and_exporter,
    start_epoch_seconds,
    end_epoch_seconds,
):
    tracer, _exporter = tracer_and_exporter

    with pytest.raises(ValueError, match="must be finite"):
        emit_span(tracer, "test.invalid", start_epoch_seconds, end_epoch_seconds)


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
