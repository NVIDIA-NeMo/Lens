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

"""Unit tests for telemetry helpers."""

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from nemo.lens.helpers import (
    DEFAULT_REDACT_KEYS,
    managed_span,
    redact_value,
    safe_set_span_attributes,
    span_cm,
    trace_fn,
)
from nemo.lens.state import set_enabled_span_groups
from tests.conftest import InMemorySpanExporter


@pytest.fixture
def tracer_and_exporter():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    tracer = trace.get_tracer("test")
    yield tracer, exporter
    provider.shutdown()


class TestRedactValue:
    def test_redacts_sensitive_key(self):
        assert redact_value("password", "secret123") == "[REDACTED]"

    def test_passes_through_normal_key(self):
        assert redact_value("iteration", "42") == "42"

    def test_custom_redact_keys(self):
        assert redact_value("custom", "val", frozenset({"custom"})) == "[REDACTED]"

    def test_default_keys(self):
        for key in DEFAULT_REDACT_KEYS:
            assert redact_value(key, "value") == "[REDACTED]"


class TestSafeSetSpanAttributes:
    def test_sets_scalar_attributes(self, tracer_and_exporter):
        tracer, exporter = tracer_and_exporter
        with tracer.start_as_current_span("test") as span:
            safe_set_span_attributes(span, {"count": 42, "name": "test", "rate": 3.14})
        spans = exporter.get_finished_spans()
        assert spans[0].attributes["count"] == 42
        assert spans[0].attributes["name"] == "test"
        assert spans[0].attributes["rate"] == 3.14

    def test_skips_none_values(self, tracer_and_exporter):
        tracer, exporter = tracer_and_exporter
        with tracer.start_as_current_span("test") as span:
            safe_set_span_attributes(span, {"key": None})
        spans = exporter.get_finished_spans()
        assert "key" not in spans[0].attributes

    def test_skips_complex_values(self, tracer_and_exporter):
        tracer, exporter = tracer_and_exporter
        with tracer.start_as_current_span("test") as span:
            safe_set_span_attributes(span, {"nested": {"a": 1}})
        spans = exporter.get_finished_spans()
        assert "nested" not in spans[0].attributes

    def test_redacts_sensitive_strings(self, tracer_and_exporter):
        tracer, exporter = tracer_and_exporter
        with tracer.start_as_current_span("test") as span:
            safe_set_span_attributes(span, {"password": "secret123"})
        spans = exporter.get_finished_spans()
        assert spans[0].attributes["password"] == "[REDACTED]"

    def test_list_of_scalars(self, tracer_and_exporter):
        tracer, exporter = tracer_and_exporter
        with tracer.start_as_current_span("test") as span:
            safe_set_span_attributes(span, {"tags": ["a", "b", "c"]})
        spans = exporter.get_finished_spans()
        assert spans[0].attributes["tags"] == ("a", "b", "c")


class TestSpanCm:
    def test_creates_span(self, tracer_and_exporter):
        tracer, exporter = tracer_and_exporter
        with span_cm("test.op", tracer=tracer, iteration=1) as span:
            assert span is not None
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "test.op"
        assert spans[0].attributes["iteration"] == 1

    def test_default_tracer(self, tracer_and_exporter):
        tracer, exporter = tracer_and_exporter
        with span_cm("test.default") as span:
            assert span is not None
        spans = exporter.get_finished_spans()
        assert len(spans) == 1


class TestManagedSpan:
    def test_disabled_group_yields_none(self):
        with managed_span("step", "test.step") as span:
            assert span is None

    def test_enabled_group_creates_span(self, tracer_and_exporter):
        tracer, exporter = tracer_and_exporter
        set_enabled_span_groups(frozenset(["step"]))
        with managed_span("step", "test.step", tracer=tracer, iteration=1) as span:
            assert span is not None
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "test.step"

    def test_exception_recorded(self, tracer_and_exporter):
        tracer, exporter = tracer_and_exporter
        set_enabled_span_groups(frozenset(["step"]))
        with (
            pytest.raises(ValueError, match="boom"),
            managed_span("step", "test.fail", tracer=tracer),
        ):
            raise ValueError("boom")
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].status.status_code.name == "ERROR"


class TestTraceFn:
    def test_disabled_group_runs_normally(self):
        @trace_fn("step", "test.fn")
        def my_func(x):
            return x * 2

        assert my_func(5) == 10

    def test_enabled_group_creates_span(self, tracer_and_exporter):
        tracer, exporter = tracer_and_exporter
        set_enabled_span_groups(frozenset(["step"]))

        @trace_fn("step", "test.fn")
        def my_func(x):
            return x * 2

        assert my_func(5) == 10
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "test.fn"
