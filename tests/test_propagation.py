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

"""Unit tests for W3C context propagation."""

import pytest
from opentelemetry import propagate, trace
from opentelemetry.baggage.propagation import W3CBaggagePropagator
from opentelemetry.propagators.composite import CompositePropagator
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from nemo.lens.propagation import extract_context, inject_context
from tests.conftest import InMemorySpanExporter


@pytest.fixture
def setup_propagator():
    propagate.set_global_textmap(
        CompositePropagator([TraceContextTextMapPropagator(), W3CBaggagePropagator()])
    )


@pytest.fixture
def tracer():
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    yield trace.get_tracer("test")
    provider.shutdown()


class TestInjectContext:
    def test_injects_traceparent(self, setup_propagator, tracer):
        with tracer.start_as_current_span("root"):
            carrier = {}
            inject_context(carrier)
            assert "traceparent" in carrier
            parts = carrier["traceparent"].split("-")
            assert len(parts) == 4
            assert parts[0] == "00"


class TestExtractContext:
    def test_extract_valid_context(self, setup_propagator, tracer):
        with tracer.start_as_current_span("root") as span:
            carrier = {}
            inject_context(carrier)
            trace_id = span.get_span_context().trace_id

        ctx = extract_context(carrier)
        remote_span = trace.get_current_span(ctx)
        assert remote_span.get_span_context().trace_id == trace_id

    def test_roundtrip_preserves_ids(self, setup_propagator, tracer):
        with tracer.start_as_current_span("root") as span:
            original_ctx = span.get_span_context()
            carrier = {}
            inject_context(carrier)

        ctx = extract_context(carrier)
        remote_ctx = trace.get_current_span(ctx).get_span_context()
        assert remote_ctx.trace_id == original_ctx.trace_id
        assert remote_ctx.span_id == original_ctx.span_id

    def test_extract_empty_carrier(self, setup_propagator):
        ctx = extract_context({})
        span = trace.get_current_span(ctx)
        assert not span.get_span_context().is_valid
