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

"""Unit tests for distributed trace utilities."""

import pytest
from opentelemetry import propagate, trace
from opentelemetry.baggage.propagation import W3CBaggagePropagator
from opentelemetry.propagators.composite import CompositePropagator
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from nemo.lens.contrib.nccl import deserialize_context, extract_nccl_context, serialize_context
from nemo.lens.distributed import create_linked_span
from nemo.lens.propagation import inject_context
from tests.conftest import InMemorySpanExporter


@pytest.fixture
def setup_tracing():
    propagate.set_global_textmap(
        CompositePropagator([TraceContextTextMapPropagator(), W3CBaggagePropagator()])
    )
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    tracer = trace.get_tracer("test")
    yield tracer, exporter
    provider.shutdown()


class TestCreateLinkedSpan:
    def test_creates_span_with_link(self, setup_tracing):
        tracer, exporter = setup_tracing
        with tracer.start_as_current_span("remote") as remote_span:
            carrier = {}
            inject_context(carrier)

        span = create_linked_span(tracer, "local.linked", remote_carrier=carrier)
        span.end()

        spans = exporter.get_finished_spans()
        linked_span = [s for s in spans if s.name == "local.linked"][0]
        assert len(linked_span.links) == 1
        assert linked_span.links[0].context.trace_id == remote_span.get_span_context().trace_id

    def test_creates_span_without_link(self, setup_tracing):
        tracer, exporter = setup_tracing
        span = create_linked_span(tracer, "no.link")
        span.end()
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].links == ()

    def test_with_attributes(self, setup_tracing):
        tracer, exporter = setup_tracing
        span = create_linked_span(tracer, "with.attrs", rank=0, stage=2)
        span.end()
        spans = exporter.get_finished_spans()
        assert spans[0].attributes["rank"] == 0
        assert spans[0].attributes["stage"] == 2


class TestNCCLContextPropagation:
    def test_serialize_deserialize_roundtrip(self, setup_tracing):
        tracer, _ = setup_tracing
        with tracer.start_as_current_span("origin"):
            data = serialize_context()

        carrier = deserialize_context(data)
        assert carrier is not None
        assert "traceparent" in carrier

    def test_deserialize_invalid_data(self):
        result = deserialize_context(b"not json")
        assert result is None

    def test_extract_nccl_context(self, setup_tracing):
        tracer, _ = setup_tracing
        with tracer.start_as_current_span("origin") as span:
            original_trace_id = span.get_span_context().trace_id
            data = serialize_context()

        ctx = extract_nccl_context(data)
        remote_span = trace.get_current_span(ctx)
        assert remote_span.get_span_context().trace_id == original_trace_id

    def test_extract_nccl_invalid_returns_current(self):
        ctx = extract_nccl_context(b"bad data")
        assert ctx is not None
