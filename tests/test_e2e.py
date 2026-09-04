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

"""End-to-end integration tests for nemo-lens."""

from opentelemetry import trace

from nemo.lens import (
    NemoLensConfig,
    TelemetryHandle,
    extract_context,
    get_tracer,
    inject_context,
    managed_span,
    setup_telemetry,
)


class TestE2EConsoleExporter:
    def test_full_lifecycle(self, demo_groups):
        """Test complete setup -> use -> shutdown lifecycle."""
        cfg = NemoLensConfig(enabled=True, exporter="console", span_groups="all")
        handle = setup_telemetry(cfg)

        assert isinstance(handle, TelemetryHandle)
        assert handle.is_exporting is True

        # Create spans
        with managed_span("job", "test.job", tracer=handle.tracer) as span:
            assert span is not None
            with managed_span("step", "test.step", tracer=handle.tracer) as step_span:
                assert step_span is not None

        handle.shutdown(timeout_ms=100)

    def test_disabled_zero_overhead(self, demo_groups):
        """When disabled, no spans should be created."""
        cfg = NemoLensConfig(enabled=False, span_groups="all")
        handle = setup_telemetry(cfg)

        assert handle.is_exporting is False

        with managed_span("job", "test.job") as span:
            assert span is None

        handle.shutdown(timeout_ms=100)


class TestE2ESpanHierarchy:
    def test_nested_spans(self, demo_groups):
        cfg = NemoLensConfig(enabled=True, exporter="console", span_groups="all")
        handle = setup_telemetry(cfg)

        with managed_span("job", "dl.train", tracer=handle.tracer) as job:
            assert job is not None
            with managed_span("step", "dl.train_step", tracer=handle.tracer, iteration=1) as step:
                assert step is not None
                with managed_span(
                    "forward_backward", "dl.forward_backward", tracer=handle.tracer
                ) as fb:
                    assert fb is not None

        handle.shutdown(timeout_ms=100)


class TestE2EContextPropagation:
    def test_inject_extract_roundtrip(self):
        cfg = NemoLensConfig(enabled=True, exporter="console", span_groups="default")
        handle = setup_telemetry(cfg)

        tracer = get_tracer()
        with tracer.start_as_current_span("origin") as span:
            carrier = {}
            inject_context(carrier)
            assert "traceparent" in carrier

        ctx = extract_context(carrier)
        remote_span = trace.get_current_span(ctx)
        assert remote_span.get_span_context().trace_id == span.get_span_context().trace_id

        handle.shutdown(timeout_ms=100)
