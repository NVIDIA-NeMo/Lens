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

"""Unit tests for optional integration helpers."""

import sys
import types

import pytest

from nemo.lens.contrib.aiohttp import instrument_aiohttp_client
from nemo.lens.contrib.fastapi import instrument_fastapi
from nemo.lens.contrib.ray import (
    extract_ray_context,
    inject_ray_context,
    ray_dispatch_with_context,
    traced_remote_call,
)


def test_aiohttp_instrumentor_is_invoked(monkeypatch):
    """Verify aiohttp instrumentation delegates to the optional OTel instrumentor."""
    calls = []
    module = types.ModuleType("opentelemetry.instrumentation.aiohttp_client")

    class AioHttpClientInstrumentor:
        def instrument(self):
            calls.append("instrument")

    module.AioHttpClientInstrumentor = AioHttpClientInstrumentor
    monkeypatch.setitem(sys.modules, module.__name__, module)

    instrument_aiohttp_client()

    assert calls == ["instrument"]


def test_aiohttp_missing_dependency_raises_helpful_error(monkeypatch):
    """Verify missing aiohttp instrumentation dependencies raise install guidance."""
    monkeypatch.setitem(sys.modules, "opentelemetry.instrumentation.aiohttp_client", None)

    with pytest.raises(ImportError, match="nemo-lens\\[aiohttp\\]"):
        instrument_aiohttp_client()


def test_fastapi_instrumentor_is_invoked(monkeypatch):
    """Verify FastAPI instrumentation delegates to the optional OTel instrumentor."""
    calls = []
    module = types.ModuleType("opentelemetry.instrumentation.fastapi")

    class FastAPIInstrumentor:
        @staticmethod
        def instrument_app(app, tracer_provider=None):
            calls.append((app, tracer_provider))

    module.FastAPIInstrumentor = FastAPIInstrumentor
    monkeypatch.setitem(sys.modules, module.__name__, module)

    app = object()
    instrument_fastapi(app, service_name="my-service")

    assert len(calls) == 1
    assert calls[0][0] is app
    assert calls[0][1] is not None
    assert calls[0][1].resource.attributes["service.name"] == "my-service"


def test_fastapi_missing_dependency_raises_helpful_error(monkeypatch):
    """Verify missing FastAPI instrumentation dependencies raise install guidance."""
    monkeypatch.setitem(sys.modules, "opentelemetry.instrumentation.fastapi", None)

    with pytest.raises(ImportError, match="nemo-lens\\[fastapi\\]"):
        instrument_fastapi(object())


def test_ray_context_helpers_roundtrip():
    """Verify Ray carrier helpers produce and extract usable OTel contexts."""
    carrier = inject_ray_context()

    assert isinstance(carrier, dict)
    assert extract_ray_context(carrier) is not None
    assert extract_ray_context(None) is not None


def test_traced_remote_call_preserves_wrapped_function_name():
    """Verify traced Ray wrappers preserve metadata and call the wrapped function."""
    calls = []

    def work(value, scale=1):
        calls.append((value, scale))
        return value * scale

    wrapped = traced_remote_call(work)

    assert wrapped.__name__ == "work"
    assert wrapped(3, scale=2) == 6
    assert calls == [(3, 2)]


def test_ray_dispatch_injects_context():
    """Verify Ray dispatch adds trace context to the remote call kwargs."""

    class RemoteFunction:
        def __init__(self):
            self.call = None

        def remote(self, *args, **kwargs):
            self.call = (args, kwargs)
            return "object-ref"

    remote_fn = RemoteFunction()

    assert ray_dispatch_with_context(remote_fn, 1, key="value") == "object-ref"
    assert remote_fn.call[0] == (1,)
    assert remote_fn.call[1]["key"] == "value"
    assert "_otel_carrier" in remote_fn.call[1]
