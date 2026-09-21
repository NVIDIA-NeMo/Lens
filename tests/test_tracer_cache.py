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

"""Default span helpers must not repeatedly invalidate warning registries."""

import multiprocessing
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from nemo.lens import _tracers, get_tracer, managed_span, span_cm, trace_fn
from nemo.lens.contrib.ray import traced_remote_call
from nemo.lens.span_utilities import emit_span
from nemo.lens.state import set_enabled_span_groups


def _warning_site():
    warnings.warn("representative PyTorch collective warning", FutureWarning, stacklevel=1)


@pytest.mark.parametrize("kind", ["span_cm", "managed_span", "trace_fn", "emit_span", "ray"])
def test_repeated_spans_preserve_warning_deduplication(kind):
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    set_enabled_span_groups(frozenset({"test"}))

    if kind == "span_cm":

        def operation():
            with span_cm("test"):
                _warning_site()
    elif kind == "managed_span":

        def operation():
            with managed_span("test", "test"):
                _warning_site()
    elif kind == "trace_fn":
        operation = trace_fn("test", "test")(_warning_site)
    elif kind == "ray":
        operation = traced_remote_call(_warning_site)
    else:

        def operation():
            emit_span(None, "test", 1, 2)
            _warning_site()

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("default")
            for _ in range(100):
                operation()
        assert (
            sum(str(w.message) == "representative PyTorch collective warning" for w in caught) == 1
        )
        assert len(exporter.get_finished_spans()) == 100
    finally:
        provider.shutdown()


def test_provider_replacement_and_late_setup(monkeypatch):
    before = get_tracer("test")
    first = TracerProvider()
    second = TracerProvider()
    try:
        trace.set_tracer_provider(first)
        live = get_tracer("test")
        assert live is not before
        assert live is get_tracer("test")
        monkeypatch.setattr(trace, "_TRACER_PROVIDER", second)
        assert get_tracer("test") is not live
    finally:
        first.shutdown()
        second.shutdown()


def test_concurrent_lookup_resolves_once(monkeypatch):
    class Provider:
        def __init__(self):
            self.calls = 0

        def get_tracer(self, name):
            self.calls += 1
            return object()

    provider = Provider()
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)
    with ThreadPoolExecutor(max_workers=8) as pool:
        tracers = list(pool.map(lambda _: get_tracer("test"), range(100)))
    assert provider.calls == 1
    assert all(t is tracers[0] for t in tracers)


def test_disabled_groups_do_not_resolve_tracer(monkeypatch):
    def fail():
        raise AssertionError("disabled span touched OTel")

    monkeypatch.setattr(trace, "get_tracer_provider", fail)
    with managed_span("disabled", "test") as span:
        assert span is None
    assert trace_fn("disabled", "test")(lambda: 42)() == 42
    assert emit_span(None, "test", 1, 2, group="disabled") is None


def test_explicit_tracer_bypasses_cache(monkeypatch):
    tracer = trace.NoOpTracer()

    def fail():
        raise AssertionError("explicit tracer went through global provider")

    monkeypatch.setattr(trace, "get_tracer_provider", fail)
    set_enabled_span_groups(frozenset({"test"}))
    with span_cm("test", tracer=tracer):
        pass
    with managed_span("test", "test", tracer=tracer):
        pass
    trace_fn("test", "test", tracer=tracer)(lambda: None)()
    traced_remote_call(lambda: None, tracer=tracer)()
    emit_span(tracer, "test", 1, 2)


@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(), reason="requires fork")
def test_child_resolves_fresh_tracer_even_if_parent_lock_is_held():
    ctx = multiprocessing.get_context("fork")
    parent, child = ctx.Pipe()
    tracer = get_tracer("fork-test")
    locked, release = threading.Event(), threading.Event()

    def hold_lock():
        with _tracers._LOCK:
            locked.set()
            release.wait(10)

    thread = threading.Thread(target=hold_lock)
    thread.start()
    assert locked.wait(5)

    def run_child():
        child.send(get_tracer("fork-test") is not tracer)
        child.close()

    process = ctx.Process(target=run_child)
    try:
        process.start()
        assert parent.poll(5), "child deadlocked on inherited tracer-cache lock"
        assert parent.recv()
        process.join(5)
        assert process.exitcode == 0
    finally:
        release.set()
        thread.join(5)
        if process.is_alive():
            process.terminate()
            process.join(5)
        parent.close()
        child.close()
