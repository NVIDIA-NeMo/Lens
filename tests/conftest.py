# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared fixtures for nemo-lens unit tests."""

import opentelemetry.metrics._internal as _metrics_mod
import opentelemetry.trace as _trace_mod
import pytest
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.util._once import Once


class InMemorySpanExporter(SpanExporter):
    """Simple in-memory span exporter for tests."""

    def __init__(self):
        self._spans = []
        self._stopped = False

    def export(self, spans):
        if self._stopped:
            return SpanExportResult.FAILURE
        self._spans.extend(spans)
        return SpanExportResult.SUCCESS

    def get_finished_spans(self):
        return list(self._spans)

    def clear(self):
        self._spans.clear()

    def shutdown(self):
        self._stopped = True

    def force_flush(self, timeout_millis=0):
        return True


def _reset_otel_globals() -> None:
    """Reset the global OTel tracer/meter providers to an unset state."""
    _trace_mod._TRACER_PROVIDER = None
    _trace_mod._TRACER_PROVIDER_SET_ONCE = Once()
    _metrics_mod._METER_PROVIDER = None
    _metrics_mod._METER_PROVIDER_SET_ONCE = Once()

    # Reset the initialization guard
    import nemo.lens.handle as _handle_mod

    _handle_mod._INITIALIZED = False


@pytest.fixture(autouse=True)
def reset_otel_providers():
    """Reset global OTel providers before and after each test."""
    _reset_otel_globals()
    yield
    _reset_otel_globals()


@pytest.fixture(autouse=True)
def reset_span_groups():
    """Reset span group state before and after each test."""
    from nemo.lens.state import set_enabled_span_groups, set_pp_trace_carrier

    set_enabled_span_groups(frozenset())
    set_pp_trace_carrier(None)
    yield
    set_enabled_span_groups(frozenset())
    set_pp_trace_carrier(None)


@pytest.fixture(autouse=True)
def reset_strategy_registry():
    """Snapshot the strategy registry before each test, restore after."""
    from nemo.lens.strategies import _REGISTRY, _REGISTRY_LOCK

    with _REGISTRY_LOCK:
        snapshot = dict(_REGISTRY)
    yield
    with _REGISTRY_LOCK:
        _REGISTRY.clear()
        _REGISTRY.update(snapshot)
