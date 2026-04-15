# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for nemo.lens.fallbacks — canonical no-op implementations."""

from nemo.lens.fallbacks import (
    is_span_group_enabled,
    managed_span,
    safe_set_span_attributes,
    span_cm,
    trace_fn,
)


class TestFallbackTraceFn:
    def test_returns_function_unchanged(self):
        def my_func(x):
            return x + 1

        decorated = trace_fn("group", "name")(my_func)
        assert decorated is my_func

    def test_decorated_function_works(self):
        @trace_fn("group", "name")
        def my_func(x):
            return x * 2

        assert my_func(5) == 10


class TestFallbackManagedSpan:
    def test_yields_none(self):
        with managed_span("group", "name") as span:
            assert span is None

    def test_body_executes(self):
        result = []
        with managed_span("group", "name"):
            result.append(42)
        assert result == [42]

    def test_accepts_kwargs(self):
        with managed_span("group", "name", iteration=1, loss=0.5) as span:
            assert span is None


class TestFallbackSpanCm:
    def test_yields_none(self):
        with span_cm("name") as span:
            assert span is None

    def test_body_executes(self):
        result = []
        with span_cm("name"):
            result.append(42)
        assert result == [42]


class TestFallbackIsSpanGroupEnabled:
    def test_always_returns_false(self):
        assert is_span_group_enabled("job") is False
        assert is_span_group_enabled("step") is False
        assert is_span_group_enabled("anything") is False


class TestFallbackSafeSetSpanAttributes:
    def test_noop_on_none_span(self):
        safe_set_span_attributes(None, {"key": "value"})

    def test_noop_with_empty_dict(self):
        safe_set_span_attributes(None, {})
