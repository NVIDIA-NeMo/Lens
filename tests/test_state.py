# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for span group state management."""

from nemo.lens.state import is_span_group_enabled, set_enabled_span_groups


class TestSpanGroupState:
    def test_default_all_disabled(self):
        assert is_span_group_enabled('job') is False
        assert is_span_group_enabled('step') is False

    def test_enable_groups(self):
        set_enabled_span_groups(frozenset(['job', 'checkpoint']))
        assert is_span_group_enabled('job') is True
        assert is_span_group_enabled('checkpoint') is True
        assert is_span_group_enabled('step') is False

    def test_override_groups(self):
        set_enabled_span_groups(frozenset(['job']))
        assert is_span_group_enabled('job') is True
        set_enabled_span_groups(frozenset(['step']))
        assert is_span_group_enabled('job') is False
        assert is_span_group_enabled('step') is True

    def test_clear_groups(self):
        set_enabled_span_groups(frozenset(['job', 'step']))
        assert is_span_group_enabled('job') is True
        set_enabled_span_groups(frozenset())
        assert is_span_group_enabled('job') is False

    def test_unknown_group_returns_false(self):
        set_enabled_span_groups(frozenset(['job']))
        assert is_span_group_enabled('nonexistent') is False


class TestSpanGroupPublicAPI:
    def test_importable_from_package(self):
        from nemo.lens import is_span_group_enabled, SpanGroup
        assert callable(is_span_group_enabled)
        assert hasattr(SpanGroup, 'JOB')
