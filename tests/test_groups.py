# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for SpanGroup base class."""

import pytest

from nemo.lens.groups import SpanGroup


class TestSpanGroupResolve:
    def test_default_preset(self):
        groups = SpanGroup.resolve("default")
        assert groups == frozenset([SpanGroup.JOB, SpanGroup.CHECKPOINT, SpanGroup.EVALUATE])

    def test_per_step_preset(self):
        groups = SpanGroup.resolve("per_step")
        assert SpanGroup.STEP in groups
        assert SpanGroup.FORWARD_BACKWARD in groups
        assert SpanGroup.OPTIMIZER in groups

    def test_all_preset(self):
        groups = SpanGroup.resolve("all")
        assert groups == SpanGroup.ALL_GROUPS

    def test_individual_group(self):
        groups = SpanGroup.resolve("step")
        assert groups == frozenset(["step"])

    def test_comma_separated(self):
        groups = SpanGroup.resolve("job,checkpoint")
        assert groups == frozenset([SpanGroup.JOB, SpanGroup.CHECKPOINT])

    def test_mix_preset_and_individual(self):
        groups = SpanGroup.resolve("default,step")
        assert SpanGroup.JOB in groups
        assert SpanGroup.STEP in groups

    def test_case_insensitive(self):
        assert SpanGroup.resolve("DEFAULT") == SpanGroup.resolve("default")

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown span group"):
            SpanGroup.resolve("unknown_group")

    def test_whitespace_tolerant(self):
        groups = SpanGroup.resolve("job , checkpoint")
        assert SpanGroup.JOB in groups
        assert SpanGroup.CHECKPOINT in groups

    def test_all_individual_groups_valid(self):
        for group in SpanGroup.ALL_GROUPS:
            result = SpanGroup.resolve(group)
            assert group in result
