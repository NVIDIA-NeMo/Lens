# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for SpanGroup and library-specific subclasses."""

import pytest

from nemo.lens.groups import SpanGroup
from nemo.lens.groups_megatron import MegatronSpanGroup
from nemo.lens.groups_rl import RLSpanGroup
from nemo.lens.groups_gym import GymSpanGroup


class TestSpanGroupResolve:
    def test_default_preset(self):
        groups = SpanGroup.resolve('default')
        assert groups == frozenset([SpanGroup.JOB, SpanGroup.CHECKPOINT, SpanGroup.EVALUATE])

    def test_per_step_preset(self):
        groups = SpanGroup.resolve('per_step')
        assert SpanGroup.STEP in groups
        assert SpanGroup.FORWARD_BACKWARD in groups
        assert SpanGroup.OPTIMIZER in groups

    def test_all_preset(self):
        groups = SpanGroup.resolve('all')
        assert groups == SpanGroup.ALL_GROUPS

    def test_individual_group(self):
        groups = SpanGroup.resolve('step')
        assert groups == frozenset(['step'])

    def test_comma_separated(self):
        groups = SpanGroup.resolve('job,checkpoint')
        assert groups == frozenset([SpanGroup.JOB, SpanGroup.CHECKPOINT])

    def test_mix_preset_and_individual(self):
        groups = SpanGroup.resolve('default,step')
        assert SpanGroup.JOB in groups
        assert SpanGroup.STEP in groups

    def test_case_insensitive(self):
        assert SpanGroup.resolve('DEFAULT') == SpanGroup.resolve('default')

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match='Unknown span group'):
            SpanGroup.resolve('unknown_group')

    def test_whitespace_tolerant(self):
        groups = SpanGroup.resolve('job , checkpoint')
        assert SpanGroup.JOB in groups
        assert SpanGroup.CHECKPOINT in groups

    def test_all_individual_groups_valid(self):
        for group in SpanGroup.ALL_GROUPS:
            result = SpanGroup.resolve(group)
            assert group in result


class TestMegatronSpanGroup:
    def test_includes_base_groups(self):
        assert SpanGroup.JOB in MegatronSpanGroup.ALL_GROUPS
        assert SpanGroup.STEP in MegatronSpanGroup.ALL_GROUPS

    def test_has_microbatch(self):
        assert MegatronSpanGroup.MICROBATCH in MegatronSpanGroup.ALL_GROUPS

    def test_has_inference(self):
        assert MegatronSpanGroup.INFERENCE in MegatronSpanGroup.ALL_GROUPS

    def test_default_includes_inference(self):
        groups = MegatronSpanGroup.resolve('default')
        assert MegatronSpanGroup.INFERENCE in groups

    def test_all_includes_microbatch(self):
        groups = MegatronSpanGroup.resolve('all')
        assert MegatronSpanGroup.MICROBATCH in groups

    def test_resolve_microbatch(self):
        groups = MegatronSpanGroup.resolve('microbatch')
        assert MegatronSpanGroup.MICROBATCH in groups

    def test_per_step_no_microbatch(self):
        groups = MegatronSpanGroup.resolve('per_step')
        assert MegatronSpanGroup.MICROBATCH not in groups
        assert SpanGroup.STEP in groups


class TestRLSpanGroup:
    def test_has_rl_groups(self):
        assert RLSpanGroup.ROLLOUT in RLSpanGroup.ALL_GROUPS
        assert RLSpanGroup.GENERATION in RLSpanGroup.ALL_GROUPS
        assert RLSpanGroup.REWARD in RLSpanGroup.ALL_GROUPS
        assert RLSpanGroup.POLICY_UPDATE in RLSpanGroup.ALL_GROUPS

    def test_per_step_includes_rl_groups(self):
        groups = RLSpanGroup.resolve('per_step')
        assert RLSpanGroup.ROLLOUT in groups
        assert RLSpanGroup.GENERATION in groups

    def test_resolve_individual_rl_group(self):
        groups = RLSpanGroup.resolve('rollout')
        assert RLSpanGroup.ROLLOUT in groups


class TestGymSpanGroup:
    def test_has_gym_groups(self):
        assert GymSpanGroup.SERVER in GymSpanGroup.ALL_GROUPS
        assert GymSpanGroup.ROLLOUT_COLLECTION in GymSpanGroup.ALL_GROUPS
        assert GymSpanGroup.VERIFY in GymSpanGroup.ALL_GROUPS

    def test_default_includes_server(self):
        groups = GymSpanGroup.resolve('default')
        assert GymSpanGroup.SERVER in groups

    def test_per_step_includes_all_gym(self):
        groups = GymSpanGroup.resolve('per_step')
        assert GymSpanGroup.VERIFY in groups
        assert GymSpanGroup.AGGREGATE_METRICS in groups
