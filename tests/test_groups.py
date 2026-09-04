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


"""Unit tests for SpanRegistry."""

import logging

import pytest

from nemo.lens.groups import ALL, SpanRegistry


class TestEmptyRegistry:
    def test_lens_ships_no_groups(self):
        """The whole point: group names belong to consuming libraries."""
        assert SpanRegistry.groups() == frozenset()
        assert SpanRegistry.namespaces() == []

    def test_all_over_an_empty_registry_is_empty(self):
        enabled, pending = SpanRegistry.resolve("all")
        assert enabled == frozenset()
        assert pending == frozenset()

    def test_everything_is_pending_before_anyone_registers(self):
        enabled, pending = SpanRegistry.resolve("default,step")
        assert enabled == frozenset()
        assert pending == frozenset({"default", "step"})


class TestRegister:
    def test_groups_become_resolvable(self):
        SpanRegistry.register("mega", {"step", "layer"})
        assert SpanRegistry.groups() == frozenset({"step", "layer"})
        assert SpanRegistry.resolve("step")[0] == frozenset({"step"})

    def test_namespaces_listed(self):
        SpanRegistry.register("mega", {"step"})
        SpanRegistry.register("rl", {"rollout"})
        assert SpanRegistry.namespaces() == ["mega", "rl"]

    def test_preset_resolves(self):
        SpanRegistry.register("mega", {"step", "layer"}, {"default": {"step"}})
        assert SpanRegistry.resolve("default")[0] == frozenset({"step"})

    def test_all_is_a_wildcard_over_the_registry(self):
        SpanRegistry.register("mega", {"step", "layer"})
        SpanRegistry.register("rl", {"rollout"})
        assert SpanRegistry.resolve("all")[0] == frozenset({"step", "layer", "rollout"})

    def test_names_are_lowercased_and_stripped(self):
        SpanRegistry.register("mega", {"  STEP  "})
        assert SpanRegistry.groups() == frozenset({"step"})

    def test_empty_namespace_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            SpanRegistry.register("", {"step"})

    def test_empty_group_name_raises(self):
        with pytest.raises(ValueError, match="empty group name"):
            SpanRegistry.register("mega", {"step", "  "})

    def test_preset_naming_an_unregistered_group_warns_and_keeps_going(self, caplog):
        with caplog.at_level(logging.WARNING, logger="nemo.lens.groups"):
            SpanRegistry.register("mega", {"step"}, {"default": {"step", "typo"}})
        assert "typo" in caplog.text
        # The registration stands and the unresolved member simply contributes
        # nothing -- it does not take the importing process down with it.
        assert SpanRegistry.presets()["default"] == frozenset({"step"})

    def test_all_is_a_reserved_preset_name(self):
        with pytest.raises(ValueError, match="reserved"):
            SpanRegistry.register("mega", {"step"}, {"all": {"step"}})

    def test_all_is_a_reserved_group_name_too(self):
        """resolve() checks presets first, so such a group is unselectable.

        Accepted, it would sit in the registry permanently unreachable: asking
        for it by name enables every group in the process instead.
        """
        with pytest.raises(ValueError, match="reserved"):
            SpanRegistry.register("mega", {"all", "quiet"})


class TestPresetsUnionAcrossNamespaces:
    """The wart this replaces: a subclass overrode _PRESETS wholesale."""

    def test_two_libraries_both_contribute_to_default(self):
        SpanRegistry.register("mega", {"step"}, {"default": {"step"}})
        SpanRegistry.register("rl", {"rollout"}, {"default": {"rollout"}})
        assert SpanRegistry.resolve("default")[0] == frozenset({"step", "rollout"})

    def test_a_late_registration_does_not_displace_an_earlier_one(self):
        SpanRegistry.register("mega", {"step"}, {"default": {"step"}})
        SpanRegistry.register("gym", {"episode"}, {"default": {"episode"}})
        assert "step" in SpanRegistry.presets()["default"]


class TestCollisions:
    def test_reregistering_a_namespace_raises(self):
        SpanRegistry.register("mega", {"step"})
        with pytest.raises(ValueError, match="already registered"):
            SpanRegistry.register("mega", {"layer"})

    def test_reregistering_a_namespace_with_override_replaces(self):
        SpanRegistry.register("mega", {"step"})
        SpanRegistry.register("mega", {"layer"}, allow_override=True)
        assert SpanRegistry.groups() == frozenset({"layer"})

    def test_an_override_does_not_keep_the_groups_it_is_dropping(self, caplog):
        """Its own previous groups are not referenceable: this call replaces them."""
        SpanRegistry.register("mega", {"step", "layer"})
        with caplog.at_level(logging.WARNING, logger="nemo.lens.groups"):
            SpanRegistry.register(
                "mega", {"layer"}, {"default": {"layer", "step"}}, allow_override=True
            )
        assert "step" in caplog.text
        assert SpanRegistry.presets()["default"] == frozenset({"layer"})

    def test_an_override_may_still_reference_another_namespace_s_groups(self):
        SpanRegistry.register("mega", {"step"})
        SpanRegistry.register("rl", {"rollout"})
        SpanRegistry.register(
            "rl", {"rollout"}, {"default": {"rollout", "step"}}, allow_override=True
        )
        assert SpanRegistry.resolve("default")[0] == frozenset({"rollout", "step"})

    def test_two_namespaces_claiming_one_group_warns_and_shares_it(self, caplog):
        """Neither library knows it is second, so this cannot be an error.

        It happens while a module is being imported, where there is no caller to
        catch it. The no-op registry never raises either, so raising here would
        mean installing lens breaks a job that worked without it.
        """
        SpanRegistry.register("mega", {"step"})
        with caplog.at_level(logging.WARNING, logger="nemo.lens.groups"):
            SpanRegistry.register("rl", {"step", "rollout"})
        assert "step" in caplog.text and "mega" in caplog.text
        assert SpanRegistry.resolve("step")[0] == frozenset({"step"})
        assert SpanRegistry.groups() == frozenset({"step", "rollout"})

    def test_the_collision_warning_is_symmetric_in_import_order(self, caplog):
        """Whichever imports second warns; the end state is the same either way."""
        for first, second in (("mega", "rl"), ("rl", "mega")):
            SpanRegistry.clear()
            caplog.clear()
            SpanRegistry.register(first, {"step"})
            with caplog.at_level(logging.WARNING, logger="nemo.lens.groups"):
                SpanRegistry.register(second, {"step"})
            assert "step" in caplog.text, f"{second} after {first}"
            assert SpanRegistry.groups() == frozenset({"step"})

    def test_a_shared_group_outlives_one_of_its_owners(self):
        SpanRegistry.register("mega", {"step", "layer"})
        SpanRegistry.register("rl", {"step", "rollout"})
        SpanRegistry.unregister("rl")
        assert "step" in SpanRegistry.groups()

    def test_allow_override_silences_the_collision_warning(self, caplog):
        SpanRegistry.register("mega", {"step"})
        with caplog.at_level(logging.WARNING, logger="nemo.lens.groups"):
            SpanRegistry.register("rl", {"step"}, allow_override=True)
        assert caplog.text == ""
        assert SpanRegistry.resolve("step")[0] == frozenset({"step"})


class TestUnregister:
    def test_removes_the_groups(self):
        SpanRegistry.register("mega", {"step"})
        SpanRegistry.unregister("mega")
        assert SpanRegistry.groups() == frozenset()

    def test_leaves_other_namespaces_alone(self):
        SpanRegistry.register("mega", {"step"})
        SpanRegistry.register("rl", {"rollout"})
        SpanRegistry.unregister("mega")
        assert SpanRegistry.groups() == frozenset({"rollout"})

    def test_unknown_namespace_raises(self):
        with pytest.raises(ValueError, match="is not registered"):
            SpanRegistry.unregister("nope")


class TestResolveSpec:
    def setup_method(self):
        SpanRegistry.clear()
        SpanRegistry.register(
            "mega", {"job", "step", "layer"}, {"default": {"job"}, "per_step": {"job", "step"}}
        )

    def test_comma_separated(self):
        assert SpanRegistry.resolve("job,layer")[0] == frozenset({"job", "layer"})

    def test_mix_preset_and_bare_name(self):
        assert SpanRegistry.resolve("default,layer")[0] == frozenset({"job", "layer"})

    def test_case_insensitive(self):
        assert SpanRegistry.resolve("DEFAULT")[0] == SpanRegistry.resolve("default")[0]

    def test_whitespace_tolerant(self):
        assert SpanRegistry.resolve(" job , layer ")[0] == frozenset({"job", "layer"})

    def test_empty_spec(self):
        assert SpanRegistry.resolve("")[0] == frozenset()

    def test_unknown_entry_is_pending_not_an_error(self):
        """A registry is per process; a spec is usually job-wide."""
        enabled, pending = SpanRegistry.resolve("job,not_yet_imported")
        assert enabled == frozenset({"job"})
        assert pending == frozenset({"not_yet_imported"})


class TestCrossNamespacePresets:
    """A library layering on one it depends on imports it, then names its groups."""

    def test_a_preset_may_reference_another_namespace_s_group(self):
        SpanRegistry.register("mega", {"step"})
        SpanRegistry.register("rl", {"rollout"}, {"default": {"rollout", "step"}})
        assert SpanRegistry.resolve("default")[0] == frozenset({"rollout", "step"})

    def test_referencing_a_group_whose_owner_is_not_imported_warns(self, caplog):
        """A missing optional dependency should not stop the library importing."""
        with caplog.at_level(logging.WARNING, logger="nemo.lens.groups"):
            SpanRegistry.register("rl", {"rollout"}, {"default": {"rollout", "step"}})
        assert "step" in caplog.text
        assert SpanRegistry.presets()["default"] == frozenset({"rollout"})

    def test_a_late_owner_makes_the_borrowed_group_start_counting(self):
        """Import order does not matter: the member starts working when its owner registers.

        The borrowing library may well be imported first, since it is the one that
        depends on the other, so this is the ordinary case rather than a recovery
        path.
        """
        SpanRegistry.register("rl", {"rollout"}, {"default": {"rollout", "step"}})
        assert SpanRegistry.resolve("default")[0] == frozenset({"rollout"})

        SpanRegistry.register("mega", {"step"})
        assert SpanRegistry.resolve("default")[0] == frozenset({"rollout", "step"})

    def test_the_reference_does_not_transfer_ownership(self):
        SpanRegistry.register("mega", {"step"})
        SpanRegistry.register("rl", {"rollout"}, {"default": {"rollout", "step"}})
        SpanRegistry.unregister("mega")
        assert "step" not in SpanRegistry.groups()

    def test_unregistering_prunes_the_borrowing_preset(self):
        """A borrowed group must not outlive its owner's registration.

        Left dangling, `default` enabled a group absent from `all` -- and spans
        were actually emitted for a group no namespace owned. Pruning on every
        read is also what lets registration keep an unresolved member instead of
        refusing it: the member is ignored until someone owns it.
        """
        SpanRegistry.register("mega", {"step"})
        SpanRegistry.register("rl", {"rollout"}, {"default": {"rollout", "step"}})
        SpanRegistry.unregister("mega")

        presets = SpanRegistry.presets()
        assert presets["default"] == frozenset({"rollout"})
        assert SpanRegistry.resolve("default")[0] == frozenset({"rollout"})
        assert presets["default"] <= presets[ALL]

    def test_override_prunes_a_dropped_group_from_a_borrowing_preset(self):
        """Same leak by the other route: re-registering replaces wholesale."""
        SpanRegistry.register("mega", {"step"})
        SpanRegistry.register("rl", {"rollout"}, {"default": {"rollout", "step"}})
        SpanRegistry.register("mega", {"other"}, allow_override=True)

        presets = SpanRegistry.presets()
        assert "step" not in presets["default"]
        assert presets["default"] <= presets[ALL]


class TestLateRegistrationWarning:
    def test_registering_after_setup_telemetry_warns(self, caplog):
        from nemo.lens.config import NemoLensConfig
        from nemo.lens.handle import setup_telemetry

        SpanRegistry.register("mega", {"step"}, {"per_step": {"step"}})
        setup_telemetry(NemoLensConfig(enabled=True, exporter="console", span_groups="per_step"))

        with caplog.at_level("WARNING"):
            SpanRegistry.register("late", {"extra"})

        assert "registered after setup_telemetry()" in caplog.text
        assert "'late'" in caplog.text

    def test_registering_before_setup_telemetry_is_silent(self, caplog):
        with caplog.at_level("WARNING"):
            SpanRegistry.register("mega", {"step"})
        assert "registered after setup_telemetry()" not in caplog.text

    def test_a_disabled_process_does_not_warn(self, caplog):
        """_INITIALIZED stays false when disabled, so late registration is harmless."""
        from nemo.lens.config import NemoLensConfig
        from nemo.lens.handle import setup_telemetry

        setup_telemetry(NemoLensConfig(enabled=False))
        with caplog.at_level("WARNING"):
            SpanRegistry.register("late", {"extra"})
        assert "registered after setup_telemetry()" not in caplog.text
