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

"""Unit tests for span group state management."""

import logging
import threading

from nemo.lens.groups import SpanRegistry
from nemo.lens.state import (
    enabled_span_groups,
    is_span_group_enabled,
    pending_span_groups,
    refresh_enabled_span_groups,
    set_enabled_span_groups,
    set_span_group_spec,
)


class TestSpanGroupState:
    def test_default_all_disabled(self):
        assert is_span_group_enabled("job") is False
        assert is_span_group_enabled("step") is False

    def test_enable_groups(self):
        set_enabled_span_groups(frozenset(["job", "checkpoint"]))
        assert is_span_group_enabled("job") is True
        assert is_span_group_enabled("checkpoint") is True
        assert is_span_group_enabled("step") is False

    def test_override_groups(self):
        set_enabled_span_groups(frozenset(["job"]))
        assert is_span_group_enabled("job") is True
        set_enabled_span_groups(frozenset(["step"]))
        assert is_span_group_enabled("job") is False
        assert is_span_group_enabled("step") is True

    def test_clear_groups(self):
        set_enabled_span_groups(frozenset(["job", "step"]))
        assert is_span_group_enabled("job") is True
        set_enabled_span_groups(frozenset())
        assert is_span_group_enabled("job") is False

    def test_unknown_group_returns_false(self):
        set_enabled_span_groups(frozenset(["job"]))
        assert is_span_group_enabled("nonexistent") is False


class TestSpecDrivenState:
    def test_spec_resolves_against_the_registry(self):
        SpanRegistry.register("mega", {"job", "step"}, {"default": {"job"}})
        set_span_group_spec("default")
        assert is_span_group_enabled("job") is True
        assert is_span_group_enabled("step") is False

    def test_a_name_nobody_registered_warns_and_stays_pending(self, caplog):
        """Never raises: this process's registry is not the authority on a
        job-wide spec, and telemetry setup must not take the process down."""
        SpanRegistry.register("mega", {"step"})
        with caplog.at_level("WARNING"):
            set_span_group_spec("step,typo")
        assert is_span_group_enabled("step") is True
        assert pending_span_groups() == frozenset({"typo"})
        assert "no library registered in this process provides" in caplog.text

    def test_the_warning_names_what_is_available(self, caplog):
        SpanRegistry.register("mega", {"step"}, {"per_step": {"step"}})
        with caplog.at_level("WARNING"):
            set_span_group_spec("typo")
        assert "'step'" in caplog.text
        assert "per_step" in caplog.text
        assert "mega" in caplog.text

    def test_an_empty_registry_gets_its_own_message(self, caplog):
        with caplog.at_level("WARNING"):
            set_span_group_spec("default")
        assert "no library has registered any span groups" in caplog.text

    def test_all_against_an_empty_registry_still_reports(self, caplog):
        """The quietest version of the loudest problem.

        "all" is always a preset, so it resolves cleanly and leaves nothing
        pending -- which used to mean zero telemetry and zero diagnostics. An
        empty registry is the expected state of a consumer that has not yet
        migrated to SpanRegistry.register(), and "all" is the likeliest spec for
        someone asking for everything.
        """
        with caplog.at_level("WARNING"):
            set_span_group_spec("all")
        assert enabled_span_groups() == frozenset()
        assert "no library has registered any span groups" in caplog.text

    def test_an_empty_spec_against_an_empty_registry_stays_quiet(self, caplog):
        """Asking for nothing and getting nothing is not a problem to report."""
        with caplog.at_level("WARNING"):
            set_span_group_spec("")
        assert caplog.text == ""

    def test_a_typo_is_re_reported_once_the_registry_fills(self, caplog):
        """The repeat-suppression key must include the registry-empty flag.

        Keyed on the unresolved set alone, the first message ("no library has
        registered any span groups") suppressed the second and far more useful
        one -- the message listing the registered groups, which is what turns
        "stepp" into a visible typo of "step".
        """
        set_span_group_spec("stepp")
        caplog.clear()  # the empty-registry message above legitimately warned once
        with caplog.at_level("WARNING"):
            SpanRegistry.register("mega", {"step"})
        assert "'step'" in caplog.text
        assert "mega" in caplog.text

    def test_an_identical_unresolved_set_is_not_re_reported(self, caplog):
        """Re-resolution happens on every registration; the log should not."""
        SpanRegistry.register("mega", {"step"})
        set_span_group_spec("step,typo")
        caplog.clear()  # the setup call above legitimately warned once
        with caplog.at_level("WARNING"):
            SpanRegistry.register("other", {"another"})
        assert "no library registered in this process provides" not in caplog.text

    def test_unregistering_warns_rather_than_raising(self, caplog):
        SpanRegistry.register("mega", {"step"})
        set_span_group_spec("step")
        with caplog.at_level("WARNING"):
            SpanRegistry.unregister("mega")
        assert "no library has registered any span groups" in caplog.text

    def test_late_registration_still_takes_effect(self):
        """A misconfiguration to warn about, not one to silently drop spans over."""
        SpanRegistry.register("mega", {"job", "step"}, {"per_step": {"job", "step"}})
        set_span_group_spec("per_step")
        SpanRegistry.register("late", {"extra"}, {"per_step": {"extra"}})
        assert is_span_group_enabled("extra") is True

    def test_unregistering_switches_groups_back_off(self):
        SpanRegistry.register("mega", {"step"})
        set_span_group_spec("step")
        assert is_span_group_enabled("step") is True
        SpanRegistry.unregister("mega")
        assert is_span_group_enabled("step") is False

    def test_pinning_drops_the_spec_so_registration_cannot_reopen_it(self):
        """How a disabled process stays disabled."""
        SpanRegistry.register("mega", {"step"})
        set_span_group_spec("all")
        set_enabled_span_groups(frozenset())
        SpanRegistry.register("other", {"another"})
        assert is_span_group_enabled("step") is False
        assert is_span_group_enabled("another") is False
        assert enabled_span_groups() == frozenset()

    def test_enabled_span_groups_reports_the_live_set(self):
        SpanRegistry.register("mega", {"job", "step"})
        set_span_group_spec("all")
        assert enabled_span_groups() == frozenset({"job", "step"})


class TestTheDiagnosticDescribesOneRegistryGeneration:
    """`_report` runs after the lock, so it must report what the resolution saw.

    Building the message from a fresh query let a registration land in between,
    producing a warning that called a group unresolved and also listed it as
    registered.
    """

    def test_the_warning_is_built_without_asking_the_registry_again(self, monkeypatch, caplog):
        SpanRegistry.register("mega", {"step"})

        def boom(*_args, **_kwargs):
            raise AssertionError("_report queried the registry a second time")

        for name in ("groups", "presets", "namespaces"):
            monkeypatch.setattr(SpanRegistry, name, classmethod(boom))

        with caplog.at_level(logging.WARNING, logger="nemo.lens.state"):
            set_span_group_spec("typo")

        # Rendered, and rendered from the same generation the resolution saw.
        assert "typo" in caplog.text
        assert "'step'" in caplog.text
        assert "'mega'" in caplog.text

    def test_a_concurrent_registration_cannot_contradict_the_message(self, monkeypatch, caplog):
        """The group named unresolved must not also appear as registered."""
        SpanRegistry.register("base", {"other"})
        real_snapshot = SpanRegistry._snapshot.__func__

        def snapshot_then_register(cls):
            result = real_snapshot(cls)
            # Lands between the resolution and the report -- the window the old
            # code re-queried in.
            if "owner" not in cls._GROUPS:
                cls._GROUPS["owner"] = frozenset({"step"})
            return result

        monkeypatch.setattr(SpanRegistry, "_snapshot", classmethod(snapshot_then_register))

        with caplog.at_level(logging.WARNING, logger="nemo.lens.state"):
            set_span_group_spec("step")

        assert "step" in caplog.text
        assert "'owner'" not in caplog.text, f"self-contradictory warning: {caplog.text}"


class TestASpecThatSelectsNothingIsReported:
    """Silence has two causes, and only one of them used to be reported.

    A preset that borrowed groups from a namespace that has gone away is pruned
    to empty. Nothing is unresolved, so `pending_span_groups()` -- the diagnostic
    the troubleshooting guide points at -- is empty as well, leaving the user with
    no telemetry and nothing to look at.
    """

    def test_a_preset_pruned_to_empty_warns(self, caplog):
        SpanRegistry.register("mega", {"step"})
        SpanRegistry.register("rl", {"rollout"}, {"prod": {"step"}})
        set_span_group_spec("prod")
        assert enabled_span_groups() == frozenset({"step"})

        with caplog.at_level(logging.WARNING, logger="nemo.lens.state"):
            SpanRegistry.unregister("mega")

        assert enabled_span_groups() == frozenset()
        assert pending_span_groups() == frozenset()
        assert "resolved to no span groups" in caplog.text

    def test_a_spec_that_selects_something_stays_quiet(self, caplog):
        SpanRegistry.register("mega", {"step"})
        with caplog.at_level(logging.WARNING, logger="nemo.lens.state"):
            set_span_group_spec("step")
        assert caplog.text == ""

    def test_going_empty_twice_warns_twice(self, caplog):
        """The suppression key must include whether the spec selected anything.

        A spec that resolved fine and then went empty has the same unresolved set
        as when it was working. Without that flag in the key, the second drop
        looks like a repeat of a report that was never made.
        """
        SpanRegistry.register("rl", {"rollout"}, {"prod": {"step"}})
        SpanRegistry.register("mega", {"step"})
        set_span_group_spec("prod")
        assert enabled_span_groups() == frozenset({"step"})

        with caplog.at_level(logging.WARNING, logger="nemo.lens.state"):
            SpanRegistry.unregister("mega")
            assert caplog.text.count("resolved to no span groups") == 1

            SpanRegistry.register("mega", {"step"})
            assert enabled_span_groups() == frozenset({"step"})

            SpanRegistry.unregister("mega")
            assert enabled_span_groups() == frozenset()
        assert caplog.text.count("resolved to no span groups") == 2


class TestSpanGroupPublicAPI:
    def test_importable_from_package(self):
        from nemo.lens import SpanRegistry as Exported
        from nemo.lens import is_span_group_enabled as exported_check

        assert callable(exported_check)
        assert hasattr(Exported, "register")


class TestResolutionIsAtomic:
    """Resolve and store happen under one lock hold.

    Split apart, a slow resolution can land on top of a newer one and the enabled
    set keeps the older answer indefinitely — nothing refreshes it again unless
    another registration happens to come along. Silently-wrong span groups for the
    life of the process is the exact failure this design exists to avoid.
    """

    def test_a_stale_resolution_cannot_clobber_a_newer_one(self, monkeypatch):
        SpanRegistry.register("mega", {"step"})
        set_span_group_spec("all")

        # Patches the entry point `state` actually calls. _resolve_snapshot also
        # returns the registry-empty flag, so the resolution and the diagnostic
        # come from one registry hold.
        real_resolve = SpanRegistry._resolve_snapshot
        gate = threading.Event()
        slow_started = threading.Event()

        def slow_resolve(spec):
            # Compute against the registry as it is now, then stall *before* the
            # caller gets to store the result.
            result = real_resolve(spec)
            if threading.current_thread().name == "slow-resolver":
                slow_started.set()
                gate.wait(10)
            return result

        monkeypatch.setattr(SpanRegistry, "_resolve_snapshot", slow_resolve)

        slow = threading.Thread(target=refresh_enabled_span_groups, name="slow-resolver")
        slow.start()
        try:
            assert slow_started.wait(10), "slow resolver never started"

            # Released from a timer, not from this thread: with the fix, the
            # register() below blocks until the slow resolver lets go of the lock.
            timer = threading.Timer(0.3, gate.set)
            timer.start()
            try:
                SpanRegistry.register("late", {"extra"})
            finally:
                timer.cancel()
                gate.set()
        finally:
            gate.set()
            slow.join(10)
            assert not slow.is_alive(), "slow resolver deadlocked"

        # The newer registration must win regardless of which resolution finished
        # first. Ordering is enforced by the lock, so this does not depend on the
        # timer firing at any particular moment.
        assert "extra" in enabled_span_groups()
        assert enabled_span_groups() == SpanRegistry.groups()

    def test_the_initial_spec_resolution_is_also_atomic(self, monkeypatch):
        """`set_span_group_spec` carries the same read-resolve-write pattern.

        Split apart, a `setup_telemetry` racing an import-time `register()` pins
        the pre-registration resolution for the life of the process: the stalled
        resolver stores its stale answer *and* the spec, while the registration
        that would have refreshed it saw no spec yet and returned early.
        """
        SpanRegistry.register("mega", {"step"})

        real_resolve = SpanRegistry._resolve_snapshot
        gate = threading.Event()
        slow_started = threading.Event()

        def slow_resolve(spec):
            result = real_resolve(spec)
            if threading.current_thread().name == "slow-spec":
                slow_started.set()
                gate.wait(10)
            return result

        monkeypatch.setattr(SpanRegistry, "_resolve_snapshot", slow_resolve)

        slow = threading.Thread(target=set_span_group_spec, args=("all",), name="slow-spec")
        slow.start()
        try:
            assert slow_started.wait(10), "slow resolver never started"

            timer = threading.Timer(0.3, gate.set)
            timer.start()
            try:
                SpanRegistry.register("late", {"extra"})
            finally:
                timer.cancel()
                gate.set()
        finally:
            gate.set()
            slow.join(10)
            assert not slow.is_alive(), "slow resolver deadlocked"

        assert "extra" in enabled_span_groups()
        assert enabled_span_groups() == SpanRegistry.groups()

    def test_concurrent_registration_leaves_the_enabled_set_consistent(self):
        """Whatever the interleaving, the end state matches a fresh resolve."""
        SpanRegistry.register("base", {"job"})
        set_span_group_spec("all")

        barrier = threading.Barrier(6)
        errors: list = []

        def worker(i):
            try:
                barrier.wait(10)
                for r in range(25):
                    ns = f"ns{i}_{r}"
                    SpanRegistry.register(ns, {f"g{i}_{r}"}, {"default": {f"g{i}_{r}", "job"}})
                    SpanRegistry.unregister(ns)
            except Exception as exc:  # pragma: no cover - only on a regression
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
            assert not t.is_alive(), "registration deadlocked"

        assert not errors, errors[:3]
        assert enabled_span_groups() == SpanRegistry.resolve("all")[0]
