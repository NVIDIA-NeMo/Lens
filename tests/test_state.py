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

        real_resolve = SpanRegistry.resolve
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

        monkeypatch.setattr(SpanRegistry, "resolve", slow_resolve)

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
