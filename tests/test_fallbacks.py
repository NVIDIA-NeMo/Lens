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

"""Unit tests for nemo.lens.fallbacks — canonical no-op implementations."""

from nemo.lens.fallbacks import (
    encode_resource_attributes as noop_encode_resource_attributes,
)
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


class TestFallbackEncodeResourceAttributes:
    """Parity lives here because AGENTS.md invariant 3 and contributing.mdx both
    name this file as the enforcement point for the fallback surface."""

    def test_signature_matches_the_real_one(self):
        import inspect

        from nemo.lens.resources import encode_resource_attributes as real

        real_sig = inspect.signature(real)
        noop_sig = inspect.signature(noop_encode_resource_attributes)
        assert list(real_sig.parameters) == list(noop_sig.parameters)
        # Defaults too, not just names. Comparing names alone let the two read
        # different sources on the `inherited=None` path -- the no-op live, the
        # real one an import-time snapshot -- while this test stayed green.
        assert [p.default for p in real_sig.parameters.values()] == [
            p.default for p in noop_sig.parameters.values()
        ]

    def test_both_read_live_environ_when_inherited_is_omitted(self, monkeypatch):
        """The default path an explicit `inherited=` argument cannot reach.

        Passing it explicitly bypasses exactly the branch where the two
        implementations previously disagreed.
        """
        from nemo.lens.resources import encode_resource_attributes as real

        monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", "launcher.id=abc")
        assert noop_encode_resource_attributes({"dl.rank": 3}) == "launcher.id=abc"
        assert real({"dl.rank": 3}).startswith("launcher.id=abc,")

    def test_preserves_a_launcher_supplied_value(self):
        """Deliberately not `""` — a launcher-set value must still reach a child
        that does have lens installed."""
        assert noop_encode_resource_attributes({"dl.rank": 3}, inherited="job=abc") == "job=abc"
