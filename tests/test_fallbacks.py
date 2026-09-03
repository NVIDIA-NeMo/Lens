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

import inspect

import pytest

from nemo.lens.fallbacks import (
    SpanRegistry,
    is_span_group_enabled,
    managed_span,
    safe_set_span_attributes,
    span_cm,
    trace_fn,
)
from nemo.lens.fallbacks import (
    encode_resource_attributes as noop_encode_resource_attributes,
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


class TestFallbackSpanRegistry:
    """Consumers register at import time, so this has to work without lens."""

    def test_register_accepts_the_real_signature(self):
        SpanRegistry.register("mega", {"step"}, {"default": {"step"}}, allow_override=True)

    def test_register_forgets_everything(self):
        SpanRegistry.register("mega", {"step"})
        assert SpanRegistry.groups() == frozenset()
        assert SpanRegistry.namespaces() == []

    def test_unregister_does_not_raise_on_unknown(self):
        SpanRegistry.unregister("never-registered")

    def test_clear_is_a_noop(self):
        SpanRegistry.clear()

    def test_presets_expose_only_the_wildcard(self):
        assert SpanRegistry.presets() == {"all": frozenset()}

    def test_resolve_returns_the_two_tuple_shape(self):
        enabled, pending = SpanRegistry.resolve("default,step")
        assert enabled == frozenset()
        assert pending == frozenset({"default", "step"})


class TestSignatureParity:
    """Invariant 3: the no-op surface must match the real API, defaults included.

    Parameter names alone are not the signature. AGENTS.md says defaults must
    resolve to the same source, because a no-op whose default differs changes
    behaviour for consumers running without lens installed, and only for them --
    which is the kind of divergence this file exists to catch. Comparing names
    alone cannot see that, nor a parameter promoted to keyword-only.
    """

    #: Parameters whose no-op default deliberately differs, and why. Each consumer
    #: hand-writes inline copies of these no-ops in its own `_fallbacks.py` for the
    #: case where lens is absent. A default that points at a lens constant has to
    #: be reinvented in every one of those copies, which spreads the divergence
    #: rather than fixing it. The no-op body here is `pass`, so the value is never
    #: read. Name and kind are still compared; only the default is exempt.
    _EXEMPT_DEFAULTS = {"safe_set_span_attributes": {"redact_keys"}}

    @classmethod
    def _shape(cls, fn, symbol):
        # Annotations are excluded deliberately: the no-ops are unannotated on
        # purpose, so comparing them would fail on every symbol.
        exempt = cls._EXEMPT_DEFAULTS.get(symbol, frozenset())
        return [
            (p.name, p.kind, "<exempt>" if p.name in exempt else p.default)
            for p in inspect.signature(fn).parameters.values()
        ]

    @pytest.mark.parametrize(
        "name",
        [
            "trace_fn",
            "managed_span",
            "span_cm",
            "is_span_group_enabled",
            "safe_set_span_attributes",
            "encode_resource_attributes",
        ],
    )
    def test_module_level_symbol(self, name):
        import nemo.lens
        import nemo.lens.fallbacks

        real = getattr(nemo.lens, name)
        noop = getattr(nemo.lens.fallbacks, name)
        assert self._shape(real, name) == self._shape(noop, name), name

    @pytest.mark.parametrize(
        "name", ["register", "unregister", "clear", "groups", "namespaces", "presets", "resolve"]
    )
    def test_span_registry_method(self, name):
        from nemo.lens.groups import SpanRegistry as Real

        assert self._shape(getattr(Real, name), name) == self._shape(
            getattr(SpanRegistry, name), name
        ), name
