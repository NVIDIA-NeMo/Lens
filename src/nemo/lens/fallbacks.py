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

"""Canonical no-op fallbacks for when nemo-lens telemetry is not active.

These match the nemo.lens public API signatures so that instrumented code
works unchanged regardless of whether telemetry is initialised.

Consumer libraries should use these as their ImportError fallback::

    try:
        from nemo.lens.groups import SpanRegistry
        from nemo.lens.helpers import managed_span
        from nemo.lens.state import is_span_group_enabled
    except ImportError:
        from nemo.lens.fallbacks import SpanRegistry, managed_span, is_span_group_enabled
"""

import os
from contextlib import contextmanager


def trace_fn(group, name, tracer=None):
    """No-op decorator — returns the function unchanged."""

    def decorator(func):
        return func

    return decorator


@contextmanager
def managed_span(group, name, tracer=None, **attributes):
    """No-op context manager — yields None."""
    yield None


@contextmanager
def span_cm(name, tracer=None, record_exception=True, **attributes):
    """No-op context manager — yields None."""
    yield None


def is_span_group_enabled(group):
    """Always returns False."""
    return False


def safe_set_span_attributes(span, attributes, redact_keys=None):
    """No-op."""
    pass


def encode_resource_attributes(attributes, inherited=None):
    """No-op — returns the inherited value unchanged.

    Deliberately not ``""``. With lens absent the child has no lens either, so
    the attributes are moot, but a launcher-set OTEL_RESOURCE_ATTRIBUTES must
    still reach the child: clobbering it would break telemetry for anything
    downstream that does have lens installed.
    """
    if inherited is not None:
        return inherited
    return os.environ.get("OTEL_RESOURCE_ATTRIBUTES", "")


class SpanRegistry:
    """No-op registry — accepts registrations and forgets them."""

    @classmethod
    def register(cls, namespace, groups, presets=None, *, allow_override=False):
        """No-op."""

    @classmethod
    def unregister(cls, namespace):
        """No-op."""

    @classmethod
    def clear(cls):
        """No-op."""

    @classmethod
    def groups(cls):
        """Always empty."""
        return frozenset()

    @classmethod
    def namespaces(cls):
        """Always empty."""
        return []

    @classmethod
    def presets(cls):
        """Only the built-in wildcard, over an empty registry."""
        return {"all": frozenset()}

    @classmethod
    def resolve(cls, spec):
        """Nothing resolves; every spec entry comes back unknown."""
        return frozenset(), frozenset(p.strip().lower() for p in spec.split(",") if p.strip())
