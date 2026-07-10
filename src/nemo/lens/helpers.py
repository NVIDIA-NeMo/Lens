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

"""Telemetry helper utilities: span_cm, managed_span, trace_fn, safe_set_span_attributes."""

import functools
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace

# ---------------------------------------------------------------------------
# Attribute key redaction
# ---------------------------------------------------------------------------

DEFAULT_REDACT_KEYS: frozenset = frozenset(
    {"prompt", "input_text", "output_text", "text", "password", "token", "secret", "key"}
)

_SCALAR_TYPES = (bool, int, float, str)


def redact_value(key: str, value: str, redact_keys: frozenset = DEFAULT_REDACT_KEYS) -> str:
    """Return ``'[REDACTED]'`` if *key* is in *redact_keys*, else *value*."""
    return "[REDACTED]" if key in redact_keys else value


def safe_set_span_attributes(
    span: trace.Span,
    attributes: dict,
    redact_keys: frozenset = DEFAULT_REDACT_KEYS,
) -> None:
    """Set span attributes, silently skipping non-scalar values.

    OTel span attributes must be scalars (bool, int, float, str) or sequences
    of scalars. Redacts string values whose keys are in *redact_keys*.
    """
    if not span.is_recording():
        return
    for key, value in attributes.items():
        if value is None:
            continue
        if isinstance(value, _SCALAR_TYPES):
            if isinstance(value, str):
                value = redact_value(key, value, redact_keys)
            span.set_attribute(key, value)
        elif isinstance(value, list | tuple) and all(isinstance(v, _SCALAR_TYPES) for v in value):
            span.set_attribute(key, list(value))


@contextmanager
def span_cm(
    name: str,
    tracer: trace.Tracer | None = None,
    record_exception: bool = True,
    group: str | None = None,
    **attributes: Any,
):
    """Context manager that creates an OTel span for a code block.

    Safe to use with no-op tracers.

    Unlike :func:`managed_span`, this does NOT gate on a span group (the caller
    decides whether to enter). Pass *group* purely so the span still carries the
    ``lens.group`` / ``lens.span_category`` attributes for offline slicing.

    Args:
        name: Span name.
        tracer: OTel tracer. Defaults to the global tracer.
        record_exception: If True, record exceptions as span events.
        group: Optional span group for ``lens.group``/``lens.span_category``
            tagging only (no gating).
        **attributes: Key/value pairs set as span attributes.

    Yields:
        The active Span.
    """
    if tracer is None:
        tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span(name, record_exception=record_exception) as span:
        if group is not None:
            from nemo.lens.state import category_of

            span.set_attribute("lens.group", group)
            _category = category_of(group)
            if _category is not None:
                span.set_attribute("lens.span_category", _category)
        if attributes:
            safe_set_span_attributes(span, attributes)
        yield span


@contextmanager
def managed_span(
    group: str,
    name: str,
    tracer: trace.Tracer | None = None,
    **attributes: Any,
):
    """Explicit-lifecycle span guarded by a span-group check.

    When the group is disabled the body executes normally and ``None`` is
    yielded — no span object is created. When enabled, the span is started,
    its context attached, and always ended in a ``finally`` block.

    Every emitted span is tagged with ``lens.group`` (this group) and
    ``lens.span_category`` (``goodput``/``profiling``, derived from the group).
    A span has exactly ONE category. If a code region is *both* a goodput
    boundary and something you want profiled with depth, do NOT try to make one
    span serve both — wrap it in two nested spans: an outer goodput-group span
    (the stable semantic boundary) and an inner profiling-group span (the
    detail). Two spans, two categories, each cleanly filterable.

    Args:
        group: Span group name. If not enabled, this is a zero-overhead no-op.
        name: Span name.
        tracer: OTel tracer. Defaults to the global tracer.
        **attributes: Key/value pairs set as span attributes.

    Yields:
        The active Span, or ``None`` when the group is disabled.
    """
    from nemo.lens.state import category_of, is_span_group_enabled

    if not is_span_group_enabled(group):
        yield None
        return

    from opentelemetry import context as otel_ctx
    from opentelemetry.trace import StatusCode, set_span_in_context

    if tracer is None:
        tracer = trace.get_tracer(__name__)

    span = tracer.start_span(name)
    span.set_attribute("lens.group", group)
    _category = category_of(group)
    if _category is not None:
        span.set_attribute("lens.span_category", _category)
    if attributes:
        safe_set_span_attributes(span, attributes)
    token = otel_ctx.attach(set_span_in_context(span))
    try:
        yield span
    except Exception as exc:
        span.record_exception(exc)
        span.set_status(StatusCode.ERROR, str(exc))
        raise
    finally:
        otel_ctx.detach(token)
        span.end()


def trace_fn(group: str, name: str, tracer: trace.Tracer | None = None):
    """Decorator that wraps a function in a group-gated OTel span.

    The span group is checked at **call time** (not decoration time).

    Args:
        group: Span group name.
        name: OTel span name.
        tracer: OTel tracer. Defaults to the global tracer.

    Returns:
        A decorator that wraps the target function.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            from nemo.lens.state import category_of, is_span_group_enabled

            if not is_span_group_enabled(group):
                return func(*args, **kwargs)
            t = tracer if tracer is not None else trace.get_tracer("nemo.lens")
            with t.start_as_current_span(name) as span:
                span.set_attribute("lens.group", group)
                _category = category_of(group)
                if _category is not None:
                    span.set_attribute("lens.span_category", _category)
                return func(*args, **kwargs)

        return wrapper

    return decorator
