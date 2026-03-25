# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Ray context propagation helpers.

Provides utilities for propagating OTel trace context across Ray remote
calls, enabling distributed traces that span driver -> worker boundaries.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from opentelemetry import trace

from nemo.lens.propagation import extract_context, inject_context


def inject_ray_context() -> dict:
    """Create a carrier dict with the current trace context for Ray.

    Call this on the driver side before dispatching a remote call.

    Returns:
        A dict containing W3C trace context headers.
    """
    carrier: dict = {}
    inject_context(carrier)
    return carrier


def extract_ray_context(carrier: dict | None = None):
    """Extract trace context from a Ray carrier.

    Call this on the worker side to resume the trace.

    Args:
        carrier: Dict with W3C trace context headers. If None, returns
            an empty context.

    Returns:
        An OTel Context.
    """
    if carrier is None:
        from opentelemetry.context import get_current

        return get_current()
    return extract_context(carrier)


def traced_remote_call(
    method: Callable,
    tracer: trace.Tracer | None = None,
) -> Callable:
    """Wrapper for Ray remote methods that auto-injects/extracts trace context.

    Apply to worker methods to automatically propagate traces across
    Ray remote boundaries.

    Args:
        method: The function to wrap.
        tracer: OTel tracer. Defaults to the global tracer.

    Returns:
        Wrapped function that accepts ``_otel_carrier`` kwarg.
    """

    @functools.wraps(method)
    def wrapper(*args: Any, _otel_carrier: dict | None = None, **kwargs: Any) -> Any:
        ctx = extract_ray_context(_otel_carrier)
        t = tracer or trace.get_tracer("nemo.lens.ray")
        with t.start_as_current_span(method.__qualname__, context=ctx):
            return method(*args, **kwargs)

    return wrapper


def ray_dispatch_with_context(
    remote_fn,
    *args: Any,
    _carrier: dict | None = None,
    **kwargs: Any,
):
    """Dispatch a Ray remote call with trace context injection.

    Args:
        remote_fn: The Ray remote function/method reference.
        *args: Positional arguments for the remote call.
        _carrier: Pre-built carrier (if None, injects current context).
        **kwargs: Keyword arguments for the remote call.

    Returns:
        The Ray ObjectRef from the remote call.
    """
    if _carrier is None:
        _carrier = inject_ray_context()
    return remote_fn.remote(*args, _otel_carrier=_carrier, **kwargs)
