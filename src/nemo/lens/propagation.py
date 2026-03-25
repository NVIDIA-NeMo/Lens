# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""W3C TraceContext / Baggage propagation helpers."""

from opentelemetry import context, propagate


def inject_context(carrier: dict) -> None:
    """Inject the current span context into *carrier* (W3C TraceContext + Baggage).

    Use this to propagate trace context across process boundaries, e.g. into
    HTTP header dicts or gRPC metadata.

    Args:
        carrier: A mutable dict that will receive the ``traceparent``
            (and optionally ``tracestate``, ``baggage``) headers.
    """
    propagate.inject(carrier)


def extract_context(carrier: dict) -> context.Context:
    """Extract span context from *carrier* and return an OTel Context.

    Use this to resume a distributed trace from incoming headers/metadata.

    Args:
        carrier: A dict containing W3C ``traceparent`` (and optionally
            ``tracestate``, ``baggage``) headers.

    Returns:
        An OTel Context. If no valid trace context is present the returned
        context is empty.
    """
    return propagate.extract(carrier)
