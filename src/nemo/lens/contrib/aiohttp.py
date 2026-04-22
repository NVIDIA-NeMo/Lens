# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""aiohttp client auto-instrumentation helper.

Automatically injects W3C trace context into all outgoing aiohttp client
requests, replacing manual ``inject_context()`` calls.
"""

from __future__ import annotations


def instrument_aiohttp_client() -> None:
    """Enable automatic trace context injection for aiohttp client sessions.

    Requires ``opentelemetry-instrumentation-aiohttp-client >= 0.40b0``.
    """
    try:
        from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
    except ImportError as exc:
        raise ImportError(
            "aiohttp client instrumentation requires "
            "opentelemetry-instrumentation-aiohttp-client. "
            "Install with: pip install 'nemo-lens[aiohttp]'"
        ) from exc

    AioHttpClientInstrumentor().instrument()
