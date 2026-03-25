# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""FastAPI auto-instrumentation helper."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def instrument_fastapi(app, service_name: str = 'nemo-gym') -> None:
    """Apply OpenTelemetry auto-instrumentation to a FastAPI app.

    Requires ``opentelemetry-instrumentation-fastapi >= 0.40b0``.

    Args:
        app: The FastAPI application instance.
        service_name: Service name for span naming.
    """
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    except ImportError:
        raise ImportError(
            "FastAPI instrumentation requires opentelemetry-instrumentation-fastapi. "
            "Install with: pip install 'nemo-lens[fastapi]'"
        )

    FastAPIInstrumentor.instrument_app(app)
