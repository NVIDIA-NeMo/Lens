# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""nemo-lens: Shared OpenTelemetry library for the NVIDIA NeMo ecosystem.

Public API
----------

.. code-block:: python

    from nemo.lens import (
        NemoLensConfig,
        SpanGroup,
        TelemetryHandle,
        setup_telemetry,
        span_cm,
        managed_span,
        trace_fn,
        inject_context,
        extract_context,
        get_tracer,
        get_meter,
        is_span_group_enabled,
    )

Quick start
-----------

1. Set ``NEMO_LENS_ENABLED=1`` and optionally
   ``OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317``.
2. Call ``setup_telemetry(NemoLensConfig.from_env(), rank, world_size)``.
3. Use ``managed_span``, ``trace_fn``, or ``span_cm`` at instrumentation sites.
"""

from opentelemetry import metrics as _metrics_mod
from opentelemetry import trace as _trace_mod

from nemo.lens._version import __version__
from nemo.lens.config import NemoLensConfig
from nemo.lens.groups import SpanGroup
from nemo.lens.groups_gym import GymSpanGroup
from nemo.lens.groups_megatron import MegatronSpanGroup
from nemo.lens.groups_rl import RLSpanGroup
from nemo.lens.handle import TelemetryHandle, setup_telemetry
from nemo.lens.helpers import (
    DEFAULT_REDACT_KEYS,
    managed_span,
    redact_value,
    safe_set_span_attributes,
    span_cm,
    trace_fn,
)
from nemo.lens.propagation import extract_context, inject_context
from nemo.lens.state import is_span_group_enabled, set_enabled_span_groups


def get_tracer(name: str = "nemo.lens") -> _trace_mod.Tracer:
    """Return the globally registered tracer."""
    return _trace_mod.get_tracer(name)


def get_meter(name: str = "nemo.lens") -> _metrics_mod.Meter:
    """Return the globally registered meter."""
    return _metrics_mod.get_meter(name)


__all__ = [
    "__version__",
    "NemoLensConfig",
    "SpanGroup",
    "MegatronSpanGroup",
    "RLSpanGroup",
    "GymSpanGroup",
    "TelemetryHandle",
    "setup_telemetry",
    "span_cm",
    "managed_span",
    "trace_fn",
    "safe_set_span_attributes",
    "redact_value",
    "DEFAULT_REDACT_KEYS",
    "inject_context",
    "extract_context",
    "get_tracer",
    "get_meter",
    "is_span_group_enabled",
    "set_enabled_span_groups",
]
