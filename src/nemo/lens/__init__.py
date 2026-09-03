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
        safe_set_span_attributes,
        redact_value,
        DEFAULT_REDACT_KEYS,
        inject_context,
        extract_context,
        encode_resource_attributes,
        broadcast_trace_context,
        create_linked_span,
        get_tracer,
        get_meter,
        is_span_group_enabled,
        set_enabled_span_groups,
    )

Consumers that need to tolerate lens being absent at runtime can import
no-op fallbacks from :mod:`nemo.lens.fallbacks`.

Quick start
-----------

1. Set ``NEMO_LENS_ENABLED=1`` and optionally
   ``OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317``.
2. Call ``setup_telemetry(NemoLensConfig.from_env())`` once per process
   (raises ``RuntimeError`` on second call when enabled). Distributed callers
   pass their own identity via ``resource_attributes={DL_RANK: rank, ...}``.
3. Use ``managed_span``, ``trace_fn``, or ``span_cm`` at instrumentation sites.
"""

from opentelemetry import metrics as _metrics_mod
from opentelemetry import trace as _trace_mod

from nemo.lens.config import NemoLensConfig
from nemo.lens.distributed import broadcast_trace_context, create_linked_span
from nemo.lens.groups import SpanGroup
from nemo.lens.handle import TelemetryHandle, setup_telemetry
from nemo.lens.helpers import (
    DEFAULT_REDACT_KEYS,
    managed_span,
    redact_value,
    safe_set_span_attributes,
    span_cm,
    trace_fn,
)
from nemo.lens.package_info import (
    __contact_emails__,
    __contact_names__,
    __download_url__,
    __homepage__,
    __package_name__,
    __repository_url__,
    __version__,
)
from nemo.lens.propagation import extract_context, inject_context
from nemo.lens.resources import encode_resource_attributes
from nemo.lens.state import is_span_group_enabled, set_enabled_span_groups


def get_tracer(name: str = "nemo.lens") -> _trace_mod.Tracer:
    """Return the globally registered tracer."""
    return _trace_mod.get_tracer(name)


def get_meter(name: str = "nemo.lens") -> _metrics_mod.Meter:
    """Return the globally registered meter."""
    return _metrics_mod.get_meter(name)


__all__ = [
    "__version__",
    "__package_name__",
    "__contact_names__",
    "__contact_emails__",
    "__homepage__",
    "__repository_url__",
    "__download_url__",
    "NemoLensConfig",
    "SpanGroup",
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
    "encode_resource_attributes",
    "get_tracer",
    "get_meter",
    "is_span_group_enabled",
    "set_enabled_span_groups",
    "broadcast_trace_context",
    "create_linked_span",
]
