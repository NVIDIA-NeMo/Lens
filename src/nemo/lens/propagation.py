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
