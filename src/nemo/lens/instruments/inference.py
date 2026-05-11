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

"""Inference metric instruments (GenAI semconv)."""

from __future__ import annotations

import logging
import weakref

from opentelemetry import metrics

_logger = logging.getLogger(__name__)
_INFERENCE_INSTRUMENTS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _get_inference_instruments(meter: metrics.Meter) -> dict:
    instruments = _INFERENCE_INSTRUMENTS.get(meter)
    if instruments is None:
        instruments = {
            "server_request_duration": meter.create_histogram(
                name="gen_ai.server.request.duration",
                unit="s",
                description="GenAI server request duration.",
            ),
            "token_usage": meter.create_histogram(
                name="gen_ai.client.token.usage",
                unit="{token}",
                description="Number of input and output tokens used.",
            ),
        }
        _INFERENCE_INSTRUMENTS[meter] = instruments
    return instruments


_PROVIDER_NAME = "nemo"
_OPERATION_NAME = "text_completion"


def record_inference_metrics(
    meter: metrics.Meter,
    request_duration_s: float | None = None,
    model: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    provider_name: str = _PROVIDER_NAME,
    operation_name: str = _OPERATION_NAME,
) -> None:
    """Record inference metrics (GenAI semconv).

    Emits ``gen_ai.server.request.duration`` and ``gen_ai.client.token.usage``.
    """
    try:
        instruments = _get_inference_instruments(meter)
    except Exception:
        _logger.warning("Failed to create inference metric instruments", exc_info=True)
        return

    base_attrs: dict = {
        "gen_ai.operation.name": operation_name,
        "gen_ai.provider.name": provider_name,
    }
    if model:
        base_attrs["gen_ai.request.model"] = str(model)

    if request_duration_s is not None:
        instruments["server_request_duration"].record(request_duration_s, attributes=base_attrs)
    if input_tokens is not None:
        instruments["token_usage"].record(
            input_tokens,
            attributes={**base_attrs, "gen_ai.token.type": "input"},
        )
    if output_tokens is not None:
        instruments["token_usage"].record(
            output_tokens,
            attributes={**base_attrs, "gen_ai.token.type": "output"},
        )
