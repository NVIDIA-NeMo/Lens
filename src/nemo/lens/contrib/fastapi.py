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

"""FastAPI auto-instrumentation helper."""

from __future__ import annotations


def instrument_fastapi(app, service_name: str = "nemo-gym") -> None:
    """Apply OpenTelemetry auto-instrumentation to a FastAPI app.

    Requires ``opentelemetry-instrumentation-fastapi >= 0.40b0``.

    Args:
        app: The FastAPI application instance.
        service_name: Service name for span naming.
    """
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    except ImportError as exc:
        raise ImportError(
            "FastAPI instrumentation requires opentelemetry-instrumentation-fastapi. "
            "Install with: pip install 'nemo-lens[fastapi]'"
        ) from exc

    FastAPIInstrumentor.instrument_app(app)
