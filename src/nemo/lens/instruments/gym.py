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

"""Gym metric instruments (gym.* namespace)."""

from __future__ import annotations

import logging
import weakref

from opentelemetry import metrics

_logger = logging.getLogger(__name__)
_GYM_INSTRUMENTS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _get_gym_instruments(meter: metrics.Meter) -> dict:
    instruments = _GYM_INSTRUMENTS.get(meter)
    if instruments is None:
        instruments = {
            "server_request_duration_ms": meter.create_histogram(
                name="gym.server.request_duration_ms",
                unit="ms",
                description="Gym server request duration in milliseconds.",
            ),
            "rollout_duration_ms": meter.create_histogram(
                name="gym.rollout.duration_ms",
                unit="ms",
                description="Rollout collection duration in milliseconds.",
            ),
            "verify_duration_ms": meter.create_histogram(
                name="gym.verify.duration_ms",
                unit="ms",
                description="Verification endpoint duration in milliseconds.",
            ),
            "verify_success_rate": meter.create_gauge(
                name="gym.verify.success_rate",
                description="Fraction of successful verifications.",
            ),
            "active_servers": meter.create_gauge(
                name="gym.servers.active",
                description="Number of active Gym servers.",
            ),
        }
        _GYM_INSTRUMENTS[meter] = instruments
    return instruments


def record_gym_metrics(
    meter: metrics.Meter,
    server_request_duration_ms: float | None = None,
    rollout_duration_ms: float | None = None,
    verify_duration_ms: float | None = None,
    verify_success_rate: float | None = None,
    active_servers: int | None = None,
) -> None:
    """Record Gym metrics. All arguments optional; None values skipped."""
    try:
        instruments = _get_gym_instruments(meter)
    except Exception:
        _logger.warning("Failed to create Gym metric instruments", exc_info=True)
        return

    if server_request_duration_ms is not None:
        instruments["server_request_duration_ms"].record(server_request_duration_ms)
    if rollout_duration_ms is not None:
        instruments["rollout_duration_ms"].record(rollout_duration_ms)
    if verify_duration_ms is not None:
        instruments["verify_duration_ms"].record(verify_duration_ms)
    if verify_success_rate is not None:
        instruments["verify_success_rate"].set(verify_success_rate)
    if active_servers is not None:
        instruments["active_servers"].set(active_servers)
