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

"""RL metric instruments (rl.* namespace)."""

from __future__ import annotations

import logging
import weakref

from opentelemetry import metrics

_logger = logging.getLogger(__name__)
_RL_INSTRUMENTS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _get_rl_instruments(meter: metrics.Meter) -> dict:
    instruments = _RL_INSTRUMENTS.get(meter)
    if instruments is None:
        instruments = {
            "reward_mean": meter.create_gauge(
                name="rl.reward.mean",
                description="Mean reward across rollout batch.",
            ),
            "kl_divergence": meter.create_gauge(
                name="rl.kl_divergence",
                description="KL divergence between policy and reference.",
            ),
            "policy_loss": meter.create_gauge(
                name="rl.policy_loss",
                description="Policy gradient loss.",
            ),
            "value_loss": meter.create_gauge(
                name="rl.value_loss",
                description="Value function loss.",
            ),
            "entropy": meter.create_gauge(
                name="rl.entropy",
                description="Policy entropy.",
            ),
            "response_length_mean": meter.create_gauge(
                name="rl.response_length.mean",
                description="Mean generated response length (tokens).",
            ),
            "grad_norm": meter.create_gauge(
                name="rl.grad_norm",
                description="Gradient norm of the policy update.",
            ),
            "learning_rate": meter.create_gauge(
                name="rl.learning_rate",
                description="Current optimizer learning rate.",
            ),
            "tokens_per_sec": meter.create_gauge(
                name="rl.tokens_per_sec",
                description="Training throughput in tokens per second.",
            ),
            "generation_duration_ms": meter.create_histogram(
                name="rl.generation.duration_ms",
                unit="ms",
                description="Duration of text generation in milliseconds.",
            ),
            "rollout_duration_ms": meter.create_histogram(
                name="rl.rollout.duration_ms",
                unit="ms",
                description="Duration of rollout collection in milliseconds.",
            ),
        }
        _RL_INSTRUMENTS[meter] = instruments
    return instruments


def record_rl_metrics(
    meter: metrics.Meter,
    reward_mean: float | None = None,
    kl_divergence: float | None = None,
    policy_loss: float | None = None,
    value_loss: float | None = None,
    entropy: float | None = None,
    response_length_mean: float | None = None,
    grad_norm: float | None = None,
    learning_rate: float | None = None,
    tokens_per_sec: float | None = None,
    generation_duration_ms: float | None = None,
    rollout_duration_ms: float | None = None,
) -> None:
    """Record RL training metrics. All arguments optional; None values skipped."""
    try:
        instruments = _get_rl_instruments(meter)
    except Exception:
        _logger.warning("Failed to create RL metric instruments", exc_info=True)
        return

    if reward_mean is not None:
        instruments["reward_mean"].set(reward_mean)
    if kl_divergence is not None:
        instruments["kl_divergence"].set(kl_divergence)
    if policy_loss is not None:
        instruments["policy_loss"].set(policy_loss)
    if value_loss is not None:
        instruments["value_loss"].set(value_loss)
    if entropy is not None:
        instruments["entropy"].set(entropy)
    if response_length_mean is not None:
        instruments["response_length_mean"].set(response_length_mean)
    if grad_norm is not None:
        instruments["grad_norm"].set(grad_norm)
    if learning_rate is not None:
        instruments["learning_rate"].set(learning_rate)
    if tokens_per_sec is not None:
        instruments["tokens_per_sec"].set(tokens_per_sec)
    if generation_duration_ms is not None:
        instruments["generation_duration_ms"].record(generation_duration_ms)
    if rollout_duration_ms is not None:
        instruments["rollout_duration_ms"].record(rollout_duration_ms)
