# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Training metric instruments (dl.training.* namespace)."""

from __future__ import annotations

import logging
import weakref

from opentelemetry import metrics

_logger = logging.getLogger(__name__)
_TRAINING_INSTRUMENTS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _get_training_instruments(meter: metrics.Meter) -> dict:
    instruments = _TRAINING_INSTRUMENTS.get(meter)
    if instruments is None:
        instruments = {
            "step_duration_ms": meter.create_histogram(
                name="dl.training.step_duration_ms",
                unit="ms",
                description="Duration of one training step in milliseconds.",
            ),
            "loss": meter.create_gauge(
                name="dl.training.loss",
                description="Training loss value at each log interval.",
            ),
            "throughput_tflops": meter.create_gauge(
                name="dl.training.throughput_tflops",
                description="Training throughput in TFLOP/s/GPU.",
            ),
            "grad_norm": meter.create_gauge(
                name="dl.training.grad_norm",
                description="Global gradient norm.",
            ),
            "skipped_iters": meter.create_counter(
                name="dl.training.skipped_iters",
                description="Number of training iterations skipped.",
            ),
            "learning_rate": meter.create_gauge(
                name="dl.training.learning_rate",
                description="Current learning rate.",
            ),
            "tokens_per_sec": meter.create_gauge(
                name="dl.training.tokens_per_sec",
                description="Training throughput in tokens/second.",
            ),
        }
        _TRAINING_INSTRUMENTS[meter] = instruments
    return instruments


def record_training_metrics(
    meter: metrics.Meter,
    step_duration_ms: float | None = None,
    loss: float | None = None,
    throughput_tflops: float | None = None,
    grad_norm: float | None = None,
    skipped_iters: int | None = None,
    learning_rate: float | None = None,
    tokens_per_sec: float | None = None,
) -> None:
    """Record training metrics to the OTel meter.

    All arguments are optional; ``None`` values are silently skipped.
    Safe to call when telemetry is disabled (meter is no-op).
    """
    try:
        instruments = _get_training_instruments(meter)
    except Exception:
        _logger.warning("Failed to create training metric instruments", exc_info=True)
        return

    if step_duration_ms is not None:
        instruments["step_duration_ms"].record(step_duration_ms)
    if loss is not None:
        instruments["loss"].set(loss)
    if throughput_tflops is not None:
        instruments["throughput_tflops"].set(throughput_tflops)
    if grad_norm is not None:
        instruments["grad_norm"].set(float(grad_norm))
    if skipped_iters is not None and skipped_iters > 0:
        instruments["skipped_iters"].add(skipped_iters)
    if learning_rate is not None:
        instruments["learning_rate"].set(learning_rate)
    if tokens_per_sec is not None:
        instruments["tokens_per_sec"].set(tokens_per_sec)
