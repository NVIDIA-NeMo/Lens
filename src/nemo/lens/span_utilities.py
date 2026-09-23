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

"""Utilities for emitting spans from caller-supplied timestamps."""

from __future__ import annotations

import logging
import os
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from opentelemetry import trace
from opentelemetry.context import Context

from nemo.lens._tracers import get_tracer
from nemo.lens.helpers import safe_set_span_attributes

_LOGGER = logging.getLogger(__name__)

_NANOSECONDS_PER_SECOND = 1_000_000_000


def emit_span(
    tracer: trace.Tracer | None,
    name: str,
    start_epoch_seconds: float,
    end_epoch_seconds: float,
    *,
    group: str | None = None,
    eps_ms: float = 30.0,
    context: Context | None = None,
    attributes: dict[str, Any] | None = None,
) -> trace.Span | None:
    """Emit a completed span for an interval in Unix-epoch seconds.

    Unless context is supplied, use the context active at this call, not the
    context that was active during the interval. Pass Context() for a root span.
    The emitted span is never made current.

    Pass None for the default tracer. A disabled group returns None before
    validation. Otherwise return the completed span.

    Inversions up to eps_ms (default 30 ms) set the end to the start and warn.
    Larger inversions raise ValueError; eps_ms=0 requires strict ordering.
    Timestamps must be finite and eps_ms must be finite and non-negative.
    """
    if group is not None:
        from nemo.lens.state import is_span_group_enabled

        if not is_span_group_enabled(group):
            return None
    start_seconds = _finite_epoch_seconds(start_epoch_seconds, "start_epoch_seconds")
    end_seconds = _finite_epoch_seconds(end_epoch_seconds, "end_epoch_seconds")
    tolerance_ms = _finite_epoch_seconds(eps_ms, "eps_ms")
    if tolerance_ms < 0:
        raise ValueError("eps_ms must be non-negative")
    tolerance_seconds = tolerance_ms / 1000
    if end_seconds < start_seconds:
        inversion_seconds = start_seconds - end_seconds
        if inversion_seconds > tolerance_seconds:
            raise ValueError(
                f"end_epoch_seconds={end_seconds} precedes start_epoch_seconds={start_seconds} "
                f"by {inversion_seconds} seconds, exceeding eps_ms={tolerance_ms} milliseconds"
            )
        _LOGGER.warning(
            "Span %s has inverted timestamps: start_epoch_seconds=%s, end_epoch_seconds=%s, "
            "inversion=%s seconds; clamping end to start (zero duration)",
            name,
            start_seconds,
            end_seconds,
            inversion_seconds,
        )
        end_seconds = start_seconds
    start_time = int(start_seconds * _NANOSECONDS_PER_SECOND)
    end_time = int(end_seconds * _NANOSECONDS_PER_SECOND)

    if tracer is None:
        tracer = get_tracer("nemo.lens")
    span = tracer.start_span(name, context=context, start_time=start_time)
    try:
        if attributes:
            safe_set_span_attributes(span, attributes)
    finally:
        span.end(end_time=end_time)
    return span


def linux_process_create_time(
    *,
    stat_text: str | None = None,
    uptime_text: str | None = None,
    read_time: float | None = None,
    stat_path: str | os.PathLike[str] = "/proc/self/stat",
    uptime_path: str | os.PathLike[str] = "/proc/uptime",
    clock_ticks_per_second: int | None = None,
) -> float:
    """Return process creation time in Unix-epoch seconds from Linux ``/proc`` data."""
    try:
        if stat_text is None:
            stat_text = Path(stat_path).read_text(encoding="utf-8")
        if uptime_text is None:
            uptime_text = Path(uptime_path).read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(
            "Linux process create time requires readable process stat and uptime data"
        ) from exc

    if read_time is None:
        read_time = time.time()
    if clock_ticks_per_second is None:
        clock_ticks_per_second = os.sysconf("SC_CLK_TCK")

    try:
        uptime_seconds = float(uptime_text.split()[0])
        process_age_seconds = (
            int(stat_text[stat_text.rindex(")") + 2 :].split()[19]) / clock_ticks_per_second
        )
    except (IndexError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError("Malformed Linux process stat or uptime data") from exc

    return read_time - (uptime_seconds - process_age_seconds)


def _finite_epoch_seconds(value: float, label: str) -> Decimal:
    try:
        seconds = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not seconds.is_finite():
        raise ValueError(f"{label} must be finite")
    return seconds
