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

import os
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from opentelemetry import trace
from opentelemetry.context import Context

from nemo.lens.helpers import safe_set_span_attributes

_NANOSECONDS_PER_SECOND = 1_000_000_000


def emit_span(
    tracer: trace.Tracer,
    name: str,
    start_epoch_seconds: float,
    end_epoch_seconds: float,
    *,
    context: Context | None = None,
    attributes: dict[str, Any] | None = None,
) -> trace.Span:
    """Emit one span with explicit Unix-epoch second start and end times."""
    start_time = _epoch_seconds_to_nanoseconds(start_epoch_seconds, "start_epoch_seconds")
    end_time = _epoch_seconds_to_nanoseconds(end_epoch_seconds, "end_epoch_seconds")
    if end_time < start_time:
        raise ValueError("end_epoch_seconds must be greater than or equal to start_epoch_seconds")

    span = tracer.start_span(name, context=context, start_time=start_time)
    if attributes:
        safe_set_span_attributes(span, attributes)
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


def _epoch_seconds_to_nanoseconds(value: float, label: str) -> int:
    try:
        seconds = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not seconds.is_finite():
        raise ValueError(f"{label} must be finite")
    return int(seconds * _NANOSECONDS_PER_SECOND)
