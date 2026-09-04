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

"""Command line tools for nemo-lens."""

from __future__ import annotations

import argparse
import math
import os
import shlex
import socket
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TextIO

from opentelemetry import trace
from opentelemetry.context import Context

from nemo.lens.providers import _build_span_emitter_provider
from nemo.lens.resources.attributes import (
    OTEL_RESOURCE_ATTRIBUTES_ENV,
    set_otel_resource_attributes,
)
from nemo.lens.resources.slurm import derive_slurm_resource_attributes
from nemo.lens.semconv import (
    HOST_NAME,
    NV_DL_LAUNCH_CONTAINER_IMAGE,
    SLURM_TOPOLOGY_ADDR,
    SLURM_TOPOLOGY_ADDR_PATTERN,
)
from nemo.lens.span_utilities import emit_span

_TASK_LOCAL_SLURM_RESOURCE_KEYS = frozenset(
    {
        SLURM_TOPOLOGY_ADDR,
        SLURM_TOPOLOGY_ADDR_PATTERN,
    }
)


@dataclass(frozen=True, slots=True)
class _SpanSpec:
    name: str
    start: float
    end: float
    parent: str | None = None


@dataclass(frozen=True, slots=True)
class _SpanValidation:
    order: list[_SpanSpec]
    invalid_count: int = 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``nemo-lens`` command line interface."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "emit-spans":
        return _run_emit_spans(args)
    if args.command == "set-slurm-resource-attrs":
        return _run_set_slurm_resource_attrs(args)

    parser.error("missing command")
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nemo-lens")
    subparsers = parser.add_subparsers(dest="command", required=True)

    emit_spans = subparsers.add_parser(
        "emit-spans",
        description="Emit spans from explicit epoch-second intervals.",
        help="emit spans from explicit intervals",
    )
    emit_spans.add_argument("--service", required=True, help="OpenTelemetry service.name")
    emit_spans.add_argument(
        "--span",
        action="append",
        default=[],
        required=True,
        metavar="NAME,START,END[,PARENT]",
        help="Span name, start epoch seconds, end epoch seconds, and optional parent name.",
    )

    set_slurm_attrs = subparsers.add_parser(
        "set-slurm-resource-attrs",
        description=(
            "Print a shell export that appends canonical SLURM resource attributes to "
            "OTEL_RESOURCE_ATTRIBUTES."
        ),
        help="print shell export for SLURM resource attributes",
    )
    set_slurm_attrs.add_argument(
        "--stage",
        required=True,
        choices=("sbatch", "task"),
        help="Launch stage where the command runs.",
    )
    set_slurm_attrs.add_argument(
        "--container-image",
        help=f"Container image path/name to add as {NV_DL_LAUNCH_CONTAINER_IMAGE}.",
    )
    return parser


def _run_emit_spans(
    args: argparse.Namespace,
    *,
    span_exporter: Any | None = None,
    stderr: TextIO | None = None,
) -> int:
    stderr = stderr or sys.stderr
    validation = _validate_span_args(args.span, stderr=stderr)
    if validation.invalid_count or not validation.order:
        print("nothing emitted: invalid span input", file=stderr)
        return 1
    order = validation.order

    try:
        provider, tracked_exporter = _build_span_emitter_provider(
            args.service,
            span_exporter=span_exporter,
        )
    except (ImportError, ValueError) as exc:
        print(f"error: {exc}", file=stderr)
        return 2

    tracer = trace.get_tracer(args.service)
    emitted: dict[str, trace.Span] = {}
    flush_succeeded = False
    try:
        for spec in order:
            context = trace.set_span_in_context(emitted[spec.parent]) if spec.parent else Context()
            emitted[spec.name] = emit_span(
                tracer,
                spec.name,
                spec.start,
                spec.end,
                context=context,
            )

        flush_succeeded = provider.force_flush()
    finally:
        provider.shutdown()

    if not flush_succeeded:
        print("error: timed out while flushing spans", file=stderr)
        return 2
    if tracked_exporter.failed:
        print("error: failed to export spans", file=stderr)
        return 2

    for spec in order:
        print(
            f"  {spec.name:34s} {spec.end - spec.start:8.3f}s"
            f"{'' if spec.parent is None else '  -> ' + spec.parent}",
            file=stderr,
        )
    return 0


def _run_set_slurm_resource_attrs(
    args: argparse.Namespace,
    *,
    environ: Mapping[str, str] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    if args.container_image and args.stage != "task":
        print("error: --container-image requires --stage task", file=stderr)
        return 2

    env = dict(os.environ if environ is None else environ)
    additions = derive_slurm_resource_attributes(env)

    if args.stage == "sbatch":
        for key in _TASK_LOCAL_SLURM_RESOURCE_KEYS:
            additions.pop(key, None)

    if args.stage == "task":
        additions[HOST_NAME] = socket.gethostname()
        if args.container_image:
            additions[NV_DL_LAUNCH_CONTAINER_IMAGE] = args.container_image

    value = set_otel_resource_attributes(additions, environ=env, overwrite=False)
    print(f"export {OTEL_RESOURCE_ATTRIBUTES_ENV}={shlex.quote(value)}", file=stdout)
    return 0


def _validate_span_args(
    span_args: Sequence[str], *, stderr: TextIO | None = None
) -> _SpanValidation:
    stderr = stderr or sys.stderr
    spans: list[_SpanSpec] = []
    invalid_count = 0

    for span_arg in span_args:
        spec = _parse_span_arg(span_arg, stderr=stderr)
        if spec is None:
            invalid_count += 1
            continue
        spans.append(spec)

    ordered = _order_span_specs(spans, stderr=stderr)
    return _SpanValidation(ordered.order, invalid_count + ordered.invalid_count)


def _parse_span_arg(arg: str, *, stderr: TextIO | None = None) -> _SpanSpec | None:
    stderr = stderr or sys.stderr
    parts = [part.strip() for part in arg.split(",")]
    if len(parts) not in (3, 4):
        print(f"error: skipping {arg!r}: expected NAME,START,END[,PARENT]", file=stderr)
        return None

    name = parts[0]
    start = parts[1]
    end = parts[2]
    parent = parts[3] if len(parts) == 4 else ""

    if not name:
        print(f"error: skipping {arg!r}: missing span name", file=stderr)
        return None

    if not start or not end:
        print(f"error: skipping {name!r}: missing timestamp", file=stderr)
        return None

    try:
        start_epoch_seconds = float(start)
        end_epoch_seconds = float(end)
    except ValueError:
        print(f"error: skipping {name!r}: unparseable timestamp", file=stderr)
        return None

    if not math.isfinite(start_epoch_seconds) or not math.isfinite(end_epoch_seconds):
        print(f"error: skipping {name!r}: timestamp must be finite", file=stderr)
        return None

    if end_epoch_seconds < start_epoch_seconds:
        print(f"error: skipping {name!r}: end timestamp precedes start timestamp", file=stderr)
        return None

    return _SpanSpec(name, start_epoch_seconds, end_epoch_seconds, parent or None)


def _order_span_specs(
    spans: Sequence[_SpanSpec], *, stderr: TextIO | None = None
) -> _SpanValidation:
    stderr = stderr or sys.stderr
    invalid_count = 0

    name_counts = Counter(spec.name for spec in spans)
    duplicate_names = {name for name, count in name_counts.items() if count > 1}
    for name in sorted(duplicate_names):
        print(f"error: invalid span {name!r}: duplicate span name", file=stderr)
        invalid_count += 1

    candidate_specs: list[_SpanSpec] = []
    candidate_names = {spec.name for spec in spans if spec.name not in duplicate_names}
    for spec in spans:
        if spec.name in duplicate_names:
            continue
        if spec.parent in duplicate_names:
            print(
                f"error: invalid span {spec.name!r}: parent {spec.parent!r} is duplicated",
                file=stderr,
            )
            invalid_count += 1
        elif spec.parent is not None and spec.parent not in candidate_names:
            print(f"error: invalid span {spec.name!r}: missing parent {spec.parent!r}", file=stderr)
            invalid_count += 1
        else:
            candidate_specs.append(spec)

    if invalid_count:
        return _SpanValidation([], invalid_count)

    order: list[_SpanSpec] = []
    ordered_names: set[str] = set()
    remaining = list(candidate_specs)

    while remaining:
        ready = [spec for spec in remaining if spec.parent is None or spec.parent in ordered_names]
        if not ready:
            for spec in remaining:
                print(
                    f"error: invalid span graph: parent cycle prevents emitting {spec.name!r}",
                    file=stderr,
                )
                invalid_count += 1
            return _SpanValidation([], invalid_count)

        order.extend(ready)
        ordered_names.update(spec.name for spec in ready)
        remaining = [spec for spec in remaining if spec.name not in ordered_names]

    return _SpanValidation(order)


if __name__ == "__main__":
    raise SystemExit(main())
