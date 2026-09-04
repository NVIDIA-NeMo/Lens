#!/usr/bin/env python3
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

"""Proxy for the training workload.

Stands in for Megatron so the workload side of the launch path can be tested
without one. Its own service, measuring itself: it takes no timestamps from the
launch layer and joins to it by identity, through OTEL_RESOURCE_ATTRIBUTES.
"""

import os
import sys
import time
from importlib import import_module

_IMPORTS_STARTED = time.time()


def _import_runtime_dependencies():
    context_module = import_module("opentelemetry.context")
    lens_module = import_module("nemo.lens")
    span_utilities_module = import_module("nemo.lens.span_utilities")
    return (
        context_module.Context,
        lens_module.NemoLensConfig,
        lens_module.setup_telemetry,
        span_utilities_module.emit_span,
        span_utilities_module.linux_process_create_time,
    )


(
    Context,
    NemoLensConfig,
    setup_telemetry,
    emit_span,
    linux_process_create_time,
) = _import_runtime_dependencies()

_IMPORTS_FINISHED = time.time()

SERVICE = "nv.dl.training"
STARTUP_SPAN = "nv.dl.training.python_startup"
IMPORTS_SPAN = "nv.dl.training.python_imports"


def main():
    created = linux_process_create_time()

    config = NemoLensConfig(
        enabled=True,
        service_name=SERVICE,
        traces_enabled=True,
        metrics_enabled=False,
        logs_enabled=False,
        exporter=("otlp" if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") else "console"),
    )
    handle = setup_telemetry(config)
    tracer = handle.tracer

    try:
        for name, a, b in (
            (STARTUP_SPAN, created, _IMPORTS_STARTED),
            (IMPORTS_SPAN, _IMPORTS_STARTED, _IMPORTS_FINISHED),
        ):
            emit_span(tracer, name, a, b, context=Context())
    finally:
        handle.shutdown()

    print(f"  {STARTUP_SPAN}  {_IMPORTS_STARTED - created:.3f}s", file=sys.stderr)
    print(
        f"  {IMPORTS_SPAN}  {_IMPORTS_FINISHED - _IMPORTS_STARTED:.3f}s",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
