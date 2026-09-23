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

"""Reuse default tracers without repeated SDK warning-filter mutations."""

import os
import threading

from opentelemetry import trace

_LOCK = threading.RLock()
_PROVIDER = None
_TRACERS: dict[str, trace.Tracer] = {}


def _reset_after_fork() -> None:
    # Never acquire a lock inherited from a vanished parent thread.
    global _LOCK, _PROVIDER, _TRACERS
    _LOCK = threading.RLock()
    _PROVIDER = None
    _TRACERS = {}


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)


def get_tracer(name: str) -> trace.Tracer:
    """Resolve once per scope for the current provider and process.

    SDK 1.44 get_tracer() mutates warning filters even on a cache hit. Keeping
    that call off the span hot path preserves Python warning deduplication.
    Provider identity is checked so late setup and reinitialization still work.
    No SDK imports, rank policy, or warning filters belong in this helper.
    """
    global _PROVIDER
    provider = trace.get_tracer_provider()
    with _LOCK:
        if provider is not _PROVIDER:
            _TRACERS.clear()
            _PROVIDER = provider
        if name not in _TRACERS:
            _TRACERS[name] = provider.get_tracer(name)
        return _TRACERS[name]
