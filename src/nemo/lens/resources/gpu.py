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

"""GPU identity detection."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import TYPE_CHECKING

from nemo.lens.semconv import (
    NV_GPU_COMPUTE_CAPABILITY,
    NV_GPU_DRIVER_VERSION,
    NV_GPU_INDEX,
    NV_GPU_MEMORY_TOTAL,
    NV_GPU_MODEL,
    NV_GPU_PCI_BUS_ID,
    NV_GPU_SERIAL,
    NV_GPU_UUID,
)

if TYPE_CHECKING:
    from nemo.lens.resources.attributes import ResourceAttributeValue


def detect_gpu(
    local_rank: int | str | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, ResourceAttributeValue]:
    """Detect worker-local GPU identity from NVML.

    ``local_rank`` is the rank's index within ``CUDA_VISIBLE_DEVICES``. Callers
    that already know their local rank should pass it explicitly; otherwise the
    helper falls back to common launcher environment variables and then rank 0.
    """
    env = os.environ if environ is None else environ
    physical_index = _physical_gpu_index(local_rank, env)
    if physical_index is None:
        return {}

    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(physical_index)
            attrs: dict[str, ResourceAttributeValue] = {
                NV_GPU_INDEX: physical_index,
                NV_GPU_MODEL: _decode(pynvml.nvmlDeviceGetName(handle)),
                NV_GPU_UUID: _decode(pynvml.nvmlDeviceGetUUID(handle)),
            }
            _set_optional(attrs, NV_GPU_SERIAL, pynvml.nvmlDeviceGetSerial, handle)
            _set_optional_field(
                attrs,
                NV_GPU_PCI_BUS_ID,
                lambda: pynvml.nvmlDeviceGetPciInfo(handle).busId,
            )
            _set_compute_capability(attrs, pynvml, handle)
            _set_optional_field(
                attrs,
                NV_GPU_MEMORY_TOTAL,
                lambda: int(pynvml.nvmlDeviceGetMemoryInfo(handle).total),
            )
            _set_optional_field(
                attrs,
                NV_GPU_DRIVER_VERSION,
                pynvml.nvmlSystemGetDriverVersion,
            )
            return attrs
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return {}


def _physical_gpu_index(
    local_rank: int | str | None,
    env: Mapping[str, str],
) -> int | None:
    try:
        rank = int(_first_value(local_rank, env, "LOCAL_RANK", "SLURM_LOCALID") or 0)
    except (TypeError, ValueError):
        rank = 0

    cuda_visible = env.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not cuda_visible:
        return rank

    visible = [item.strip() for item in cuda_visible.split(",")]
    if rank >= len(visible):
        return None
    try:
        return int(visible[rank])
    except ValueError:
        return None


def _first_value(
    explicit: int | str | None,
    env: Mapping[str, str],
    *env_names: str,
) -> int | str | None:
    if explicit is not None:
        return explicit
    for name in env_names:
        value = env.get(name)
        if value not in (None, ""):
            return value
    return None


def _set_optional(
    attrs: dict[str, ResourceAttributeValue],
    key: str,
    getter,
    handle,
) -> None:
    _set_optional_field(attrs, key, lambda: getter(handle))


def _set_optional_field(
    attrs: dict[str, ResourceAttributeValue],
    key: str,
    getter,
) -> None:
    try:
        value = getter()
    except Exception:
        return
    attrs[key] = _decode(value)


def _set_compute_capability(attrs, pynvml, handle) -> None:
    try:
        major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
    except Exception:
        return
    attrs[NV_GPU_COMPUTE_CAPABILITY] = f"{major}.{minor}"


def _decode(value):
    return value.decode() if isinstance(value, bytes) else value
