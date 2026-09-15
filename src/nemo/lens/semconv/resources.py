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

"""Resource attribute field sets and type mappings."""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping

from .attributes import (
    NV_DL_JOB_UUID,
    NV_DL_LOCAL_RANK,
    NV_DL_PROVIDER_NAME,
    NV_DL_RANK,
    NV_DL_ROLE,
    NV_DL_RUN_UUID,
    NV_DL_SOFTWARE_CUDA,
    NV_DL_SOFTWARE_NCCL,
    NV_DL_SOFTWARE_TORCH,
    NV_DL_SOFTWARE_TRANSFORMER_ENGINE,
    NV_DL_TOPOLOGY_SIZE_DP,
    NV_DL_TOPOLOGY_SIZE_PP,
    NV_DL_TOPOLOGY_SIZE_TP,
    NV_DL_TRAINING_CONFIG_GLOBAL_BATCH_SIZE,
    NV_DL_TRAINING_CONFIG_MICRO_BATCH_SIZE,
    NV_DL_TRAINING_CONFIG_OPTIMIZER,
    NV_DL_TRAINING_CONFIG_RECOMPUTE_GRANULARITY,
    NV_DL_TRAINING_CONFIG_SEQUENCE_LENGTH,
    NV_DL_TRAINING_TARGET_TRAIN_ITERS,
    NV_DL_TRAINING_TARGET_TRAIN_SAMPLES,
    NV_DL_TRAINING_TARGET_TRAIN_TOKENS,
    NV_DL_WORLD_SIZE,
    NV_GPU_INDEX,
    NV_GPU_MEMORY_TOTAL,
    SLURM_ARRAY_COUNT,
    SLURM_ARRAY_JOB_ID,
    SLURM_ARRAY_SLUID,
    SLURM_ARRAY_TASK_ID,
    SLURM_CLUSTER,
    SLURM_CLUSTER_NAME,
    SLURM_HEAD_NODE_NAME,
    SLURM_JOB_ACCOUNT,
    SLURM_JOB_ID,
    SLURM_JOB_ID_RAW,
    SLURM_JOB_NAME,
    SLURM_JOB_QOS,
    SLURM_JOB_RESERVATION,
    SLURM_JOB_USER,
    SLURM_NNODES,
    SLURM_NODELIST,
    SLURM_NTASKS,
    SLURM_PARTITION,
    SLURM_RESTART_COUNT,
    SLURM_SEGMENT,
    SLURM_SLUID,
    SLURM_TOPOLOGY_ADDR,
    SLURM_TOPOLOGY_ADDR_PATTERN,
    SLURM_TORCHELASTIC_RESTART_COUNT,
)
from .encoding import AttributeConverter, AttributeValue, convert_attribute

_LOG = logging.getLogger(__name__)

SLURM_RETIRED_RESOURCE_ATTRIBUTE_KEYS = frozenset(
    {
        SLURM_CLUSTER,
        SLURM_NODELIST,
        SLURM_TORCHELASTIC_RESTART_COUNT,
    }
)


def _resource_integer(value: AttributeValue) -> int:
    """Accept integers or complete ASCII base-10 strings without truncation."""
    if type(value) is int:
        return value
    if isinstance(value, str) and re.fullmatch(r"[+-]?[0-9]+", value.strip()):
        return int(value.strip(), 10)
    raise ValueError("expected a base-10 integer")


SLURM_RESOURCE_TYPES: dict[str, AttributeConverter] = {
    SLURM_JOB_ID: str,
    SLURM_JOB_ID_RAW: str,
    SLURM_ARRAY_JOB_ID: str,
    SLURM_ARRAY_TASK_ID: str,
    SLURM_ARRAY_COUNT: _resource_integer,
    SLURM_SLUID: str,
    SLURM_ARRAY_SLUID: str,
    SLURM_JOB_NAME: str,
    SLURM_CLUSTER_NAME: str,
    SLURM_PARTITION: str,
    SLURM_HEAD_NODE_NAME: str,
    SLURM_NNODES: _resource_integer,
    SLURM_NTASKS: _resource_integer,
    SLURM_RESTART_COUNT: _resource_integer,
    SLURM_JOB_USER: str,
    SLURM_JOB_ACCOUNT: str,
    SLURM_JOB_QOS: str,
    SLURM_JOB_RESERVATION: str,
    SLURM_SEGMENT: str,
    SLURM_TOPOLOGY_ADDR: str,
    SLURM_TOPOLOGY_ADDR_PATTERN: str,
    NV_DL_JOB_UUID: str,
    NV_DL_RUN_UUID: str,
}


DL_RESOURCE_TYPES: dict[str, AttributeConverter] = {
    NV_DL_RANK: _resource_integer,
    NV_DL_WORLD_SIZE: _resource_integer,
    NV_DL_LOCAL_RANK: _resource_integer,
    NV_DL_ROLE: str,
    NV_DL_PROVIDER_NAME: str,
    NV_DL_TOPOLOGY_SIZE_TP: _resource_integer,
    NV_DL_TOPOLOGY_SIZE_PP: _resource_integer,
    NV_DL_TOPOLOGY_SIZE_DP: _resource_integer,
    NV_DL_TRAINING_CONFIG_GLOBAL_BATCH_SIZE: _resource_integer,
    NV_DL_TRAINING_CONFIG_MICRO_BATCH_SIZE: _resource_integer,
    NV_DL_TRAINING_CONFIG_SEQUENCE_LENGTH: _resource_integer,
    NV_DL_TRAINING_CONFIG_OPTIMIZER: str,
    NV_DL_TRAINING_CONFIG_RECOMPUTE_GRANULARITY: str,
    NV_DL_TRAINING_TARGET_TRAIN_ITERS: _resource_integer,
    NV_DL_TRAINING_TARGET_TRAIN_SAMPLES: _resource_integer,
    NV_DL_TRAINING_TARGET_TRAIN_TOKENS: _resource_integer,
    NV_DL_SOFTWARE_TORCH: str,
    NV_DL_SOFTWARE_CUDA: str,
    NV_DL_SOFTWARE_NCCL: str,
    NV_DL_SOFTWARE_TRANSFORMER_ENGINE: str,
}

GPU_RESOURCE_TYPES: dict[str, AttributeConverter] = {
    NV_GPU_INDEX: _resource_integer,
    NV_GPU_MEMORY_TOTAL: _resource_integer,
}

RESOURCE_TYPES: dict[str, AttributeConverter] = {
    **SLURM_RESOURCE_TYPES,
    **DL_RESOURCE_TYPES,
    **GPU_RESOURCE_TYPES,
}


def normalize_resource_attributes(
    attrs: Mapping[str, AttributeValue],
) -> dict[str, AttributeValue]:
    """Reconstruct known Resource values while preserving invalid non-Slurm input.

    Slurm applies its own selection and fallback policy in ``detect_slurm()``.
    Unknown fields retain their decoded string values.
    """
    normalized = dict(attrs)
    for key, value in attrs.items():
        if key in SLURM_RESOURCE_TYPES:
            continue
        try:
            normalized[key] = convert_attribute(key, value, RESOURCE_TYPES)
        except ValueError:
            _LOG.warning(
                "Resource attribute %s requires a base-10 integer; preserving invalid value %r.",
                key,
                value,
            )
    return normalized


# Compatibility aliases retained for imports from ``resources.slurm``.
SLURM_RESOURCE_ATTRIBUTE_NORMALIZERS = SLURM_RESOURCE_TYPES
SLURM_RESOURCE_ATTRIBUTE_KEYS = frozenset(SLURM_RESOURCE_TYPES)
