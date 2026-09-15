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

from collections.abc import Callable

from .attributes import (
    NV_DL_JOB_UUID,
    NV_DL_RUN_UUID,
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

ResourceAttributeConverter = Callable[[str], str | int]

SLURM_RETIRED_RESOURCE_ATTRIBUTE_KEYS = frozenset(
    {
        SLURM_CLUSTER,
        SLURM_NODELIST,
        SLURM_TORCHELASTIC_RESTART_COUNT,
    }
)

SLURM_RESOURCE_TYPES: dict[str, ResourceAttributeConverter] = {
    SLURM_JOB_ID: str,
    SLURM_JOB_ID_RAW: str,
    SLURM_ARRAY_JOB_ID: str,
    SLURM_ARRAY_TASK_ID: str,
    SLURM_ARRAY_COUNT: int,
    SLURM_SLUID: str,
    SLURM_ARRAY_SLUID: str,
    SLURM_JOB_NAME: str,
    SLURM_CLUSTER_NAME: str,
    SLURM_PARTITION: str,
    SLURM_HEAD_NODE_NAME: str,
    SLURM_NNODES: int,
    SLURM_NTASKS: int,
    SLURM_RESTART_COUNT: int,
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

# Compatibility aliases retained for imports from ``resources.slurm``.
SLURM_RESOURCE_ATTRIBUTE_NORMALIZERS = SLURM_RESOURCE_TYPES
SLURM_RESOURCE_ATTRIBUTE_KEYS = frozenset(SLURM_RESOURCE_TYPES)
