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

"""SLURM environment detection."""

import os


def detect_slurm() -> dict:
    """Detect SLURM environment variables and return resource attributes.

    Returns an empty dict if not running under SLURM.
    """
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        return {}

    attrs = {"slurm.job.id": job_id}

    _mappings = {
        "SLURM_JOB_NAME": "slurm.job.name",
        "SLURM_NODELIST": "slurm.nodelist",
        "SLURM_NNODES": "slurm.nnodes",
        "SLURM_NTASKS": "slurm.ntasks",
        "SLURM_PARTITION": "slurm.partition",
        "SLURM_CLUSTER_NAME": "slurm.cluster.name",
    }
    for env_var, attr_name in _mappings.items():
        val = os.environ.get(env_var)
        if val:
            attrs[attr_name] = val

    return attrs
