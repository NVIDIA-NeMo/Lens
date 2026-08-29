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

from __future__ import annotations

import os
import uuid
from collections.abc import Mapping

from nemo.lens.resources.attributes import (
    ResourceAttributeValue,
    get_otel_resource_attributes,
    merge_resource_attributes,
)
from nemo.lens.semconv import (
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

SLURM_RESOURCE_ATTRIBUTE_KEYS = frozenset(
    {
        SLURM_JOB_ID,
        SLURM_JOB_ID_RAW,
        SLURM_ARRAY_JOB_ID,
        SLURM_ARRAY_TASK_ID,
        SLURM_ARRAY_COUNT,
        SLURM_SLUID,
        SLURM_ARRAY_SLUID,
        SLURM_JOB_NAME,
        SLURM_CLUSTER_NAME,
        SLURM_PARTITION,
        SLURM_HEAD_NODE_NAME,
        SLURM_NNODES,
        SLURM_NTASKS,
        SLURM_RESTART_COUNT,
        SLURM_JOB_USER,
        SLURM_JOB_ACCOUNT,
        SLURM_JOB_QOS,
        SLURM_JOB_RESERVATION,
        SLURM_SEGMENT,
        SLURM_TOPOLOGY_ADDR,
        SLURM_TOPOLOGY_ADDR_PATTERN,
    }
)

SLURM_RETIRED_RESOURCE_ATTRIBUTE_KEYS = frozenset(
    {
        SLURM_CLUSTER,
        SLURM_NODELIST,
        SLURM_TORCHELASTIC_RESTART_COUNT,
    }
)

SLURM_IDENTITY_RESOURCE_ATTRIBUTE_KEYS = frozenset(
    {
        NV_DL_JOB_UUID,
        NV_DL_RUN_UUID,
    }
)


def detect_slurm(environ: Mapping[str, str] | None = None) -> dict[str, ResourceAttributeValue]:
    """Detect SLURM environment variables and return resource attributes.

    Existing ``OTEL_RESOURCE_ATTRIBUTES`` values are authoritative. Locally
    derived values are an additive fallback and never overwrite inherited keys.
    """
    env = os.environ if environ is None else environ
    inherited = _select_slurm_resource_attributes(get_otel_resource_attributes(env))
    fallback = derive_slurm_resource_attributes(env)

    if not inherited and not fallback:
        return {}

    return merge_resource_attributes(inherited, fallback, overwrite=False)


def derive_slurm_resource_attributes(
    environ: Mapping[str, str] | None = None,
) -> dict[str, ResourceAttributeValue]:
    """Derive v0.1 SLURM resource attributes from raw SLURM environment variables."""
    env = os.environ if environ is None else environ
    raw_job_id = env.get("SLURM_JOB_ID", "")
    if not raw_job_id:
        return {}

    attrs: dict[str, ResourceAttributeValue] = {}

    is_array = "SLURM_ARRAY_TASK_ID" in env
    array_job_id = env.get("SLURM_ARRAY_JOB_ID", "")
    array_task_id = env.get("SLURM_ARRAY_TASK_ID", "0")
    job_id = f"{array_job_id}_{array_task_id}" if is_array and array_job_id else raw_job_id

    _set(attrs, SLURM_JOB_ID, job_id)
    _set(attrs, SLURM_JOB_ID_RAW, raw_job_id)
    _set(attrs, SLURM_ARRAY_JOB_ID, array_job_id or raw_job_id)
    _set(attrs, SLURM_ARRAY_TASK_ID, array_task_id)
    _set_int(attrs, SLURM_ARRAY_COUNT, env.get("SLURM_ARRAY_TASK_COUNT"), default=1)

    _set(attrs, SLURM_SLUID, _first_env(env, "LENS_SLURM_SLUID", "NV_LAUNCH_SLUID"))
    _set(
        attrs,
        SLURM_ARRAY_SLUID,
        _first_env(env, "LENS_SLURM_ARRAY_SLUID", "NV_LAUNCH_ARRAY_SLUID"),
    )
    _set(attrs, SLURM_JOB_NAME, env.get("SLURM_JOB_NAME"))
    _set(attrs, SLURM_CLUSTER_NAME, env.get("SLURM_CLUSTER_NAME"))
    _set(attrs, SLURM_PARTITION, _first_env(env, "SLURM_JOB_PARTITION", "SLURM_PARTITION"))
    _set_int(attrs, SLURM_NNODES, _first_env(env, "SLURM_JOB_NUM_NODES", "SLURM_NNODES"))
    _set_int(attrs, SLURM_NTASKS, env.get("SLURM_NTASKS"))
    _set_int(attrs, SLURM_RESTART_COUNT, env.get("SLURM_RESTART_COUNT"), default=0)
    _set(attrs, SLURM_JOB_USER, env.get("SLURM_JOB_USER"))
    _set(attrs, SLURM_JOB_ACCOUNT, env.get("SLURM_JOB_ACCOUNT"))
    _set(attrs, SLURM_JOB_QOS, env.get("SLURM_JOB_QOS"))
    _set(attrs, SLURM_JOB_RESERVATION, env.get("SLURM_JOB_RESERVATION"))
    _set(attrs, SLURM_SEGMENT, env.get("SLURM_SEGMENT"))
    _set(attrs, SLURM_TOPOLOGY_ADDR, env.get("SLURM_TOPOLOGY_ADDR"))
    _set(attrs, SLURM_TOPOLOGY_ADDR_PATTERN, env.get("SLURM_TOPOLOGY_ADDR_PATTERN"))

    if not _in_slurm_step(env):
        _set(attrs, SLURM_HEAD_NODE_NAME, env.get("SLURMD_NODENAME"))

    _set(attrs, NV_DL_JOB_UUID, derive_nv_dl_job_uuid(env))
    return attrs


def derive_nv_dl_job_uuid(environ: Mapping[str, str] | None = None) -> str:
    """Derive the stable submitted-job UUID from raw SLURM environment variables."""
    env = os.environ if environ is None else environ
    cluster = env.get("SLURM_CLUSTER_NAME") or "nocluster"
    job_key = _slurm_identity_job_key(env)
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"nemo.lens.job/{cluster}/{job_key}"))


def derive_nv_dl_run_uuid(environ: Mapping[str, str] | None = None) -> str:
    """Derive the run-attempt UUID from raw SLURM and torchelastic environment variables."""
    env = os.environ if environ is None else environ
    cluster = env.get("SLURM_CLUSTER_NAME") or "nocluster"
    run_parts = [cluster, _slurm_identity_job_key(env)]

    if "SLURM_ARRAY_TASK_ID" not in env:
        run_parts.append("sr" + (env.get("SLURM_RESTART_COUNT") or "0"))
    if "TORCHELASTIC_RESTART_COUNT" in env:
        run_parts.append("te" + env["TORCHELASTIC_RESTART_COUNT"])

    return str(uuid.uuid5(uuid.NAMESPACE_URL, "nemo.lens.run/" + "/".join(run_parts)))


def _select_slurm_resource_attributes(attrs: Mapping[str, str]) -> dict[str, str]:
    allowed = SLURM_RESOURCE_ATTRIBUTE_KEYS | SLURM_IDENTITY_RESOURCE_ATTRIBUTE_KEYS
    retired = SLURM_RETIRED_RESOURCE_ATTRIBUTE_KEYS
    return {key: value for key, value in attrs.items() if key in allowed and key not in retired}


def _slurm_identity_job_key(env: Mapping[str, str]) -> str:
    if "SLURM_ARRAY_TASK_ID" in env:
        return env.get("SLURM_ARRAY_JOB_ID") or "nojob"
    return env.get("SLURM_JOB_ID") or "nojob"


def _first_env(env: Mapping[str, str], *keys: str) -> str | None:
    for key in keys:
        value = env.get(key)
        if value:
            return value
    return None


def _set(
    attrs: dict[str, ResourceAttributeValue],
    key: str,
    value: ResourceAttributeValue | None,
) -> None:
    if value is not None and value != "":
        attrs[key] = value


def _set_int(
    attrs: dict[str, ResourceAttributeValue],
    key: str,
    value: str | None,
    *,
    default: int | None = None,
) -> None:
    if value is None or value == "":
        if default is not None:
            attrs[key] = default
        return
    try:
        attrs[key] = int(value)
    except ValueError:
        return


def _in_slurm_step(env: Mapping[str, str]) -> bool:
    return bool(env.get("SLURM_STEP_ID") or env.get("SLURM_STEPID"))
