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

"""Tests for semantic definitions and attribute reconstruction."""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from nemo.lens.resources.attributes import format_otel_resource_attributes
from nemo.lens.semconv import (
    NEMO_RUN_ID,
    NV_DL_LOCAL_RANK,
    NV_DL_PROVIDER_NAME,
    NV_DL_RANK,
    NV_DL_ROLE,
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
    SLURM_NNODES,
    SLURM_NTASKS,
    SLURM_RESTART_COUNT,
)
from nemo.lens.semconv.encoding import (
    compose_attributes,
    convert_attribute,
)
from nemo.lens.semconv.resources import (
    DL_RESOURCE_TYPES,
    RESOURCE_TYPES,
    normalize_resource_attributes,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

_CARRIED_INTEGERS = {
    NV_DL_RANK: 7,
    NV_DL_WORLD_SIZE: 16,
    NV_DL_LOCAL_RANK: 3,
    NV_GPU_INDEX: 3,
    NV_GPU_MEMORY_TOTAL: 85899345920,
    NV_DL_TOPOLOGY_SIZE_TP: 2,
    NV_DL_TOPOLOGY_SIZE_PP: 4,
    NV_DL_TOPOLOGY_SIZE_DP: 2,
    NV_DL_TRAINING_CONFIG_GLOBAL_BATCH_SIZE: 128,
    NV_DL_TRAINING_CONFIG_MICRO_BATCH_SIZE: 1,
    NV_DL_TRAINING_CONFIG_SEQUENCE_LENGTH: 8192,
    NV_DL_TRAINING_TARGET_TRAIN_ITERS: 1000,
    NV_DL_TRAINING_TARGET_TRAIN_SAMPLES: 128000,
    NV_DL_TRAINING_TARGET_TRAIN_TOKENS: 2**53 + 1,
}

_RESOURCE_INTEGER_FIELDS = {
    SLURM_ARRAY_COUNT,
    SLURM_NNODES,
    SLURM_NTASKS,
    SLURM_RESTART_COUNT,
    *_CARRIED_INTEGERS,
}

_CURRENT_RESOURCE_FIELDS = (
    (NV_DL_ROLE, "nv.dl.role"),
    (NV_DL_PROVIDER_NAME, "nv.dl.provider.name"),
    (NV_DL_TOPOLOGY_SIZE_TP, "nv.dl.topology.size.tp"),
    (NV_DL_TOPOLOGY_SIZE_PP, "nv.dl.topology.size.pp"),
    (NV_DL_TOPOLOGY_SIZE_DP, "nv.dl.topology.size.dp"),
    (NV_DL_TRAINING_CONFIG_GLOBAL_BATCH_SIZE, "nv.dl.training.config.global_batch_size"),
    (NV_DL_TRAINING_CONFIG_MICRO_BATCH_SIZE, "nv.dl.training.config.micro_batch_size"),
    (NV_DL_TRAINING_CONFIG_SEQUENCE_LENGTH, "nv.dl.training.config.sequence_length"),
    (NV_DL_TRAINING_CONFIG_OPTIMIZER, "nv.dl.training.config.optimizer"),
    (
        NV_DL_TRAINING_CONFIG_RECOMPUTE_GRANULARITY,
        "nv.dl.training.config.recompute_granularity",
    ),
    (NV_DL_TRAINING_TARGET_TRAIN_ITERS, "nv.dl.training.target.train_iters"),
    (NV_DL_TRAINING_TARGET_TRAIN_SAMPLES, "nv.dl.training.target.train_samples"),
    (NV_DL_TRAINING_TARGET_TRAIN_TOKENS, "nv.dl.training.target.train_tokens"),
    (NV_DL_SOFTWARE_TORCH, "nv.dl.software.torch"),
    (NV_DL_SOFTWARE_CUDA, "nv.dl.software.cuda"),
    (NV_DL_SOFTWARE_NCCL, "nv.dl.software.nccl"),
    (NV_DL_SOFTWARE_TRANSFORMER_ENGINE, "nv.dl.software.transformer_engine"),
)


def test_current_resource_field_inventory_and_types():
    assert all(actual == expected for actual, expected in _CURRENT_RESOURCE_FIELDS)
    assert all(actual in DL_RESOURCE_TYPES for actual, _ in _CURRENT_RESOURCE_FIELDS)

    strings = {
        NV_DL_ROLE: "trainer",
        NV_DL_PROVIDER_NAME: "megatron-lm",
        NV_DL_TRAINING_CONFIG_OPTIMIZER: "adam",
        NV_DL_TRAINING_CONFIG_RECOMPUTE_GRANULARITY: "full",
        NV_DL_SOFTWARE_TORCH: "2.8.0",
        NV_DL_SOFTWARE_CUDA: "12.8",
        NV_DL_SOFTWARE_NCCL: "2.27.3",
        NV_DL_SOFTWARE_TRANSFORMER_ENGINE: "2.5.0",
    }
    assert normalize_resource_attributes(strings) == strings


def test_convert_attribute_uses_only_the_explicit_mapping():
    converters = {"example.count": int}
    decoded = {"example.count": "7", "example.id": "007"}

    assert convert_attribute("example.count", decoded["example.count"], converters) == 7
    assert convert_attribute("example.id", decoded["example.id"], converters) == "007"


@pytest.mark.parametrize(
    ("current", "defaults", "overrides", "expected"),
    [
        ({}, {"example.value": 3}, {}, {"example.value": 3}),
        ({"example.value": 7}, {"example.value": 3}, {}, {"example.value": 7}),
        ({"example.value": ""}, {"example.value": 3}, {}, {"example.value": ""}),
        (
            {"example.value": 7},
            {"example.value": 3},
            {"example.value": ""},
            {"example.value": ""},
        ),
        (
            {"example.value": 7},
            {"example.value": 3},
            {"example.value": None},
            {"example.value": None},
        ),
    ],
)
def test_compose_attributes_uses_dictionary_precedence_without_mutation(
    current, defaults, overrides, expected
):
    original_layers = (dict(current), dict(defaults), dict(overrides))

    resolved = compose_attributes(current, defaults=defaults, overrides=overrides)

    assert resolved == expected
    assert (current, defaults, overrides) == original_layers
    assert resolved is not current


@pytest.mark.parametrize("key", sorted(_RESOURCE_INTEGER_FIELDS))
def test_resource_integer_fields_share_strict_conversion(key):
    assert convert_attribute(key, " +007 ", RESOURCE_TYPES) == 7
    with pytest.raises(ValueError, match="expected a base-10 integer"):
        convert_attribute(key, "1_0", RESOURCE_TYPES)


@pytest.mark.parametrize(
    "key,value,expected,warns",
    [
        (NV_DL_RANK, "7", 7, False),
        (NV_DL_WORLD_SIZE, " +007 ", 7, False),
        (NV_DL_LOCAL_RANK, -1, -1, False),
        (NV_GPU_INDEX, "0", 0, False),
        (NV_GPU_MEMORY_TOTAL, str(2**53 + 1), 2**53 + 1, False),
        (NV_DL_RANK, True, True, True),
        (NV_DL_RANK, "1_0", "1_0", True),
        (NV_DL_RANK, "3.5", "3.5", True),
        ("example.id", "007", "007", False),
    ],
)
def test_normalize_resource_attributes_uses_explicit_types(caplog, key, value, expected, warns):
    with caplog.at_level(logging.WARNING, logger="nemo.lens.semconv.resources"):
        normalized = normalize_resource_attributes({key: value})

    assert normalized[key] == expected
    assert type(normalized[key]) is type(expected)
    assert ("requires a base-10 integer" in caplog.text) is warns


def test_supported_resource_integer_types_survive_environment_boundary():
    values = {**_CARRIED_INTEGERS, NEMO_RUN_ID: "007"}
    env = os.environ.copy()
    env["OTEL_RESOURCE_ATTRIBUTES"] = format_otel_resource_attributes(values)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT / "src"), env.get("PYTHONPATH", "")])
    code = f"""
import json
from opentelemetry import trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from nemo.lens.config import NemoLensConfig
from nemo.lens.providers import build_providers

keys = {list(values)!r}
exporter = InMemorySpanExporter()
build_providers(
    NemoLensConfig(enabled=True, metrics_enabled=False),
    span_exporter=exporter,
)
provider = trace.get_tracer_provider()
with provider.get_tracer(__name__).start_as_current_span("child"):
    pass
provider.force_flush()
attrs = exporter.get_finished_spans()[0].resource.attributes
print(json.dumps({{key: attrs[key] for key in keys}}))
provider.shutdown()
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    exported = json.loads(result.stdout.strip().splitlines()[-1])
    for key, value in _CARRIED_INTEGERS.items():
        assert exported[key] == value
        assert type(exported[key]) is int
    assert exported[NEMO_RUN_ID] == "007"
