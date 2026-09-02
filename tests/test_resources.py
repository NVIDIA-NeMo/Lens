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

"""Unit tests for resource detection and attribute propagation."""

import logging
import os
import sys
import uuid

import pytest

from nemo.lens.resources import (
    detect_gpu,
    detect_resource,
    extend_otel_resource_attributes,
    publish_otel_resource_attributes,
    set_otel_resource_attributes,
)
from nemo.lens.resources.attributes import (
    check_resource_attributes,
    duplicate_otel_resource_attribute_keys,
    format_otel_resource_attributes,
    merge_resource_attributes,
    parse_otel_resource_attributes,
)
from nemo.lens.resources.kubernetes import detect_kubernetes
from nemo.lens.resources.local import detect_local
from nemo.lens.resources.slurm import (
    derive_nv_dl_job_uuid,
    derive_nv_dl_run_uuid,
    derive_slurm_resource_attributes,
    detect_slurm,
)


class TestDetectSlurm:
    def test_no_slurm_returns_empty(self, monkeypatch):
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.delenv("OTEL_RESOURCE_ATTRIBUTES", raising=False)
        assert detect_slurm() == {}

    def test_inherited_identity_activates_without_local_slurm(self):
        result = detect_slurm(
            {
                "OTEL_RESOURCE_ATTRIBUTES": (
                    "slurm.job.id=from-launch,nv.dl.job.uuid=job-from-launch,service.name=ignored"
                )
            }
        )

        assert result == {
            "slurm.job.id": "from-launch",
            "nv.dl.job.uuid": "job-from-launch",
        }

    def test_detects_slurm_job(self, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "12345")
        monkeypatch.setenv("SLURM_JOB_NAME", "train-gpt")
        monkeypatch.setenv("SLURM_JOB_NUM_NODES", "4")
        monkeypatch.setenv("SLURM_NTASKS", "16")
        monkeypatch.setenv("SLURM_CLUSTER_NAME", "test-cluster")
        monkeypatch.setenv("SLURM_JOB_PARTITION", "batch")
        monkeypatch.setenv("SLURMD_NODENAME", "head-01")
        result = detect_slurm()
        assert result["slurm.job.id"] == "12345"
        assert result["slurm.job.id.raw"] == "12345"
        assert result["slurm.array.job_id"] == "12345"
        assert result["slurm.array.task_id"] == "0"
        assert result["slurm.array.count"] == 1
        assert result["slurm.job.name"] == "train-gpt"
        assert result["slurm.nnodes"] == 4
        assert result["slurm.ntasks"] == 16
        assert result["slurm.cluster.name"] == "test-cluster"
        assert result["slurm.partition"] == "batch"
        assert result["slurm.head_node.name"] == "head-01"
        assert "slurm.nodelist" not in result
        assert "nv.dl.job.uuid" in result
        assert "nv.dl.run.uuid" not in result

    def test_derive_slurm_resource_attributes_uses_v01_fallback_keys(self):
        result = derive_slurm_resource_attributes(
            {
                "SLURM_JOB_ID": "12345",
                "SLURM_JOB_NAME": "train-gpt",
                "SLURM_JOB_NUM_NODES": "4",
                "SLURM_NTASKS": "16",
                "SLURM_CLUSTER_NAME": "test-cluster",
                "SLURM_JOB_PARTITION": "batch",
                "SLURMD_NODENAME": "head-01",
                "SLURM_RESTART_COUNT": "2",
            }
        )

        assert result["slurm.job.id"] == "12345"
        assert result["slurm.job.id.raw"] == "12345"
        assert result["slurm.array.job_id"] == "12345"
        assert result["slurm.array.task_id"] == "0"
        assert result["slurm.array.count"] == 1
        assert result["slurm.cluster.name"] == "test-cluster"
        assert result["slurm.partition"] == "batch"
        assert result["slurm.head_node.name"] == "head-01"
        assert result["slurm.nnodes"] == 4
        assert result["slurm.ntasks"] == 16
        assert result["slurm.restart_count"] == 2

    def test_partial_slurm_vars(self, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "99")
        monkeypatch.delenv("SLURM_JOB_NAME", raising=False)
        result = detect_slurm()
        assert result["slurm.job.id"] == "99"
        assert "slurm.job.name" not in result

    def test_existing_otel_resource_attributes_win(self, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "12345")
        monkeypatch.setenv("SLURM_JOB_NUM_NODES", "4")
        monkeypatch.setenv("SLURM_NTASKS", "16")
        monkeypatch.setenv("SLURM_ARRAY_TASK_COUNT", "4")
        monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
        monkeypatch.setenv("SLURM_CLUSTER_NAME", "cluster-a")
        monkeypatch.setenv(
            "OTEL_RESOURCE_ATTRIBUTES",
            "slurm.job.id=from-launch,"
            "slurm.nnodes=8,"
            "slurm.ntasks=32,"
            "slurm.array.count=64,"
            "slurm.restart_count=2,"
            "nv.dl.job.uuid=job-from-launch",
        )

        result = detect_slurm()

        assert result["slurm.job.id"] == "from-launch"
        assert result["slurm.nnodes"] == 8
        assert result["slurm.ntasks"] == 32
        assert result["slurm.array.count"] == 64
        assert result["slurm.restart_count"] == 2
        assert result["nv.dl.job.uuid"] == "job-from-launch"
        assert result["slurm.job.id.raw"] == "12345"

    def test_invalid_inherited_integer_attributes_use_derived_values(self):
        result = detect_slurm(
            {
                "SLURM_JOB_ID": "12345",
                "SLURM_JOB_NUM_NODES": "4",
                "SLURM_NTASKS": "16",
                "SLURM_ARRAY_TASK_COUNT": "8",
                "SLURM_RESTART_COUNT": "1",
                "OTEL_RESOURCE_ATTRIBUTES": (
                    "slurm.nnodes=invalid,"
                    "slurm.ntasks=invalid,"
                    "slurm.array.count=invalid,"
                    "slurm.restart_count=invalid"
                ),
            }
        )

        assert result["slurm.nnodes"] == 4
        assert result["slurm.ntasks"] == 16
        assert result["slurm.array.count"] == 8
        assert result["slurm.restart_count"] == 1

    def test_empty_inherited_attributes_use_derived_values(self):
        result = detect_slurm(
            {
                "SLURM_JOB_ID": "12345",
                "SLURM_JOB_NAME": "fallback-name",
                "SLURM_JOB_NUM_NODES": "4",
                "OTEL_RESOURCE_ATTRIBUTES": (
                    "slurm.job.name=,slurm.nnodes=,nv.dl.job.uuid=,nv.dl.run.uuid="
                ),
            }
        )

        assert result["slurm.job.name"] == "fallback-name"
        assert result["slurm.nnodes"] == 4
        assert result["nv.dl.job.uuid"] == derive_nv_dl_job_uuid({"SLURM_JOB_ID": "12345"})
        assert "nv.dl.run.uuid" not in result

    def test_detect_slurm_preserves_valid_launch_attrs_and_fills_missing(self):
        result = detect_slurm(
            {
                "SLURM_JOB_ID": "12345",
                "SLURM_JOB_NAME": "fallback-name",
                "SLURM_JOB_NUM_NODES": "4",
                "SLURM_NTASKS": "16",
                "SLURM_CLUSTER_NAME": "fallback-cluster",
                "SLURM_JOB_PARTITION": "batch",
                "SLURMD_NODENAME": "fallback-head",
                "OTEL_RESOURCE_ATTRIBUTES": (
                    "slurm.job.id=from-launch,"
                    "slurm.cluster.name=launch-cluster,"
                    "slurm.nnodes=8,"
                    "slurm.head_node.name=launch-head,"
                    "nv.dl.job.uuid=job-from-launch,"
                    "nv.dl.run.uuid=run-from-launch"
                ),
            }
        )

        assert result["slurm.job.id"] == "from-launch"
        assert result["slurm.cluster.name"] == "launch-cluster"
        assert result["slurm.nnodes"] == 8
        assert result["slurm.head_node.name"] == "launch-head"
        assert result["nv.dl.job.uuid"] == "job-from-launch"
        assert result["nv.dl.run.uuid"] == "run-from-launch"
        assert result["slurm.job.id.raw"] == "12345"
        assert result["slurm.job.name"] == "fallback-name"
        assert result["slurm.ntasks"] == 16
        assert result["slurm.partition"] == "batch"

    def test_array_ids_use_array_of_one_shape(self, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "12350")
        monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "12345")
        monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "7")
        monkeypatch.setenv("SLURM_ARRAY_TASK_COUNT", "64")

        result = detect_slurm()

        assert result["slurm.job.id"] == "12345_7"
        assert result["slurm.job.id.raw"] == "12350"
        assert result["slurm.array.job_id"] == "12345"
        assert result["slurm.array.task_id"] == "7"
        assert result["slurm.array.count"] == 64

    def test_head_node_not_derived_inside_slurm_step(self, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "12345")
        monkeypatch.setenv("SLURM_STEP_ID", "0")
        monkeypatch.setenv("SLURMD_NODENAME", "compute-01")

        result = detect_slurm()

        assert "slurm.head_node.name" not in result

    def test_head_node_kept_when_launch_layer_supplied_it(self, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "12345")
        monkeypatch.setenv("SLURM_STEP_ID", "0")
        monkeypatch.setenv("SLURMD_NODENAME", "compute-01")
        monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", "slurm.head_node.name=head-01")

        result = detect_slurm()

        assert result["slurm.head_node.name"] == "head-01"

    def test_run_uuid_kept_when_launch_layer_supplied_it(self, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "12345")
        monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", "nv.dl.run.uuid=run-from-launch")

        result = detect_slurm()

        assert result["nv.dl.run.uuid"] == "run-from-launch"

    def test_derive_nv_dl_job_uuid_uses_array_base_id(self, monkeypatch):
        monkeypatch.setenv("SLURM_CLUSTER_NAME", "cluster-a")
        monkeypatch.setenv("SLURM_JOB_ID", "12350")
        monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "12345")
        monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "7")

        expected = uuid.uuid5(uuid.NAMESPACE_URL, "nemo.lens.job/cluster-a/12345")

        assert derive_nv_dl_job_uuid() == str(expected)

    def test_derive_nv_dl_run_uuid_plain_restart_matrix(self):
        env = {"SLURM_CLUSTER_NAME": "cluster-a", "SLURM_JOB_ID": "12345"}
        restarted_env = {**env, "SLURM_RESTART_COUNT": "1"}

        default_restart = uuid.uuid5(uuid.NAMESPACE_URL, "nemo.lens.run/cluster-a/12345/sr0")
        restarted = uuid.uuid5(uuid.NAMESPACE_URL, "nemo.lens.run/cluster-a/12345/sr1")

        assert derive_nv_dl_run_uuid(env) == str(default_restart)
        assert derive_nv_dl_run_uuid({**env, "SLURM_RESTART_COUNT": "0"}) == str(default_restart)
        assert derive_nv_dl_run_uuid(restarted_env) == str(restarted)
        assert derive_nv_dl_run_uuid(restarted_env) != derive_nv_dl_run_uuid(env)

    def test_derive_nv_dl_run_uuid_omits_local_constant_without_run_id(self):
        assert derive_nv_dl_run_uuid({}) is None

    def test_derive_nv_dl_run_uuid_uses_local_run_id(self):
        first = derive_nv_dl_run_uuid({}, run_id="local-1")
        second = derive_nv_dl_run_uuid({}, run_id="local-2")
        restarted = derive_nv_dl_run_uuid(
            {"TORCHELASTIC_RESTART_COUNT": "1"},
            run_id="local-1",
        )

        expected = uuid.uuid5(uuid.NAMESPACE_URL, "nemo.lens.run/local/local-1/te0")
        restarted_expected = uuid.uuid5(
            uuid.NAMESPACE_URL,
            "nemo.lens.run/local/local-1/te1",
        )

        assert first == str(expected)
        assert second != first
        assert restarted == str(restarted_expected)

    def test_uuid_derivation_ignores_exported_array_task_id_for_plain_job(self):
        env = {
            "SLURM_CLUSTER_NAME": "cluster-a",
            "SLURM_JOB_ID": "12345",
            "SLURM_RESTART_COUNT": "4",
            "OTEL_RESOURCE_ATTRIBUTES": ("slurm.array.job_id=99999,slurm.array.task_id=0"),
        }

        job_uuid = uuid.uuid5(uuid.NAMESPACE_URL, "nemo.lens.job/cluster-a/12345")
        run_uuid = uuid.uuid5(uuid.NAMESPACE_URL, "nemo.lens.run/cluster-a/12345/sr4")

        assert derive_nv_dl_job_uuid(env) == str(job_uuid)
        assert derive_nv_dl_run_uuid(env) == str(run_uuid)

    def test_derive_nv_dl_job_uuid_is_stable_across_array_elements(self):
        first = {
            "SLURM_CLUSTER_NAME": "cluster-a",
            "SLURM_JOB_ID": "12350",
            "SLURM_ARRAY_JOB_ID": "12345",
            "SLURM_ARRAY_TASK_ID": "1",
        }
        second = {
            "SLURM_CLUSTER_NAME": "cluster-a",
            "SLURM_JOB_ID": "12351",
            "SLURM_ARRAY_JOB_ID": "12345",
            "SLURM_ARRAY_TASK_ID": "7",
        }
        expected = uuid.uuid5(uuid.NAMESPACE_URL, "nemo.lens.job/cluster-a/12345")

        assert derive_nv_dl_job_uuid(first) == str(expected)
        assert derive_nv_dl_job_uuid(second) == str(expected)

    def test_derive_nv_dl_run_uuid_excludes_slurm_restart_count_for_arrays(self):
        base = {
            "SLURM_CLUSTER_NAME": "cluster-a",
            "SLURM_JOB_ID": "12350",
            "SLURM_ARRAY_JOB_ID": "12345",
            "SLURM_ARRAY_TASK_ID": "7",
            "TORCHELASTIC_RESTART_COUNT": "3",
        }
        slurm_restarted = {**base, "SLURM_RESTART_COUNT": "99"}
        elastic_restarted = {**slurm_restarted, "TORCHELASTIC_RESTART_COUNT": "4"}

        expected = uuid.uuid5(uuid.NAMESPACE_URL, "nemo.lens.run/cluster-a/12345/te3")
        elastic_expected = uuid.uuid5(uuid.NAMESPACE_URL, "nemo.lens.run/cluster-a/12345/te4")

        assert derive_nv_dl_run_uuid(base) == str(expected)
        assert derive_nv_dl_run_uuid(slurm_restarted) == str(expected)
        assert derive_nv_dl_run_uuid(elastic_restarted) == str(elastic_expected)
        assert derive_nv_dl_run_uuid(elastic_restarted) != derive_nv_dl_run_uuid(base)


class TestResourceAttributes:
    def test_parse_and_format_round_trip(self):
        attrs = {
            "service.name": "nv.dl.launch",
            "slurm.job.name": "train,comma",
            "nv.dl.launch.container.image": "image=name:tag",
            "slurm.array.count": 4,
        }

        encoded = format_otel_resource_attributes(attrs)
        parsed = parse_otel_resource_attributes(encoded)

        assert parsed == {
            "service.name": "nv.dl.launch",
            "slurm.job.name": "train,comma",
            "nv.dl.launch.container.image": "image=name:tag",
            "slurm.array.count": "4",
        }

    def test_parser_does_not_decode_keys(self):
        assert parse_otel_resource_attributes("key%2Cpart=value%2Cpart") == {
            "key%2Cpart": "value,part"
        }

    def test_formatter_encodes_values_but_not_keys(self):
        encoded = format_otel_resource_attributes({"nemo.run.id": "exp/1,seed=7"})

        assert encoded == "nemo.run.id=exp%2F1%2Cseed%3D7"
        assert parse_otel_resource_attributes(encoded) == {"nemo.run.id": "exp/1,seed=7"}

    def test_formatter_normalizes_scalar_subclasses(self):
        from enum import Enum

        class Backend(str, Enum):
            NCCL = "nccl"

        class Size(int, Enum):
            LARGE = 5

        class Rate(float, Enum):
            LR = 0.5

        encoded = format_otel_resource_attributes(
            {"backend": Backend.NCCL, "size": Size.LARGE, "rate": Rate.LR, "active": True}
        )

        assert encoded == "backend=nccl,size=5,rate=0.5,active=True"

    def test_formatter_drops_unrepresentable_keys_and_values(self, caplog):
        with caplog.at_level(logging.WARNING, logger="nemo.lens.resources"):
            encoded = format_otel_resource_attributes(
                {
                    None: 1,
                    "bad,key": "value",
                    "bad=key": "value",
                    "bytes": b"value",
                    "list": [1, 2],
                    "good": 3,
                }
            )

        assert encoded == "good=3"
        assert "is not a string" in caplog.text
        assert "cannot be carried" in caplog.text
        assert "bytes" in caplog.text
        assert "list" in caplog.text

    def test_formatter_drops_invalid_utf8_without_raising(self, caplog):
        bad = b"run-\xff".decode("utf-8", "surrogateescape")

        with caplog.at_level(logging.WARNING, logger="nemo.lens.resources"):
            encoded = format_otel_resource_attributes({"bad": bad, "good": 1})

        assert encoded == "good=1"
        assert "not valid UTF-8" in caplog.text

    def test_formatter_emits_trimmed_duplicate_key_once(self):
        assert format_otel_resource_attributes({"a": 1, " a ": 2}) == "a=2"

    def test_merge_resource_attributes_preserves_base_by_default(self):
        merged = merge_resource_attributes(
            {"slurm.job.id": "from-launch"},
            {"slurm.job.id": "from-fallback", "slurm.job.id.raw": "12345"},
            overwrite=False,
        )

        assert merged == {
            "slurm.job.id": "from-launch",
            "slurm.job.id.raw": "12345",
        }

    def test_merge_resource_attributes_ignores_empty_values(self):
        for empty in (None, ""):
            merged = merge_resource_attributes(
                {"slurm.job.id": empty},
                {"slurm.job.id": "from-fallback", "slurm.job.name": empty},
            )

            assert merged == {"slurm.job.id": "from-fallback"}

    def test_extend_preserves_inherited_values(self):
        encoded = extend_otel_resource_attributes(
            "slurm.job.id=from-launch",
            {"slurm.job.id": "from-fallback", "slurm.job.id.raw": "12345"},
            overwrite=False,
        )

        assert parse_otel_resource_attributes(encoded) == {
            "slurm.job.id": "from-launch",
            "slurm.job.id.raw": "12345",
        }

    def test_extend_can_replace_inherited_values(self):
        encoded = extend_otel_resource_attributes(
            "slurm.job.id=stale,launcher.id=abc",
            {"slurm.job.id": "current"},
            overwrite=True,
        )

        assert encoded == "launcher.id=abc,slurm.job.id=current"

    def test_extend_overwrite_removes_inherited_duplicates(self):
        encoded = extend_otel_resource_attributes(
            "nv.dl.rank=1,launcher.id=abc,nv.dl.rank=2",
            {"nv.dl.rank": 3},
            overwrite=True,
        )

        assert encoded == "launcher.id=abc,nv.dl.rank=3"

    def test_extend_rejects_non_mapping_additions(self):
        with pytest.raises(
            TypeError,
            match=r"additions must be a mapping of name -> value, got list\.",
        ):
            extend_otel_resource_attributes("launcher.id=abc", [("nv.dl.rank", 2)])

    def test_extend_treats_empty_additions_as_absent_when_overwriting(self):
        encoded = extend_otel_resource_attributes(
            "run.id=current,launcher.id=abc",
            {"run.id": "", "launcher.id": None},
            overwrite=True,
        )

        assert encoded == "run.id=current,launcher.id=abc"

    def test_extend_preserves_untouched_inherited_segments(self):
        encoded = extend_otel_resource_attributes(
            "odd=%2Fpre%2Dencoded, malformed ",
            {"slurm.job.id": "12345"},
            overwrite=False,
        )

        assert encoded == "odd=%2Fpre%2Dencoded,slurm.job.id=12345"

    def test_set_otel_resource_attributes_updates_env(self):
        env = {"OTEL_RESOURCE_ATTRIBUTES": "slurm.job.id=from-launch"}

        value = set_otel_resource_attributes(
            {"slurm.job.id": "from-fallback", "host.name": "node-01"},
            environ=env,
            overwrite=False,
        )

        assert env["OTEL_RESOURCE_ATTRIBUTES"] == value
        assert parse_otel_resource_attributes(value) == {
            "slurm.job.id": "from-launch",
            "host.name": "node-01",
        }

    def test_set_can_replace_inherited_values(self):
        env = {"OTEL_RESOURCE_ATTRIBUTES": "nv.dl.rank=1,launcher.id=abc"}

        value = set_otel_resource_attributes(
            {"nv.dl.rank": 2},
            environ=env,
            overwrite=True,
        )

        assert value == "launcher.id=abc,nv.dl.rank=2"
        assert env["OTEL_RESOURCE_ATTRIBUTES"] == value

    def test_set_reads_live_environment_without_accumulating(self, monkeypatch):
        monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", "launcher.id=abc")

        for rank in range(5):
            value = set_otel_resource_attributes({"nv.dl.rank": rank}, overwrite=True)
            assert value == f"launcher.id=abc,nv.dl.rank={rank}"
            assert os.environ["OTEL_RESOURCE_ATTRIBUTES"] == value

    def test_publish_temporarily_replaces_identity_and_restores_environment(self):
        env = {"OTEL_RESOURCE_ATTRIBUTES": "nv.dl.rank=1,launcher.id=abc"}

        with publish_otel_resource_attributes({"nv.dl.rank": 2}, environ=env):
            assert env["OTEL_RESOURCE_ATTRIBUTES"] == "launcher.id=abc,nv.dl.rank=2"

        assert env == {"OTEL_RESOURCE_ATTRIBUTES": "nv.dl.rank=1,launcher.id=abc"}

    def test_publish_can_preserve_inherited_identity(self):
        env = {"OTEL_RESOURCE_ATTRIBUTES": "nv.dl.rank=1,launcher.id=abc"}

        with publish_otel_resource_attributes(
            {"nv.dl.rank": 2, "nv.dl.world_size": 8},
            environ=env,
            overwrite=False,
        ):
            assert (
                env["OTEL_RESOURCE_ATTRIBUTES"] == "nv.dl.rank=1,launcher.id=abc,nv.dl.world_size=8"
            )

        assert env == {"OTEL_RESOURCE_ATTRIBUTES": "nv.dl.rank=1,launcher.id=abc"}

    def test_publish_restores_nested_scopes_in_lifo_order(self):
        env = {"OTEL_RESOURCE_ATTRIBUTES": "nv.dl.rank=0"}

        with publish_otel_resource_attributes({"nv.dl.rank": 1}, environ=env):
            assert env["OTEL_RESOURCE_ATTRIBUTES"] == "nv.dl.rank=1"
            with publish_otel_resource_attributes({"nv.dl.rank": 2}, environ=env):
                assert env["OTEL_RESOURCE_ATTRIBUTES"] == "nv.dl.rank=2"
            assert env["OTEL_RESOURCE_ATTRIBUTES"] == "nv.dl.rank=1"

        assert env == {"OTEL_RESOURCE_ATTRIBUTES": "nv.dl.rank=0"}

    def test_publish_removes_new_environment_value_after_error(self):
        env = {}

        with (
            pytest.raises(RuntimeError),
            publish_otel_resource_attributes({"nv.dl.rank": 2}, environ=env),
        ):
            assert env["OTEL_RESOURCE_ATTRIBUTES"] == "nv.dl.rank=2"
            raise RuntimeError("stop")

        assert env == {}

    def test_check_resource_attributes_reports_problems(self):
        check = check_resource_attributes(
            {"slurm.job.id": "", "slurm.nodelist": "node-[1-4]"},
            required=("slurm.job.id", "slurm.array.job_id"),
            forbidden=("slurm.nodelist",),
            env_value="slurm.job.id=1,slurm.job.id=2",
        )

        assert not check.ok
        assert check.empty == ("slurm.job.id",)
        assert check.missing == ("slurm.array.job_id",)
        assert check.forbidden == ("slurm.nodelist",)
        assert check.duplicates == ("slurm.job.id",)

    def test_duplicate_otel_resource_attribute_keys(self):
        duplicates = duplicate_otel_resource_attribute_keys("a=1,b=2,a=3,b=4")

        assert duplicates == ("a", "b")


class TestDetectKubernetes:
    def test_no_k8s_returns_empty(self, monkeypatch):
        monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
        result = detect_kubernetes()
        assert result == {}

    def test_detects_k8s(self, monkeypatch):
        monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
        monkeypatch.setenv("K8S_POD_NAME", "trainer-0")
        monkeypatch.setenv("K8S_NAMESPACE", "ml")
        result = detect_kubernetes()
        assert result["k8s.pod.name"] == "trainer-0"
        assert result["k8s.namespace.name"] == "ml"


class TestDetectLocal:
    def test_detects_hostname(self):
        result = detect_local()
        assert "host.name" in result
        assert isinstance(result["host.name"], str)

    def test_detects_pid(self):
        result = detect_local()
        assert "process.pid" in result
        assert isinstance(result["process.pid"], int)

    def test_detects_gpu_from_cuda_visible(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
        result = detect_local()
        assert result.get("host.gpu.count") == 4

    def test_empty_cuda_visible_means_zero_gpus(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
        result = detect_local()
        assert result.get("host.gpu.count") == 0


class TestDetectGpu:
    def test_detects_worker_gpu_identity(self, monkeypatch):
        class FakePciInfo:
            busId = b"00000000:81:00.0"

        class FakeMemoryInfo:
            total = 80_000_000_000

        fake_pynvml = type(
            "FakePynvml",
            (),
            {
                "nvmlInit": staticmethod(lambda: None),
                "nvmlShutdown": staticmethod(lambda: None),
                "nvmlDeviceGetHandleByIndex": staticmethod(lambda index: f"gpu-{index}"),
                "nvmlDeviceGetName": staticmethod(lambda handle: b"NVIDIA H100"),
                "nvmlDeviceGetUUID": staticmethod(lambda handle: b"GPU-123"),
                "nvmlDeviceGetSerial": staticmethod(lambda handle: b"serial-123"),
                "nvmlDeviceGetPciInfo": staticmethod(lambda handle: FakePciInfo()),
                "nvmlDeviceGetCudaComputeCapability": staticmethod(lambda handle: (9, 0)),
                "nvmlDeviceGetMemoryInfo": staticmethod(lambda handle: FakeMemoryInfo()),
                "nvmlSystemGetDriverVersion": staticmethod(lambda: b"550.54.15"),
            },
        )
        monkeypatch.setitem(sys.modules, "pynvml", fake_pynvml)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")

        result = detect_gpu(local_rank=1)

        assert result == {
            "nv.gpu.index": 5,
            "nv.gpu.model": "NVIDIA H100",
            "nv.gpu.uuid": "GPU-123",
            "nv.gpu.serial": "serial-123",
            "nv.gpu.pci_bus_id": "00000000:81:00.0",
            "nv.gpu.compute_capability": "9.0",
            "nv.gpu.memory_total": 80_000_000_000,
            "nv.gpu.driver_version": "550.54.15",
        }

    def test_non_numeric_visible_device_is_not_guessed(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-123")

        assert detect_gpu(local_rank=0) == {}


class TestDetectResource:
    def test_always_returns_dict(self, monkeypatch):
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
        result = detect_resource()
        assert isinstance(result, dict)
        assert "host.name" in result


def _child_reports_resource(queue):
    """Module-level so `spawn` can pickle and re-import it."""
    from opentelemetry.sdk.resources import OTELResourceDetector

    queue.put(dict(OTELResourceDetector().detect().attributes))


class TestEndToEndAcrossAProcessBoundary:
    """The claim is that a child picks the attributes up. Prove it with a child."""

    CHILD = (
        "import json;"
        "from opentelemetry.sdk.resources import OTELResourceDetector;"
        "print(json.dumps(dict(OTELResourceDetector().detect().attributes)))"
    )

    def test_a_subprocess_receives_the_encoded_attributes(self, tmp_path):
        import json
        import subprocess
        import sys

        value = extend_otel_resource_attributes(
            "",
            {"nv.dl.rank": 5, "nv.dl.world_size": 8, "nemo.run.id": "exp,2026"},
            overwrite=True,
        )
        out = subprocess.run(
            [sys.executable, "-c", self.CHILD],
            env={**os.environ, "OTEL_RESOURCE_ATTRIBUTES": value},
            capture_output=True,
            text=True,
            check=True,
            cwd=tmp_path,
        )
        got = json.loads(out.stdout)
        assert got["nv.dl.rank"] == "5"
        assert got["nv.dl.world_size"] == "8"
        assert got["nemo.run.id"] == "exp,2026"  # the comma survived the boundary

    @pytest.mark.parametrize("method", ["fork", "spawn"])
    def test_a_multiprocessing_child_inherits_via_os_environ(self, monkeypatch, method):
        """The case the docstring is actually about.

        `multiprocessing.Process` has no `env=` parameter, so mutating os.environ
        before start() is the only channel -- unlike subprocess, which the tests
        above use. Both start methods matter and they fail differently: `spawn`
        re-imports lens in the child, `fork` carries the parent's already-imported
        module across. An import-time snapshot of OTEL_RESOURCE_ATTRIBUTES passes
        under `spawn` and silently drops the parent's attributes under `fork`,
        which is the default on Linux.
        """
        import multiprocessing as mp

        if method not in mp.get_all_start_methods():
            pytest.skip(f"{method} is unavailable on this platform")

        monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", "launcher.id=abc")
        ctx = mp.get_context(method)
        queue = ctx.Queue()
        proc = ctx.Process(target=_child_reports_resource, args=(queue,))
        with publish_otel_resource_attributes({"nv.dl.rank": 2, "nv.dl.world_size": 8}):
            proc.start()
        # Drain before joining. A child cannot exit until its queue feeder has
        # flushed to the pipe, and the feeder cannot flush once the pipe fills,
        # so join()-then-get() deadlocks on any payload past the buffer. This
        # one is small enough to survive it today, which is exactly why the
        # ordering has to be right rather than lucky.
        got = queue.get(timeout=60)
        proc.join(60)
        assert proc.exitcode == 0
        assert got["launcher.id"] == "abc"  # the inherited value survived
        assert got["nv.dl.rank"] == "2"
        assert got["nv.dl.world_size"] == "8"


class TestTheseTestsLeakNothing:
    """Verify process-boundary tests restore process-global attribute state.

    Declared last so this check catches any preceding propagation path that
    leaves ``OTEL_RESOURCE_ATTRIBUTES`` set after its test completes.
    """

    def test_no_resource_attributes_survive_the_propagation_tests(self):
        assert "OTEL_RESOURCE_ATTRIBUTES" not in os.environ
