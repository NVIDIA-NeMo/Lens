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

"""Unit tests for resource detection."""

import uuid

from nemo.lens.resources import detect_resource
from nemo.lens.resources.attributes import (
    check_resource_attributes,
    duplicate_otel_resource_attribute_keys,
    extend_otel_resource_attributes,
    format_otel_resource_attributes,
    merge_resource_attributes,
    parse_otel_resource_attributes,
    set_otel_resource_attributes,
)
from nemo.lens.resources.kubernetes import detect_kubernetes
from nemo.lens.resources.local import detect_local
from nemo.lens.resources.slurm import (
    derive_nv_dl_job_uuid,
    derive_nv_dl_run_uuid,
    detect_slurm,
)


class TestDetectSlurm:
    def test_no_slurm_returns_empty(self, monkeypatch):
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.delenv("OTEL_RESOURCE_ATTRIBUTES", raising=False)
        assert detect_slurm() == {}

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

    def test_partial_slurm_vars(self, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "99")
        monkeypatch.delenv("SLURM_JOB_NAME", raising=False)
        result = detect_slurm()
        assert result["slurm.job.id"] == "99"
        assert "slurm.job.name" not in result

    def test_existing_otel_resource_attributes_win(self, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "12345")
        monkeypatch.setenv("SLURM_JOB_NUM_NODES", "4")
        monkeypatch.setenv("SLURM_CLUSTER_NAME", "cluster-a")
        monkeypatch.setenv(
            "OTEL_RESOURCE_ATTRIBUTES",
            "slurm.job.id=from-launch,slurm.nnodes=8,nv.dl.job.uuid=job-from-launch",
        )

        result = detect_slurm()

        assert result["slurm.job.id"] == "from-launch"
        assert result["slurm.nnodes"] == "8"
        assert result["nv.dl.job.uuid"] == "job-from-launch"
        assert result["slurm.job.id.raw"] == "12345"

    def test_retired_slurm_keys_are_not_returned(self, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "12345")
        monkeypatch.setenv("SLURM_NODELIST", "node-[1-4]")
        monkeypatch.setenv(
            "OTEL_RESOURCE_ATTRIBUTES",
            "slurm.nodelist=node-%5B1-4%5D,slurm.cluster=old,slurm.job.id=12345",
        )

        result = detect_slurm()

        assert "slurm.nodelist" not in result
        assert "slurm.cluster" not in result
        assert result["slurm.job.id"] == "12345"

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

    def test_derive_nv_dl_run_uuid_excludes_slurm_restart_count_for_arrays(self, monkeypatch):
        monkeypatch.setenv("SLURM_CLUSTER_NAME", "cluster-a")
        monkeypatch.setenv("SLURM_JOB_ID", "12350")
        monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "12345")
        monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "7")
        monkeypatch.setenv("SLURM_RESTART_COUNT", "99")
        monkeypatch.setenv("TORCHELASTIC_RESTART_COUNT", "3")

        expected = uuid.uuid5(uuid.NAMESPACE_URL, "nemo.lens.run/cluster-a/12345/te3")

        assert derive_nv_dl_run_uuid() == str(expected)


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

    def test_merge_resource_attributes_additive_by_default(self):
        merged = merge_resource_attributes(
            {"slurm.job.id": "from-launch"},
            {"slurm.job.id": "from-fallback", "slurm.job.id.raw": "12345"},
        )

        assert merged == {
            "slurm.job.id": "from-launch",
            "slurm.job.id.raw": "12345",
        }

    def test_extend_otel_resource_attributes(self):
        encoded = extend_otel_resource_attributes(
            "slurm.job.id=from-launch",
            {"slurm.job.id": "from-fallback", "slurm.job.id.raw": "12345"},
        )

        assert parse_otel_resource_attributes(encoded) == {
            "slurm.job.id": "from-launch",
            "slurm.job.id.raw": "12345",
        }

    def test_set_otel_resource_attributes_updates_env(self):
        env = {"OTEL_RESOURCE_ATTRIBUTES": "slurm.job.id=from-launch"}

        value = set_otel_resource_attributes(
            {"slurm.job.id": "from-fallback", "host.name": "node-01"},
            environ=env,
        )

        assert env["OTEL_RESOURCE_ATTRIBUTES"] == value
        assert parse_otel_resource_attributes(value) == {
            "slurm.job.id": "from-launch",
            "host.name": "node-01",
        }

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


class TestDetectResource:
    def test_always_returns_dict(self, monkeypatch):
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
        result = detect_resource()
        assert isinstance(result, dict)
        assert "host.name" in result
