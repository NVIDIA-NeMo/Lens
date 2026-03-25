# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for resource detection."""

from nemo.lens.resources import detect_resource
from nemo.lens.resources.kubernetes import detect_kubernetes
from nemo.lens.resources.local import detect_local
from nemo.lens.resources.slurm import detect_slurm


class TestDetectSlurm:
    def test_no_slurm_returns_empty(self, monkeypatch):
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        assert detect_slurm() == {}

    def test_detects_slurm_job(self, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "12345")
        monkeypatch.setenv("SLURM_JOB_NAME", "train-gpt")
        monkeypatch.setenv("SLURM_NNODES", "4")
        result = detect_slurm()
        assert result["slurm.job.id"] == "12345"
        assert result["slurm.job.name"] == "train-gpt"
        assert result["slurm.nnodes"] == "4"

    def test_partial_slurm_vars(self, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "99")
        monkeypatch.delenv("SLURM_JOB_NAME", raising=False)
        result = detect_slurm()
        assert result["slurm.job.id"] == "99"
        assert "slurm.job.name" not in result


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
