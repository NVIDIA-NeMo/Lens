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

from nemo.lens.resources import detect_resource, encode_resource_attributes
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


class TestEncodeResourceAttributes:
    """The write side: carrying identity into a process with no call site.

    `multiprocessing.Process` has no `env` parameter, so mutating os.environ
    before start() is the only channel for a spawned child.
    """

    @staticmethod
    def _parse(value):
        """Decode through the real SDK parser, not a reimplementation of it."""
        import os

        from opentelemetry.sdk.resources import OTELResourceDetector

        os.environ["OTEL_RESOURCE_ATTRIBUTES"] = value
        return dict(OTELResourceDetector().detect().attributes)

    def test_basic_round_trip(self):
        out = encode_resource_attributes({"dl.rank": 3}, inherited="")
        assert self._parse(out) == {"dl.rank": "3"}

    def test_a_comma_in_a_value_survives(self):
        """Unencoded, this truncates the value AND invents a second key."""
        value = "exp/2026-01,seed=7"
        out = encode_resource_attributes({"nemo.run.id": value}, inherited="")
        parsed = self._parse(out)
        assert parsed == {"nemo.run.id": value}

    def test_equals_space_and_unicode_survive(self):
        attrs = {"a.b": "x=y", "c.d": "two words", "e.f": "café→"}
        out = encode_resource_attributes(attrs, inherited="")
        assert self._parse(out) == attrs

    def test_values_are_stringified(self):
        out = encode_resource_attributes({"a": 5, "b": True, "c": 1.5}, inherited="")
        assert self._parse(out) == {"a": "5", "b": "True", "c": "1.5"}

    def test_none_values_are_dropped_not_written(self):
        out = encode_resource_attributes({"a": 1, "b": None}, inherited="")
        assert self._parse(out) == {"a": "1"}

    def test_inherited_is_extended_not_replaced(self):
        out = encode_resource_attributes({"dl.rank": 3}, inherited="job=abc")
        assert self._parse(out) == {"job": "abc", "dl.rank": "3"}

    def test_new_attributes_override_inherited_ones(self):
        """Relies on the SDK resolving duplicate keys last-wins."""
        out = encode_resource_attributes({"dl.rank": 9}, inherited="dl.rank=1,job=abc")
        assert self._parse(out) == {"dl.rank": "9", "job": "abc"}

    def test_inherited_bytes_pass_through_untouched(self):
        """Appended, never re-encoded: a launcher's value is not round-tripped."""
        out = encode_resource_attributes({"dl.rank": 1}, inherited="odd=a%2Cb")
        assert out.startswith("odd=a%2Cb,")
        assert self._parse(out)["odd"] == "a,b"

    def test_empty_attributes_returns_inherited_unchanged(self):
        assert encode_resource_attributes({}, inherited="job=abc") == "job=abc"

    def test_empty_inherited_produces_no_leading_comma(self):
        """A stray comma makes the SDK log an 'invalid pair' warning."""
        assert not encode_resource_attributes({"a": 1}, inherited="").startswith(",")
        assert not encode_resource_attributes({"a": 1}, inherited=" , ").startswith(",")

    def test_a_key_with_a_separator_is_dropped_with_a_warning(self, caplog):
        """The SDK unquotes only the value half, so such a key cannot round-trip.

        Encoding it would ship the escape sequence as the attribute name.
        """
        with caplog.at_level(logging.WARNING, logger="nemo.lens.resources"):
            out = encode_resource_attributes(
                {"k=weird": "v", "k,weird": "v", "dl.rank": 1}, inherited=""
            )
        assert self._parse(out) == {"dl.rank": "1"}
        assert "cannot be carried" in caplog.text
        assert "%" not in out

    def test_repeated_calls_do_not_accumulate(self, monkeypatch):
        """The default inherited value is an import-time snapshot.

        Merging against live os.environ instead is what makes keys pile up when a
        parent spawns in a loop, or re-execs.
        """
        monkeypatch.setattr("nemo.lens.resources._INHERITED", "job=abc")
        for rank in range(3):
            out = encode_resource_attributes({"dl.rank": rank})
            assert out == f"job=abc,dl.rank={rank}"

    def test_does_not_mutate_the_environment(self, monkeypatch):
        monkeypatch.delenv("OTEL_RESOURCE_ATTRIBUTES", raising=False)
        encode_resource_attributes({"dl.rank": 3}, inherited="")
        assert "OTEL_RESOURCE_ATTRIBUTES" not in os.environ


class TestEncodeResourceAttributesFallbackParity:
    def test_signature_matches_the_real_one(self):
        import inspect

        from nemo.lens.fallbacks import encode_resource_attributes as noop

        real = inspect.signature(encode_resource_attributes)
        assert list(real.parameters) == list(inspect.signature(noop).parameters)

    def test_fallback_preserves_a_launcher_supplied_value(self):
        from nemo.lens.fallbacks import encode_resource_attributes as noop

        assert noop({"dl.rank": 3}, inherited="job=abc") == "job=abc"


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

        value = encode_resource_attributes(
            {"dl.rank": 5, "dl.world_size": 8, "nemo.run.id": "exp,2026"},
            inherited="",
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
        assert got["dl.rank"] == "5"
        assert got["dl.world_size"] == "8"
        assert got["nemo.run.id"] == "exp,2026"  # the comma survived the boundary

    def test_a_spawned_process_inherits_via_os_environ(self, monkeypatch):
        """multiprocessing.Process has no env= parameter, so this is the only way."""
        import json
        import subprocess
        import sys

        monkeypatch.setenv(
            "OTEL_RESOURCE_ATTRIBUTES",
            encode_resource_attributes({"dl.rank": 2}, inherited=""),
        )
        out = subprocess.run(
            [sys.executable, "-c", self.CHILD], capture_output=True, text=True, check=True
        )
        assert json.loads(out.stdout)["dl.rank"] == "2"
