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

import pytest

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
    before start() is the only channel for such a child -- see
    TestEndToEndAcrossAProcessBoundary for the test that actually crosses one.
    """

    @pytest.fixture(autouse=True)
    def _isolate_env(self, monkeypatch):
        """Give every test a clean OTEL_RESOURCE_ATTRIBUTES, and take it away after.

        `_parse` sets the variable and the default `inherited` reads it, so both
        directions matter. The monkeypatch handle is stashed so `_parse` can go
        through it: a raw `os.environ` write is invisible to monkeypatch, which
        then has nothing to undo, and the value outlives the test -- and the
        *next* test's `delenv` records that leaked value as the "original" and
        faithfully restores it at its own teardown. The result is a value that
        survives to the end of the session and escapes into other test files.
        """
        monkeypatch.delenv("OTEL_RESOURCE_ATTRIBUTES", raising=False)
        self._monkeypatch = monkeypatch

    def _parse(self, value):
        """Decode through the real SDK parser, not a reimplementation of it."""
        from opentelemetry.sdk.resources import OTELResourceDetector

        self._monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", value)
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

    def test_a_scalar_subclass_encodes_as_its_value_not_its_repr(self):
        """`isinstance` is not enough — the conversion has to be too.

        A `str`/`int` mixin enum passes the scalar check, but `str()` on one
        renders its *name*: `str(Backend.NCCL)` is "Backend.NCCL", not "nccl".
        Shipping that is the same silent corruption the scalar check exists to
        prevent, one level further in. (`IntEnum` happens to override `__str__`;
        a plain `int` mixin does not, so both are covered here.)
        """
        from enum import Enum

        class Backend(str, Enum):
            NCCL = "nccl"

        class Size(int, Enum):
            LARGE = 5

        out = encode_resource_attributes(
            {"backend": Backend.NCCL, "size": Size.LARGE}, inherited=""
        )
        assert self._parse(out) == {"backend": "nccl", "size": "5"}

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

    @pytest.mark.parametrize("inherited", ["", " , ", "  , , ", ",,", "   "])
    def test_a_degenerate_inherited_value_yields_no_empty_pair(self, inherited):
        """Asserted on the decode, not on a string prefix.

        `.strip().strip(",")` left whitespace exposed underneath a comma, so
        "  , , " produced " ,a=1" -- a leading *space*, which a startswith(",")
        assertion cannot see and which makes the SDK log "invalid key value
        resource attribute pair".
        """
        out = encode_resource_attributes({"a": 1}, inherited=inherited)
        # Asserted on the segments, not the decode: the SDK *discards* a malformed
        # pair (with a warning) rather than surfacing it, so the decoded mapping
        # looks correct either way and cannot witness the bug.
        assert all("=" in seg and seg.strip() for seg in out.split(",")), repr(out)
        assert self._parse(out) == {"a": "1"}

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
        """A caller feeding its own output back must not pile up dead copies.

        Asserted against the real environment rather than by patching a module
        global: the previous version patched the import-time snapshot, so it
        silently became a no-op the moment the default started reading live
        os.environ, and could not see the accumulation it was named for.
        """
        monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", "job=abc")
        for rank in range(5):
            out = encode_resource_attributes({"dl.rank": rank})
            monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", out)
            assert out == f"job=abc,dl.rank={rank}"

    def test_the_default_sees_a_value_written_after_import(self, monkeypatch):
        """The default reads live os.environ, not a snapshot taken at import.

        A trainer imports lens at module load and only learns its rank later, so
        a snapshot is stale for the normal ordering. Under fork the child
        inherits that stale module and silently drops what the parent added;
        under spawn the re-import hides it.
        """
        monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", "launcher.id=abc,dl.rank=3")
        out = encode_resource_attributes({"dl.local_rank": 7})
        assert self._parse(out) == {
            "launcher.id": "abc",
            "dl.rank": "3",
            "dl.local_rank": "7",
        }

    def test_an_overridden_key_is_replaced_not_duplicated(self, monkeypatch):
        monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", "job=abc,dl.rank=1")
        out = encode_resource_attributes({"dl.rank": 9})
        assert out.count("dl.rank") == 1
        assert self._parse(out) == {"job": "abc", "dl.rank": "9"}

    def test_untouched_inherited_keys_pass_through_byte_for_byte(self):
        """Foreign segments are never re-encoded, only filtered."""
        out = encode_resource_attributes({"b": 2}, inherited="weird=%2Fpre%2Dencoded")
        assert out.startswith("weird=%2Fpre%2Dencoded,")

    def test_non_scalar_values_are_dropped_with_a_warning(self, caplog):
        """This channel is string-typed; a repr is not a round trip.

        bytes would arrive as its Python repr and a list as "[1, 2]", neither of
        which a consumer can decode. A dict is not a valid OTel attribute value
        at all -- resource_attributes= rejects it outright.
        """
        with caplog.at_level(logging.WARNING, logger="nemo.lens.resources"):
            out = encode_resource_attributes(
                {"ok": 1, "b": b"ab", "l": [1, 2], "d": {"k": 1}}, inherited=""
            )
        assert self._parse(out) == {"ok": "1"}
        for kind in ("bytes", "list", "dict"):
            assert kind in caplog.text

    def test_a_non_string_key_is_dropped_with_a_warning(self, caplog):
        """str(key) would ship "None" as a literal attribute name."""
        with caplog.at_level(logging.WARNING, logger="nemo.lens.resources"):
            out = encode_resource_attributes({None: 1, "good": 3}, inherited="")
        assert self._parse(out) == {"good": "3"}
        assert "is not a string" in caplog.text

    def test_does_not_mutate_the_environment(self, monkeypatch):
        monkeypatch.delenv("OTEL_RESOURCE_ATTRIBUTES", raising=False)
        encode_resource_attributes({"dl.rank": 3}, inherited="")
        assert "OTEL_RESOURCE_ATTRIBUTES" not in os.environ


class TestEncodeResourceAttributesFallbackParity:
    def test_signature_matches_the_real_one(self):
        import inspect

        from nemo.lens.fallbacks import encode_resource_attributes as noop

        real = inspect.signature(encode_resource_attributes)
        noop_sig = inspect.signature(noop)
        assert list(real.parameters) == list(noop_sig.parameters)
        # Defaults too, not just names. Comparing names alone let the two read
        # different sources on the `inherited=None` path -- the no-op live, the
        # real one an import-time snapshot -- while this test stayed green.
        assert [p.default for p in real.parameters.values()] == [
            p.default for p in noop_sig.parameters.values()
        ]

    def test_both_read_live_environ_when_inherited_is_omitted(self, monkeypatch):
        """The default path the explicit-argument test below cannot reach.

        Passing `inherited=` explicitly bypasses exactly the branch where the two
        implementations previously disagreed.
        """
        from nemo.lens.fallbacks import encode_resource_attributes as noop

        monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", "launcher.id=abc")
        assert noop({"dl.rank": 3}) == "launcher.id=abc"
        assert encode_resource_attributes({"dl.rank": 3}).startswith("launcher.id=abc,")

    def test_fallback_preserves_a_launcher_supplied_value(self):
        from nemo.lens.fallbacks import encode_resource_attributes as noop

        assert noop({"dl.rank": 3}, inherited="job=abc") == "job=abc"


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
        monkeypatch.setenv(
            "OTEL_RESOURCE_ATTRIBUTES",
            encode_resource_attributes({"dl.rank": 2, "dl.world_size": 8}),
        )

        ctx = mp.get_context(method)
        queue = ctx.Queue()
        proc = ctx.Process(target=_child_reports_resource, args=(queue,))
        proc.start()
        proc.join(60)
        assert proc.exitcode == 0

        got = queue.get(timeout=10)
        assert got["launcher.id"] == "abc"  # the inherited value survived
        assert got["dl.rank"] == "2"
        assert got["dl.world_size"] == "8"


class TestTheseTestsLeakNothing:
    """Declared last on purpose: the leak is only visible *after* the class above.

    `_parse` used to write `os.environ` directly. monkeypatch had nothing to undo,
    so the value outlived its test -- and the next test's `delenv` then recorded
    that leaked value as the "original" and faithfully restored it at its own
    teardown, carrying it to the end of the session and into other test files.
    Every test in that class still passed throughout, because its fixture clears
    the variable on the way *in*. Only a check that runs afterwards can see it.
    """

    def test_no_resource_attributes_survive_the_encode_tests(self):
        assert "OTEL_RESOURCE_ATTRIBUTES" not in os.environ
