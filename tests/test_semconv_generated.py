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

"""The committed generated constants must be exactly what weaver produces.

Without this, ``_generated/attributes.py`` is just more hand-written constants
that happen to have been machine-written once: a model edit could land with
stale generated code and nothing would notice. This regenerates into a temp dir
and asserts byte-identity.

Weaver is a Rust binary, not a Python dependency, so it is absent from the plain
unit-test environment. There the test skips — EXCEPT when ``REQUIRE_WEAVER`` is
set, which the CI job that installs weaver sets, so a missing weaver there is a
red build rather than a silently skipped gate. (``test_semconv_model_parity.py``
is the weaver-independent guard that always runs.)
"""

from __future__ import annotations

import ast
import filecmp
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_MODEL = _ROOT / "src" / "nemo" / "lens" / "semconv" / "model"
_TEMPLATES = _ROOT / "templates"
_GENERATED = _ROOT / "src" / "nemo" / "lens" / "semconv" / "_generated" / "attributes.py"


def _weaver() -> str | None:
    found = shutil.which("weaver")
    if found:
        return found
    local = Path.home() / ".local" / "weaver" / "weaver"
    return str(local) if local.is_file() and os.access(local, os.X_OK) else None


def _require_weaver() -> str:
    exe = _weaver()
    if exe:
        return exe
    if os.environ.get("REQUIRE_WEAVER"):
        pytest.fail(
            "REQUIRE_WEAVER is set but weaver was not found — the regeneration gate "
            "did not run in the job responsible for it"
        )
    pytest.skip("weaver not installed; see semconv/model/README.md for the install command")


def test_generated_is_current() -> None:
    """Regenerate into a temp dir; the generated module must be byte-identical."""
    exe = _require_weaver()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "gen"
        proc = subprocess.run(
            [
                exe,
                "registry",
                "generate",
                "-r",
                str(_MODEL),
                "-t",
                str(_TEMPLATES),
                "python",
                str(out),
            ],
            cwd=_ROOT,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, f"weaver generate failed:\n{proc.stderr}"
        fresh = out / "attributes.py"
        assert fresh.is_file(), "weaver did not produce attributes.py"
        assert _GENERATED.is_file(), "_generated/attributes.py is missing"
        assert filecmp.cmp(fresh, _GENERATED, shallow=False), (
            "_generated/attributes.py is stale relative to model/. Regenerate and commit:\n"
            "  weaver registry generate -r src/nemo/lens/semconv/model -t templates "
            "python <out> && cp <out>/attributes.py src/nemo/lens/semconv/_generated/"
        )


def test_generated_module_is_valid_python() -> None:
    """A template can emit text that is not valid Python; parse, do not count."""
    ast.parse(_GENERATED.read_text())
