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

"""Exercise the Git -> sdist -> wheel version contract."""

import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from email.parser import Parser
from pathlib import Path

import pytest
from packaging.specifiers import SpecifierSet


@pytest.mark.parametrize("annotated", [False, True])
def test_tagged_sdist_preserves_version_without_git(tmp_path, annotated):
    root = Path(__file__).resolve().parents[1]
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy2(root / name, checkout / name)
    package = checkout / "src/nemo/lens"
    package.mkdir(parents=True)
    shutil.copy2(root / "src/nemo/lens/package_info.py", package)
    (package / "__init__.py").write_text("from .package_info import __version__\n")
    env = {key: value for key, value in os.environ.items() if not key.startswith("SETUPTOOLS_SCM_")}
    env["NO_VCS_VERSION"] = "1"  # The old CI switch must not disable SCM versioning.

    def run(*args, cwd=None):
        return subprocess.run(
            args, cwd=cwd or checkout, env=env, check=True, capture_output=True, text=True
        )

    run("git", "init")
    run("git", "add", ".")
    run(
        "git",
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-m",
        "initial",
    )
    tag_args = ("-a", "-m", "Release 0.3.0") if annotated else ()
    run(
        "git",
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "-c",
        "tag.gpgsign=false",
        "tag",
        *tag_args,
        "v0.3.0",
    )
    # Match actions/checkout's shallow fetch of an explicit release tag.
    shallow = tmp_path / "shallow"
    shallow.mkdir()
    run("git", "init", cwd=shallow)
    run("git", "remote", "add", "origin", str(checkout), cwd=shallow)
    run("git", "fetch", "--depth=1", "origin", "+refs/tags/v0.3.0:refs/tags/v0.3.0", cwd=shallow)
    run("git", "checkout", "refs/tags/v0.3.0", cwd=shallow)
    checkout = shallow
    run(sys.executable, "-m", "build", "--no-isolation", "--sdist")
    sdist = next((checkout / "dist").glob("*.tar.gz"))
    extracted = tmp_path / "extracted"
    with tarfile.open(sdist) as archive:
        archive.extractall(extracted, filter="data")
    source = next(extracted.iterdir())
    run(sys.executable, "-m", "build", "--no-isolation", "--wheel", cwd=source)
    wheel = next((source / "dist").glob("*.whl"))
    installed = tmp_path / "installed"
    with zipfile.ZipFile(wheel) as archive:
        metadata_path = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        metadata = Parser().parsestr(archive.read(metadata_path).decode())
        assert metadata["Version"] == "0.3.0"
        assert metadata["Version"] in SpecifierSet(">=0.3.0")
        archive.extractall(installed)
    # -S excludes installed packages, including setuptools-scm and older Lens metadata.
    result = run(
        sys.executable,
        "-S",
        "-c",
        f"import sys; sys.path.insert(0, {str(installed)!r}); import nemo.lens; print(nemo.lens.__version__)",
        cwd=tmp_path,
    )
    assert result.stdout.strip() == "0.3.0"
