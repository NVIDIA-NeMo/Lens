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

"""Import-time dependency isolation tests."""

import os
import subprocess
import sys
from pathlib import Path


def test_import_does_not_require_opentelemetry_sdk():
    """The public package must import with only the OTel API available."""
    source_root = Path(__file__).parents[1] / "src"
    script = """
import importlib.abc
import sys


class BlockSdkImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "opentelemetry.sdk" or fullname.startswith("opentelemetry.sdk."):
            raise ImportError("opentelemetry.sdk is unavailable in this process")
        return None


sys.meta_path.insert(0, BlockSdkImports())
import nemo.lens

assert not any(name == "opentelemetry.sdk" or name.startswith("opentelemetry.sdk.") for name in sys.modules)
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(source_root)
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
