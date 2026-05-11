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

"""Local environment detection: hostname, PID, GPU count."""

import os
import socket


def detect_local() -> dict:
    """Detect local environment attributes."""
    attrs = {
        "host.name": socket.gethostname(),
        "process.pid": os.getpid(),
    }

    # GPU count detection (best-effort)
    gpu_count = _detect_gpu_count()
    if gpu_count is not None:
        attrs["host.gpu.count"] = gpu_count

    return attrs


def _detect_gpu_count() -> int | None:
    """Detect the number of GPUs available (best-effort)."""
    # Check CUDA_VISIBLE_DEVICES first
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible is not None:
        if cuda_visible.strip() == "":
            return 0
        return len(cuda_visible.split(","))

    # Try nvidia-smi
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            lines = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
            if lines:
                return len(lines)
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass

    return None
