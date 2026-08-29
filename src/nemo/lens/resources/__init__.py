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

"""Resource detection: auto-detect SLURM, K8s, and local environment attributes."""

from nemo.lens.resources.kubernetes import detect_kubernetes
from nemo.lens.resources.local import detect_local
from nemo.lens.resources.slurm import detect_slurm


def detect_resource() -> dict:
    """Detect deployment environment and return resource attributes.

    Checks SLURM, Kubernetes, and local environment in order.
    All detected attributes are merged.
    """
    attrs = {}
    attrs.update(detect_local())
    attrs.update(detect_slurm())
    attrs.update(detect_kubernetes())
    return attrs


__all__ = [
    "detect_resource",
    "detect_slurm",
    "detect_kubernetes",
    "detect_local",
]
