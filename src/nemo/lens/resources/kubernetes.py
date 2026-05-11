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

"""Kubernetes environment detection via downward API."""

import os


def detect_kubernetes() -> dict:
    """Detect Kubernetes environment and return resource attributes.

    Uses standard K8s downward API env vars. Returns empty dict if not
    running in K8s.
    """
    # Check for common K8s indicators
    if not (
        os.environ.get("KUBERNETES_SERVICE_HOST")
        or os.path.exists("/var/run/secrets/kubernetes.io")
    ):
        return {}

    attrs = {}
    _mappings = {
        "K8S_NAMESPACE": "k8s.namespace.name",
        "K8S_POD_NAME": "k8s.pod.name",
        "K8S_POD_UID": "k8s.pod.uid",
        "K8S_NODE_NAME": "k8s.node.name",
        "K8S_CONTAINER_NAME": "k8s.container.name",
        "K8S_JOB_NAME": "k8s.job.name",
        "HOSTNAME": "k8s.pod.name",  # fallback
    }
    for env_var, attr_name in _mappings.items():
        val = os.environ.get(env_var)
        if val and attr_name not in attrs:
            attrs[attr_name] = val

    return attrs
