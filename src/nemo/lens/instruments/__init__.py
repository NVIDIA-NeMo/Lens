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

"""OTel metric instruments for the NeMo ecosystem.

Lens owns the ``gen_ai.*`` inference instruments (:func:`record_inference_metrics`).
Consumer-owned metric families — RL series in NeMo-RL, environment series in
NeMo-Gym — are declared by the consumer through the registry
(:class:`MetricSpec`, :func:`register_metric_group`, :func:`record_metrics`)
rather than shipped as a per-consumer module here.
"""

from nemo.lens.instruments.inference import record_inference_metrics
from nemo.lens.instruments.registry import (
    METRIC_KINDS,
    MetricSpec,
    record_metrics,
    register_metric_group,
    registered_metric_groups,
    unregister_metric_group,
)

__all__ = [
    "record_inference_metrics",
    "MetricSpec",
    "METRIC_KINDS",
    "register_metric_group",
    "unregister_metric_group",
    "registered_metric_groups",
    "record_metrics",
]
