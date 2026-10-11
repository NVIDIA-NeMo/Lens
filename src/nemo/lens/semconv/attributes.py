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

"""Semantic convention attribute name constants for the NeMo ecosystem.

The attribute-name constants are **generated** from ``semconv/model/`` and live
in ``_generated/attributes.py``; this module re-exports them and owns the one
piece of non-attribute state, ``SEMCONV_VERSION``.

To add, rename, or re-type a name you edit the model and regenerate — never this
file or the generated one. See ``semconv/model/README.md``. Follows OTel semconv
naming: ``<namespace>.<entity>.<attribute>``.
"""

# The names are the generated registry output — one place, machine-written.
from nemo.lens.semconv._generated.attributes import *  # noqa: F403

# ------------------------------------------------------------------ #
# Version tracking (non-attribute state; hand-maintained here)
# ------------------------------------------------------------------ #
# Tracks which upstream OTel semconv version these constants align with. Update
# when syncing with a new semconv release.
# Reference: https://github.com/open-telemetry/semantic-conventions
# ------------------------------------------------------------------ #

SEMCONV_VERSION = "1.29.0"
"""OTel semantic conventions version these constants are aligned with.

Standard namespaces (``gen_ai.*``, ``host.*``, ``k8s.*``) follow the upstream
spec at this version. Custom namespaces (``nv.dl.*``, ``nv.gpu.*``, ``dl.*``,
``rl.*``, ``gym.*``, ``slurm.*``, ``nemo.*``, ``wandb.*``, ``inverted.*``) are
NeMo-specific extensions that do not exist upstream.
"""
