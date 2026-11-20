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

"""Span-name constants for the NeMo ecosystem.

Generated from ``semconv/model/`` into ``_generated/spans.py`` and re-exported
here; edit the model, never this file or the generated one. See
``semconv/model/README.md``.

Kept out of ``nemo.lens.semconv``'s star-export on purpose: a span and an attribute
with the same id would generate the same constant, so each kind keeps its own
module::

    from nemo.lens.semconv.spans import NV_DL_TRAINING_ITERATION

``ALL_SPANS`` is the full set, for consumers that check names they read against
the registry.
"""

from nemo.lens.semconv._generated.spans import *  # noqa: F403
from nemo.lens.semconv._generated.spans import ALL_SPANS  # noqa: F401
