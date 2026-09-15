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

"""Attribute composition and explicit type conversion."""

from __future__ import annotations

from collections.abc import Callable, Mapping

AttributeValue = str | bool | int | float
AttributeConverter = Callable[[AttributeValue], AttributeValue]
AttributeConverters = Mapping[str, AttributeConverter]


def compose_attributes(
    current: Mapping[str, AttributeValue | None],
    *,
    defaults: Mapping[str, AttributeValue | None] | None = None,
    overrides: Mapping[str, AttributeValue | None] | None = None,
) -> dict[str, AttributeValue | None]:
    """Compose attribute layers with ordinary dictionary precedence.

    Defaults apply only to keys absent from *current* and *overrides*. Later
    layers win even when their value is ``None`` or an empty string.
    """
    resolved = dict(defaults or {})
    resolved.update(current)
    resolved.update(overrides or {})
    return resolved


def convert_attribute(
    key: str,
    value: AttributeValue,
    converters: AttributeConverters,
) -> AttributeValue:
    """Apply the converter registered for *key*, or return an unknown value unchanged.

    A supplied converter can raise ``ValueError`` when a supported input cannot
    be converted. This function does not infer types or handle that failure.
    """
    converter = converters.get(key)
    return value if converter is None else converter(value)
