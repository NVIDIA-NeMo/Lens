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

"""Helpers for OpenTelemetry resource attribute maps."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass
from urllib.parse import quote, unquote

OTEL_RESOURCE_ATTRIBUTES_ENV = "OTEL_RESOURCE_ATTRIBUTES"

ResourceAttributeValue = str | bool | int | float
ResourceAttributes = Mapping[str, ResourceAttributeValue]


@dataclass(frozen=True)
class ResourceAttributeCheck:
    """Result from checking a resource attribute map."""

    missing: tuple[str, ...] = ()
    empty: tuple[str, ...] = ()
    forbidden: tuple[str, ...] = ()
    duplicates: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """Whether the attribute map passed all checks."""
        return not (self.missing or self.empty or self.forbidden or self.duplicates)


def parse_otel_resource_attributes(value: str | None) -> dict[str, str]:
    """Parse an ``OTEL_RESOURCE_ATTRIBUTES`` value into a key/value map."""
    if not value:
        return {}

    attrs: dict[str, str] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        key, sep, raw_value = item.partition("=")
        key = unquote(key.strip())
        if not sep or not key:
            continue
        attrs[key] = unquote(raw_value.strip())
    return attrs


def format_otel_resource_attributes(attrs: ResourceAttributes) -> str:
    """Format resource attributes for ``OTEL_RESOURCE_ATTRIBUTES``."""
    parts = []
    for key, value in attrs.items():
        if value is None:
            continue
        parts.append(f"{_quote(str(key))}={_quote(_format_value(value))}")
    return ",".join(parts)


def get_otel_resource_attributes(
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Read and parse ``OTEL_RESOURCE_ATTRIBUTES`` from an environment mapping."""
    env = os.environ if environ is None else environ
    return parse_otel_resource_attributes(env.get(OTEL_RESOURCE_ATTRIBUTES_ENV))


def extend_otel_resource_attributes(
    value: str | None,
    additions: ResourceAttributes,
    *,
    overwrite: bool = False,
) -> str:
    """Add attributes to an ``OTEL_RESOURCE_ATTRIBUTES`` value."""
    attrs = parse_otel_resource_attributes(value)
    return format_otel_resource_attributes(
        merge_resource_attributes(attrs, additions, overwrite=overwrite)
    )


def set_otel_resource_attributes(
    additions: ResourceAttributes,
    *,
    environ: MutableMapping[str, str] | None = None,
    overwrite: bool = False,
) -> str:
    """Add attributes to ``OTEL_RESOURCE_ATTRIBUTES`` in an environment mapping."""
    env = os.environ if environ is None else environ
    value = extend_otel_resource_attributes(
        env.get(OTEL_RESOURCE_ATTRIBUTES_ENV),
        additions,
        overwrite=overwrite,
    )
    env[OTEL_RESOURCE_ATTRIBUTES_ENV] = value
    return value


def merge_resource_attributes(
    base: ResourceAttributes,
    additions: ResourceAttributes,
    *,
    overwrite: bool = False,
) -> dict[str, ResourceAttributeValue]:
    """Merge resource attributes, optionally preserving existing keys."""
    merged: dict[str, ResourceAttributeValue] = dict(base)
    for key, value in additions.items():
        if value is None:
            continue
        if overwrite or key not in merged:
            merged[key] = value
    return merged


def check_resource_attributes(
    attrs: ResourceAttributes,
    *,
    required: Iterable[str] = (),
    forbidden: Iterable[str] = (),
    env_value: str | None = None,
) -> ResourceAttributeCheck:
    """Check required, forbidden, empty, and duplicate resource attributes."""
    missing = []
    empty = []
    for key in required:
        if key not in attrs:
            missing.append(key)
        elif _is_empty(attrs[key]):
            empty.append(key)

    forbidden_present = [key for key in forbidden if key in attrs]
    duplicates = duplicate_otel_resource_attribute_keys(env_value) if env_value else ()

    return ResourceAttributeCheck(
        missing=tuple(missing),
        empty=tuple(empty),
        forbidden=tuple(forbidden_present),
        duplicates=duplicates,
    )


def duplicate_otel_resource_attribute_keys(value: str | None) -> tuple[str, ...]:
    """Return duplicate keys in an ``OTEL_RESOURCE_ATTRIBUTES`` value."""
    if not value:
        return ()

    seen: set[str] = set()
    duplicates: list[str] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        key, sep, _ = item.partition("=")
        key = unquote(key.strip())
        if not sep or not key:
            continue
        if key in seen and key not in duplicates:
            duplicates.append(key)
        seen.add(key)
    return tuple(duplicates)


def _quote(value: str) -> str:
    return quote(value, safe="/:._-")


def _format_value(value: ResourceAttributeValue) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _is_empty(value: ResourceAttributeValue) -> bool:
    return isinstance(value, str) and value == ""
