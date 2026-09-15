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

import logging
import os
from collections.abc import Iterable, Iterator, Mapping, MutableMapping
from contextlib import contextmanager
from dataclasses import dataclass
from urllib.parse import quote, unquote

from nemo.lens.semconv.encoding import compose_attributes

_LOG = logging.getLogger(__name__)

OTEL_RESOURCE_ATTRIBUTES_ENV = "OTEL_RESOURCE_ATTRIBUTES"

ResourceAttributeValue = str | bool | int | float
ResourceAttributes = Mapping[str, ResourceAttributeValue | None]
_SCALAR_TYPES = (str, bool, int, float)


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


def parse_otel_resource_attributes(text: str | None) -> dict[str, str]:
    """Parse an ``OTEL_RESOURCE_ATTRIBUTES`` value like the OTel SDK."""
    attrs: dict[str, str] = {}
    for key, item in _otel_resource_attribute_segments(text):
        _, _, raw_value = item.partition("=")
        attrs[key] = unquote(raw_value.strip())
    return attrs


def format_otel_resource_attributes(attributes: ResourceAttributes) -> str:
    """Format representable scalar attributes for ``OTEL_RESOURCE_ATTRIBUTES``."""
    parts: dict[str, str] = {}
    for key, value in attributes.items():
        if value is None:
            continue
        if not isinstance(key, str):
            _LOG.warning("Resource attribute key %r is not a string and was dropped.", key)
            continue

        name = key.strip()
        if not name or "," in name or "=" in name:
            _LOG.warning(
                "Resource attribute key %r cannot be carried in %s and was dropped; "
                "',' and '=' have no encoding in this format.",
                key,
                OTEL_RESOURCE_ATTRIBUTES_ENV,
            )
            continue
        if not isinstance(value, _SCALAR_TYPES):
            _LOG.warning(
                "Resource attribute %r has value type %s, which cannot survive %s "
                "and was dropped; only str, bool, int and float round-trip.",
                name,
                type(value).__name__,
                OTEL_RESOURCE_ATTRIBUTES_ENV,
            )
            continue

        try:
            encoded_value = quote(_format_resource_attribute_value(value), safe="")
        except UnicodeEncodeError:
            _LOG.warning(
                "Resource attribute %r has a value that is not valid UTF-8 and was "
                "dropped; %s cannot carry unpaired surrogates.",
                name,
                OTEL_RESOURCE_ATTRIBUTES_ENV,
            )
            continue

        # Trimming can make distinct input keys equivalent. Keep only the last,
        # matching both dict assignment and the OTel detector's duplicate policy.
        parts[name] = f"{name}={encoded_value}"
    return ",".join(parts.values())


def get_otel_resource_attributes(
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Read and parse ``OTEL_RESOURCE_ATTRIBUTES`` from an environment mapping."""
    env = os.environ if environ is None else environ
    return parse_otel_resource_attributes(env.get(OTEL_RESOURCE_ATTRIBUTES_ENV))


def extend_otel_resource_attributes(
    text: str | None,
    *,
    defaults: ResourceAttributes | None = None,
    overrides: ResourceAttributes | None = None,
) -> str:
    """Compose an ``OTEL_RESOURCE_ATTRIBUTES`` string without environment I/O."""
    current = parse_otel_resource_attributes(text)
    resolved = compose_attributes(current, defaults=defaults, overrides=overrides)
    return format_otel_resource_attributes(resolved)


def set_otel_resource_attributes(
    attributes: ResourceAttributes,
    *,
    environ: MutableMapping[str, str] | None = None,
) -> str:
    """Format and replace ``OTEL_RESOURCE_ATTRIBUTES`` in an environment mapping."""
    env = os.environ if environ is None else environ
    value = format_otel_resource_attributes(attributes)
    env[OTEL_RESOURCE_ATTRIBUTES_ENV] = value
    return value


@contextmanager
def publish_otel_resource_attributes(
    attributes: ResourceAttributes,
    *,
    environ: MutableMapping[str, str] | None = None,
) -> Iterator[None]:
    """Temporarily publish an exact attribute map for child processes.

    Composition belongs to the caller: this scope installs only the encoded
    *attributes* supplied here and never retains values from the current
    environment implicitly. The exact prior environment state is restored when
    the scope exits, including on error. Because environment variables are
    process-global, publisher scopes may nest in one thread but must not overlap
    across threads; callers must serialize cross-thread publication.
    """
    env = os.environ if environ is None else environ
    had_previous = OTEL_RESOURCE_ATTRIBUTES_ENV in env
    previous = env.get(OTEL_RESOURCE_ATTRIBUTES_ENV)
    set_otel_resource_attributes(attributes, environ=env)
    try:
        yield
    finally:
        if had_previous:
            assert previous is not None
            env[OTEL_RESOURCE_ATTRIBUTES_ENV] = previous
        else:
            env.pop(OTEL_RESOURCE_ATTRIBUTES_ENV, None)


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
    seen: set[str] = set()
    duplicates: list[str] = []
    for key, _ in _otel_resource_attribute_segments(value):
        if key in seen and key not in duplicates:
            duplicates.append(key)
        seen.add(key)
    return tuple(duplicates)


def _otel_resource_attribute_segments(text: str | None) -> list[tuple[str, str]]:
    """Return valid ``(key, raw_segment)`` pairs without rewriting bytes."""
    if not text:
        return []

    segments = []
    for item in text.split(","):
        if not item.strip() or "=" not in item:
            continue
        key = item.partition("=")[0].strip()
        if key:
            segments.append((key, item))
    return segments


def _format_resource_attribute_value(value: ResourceAttributeValue) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(int(value))
    return str(float(value))


def _is_empty(value: ResourceAttributeValue | None) -> bool:
    return value is None or value == ""
