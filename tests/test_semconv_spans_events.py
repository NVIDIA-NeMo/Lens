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

"""Parity between the span and event records in ``semconv/model/`` and their constants.

The weaver-independent guard for ``nemo.lens.semconv.spans`` and
``nemo.lens.semconv.events``, as ``test_semconv_model_parity.py`` is for attributes:
it reads the registry YAML directly, so it runs with or without weaver.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import nemo.lens.semconv.attributes as attributes
import nemo.lens.semconv.events as events
import nemo.lens.semconv.spans as spans

_MODEL_DIR = Path(attributes.__file__).parent / "model"


def _groups(kind: str) -> list[tuple[dict, str]]:
    """``(group, source_file)`` for every ``type: <kind>`` group in the model."""
    found = []
    for path in sorted(_MODEL_DIR.glob("*.yaml")):
        doc = yaml.safe_load(path.read_text())
        if not isinstance(doc, dict):
            continue
        found += [(g, path.name) for g in doc.get("groups", []) if g.get("type") == kind]
    return found


def _wire_name(group: dict) -> str:
    """The emitted name: an event carries it in ``name``; a span id is ``span.<name>``."""
    if group["type"] == "event":
        return group["name"]
    assert group["id"].startswith("span."), f"span id must start with 'span.': {group['id']}"
    return group["id"].removeprefix("span.")


def _attribute_ids() -> set[str]:
    ids = set()
    for path in sorted(_MODEL_DIR.glob("*.yaml")):
        doc = yaml.safe_load(path.read_text())
        if not isinstance(doc, dict):
            continue
        for group in doc.get("groups", []):
            ids |= {a["id"] for a in group.get("attributes", []) if "id" in a}
    return ids


@pytest.mark.parametrize(
    ("kind", "module", "all_name"),
    [("span", spans, "ALL_SPANS"), ("event", events, "ALL_EVENTS")],
)
def test_model_matches_constants(kind: str, module, all_name: str) -> None:
    """Every record has a constant holding its wire name, and nothing else is exported."""
    groups = _groups(kind)
    missing_constant = [
        f"{g['id']} ({src})" for g, src in groups if not g.get("annotations", {}).get("constant")
    ]
    assert not missing_constant, (
        f"{kind} records with no `annotations.constant`: {missing_constant}"
    )

    expected = {g["annotations"]["constant"]: _wire_name(g) for g, _ in groups}
    exported = {
        name: getattr(module, name)
        for name in dir(module)
        if name.isupper() and isinstance(getattr(module, name), str)
    }
    assert exported == expected, f"{module.__name__} is out of parity with the model; regenerate"
    assert getattr(module, all_name) == frozenset(expected.values())


@pytest.mark.parametrize("kind", ["span", "event"])
def test_refs_resolve_to_model_attributes(kind: str) -> None:
    """A span or event may only reference attributes this registry defines."""
    known = _attribute_ids()
    dangling = [
        f"{g['id']} -> {a['ref']} ({src})"
        for g, src in _groups(kind)
        for a in g.get("attributes", [])
        if "ref" in a and a["ref"] not in known
    ]
    assert not dangling, f"{kind} records referencing undefined attributes: {dangling}"


def test_no_duplicate_wire_names() -> None:
    names = [_wire_name(g) for kind in ("span", "event") for g, _ in _groups(kind)]
    dups = sorted({n for n in names if names.count(n) > 1})
    assert not dups, f"duplicate span/event names across model/: {dups}"
