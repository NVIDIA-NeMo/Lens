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

"""Parity between ``semconv/model/`` and the public attribute constants.

This is the **weaver-independent** guard: it reads the registry YAML directly
and asserts the model and the importable constants describe the same name set,
symbol-for-symbol and value-for-value. It always runs (CI with or without
weaver installed); ``test_semconv_generated.py`` adds the stronger byte-level
regeneration check when weaver is present.

The constants are generated from the model, so in normal operation these can
only disagree if the generated file is stale — which this test turns into a red
build instead of a silent drift.
"""

from __future__ import annotations

from pathlib import Path

import yaml

import nemo.lens.semconv.attributes as attributes

_MODEL_DIR = Path(attributes.__file__).parent / "model"
_NON_ATTRIBUTE_CONSTANTS = frozenset({"SEMCONV_VERSION"})


def _attributes_constants() -> dict[str, str]:
    """``{SYMBOL: value}`` for every importable attribute-name constant.

    Resolved through the public module (shim + generated), minus the
    explicitly non-attribute names, so it reflects what consumers actually get.
    """
    consts = {
        name: getattr(attributes, name)
        for name in dir(attributes)
        if name.isupper() and isinstance(getattr(attributes, name), str)
    }
    for name in _NON_ATTRIBUTE_CONSTANTS:
        consts.pop(name, None)
    return consts


def _model_records() -> list[tuple[str, str, str]]:
    """``(id, python_constant, source_file)`` for every attribute in the model."""
    records: list[tuple[str, str, str]] = []
    for path in sorted(_MODEL_DIR.glob("*.yaml")):
        doc = yaml.safe_load(path.read_text())
        if not isinstance(doc, dict):
            continue
        for group in doc.get("groups", []):
            for attr in group.get("attributes", []):
                constant = attr.get("annotations", {}).get("constant")
                records.append((attr["id"], constant, path.name))
    return records


def test_model_dir_exists() -> None:
    assert _MODEL_DIR.is_dir(), f"missing semconv model dir: {_MODEL_DIR}"


def test_every_attribute_has_a_constant_annotation() -> None:
    missing = [f"{rid} ({src})" for rid, const, src in _model_records() if not const]
    assert not missing, (
        "attributes with no `annotations.constant` (weaver needs it to name the "
        "Python constant):\n  " + "\n  ".join(missing)
    )


def test_no_duplicate_records() -> None:
    records = _model_records()
    ids = [rid for rid, _, _ in records]
    consts = [c for _, c, _ in records]
    dup_ids = sorted({x for x in ids if ids.count(x) > 1})
    dup_consts = sorted({x for x in consts if consts.count(x) > 1})
    assert not dup_ids, f"duplicate ids across model/: {dup_ids}"
    assert not dup_consts, f"duplicate constants across model/: {dup_consts}"


def test_model_matches_attributes() -> None:
    """Model and importable constants describe the exact same name set."""
    consts = _attributes_constants()
    records = _model_records()
    model_consts = {c for _, c, _ in records}

    unknown = [f"{c} (id={rid!r}, {src})" for rid, c, src in records if c not in consts]
    mismatch = [
        f"{c}: model id={rid!r} != constant value={consts[c]!r} ({src})"
        for rid, c, src in records
        if c in consts and consts[c] != rid
    ]
    missing = sorted(set(consts) - model_consts)

    problems: list[str] = []
    if unknown:
        problems.append("model constants not importable:\n    " + "\n    ".join(unknown))
    if mismatch:
        problems.append("id / value disagreements:\n    " + "\n    ".join(mismatch))
    if missing:
        problems.append(
            "importable constants with no model record (regenerate, or the model "
            "and generated code are out of sync):\n    " + "\n    ".join(missing)
        )
    assert not problems, (
        "semconv model/ and the generated constants are out of parity:\n\n" + "\n\n".join(problems)
    )


def test_semconv_version_is_not_a_record() -> None:
    assert "SEMCONV_VERSION" not in {c for _, c, _ in _model_records()}


def test_model_files_are_valid_yaml() -> None:
    for path in sorted(_MODEL_DIR.glob("*.yaml")):
        doc = yaml.safe_load(path.read_text())
        assert isinstance(doc, dict), f"{path.name}: expected a mapping"
        if path.name == "manifest.yaml":
            assert {"name", "semconv_version", "schema_url"} <= set(doc)
        else:
            assert isinstance(doc.get("groups"), list), f"{path.name}: expected a 'groups' list"
