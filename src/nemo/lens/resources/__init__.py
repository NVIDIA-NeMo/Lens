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

"""Resource attributes in both directions.

:func:`detect_resource` reads what this process can discover about itself.
:func:`encode_resource_attributes` writes attributes out for a child process to
pick up, which is the only way to reach one that has no ``setup_telemetry`` call
site of its own.
"""

import logging
import os
from collections.abc import Mapping
from urllib.parse import quote

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


#: The OTel env var carrying resource attributes into a process.
OTEL_RESOURCE_ATTRIBUTES = "OTEL_RESOURCE_ATTRIBUTES"

#: Value types that survive the round trip. The SDK stores an attribute from this
#: channel as a string, so anything whose ``str()`` is not its own value arrives
#: corrupted -- ``bytes`` as its Python repr, a list as ``"[1, 2]"``, a dict as a
#: repr the SDK would have rejected outright through ``resource_attributes=``.
_SCALARS = (str, bool, int, float)


def _as_text(value: object) -> str:
    """Render a scalar without going through a subclass's ``__str__``.

    ``str()`` is not safe here even after the ``_SCALARS`` check, because a
    subclass may render as its own repr rather than its value::

        class Backend(str, Enum):
            NCCL = "nccl"

        str(Backend.NCCL)   # "Backend.NCCL", not "nccl"

    That passes ``isinstance(value, str)`` and would ship the enum's name as the
    attribute value, which is the same silent corruption the ``_SCALARS`` check
    exists to prevent -- just one level further in. ``int`` mixin enums have the
    identical flaw (``str`` of an ``int``-mixin member is its name, though
    ``IntEnum`` overrides that). Normalising through the base type is what makes
    the check mean what it says.

    ``bool`` is handled before ``int`` because it is a subclass of it, and is
    rendered as ``"True"``/``"False"`` to match what ``resource_attributes=``
    produces for the same value.
    """
    if isinstance(value, str):
        # quote() reads the underlying character data, so a str subclass encodes
        # as its value without any conversion.
        return value
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(int(value))
    return str(float(value))


def encode_resource_attributes(
    attributes: Mapping[str, object],
    inherited: str | None = None,
) -> str:
    """Build an ``OTEL_RESOURCE_ATTRIBUTES`` value carrying *attributes* onward.

    For a process that cannot reach a ``setup_telemetry`` call site -- a spawned
    checkpoint worker, an ``exec``'d relaunch, a launcher-started agent -- this
    variable is the only channel that survives, so it is how such a process
    receives its own identity. ``multiprocessing.Process`` has no ``env``
    parameter at all, which is why the mutation form below is not merely one
    option among several::

        import os
        from nemo.lens.resources import encode_resource_attributes
        from nemo.lens.semconv import DL_RANK, DL_WORLD_SIZE

        os.environ["OTEL_RESOURCE_ATTRIBUTES"] = encode_resource_attributes(
            {DL_RANK: rank, DL_WORLD_SIZE: world_size}
        )
        ctx.Process(target=worker).start()

    For ``subprocess``, build the child's environment instead::

        env = {**os.environ, "OTEL_RESOURCE_ATTRIBUTES": encode_resource_attributes(...)}
        subprocess.Popen(cmd, env=env)

    Values are percent-encoded, which the OTel SDK decodes on the way in.
    Skipping that is silent data loss rather than an error: an unescaped comma
    truncates its value *and* invents a key, so a run id of
    ``exp/2026-01,seed=7`` arrives as ``nemo.run.id="exp/2026-01"`` plus a
    spurious ``seed="7"``.

    Keys are **not** encoded, because the SDK only unquotes the value half --
    it stores ``key.strip()`` verbatim. Encoding a key would therefore ship the
    escape sequence as the attribute name. A key containing ``,`` or ``=`` has no
    representation in this format at all; such a key is dropped with a warning
    rather than mangled. No semconv name contains either character, so
    percent-encoding is a no-op for every real key.

    Values must be ``str``, ``bool``, ``int`` or ``float``. Anything else is
    dropped with a warning: this channel is string-typed, so a ``bytes`` value
    would arrive as its Python repr and a list would arrive as ``"[1, 2]"``,
    neither of which any consumer can decode back. Dropping loudly beats
    shipping a corrupted value that then has to be traced back here.

    Args:
        attributes: Attributes to add. ``None`` values are dropped rather than
            written as ``"None"``. An empty mapping is allowed and simply
            returns *inherited*.
        inherited: Existing value to extend. Defaults to the live
            ``OTEL_RESOURCE_ATTRIBUTES``. Any key in *attributes* replaces an
            earlier occurrence of that key in the inherited value, so calling
            this repeatedly -- a parent spawning in a loop, a relaunch that
            re-execs itself -- does not pile up copies. Every other key is
            carried through byte-for-byte. Pass ``""`` to start clean.

    Returns:
        A value for ``OTEL_RESOURCE_ATTRIBUTES``. Does not mutate the
        environment -- the caller decides where it goes.
    """
    log = logging.getLogger(__name__)
    # Read live, not from an import-time snapshot. A snapshot cannot see anything
    # written after this module was imported -- which is the normal order, since a
    # trainer imports lens at module load and only learns its rank later. Under
    # fork the child inherits that stale module and silently drops whatever the
    # parent had added; under spawn the re-import hides the bug. It also put this
    # function out of step with its own no-op in fallbacks.py, which reads live.
    base = os.environ.get(OTEL_RESOURCE_ATTRIBUTES, "") if inherited is None else inherited

    encoded: list[str] = []
    superseded: set[str] = set()
    for key, value in attributes.items():
        if value is None:
            continue
        if not isinstance(key, str):
            log.warning(
                "Resource attribute key %r is not a string and was dropped; "
                "str() would ship %r as the literal attribute name.",
                key,
                str(key),
            )
            continue
        name = key.strip()
        if not name or "," in name or "=" in name:
            # Unrepresentable: the SDK splits on these before it decodes anything,
            # so there is no escaping that survives. Dropping it loudly beats
            # shipping a mangled attribute name the consumer then hunts for.
            log.warning(
                "Resource attribute key %r cannot be carried in %s and was dropped; "
                "',' and '=' have no encoding in this format.",
                key,
                OTEL_RESOURCE_ATTRIBUTES,
            )
            continue
        if not isinstance(value, _SCALARS):
            log.warning(
                "Resource attribute %r has value type %s, which cannot survive %s "
                "and was dropped; only str, bool, int and float round-trip.",
                name,
                type(value).__name__,
                OTEL_RESOURCE_ATTRIBUTES,
            )
            continue
        superseded.add(name)
        encoded.append(f"{name}={quote(_as_text(value), safe='')}")

    # Inherited segments pass through byte-for-byte -- re-encoding them would
    # round-trip a launcher's bytes through our parser and mangle anything it
    # escaped differently. Only two are dropped: empties (a stray comma or a bare
    # space makes the SDK log "invalid key value resource attribute pair"), and
    # keys this call replaces, which would otherwise accumulate one dead copy per
    # call for a caller that feeds its own output back through os.environ.
    kept = []
    for segment in base.split(","):
        if not segment.strip():
            continue
        if segment.split("=", 1)[0].strip() in superseded:
            continue
        kept.append(segment)

    return ",".join(kept + encoded)


__all__ = [
    "detect_resource",
    "detect_slurm",
    "detect_kubernetes",
    "detect_local",
    "encode_resource_attributes",
]
