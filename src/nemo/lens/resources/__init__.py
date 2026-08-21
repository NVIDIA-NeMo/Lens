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

#: Snapshot taken at import, before this process has had a chance to add anything
#: of its own. Merging against the *live* variable instead is what makes keys pile
#: up: a launcher that re-execs, or a parent that spawns in a loop after setting
#: its own value, would append to a string it had already appended to.
_INHERITED: str = os.environ.get(OTEL_RESOURCE_ATTRIBUTES, "")


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

    Args:
        attributes: Attributes to add. Values are stringified; ``None`` values
            are dropped rather than written as ``"None"``. An empty mapping is
            allowed and simply returns *inherited*.
        inherited: Existing value to extend. Defaults to a snapshot of the
            variable taken when this module was imported, so calling this
            repeatedly does not accumulate keys. Pass ``""`` to start clean, or
            an explicit string to merge against something else.

    Returns:
        A value for ``OTEL_RESOURCE_ATTRIBUTES``. Does not mutate the
        environment -- the caller decides where it goes.
    """
    base = _INHERITED if inherited is None else inherited
    # Appended, never re-encoded. The SDK resolves duplicate keys last-wins, so
    # concatenating is enough to override an inherited key -- and it lets whatever
    # a launcher wrote pass through byte-for-byte instead of being round-tripped
    # through our own parser, which would mangle anything it encodes differently.
    base = base.strip().strip(",")
    parts = [base] if base else []

    for key, value in attributes.items():
        if value is None:
            continue
        name = str(key).strip()
        if not name or "," in name or "=" in name:
            # Unrepresentable: the SDK splits on these before it decodes anything,
            # so there is no escaping that survives. Dropping it loudly beats
            # shipping a mangled attribute name the consumer then hunts for.
            logging.getLogger(__name__).warning(
                "Resource attribute key %r cannot be carried in %s and was dropped; "
                "',' and '=' have no encoding in this format.",
                key,
                OTEL_RESOURCE_ATTRIBUTES,
            )
            continue
        parts.append(f"{name}={quote(str(value), safe='')}")

    return ",".join(parts)


__all__ = [
    "detect_resource",
    "detect_slurm",
    "detect_kubernetes",
    "detect_local",
    "encode_resource_attributes",
]
