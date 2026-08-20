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

"""NemoLensConfig: unified OTel configuration for the NeMo ecosystem."""

import os
from dataclasses import dataclass, field


@dataclass
class NemoLensConfig:
    """Configuration for OpenTelemetry instrumentation across NeMo libraries.

    Library-specific settings use a prefix/fallback model: each library reads
    its own ``<PREFIX>_OTEL_*`` env vars first, falling back to ``NEMO_LENS_*``.
    Standard OTel SDK vars (``OTEL_EXPORTER_OTLP_ENDPOINT``, etc.) are handled
    automatically by the SDK.
    """

    #: Must be explicitly True to activate telemetry.
    enabled: bool = False

    #: Human-readable service name for the OTLP backend.
    service_name: str = "nemo"

    #: Enable trace spans.
    traces_enabled: bool = True

    #: Enable metrics instruments.
    metrics_enabled: bool = True

    #: Enable OTel log bridge.
    logs_enabled: bool = False

    #: Comma-separated span-group spec (preset or individual group names).
    span_groups: str = "default"

    #: Exporter backend: ``"otlp"`` or ``"console"``.
    exporter: str = "otlp"

    #: Unique run identifier. Shared across all ranks in a distributed job.
    #: Auto-generated from SLURM_JOB_ID or UUID if not set explicitly.
    run_id: str = ""

    #: Optional user/team label for filtering runs by owner.
    user: str = ""

    #: W&B entity (team/user) name. Required when exporting traces to W&B Weave.
    wandb_entity: str = ""

    #: W&B project name. Required when exporting traces to W&B Weave.
    wandb_project: str = ""

    #: Span group class used for resolution. Set by library-specific subclasses.
    _span_group_cls: type | None = field(default=None, repr=False)

    @property
    def resolved_span_groups(self) -> frozenset:
        """Resolve :attr:`span_groups` to a frozenset of group names."""
        from nemo.lens.groups import SpanGroup

        cls = self._span_group_cls or SpanGroup
        return cls.resolve(self.span_groups)

    @classmethod
    def from_env(
        cls,
        prefix: str = "NEMO_LENS",
        fallback_prefix: str | None = None,
        span_group_cls: type | None = None,
    ) -> "NemoLensConfig":
        """Build config from environment variables.

        Args:
            prefix: Primary env var prefix (e.g. ``"MEGATRON_OTEL"``).
            fallback_prefix: Fallback prefix (e.g. ``"NEMO_LENS"``).
            span_group_cls: SpanGroup subclass for resolution.
        """

        def _env(key: str, default: str = "") -> str:
            val = os.environ.get(f"{prefix}_{key}", "").strip()
            if not val and fallback_prefix:
                val = os.environ.get(f"{fallback_prefix}_{key}", "").strip()
            return val if val else default

        def _bool(key: str, default: bool) -> bool:
            val = _env(key).lower()
            if not val:
                return default
            if val in ("1", "true", "yes", "on"):
                return True
            if val in ("0", "false", "no", "off"):
                return False
            raise ValueError(
                f"Invalid boolean for {prefix}_{key}: {val!r}. "
                "Expected '1'/'0', 'true'/'false', 'yes'/'no', 'on'/'off'."
            )

        service_name = os.environ.get("OTEL_SERVICE_NAME", "").strip() or "nemo"

        return cls(
            enabled=_bool("ENABLED", False),
            service_name=service_name,
            traces_enabled=_bool("TRACES_ENABLED", True),
            metrics_enabled=_bool("METRICS_ENABLED", True),
            logs_enabled=_bool("LOGS_ENABLED", False),
            span_groups=_env("SPAN_GROUPS", "default"),
            exporter=_env("EXPORTER", "otlp"),
            run_id=_env("RUN_ID", ""),
            user=_env("USER_ID", ""),
            wandb_entity=os.environ.get("WANDB_ENTITY", ""),
            wandb_project=os.environ.get("WANDB_PROJECT", ""),
            _span_group_cls=span_group_cls,
        )
