# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""NemoLensConfig: unified OTel configuration for the NeMo ecosystem."""

import os
from dataclasses import dataclass, field
from typing import Optional


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
    service_name: str = 'nemo'

    #: Export strategy: ``"all_ranks"``, ``"sampled"``, ``"single_rank"``.
    export_strategy: str = 'single_rank'

    #: For ``single_rank``: which rank exports (-1 = last rank).
    export_rank: int = -1

    #: For ``sampled``: fraction of ranks that export (0.0–1.0).
    export_sample_rate: float = 1.0

    #: Enable trace spans.
    traces_enabled: bool = True

    #: Enable metrics instruments.
    metrics_enabled: bool = True

    #: Enable OTel log bridge.
    logs_enabled: bool = False

    #: Comma-separated span-group spec (preset or individual group names).
    span_groups: str = 'default'

    #: Exporter backend: ``"otlp"`` or ``"console"``.
    exporter: str = 'otlp'

    #: Span group class used for resolution. Set by library-specific subclasses.
    _span_group_cls: Optional[type] = field(default=None, repr=False)

    @property
    def resolved_span_groups(self) -> frozenset:
        """Resolve :attr:`span_groups` to a frozenset of group names."""
        from nemo.lens.groups import SpanGroup
        cls = self._span_group_cls or SpanGroup
        return cls.resolve(self.span_groups)

    @classmethod
    def from_env(
        cls,
        prefix: str = 'NEMO_LENS',
        fallback_prefix: Optional[str] = None,
        span_group_cls: Optional[type] = None,
    ) -> 'NemoLensConfig':
        """Build config from environment variables.

        Args:
            prefix: Primary env var prefix (e.g. ``"MEGATRON_OTEL"``).
            fallback_prefix: Fallback prefix (e.g. ``"NEMO_LENS"``).
            span_group_cls: SpanGroup subclass for resolution.
        """

        def _env(key: str, default: str = '') -> str:
            val = os.environ.get(f'{prefix}_{key}', '').strip()
            if not val and fallback_prefix:
                val = os.environ.get(f'{fallback_prefix}_{key}', '').strip()
            return val if val else default

        def _bool(key: str, default: bool) -> bool:
            val = _env(key).lower()
            if not val:
                return default
            if val in ('1', 'true', 'yes', 'on'):
                return True
            if val in ('0', 'false', 'no', 'off'):
                return False
            raise ValueError(
                f"Invalid boolean for {prefix}_{key}: {val!r}. "
                "Expected '1'/'0', 'true'/'false', 'yes'/'no', 'on'/'off'."
            )

        def _int(key: str, default: int) -> int:
            val = _env(key)
            if not val:
                return default
            try:
                return int(val)
            except ValueError:
                raise ValueError(f"Invalid integer for {prefix}_{key}: {val!r}.")

        def _float(key: str, default: float) -> float:
            val = _env(key)
            if not val:
                return default
            try:
                return float(val)
            except ValueError:
                raise ValueError(f"Invalid float for {prefix}_{key}: {val!r}.")

        service_name = os.environ.get('OTEL_SERVICE_NAME', '').strip() or 'nemo'

        return cls(
            enabled=_bool('ENABLED', False),
            service_name=service_name,
            export_strategy=_env('EXPORT_STRATEGY', 'single_rank'),
            export_rank=_int('EXPORT_RANK', -1),
            export_sample_rate=_float('EXPORT_SAMPLE_RATE', 1.0),
            traces_enabled=_bool('TRACES_ENABLED', True),
            metrics_enabled=_bool('METRICS_ENABLED', True),
            logs_enabled=_bool('LOGS_ENABLED', False),
            span_groups=_env('SPAN_GROUPS', 'default'),
            exporter=_env('EXPORTER', 'otlp'),
            _span_group_cls=span_group_cls,
        )
