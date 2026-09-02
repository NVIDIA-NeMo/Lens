<div align="center">

# NeMo Lens

[![codecov](https://codecov.io/github/NVIDIA-NeMo/Lens/graph/badge.svg?token=16WHHLY0RA)](https://codecov.io/github/NVIDIA-NeMo/Lens)
[![CICD NeMo Lens](https://github.com/NVIDIA-NeMo/Lens/actions/workflows/cicd-main.yml/badge.svg)](https://github.com/NVIDIA-NeMo/Lens/actions/workflows/cicd-main.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![GitHub Stars](https://img.shields.io/github/stars/NVIDIA-NeMo/Lens.svg?style=social&label=Star&cacheSeconds=14400)](https://github.com/NVIDIA-NeMo/Lens/stargazers/)

</div>

> **Early development:** This library is under active development. Expect breaking changes between releases.

Shared OpenTelemetry instrumentation library for the NVIDIA NeMo ecosystem (Megatron-LM, NeMo-RL, NeMo-Gym).

Provides unified tracing, metrics, and log bridging across distributed training jobs. Cheap when disabled — group-gated calls (`managed_span`, `@trace_fn`) cost only a single `frozenset` lookup when their span group is off. `managed_span` then yields `None` (its body still runs); `@trace_fn` just calls the wrapped function. (`span_cm` is always-on and not gated.) Only `opentelemetry-api` (no-op) is required at import time; the full SDK loads only where telemetry is enabled.

## Install

```bash
pip install nemo-lens           # API only — no-op at runtime, no SDK overhead
pip install 'nemo-lens[sdk]'    # adds SDK + OTLP exporters, required wherever telemetry is enabled
```

## Quickstart

```python
from nemo.lens import NemoLensConfig, setup_telemetry, managed_span
from nemo.lens.semconv import DL_RANK, DL_WORLD_SIZE

config = NemoLensConfig.from_env()
handle = setup_telemetry(
    config,
    resource_attributes={DL_RANK: rank, DL_WORLD_SIZE: world_size},
)

try:
    for i in range(steps):
        with managed_span('step', 'train.step', iteration=i) as span:
            loss = train_step()
            if span:
                span.set_attribute('loss', loss)
finally:
    handle.shutdown()
```

Enable with environment variables:

```bash
NEMO_LENS_ENABLED=1
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
NEMO_LENS_SPAN_GROUPS=per_step   # includes the 'step' group used above (default={job,checkpoint,evaluate} omits it)
```

## Three instrumentation primitives

| Primitive | Use when |
|---|---|
| `managed_span(group, name, **attrs)` | Context manager; group-gated, yields `None` when disabled |
| `@trace_fn(group, name)` | Decorator; same gating, no re-indentation |
| `span_cm(name, tracer=...)` | Always-on context manager; use for top-level spans |

## Distributed training

Lens has no notion of a rank. Every process where `enabled` is true exports, and
the caller passes its own position as resource attributes. Deciding which ranks
report is yours to make — enabling everywhere multiplies span volume by the world
size:

```python
config = NemoLensConfig.from_env()
config.enabled = config.enabled and rank == world_size - 1   # last rank only
```

Or let every rank export and filter on the `dl.rank` resource attribute in the
collector. See [docs/user-guide/production.mdx](docs/user-guide/production.mdx).

A process with no `setup_telemetry` call site to reach — a spawned checkpoint
worker, an `exec`'d relaunch — supplies the same identity through the standard
`OTEL_RESOURCE_ATTRIBUTES=dl.rank=5,dl.world_size=8`, which lens resolves
alongside `resource_attributes`.

Omitting `dl.rank` from both logs a warning at startup — without it, telemetry
cannot be filtered or grouped by rank in the collector, and `service.instance.id`
is not rank-derived. Lens will not guess it from `RANK`/`WORLD_SIZE`; a genuinely
single-process caller can pass rank `0` to silence the warning.

## Local observability stack

```bash
docker compose -f docker-compose.otel.yml up -d
# Jaeger   → http://localhost:16686
# Grafana  → http://localhost:3000
# Kibana   → http://localhost:5601
```

## Development

```bash
git clone <repo-url> && cd lens
uv venv && uv pip install -e . --group dev
pre-commit install
pytest
```

## Docs

Full documentation: https://docs.nvidia.com/nemo/lens. For local development, see `docs/fern/README.md`.
