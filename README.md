# nemo-lens

Shared OpenTelemetry instrumentation library for the NVIDIA NeMo ecosystem (Megatron-LM, NeMo-RL, NeMo-Gym).

Provides unified tracing, metrics, and log bridging across distributed training jobs. Cheap when disabled — group-gated calls (`managed_span`, `@trace_fn`) cost only a single `frozenset` lookup when their span group is off. `managed_span` then yields `None` (its body still runs); `@trace_fn` just calls the wrapped function. (`span_cm` is always-on and not gated.) Only `opentelemetry-api` (no-op) is required at import time; the full SDK loads only on exporting ranks.

## Install

```bash
pip install nemo-lens           # API only — no-op at runtime, no SDK overhead
pip install 'nemo-lens[sdk]'    # adds SDK + OTLP exporters, required on exporting ranks
```

## Quickstart

```python
from nemo.lens import NemoLensConfig, setup_telemetry, managed_span

config = NemoLensConfig.from_env()
handle = setup_telemetry(config, rank=rank, world_size=world_size)

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

By default only one rank exports (`single_rank`, last rank). Change with:

```bash
NEMO_LENS_EXPORT_STRATEGY=all_ranks            # every rank
NEMO_LENS_EXPORT_STRATEGY=sampled              # fraction via NEMO_LENS_EXPORT_SAMPLE_RATE
NEMO_LENS_EXPORT_STRATEGY=first_rank_per_node  # one rank per node (LOCAL_RANK=0)
```

Custom strategies (your own rank-selection logic) are supported via `register_export_strategy` — see [docs/user-guide/custom-strategies.md](docs/user-guide/custom-strategies.md).

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
uv venv && uv pip install -e '.[dev]'
pre-commit install
pytest
```

## Docs

Full documentation: `cd docs && make serve` (requires `pip install --group docs -e .`).
