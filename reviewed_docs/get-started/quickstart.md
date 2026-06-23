# Quickstart

Instrument a minimal script in three steps.

## 1. Set environment variables

```bash
export NEMO_LENS_ENABLED=1
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317   # where your collector listens
export NEMO_LENS_SPAN_GROUPS=per_step                        # adds per-step boundaries (step/forward_backward/optimizer)
```

## 2. Initialize telemetry

```python
from nemo.lens import NemoLensConfig, setup_telemetry

config = NemoLensConfig.from_env()
handle = setup_telemetry(config, rank=0, world_size=1)

# handle.tracer — OTel Tracer (real on exporting rank, no-op elsewhere)
# handle.meter  — OTel Meter
# handle.is_exporting — whether this rank exports
```

Call this **once per process**, typically at startup.

## 3. Add instrumentation

Three primitives cover most cases:

```python
from nemo.lens import managed_span, trace_fn, span_cm

# Group-gated context manager — cheap when disabled (gated by a frozenset lookup)
with managed_span('step', 'train.step', iteration=42) as span:
    do_training_step()
    if span is not None:
        span.set_attribute('loss', compute_loss())

# Group-gated decorator — no re-indentation
@trace_fn('forward_backward', 'train.forward_backward')
def forward_pass(batch):
    ...

# Simple ungated context manager — always creates a span
with span_cm('demo.evaluate', tracer=handle.tracer):
    ...
```

## 4. Shut down cleanly

```python
try:
    ...  # your training loop
finally:
    handle.shutdown()
```

`handle.shutdown()` flushes pending spans and metrics, then shuts down the providers. Don't call `force_flush()` on the global providers manually — the handle encapsulates this correctly.

## Complete example

```python
import time
from nemo.lens import NemoLensConfig, setup_telemetry, managed_span

def main():
    config = NemoLensConfig.from_env()
    handle = setup_telemetry(config, rank=0, world_size=1)

    try:
        with managed_span('job', 'demo.job'):
            for i in range(5):
                with managed_span('step', 'demo.step', iteration=i):
                    time.sleep(0.1)
    finally:
        handle.shutdown()

if __name__ == "__main__":
    main()
```

Run with `NEMO_LENS_ENABLED=1` to export; without it, the script is a no-op at the OTel level.

## Next steps

- [Configuration](../user-guide/configuration.md) — full env var reference and `NemoLensConfig` options
- [Instrumentation](../user-guide/instrumentation.md) — when to use each primitive
- [Span Groups](../user-guide/span-groups.md) — how to control granularity
- [Observability Stack](../observability/stack.md) — run Jaeger + Prometheus + Grafana locally
