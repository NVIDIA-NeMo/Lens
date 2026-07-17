# Custom Exporters

`setup_telemetry` supports injecting custom `SpanExporter` and `MetricReader` instances, which bypasses the configuration-based construction. This is the extension point for the following use cases:

- In-memory exporters for testing
- Custom exporters for proprietary backends
- Exporters that require fine-grained configuration that NeMo Lens does not expose

## Signature

```python
setup_telemetry(
    config,
    rank=0,
    world_size=1,
    resource_attributes=None,
    span_exporter=None,        # optional custom SpanExporter
    metric_reader=None,         # optional custom MetricReader
)
```

When provided, these override the config's `exporter` field for that signal. You can mix these options by passing a custom `span_exporter` and letting metrics use the config's exporter.

## Custom Span Exporter

```python
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

exporter = InMemorySpanExporter()

handle = setup_telemetry(
    config,
    rank=0,
    world_size=1,
    span_exporter=exporter,
)

# ... do work ...

spans = exporter.get_finished_spans()
```

This is how the NeMo Lens test suite captures spans for assertions.

## Custom Metric Reader

```python
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

reader = InMemoryMetricReader()

handle = setup_telemetry(
    config,
    rank=0,
    world_size=1,
    metric_reader=reader,
)

# ... record metrics ...

data = reader.get_metrics_data()
```

## Write a Custom Exporter

To route telemetry to a backend not supported by OTLP, subclass `SpanExporter`:

```python
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

class MyBackendExporter(SpanExporter):
    def __init__(self, client):
        self._client = client

    def export(self, spans) -> SpanExportResult:
        try:
            for span in spans:
                self._client.send(
                    name=span.name,
                    trace_id=span.context.trace_id,
                    attributes=dict(span.attributes),
                )
            return SpanExportResult.SUCCESS
        except Exception:
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        self._client.close()

    def force_flush(self, timeout_millis=30000) -> bool:
        return True

handle = setup_telemetry(config, span_exporter=MyBackendExporter(my_client))
```

Use the same pattern for `MetricReader` (refer to the [OpenTelemetry documentation](https://opentelemetry-python.readthedocs.io/en/latest/sdk/metrics.html)).

## Custom Exporters and Sampling

Custom exporters plug into the `BatchSpanProcessor` that NeMo Lens installs. By default (`sampler_enabled=False`), no span-level sampling occurs, and a custom exporter receives every span produced on an exporting rank. When `sampler_enabled=True`, NeMo Lens's `RankAwareSampler` makes a single per-rank decision at construction time (`sampling.py`), where all spans on a rank are either kept or dropped together, meaning it is not a per-span sample. If you want a parallel processor regardless of sampling, install your own `SpanProcessor` as shown below:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace.export import BatchSpanProcessor

handle = setup_telemetry(config)
provider = trace.get_tracer_provider()
provider.add_span_processor(BatchSpanProcessor(MyArchiveExporter()))
```

:::{note}
This approach only works on an exporting rank with traces enabled, which occurs when `handle.is_exporting` is `True`. On disabled or non-exporting ranks, `setup_telemetry` installs a `NoOpTracerProvider`, so `trace.get_tracer_provider()` returns a no-op provider that has no `add_span_processor` method, and this call raises an `AttributeError`. Guard this call with `if handle.is_exporting:`:
:::

NeMo Lens does not expose an extension point for custom processors, so if you need one, add it to the provider directly after `setup_telemetry` returns. This is a lower-level interface, but it is stable within the OpenTelemetry SDK.

## Why This API

Before this extension point, connecting NeMo Lens to a non-OTLP backend required subclassing `build_providers` or monkey-patching internals. Now it is a supported extension point: pass your exporter, and NeMo Lens does the rest.

The design goal is to keep NeMo Lens's core surface small (OTLP and console are plenty for most users) while letting power users plug in whatever they need without forking NeMo Lens.
