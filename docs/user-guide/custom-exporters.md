# Custom Exporters

`setup_telemetry` supports injecting custom `SpanExporter` and `MetricReader` instances, bypassing the config-based construction. This is the extension point for:

- In-memory exporters for testing
- Custom exporters for proprietary backends
- Exporters that require fine-grained configuration lens doesn't expose

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

When provided, these override the config's `exporter` field for that signal. You can mix: pass a custom `span_exporter` and let metrics use the config's exporter.

## Custom span exporter

```python
from opentelemetry.sdk.trace.export import InMemorySpanExporter

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

This is how the lens test suite captures spans for assertions.

## Custom metric reader

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

## Writing a custom exporter

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

Same pattern for `MetricReader` — see [OTel docs](https://opentelemetry-python.readthedocs.io/en/latest/sdk/metrics.html).

## Custom exporters and sampling

Custom exporters plug into the `BatchSpanProcessor` that lens installs. They see only **sampled** spans. If you need the pre-sample stream (e.g. for archival before sampling), install your own `SpanProcessor` and use lens's config-based exporter:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace.export import BatchSpanProcessor

handle = setup_telemetry(config)
provider = trace.get_tracer_provider()
provider.add_span_processor(BatchSpanProcessor(MyArchiveExporter()))
```

Lens doesn't expose an extension point for custom processors — if you need one, add it to the provider directly after `setup_telemetry` returns. This is a lower-level interface, but it's stable within OTel SDK.

## Why this API

Before this extension point, connecting lens to a non-OTLP backend required subclassing `build_providers` or monkeypatching internals. Now it's a supported extension point: pass your exporter, lens does the rest.

The design goal is to keep lens's core surface small (OTLP + console is plenty for most users) while letting power users plug in whatever they need without forking lens.
