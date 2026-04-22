# Context Propagation

When a traced request crosses a process boundary — HTTP call, gRPC message, message queue, Ray remote call — its **trace context** (trace ID, parent span ID, baggage) must travel with it. Otherwise downstream spans end up in a different trace and the cross-service waterfall is lost.

Lens exposes two primitives for this, matching the OTel W3C TraceContext + Baggage propagators.

## `inject_context` — outbound

```python
from nemo.lens import inject_context

headers = {}
inject_context(headers)
# headers == {'traceparent': '00-<trace_id>-<span_id>-01', 'tracestate': '...', ...}

await http_client.post(url, headers=headers, json=body)
```

`inject_context(carrier)` writes the current OTel context into the `carrier` dict. The carrier is whatever your transport uses — HTTP headers, gRPC metadata, message attributes.

## `extract_context` — inbound

```python
from nemo.lens import extract_context
from opentelemetry import trace, context

# On the receiving side
ctx = extract_context(request.headers)
token = context.attach(ctx)
try:
    with trace.get_tracer(__name__).start_as_current_span("handle.request"):
        ...
finally:
    context.detach(token)
```

`extract_context(carrier)` parses W3C headers and returns an OTel `Context`. Attach it before starting child spans, detach when done.

If the carrier has no valid trace context, the returned `Context` is empty — new spans will start a fresh trace, which is the correct behaviour.

## Auto-instrumentation for common transports

Writing `inject_context` / `extract_context` by hand on every call is error-prone. For common frameworks, lens ships auto-instrumentation helpers:

- **FastAPI**: `nemo.lens.contrib.fastapi.instrument_fastapi(app)` — extracts context from every incoming request and makes it the current span's parent.
- **aiohttp client**: `nemo.lens.contrib.aiohttp.instrument_aiohttp_client()` — injects context into every outgoing request.
- **Ray**: `nemo.lens.contrib.ray.inject_ray_context()` / `extract_ray_context()` / `traced_remote_call(method)` — helpers for Ray's kwargs-based propagation.
- **NCCL**: `nemo.lens.contrib.nccl.serialize_context()` / `extract_nccl_context(data)` — helpers for piggy-backing context on NCCL byte transfers.

See [Contrib](contrib.md) for details.

**When auto-instrumentation is available, prefer it over manual injection** — it covers every call site, handles error paths, and doesn't rot when new code is added.

## Cross-rank propagation in distributed training

HTTP-style propagation doesn't apply to `torch.distributed` — those aren't carrier-based transports. For distributed training, lens provides:

- `broadcast_trace_context(rank, src_rank=0)` — uses `torch.distributed.broadcast` to share trace context across ranks.
- `create_linked_span(tracer, name, remote_carrier=carrier)` — creates a span with an OTel Link (not parent-child) to a remote span, useful for pipeline-parallel correlation.

See [Distributed Tracing](distributed-tracing.md) for the full pattern.

## Baggage

[Baggage](https://www.w3.org/TR/baggage/) is a W3C standard for propagating small key/value context alongside trace context — "which customer is this request for?", "is this a canary deployment?".

Lens's propagator is a `CompositePropagator` of `TraceContextTextMapPropagator` and `W3CBaggagePropagator`, so both flow through `inject_context` / `extract_context` automatically. Set baggage with:

```python
from opentelemetry import baggage, context

ctx = baggage.set_baggage("customer.id", "42")
token = context.attach(ctx)
try:
    # All downstream inject_context calls will include customer.id in the baggage header
    ...
finally:
    context.detach(token)
```

Baggage values are propagated across service boundaries but are not automatically added to spans. Use them for filtering/routing decisions, not for span attributes.
