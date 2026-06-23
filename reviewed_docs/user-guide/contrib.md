# Contrib Helpers

`nemo.lens.contrib` contains framework-specific integration helpers. Each is optional and isolated — installing lens without the corresponding extra raises a clear ImportError (with an install hint) when you call the helper, not when you import it.

## FastAPI — `contrib.fastapi`

```python
from nemo.lens.contrib.fastapi import instrument_fastapi

app = FastAPI()
instrument_fastapi(app)
```

Wraps `opentelemetry-instrumentation-fastapi`. After this call, every incoming HTTP request gets a span covering its lifetime, with W3C trace context automatically extracted from request headers (so upstream traces flow through).

Note: `instrument_fastapi` does accept a `service_name` parameter, but it is currently a no-op (the implementation ignores it). The service name is set via `setup_telemetry` / the `OTEL_SERVICE_NAME` environment variable, not here.

Install: `pip install 'nemo-lens[fastapi]'`

### Typical integration

Gate on a span group so FastAPI spans respect your telemetry toggles:

```python
from nemo.lens.state import is_span_group_enabled

if is_span_group_enabled('server'):
    instrument_fastapi(app)
```

## aiohttp client — `contrib.aiohttp`

```python
from nemo.lens.contrib.aiohttp import instrument_aiohttp_client

instrument_aiohttp_client()
# After this, every aiohttp ClientSession request has W3C context injected automatically.
```

Wraps `opentelemetry-instrumentation-aiohttp-client`. Eliminates the need to manually call `inject_context(kwargs['headers'])` on every outbound HTTP call.

Install: `pip install 'nemo-lens[aiohttp]'`

### When to call it

Once at startup, after `setup_telemetry` returns and confirms you're actively exporting:

```python
handle = setup_telemetry(config)
if handle.is_exporting:
    instrument_aiohttp_client()
```

On non-exporting ranks, there's no reason to pay the instrumentation cost.

## Ray — `contrib.ray`

Ray remote calls don't carry HTTP headers, so trace context must be passed explicitly. Lens exposes helpers that add a conventional `_otel_carrier` kwarg to remote calls.

### Driver side

```python
from nemo.lens.contrib.ray import inject_ray_context, ray_dispatch_with_context

# Method 1: manual carrier
carrier = inject_ray_context()
future = my_actor.method.remote(arg1, arg2, _otel_carrier=carrier)

# Method 2: dispatch helper
future = ray_dispatch_with_context(my_actor.method, arg1, arg2)
```

### Worker side

Wrap remote methods with `traced_remote_call` to auto-extract context:

```python
from nemo.lens.contrib.ray import traced_remote_call

@ray.remote
class MyActor:
    @traced_remote_call
    def method(self, arg1, arg2):
        # _otel_carrier kwarg is consumed by the decorator and used to set
        # the current span context before method body runs
        ...
```

Spans created inside `method` now appear as children of the driver's span.

No extra install needed — uses `opentelemetry-api` only.

## NCCL — `contrib.nccl`

NCCL transfers are raw bytes — no native header concept. For pipeline-parallel correlation where you want to piggy-back trace context on a tensor transfer:

```python
from nemo.lens.contrib.nccl import serialize_context, extract_nccl_context

# Sender
data = serialize_context()    # JSON-encoded W3C carrier as bytes
# ... send `data` alongside your tensor via NCCL ...

# Receiver
ctx = extract_nccl_context(data)
# attach ctx as parent context for new spans
```

If you need the intermediate carrier dict rather than a ready-to-use OTel `Context`, call `deserialize_context(data: bytes) -> dict | None` — the mid-layer that `extract_nccl_context` wraps. It returns the decoded carrier dict, or `None` if the bytes are malformed (it swallows `JSONDecodeError` and `UnicodeDecodeError`):

```python
from nemo.lens.contrib.nccl import deserialize_context

carrier = deserialize_context(data)    # dict, or None on bad input
```

In practice, most pipeline-parallel users don't need this — `broadcast_trace_context` is simpler and more idiomatic (see [Distributed Tracing](distributed-tracing.md)). NCCL helpers exist for advanced cases where you're already passing metadata alongside tensors and trace context can piggy-back for free.

No extra install needed.

## Design notes

Contrib modules are **thin** — each wraps an existing OTel instrumentation package or provides a couple of helper functions. They don't implement tracing logic themselves.

If you need to add a contrib module:

1. Check if an `opentelemetry-instrumentation-<framework>` package exists upstream. If yes, your module should be a single function that imports and calls it.
2. Add the package as an optional extra in `pyproject.toml` (`nemo-lens[<framework>]`).
3. Raise an `ImportError` with an actionable install hint if the instrumentation package isn't present.

This keeps the contrib surface small and maintenance burden low.
