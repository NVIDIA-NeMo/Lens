# Troubleshooting

You enabled lens and something isn't showing up where you expect. This page walks the common failure modes in the order you should check them.

If you're here because of a `RuntimeError` from `setup_telemetry`, skip ahead to [the double-init section](#double-init).

## Nothing is being exported

Lens never crashes your training job over a broken pipe — export failures are logged and swallowed. That's the right default, but it means you have to go looking when nothing shows up.

### Is `NEMO_LENS_ENABLED` actually set for the Python process?

The env var is read inside the Python process, not by your shell. Check what Python sees:

```python
import os
print(os.environ.get("NEMO_LENS_ENABLED"))
```

Common causes of a mismatch:

- `sudo`, `srun`, or `torchrun` wrappers that strip env vars.
- Docker `ENV` vs `docker run -e` — one gets baked in, the other is per-container.
- A virtualenv activation script that overrides the parent shell's env.

If `os.environ` disagrees with your shell, fix the launcher, not lens.

### Is the OTLP endpoint reachable?

`OTEL_EXPORTER_OTLP_ENDPOINT` defaults to `localhost:4317` when unset. If your collector is elsewhere, set it explicitly:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://collector.internal:4317
```

From the training host, verify reachability with `curl -v $OTEL_EXPORTER_OTLP_ENDPOINT`. A firewall between training and collector is a common and silent failure mode.

### Is this rank the exporting rank?

With the default `single_rank` + `export_rank=-1`, only the last rank exports. If you're poking at rank 0's logs and nothing shows up, that's why.

```python
handle = setup_telemetry(config, rank=rank, world_size=world_size)
print(f"rank {rank}: is_exporting={handle.is_exporting}")
```

See [Sampling](sampling.md) for how to change which rank(s) export.

### Wrong OTLP protocol

If you only have `opentelemetry-exporter-otlp-proto-http` installed (not `...-grpc`), the SDK will not silently fall back — span export fails. Set the protocol explicitly:

```bash
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
```

### Verify locally first

Before debugging collector plumbing, confirm lens itself is producing spans by routing to a file. See [Send telemetry to a file](../observability/backends.md#file). If you see spans in the file, the problem is downstream; if you don't, it's in lens or your config.

## Spans show up in Jaeger but they're fragmented

You see isolated per-rank traces instead of one cross-rank waterfall. Two common causes:

### PP ranks aren't linked

For pipeline-parallel Megatron jobs, the `step` / `microbatch` span groups must be enabled and the PP broadcast must be running. Look for `megatron.pp.recv_forward.linked` spans on non-first-stage ranks — their absence means the broadcast step didn't happen.

If you're running `all_ranks` without the appropriate span groups, each process creates its own root trace and nothing links them together.

### Incoming context not extracted

For HTTP or gRPC services receiving traced requests (e.g. Gym servers), the request headers carry a `traceparent` that must be extracted, or the server-side span starts a fresh trace.

Use the FastAPI helper:

```python
from nemo.lens.contrib.fastapi import instrument_fastapi
instrument_fastapi(app)
```

Or extract manually — see [Context Propagation](context-propagation.md).

(double-init)=
## `setup_telemetry` raises "already been initialised"

```
RuntimeError: setup_telemetry() has already been initialised for this process.
```

You're calling `setup_telemetry` twice with `config.enabled=True`. This is almost always a bug — double-init silently breaks telemetry on the OTel SDK, and lens fails loudly instead. See [Double-Init Guard](../design/double-init-guard.md) for the rationale.

If you have a legitimate need (multi-rank simulation in one process, tests), pass `_allow_reinit=True`:

```python
setup_telemetry(config, rank=rank, world_size=world_size, _allow_reinit=True)
```

The leading underscore is deliberate. If you're reaching for it in production code, step back.

## Span group isn't active

You instrumented a site with `managed_span('my_group', ...)` and nothing exports. Two usual suspects:

### `my_group` isn't in the enabled set

`default` includes only `job`, `checkpoint`, and `evaluate`. Groups like `step`, `microbatch`, `forward_backward` live in `per_step` or `all`. Check what's enabled:

```python
from nemo.lens.state import is_span_group_enabled
print(is_span_group_enabled("my_group"))
```

Set `NEMO_LENS_SPAN_GROUPS` to include it:

```bash
NEMO_LENS_SPAN_GROUPS=per_step
# or
NEMO_LENS_SPAN_GROUPS=default,my_group
```

### The preset resolved but the name is wrong

`span_groups` resolution is strict — unknown names raise `ValueError` at config time with the valid list. If your app started, the name is real. If you just added a new group, double-check it's in `ALL_GROUPS` of the `SpanGroup` subclass being passed to `from_env(span_group_cls=...)`.

### You're on a non-exporting rank

On non-exporting ranks, `setup_telemetry` sets the enabled group set to empty regardless of config. Nothing in this rank will export, by design — see [Sampling](sampling.md).

## Metrics not showing up in Prometheus

### Give it one export cycle

Metrics go through `PeriodicExportingMetricReader`. The default interval is 10 seconds — wait at least one cycle after your first metric record before expecting to see anything.

### Prometheus scrapes the collector, not you

The application exports OTLP; the collector converts that into the Prometheus-format endpoint Prometheus scrapes. Confirm the collector is exposing its metrics endpoint (typically `:8889/metrics`) and that Prometheus has it in its scrape config.

### SDK-appended unit suffixes

The OTel SDK appends units to metric names when it has them. `megatron.training.loss` becomes `megatron_training_loss` (dots → underscores); duration metrics ending in `_ms` may get `_milliseconds` appended. Use the Prometheus metric browser to find the exact name rather than guessing.

## Tracing is slow or blocking my training loop

### Confirm it's telemetry, not something else

Toggle lens off (`NEMO_LENS_ENABLED=0`) and retime the workload. If speed comes back, the overhead is real and configurable; if it doesn't, look elsewhere.

### Narrow the scope

Switch from `per_step` or `all` to `default`. Spans in the hot loop (microbatch, communication) are the likely cost. See [Span Groups](span-groups.md) and [Sampling](sampling.md) for how to dial this in.

### BatchSpanProcessor should never block

`BatchSpanProcessor` is async — it batches on a background thread. If it's blocking, the collector is probably unreachable and the internal queue is full, so span creation starts backpressuring. Fix the collector or route to a file while you debug.

## My custom exporter isn't receiving spans

### Wire it at `setup_telemetry` time

Custom exporters are attached through the `span_exporter=` argument to `setup_telemetry`. Constructing one after `setup_telemetry` returned doesn't do anything — the provider is already built.

```python
handle = setup_telemetry(config, rank=0, world_size=1, span_exporter=my_exporter)
```

See [Custom Exporters](custom-exporters.md).

### Check `traces_enabled`

If `config.traces_enabled=False` (or `NEMO_LENS_TRACES_ENABLED=0`), the TracerProvider is a no-op regardless of what exporter you passed in. Metrics toggles independently.
