# Troubleshooting

You enabled NeMo Lens, and something is not showing up where you expect. This page walks through the common failure modes in the order you should check them.

If you're here because of a `RuntimeError` from `setup_telemetry`, skip ahead to [the double-init section](#double-init).

## Nothing Is Being Exported

NeMo Lens never crashes your training job over a broken pipe; export failures are logged and swallowed. That is the right default, but it means you have to go looking when nothing shows up.

### Is `NEMO_LENS_ENABLED` Actually Set for the Python Process?

The env var is read inside the Python process, not by your shell. Check what Python sees:

```python
import os
print(os.environ.get("NEMO_LENS_ENABLED"))
```

Common causes of a mismatch:

- `sudo`, `srun`, or `torchrun` wrappers that strip env vars.
- Docker `ENV` vs. `docker run -e`, where one gets baked in and the other is per-container.
- A virtualenv activation script that overrides the parent shell's env.

If `os.environ` disagrees with your shell, fix the launcher, not NeMo Lens.

### Is the OTLP Endpoint Reachable?

`OTEL_EXPORTER_OTLP_ENDPOINT` defaults to `localhost:4317` when unset. If your collector is elsewhere, set it explicitly:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://collector.internal:4317
```

From the training host, verify reachability with `curl -v $OTEL_EXPORTER_OTLP_ENDPOINT`. A firewall between training and collector is a common and silent failure mode.

### Is This Rank the Exporting Rank?

With the default `single_rank` + `export_rank=-1`, only the last rank exports. If you're poking at rank 0's logs and nothing shows up, that's why.

```python
handle = setup_telemetry(config, rank=rank, world_size=world_size)
print(f"rank {rank}: is_exporting={handle.is_exporting}")
```

See [Sampling](sampling.md) for how to change which rank(s) export.

### Wrong OTLP Protocol

NeMo Lens's exporter builder falls back between gRPC and HTTP automatically (it tries the protocol you ask for, then the other), and `nemo-lens[sdk]` installs both exporter packages, so a missing exporter package is rarely the issue. The real failure mode is a protocol/endpoint mismatch: NeMo Lens defaults to gRPC on port 4317, but your collector might only listen for HTTP on 4318 (or vice versa). Match NeMo Lens to the collector's listener:

```bash
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
export OTEL_EXPORTER_OTLP_ENDPOINT=http://collector.internal:4318
```

### Verify Locally First

Before debugging collector plumbing, confirm NeMo Lens itself is producing spans by routing to a file. See [Send telemetry to a file](../observability/backends.md#file). If you see spans in the file, the problem is downstream; if you don't, it's in NeMo Lens or your config.

## Spans Show Up in Jaeger but They Are Fragmented

You see isolated per-rank traces instead of one cross-rank waterfall. There are two common causes:

### PP Ranks Are Not Linked

For pipeline-parallel Megatron jobs, the `step` / `microbatch` span groups must be enabled, and the PP broadcast must be running. Look for `megatron.pp.recv_forward.linked` spans on non-first-stage ranks; their absence means the broadcast step didn't happen.

If you're running `all_ranks` without the appropriate span groups, each process creates its own root trace, and nothing links them together.

### Incoming Context Is Not Extracted

For HTTP or gRPC services receiving traced requests (e.g., NeMo Gym servers), the request headers carry a `traceparent` that must be extracted, or the server-side span starts a fresh trace.

Use the FastAPI helper:

```python
from nemo.lens.contrib.fastapi import instrument_fastapi
instrument_fastapi(app)
```

Alternatively, you can extract the context manually. See [Context Propagation](context-propagation.md).

(double-init)=
## `setup_telemetry` Raises Already Been Initialized


```
RuntimeError: setup_telemetry() has already been initialised for this process.
```

You're calling `setup_telemetry` twice with `config.enabled=True`. This is almost always a bug, as double-init silently breaks telemetry on the OTel SDK, and NeMo Lens fails loudly instead. See [Double-Init Guard](../design/double-init-guard.md) for the rationale.

If you have a legitimate need (multi-rank simulation in one process, tests), pass `_allow_reinit=True`:

```python
setup_telemetry(config, rank=rank, world_size=world_size, _allow_reinit=True)
```

The leading underscore is deliberate. If you're reaching for it in production code, step back.

## Span Group Is Not Active

You instrumented a site with `managed_span('my_group', ...)`, and nothing exports. There are two usual suspects:

### `my_group` Is Not in the Enabled Set

The base `default` preset includes only `job`, `checkpoint`, and `evaluate`, but consumer subclasses extend it; Megatron's `default` adds `inference`, and Gym's adds `server` (RL's matches the base). The authoritative set is the `_PRESETS`/`ALL_GROUPS` of the `SpanGroup` subclass you pass to `from_env(span_group_cls=...)`. Groups like `step`, `microbatch`, `forward_backward` live in `per_step` or `all`. Check what's enabled:

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

### The Preset Resolved but the Name Is Wrong

`span_groups` resolution is strict; unknown names raise `ValueError` at config time with the valid list. If your app started, the name is real. If you just added a new group, double-check it's in `ALL_GROUPS` of the `SpanGroup` subclass being passed to `from_env(span_group_cls=...)`.

### You Are on a Non-Exporting Rank

On non-exporting ranks, `setup_telemetry` sets the enabled group set to empty regardless of config. Nothing in this rank will export, by design. See [Sampling](sampling.md).

## Metrics Not Showing Up in Prometheus

### Give It One Export Cycle

Metrics go through `PeriodicExportingMetricReader`. The default interval is 10 seconds; wait at least one cycle after your first metric record before expecting to see anything.

### Prometheus Scrapes the Collector, Not You

The application exports OTLP. The collector converts this data into the Prometheus-format endpoint scraped by Prometheus. Confirm the collector is exposing its metrics endpoint (typically `:8889/metrics`) and that Prometheus has it in its scrape config.

### SDK-Appended Unit Suffixes

The OTel SDK appends units to metric names when it has them. A metric like `rl.reward.mean` becomes `rl_reward_mean` (dots to underscores); duration metrics ending in `_ms` (e.g., `gym.server.request_duration_ms`) might get `_milliseconds` appended. Use the Prometheus metric browser to find the exact name rather than guessing.

## Tracing Is Slow or Blocking My Training Loop

Telemetry collection is designed to be lightweight and non-blocking. If you experience performance degradation or backpressure in your training loop, check these potential bottlenecks.

### Confirm It's Telemetry, Not Something Else

Toggle NeMo Lens off (`NEMO_LENS_ENABLED=0`) and retime the workload. If speed comes back, the overhead is real and configurable; if it doesn't, look elsewhere.

### Narrow the Scope

Switch from `per_step` or `all` to `default`. Spans in the hot loop (microbatch, communication) are the likely cost. See [Span Groups](span-groups.md) and [Sampling](sampling.md) for how to dial this in.

### BatchSpanProcessor Should Never Block

`BatchSpanProcessor` is asynchronous and batches on a background thread. If it's blocking, the collector is probably unreachable and the internal queue is full, so span creation starts backpressuring. Fix the collector or route to a file while you debug.

## My Custom Exporter Is Not Receiving Spans

If you are using a custom exporter but it is not receiving any telemetry data, ensure it is wired correctly. Check these common setup mistakes.

### Wire It at `setup_telemetry` Time

Custom exporters are attached through the `span_exporter=` argument to `setup_telemetry`. Constructing one after `setup_telemetry` returned doesn't do anything, as the provider is already built.

```python
handle = setup_telemetry(config, rank=0, world_size=1, span_exporter=my_exporter)
```

See [Custom Exporters](custom-exporters.md).

### Check `traces_enabled`

If `config.traces_enabled=False` (or `NEMO_LENS_TRACES_ENABLED=0`), the TracerProvider is a no-op regardless of what exporter you passed in; metrics toggle independently.
