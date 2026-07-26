# Metrics

Lens ships opinionated metric instruments under `nemo.lens.instruments` for common observability needs: GenAI inference, RL training, and Gym servers. Training-specific metrics (e.g. `megatron.training.loss`) live in the consumer project — they are not generic.

Import each record function from its submodule, e.g. `from nemo.lens.instruments.rl import record_rl_metrics`. Only `record_inference_metrics` is also re-exported at the package level (`from nemo.lens.instruments import record_inference_metrics`); the RL and Gym functions are available only via their submodules.

The `meter` argument is the OTel `Meter` to record on. You can use `handle.meter` from `setup_telemetry()`, or grab one directly with `get_meter(name="nemo.lens")` (`from nemo.lens import get_meter`).

## Architecture

Each module under `instruments/` follows the same pattern:

- A module-level `WeakKeyDictionary` caches instruments per `Meter`, so re-initialising the meter doesn't leak memory.
- A `_get_*_instruments(meter)` helper creates (and caches) all instruments for a meter on first call.
- A `record_*_metrics(meter, ...)` function takes a required `meter` plus optional per-metric arguments (best passed by keyword) and records only the ones that are not `None`.

This means callers can record partial data without conditional logic:

```python
record_rl_metrics(handle.meter, reward_mean=r, policy_loss=p)   # only these two
record_rl_metrics(handle.meter, kl_divergence=kl)                # just one
```

## Inference — `instruments/inference.py`

Emits metrics following the [OTel GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/).

```python
from nemo.lens.instruments.inference import record_inference_metrics

record_inference_metrics(
    handle.meter,
    request_duration_s=0.42,
    model="llama-3-8b",
    input_tokens=128,
    output_tokens=256,
)
```

| Instrument | OTel name | Type | Unit |
|---|---|---|---|
| Request duration | `gen_ai.server.request.duration` | Histogram | s |
| Token usage | `gen_ai.client.token.usage` | Histogram | `{token}` |

Token usage is labelled with `gen_ai.token.type` (`"input"` or `"output"`) — filter on that label in Prometheus/Grafana.

All data points carry `gen_ai.operation.name = "text_completion"` and `gen_ai.provider.name = "nemo"` by default; override via `operation_name=` / `provider_name=` args.

## RL — `instruments/rl.py`

Emits RL-specific gauges and histograms in the `rl.*` namespace.

```python
from nemo.lens.instruments.rl import record_rl_metrics

record_rl_metrics(
    handle.meter,
    reward_mean=0.75,
    kl_divergence=0.01,
    policy_loss=0.3,
    value_loss=0.5,
    entropy=2.1,
    response_length_mean=128.0,
    grad_norm=1.7,
    learning_rate=3e-6,
    tokens_per_sec=18500.0,
    generation_duration_ms=450.0,
    rollout_duration_ms=2300.0,
)
```

| Metric | Type | Description |
|---|---|---|
| `rl.reward.mean` | Gauge | Mean reward across rollout batch |
| `rl.kl_divergence` | Gauge | KL divergence between policy and reference |
| `rl.policy_loss` | Gauge | Policy gradient loss |
| `rl.value_loss` | Gauge | Value function loss |
| `rl.entropy` | Gauge | Policy entropy |
| `rl.response_length.mean` | Gauge | Mean generated response length (tokens) |
| `rl.grad_norm` | Gauge | Gradient norm of the policy update |
| `rl.learning_rate` | Gauge | Current optimizer learning rate |
| `rl.throughput.tokens_per_sec` | Gauge (`{token}/s`) | Training throughput (tokens/sec) |
| `rl.generation.duration_ms` | Histogram (ms) | Text generation duration |
| `rl.rollout.duration_ms` | Histogram (ms) | Rollout collection duration |

## Gym — `instruments/gym.py`

Emits Gym server metrics in the `gym.*` namespace.

```python
from nemo.lens.instruments.gym import record_gym_metrics

record_gym_metrics(
    handle.meter,
    server_request_duration_ms=42.0,
    verify_duration_ms=120.0,
    verify_success_rate=0.87,
    active_servers=4,
)
```

| Metric | Type | Description |
|---|---|---|
| `gym.server.request_duration_ms` | Histogram (ms) | Incoming request duration |
| `gym.rollout.duration_ms` | Histogram (ms) | Rollout collection duration |
| `gym.verify.duration_ms` | Histogram (ms) | Verification endpoint duration |
| `gym.verify.success_rate` | Gauge | Fraction of successful verifications |
| `gym.servers.active` | Gauge | Number of active Gym servers |

## Writing custom instruments

The same `WeakKeyDictionary` pattern works for project-specific metrics. Megatron's `instruments/training.py` (shipped in the Megatron repo, not lens) emits `megatron.training.*` metrics the same way.

If your project has a recurring metric shape, add a module under `nemo.lens.instruments.<domain>` with:

```python
import weakref
from opentelemetry import metrics

_INSTRUMENTS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

def _get_instruments(meter: metrics.Meter) -> dict:
    instruments = _INSTRUMENTS.get(meter)
    if instruments is None:
        instruments = {
            "latency": meter.create_histogram("my.op.latency_ms", unit="ms"),
            "queue_depth": meter.create_gauge("my.queue.depth"),
        }
        _INSTRUMENTS[meter] = instruments
    return instruments

def record_my_metrics(meter, latency_ms=None, queue_depth=None):
    i = _get_instruments(meter)
    if latency_ms is not None:
        i["latency"].record(latency_ms)
    if queue_depth is not None:
        i["queue_depth"].set(queue_depth)
```

## Gauge vs Histogram

- **Gauge**: point-in-time value. Prometheus shows the last reported value. Good for losses, rates, counts.
- **Histogram**: distribution of values. Prometheus computes quantiles. Good for durations, sizes, anything where percentiles matter.
- **Counter**: monotonic cumulative value. Prometheus shows the rate of change. Use for event counts (`skipped_iters`, `errors`).

Don't put durations on gauges — you lose p99. Don't put event counts on histograms — the cardinality is wrong.

## Metrics vs span attributes vs resource attributes

A common mistake is mixing these up. Use this decision table:

| Value | Where to put it |
|---|---|
| Changes over time, numerical (loss, throughput) | **Metric** |
| Categorical per-span context (iteration, microbatch_id, skipped) | **Span attribute** |
| Stable for the process lifetime (rank, parallelism config, model arch) | **Resource attribute** (via `resource_attributes=` in `setup_telemetry`) |

In particular: **never put a continuously-varying metric like loss on a span attribute**. It wastes span storage and Jaeger can't aggregate across spans. Use `record_*_metrics()` to emit it as a real metric.
