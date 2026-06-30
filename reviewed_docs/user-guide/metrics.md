# Metrics

NeMo Lens ships opinionated metric instruments under `nemo.lens.instruments` for common observability needs: GenAI inference, RL training, and Gym servers. Training-specific metrics (e.g., `megatron.training.loss`) live in the consumer project; they are not generic.

Import each record function from its submodule, e.g., `from nemo.lens.instruments.rl import record_rl_metrics`. Only `record_inference_metrics` is re-exported at the package level (`from nemo.lens.instruments import record_inference_metrics`); the RL and Gym functions are available only through their submodules.

The `meter` argument is the OTel `Meter` to record on. You can use `handle.meter` from `setup_telemetry()`, or grab one directly with `get_meter(name="nemo.lens")` (`from nemo.lens import get_meter`).

## Architecture

Each module under `instruments/` follows the same pattern:

- A module-level `WeakKeyDictionary` caches instruments per `Meter`, so re-initializing the meter does not leak memory.
- A `_get_*_instruments(meter)` helper creates (and caches) all instruments for a meter on first call.
- A `record_*_metrics(meter, ...)` function takes a required `meter` plus optional per-metric arguments (best passed by keyword) and records only the ones that are not `None`.

Callers can record partial data without conditional logic:

```python
record_rl_metrics(handle.meter, reward_mean=r, policy_loss=p)   # only these two
record_rl_metrics(handle.meter, kl_divergence=kl)                # just one
```

## Inference with `instruments/inference.py`

This module emits metrics following the [OTel GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/).

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

Token usage is labeled with `gen_ai.token.type` (`"input"` or `"output"`); filter on that label in Prometheus or Grafana.

All data points carry `gen_ai.operation.name = "text_completion"` and `gen_ai.provider.name = "nemo"` by default; override through the `operation_name=` or `provider_name=` args.

## NeMo RL with `instruments/rl.py`

This module emits NeMo RL-specific gauges and histograms in the `rl.*` namespace.

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
| `rl.generation.duration_ms` | Histogram (ms) | Text generation duration |
| `rl.rollout.duration_ms` | Histogram (ms) | Rollout collection duration |

## NeMo Gym with `instruments/gym.py`

This module emits NeMo Gym server metrics in the `gym.*` namespace.

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

## Write Custom Instruments

The same `WeakKeyDictionary` pattern works for project-specific metrics. Megatron's `instruments/training.py` (shipped in the Megatron repository, not NeMo Lens) emits `megatron.training.*` metrics the same way.

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

## Choose Between Gauges and Histograms

- **Gauge**: Point-in-time value. Prometheus shows the last reported value. Good for losses, rates, and counts.
- **Histogram**: Distribution of values. Prometheus computes quantiles. Good for durations, sizes, and any value where percentiles matter.
- **Counter**: Monotonic cumulative value. Prometheus shows the rate of change. Use for event counts (`skipped_iters` and `errors`).

Do not put durations on gauges; you lose the ninety-ninth percentile. Do not put event counts on histograms; the cardinality is incorrect.

## Metrics, Span Attributes, and Resource Attributes

Avoid mixing these concepts. Use the following decision table to choose the correct telemetry type:

| Value | Where to put it |
|---|---|
| Changes over time, numerical (loss, throughput) | **Metric** |
| Categorical per-span context (iteration, microbatch_id, skipped) | **Span attribute** |
| Stable for the process lifetime (rank, parallelism configuration, model architecture) | **Resource attribute** (through `resource_attributes=` in `setup_telemetry`) |

Specifically, do not record a continuously-varying metric, such as loss, as a span attribute. Doing so wastes span storage, and Jaeger cannot aggregate across spans. Use `record_*_metrics()` to emit the value as a real metric.
