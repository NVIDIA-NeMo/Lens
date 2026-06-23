# Configuration

`NemoLensConfig` is the single configuration object consumed by `setup_telemetry`. It holds every knob exposed by the library.

## Construction

### From environment

```python
from nemo.lens import NemoLensConfig

cfg = NemoLensConfig.from_env()
```

Reads `NEMO_LENS_*` env vars. For library-specific prefixes, pass `prefix` and `fallback_prefix`:

```python
cfg = NemoLensConfig.from_env(
    prefix='MEGATRON_OTEL',
    fallback_prefix='NEMO_LENS',
    span_group_cls=MegatronSpanGroup,
)
```

The **prefix/fallback** pattern lets each consumer have library-scoped env vars while sharing common defaults. Primary prefix wins; fallback applies only if the primary is unset.

### Direct construction

```python
cfg = NemoLensConfig(
    enabled=True,
    service_name='my-training-run',
    export_strategy='all_ranks',
    span_groups='per_step',
)
```

Field validation runs in `__post_init__`: `export_sample_rate` must be in `[0.0, 1.0]`, otherwise `ValueError`.

## Fields

### Core

| Field | Default | Description |
|---|---|---|
| `enabled` | `False` | Master toggle. Must be `True` to activate any telemetry. |
| `service_name` | `"nemo"` | OTLP service name. Overridden by `OTEL_SERVICE_NAME`. |

### Export strategy

Controls which ranks send telemetry to the collector. Three strategies are available: `single_rank` (default), `all_ranks`, and `sampled`. See [Sampling](sampling.md) for detailed semantics, when to use each, and how they compose with OTel SDK samplers. Unknown strategy names raise `ValueError` at `setup_telemetry` time, not at config construction — register custom strategies before initialising telemetry. See [Custom Strategies](custom-strategies.md).

| Field | Default | Description |
|---|---|---|
| `export_strategy` | `"single_rank"` | `"single_rank"`, `"all_ranks"`, `"sampled"`, `"first_rank_per_node"`, or any name registered via [`register_export_strategy`](custom-strategies.md). |
| `export_rank` | `-1` | For `single_rank`: which rank exports. `-1` means the last rank. |
| `export_sample_rate` | `1.0` | For `sampled`: fraction of ranks in `[0.0, 1.0]`. Validated at config time. |
| `sampler_enabled` | `False` | Install `RankAwareSampler` on the TracerProvider for SDK-level per-rank filtering. See [Sampling](sampling.md). |

### Signal toggles

| Field | Default | Description |
|---|---|---|
| `traces_enabled` | `True` | Enable trace spans. |
| `metrics_enabled` | `True` | Enable metric instruments. |
| `logs_enabled` | `False` | Enable the OTel log bridge. See [Logging Bridge](logging-bridge.md). |

### Granularity

| Field | Default | Description |
|---|---|---|
| `span_groups` | `"default"` | Comma-separated spec of preset keywords or group names. See [Span Groups](span-groups.md). |

### Backend

| Field | Default | Description |
|---|---|---|
| `exporter` | `"otlp"` | `"otlp"` (gRPC, falls back to HTTP) or `"console"` (stdout, for local debugging). |

With `exporter="console"` (env `NEMO_LENS_EXPORTER=console`), lens uses the SDK's `ConsoleSpanExporter` / `ConsoleMetricExporter`, which print spans and metrics to stdout. Any value other than `"otlp"` or `"console"` raises `ValueError("Unknown exporter type: ...")` when providers are built.

### Identification

| Field | Default | Description |
|---|---|---|
| `run_id` | auto | Unique run ID. Auto-generated from `SLURM_JOB_ID` or a UUID if empty. Shared across all ranks. |
| `user` | `""` | Optional user/team label. Emitted as `nemo.user.id`. |

### W&B Weave

| Field | Default | Description |
|---|---|---|
| `wandb_entity` | `""` | W&B team/user name — set as `wandb.entity` resource attribute. |
| `wandb_project` | `""` | W&B project name — set as `wandb.project` resource attribute. |

## Environment variables

Most config fields have a corresponding `<PREFIX>_<KEY>` env var. Three are exceptions that bypass the prefix/fallback model entirely: `service_name` reads bare `OTEL_SERVICE_NAME`, and `wandb_entity`/`wandb_project` read bare `WANDB_ENTITY`/`WANDB_PROJECT` (no prefix or fallback). Note also that the `user` field's env var is `<PREFIX>_USER_ID` (e.g. `NEMO_LENS_USER_ID`), not `_USER`. Using `NEMO_LENS` as the prefix:

| Variable | Field |
|---|---|
| `NEMO_LENS_ENABLED` | `enabled` |
| `NEMO_LENS_EXPORT_STRATEGY` | `export_strategy` |
| `NEMO_LENS_EXPORT_RANK` | `export_rank` |
| `NEMO_LENS_EXPORT_SAMPLE_RATE` | `export_sample_rate` |
| `NEMO_LENS_SAMPLER_ENABLED` | `sampler_enabled` |
| `NEMO_LENS_TRACES_ENABLED` | `traces_enabled` |
| `NEMO_LENS_METRICS_ENABLED` | `metrics_enabled` |
| `NEMO_LENS_LOGS_ENABLED` | `logs_enabled` |
| `NEMO_LENS_SPAN_GROUPS` | `span_groups` |
| `NEMO_LENS_EXPORTER` | `exporter` |
| `NEMO_LENS_RUN_ID` | `run_id` |
| `NEMO_LENS_USER_ID` | `user` |
| `WANDB_ENTITY` | `wandb_entity` |
| `WANDB_PROJECT` | `wandb_project` |

Boolean parsing accepts: `1`/`0`, `true`/`false`, `yes`/`no`, `on`/`off` (case-insensitive). Any other value raises `ValueError`.

## Standard OTel SDK variables

Standard OTel SDK env vars are honoured — most are read by the SDK directly, but a few are consumed by lens's provider-building code: `OTEL_EXPORTER_OTLP_PROTOCOL` (and the signal-specific `OTEL_EXPORTER_OTLP_{TRACES,METRICS,LOGS}_PROTOCOL`, which wins; default `grpc`) selects the OTLP transport, and `OTEL_METRIC_EXPORT_INTERVAL` (default `10000` ms) sets the metric reader's export interval:

| Variable | Example |
|---|---|
| `OTEL_SERVICE_NAME` | `my-training-run` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4317` |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | `grpc` or `http/protobuf` |
| `OTEL_EXPORTER_OTLP_HEADERS` | `Authorization=Bearer <token>` |
| `OTEL_TRACES_SAMPLER` | `parentbased_traceidratio` |
| `OTEL_TRACES_SAMPLER_ARG` | `0.1` |
| `OTEL_METRIC_EXPORT_INTERVAL` | `10000` (ms) |
| `OTEL_SDK_DISABLED` | `true` |

In addition, `build_providers` reads the non-prefixed `DEPLOYMENT_ENV` (falling back to `ENVIRONMENT`) and, when set, emits it as the `deployment.environment` resource attribute.

## setup_telemetry signature

```python
setup_telemetry(
    config: NemoLensConfig,
    rank: int = 0,
    world_size: int = 1,
    resource_attributes: dict | None = None,
    span_exporter=None,
    metric_reader=None,
    export_strategy: ExportStrategy | None = None,
    _allow_reinit: bool = False,
) -> TelemetryHandle
```

| Parameter | Description |
|---|---|
| `config` | The `NemoLensConfig` — typically from `from_env()`. |
| `rank` / `world_size` | Distributed position, used for export strategy and resource attributes. |
| `resource_attributes` | Extra attributes to merge into the OTel `Resource` (become Jaeger "Process" tags). |
| `span_exporter` | Optional custom `SpanExporter`, bypasses config-based construction. See [Custom Exporters](custom-exporters.md). |
| `metric_reader` | Optional custom `MetricReader`, bypasses config-based construction. |
| `export_strategy` | Optional callable `(config, rank, world_size) -> bool` that bypasses the registry-based strategy dispatch (per-call override, no global registration needed). See [Custom Strategies](custom-strategies.md). |
| `_allow_reinit` | Escape hatch for testing only — bypasses the [double-init guard](../design/double-init-guard.md). |

Returns a `TelemetryHandle` exposing:

- `.tracer` / `.meter` — read-only properties holding the OTel tracer and meter (no-op objects on non-exporting ranks).
- `.is_exporting` — `bool` indicating whether this rank built real exporting providers.
- `.shutdown(timeout_ms=5000)` — force-flushes and shuts down both the tracer and meter providers.

**Call once per process.** A second call with `config.enabled=True` raises `RuntimeError`. See [Double-Init Guard](../design/double-init-guard.md) for rationale.
