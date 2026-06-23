# Sending Telemetry to a Backend

**Lens does not provide or recommend an observability solution.** It emits OTLP. Where that OTLP goes, how it's stored, how it's queried, and how it's visualised are your decisions — shaped by your organisation's existing observability investments and the scale of your workloads.

This page shows how to wire lens up to common destinations. Lens exports via standard OTLP, so any OTLP-compatible backend works without code changes. Four destinations are covered in depth:

- [File](#file) — local trace/metric capture for offline analysis or archival
- [W&B Weave](#wb-weave) — Weights & Biases' trace UI, co-located with training run metadata
- [Honeycomb](#honeycomb) — hosted APM that accepts all three signals on one OTLP endpoint
- [OTel Collector](#otel-collector) — a routing / aggregation layer in front of other backends

Plus a quick reference for other hosted backends.

---

## File

Writing traces to a local file is useful for offline analysis, CI captures, and archival. Three approaches, in order of simplicity:

### Approach 1: Console exporter + shell redirect (simplest)

```bash
export NEMO_LENS_ENABLED=1
export NEMO_LENS_EXPORTER=console

python train.py > traces.jsonl 2>&1
```

`NEMO_LENS_EXPORTER=console` installs `ConsoleSpanExporter`, which writes one JSON line per span to stdout. Redirect stdout to a file and you have a span log.

**Drawbacks:**

- Mixes application stdout with span data — separate them with selective logging to stderr.
- Doesn't capture metrics (the metric exporter writes a different format).

### Approach 2: Custom `ConsoleSpanExporter` pointing at a file handle

```python
from pathlib import Path
from opentelemetry.sdk.trace.export import ConsoleSpanExporter
from nemo.lens import NemoLensConfig, setup_telemetry

trace_file = Path("traces.jsonl").open("a", buffering=1)   # line-buffered
exporter = ConsoleSpanExporter(out=trace_file)

config = NemoLensConfig(enabled=True, exporter="console")  # falls back if custom skipped
handle = setup_telemetry(
    config,
    rank=0,
    world_size=1,
    span_exporter=exporter,
)

try:
    # ... your workload ...
finally:
    handle.shutdown()
    trace_file.close()
```

`ConsoleSpanExporter` accepts any file-like object via `out=`. This separates trace data from application stdout without the shell-redirect hack.

**Caveats:**

- Line-buffer (`buffering=1`) so lines aren't lost if the process crashes.
- Remember to close the file after `handle.shutdown()`.
- Each line is a Python `repr` of the span, not strict JSON. For strict JSON, write a custom exporter (next approach).

### Approach 3: Custom `SpanExporter` (full control)

For structured JSON, compression, rotation, or any custom format:

```python
import json
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

class JSONLFileSpanExporter(SpanExporter):
    def __init__(self, path: str):
        self._fh = open(path, "a", buffering=1)

    def export(self, spans) -> SpanExportResult:
        try:
            for span in spans:
                record = {
                    "name": span.name,
                    "trace_id": f"{span.context.trace_id:032x}",
                    "span_id": f"{span.context.span_id:016x}",
                    "parent_id": (
                        f"{span.parent.span_id:016x}" if span.parent else None
                    ),
                    "start_time_unix_nano": span.start_time,
                    "end_time_unix_nano": span.end_time,
                    "attributes": dict(span.attributes or {}),
                    "status": str(span.status.status_code),
                }
                self._fh.write(json.dumps(record) + "\n")
            return SpanExportResult.SUCCESS
        except Exception:
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        self._fh.close()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        self._fh.flush()
        return True


handle = setup_telemetry(
    config,
    span_exporter=JSONLFileSpanExporter("traces.jsonl"),
)
```

This gives you strict JSONL that's trivial to `jq` over. Extend with gzip, rotation, or remote write as needed.

### Approach 4: OTel Collector file exporter

If you're already running an OTel Collector (see below), add a `file` exporter to its pipeline:

```yaml
# otel-collector.yaml
exporters:
  file/traces:
    path: /var/log/otel/traces.jsonl
    rotation:
      max_megabytes: 100
      max_days: 7

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [file/traces, jaeger]   # fan out to both
```

Your application still exports OTLP as normal; the Collector handles file writes, rotation, and retention.

**Use this** when you want a single file with spans from multiple ranks or multiple services.

### Metrics to file

For metrics, use a `PeriodicExportingMetricReader` with `ConsoleMetricExporter`:

```python
from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader

metric_file = open("metrics.jsonl", "a", buffering=1)
reader = PeriodicExportingMetricReader(
    ConsoleMetricExporter(out=metric_file),
    export_interval_millis=10000,
)

handle = setup_telemetry(config, metric_reader=reader)
```

Or use the Collector's file exporter in its metrics pipeline (same pattern as traces).

---

## W&B Weave

[Weave](https://wandb.ai/site/weave) is Weights & Biases' trace visualisation tool. It ingests OTLP spans and renders them in the same UI as your W&B training runs — so traces and training metrics live together.

### Configure

Two patterns work. Pick one.

#### Pattern A: direct from app (no collector)

```bash
# Required: W&B identification (set as resource attributes)
export WANDB_ENTITY=my-team                    # or your personal entity
export WANDB_PROJECT=megatron-training

# Required: traces-signal-specific endpoint + auth header
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=https://trace.wandb.ai/otel/v1/traces
export OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf
export OTEL_EXPORTER_OTLP_TRACES_HEADERS="wandb-api-key=$WANDB_API_KEY"

# Weave ingests traces only; disable the other two signals.
export MEGATRON_OTEL_METRICS_ENABLED=0
export NEMO_LENS_LOGS_ENABLED=0

# Required: lens activation
export NEMO_LENS_ENABLED=1
```

A ready-to-run compose file for this pattern is at `docker-compose.weave.yml` — brings up only the Megatron container and points traces straight at Weave. Use when you don't want to run the local stack.

#### Pattern B: through a collector

Useful when you want batching, filtering, or multi-backend fan-out. See `observability/otel-collector-weave.yaml` in the repo for a ready-to-run example, toggled via `docker-compose.otel.yml`'s `--config=/etc/otel/collector-weave.yaml` mode.

### Notes on the direct path

- Weave currently ingests **traces only** (as of early 2026). Metrics still need a separate sink (Prometheus, OTel Collector, native `wandb.log()`, etc.).
- Use the `OTEL_EXPORTER_OTLP_TRACES_*` variants (not the signal-agnostic `OTEL_EXPORTER_OTLP_*`). The Weave URL is a full path ending in `/v1/traces`; signal-specific env vars are treated as full URLs, the generic variant appends `/v1/traces` automatically — setting both would produce `/v1/traces/v1/traces` and 404.
- Lens honours `OTEL_EXPORTER_OTLP_PROTOCOL` and signal-specific variants, so `http/protobuf` routes to the HTTP exporter class (Weave is HTTP-only).
- `NemoLensConfig.from_env()` reads `WANDB_ENTITY` and `WANDB_PROJECT` directly and sets them as `wandb.entity` / `wandb.project` resource attributes on every span — required for Weave to route correctly.

### Run

```bash
python pretrain_gpt.py ...
```

Traces appear in the Weave tab of your W&B run within a few seconds. The trace tree mirrors Jaeger's structure: a `megatron.train_step` root span with child spans for forward_backward, optimizer, etc.

### Linking traces to runs

Because `WANDB_ENTITY` / `WANDB_PROJECT` are set as span attributes, Weave automatically associates traces with the right W&B run. The `nemo.run.id` resource attribute (auto-generated or from `SLURM_JOB_ID`) serves as a unique run identifier you can filter on in the Weave UI.

### Sampling for cost

W&B bills by ingested trace volume. For long `per_step` runs, sample aggressively:

```bash
export OTEL_TRACES_SAMPLER=parentbased_traceidratio
export OTEL_TRACES_SAMPLER_ARG=0.01    # keep 1% of traces
```

See [sampling](../user-guide/sampling.md) for how this composes with lens's export strategies.

### What gets sent

Everything the SDK exports — span names, attributes, events, links, status. Weave renders:

- Span waterfall timelines
- Attribute key/value pairs
- Error events (via `span.record_exception`)
- OTel Links as clickable references (useful for pipeline-parallel correlation)

---

## Honeycomb

[Honeycomb](https://honeycomb.io) is a hosted APM that ingests OpenTelemetry data natively. Unlike Weave, it accepts all three signals — traces, metrics, and logs — on a single OTLP endpoint. Good fit if you want one hosted destination for everything and already have (or are happy to adopt) Honeycomb's query model.

### Configure

Two patterns work. Pick one.

#### Pattern A: direct from app (no collector)

```bash
# One endpoint covers all three signals.
export OTEL_EXPORTER_OTLP_ENDPOINT=https://api.honeycomb.io:443
export OTEL_EXPORTER_OTLP_HEADERS="x-honeycomb-team=${HONEYCOMB_API_KEY},x-honeycomb-dataset=${HONEYCOMB_DATASET}"
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf

export NEMO_LENS_ENABLED=1
```

- **`x-honeycomb-team`**: your ingest API key. Find it in the Honeycomb UI under **Environment settings → API Keys**.
- **`x-honeycomb-dataset`**: dataset name. Required for metrics; optional but recommended for traces and logs. Pick anything meaningful; Honeycomb auto-creates the dataset on first write.
- **`OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf`**: Honeycomb supports both gRPC and HTTP, but HTTP is more forgiving behind load balancers. Default to HTTP unless you have a reason otherwise.

Lens honours `OTEL_EXPORTER_OTLP_PROTOCOL` (and the signal-specific variants `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL`, `OTEL_EXPORTER_OTLP_METRICS_PROTOCOL`) when picking between gRPC and HTTP exporters, so this works without code changes.

For EU instance, substitute `https://api.eu1.honeycomb.io:443`.

A ready-to-run compose file for this pattern is at `docker-compose.honeycomb.yml` — brings up only the Megatron container and points it straight at Honeycomb. Use when you don't want to run the local stack.

#### Pattern B: through a collector

Useful when you want batching, filtering, or multi-backend fan-out between the app and Honeycomb. See [collector-honeycomb.yaml](../../observability/otel-collector-honeycomb.yaml) in the repo for a ready-to-run example:

```yaml
exporters:
  otlphttp/honeycomb:
    endpoint: https://api.honeycomb.io:443
    headers:
      x-honeycomb-team: ${env:HONEYCOMB_API_KEY}
      x-honeycomb-dataset: ${env:HONEYCOMB_DATASET}

service:
  pipelines:
    traces:   { receivers: [otlp], processors: [batch], exporters: [otlphttp/honeycomb] }
    metrics:  { receivers: [otlp], processors: [batch], exporters: [otlphttp/honeycomb] }
    logs:     { receivers: [otlp], processors: [batch], exporters: [otlphttp/honeycomb] }
```

The application then points at your collector (`OTEL_EXPORTER_OTLP_ENDPOINT=http://collector:4317`), and the collector handles Honeycomb auth and routing.

The repo's `docker-compose.otel.yml` has a one-line toggle for this: uncomment `--config=/etc/otel/collector-honeycomb.yaml` and set `HONEYCOMB_API_KEY` / `HONEYCOMB_DATASET` in `.env`.

### Classic vs current Honeycomb

Honeycomb migrated from dataset-per-service (Classic) to environment-based organisation. If you're on a Classic account, the `x-honeycomb-dataset` header is required for every signal, and the dataset field has specific semantics. For current Honeycomb it's still required for metrics and optional-but-recommended for traces/logs. If you're unsure which you have, your account page will tell you.

### Sampling

Honeycomb bills on event volume. A `per_step` Megatron run on many ranks will ship a lot of events. Layer your sampling:

1. Lens `export_strategy` — rank-level (start with `single_rank`).
2. OTel SDK `OTEL_TRACES_SAMPLER=parentbased_traceidratio` — per-trace.
3. Honeycomb **Refinery** — tail sampling with access to the full trace before deciding. Recommended for production; see [Honeycomb's Refinery docs](https://docs.honeycomb.io/manage-data-volume/refinery/).

### What gets sent

Every attribute, event, and link the SDK exports. Honeycomb's UI is especially good at high-cardinality attribute queries (`BubbleUp`, `HEATMAP`, etc.), so set span attributes liberally — attribute cardinality is what Honeycomb is best at.

---

## OTel Collector

The OpenTelemetry Collector is a common intermediary between your application and your observability backends. Running a Collector (vs. exporting directly from the SDK) can give you:

- **Fan-out**: send the same telemetry to multiple backends (Jaeger + Prometheus + S3 archive).
- **Sampling and filtering**: drop spans at the Collector rather than in every SDK instance.
- **Batching and resilience**: buffer during network outages without losing data.
- **Transforms**: rename attributes, redact PII, enrich with external metadata.
- **Centralised config**: change backends without restarting training jobs.

### Minimum configuration

```yaml
# otel-collector.yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

processors:
  batch:
    timeout: 5s
    send_batch_size: 1024

exporters:
  # Traces to Jaeger
  otlp/jaeger:
    endpoint: jaeger:4317
    tls:
      insecure: true

  # Metrics to Prometheus (pull model)
  prometheus:
    endpoint: 0.0.0.0:8889

  # Optional: file archival
  file:
    path: /var/log/otel/telemetry.jsonl

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [otlp/jaeger, file]

    metrics:
      receivers: [otlp]
      processors: [batch]
      exporters: [prometheus]
```

### Run the Collector

Docker (simplest):

```bash
docker run --rm \
  -p 4317:4317 -p 4318:4318 -p 8889:8889 \
  -v $(pwd)/otel-collector.yaml:/etc/otel-collector.yaml \
  otel/opentelemetry-collector-contrib:latest \
  --config=/etc/otel-collector.yaml
```

This is a generic standalone example. The repo's bundled configs live under `observability/` and are mounted at `/etc/otel/collector*.yaml` by `docker-compose.otel.yml` (selected via the `--config` line) — adjust the path if you copy from there.

On a cluster, deploy as a sidecar, DaemonSet, or shared service. Typical patterns:

- **Sidecar**: one Collector per application pod. Low latency, isolated failure domain.
- **DaemonSet**: one Collector per host, every local app exports to it. Good for Kubernetes.
- **Shared service**: one fleet of Collectors behind a load balancer. Cheapest, but adds a hop.

### Configure the application

```bash
export NEMO_LENS_ENABLED=1
export OTEL_EXPORTER_OTLP_ENDPOINT=http://collector.internal:4317
```

That's it. Lens discovers the endpoint from the standard env var; no code changes.

### Useful processors

Beyond `batch`, consider:

```yaml
processors:
  # Drop noisy spans at the Collector
  filter/drop_health_checks:
    traces:
      span:
        - 'name == "GET /healthz"'

  # Sample smart: keep errors, sample 10% of successes
  tail_sampling:
    decision_wait: 10s
    policies:
      - name: keep-errors
        type: status_code
        status_code: { status_codes: [ERROR] }
      - name: sample-successes
        type: probabilistic
        probabilistic: { sampling_percentage: 10 }

  # Enrich with cluster metadata
  resource:
    attributes:
      - key: cluster.name
        value: prod-us-west
        action: upsert

  # Redact sensitive attributes
  attributes/redact:
    actions:
      - key: user.email
        action: delete
```

Attach to a pipeline:

```yaml
service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [filter/drop_health_checks, tail_sampling, resource, batch]
      exporters: [otlp/jaeger, file]
```

### Multi-backend routing

Send traces to two places simultaneously — e.g. Jaeger for interactive debugging and W&B Weave for run history:

```yaml
exporters:
  otlp/jaeger:
    endpoint: jaeger:4317
    tls: { insecure: true }

  otlphttp/wandb:
    endpoint: https://trace.wandb.ai/otel/v1/traces
    headers:
      wandb-api-key: ${env:WANDB_API_KEY}

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [otlp/jaeger, otlphttp/wandb]
```

Your application exports to one endpoint (the Collector); the Collector fans out.

### Collector-side sampling

Instead of sampling at the SDK (via `OTEL_TRACES_SAMPLER`), sample at the Collector. Advantage: you can make the decision based on the complete trace (e.g. keep all traces containing an error), which the SDK can't do because it hasn't seen the whole trace yet.

The `tail_sampling` processor is the standard tool. See the full [tail sampling docs](https://github.com/open-telemetry/opentelemetry-collector-contrib/tree/main/processor/tailsamplingprocessor).

### Production considerations

- **Backpressure**: if a backend is slow, the Collector buffers. Configure `sending_queue` limits to cap memory.
- **TLS**: enable TLS between SDK and Collector, and between Collector and backends, in any multi-tenant setup.
- **Health checks**: enable the `health_check` extension (as the bundled configs do) to expose `:13133/`. Monitor it.
- **Version pinning**: the `opentelemetry-collector-contrib` image changes — pin to a version and upgrade deliberately.

### Debugging the Collector

```bash
# Increase logging
--set service.telemetry.logs.level=debug

# Enable the debug exporter to see spans on stdout
exporters:
  debug:
    verbosity: detailed

# Add to a pipeline for testing
service:
  pipelines:
    traces:
      exporters: [debug]
```

The Collector's own telemetry (`:8888/metrics`) shows incoming span rates, processor queue depth, and exporter success counts — scrape it with Prometheus to monitor the monitoring.

---

## Other hosted backends

### Grafana Cloud

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=https://otlp-gateway-<region>.grafana.net/otlp
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Basic <base64-encoded-instance-id-and-token>"
```

Traces → Tempo, metrics → Mimir, logs → Loki — queryable from a unified Grafana UI.

### Datadog

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=https://trace.agent.datadoghq.com
export OTEL_EXPORTER_OTLP_HEADERS="DD-API-KEY=<your-api-key>"
```

Datadog also ships their own Collector preset; see their docs for advanced config.

### New Relic

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=https://otlp.nr-data.net
export OTEL_EXPORTER_OTLP_HEADERS="api-key=<your-ingest-key>"
```

### Self-hosted Jaeger or Tempo

Both accept OTLP natively:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger.internal:4317    # or tempo.internal
```

---

## Choosing between them

| Factor | Prefer |
|---|---|
| Quick local iteration | [Console / File](#file) |
| Small team, no infra team | Hosted ([W&B Weave](#wb-weave), [Honeycomb](#honeycomb), Grafana Cloud) |
| Training runs tied to W&B | [W&B Weave](#wb-weave) |
| One hosted destination for all three signals | [Honeycomb](#honeycomb) |
| High-cardinality attribute queries matter | [Honeycomb](#honeycomb) |
| Production with multi-backend routing | [OTel Collector](#otel-collector) + your chosen backends |
| Data residency / compliance | Self-hosted Collector + self-hosted backends |
| Already have one APM vendor | Their OTLP endpoint |

All destinations work the same from lens's perspective — the choice is about cost, operational burden, and integration with your existing stack.

## Partitioning by run

Regardless of backend, filter by `nemo.run.id` (auto-set by lens) to isolate a specific training run's data:

- **Jaeger**: tag filter `nemo.run.id=<value>`
- **Grafana**: dashboard variable `nemo_run_id`
- **Honeycomb**: filter `nemo.run.id`
- **Datadog**: facet `@nemo.run.id`
- **Weave**: run-level association via `WANDB_ENTITY` + `WANDB_PROJECT`

Multiple runs land in the same index/project; the attribute is the partition key.

## gRPC vs HTTP

OTLP has two transport variants:

```bash
# gRPC (default, faster, persistent connections)
OTEL_EXPORTER_OTLP_PROTOCOL=grpc
OTEL_EXPORTER_OTLP_ENDPOINT=http://collector:4317

# HTTP/Protobuf (firewall-friendly, HTTPS works easily)
OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
OTEL_EXPORTER_OTLP_ENDPOINT=http://collector:4318
```

Lens's `providers.py` tries gRPC first, falls back to HTTP if the gRPC exporter isn't installed. If you only installed `opentelemetry-exporter-otlp-proto-http`, set the protocol explicitly.

## Don't send everything

A `per_step` run on 1,000 ranks can produce 100k+ spans/second. Most backends can't (or won't affordably) ingest that. Layer your sampling:

1. **Export strategy** (rank level): `single_rank` sends one rank's data. Usually the right starting point.
2. **OTel SDK sampler** (trace level): `OTEL_TRACES_SAMPLER=parentbased_traceidratio` with `OTEL_TRACES_SAMPLER_ARG=0.1` keeps 10% of traces.
3. **Collector tail sampling** (smart): keep all errors, sample 1% of successes.

Combine aggressively. It's easier to re-enable telemetry when you're debugging than to pay for ingestion nobody looks at.
