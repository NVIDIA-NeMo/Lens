# Demo Stack

This page describes the **demo** `docker-compose.otel.yml` included in the repository, a Docker Compose stack for trying `nemo-lens` locally. It stands up one plausible observability pipeline (OTel Collector to Jaeger, Prometheus, Grafana, Elasticsearch, and Kibana) so you can point an instrumented application to it and see spans, metrics, and logs without wiring up a full backend first. It is a proof of concept and a development convenience, not a reference architecture, not a supported deployment, and not something NeMo Lens prescribes.

**NeMo Lens does not provide an observability solution.** It is an instrumentation library: it emits OTLP. Where that OTLP goes, how long it is retained, how it is queried, and how it is visualized are your choices, which are driven by your organization's existing observability investments and the scale of your workloads. The demo stack exists only to give you a destination while you evaluate NeMo Lens.

For production, see [Send Telemetry to a Backend](backends.md). The same OTLP stream can go to any compliant destination.

## What the Demo Stack Includes

```
Your application
  │ OTLP (:4317 gRPC or :4318 HTTP)
  ▼
┌─────────────────────────┐
│ OpenTelemetry Collector │
└─────────────────────────┘
  ├─ traces  ─► Jaeger + Elasticsearch
  ├─ metrics ─► Prometheus + Elasticsearch
  └─ logs    ─► Elasticsearch ─► Kibana

Prometheus ─► Grafana (dashboards)
```

| Component | Port | Role in the demo |
|---|---|---|
| OpenTelemetry Collector | 4317 / 4318 | OTLP receiver, fans out to storage (internal network only, not published to host) |
| Jaeger | 16686 | Trace search UI |
| Prometheus | 9090 | Metric storage |
| Grafana | 3000 | Metric dashboards |
| Elasticsearch | 9200 | Trace and log storage |
| Kibana | 5601 | Log UI |
| DCGM Exporter | 9400 | GPU metrics (internal scrape target, not published to host) |
| Node Exporter | 9100 | Host metrics (internal scrape target, not published to host) |

This setup includes many components for a demo environment. If you only need to view traces, you can run Jaeger alone. If you only need to analyze metrics, Prometheus and Grafana are sufficient. Select the services that match your requirements; the Docker Compose file is a starting point rather than a strict requirement.

## Start the Demo

```{note}
The committed `docker-compose.otel.yml` ships with the **W&B Weave** collector mode active. For the local Jaeger, Prometheus, Grafana, Elasticsearch, and Kibana pipeline described here, edit the `otel-collector` service in `docker-compose.otel.yml`: uncomment `command: ["--config=/etc/otel/collector.yaml"]` and comment out the `--config=/etc/otel/collector-weave.yaml` line. With Weave active, traces go to W&B Weave, and metrics and logs are accepted and dropped, so the local UIs below stay empty.
```

```bash
docker compose -f docker-compose.otel.yml up -d
```

The compose file is built around running an instrumented application **inside** the compose network. The bundled Megatron-LM container already exports to the collector over the internal network (`OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317`):

```bash
docker compose -f docker-compose.otel.yml exec megatron bash
```

The collector's OTLP ports (4317/4318) are not published to the host. To export from a host-side application instead, add `"4317:4317"` and `"4318:4318"` to the `otel-collector` `ports:` block, then point your app at it:

```bash
export NEMO_LENS_ENABLED=1
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
```

Access the UIs:

| Service | URL |
|---|---|
| Grafana | http://localhost:3000 (anonymous admin) |
| Jaeger | http://localhost:16686 |
| Prometheus | http://localhost:9090 |
| Kibana | http://localhost:5601 |

Stop the demo:

```bash
docker compose -f docker-compose.otel.yml down        # keep volumes
docker compose -f docker-compose.otel.yml down -v     # delete volumes too
```

## Intended Benefits

- **Local evaluation.** Decide whether you want telemetry in your workflow.
- **Instrumentation development.** Spans you add show up in Jaeger in seconds, so you can iterate on naming and attribute choices.
- **Issue troubleshooting.** Run training against the local stack, reproduce the issue, and inspect the trace.
- **Observability reproduction.** Share an observability question, such as "I see this in Jaeger on the demo stack. Is this what you expected?"

## Design Limitations

- **Production**. Nothing here is hardened, authenticated, or scaled. Jaeger in-memory storage will happily lose your traces on restart. Elasticsearch is a single node with no replication. Grafana has anonymous admin with no authentication.
- **Long-running analysis**. Elasticsearch's default retention and Jaeger's in-memory store will exhaust disk or RAM on a multi-day training run.
- **Multi-user access**. Everything binds to `localhost`; extending it to a shared host is explicitly out of scope for the demo.
- **A recommendation for any specific backend**. Jaeger, Prometheus, and Grafana are in the demo because they are free to run locally, not because NeMo Lens endorses them over alternatives.

## Choose a Production Backend

After you confirm that NeMo Lens is instrumenting what you expect, move to a real backend:

- **Managed or hosted.** Use [W&B Weave](backends.md#integrate-with-wb-weave), Grafana Cloud, Honeycomb, Datadog, New Relic, or another hosted backend. See [Send Telemetry to Other Hosted Backends](backends.md#send-telemetry-to-other-hosted-backends).
- **Self-hosted production.** Use an [OTel Collector](backends.md#configure-otel-collector) in front of your chosen storage, such as Tempo, Mimir, Loki, Jaeger, or Prometheus at scale.
- **File export for offline analysis.** See [Export Telemetry to a File](backends.md#export-telemetry-to-a-file).

NeMo Lens does not change between these options. Point to a different `OTEL_EXPORTER_OTLP_ENDPOINT`, and everything else stays the same.

## Demo File Layout

The compose file mounts its observability configuration from this NeMo Lens repository under `observability/`. Only the application container (Megatron-LM in this case) is built and mounted from the parent directory. In `lens/`:

```
docker-compose.otel.yml
observability/
├── jaeger.yaml                   — Jaeger v2 config (native OTLP)
├── otel-collector.yaml           — local-stack receiver + pipelines + exporters
├── otel-collector-file.yaml      — file-export mode
├── otel-collector-weave.yaml     — W&B Weave mode (default in shipped compose)
├── otel-collector-honeycomb.yaml — Honeycomb mode
├── prometheus.yml                — scrape config (collector, dcgm, node-exporter)
├── kibana/                       — kibana.yml + setup.sh + saved-objects.ndjson
└── grafana/
    ├── provisioning/             — auto-wire Prometheus + Jaeger + Elasticsearch data sources
    └── dashboards/
        ├── megatron-training.json — training + inference dashboard
        └── system-overview.json   — host/GPU/network overview dashboard
```

If you are using NeMo Lens from a different consumer (NeMo-RL, NeMo-Gym, or a fresh project), you need your own collector configuration and dashboards. The demo is not trying to be a one-size-fits-all deployment; it is a worked example.

## Compare What NeMo Lens Provides and What You Provide

| Concern | Provided by NeMo Lens | Provided by you |
|---|---|---|
| SDK initialization | ✅ `setup_telemetry()` | — |
| Instrumentation primitives | ✅ `managed_span`, `trace_fn`, `span_cm` | — |
| OTLP wire format | ✅ (via OTel SDK) | — |
| Resource detection | ✅ SLURM, K8s, local | Any custom attributes |
| Example stack | Demo `docker-compose.otel.yml` | — |
| Storage backend | — | Jaeger / Tempo / Elasticsearch / hosted / ... |
| Dashboards | — | Grafana JSON / hosted dashboards |
| Alerting rules | — | Prometheus alerting / PagerDuty / ... |
| Retention policies | — | Configured at the backend |
| Access control | — | Configured at the backend |
| Sampling strategy | Hooks and primitives | Business policy (keep errors? cost budget?) |

NeMo Lens stops at the OTLP boundary. Everything downstream is yours.
