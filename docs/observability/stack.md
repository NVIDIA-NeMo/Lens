# Demo Stack (PoC)

This page describes the **demo** `docker-compose.otel.yml` included in the repo — a docker-compose stack for trying nemo-lens locally. It stands up one plausible observability pipeline (OTel Collector → Jaeger + Prometheus + Grafana + Elasticsearch + Kibana) so you can point an instrumented application at it and see spans, metrics, and logs without wiring up a full backend first. It is a proof of concept and a development convenience — not a reference architecture, not a supported deployment, and not something lens is opinionated about.

**nemo-lens itself does not provide an observability solution.** It's an instrumentation library: it emits OTLP. Where that OTLP goes, how long it's retained, how it's queried, and how it's visualised are the user's choices — driven by their organisation's existing observability investments and the scale of their workloads. The demo stack exists only to give you something to point at while you evaluate lens.

For production, see [Sending Telemetry to a Backend](backends.md) — the same OTLP stream can go to any compliant destination.

## What the demo stack includes

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
| OpenTelemetry Collector | 4317 / 4318 | OTLP receiver, fans out to storage |
| Jaeger | 16686 | Trace search UI |
| Prometheus | 9090 | Metric storage |
| Grafana | 3000 | Metric dashboards |
| Elasticsearch | 9200 | Trace + log storage |
| Kibana | 5601 | Log UI |
| DCGM Exporter | 9400 | GPU metrics (scrape target) |
| Node Exporter | 9100 | Host metrics (scrape target) |

This is a **lot of moving parts for a demo**. If all you need is trace viewing, you can run Jaeger alone. If you only care about metrics, Prometheus + Grafana is enough. Pick what matches your needs; the compose file is a starting point, not a prescription.

## Starting the demo

```bash
docker compose -f docker-compose.otel.yml up -d
```

Point an instrumented application at it:

```bash
export NEMO_LENS_ENABLED=1
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
```

UIs:

| Service | URL |
|---|---|
| Grafana | http://localhost:3000 (anonymous admin) |
| Jaeger | http://localhost:16686 |
| Prometheus | http://localhost:9090 |
| Kibana | http://localhost:5601 |

Stop it:

```bash
docker compose -f docker-compose.otel.yml down        # keep volumes
docker compose -f docker-compose.otel.yml down -v     # delete volumes too
```

## What this is good for

- **Trying lens locally** before deciding whether you want telemetry in your workflow at all.
- **Developing new instrumentation** — spans you add show up in Jaeger in seconds, so you can iterate on naming and attribute choices.
- **Debugging a specific issue** — run training against the local stack, reproduce the issue, inspect the trace.
- **Sharing a minimal reproduction** of an observability question — "I see this in Jaeger on the demo stack, is this what you expected?"

## What this is NOT good for

- **Production**. Nothing here is hardened, authenticated, or scaled. Jaeger in-memory storage will happily lose your traces on restart. Elasticsearch is a single node with no replication. Grafana has anonymous admin with no auth.
- **Long-running analysis**. Elasticsearch's default retention and Jaeger's in-memory store will exhaust disk / RAM on a multi-day training run.
- **Multi-user access**. Everything binds to `localhost`; extending it to a shared host is explicitly out of scope for the demo.
- **A recommendation for any specific backend**. Jaeger, Prometheus, and Grafana are in the demo because they're free to run locally, not because lens endorses them over alternatives.

## Where to go from here

Once you've confirmed lens is instrumenting what you expect, move to a real backend:

- **Managed / hosted**: [W&B Weave](backends.md#wb-weave), Grafana Cloud, Honeycomb, Datadog, New Relic, ... — see [Backends](backends.md#other-hosted-backends).
- **Self-hosted production**: an [OTel Collector](backends.md#otel-collector) fronting your chosen storage (Tempo + Mimir + Loki, Jaeger + Prometheus at scale, etc.).
- **Just a file for offline analysis**: see [File](backends.md#file).

Lens doesn't change between these. Point at a different `OTEL_EXPORTER_OTLP_ENDPOINT` and everything else stays the same.

## File layout of the demo

The compose file references configuration from the **consumer** repo (Megatron-LM in this case), because dashboards and collector config are domain-specific. In Megatron-LM:

```
docker-compose.otel.yml
observability/
├── otel-collector.yaml          — receiver + pipelines + exporters
├── prometheus.yml               — scrape config (collector, dcgm, node-exporter)
└── grafana/
    ├── provisioning/            — auto-wire Prometheus + Jaeger data sources
    └── dashboards/
        └── megatron-training.json  — training + inference dashboard
```

If you're using lens from a different consumer (NeMo-RL, NeMo-Gym, a fresh project), you'll need your own collector config and dashboards. The demo isn't trying to be a one-size-fits-all deployment — it's a worked example.

## What lens provides vs. what you provide

| Concern | Provided by lens | Provided by you |
|---|---|---|
| SDK initialisation | ✅ `setup_telemetry()` | — |
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

Lens stops at the OTLP boundary. Everything downstream is yours.
