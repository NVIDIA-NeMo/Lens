# Overview

`nemo-lens` solves a narrow but important problem: giving NeMo-ecosystem training and inference workloads a **shared, idiomatic OpenTelemetry instrumentation layer** — cheap when disabled, ergonomic when enabled — that consumers can opt into without taking on a hard dependency.

## What it is

A thin, well-tested library that wraps the OpenTelemetry Python SDK with:

- A unified configuration object (`NemoLensConfig`) with prefix/fallback env var support
- Three instrumentation primitives designed for hot paths: `managed_span`, `trace_fn`, `span_cm`
- Span-group gating for granularity control (coarse for production, fine-grained for debugging)
- Rank-aware export strategies for distributed training (single rank, all ranks, sampled)
- Cross-rank trace context broadcast and span linking for pipeline-parallel correlation
- Resource auto-detection for SLURM, Kubernetes, and local environments
- Framework contrib modules for FastAPI, aiohttp, Ray, and NCCL

## What it isn't

- **Not a tracer.** OpenTelemetry SDK does the tracing. Lens configures it and provides ergonomic primitives.
- **Not an observability solution.** Lens emits OTLP and stops at that boundary. Choosing, running, securing, scaling, and retaining a backend is the user's decision — driven by their organisation's existing observability stack and the scale of their workloads. The `docker-compose.otel.yml` shipped with the repo is a **demo / PoC** to help you try lens locally, not a recommended production deployment.
- **Not a backend.** Spans and metrics export via standard OTLP to any compliant backend (Jaeger, Grafana Tempo, Honeycomb, Datadog, W&B Weave, etc.).
- **Not a requirement.** Consumers integrate via `try/except ImportError`. Lens ships canonical no-op fallbacks so instrumented code runs unchanged when lens isn't installed.

## Architectural principles

### Cheap when disabled

Every `managed_span` / `trace_fn` call checks `is_span_group_enabled(group)` before doing any real work. The check is a `frozenset` lookup. When the group is disabled, `managed_span` yields `None` and the body executes without creating any span objects.

### Lazy SDK imports

`opentelemetry-api` is the only required dependency (it ships a no-op implementation). The full SDK (`opentelemetry-sdk`, OTLP exporters) is imported only on **exporting ranks** — non-exporting ranks never pay the import cost.

### Single entry point

`setup_telemetry(config, rank, world_size)` is the only initialization call. It decides whether this rank exports, builds the right providers (real SDK or no-op), registers enabled span groups, and returns a `TelemetryHandle`.

### Rank-aware by default

Distributed training doesn't need every rank to export telemetry. The default `single_rank` strategy exports from one rank only (last rank by default). `all_ranks` and `sampled` strategies are available for specific use cases.

## Next steps

- [Install](installation.md) nemo-lens
- Follow the [quickstart](quickstart.md) to instrument a minimal script
- Read the [user guide](../user-guide/configuration.md) for details on each feature
