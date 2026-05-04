# nemo-lens Documentation

**nemo-lens** (`nemo.lens`) is a shared [OpenTelemetry](https://opentelemetry.io/) instrumentation library for the NVIDIA NeMo ecosystem. It provides unified tracing, metrics, and log bridging across distributed training, inference, and RL workloads.

## Key Features

* **Cheap when disabled** — span group gating is a `frozenset` lookup; no span objects are allocated when a group is off
* **Distributed-training aware** — rank-based export strategies, cross-rank trace correlation
* **Framework-agnostic primitives** — `managed_span`, `trace_fn`, `span_cm` work everywhere
* **OTel idiomatic** — real `TracerProvider` / `MeterProvider`, W3C propagation, standard semconv
* **Optional dependency** — consumers import via `try/except ImportError`; lens ships canonical no-op fallbacks
* **Pluggable** — custom exporters, custom samplers, custom span groups
* **Resource auto-detection** — SLURM, Kubernetes, local GPU count out of the box

## Consumers

`nemo-lens` is consumed by three NeMo-ecosystem projects, each extending it with domain-specific span groups:

| Consumer | Domain | Span group extension |
|---|---|---|
| [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | Transformer pre-training | `MegatronSpanGroup` — microbatch, communication, activation_offload, inference |
| NeMo-RL | RLHF, GRPO, DPO | `RLSpanGroup` — rollout, generation, logprob, reward, advantage, policy_update |
| NeMo-Gym | RL environments | `GymSpanGroup` — server, rollout_collection, verify, aggregate_metrics |

```{toctree}
:maxdepth: 2
:hidden:
:caption: Get Started

get-started/overview
get-started/installation
get-started/quickstart
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: User Guide

user-guide/configuration
user-guide/instrumentation
user-guide/span-groups
user-guide/metrics
user-guide/context-propagation
user-guide/distributed-tracing
user-guide/sampling
user-guide/custom-exporters
user-guide/custom-strategies
user-guide/resources
user-guide/contrib
user-guide/logging-bridge
user-guide/troubleshooting
user-guide/production
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Design

design/architecture
design/optional-dependency
design/double-init-guard
design/semconv
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Backends & Demo Stack

observability/backends
observability/stack
```

Lens emits OTLP; choosing and running an observability stack is up to you. Quick links for sending telemetry to common destinations:

- [Send telemetry to a file](observability/backends.md#file)
- [Send telemetry to W&B Weave](observability/backends.md#wb-weave)
- [Send telemetry to Honeycomb](observability/backends.md#honeycomb)
- [Send telemetry to an OTel Collector](observability/backends.md#otel-collector)

For trying things out locally, see the [demo docker-compose stack](observability/stack.md) (proof of concept, not for production).

```{toctree}
:maxdepth: 2
:hidden:
:caption: Developer Guide

developer/testing
developer/new-span-group
developer/contributing
developer/building-docs
```

% API reference (autodoc2-generated) — the apidocs/ dir is created at build time
% when autodoc2 is enabled. Set SKIP_AUTODOC=true to skip it.
```{toctree}
:maxdepth: 1
:hidden:
:caption: API Reference
:glob:

apidocs/index
```
