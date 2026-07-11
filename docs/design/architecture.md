# Architecture

Lens is intentionally small. The design prioritises three properties:

1. **Cheap when disabled** — instrumented code should pay a trivial gating cost (a `frozenset` lookup) when telemetry is off, and never construct span objects.
2. **Optional dependency**: consumer libraries (such as Megatron-LM, NeMo RL, and NeMo Gym) can ship without bundling NeMo Lens.
3. **Clean layering** — each module has one responsibility and stable boundaries.

## Layer diagram

```
┌─────────────────────────────────────────────────────────────┐
│ Public API (__init__.py)                                    │
│ — re-exports everything consumers use                       │
└─────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│ Instrumentation layer                                        │
│ — helpers.py:  managed_span, trace_fn, span_cm              │
│ — distributed.py: broadcast_trace_context, create_linked_span│
│ — propagation.py: inject_context, extract_context           │
└─────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│ State + config layer                                         │
│ — config.py:  NemoLensConfig                                │
│ — state.py:   enabled span groups, PP trace carrier         │
│ — groups.py:  SpanGroup + presets                           │
└─────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│ Lifecycle + providers                                        │
│ — handle.py:     setup_telemetry, TelemetryHandle           │
│ — providers.py:  build_providers, build_noop_providers      │
│ — sampling.py:   RankAwareSampler                           │
└─────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│ Extras                                                        │
│ — instruments/  inference, rl, gym metrics                  │
│ — resources/    SLURM, K8s, local detection                 │
│ — contrib/      FastAPI, aiohttp, Ray, NCCL                 │
│ — logging_bridge.py                                          │
│ — fallbacks.py  canonical no-ops for consumers              │
│ — semconv.py    attribute name constants                    │
└─────────────────────────────────────────────────────────────┘
```

Each layer depends only on layers above it. No circular imports.

## Module responsibilities

| Module | Responsibility |
|---|---|
| `__init__.py` | Public API re-exports |
| `config.py` | `NemoLensConfig` dataclass, env var parsing, validation |
| `state.py` | Thread-safe global state: enabled span groups and PP trace carrier |
| `groups.py` | `SpanGroup` base class, preset resolution |
| `helpers.py` | `managed_span`, `trace_fn`, `span_cm`, attribute safety |
| `handle.py` | `setup_telemetry` entry point, `TelemetryHandle`, export strategy |
| `providers.py` | SDK provider construction (lazy imports) |
| `propagation.py` | W3C context inject/extract |
| `distributed.py` | `broadcast_trace_context`, `create_linked_span` |
| `sampling.py` | `RankAwareSampler` (OTel `Sampler` impl) |
| `semconv.py` | Attribute name constants and version tracking |
| `fallbacks.py` | Canonical no-op implementations for consumers |
| `logging_bridge.py` | Python logging → OTel LoggerProvider bridge |

## Call Flow for `setup_telemetry`

```
setup_telemetry(config, rank, world_size, resource_attributes)
  │
  ├─ if _INITIALIZED and config.enabled and not _allow_reinit:
  │     raise RuntimeError          # [double-init guard]
  │
  ├─ if not config.run_id:
  │     config.run_id = SLURM_JOB_ID or uuid4()
  │
  ├─ is_export_rank = _should_export(config, rank, world_size)
  │     └─ branches on config.export_strategy
  │
  ├─ if not config.enabled:
  │     build_noop_providers()
  │     set_enabled_span_groups(frozenset())
  │
  ├─ elif is_export_rank:
  │     build_providers(config, rank, world_size, resource_attributes, ...)
  │         ├─ imports SDK (primary SDK construction site)
  │         ├─ builds Resource with auto-detected + passed attributes
  │         ├─ builds TracerProvider (with optional RankAwareSampler)
  │         ├─ builds MeterProvider (with PeriodicExportingMetricReader)
  │         ├─ builds LoggerProvider (if config.logs_enabled)
  │         └─ sets W3C CompositePropagator
  │     set_enabled_span_groups(config.resolved_span_groups)
  │
  └─ else:    # non-exporting rank
        build_noop_providers()
        set_enabled_span_groups(frozenset())

  returns TelemetryHandle(tracer, meter, is_exporting)
```

## Call Flow for `managed_span`

```
with managed_span('step', 'train.step', iteration=42) as span:
  │
  ├─ is_span_group_enabled('step')   # frozenset lookup
  │
  ├─ if not enabled:
  │     yield None
  │     return                        # [disabled path: no span created]
  │
  ├─ tracer = tracer or trace.get_tracer(__name__)   # default tracer name "nemo.lens.helpers"
  ├─ span = tracer.start_span('train.step')
  ├─ safe_set_span_attributes(span, {'iteration': 42})
  ├─ token = context.attach(set_span_in_context(span))
  │
  ├─ try:
  │     yield span
  │   except Exception as exc:
  │     span.record_exception(exc)
  │     span.set_status(StatusCode.ERROR, str(exc))
  │     raise
  │   finally:
  │     context.detach(token)
  │     span.end()
```

The hot path when disabled is three Python statements: lookup, compare, yield `None`.

## Why this factoring

**State separated from logic** — `state.py` is just a frozenset and a lock. `managed_span` in `helpers.py` imports `is_span_group_enabled` but doesn't know how groups are stored. Swapping the state implementation (e.g. to add per-thread overrides) doesn't touch `helpers.py`.

**Config separated from providers** — `config.py` has no dependency on OTel SDK. A consumer can construct and validate `NemoLensConfig` in a process that doesn't have the SDK installed, then decide whether to initialise telemetry.

**Lazy SDK imports** — `providers.py` is the primary home for `opentelemetry.sdk.*` construction (a few other modules, such as `sampling.py` and `logging_bridge.py`, import the SDK lazily inside function bodies). Non-exporting ranks never execute `build_providers`, so they never incur the SDK import cost. On a large-rank job where most ranks don't export, avoiding that import on every non-exporting rank adds up.

**Public API minimal** — `__init__.py` exports only the documented public surface (the entries in `__all__`). Everything else is internal. Consumers can rely on the `__all__` list; we can refactor internals freely.

## Thread safety

Lens's global state is protected where it matters:

- `state._ENABLED_GROUPS` is written under a lock, read lock-free (a frozenset is immutable so readers see a consistent snapshot).
- `state._PP_TRACE_CARRIER` is written during `setup_telemetry` (single-threaded) and read from multiple threads in the pipeline schedule.
- Instrument caches use `WeakKeyDictionary`, which is thread-safe under CPython's GIL for the operations NeMo Lens performs.

`setup_telemetry` itself is not thread-safe — it's documented as "call once per process." The [double-init guard](double-init-guard.md) surfaces violations.

## What's out of scope

NeMo Lens is deliberately narrow. Log shipping beyond the OTel logging bridge remains with your existing log stack. Profiling CPU, memory, or GPU activity is the responsibility of the PyTorch profiler or `nsys`, not a tracing library. APM-style features such as service maps and error tracking are handled by whichever backend consumes the OTLP stream (such as Datadog, Sentry, or other compliant platforms).

The scope is also restricted to the NeMo ecosystem's training and inference workloads. The core primitives are domain-agnostic, but the opinionated components (span groups, metric instruments, and resource attributes) lean toward machine learning workloads. NeMo Lens does not serve as a general-purpose instrumentation library for unrelated domains.
