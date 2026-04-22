# Optional Dependency

Lens is designed so consumers (Megatron-LM, NeMo-RL, NeMo-Gym) can depend on it **optionally** — code runs whether or not lens is installed. This constraint shapes several design decisions.

## Why optional

- Consumers are foundational ML libraries. Their users shouldn't have to install an observability stack to run them.
- Not every deployment wants OTel instrumentation (local dev, batch inference, simple experiments).
- Lens's dependency footprint (`opentelemetry-api` minimum, full SDK optional) should be a user decision, not forced by a consumer.

## The pattern

Every consumer instrumentation site uses this import idiom:

```python
try:
    from nemo.lens.state import is_span_group_enabled as _otel_sg_enabled
    from nemo.lens.helpers import managed_span as _otel_managed_span
    from nemo.lens.helpers import trace_fn as _otel_trace_fn
except ImportError:
    from <project>.telemetry._fallbacks import is_span_group_enabled as _otel_sg_enabled
    from <project>.telemetry._fallbacks import managed_span as _otel_managed_span
    from <project>.telemetry._fallbacks import trace_fn as _otel_trace_fn
```

The instrumented code then uses the aliased names. When lens is installed → real implementations. When lens is absent → no-op fallbacks.

## `nemo.lens.fallbacks` — canonical no-ops

Lens ships `nemo.lens.fallbacks` with canonical no-op implementations of every consumer-facing function:

- `trace_fn(group, name)` → decorator that returns the function unchanged
- `managed_span(group, name, **kwargs)` → context manager yielding `None`
- `span_cm(name, **kwargs)` → context manager yielding `None`
- `is_span_group_enabled(group)` → always `False`
- `safe_set_span_attributes(span, attributes, redact_keys=None)` → no-op

When lens is installed, consumers can re-export these:

```python
# <project>/telemetry/_fallbacks.py
try:
    from nemo.lens.fallbacks import (
        is_span_group_enabled, managed_span, safe_set_span_attributes,
        span_cm, trace_fn,
    )
except ImportError:
    from contextlib import contextmanager

    def trace_fn(group, name, tracer=None):
        def decorator(func):
            return func
        return decorator

    @contextmanager
    def managed_span(group, name, tracer=None, **attributes):
        yield None

    # ... inline copies of the other no-ops ...
```

The nested `try/except` is needed because the consumer's `_fallbacks.py` itself imports from lens when possible. Only if lens is **completely absent** does it fall back to inline definitions.

## Why ship canonical no-ops

Before this module, each consumer maintained an identical copy of the no-op functions:

- Megatron-LM: `megatron/core/telemetry/_fallbacks.py`
- NeMo-RL: `nemo_rl/telemetry/_fallbacks.py`
- NeMo-Gym: `nemo_gym/telemetry/_fallbacks.py`

Three copies of the same file. Drift was inevitable — if lens added a new parameter to `managed_span`, all three copies needed updating separately.

Shipping canonical no-ops in `nemo.lens.fallbacks` eliminates the drift: consumer `_fallbacks.py` re-exports, the inline definitions serve only the "lens not installed" case. Both paths produce identical behaviour.

## Signature compatibility

The fallback signatures must match the real API exactly. If `managed_span` adds a new keyword argument, `fallbacks.py` must add it too (ignoring it is fine, since it's a no-op). Tests catch this — `lens/tests/test_fallbacks.py` exercises every fallback to verify signature compatibility.

## The boundary: what consumers can use without ImportError fallbacks

Anything under `nemo.lens.fallbacks` has a guaranteed no-op equivalent. Anything else doesn't.

Safe to use with fallbacks: `managed_span`, `trace_fn`, `span_cm`, `is_span_group_enabled`, `safe_set_span_attributes`.

Requires lens to be installed: `NemoLensConfig`, `setup_telemetry`, `TelemetryHandle`, `inject_context`, `extract_context`, `broadcast_trace_context`, `create_linked_span`, all of `contrib/`, all of `instruments/`.

For the latter group, consumers typically gate the entire setup behind `try/except ImportError`:

```python
try:
    from nemo.lens import NemoLensConfig, setup_telemetry
    config = NemoLensConfig.from_env(prefix='...', fallback_prefix='NEMO_LENS')
    handle = setup_telemetry(config, rank=rank, world_size=world_size)
except ImportError:
    handle = None
```

Instrumented code uses the `_otel_*` aliases from `_fallbacks.py` regardless of whether `handle` is `None` — so even without lens, the instrumentation compiles and runs as no-ops.

## What this costs

- **A bit of boilerplate** in consumer repos (the try/except import blocks).
- **Strict API compatibility** between lens's real implementations and `fallbacks.py`. This is enforced by tests.

What we get in return: consumers can honestly advertise "optional observability" and mean it.

## What this doesn't do

Being optional at the **import** level doesn't mean lens can be added to a running process dynamically. You still need to install it before the process starts. But consumers don't need to pin a lens version in their `pyproject.toml`, and CI can run their tests without lens installed to verify the fallbacks work.
