# Instrumentation Primitives

Lens provides three span primitives. They cover the full spectrum from "cheap when off, gated by a frozenset lookup" to "always-on ergonomic context manager." Pick based on the call site.

## `managed_span` — group-gated context manager

```python
from nemo.lens import managed_span

with managed_span('step', 'train.step', iteration=42) as span:
    do_work()
    if span is not None:
        span.set_attribute('loss', loss_value)
```

### Behaviour

- When the `step` group is **disabled**: yields `None`, the body runs unchanged, **no span object is created**. The gating check is a `frozenset` lookup.
- When the `step` group is **enabled**: starts a span, sets attributes, attaches its context, yields the span. On exit, detaches context and ends the span. If the body raises, the exception is recorded on the span with `StatusCode.ERROR` before re-raising.

### When to use it

- Hot paths where you need configurable granularity (training steps, microbatches, communication ops).
- Any block where the cost must be minimised when telemetry is off.

### Tracer resolution

If `tracer=` is not passed, `managed_span` obtains a named tracer from the global `TracerProvider` using the instrumentation scope `nemo.lens.helpers` (the same default used by `span_cm`; note that `trace_fn` defaults to the `nemo.lens` scope instead). Passing `tracer=handle.tracer` (scope `nemo.lens`) skips this `get_tracer` call and makes spans share the handle's instrumentation scope.

## `trace_fn` — group-gated decorator

```python
from nemo.lens import trace_fn

@trace_fn('microbatch', 'train.microbatch.forward')
def forward_step(batch):
    ...
```

### Behaviour

Identical gating semantics to `managed_span`, but applied as a function decorator — no re-indentation of the function body. The span group is checked **at call time**, not at decoration time, so toggling groups dynamically works.

### When to use it

- Instrumenting existing functions without restructuring them.
- When the span name matches the function name (which is usually the case).

### Limitations

`trace_fn` cannot set span attributes from function arguments without a custom wrapper. For attribute-rich spans, use `managed_span` inside the function body instead.

## `span_cm` — simple ungated context manager

```python
from nemo.lens import span_cm

with span_cm('evaluate', tracer=handle.tracer, dataset='mmlu'):
    ...
```

### Behaviour

**Always** creates a span — no group gating. Attributes are set via `safe_set_span_attributes` (scalars and scalar-sequences only; non-scalars silently dropped).

### When to use it

- Code paths where you always want the span if telemetry is active (evaluation, setup, shutdown).
- Utility modules that don't know about span groups.
- Spans outside hot paths where the per-call cost of always creating a span doesn't matter.

### No-op interaction

When the global `TracerProvider` is no-op (non-exporting rank or telemetry disabled), `start_as_current_span` is still called but becomes a cheap no-op inside the OTel API — no exporter round-trip, no attribute processing cost. It doesn't skip the span object entirely the way `managed_span` does when its group is disabled, but the cost is small.

## `safe_set_span_attributes`

Utility for bulk-setting span attributes with sensible filtering and redaction:

```python
from nemo.lens import safe_set_span_attributes

safe_set_span_attributes(span, {
    'iteration': 42,
    'skipped': False,
    'loss': 1.23,
    'prompt': 'user input here',     # redacted to '[REDACTED]'
    'complex_obj': {...},             # silently dropped (not scalar)
})
```

### Rules

1. If `span.is_recording()` is `False`, the call is a no-op.
2. `None` values are silently skipped.
3. Non-scalar values (dicts, objects) are silently skipped. OTel attributes must be scalars or sequences of scalars.
4. Sequences of scalars are converted to `list`.
5. String values whose key matches a **redact key** are replaced with `'[REDACTED]'`.

### Default redact keys

```python
from nemo.lens import DEFAULT_REDACT_KEYS
# frozenset({'prompt', 'input_text', 'output_text', 'text',
#            'password', 'token', 'secret', 'key'})
```

Pass a custom `redact_keys` set to override.

### `redact_value`

`redact_value(key, value, redact_keys=DEFAULT_REDACT_KEYS)` is the single-value primitive that `safe_set_span_attributes` uses internally:

```python
from nemo.lens import redact_value, DEFAULT_REDACT_KEYS

redact_value('prompt', 'user input here')   # '[REDACTED]'  (key is in DEFAULT_REDACT_KEYS)
redact_value('iteration', 'user input here') # 'user input here' (key not redacted)
```

It returns `'[REDACTED]'` iff `key` is in `redact_keys`, otherwise it returns `value` unchanged. Redaction is decided by the **attribute-key name**, not by inspecting the value.

## `get_tracer` and `get_meter`

Both are top-level exports that return the globally registered tracer/meter from the active provider:

```python
from nemo.lens import get_tracer, get_meter

tracer = get_tracer()   # default instrumentation scope 'nemo.lens'
meter = get_meter()     # default instrumentation scope 'nemo.lens'
```

Each accepts an optional `name=` argument to set the instrumentation scope (default `'nemo.lens'`). Use these when you need a tracer or meter outside the span primitives — for example, to create custom metric instruments.

## Choosing between primitives

| Call frequency | Attribute-heavy? | Need group gating? | Use |
|---|---|---|---|
| Hot (per-microbatch) | Yes | Yes | `managed_span` |
| Hot (per-microbatch) | No | Yes | `trace_fn` |
| Cold (per-job, per-eval) | Yes | No | `span_cm` |
| Cold (per-job, per-eval) | No | No | `span_cm` |

## Checking group status before expensive prep

If building the attributes dict is itself expensive, gate it:

```python
from nemo.lens import is_span_group_enabled

if is_span_group_enabled('step'):
    attrs = build_expensive_attributes()
    with managed_span('step', 'train.step', **attrs) as span:
        ...
```

This saves the `build_expensive_attributes()` cost when `step` is disabled.
