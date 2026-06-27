# Double-Init Guard

`setup_telemetry` should be called **once per process**. Calling it twice was previously a silent failure: the OTel SDK logs a warning about provider override but otherwise carries on, producing subtly broken telemetry that is hard to diagnose.

As of the architectural fixes, a second call with `config.enabled=True` raises `RuntimeError`. This page explains why, how, and how to escape it if you really need to.

## The Problem

OTel SDK enforces a one-shot rule: `trace.set_tracer_provider(p)` only works if no provider has been set yet. Later calls log a warning and silently no-op. Same for `MeterProvider`.

Before the guard, this failure mode was invisible to lens callers:

```python
setup_telemetry(config)   # builds real providers, installs them
setup_telemetry(config)   # builds new providers, but they don't install.
                          # handle.tracer points at the NEW (uninstalled) provider's
                          # tracer — which is disconnected from the active one.
                          # Spans created via handle.tracer don't export; only spans
                          # created via trace.get_tracer() do.
```

The result: partial observability, no error, no easy diagnosis. Instrumentation sites that use `handle.tracer` silently produce no data.

## The Guard

NeMo Lens tracks whether `setup_telemetry` has succeeded with `config.enabled=True`:

```python
# nemo.lens.handle
_INITIALIZED = False

def setup_telemetry(config, ..., _allow_reinit=False):
    global _INITIALIZED
    if _INITIALIZED and config.enabled and not _allow_reinit:
        raise RuntimeError(
            "setup_telemetry() has already been initialised for this process. "
            "Call it once at startup. Pass _allow_reinit=True to override (testing only)."
        )
    ...
    if config.enabled:
        _INITIALIZED = True
```

The error is immediate and actionable. Callers learn about the mistake during development, not from a half-broken trace in production.

## When the Guard Does Not Fire

The guard does not fire when calling with a disabled configuration or during testing.


### Calling Twice with a Disabled Configuration

Calling `setup_telemetry(config_disabled)` twice is fine. No provider is installed either time, so there's no conflict. The `_INITIALIZED` flag stays `False`.

### Testing

Tests need to call `setup_telemetry` many times — each test wants a fresh state. Two options:

1. **Reset the flag in a fixture** (the NeMo Lens approach):

   ```python
   # tests/conftest.py
   @pytest.fixture(autouse=True)
   def reset_otel_providers():
       _reset_otel_globals()   # resets TracerProvider, MeterProvider, and _INITIALIZED
       yield
       _reset_otel_globals()
   ```

2. **Pass `_allow_reinit=True`** (for individual test cases):

   ```python
   def test_pipeline_parallel_simulation():
       for rank in range(4):
           handle = setup_telemetry(config, rank=rank, _allow_reinit=True)
           # ...
   ```

The leading underscore signals "internal, don't use in production code." Tests are the only legitimate use.

### Double Import

If your app imports a module that calls `setup_telemetry` twice due to a broken module system, you will hit the guard immediately. Fix the import; do not pass `_allow_reinit`.

## What to Do When You See the Error

First, find the second call. The stack trace points at it. Common causes:

- A test runner without the reset fixture.
- Two entry points that both call `setup_telemetry` (e.g., training script and inference server sharing a module).
- A main program that calls a library which calls `setup_telemetry` internally.

The fix depends on the cause:

- **Test runner**: add or repair the reset fixture.
- **Two entry points**: make one of them check whether telemetry is already initialized (expose a getter) and skip its call.
- **Library that initializes telemetry**: probably should not. Libraries should accept a `TelemetryHandle` (or nothing) and let the application decide whether to initialize.

Do not `try: setup_telemetry() except RuntimeError: pass`. That hides the real bug.

## Why This Is Not Configurable

The natural alternative would be a `force_reinit` public parameter rather than `_allow_reinit`. The underscore name signals that **this is not a supported workflow**.

Production code should call `setup_telemetry` exactly once at startup. If re-initialization is needed, there is likely a structural problem: a library is initializing something the application should own, or two subsystems are competing over global state.

The escape hatch exists for testing; making it public would encourage working around the bug instead of fixing it.

## Interaction with the Global OTel State

The guard tracks the NeMo Lens `_INITIALIZED` flag. The OTel SDK's own one-shot rule also still applies. Resetting `_INITIALIZED` in a test without also resetting the SDK's `_TRACER_PROVIDER_SET_ONCE` would leave the test with lens thinking it can re-init but the SDK refusing. The test would produce spans on a no-op provider and assertions would fail mysteriously.

The NeMo Lens `conftest.py` resets both. If you have a test infrastructure that needs to reset state, copy the pattern from there.
