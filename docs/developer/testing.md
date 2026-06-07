# Testing

The lens test suite is small and focused. Over 180 tests cover every public API, with a strong emphasis on:

- **State isolation** — tests that touch global OTel state reset it before and after.
- **No-op equivalence** — `fallbacks.py` behaviour must match the real API.
- **Config edge cases** — every env var combination and validation path.

## Running tests

```bash
# Full suite
pytest

# Single file
pytest tests/test_helpers.py

# Single test
pytest tests/test_helpers.py::TestManagedSpan::test_enabled_group_creates_span

# With coverage
pytest --cov=nemo.lens --cov-report=term-missing
```

## Layout

```
tests/
├── conftest.py                     — shared fixtures (state reset, InMemorySpanExporter)
├── test_config.py                  — NemoLensConfig, from_env, validation
├── test_state.py                   — is_span_group_enabled, thread safety
├── test_groups.py                  — SpanGroup.resolve, preset handling
├── test_helpers.py                 — managed_span, trace_fn, span_cm, attribute safety
├── test_handle.py                  — setup_telemetry, TelemetryHandle, double-init guard
├── test_providers.py               — build_providers, custom exporters
├── test_sampling_integration.py    — RankAwareSampler, integration with TracerProvider
├── test_strategies.py              — register/unregister/registered export strategies
├── test_distributed.py             — broadcast_trace_context, create_linked_span
├── test_propagation.py             — inject_context, extract_context
├── test_resources.py               — SLURM, K8s, local detection
├── test_instruments.py             — metric recording functions
├── test_fallbacks.py               — fallback no-op correctness
└── test_e2e.py                     — end-to-end with real SDK + InMemorySpanExporter
```

## Global state isolation

OTel SDK stores providers globally (`trace._TRACER_PROVIDER`, `metrics._METER_PROVIDER`). Lens stores enabled span groups globally. Tests that touch these must reset between runs, or test 2 inherits test 1's state.

`conftest.py` has three `autouse` fixtures:

```python
@pytest.fixture(autouse=True)
def reset_otel_providers():
    _reset_otel_globals()
    yield
    _reset_otel_globals()

@pytest.fixture(autouse=True)
def reset_span_groups():
    set_enabled_span_groups(frozenset())
    set_pp_trace_carrier(None)
    yield
    set_enabled_span_groups(frozenset())
    set_pp_trace_carrier(None)
```

Because `reset_span_groups` clears the enabled set before every test, a test that needs a group active must opt in explicitly — `set_enabled_span_groups(...)` (a top-level export from `nemo.lens`) is the escape hatch for enabling groups inside a test.

A third autouse fixture, `reset_strategy_registry`, snapshots `nemo.lens.strategies._REGISTRY` (under `_REGISTRY_LOCK`) before each test and restores it afterward, so custom export strategies registered via `register_export_strategy` do not leak between tests:

```python
@pytest.fixture(autouse=True)
def reset_strategy_registry():
    from nemo.lens.strategies import _REGISTRY, _REGISTRY_LOCK
    with _REGISTRY_LOCK:
        snapshot = dict(_REGISTRY)
    yield
    with _REGISTRY_LOCK:
        _REGISTRY.clear()
        _REGISTRY.update(snapshot)
```

`_reset_otel_globals()` resets five pieces of state:

```python
_trace_mod._TRACER_PROVIDER = None
_trace_mod._TRACER_PROVIDER_SET_ONCE = Once()
_metrics_mod._METER_PROVIDER = None
_metrics_mod._METER_PROVIDER_SET_ONCE = Once()
_handle_mod._INITIALIZED = False     # lens's own double-init flag
```

The `Once()` pointers are OTel SDK's internal "was this set?" flag. Without resetting them, `setup_telemetry` in a test would install a new provider but the SDK would log "provider already set" and silently use the previous one.

## Capturing spans

Tests that assert on span content use `InMemorySpanExporter` (shipped in `conftest.py`):

```python
def test_my_instrumentation():
    from tests.conftest import InMemorySpanExporter
    exporter = InMemorySpanExporter()

    cfg = NemoLensConfig(enabled=True, exporter="console")
    setup_telemetry(cfg, rank=0, world_size=1, span_exporter=exporter)

    with managed_span('job', 'my.op') as span:
        ...

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == 'my.op'
```

For metrics, use `InMemoryMetricReader` from the OTel SDK.

## Testing fallbacks

`tests/test_fallbacks.py` asserts that `nemo.lens.fallbacks` signatures match the real API and behave as no-ops. Whenever you add a parameter to `managed_span` or `trace_fn`, also add it to `fallbacks.py` and extend the test.

```python
def test_managed_span_accepts_kwargs():
    # Real API accepts arbitrary kwargs — fallback must too
    with managed_span("group", "name", iteration=1, loss=0.5) as span:
        assert span is None
```

## Testing double-init

```python
def test_double_init_raises():
    cfg = NemoLensConfig(enabled=True, exporter="console")
    setup_telemetry(cfg, rank=0, world_size=1)
    with pytest.raises(RuntimeError, match="already been initialised"):
        setup_telemetry(cfg, rank=0, world_size=1)
```

Tests that legitimately need to call `setup_telemetry` multiple times in one test (e.g. simulating multiple ranks) pass `_allow_reinit=True`:

```python
def test_all_ranks_export():
    cfg = NemoLensConfig(enabled=True, export_strategy="all_ranks", exporter="console")
    for rank in range(4):
        handle = setup_telemetry(cfg, rank=rank, world_size=4, _allow_reinit=True)
        assert handle.is_exporting
```

## Testing distributed helpers

`broadcast_trace_context` uses `torch.distributed`, which can't run in a single-process test without mocks. The distributed tests use `torch.distributed.init_process_group(backend='gloo', ...)` with a single rank — the broadcast becomes a no-op but the code path exercises correctly.

For genuinely multi-rank behaviour, tests would need to spawn subprocesses; currently the single-rank path plus manual carrier construction covers the contract.

## Testing the OTel interface of `RankAwareSampler`

```python
def test_sampler_returns_proper_sampling_result():
    from opentelemetry.sdk.trace.sampling import Decision
    sampler = RankAwareSampler(rank=0, world_size=4, sample_rate=1.0)
    result = sampler.should_sample(parent_context=None, trace_id=12345, name="test")
    assert result.decision == Decision.RECORD_AND_SAMPLE
```

The sampler is wrapped in a try/except inside `should_sample` to fall back to `bool` if the SDK isn't installed — covered by a separate test.

## Linting

```bash
ruff check src tests --fix
ruff format src tests
```

Pre-commit runs both (the `ruff` and `ruff-format` hooks in `.pre-commit-config.yaml`). CI runs `pre-commit run --all-files` and rejects PRs that fail it.

## What's NOT tested

- **Actual export to a collector**. That's the SDK's job; mocking it correctly is more work than value.
- **Long-running performance**. `tests/` exercises correctness, not throughput.
- **Integration with consumer libraries**. Those have their own test suites (`Megatron-LM/tests/unit_tests/telemetry/`, etc.).

When adding features that interact with a consumer, add a corresponding test in the consumer repo. Lens tests should stay self-contained.
