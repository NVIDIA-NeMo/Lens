# Custom Export Strategies

Lens picks which ranks export by name: `NemoLensConfig.export_strategy` selects a callable from a process-wide registry. The four built-ins cover the common cases (see [Sampling](sampling.md)); the registry is the extension point for everything else — fleet-aware sampling, locality-aware selection, anything that needs more context than the built-ins expose.

## Built-in strategies

Registered automatically at import time and exposed as the frozenset `nemo.lens.strategies.BUILTIN_STRATEGIES`:

- `single_rank` (default) — one rank exports (`config.export_rank`, `-1` means the last rank).
- `all_ranks` — every rank exports.
- `sampled` — deterministic hash-based fraction of ranks, controlled by `config.export_sample_rate`.
- `first_rank_per_node` — the rank with `LOCAL_RANK=0` on each node. Reads `LOCAL_RANK` from the environment (set by torchrun, deepspeed, etc.); a missing value is treated as `"0"`.

Built-ins cannot be unregistered, and replacing one requires `allow_override=True`.

## Selecting a strategy

By name, via the config or env var:

```python
from nemo.lens import NemoLensConfig, setup_telemetry

cfg = NemoLensConfig(enabled=True, export_strategy="first_rank_per_node")
handle = setup_telemetry(cfg, rank=rank, world_size=world_size)
```

```bash
NEMO_LENS_EXPORT_STRATEGY=first_rank_per_node
```

Or as an ad-hoc one-off, by passing the callable directly to `setup_telemetry`:

```python
handle = setup_telemetry(
    cfg,
    rank=rank,
    world_size=world_size,
    export_strategy=lambda config, rank, world_size: rank in {0, 7, 42},
)
```

The `export_strategy=` argument bypasses the registry — useful when you want a one-line strategy without polluting a process-wide name.

## Registering a custom strategy

```python
from nemo.lens import register_export_strategy, NemoLensConfig, setup_telemetry

def first_two_ranks(config, rank, world_size):
    return rank < 2

register_export_strategy("first_two_ranks", first_two_ranks)

cfg = NemoLensConfig(enabled=True, export_strategy="first_two_ranks")
handle = setup_telemetry(cfg, rank=rank, world_size=world_size)
```

Notes:

- Register **before** `setup_telemetry`. Validation is lazy: unknown names raise `ValueError` at `setup_telemetry` time, not at `NemoLensConfig` construction.
- The registry is process-wide. Call `register_export_strategy` exactly once per process (typically near your import-time setup).
- Pass `allow_override=True` only when you intentionally replace an existing entry. Built-ins additionally refuse silent replacement.

## Strategy callable signature

```python
ExportStrategy = Callable[[NemoLensConfig, int, int], bool]
```

A strategy receives the resolved `NemoLensConfig`, the global `rank`, and the `world_size`. It returns `True` if **this** rank should export, `False` otherwise. The decision runs once per process at `setup_telemetry` time.

Strategies may read environment variables for context the function arguments don't carry — `LOCAL_RANK`, `NODE_RANK`, `SLURM_PROCID`, anything launcher-specific. The built-in `first_rank_per_node` is the canonical example.

## Discovery and introspection

```python
from nemo.lens import registered_strategies, unregister_export_strategy

registered_strategies()
# ['all_ranks', 'first_rank_per_node', 'first_two_ranks', 'sampled', 'single_rank']

unregister_export_strategy("first_two_ranks")
```

`registered_strategies()` returns a sorted list of every name currently in the registry. `unregister_export_strategy(name)` removes a custom entry; it raises `ValueError` if `name` refers to a built-in or is not registered.

## Why this API

Name-based dispatch lets the same env var (`NEMO_LENS_EXPORT_STRATEGY`) route to user code without touching lens internals — production runs and CI can pick a strategy from outside the program. The `setup_telemetry(export_strategy=)` argument is the lower-level escape hatch: pass any callable, skip the registry, never name it. This parallels the `span_exporter=` argument documented in [Custom Exporters](custom-exporters.md) — both keep the core surface small while staying open at the bottom for callers that need it.
