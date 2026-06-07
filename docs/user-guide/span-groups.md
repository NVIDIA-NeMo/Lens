# Span Groups

Span groups are lens's mechanism for controlling trace granularity at runtime without code changes. Every instrumentation site tags itself with a group name; at startup, only the enabled groups actually emit spans.

## Why groups

A training job might want:
- `default` in production: only coarse spans (job, checkpoint, evaluate). Lowest cost.
- `per_step` in staging: adds per-iteration boundaries. Moderate cost.
- `all` when debugging a specific hang: every instrumented site, including per-microbatch and per-layer. Highest cost.

Toggling this via one env var — with no code changes and a cheap gating check when disabled — is the goal.

## Base groups

`SpanGroup` ships with eight groups covering typical training workflows:

### Coarse-grained (in `default` preset)

| Group | Typical spans | Frequency |
|---|---|---|
| `job` | Pretrain/train root spans | once per job |
| `checkpoint` | Checkpoint save | every N iterations |
| `evaluate` | Evaluation pass | every N iterations |

### Medium-grained (in `per_step` preset)

| Group | Typical spans | Frequency |
|---|---|---|
| `model_init` | Model construction | once at startup |
| `load_checkpoint` | Checkpoint load | once at startup |
| `step` | Training step boundary | every iteration |
| `forward_backward` | Forward+backward pass | every iteration |
| `optimizer` | Optimizer step | every iteration |

## Presets

Presets bundle groups by use case:

| Preset | Groups included | Relative cost | Use case |
|---|---|---|---|
| `default` | job, checkpoint, evaluate | Lowest | Safe for production |
| `per_step` | default + model_init, load_checkpoint, step, forward_backward, optimizer | Moderate | Staging, profiling |
| `all` | Every group in the class | Highest | Debugging |

Presets are **per-subclass**: `MegatronSpanGroup.ALL_GROUPS` contains more than `SpanGroup.ALL_GROUPS`.

## The spec string

`config.span_groups` is a **comma-separated spec** that mixes preset keywords and individual group names:

```bash
NEMO_LENS_SPAN_GROUPS=default                  # just default preset
NEMO_LENS_SPAN_GROUPS=per_step                 # per_step preset
NEMO_LENS_SPAN_GROUPS=default,step             # default + one extra group
NEMO_LENS_SPAN_GROUPS=step,optimizer,checkpoint # individual groups only
NEMO_LENS_SPAN_GROUPS=all                      # everything
```

Resolution happens once at `setup_telemetry` via `config.resolved_span_groups`. The resulting `frozenset` is registered with the `state` module and consulted at every instrumentation site.

Unknown keywords raise `ValueError` at resolution time with a list of valid options.

## State machinery

Enabled groups live in a module-level `frozenset` in `nemo.lens.state`:

```python
from nemo.lens.state import is_span_group_enabled, set_enabled_span_groups

is_span_group_enabled('step')        # returns bool; a frozenset lookup
set_enabled_span_groups(frozenset(['job', 'step']))
```

The read path (`is_span_group_enabled`) is lock-free and safe to call from any thread. The write path (`set_enabled_span_groups`) is lock-protected and typically called once by `setup_telemetry`.

`set_enabled_span_groups` is also a top-level public export (`from nemo.lens import set_enabled_span_groups`), letting you override the active groups at runtime without reaching into `nemo.lens.state`.

## Extending: library-specific subclasses

Subclass `SpanGroup` to add domain-specific groups. Megatron's extension:

```python
from nemo.lens.groups import SpanGroup

class MegatronSpanGroup(SpanGroup):
    MICROBATCH = "microbatch"
    COMMUNICATION = "communication"
    ACTIVATION_OFFLOAD = "activation_offload"
    DATA_LOADING = "data_loading"
    INFERENCE = "inference"
    LAYER = "layer"

    ALL_GROUPS = frozenset([
        *SpanGroup.ALL_GROUPS,
        MICROBATCH, COMMUNICATION, ACTIVATION_OFFLOAD,
        DATA_LOADING, INFERENCE, LAYER,
    ])

    _PRESETS = {
        "default": frozenset([SpanGroup.JOB, SpanGroup.CHECKPOINT, SpanGroup.EVALUATE, INFERENCE]),
        "per_step": frozenset([
            *SpanGroup.ALL_GROUPS, COMMUNICATION, DATA_LOADING, INFERENCE,
        ]),
        "all": ALL_GROUPS,
    }
```

Pass it to `from_env`:

```python
cfg = NemoLensConfig.from_env(
    prefix='MEGATRON_OTEL',
    fallback_prefix='NEMO_LENS',
    span_group_cls=MegatronSpanGroup,
)
```

`config.resolved_span_groups` now resolves against `MegatronSpanGroup.ALL_GROUPS` and `_PRESETS`.

## Design notes

- Groups are **runtime knobs**, not compile-time. Toggling an env var and restarting is the full configuration workflow — no code changes needed.
- Groups are **orthogonal** to rank sampling (`NEMO_LENS_SAMPLER_ENABLED` / `NEMO_LENS_EXPORT_SAMPLE_RATE`) and export strategy (`NEMO_LENS_EXPORT_STRATEGY`). You can combine: enable `per_step` groups, sample 10% of ranks, export from one rank only.
- Groups are a **coarse filter**. For fine-grained control (e.g. "trace only iterations where loss > threshold"), add a runtime check inside your instrumented code — `is_span_group_enabled` is just one signal.
