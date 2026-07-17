# Add a New Span Group

Span groups are a runtime knob. You add one, wire instrumentation to it, and users get a new opt-in slice of telemetry without any code changes on their side. This page walks through adding one to either the base `SpanGroup` or a consumer-specific subclass.

See [Span Groups](../user-guide/span-groups.md) for how groups work at runtime; this page is for library authors.

## Decide Where It Belongs

The decision is about scope, not implementation:

- **Base `SpanGroup`**: only for groups meaningful across every consumer (Megatron, RL, Gym, and any hypothetical new one). The current base set (which includes `job`, `checkpoint`, `evaluate`, `model_init`, `load_checkpoint`, `step`, `forward_backward`, and `optimizer`) is tight on purpose. Adding to it is a commitment.
- **Consumer subclass** (`MegatronSpanGroup`, `RLSpanGroup`, `GymSpanGroup`): everything else. If the concept is not universal, it goes here.

When in doubt, start in the subclass. Promoting to the base later is cheap; demoting is not.

## Add to the Base Class

File: `lens/src/nemo/lens/groups.py`.

```python
class SpanGroup:
    # existing groups...
    MY_NEW_GROUP = "my_new_group"

    ALL_GROUPS: Final[frozenset] = frozenset([
        JOB, CHECKPOINT, EVALUATE, MODEL_INIT, LOAD_CHECKPOINT,
        STEP, FORWARD_BACKWARD, OPTIMIZER,
        MY_NEW_GROUP,
    ])

    _PRESETS: ClassVar[dict] = {
        "default": frozenset([...]),     # include only if cheap and always useful
        "per_step": frozenset([...]),
        "all": ALL_GROUPS,
    }
```

Checklist:

- [ ] Constant added: `MY_NEW_GROUP = "my_new_group"`.
- [ ] Constant added to `ALL_GROUPS`.
- [ ] Constant added to the appropriate presets (`default` / `per_step` / `all`).
- [ ] Test case in `tests/test_groups.py` covering resolution of the new name.
- [ ] Instrumentation added at the relevant call site(s) using `managed_span("my_new_group", ...)` or `@trace_fn("my_new_group", ...)`.
- [ ] Entry added to [Span Groups](../user-guide/span-groups.md).

## Add to a Consumer Subclass

Example: Megatron. File: `Megatron-LM/megatron/core/telemetry/span_groups.py`.

```python
class MegatronSpanGroup(SpanGroup):
    # existing Megatron groups...
    MY_NEW_GROUP = "my_new_group"

    ALL_GROUPS: Final[frozenset] = SpanGroup.ALL_GROUPS | frozenset([
        MICROBATCH, LAYER, COMMUNICATION, ACTIVATION_OFFLOAD, DATA_LOADING, INFERENCE,
        MY_NEW_GROUP,
    ])

    _PRESETS: ClassVar[dict] = {
        "default": frozenset([
            SpanGroup.JOB, SpanGroup.CHECKPOINT, SpanGroup.EVALUATE, INFERENCE,
        ]),
        "per_step": frozenset([...]),
        "all": ALL_GROUPS,
    }
```

Checklist:

- [ ] Constant added to the subclass.
- [ ] Constant added to the subclass's `ALL_GROUPS` (inheriting base via set union).
- [ ] Constant added to the appropriate subclass presets.
- [ ] Instrumentation added in the consumer source tree.
- [ ] Entry added to the consumer's observability docs (e.g. `Megatron-LM/docs/user-guide/observability/span-groups.md`).

## No Changes Needed in `fallbacks.py`

`fallbacks.py` provides canonical no-ops for `managed_span`, `trace_fn`, `span_cm`, `is_span_group_enabled`, and `safe_set_span_attributes`. Group names are plain strings. When the no-op fallback is active, `is_span_group_enabled(group)` returns `False` regardless of the name. Adding a new group does not require touching the fallbacks.

## Naming Conventions

- Lowercase, `snake_case`: `pipeline_parallel`, not `PipelineParallel` or `pipeline-parallel`.
- Concept-oriented, not verb-oriented: `checkpoint`, not `saving_checkpoint`.
- Short: `evaluate`, not `evaluation_phase`.
- Skim the existing list before inventing; if a nearby concept already has a group, extend instrumentation there rather than adding a new one.

## Preset Inclusion

Do not add to `default` casually. The contract for `default` is that it stays quiet enough for production (structural job-level spans only). If your group fires more than a handful of times per iteration, it belongs in `per_step` or `all`, and not `default`. Getting this wrong means every production user of the library inherits your instrumentation overhead.

A reasonable default: add to `per_step` and `all` only, and wait for a concrete reason before promoting to `default`.

## Testing

See [Testing](testing.md) for fixture conventions. A typical new-group test:

```python
def test_my_new_group_in_per_step_preset():
    cfg = NemoLensConfig(
        enabled=True,
        span_groups="per_step",
        exporter="console",
    )
    setup_telemetry(cfg, rank=0, world_size=1)
    assert is_span_group_enabled("my_new_group")
```

At minimum, cover:

- The group resolves from its bare name.
- The group is included in every preset you added it to, and excluded from the ones you didn't.
- An instrumentation site wrapped in `managed_span("my_new_group", ...)` emits a span when the group is enabled, and emits nothing when it is not.
