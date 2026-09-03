# AGENTS.md — NeMo Lens

Orientation for coding agents working in `NVIDIA-NeMo/Lens`. This file covers the
invariants and gotchas; `docs/` is the authoritative reference for everything
else. When the two disagree, `docs/` wins and this file should be fixed.

## Skills

`skills/` holds procedural guides for the tasks that cut across several files or
that CI will reject if done wrong. **Read the relevant `SKILL.md` before starting
a task it covers** — infer which one from the artifact in front of you rather
than waiting to be told. The routing table at the end of this file maps tasks to
skills and docs.

| Skill | Covers |
|---|---|
| `skills/lens-review/` | Reviewing a diff against the invariants below |
| `skills/lens-pre-pr-check/` | The full local gate before opening a PR |
| `skills/lens-create-issue/` | Triaging a red CI run into a filed bug |
| `skills/lens-respond-to-issue/` | Drafting a maintainer reply to a community issue |

`skills/` is the single source of truth. `.claude/skills` and `.agents/skills`
are symlinks to it, and `CLAUDE.md` is a symlink to this file, so every harness
reads the same content. Claude Code additionally defines a `lens-reviewer`
subagent under `.claude/agents/` that delegates `lens-review` to a fresh context;
its body deliberately points at the skill rather than restating it.
Edit `skills/` and `AGENTS.md` — never a symlink, and never copy content between
them.

## What this repo is

`nemo-lens` is a standalone, published Python library: shared OpenTelemetry
instrumentation for the NVIDIA NeMo ecosystem. It is consumed as an **optional**
dependency by Megatron-LM, NeMo-RL, and NeMo-Gym — those repos live elsewhere
and are not part of this checkout. Everything here is the library, its tests,
its docs, and a local observability stack for manual verification.

- Package: `nemo-lens`, importable as `nemo.lens` · Python ≥ 3.10
- Docs: <https://docs.nvidia.com/nemo/lens> · Source: `docs/` (Fern)
- `src/nemo/__init__.py` is a **PEP 420 namespace package** so `nemo.lens` can
  coexist with the upstream NeMo Framework's `nemo` package. Never add code to it.

## Layout

```
src/nemo/lens/
├── __init__.py        public API surface — __all__ is the contract
├── config.py          NemoLensConfig + from_env()
├── handle.py          setup_telemetry(), TelemetryHandle, double-init guard
├── providers.py       ONLY module allowed to import opentelemetry.sdk.*
├── groups.py          SpanRegistry — consumers declare the groups they emit
├── state.py           span-group spec + enabled frozenset; the hot-path gate
├── helpers.py         span_cm, managed_span, trace_fn, safe_set_span_attributes
├── fallbacks.py       canonical no-ops mirrored by consumer repos
├── strategies.py      export-strategy registry (which ranks export)
├── sampling.py        RankAwareSampler
├── distributed.py     broadcast_trace_context, create_linked_span
├── propagation.py     inject_context / extract_context (W3C)
├── logging_bridge.py  Python logging → OTel logs
├── semconv.py         attribute-name constants (single source)
├── package_info.py    version (bumped by release automation, not by hand)
├── instruments/       metric instruments: inference, rl, gym
├── resources/         detection (slurm, k8s, local) + encode_resource_attributes
└── contrib/           fastapi, aiohttp, ray, nccl integration helpers
```

`tests/` mirrors this module-for-module. `observability/` holds collector,
Prometheus, Grafana, Jaeger, and Kibana configs for the compose stack.

## Three invariants — do not break these

### 1. SDK imports live only in `providers.py`

`import nemo.lens` must work with only `opentelemetry-api` installed (the API's
default implementation is a no-op). `opentelemetry.sdk.*` appears **only** in
`providers.py`, and even there inside function bodies. Two classes that would
naturally subclass SDK types — `RankAwareSampler` and `SeedIndependentIdGenerator`
— are duck-typed instead, deliberately, so their modules stay SDK-free.

There is no linter for this. Grep before you add an import:
`grep -rn "opentelemetry.sdk" src/`.

### 2. Nothing happens before the span-group gate

`is_span_group_enabled()` is one `frozenset` membership test (`state.py`). Both
`managed_span` and `trace_fn` check it first and return without touching OTel
when the group is off. Any work placed *before* that check — string formatting,
`time.time()`, building an attribute dict — is paid by every user on every call
whether or not they enabled telemetry.

```python
# right: attributes computed only when the group is live
with managed_span("step", "train.step") as span:
    if span:
        span.set_attribute(DL_ITERATION, i)

# wrong: the f-string runs even when 'step' is disabled
with managed_span("step", "train.step", label=f"iter {i}") as span:
```

Passing an already-materialized value as a kwarg is fine; *building* one is not.

### 3. `fallbacks.py` signatures match the real API exactly

Consumers import from `nemo.lens.fallbacks` when lens is absent. Seven symbols
form that surface: `trace_fn`, `managed_span`, `span_cm`,
`is_span_group_enabled`, `safe_set_span_attributes`, `SpanRegistry`, and
`encode_resource_attributes`. `SpanRegistry` is there because consumers call
`SpanRegistry.register()` at import time, which has to work with lens absent.
Add a parameter to a real implementation and you must add it to the no-op too
(it may ignore it). `tests/test_fallbacks.py` is the enforcement. See
`docs/design/optional-dependency.mdx` for why this exists.

Signatures are the floor, not the ceiling: **defaults must resolve to the same
source**. `encode_resource_attributes(attrs)` with `inherited` omitted has to
read the same place in both, or the no-op preserves a launcher's value where the
real one drops it — a divergence a name-only parameter comparison cannot see.

## Instrumentation primitives

| Primitive | Gated? | Use for |
|---|---|---|
| `managed_span(group, name, **attrs)` | yes | Scoping a block. Yields `None` when the group is off — the body still runs. |
| `@trace_fn(group, name)` | yes | The whole function is the unit of work. Group checked at call time. |
| `span_cm(name, tracer=..., **attrs)` | **no** | Top-level always-on spans only (outermost job span, app startup). |

`span_cm` creates a span unconditionally. Reaching for it deeper than an entry
point is almost always a mistake.

## Span groups

**Lens defines no group names and no preset contents.** `SpanRegistry`
(`groups.py`) is a process-global registry; each consuming library declares what
it emits under its own namespace:

```python
SpanRegistry.register(
    "megatron",
    groups={"step", "microbatch", "layer"},
    presets={"default": {"step"}, "per_step": {"step", "microbatch"}},
)
```

Group names are one flat namespace across libraries (so call sites stay terse);
two namespaces claiming one name **share** it and get a `WARNING`, which
`allow_override=True` silences. Presets **union** across namespaces — `default`
means every registered library's default, which is the fix for the old subclass
scheme where `_PRESETS` was overridden wholesale. A preset may reference any
group, which is how a library layers on one it imports. A member no one has
registered is kept and warned about, and is ignored until its owner registers. A
member whose group is unregistered later is pruned the same way, so a preset can
never name a group outside `all`.

**`register()` must not raise for either of those.** Both happen while a
consuming library is being imported, before `setup_telemetry` and with no caller
in a position to catch them. Neither library can prevent it either: in an RL job
driving Megatron, both emit a `step`, and neither knows it is second. The no-op
`SpanRegistry` never raises, so raising here would mean installing lens breaks a
job that worked without it. `register()` does still raise for mistakes a single
author can fix: an empty namespace or group name, `all` on either side, and
re-registering a namespace without `allow_override=True`. `all` is built in,
means "everything registered", and is reserved as **both** a preset and a group
name — `resolve()` checks presets first, so a group called `all` would be
permanently unselectable.

**Register before `setup_telemetry`.** Importing a library registers its groups,
so by setup time everything in play is known. But resolution **never raises** —
an entry naming nothing registered here is warned about, kept in
`pending_span_groups()`, and the rest of the spec still applies. Two reasons, and
both are load-bearing:

- **A registry is per process; a spec is usually job-wide.** A launcher agent or
  a spawned checkpoint worker inherits one `NEMO_LENS_SPAN_GROUPS` from the
  trainer but imports different libraries, so a value valid in the trainer names
  nothing there. That is not a typo. Such a process should take its own prefix
  (`from_env(prefix="NVRX_OTEL", fallback_prefix="NEMO_LENS")`).
- **Raising from `setup_telemetry` is worse than it looks.** The raise used to
  land after `build_providers`, leaving real SDK providers and live batch threads
  installed while the caller got an exception instead of a handle to shut them
  down. Do not reintroduce a raise on the setup path.

Registering *after* `setup_telemetry` still takes effect — `state.py` retains the
spec and re-resolves — but warns, since the group was not selectable when the
spec was resolved. Refusing it would drop spans silently.

`set_enabled_span_groups()` *pins* a set and drops the spec, which is how a
disabled process stays disabled when a library registers afterwards.

**Lock order is state → registry.** `state._LOCK` is held across resolution, so
`state` calls into `SpanRegistry` while holding its own lock. `SpanRegistry`'s
mutators therefore notify `state` only *after* releasing `_LOCK` — every one of
them puts `cls._notify()` below its `with cls._LOCK` block. Move a `_notify()`
inside that block and the next registration hangs: `_LOCK` is a plain
`threading.Lock`, so the same thread tries to take it twice and deadlocks. No
second thread is involved, so reproduce it single-threaded rather than hunting
for a race. It hangs the suite instead of failing it, so what you see in CI is a
job timeout.

Likewise, `register()` normalises presets and commits under one hold: splitting
them lets a concurrent `unregister` land between, and the preset commits a
reference to a group that no longer exists. Warnings work the other way round —
collect them under the lock, log them after releasing it, because a logging
handler is arbitrary user code. Reads follow the same rule:
`SpanRegistry._snapshot()` returns presets, groups and namespaces from one hold,
and `_resolve_snapshot()` returns those together with the resolution and the
registry-empty flag as a `_Resolution`, so `state` sees one registry generation
instead of asking again. `_report` depends on this. It runs after `state._LOCK`
is released, and when it built its message from a fresh query the result could
contradict itself, calling a group unresolved while also listing it as
registered.

Adding to `default` raises always-on overhead for every user of every library in
the process, not just yours. Procedure: `docs/developer/new-span-group.mdx`.

## Classify before you record

The single most common instrumentation mistake is putting a value in the wrong
place. Three destinations, no overlap:

| Kind | Test | Goes to |
|---|---|---|
| Resource attribute | Fixed for the process lifetime (rank, world size, parallelism config, run id, cluster) | `resource_attributes=` on `setup_telemetry()` |
| Span attribute | Categorical, answers "which one?" per span (iteration, algorithm, backend) | `span.set_attribute()` / `managed_span` kwargs |
| Metric | A number that moves over time (loss, grad norm, throughput, reward, KL) | a `record_*_metrics()` in `instruments/` |

Time-series numbers never go on spans. Attribute *names* come from `semconv.py`
(`dl.*`, `rl.*`, `gym.*`, `nemo.*`, `slurm.*`, plus upstream `k8s.*`,
`gen_ai.*`) — add the constant there rather than inlining a string. Metric
*names* use application scope (`rl.*`, `gym.*`, `gen_ai.*`); consumer-specific
training metrics like `megatron.training.loss` live in the consumer, not here.

## Configuration

`NemoLensConfig.from_env(prefix="NEMO_LENS", fallback_prefix=...)`.
Consumers pass their own prefix (e.g. `MEGATRON_OTEL`) and fall back to
`NEMO_LENS`. Env keys, all `<PREFIX>_`-suffixed unless noted:

| Key | Field | Default |
|---|---|---|
| `ENABLED` | `enabled` | `false` — telemetry is opt-in |
| `SPAN_GROUPS` | `span_groups` | `default` |
| `EXPORTER` | `exporter` | `otlp` (or `console`) |
| `EXPORT_STRATEGY` | `export_strategy` | `single_rank` |
| `EXPORT_RANK` | `export_rank` | `-1` (last rank) |
| `EXPORT_SAMPLE_RATE` | `export_sample_rate` | `1.0`, validated to `[0,1]` |
| `SAMPLER_ENABLED` | `sampler_enabled` | `false` |
| `TRACES_ENABLED` / `METRICS_ENABLED` | | `true` |
| `LOGS_ENABLED` | `logs_enabled` | `false` |
| `RUN_ID` | `run_id` | `SLURM_JOB_ID`, else a random hex |
| `USER_ID` | `user` | `""` — note the field/key name mismatch |

Read **without** a prefix: `OTEL_SERVICE_NAME`, `WANDB_ENTITY`,
`WANDB_PROJECT`, `DEPLOYMENT_ENV`/`ENVIRONMENT`, `OTEL_METRIC_EXPORT_INTERVAL`,
`LOCAL_RANK` (by `first_rank_per_node`), `SLURM_JOB_ID`, `NO_VCS_VERSION`.
Everything `OTEL_EXPORTER_OTLP_*` is the SDK's business — don't reimplement it.

## Testing

`pytest` from the repo root. The suite runs in seconds, so always run all of it
rather than a subset. Its defining constraint is that
OTel providers, the span registry, and lens's enabled-group set are
**process-global**, so `conftest.py` has four `autouse` fixtures that reset them
around every test: providers + `_INITIALIZED`, the `SpanRegistry` + span-group
set + PP carrier, the export strategy registry, and the metric-group registry.
Consequences:

- Lens ships no groups, so a test that needs one must register it. The
  `demo_groups` fixture does that; `set_enabled_span_groups()` pins a set
  directly. Nothing carries over between tests.
- Calling `setup_telemetry()` twice in one process raises. Tests that legitimately
  need to (e.g. looping over ranks) pass `_allow_reinit=True`.
- Assert on span content with `InMemorySpanExporter` from `conftest.py`, passed
  via `setup_telemetry(..., span_exporter=...)`.

Full conventions: `docs/developer/testing.mdx`.

## Commands

```bash
uv venv && uv pip install -e . --group dev   # dev env
pytest                                        # full suite
pytest tests/test_helpers.py -v               # one file
pytest --cov=nemo.lens --cov-report=term-missing
ruff check src tests --fix && ruff format src tests
pre-commit run --all-files                    # what CI's lint job runs

npm --prefix docs/fern run generate:library:local   # docs: build API pages
npm --prefix docs/fern run check                    # docs: validate
```

## Docs

MDX under `docs/`, built by Fern. `docs/` is the **nightly** version;
`docs/fern/versions/0.1.0/pages/` is a frozen snapshot of the 0.1.0 release and
is currently a byte-identical copy. Editing one does not update the other.

Normal doc changes go in `docs/` only. Touch the `0.1.0` tree solely for a
deliberate backport. A new page must also be registered in
`docs/fern/versions/nightly.yml` or it will not appear in the sidebar. Details
and CI gates: `docs/developer/building-docs.mdx`.

## Contributing conventions

- **PR title must follow Conventional Commits** (`feat:`, `fix:`, `docs:`,
  `chore:`, `ci:`, `build:`, `perf:`, `refactor:`, `style:`, `test:`, `revert:`).
  A CI check rejects anything else.
- **Sign off every commit** (`git commit -s`) — DCO is enforced.
- **Open PRs ready for review, not as drafts.** The CI workflow carries an
  explicit fail-on-draft guard. (This is the opposite of the Megatron-LM
  convention; don't carry that habit over.)
- **New Python modules need the SPDX + Apache-2.0 header.** Copy it verbatim
  from a neighbouring file; a copyright-check workflow runs on every PR.
  `.github/workflows/*.yml` and `docs/fern/*.yml` carry it too; other config
  YAML (compose files, `observability/*`) does not. `src/nemo/__init__.py` is a
  deliberate exception — it holds two comment lines and nothing else.
- CI runs against `pull-request/NNN` mirror branches created by NVIDIA's
  copy-pr-bot, not against the PR branch directly.
- Do not hand-edit the version in `package_info.py` — the code-freeze and
  release workflows own it.

## Gotchas

- **`instruments/__init__.py` re-exports `record_inference_metrics` plus the
  metric registry** (`MetricSpec`, `register_metric_group`, `record_metrics`,
  `unregister_metric_group`, `registered_metric_groups`). `record_gym_metrics`
  is still reachable only through its submodule. There is no `rl.py`: RL metric
  series are consumer-owned and declared through the registry
  (`register_metric_group`), so their metric names live in the consumer (NeMo-RL),
  not in `semconv` (the `rl.*` span-attribute names stay in `semconv`). Ship a
  dedicated module only when a series needs per-point logic the registry cannot
  express (e.g. inference token usage); otherwise use the registry.
- **`SeedIndependentIdGenerator` exists for a real bug.** Training frameworks
  call `random.seed()` identically across data-parallel ranks, which made OTel's
  default generator emit colliding span/trace IDs. It uses a private `Random`
  and re-seeds after `fork`. Don't "simplify" it back to the default generator.
- **The compose stack's default collector mode is W&B Weave**, which needs
  `WANDB_API_KEY`. For a local Jaeger run, switch the uncommented `command:`
  line under `otel-collector` to `collector.yaml`.
- **`docker compose` fails outright without a `.env`** — both `megatron` and
  `otel-collector` declare `env_file: .env`. Run `cp .env.example .env` first.
- **The `megatron` service in `docker-compose.otel.yml` cannot build here.** Its
  `context: ..` / `COPY lens` is left over from an older monorepo layout. Bring
  up services explicitly (`up -d jaeger otel-collector prometheus grafana`)
  rather than the whole file.
- **The collector does not publish 4317/4318 to the host** — only 8889 and
  Jaeger's UI on 16686. A host process cannot reach it at `localhost:4317`;
  emit from a container on `otel-net`, or use `NEMO_LENS_EXPORTER=console`.
- `semconv.py` is excluded from coverage (`pyproject.toml`); it is constants only.

## Where to look for a procedure

| Task | Authoritative source |
|---|---|
| Register a span group | `docs/developer/new-span-group.mdx` |
| Add / change a public API symbol | `docs/design/optional-dependency.mdx` |
| Add a metric instrument | `docs/user-guide/metrics.mdx` (§ Architecture) |
| Write or fix a test | `docs/developer/testing.mdx` |
| Change docs, add a page | `docs/developer/building-docs.mdx` |
| Understand the fallback design | `docs/design/optional-dependency.mdx` |
| Understand module boundaries | `docs/design/architecture.mdx` |
| Pass attributes to a child process | `docs/user-guide/resources.mdx` |
| Run the observability stack | `docs/observability/stack.mdx` |
| Ship to a hosted backend | `docs/observability/backends.mdx` |
| Pre-PR gate | `skills/lens-pre-pr-check/` |
| Review pending changes | `skills/lens-review/` |
| File a bug from a failing CI run | `skills/lens-create-issue/` |
| Reply to a community issue | `skills/lens-respond-to-issue/` |
