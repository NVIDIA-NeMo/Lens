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

- Package: `nemo-lens`, importable as `nemo.lens` · Python ≥ 3.12
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
├── groups.py          SpanGroup base class + presets
├── state.py           enabled-group frozenset; the hot-path gate
├── helpers.py         span_cm, managed_span, trace_fn, safe_set_span_attributes
├── fallbacks.py       canonical no-ops mirrored by consumer repos
├── distributed.py     broadcast_trace_context, create_linked_span
├── propagation.py     inject_context / extract_context (W3C)
├── logging_bridge.py  Python logging → OTel logs
├── semconv.py         attribute-name constants (single source)
├── package_info.py    version (bumped by release automation, not by hand)
├── instruments/       metric instruments: inference, rl, gym
├── resources/         auto-detection: slurm, kubernetes, local
└── contrib/           fastapi, aiohttp, ray, nccl integration helpers
```

`tests/` mirrors this module-for-module. `observability/` holds collector,
Prometheus, Grafana, Jaeger, and Kibana configs for the compose stack.

## Three invariants — do not break these

### 1. SDK imports live only in `providers.py`

`import nemo.lens` must work with only `opentelemetry-api` installed (the API's
default implementation is a no-op). `opentelemetry.sdk.*` appears **only** in
`providers.py`, and even there inside function bodies. Two classes that would
naturally subclass SDK types — `SeedIndependentIdGenerator` and `_OpenSpanCloser`
— are duck-typed instead, deliberately, so the module stays importable without
the SDK.

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
with managed_span(SpanGroup.STEP, "train.step") as span:
    if span:
        span.set_attribute(DL_ITERATION, i)

# wrong: the f-string runs even when 'step' is disabled
with managed_span(SpanGroup.STEP, "train.step", label=f"iter {i}") as span:
```

Passing an already-materialized value as a kwarg is fine; *building* one is not.

### 3. `fallbacks.py` signatures match the real API exactly

Consumers import from `nemo.lens.fallbacks` when lens is absent. Five symbols
form that surface: `trace_fn`, `managed_span`, `span_cm`,
`is_span_group_enabled`, `safe_set_span_attributes`. Add a parameter to a real
implementation and you must add it to the no-op too (it may ignore it).
`tests/test_fallbacks.py` is the enforcement. See
`docs/design/optional-dependency.mdx` for why this exists.

## Instrumentation primitives

| Primitive | Gated? | Use for |
|---|---|---|
| `managed_span(group, name, **attrs)` | yes | Scoping a block. Yields `None` when the group is off — the body still runs. |
| `@trace_fn(group, name)` | yes | The whole function is the unit of work. Group checked at call time. |
| `span_cm(name, tracer=..., **attrs)` | **no** | Top-level always-on spans only (outermost job span, app startup). |

`span_cm` creates a span unconditionally. Reaching for it deeper than an entry
point is almost always a mistake.

## Span groups

`SpanGroup` (`groups.py`) is the base class. Base groups: `job`, `checkpoint`,
`evaluate` (coarse) and `model_init`, `load_checkpoint`, `step`,
`forward_backward`, `optimizer` (medium). Consumers subclass it to add their own.

Presets resolved by `SpanGroup.resolve(spec)`, where `spec` is a comma-separated
mix of preset names and bare group names:

| Preset | Contents |
|---|---|
| `default` | `job`, `checkpoint`, `evaluate` — must stay cheap enough for production |
| `per_step` | `default` + the medium-grained groups |
| `all` | everything in `ALL_GROUPS` |

Adding to `default` raises always-on overhead for every user. Subclass `_PRESETS`
is overridden wholesale, not merged — a new base group does not automatically
appear in a subclass preset. Procedure: `docs/developer/new-span-group.mdx`.

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

## Lens does not know about ranks

`setup_telemetry(config, resource_attributes=..., ...)` takes no `rank` or
`world_size`, and there is no export-strategy registry. Every process where
`config.enabled` is true exports; `handle.is_exporting` is just `config.enabled`.

A distributed caller supplies its own position as resource attributes using the
`DL_RANK` / `DL_WORLD_SIZE` constants from `semconv.py`, and decides which ranks
report by setting `enabled` per rank (or by filtering on `dl.rank` in the
collector). `service.instance.id` is derived from `run_id` plus that supplied `dl.rank`,
resolved from **both** `resource_attributes=` and `OTEL_RESOURCE_ATTRIBUTES`
(caller wins, matching the SDK). Two traps live here:

- Derive from the resolved pair, not from the local `attrs` dict — env attributes
  do not exist in it, and a spawned worker supplies its rank exactly that way.
- Do not test "is `service.instance.id` already set?" against the built
  `Resource`: the SDK auto-populates it with a per-process UUID, so the answer is
  always yes and the derivation silently never runs.

With no rank from either channel it degrades to `run_id` alone and is *not*
unique per process. `providers._warn_no_rank` logs a `WARNING` in that case —
once, at startup, off any hot path. It drops to `DEBUG` when the caller supplied
its own `service.instance.id`: the warning's stated harm does not apply to a
process that named itself, and a launcher agent with no rank to claim is a
correct configuration rather than one to nag per node per job.

Do not reintroduce a rank parameter to make this convenient, and do not silence
that warning by reading `RANK`/`WORLD_SIZE` from the environment: those are
launcher conventions, inferring the rank is the behaviour this design removed, and
it is wrong for any caller whose rank is not the process's global rank.
`OTEL_RESOURCE_ATTRIBUTES` is a different thing and is fine to read — it is the
standard resource channel carrying an attribute lens itself named in `semconv.py`,
not a guess about the launcher.

## Configuration

`NemoLensConfig.from_env(prefix="NEMO_LENS", fallback_prefix=..., span_group_cls=...)`.
Consumers pass their own prefix (e.g. `MEGATRON_OTEL`) and fall back to
`NEMO_LENS`. Env keys, all `<PREFIX>_`-suffixed unless noted:

| Key | Field | Default |
|---|---|---|
| `ENABLED` | `enabled` | `false` — telemetry is opt-in |
| `SPAN_GROUPS` | `span_groups` | `default` |
| `EXPORTER` | `exporter` | `otlp` (or `console`) |
| `TRACES_ENABLED` / `METRICS_ENABLED` | | `true` |
| `LOGS_ENABLED` | `logs_enabled` | `false` |
| `RUN_ID` | `run_id` | `SLURM_JOB_ID`, else a random hex |
| `USER_ID` | `user` | `""` — note the field/key name mismatch |

Read **without** a prefix: `OTEL_SERVICE_NAME`, `WANDB_ENTITY`,
`WANDB_PROJECT`, `DEPLOYMENT_ENV`/`ENVIRONMENT`, `OTEL_METRIC_EXPORT_INTERVAL`,
`SLURM_JOB_ID`, `NO_VCS_VERSION`.
Everything `OTEL_EXPORTER_OTLP_*` is the SDK's business — don't reimplement it.

`OTEL_RESOURCE_ATTRIBUTES` is the SDK's too, but `providers.py` reads it back via
`OTELResourceDetector().detect()` before deriving `service.instance.id`. It is the
only attribute channel that survives a `spawn`/`exec`, so a worker with no
`setup_telemetry` call site supplies `dl.rank` there. Deriving from the caller's
dict alone made lens blind to it — see the rank section below.

## Testing

`pytest` from the repo root. The suite runs in seconds, so always run all of it
rather than a subset. Its defining constraint is that
OTel providers and lens's enabled-group set are **process-global**, so
`conftest.py` has two `autouse` fixtures that reset them around every test:
providers + `_INITIALIZED`, and the span-group set + PP carrier. Consequences:

- A test that needs a group active must enable it explicitly; nothing carries over.
- Calling `setup_telemetry()` twice in one process raises. Tests that legitimately
  need to (e.g. re-initialising with a different config) pass `_allow_reinit=True`.
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

- **`instruments/__init__.py` re-exports only `record_inference_metrics`.**
  `record_rl_metrics` and `record_gym_metrics` are reachable only through their
  submodules. Intentional today; don't "fix" it silently, and match the existing
  pattern when adding one.
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
| Add a span group | `docs/developer/new-span-group.mdx` |
| Add / change a public API symbol | `docs/design/optional-dependency.mdx` |
| Add a metric instrument | `docs/user-guide/metrics.mdx` (§ Architecture) |
| Write or fix a test | `docs/developer/testing.mdx` |
| Change docs, add a page | `docs/developer/building-docs.mdx` |
| Understand the fallback design | `docs/design/optional-dependency.mdx` |
| Understand module boundaries | `docs/design/architecture.mdx` |
| Run the observability stack | `docs/observability/stack.mdx` |
| Ship to a hosted backend | `docs/observability/backends.mdx` |
| Pre-PR gate | `skills/lens-pre-pr-check/` |
| Review pending changes | `skills/lens-review/` |
| File a bug from a failing CI run | `skills/lens-create-issue/` |
| Reply to a community issue | `skills/lens-respond-to-issue/` |
