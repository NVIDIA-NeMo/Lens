---
name: lens-review
description: Reviews pending changes in the NeMo Lens repo against the library's load-bearing invariants — SDK-import isolation, hot-path gating, fallback signature parity, span-group and preset placement, semconv naming, value classification, test state isolation, and doc sync.
license: Apache-2.0
when_to_use: Reviewing a diff, branch, or pull request in this repo; checking a change before pushing; auditing instrumentation someone else wrote; 'review my changes', 'review this PR', 'did I break anything', 'check this before I push', 'audit this diff'.
user_invocable: true
argument: "[PR number, branch, or path]  # optional: defaults to the working tree"
metadata:
  author: Ahmad Kiswani <akiswani@nvidia.com>
---

# Review a NeMo Lens change

Style is ruff's job and mechanics are CI's job. This review covers the handful
of properties nothing else in the repo checks.

Read `AGENTS.md` first if you have not; it states the invariants enforced here
and the layout being read.

## Get the diff first

Do not review from memory or from a description. Establish what actually changed:

```bash
git status --short
git diff main...HEAD --stat     # or --staged / HEAD~1, whichever matches the ask
git diff main...HEAD
```

If a PR was named, read it (`gh pr diff <n>`). If the working tree is dirty and
the branch also has commits, state which one was reviewed.

## Checks

Walk these in order. Skip a check cleanly if the diff cannot touch it — say "not
applicable" rather than padding the report.

### 1. SDK import isolation

`opentelemetry.sdk.*` may appear only in `src/nemo/lens/providers.py`, and there
only inside function bodies. `import nemo.lens` must succeed with just
`opentelemetry-api` installed.

```bash
grep -rn "opentelemetry\.sdk" src/ | grep -v "src/nemo/lens/providers.py"
grep -n "^from opentelemetry.sdk\|^import opentelemetry.sdk" src/nemo/lens/providers.py
```

Both should come back empty. A module-level SDK import in `providers.py` is as
much a violation as one elsewhere. `sampling.py` and `providers.py` duck-type
SDK base classes on purpose — if the diff makes either subclass a real SDK type,
that is a blocker.

### 2. Hot-path gating

For every new or modified `managed_span` / `trace_fn` call site, and for any
change to `helpers.py` or `state.py`: does any work happen before the span-group
check?

Flag f-strings, `.format()`, dict and list comprehensions, `time.time()`, tensor
or CUDA access, and function calls in the argument list of a gated primitive.
Passing a variable that already exists is fine; computing one is not.

Also flag new work inside `is_span_group_enabled` itself — it is a single
`frozenset` membership test and must stay that way.

### 3. Fallback parity

If the diff touches `trace_fn`, `managed_span`, `span_cm`,
`is_span_group_enabled`, or `safe_set_span_attributes` in `helpers.py` or
`state.py`, the matching no-op in `fallbacks.py` must accept the same argument
shape, and `tests/test_fallbacks.py` must cover it.

Compare signatures directly rather than eyeballing:

```bash
grep -n "^def \|^@contextmanager" src/nemo/lens/fallbacks.py
grep -n "^def \|^@contextmanager" src/nemo/lens/helpers.py
```

A new name in `nemo.lens.__all__` is not automatically part of the fallback
surface — that surface is exactly those six symbols (`trace_fn`, `managed_span`,
`span_cm`, `is_span_group_enabled`, `safe_set_span_attributes`, `SpanRegistry`).
A seventh is a deliberate contract change and should be called out as one, since
consumer repos mirror this file. A changed `SpanRegistry` method signature counts
too — the no-op class must match.

### 4. Public surface

Anything added to or removed from `src/nemo/lens/__init__.py`'s `__all__` is
consumer-visible. Check the symbol is importable, is tested, and that a removal
or rename has a deprecation path rather than a hard break. Note it prominently
even when otherwise correct — the author may not have realized it was public.

A public-surface change also carries the consumer mirror obligation: each of
Megatron-LM, NeMo-RL, and NeMo-Gym keeps its own `telemetry/_fallbacks.py` and
`telemetry/span_groups.py` copy of this contract, none of them in this checkout.
Flag that the PR must name the affected files and new signatures for those repos.

### 5. Span groups and presets

Lens defines no group names of its own — a diff that hard-codes one into
`groups.py` is a blocker. Groups arrive through `SpanRegistry.register()` in a
consuming library.

Anything added to a `default` preset needs justification: presets union across
namespaces, so it raises the always-on tier for every library in the process, not
just the one registering it.

Watch for two regressions specifically:

- Work moved *before* the `is_span_group_enabled` gate, or re-resolution moved
  out of registration and onto the gating path. The gate must stay one
  `frozenset` membership test.
- Resolution made strict again (raising on an unknown spec entry), or any new
  raise on the `setup_telemetry` path. A registry is per process while the spec
  is job-wide, so an unresolvable name is as likely to be another process's
  vocabulary as a typo — and a raise after `build_providers` strands live
  exporters with no handle to close them. Warnings plus `pending_span_groups()`
  are the contract.
- A `SpanRegistry._notify()` moved *inside* a `with cls._LOCK` block, or a
  read-resolve-write in `state.py` split across two lock holds. Lock order is
  state → registry; inverting it deadlocks, and splitting the resolve lets a
  stale enabled set stick for the life of the process. Both entry points carry
  the pattern — `set_span_group_spec` and `refresh_enabled_span_groups` — and
  `tests/test_state.py::TestResolutionIsAtomic` covers each. Registry *reads*
  are the same rule: `_snapshot()` returns presets and groups from one hold and
  `_resolve_snapshot()` adds the registry-empty flag, so a caller cannot mix
  generations by asking twice.

### 6. Naming

- Attribute names: constants in `semconv.py`, not string literals at the use
  site. Namespace must be `dl.*`, `rl.*`, `gym.*`, `nemo.*`, `slurm.*`,
  `wandb.*`, or upstream (`k8s.*`, `gen_ai.*`). A new namespace is a design
  decision, not a detail — flag it as one.
- Metric names: application scope (`rl.*`, `gym.*`, `gen_ai.*`), never `dl.*`.
  Unit and description set on the instrument.
- Span names: `<library>.<operation>[.<sub_operation>]`, snake_case after the
  prefix, describing user-visible behavior rather than internals.
- Span group names: lowercase snake_case, concept- not verb-oriented
  (`checkpoint`, not `saving_checkpoint`).

### 7. Classification

Time-series numbers (loss, throughput, reward, latency) go to a
`record_*_metrics()` instrument, never onto a span attribute. Process-lifetime
constants (rank, parallel sizes, run id) belong in `resource_attributes` on
`setup_telemetry()`, not on individual spans. Catching a misclassification here
is the highest-value outcome of the review.

### 8. Tests

New behavior needs a test in the mirroring `tests/test_*.py`. Beyond presence,
check new tests cooperate with the three `autouse` fixtures in `conftest.py`:
they must not assume a span group is enabled, must not assume a provider
survives from a previous test, and must pass `_allow_reinit=True` if they call
`setup_telemetry()` more than once. A test registering an export strategy needs
no cleanup — the registry fixture handles it.

Run what the diff touches and report the real result:

```bash
pytest -q
ruff check src tests
```

### 9. Docs and headers

- User-visible change (new env var, group, metric, public symbol, behavior)
  → is the matching page under `docs/` updated? A new page also needs an entry
  in `docs/fern/versions/nightly.yml`.
- Edits landing in `docs/fern/versions/0.1.0/pages/` are a backport to a frozen
  release — flag unless clearly intended.
- New Python modules and new `.github/workflows/*.yml` files need the SPDX +
  Apache-2.0 header; a CI workflow checks this. Other config YAML in this repo
  carries none — match the directory rather than assuming.

## Report

Group findings by severity, most severe first, citing `file:line` for each.

- **Blocker** — breaks an invariant, a contract, or correctness.
- **Should fix** — wrong classification, wrong preset, missing test or doc.
- **Nit** — naming, polish. Keep these few.

Every finding states the concrete consequence. Not "wrong attribute" but
"`src/nemo/lens/instruments/rl.py:88` records `entropy` as a span attribute; it
changes every step, so it will be lost to span sampling and unqueryable as a
series — use the existing `rl.entropy` gauge."

End with what was run and what it returned. If the change is clean, say so in a
sentence and stop — do not manufacture findings to justify the review.

## Out of scope

Do not edit code during a review. Do not restyle or reformat. Do not demand
coverage of gating branches that cannot be exercised without the SDK. Do not
review `uv.lock`, generated API pages under `docs/fern/product-docs/`, or
vendored CI template SHAs.
