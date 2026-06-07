# Contributing

## Development setup

```bash
git clone <repo-url>
cd lens
pip install --group dev -e .
pre-commit install
```

The `dev` dependency group includes the OTel SDK, pytest, pytest-cov, pre-commit, and ruff.

## Code style

- **Line length**: 100 (ruff-enforced)
- **Python**: 3.13+ (use `X | Y`, not `Optional`)
- **Ruff rules**: E, W, F, I (isort), UP, B, SIM, TCH (with TC002/TC003 ignored for runtime imports)
- **Double quotes** for strings

Run:

```bash
ruff check src tests --fix
ruff format src tests
```

Pre-commit does both automatically on staged files.

## Heavy imports stay deferred

`opentelemetry-sdk` and related packages are expensive to import. Import them **inside** functions that actually need them, not at module level. `providers.py` is the only place the SDK is imported, and even there the imports happen inside `build_providers()` not at the top of the file.

This keeps `import nemo.lens` cheap — important for consumers that may import lens just to reach its fallbacks.

## Type hints everywhere

Public functions must have type hints on parameters and return types. Use `typing` + `__future__.annotations` for forward references. `TYPE_CHECKING` guards are fine for imports used only in type hints.

## No comments that repeat the code

Prefer:

```python
span = tracer.start_span(name)
```

Over:

```python
# Start a span on the tracer
span = tracer.start_span(name)
```

Comments should explain **why** (non-obvious invariants, workarounds, design choices). For most code, good names make comments unnecessary.

## Test-first for changes

For every change that touches behaviour:

1. Write a failing test.
2. Run it to confirm it fails.
3. Make the change.
4. Run the test to confirm it passes.
5. Run the full suite to confirm nothing else broke.

See [Testing](testing.md) for fixture patterns and state-isolation requirements.

## Changes to the public API

`__all__` in `__init__.py` is the public contract. Before adding or changing anything there:

1. Is this part of the Core Value Proposition, or a one-off helper? If the latter, keep it internal.
2. Does it have a test?
3. Does it have a docstring?
4. Does it have a user-guide page (or a sentence in an existing one)?
5. Do we need to update `fallbacks.py`? (Signature changes to `managed_span`/`trace_fn`/`span_cm`/`is_span_group_enabled`/`safe_set_span_attributes` must be mirrored.)

## Changes to `fallbacks.py`

Whenever a signature in `helpers.py`, `state.py`, or wherever else is mirrored in `fallbacks.py` changes, update `fallbacks.py` too — and add a test in `test_fallbacks.py` that exercises the new signature.

## Changes to semconv

When adding an attribute name constant:

1. Put it in the right namespace (`dl.*`, `<project>.*`, etc. — see [semconv](../design/semconv.md)).
2. Update the stability-marker comment block if the namespace is new.
3. Use it. Unused constants are dead code.

When bumping `SEMCONV_VERSION`:

1. Review upstream changelog for breaking changes.
2. Update constants that were renamed or removed upstream.
3. Mention in the PR description so downstream consumers know.

## Changes that affect consumers

Megatron-LM, NeMo-RL, and NeMo-Gym all depend on lens. Changes that break them are blocking:

- Renaming a public function → break.
- Changing `managed_span` to require a new positional argument → break.
- Removing a span group → break.

For such changes, coordinate with the consumer repos (feature branches, paired PRs). Prefer additive changes over breaking ones.

## PR checklist

- [ ] Tests added and passing (`pytest -v`)
- [ ] Ruff clean (`ruff check src tests && ruff format --check src tests`)
- [ ] Public API changes mirrored in `fallbacks.py` if applicable
- [ ] Docstrings updated for changed signatures
- [ ] User-guide page added or updated if the change is user-visible
- [ ] `__all__` updated if exports changed
- [ ] Changelog entry (if one exists)

## Docs

Docs source lives in `docs/` and is built with Sphinx + MyST parser + NVIDIA theme.

```bash
pip install --group docs -e .
cd docs && make html          # full build
cd docs && make serve         # live-reload on :8000 (preferred while writing)
```

See [Building the Docs](building-docs.md) for the full workflow — build commands, iteration patterns, writing conventions, and debugging build failures.

## Questions

Open an issue on the repo, or ping the NeMo ecosystem team.
