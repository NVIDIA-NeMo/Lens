# Building the Docs

This page explains how to build and iterate on the lens documentation locally.

## Toolchain

The docs use:

- **Sphinx** — the build engine
- **MyST Parser** — lets us write docs in Markdown instead of reStructuredText
- **nvidia-sphinx-theme** — NVIDIA's themed Sphinx theme (same one Megatron-LM uses)
- **sphinx-autodoc2** — auto-generates API reference pages from the source tree
- **sphinx-copybutton** — copy-to-clipboard button on code blocks
- **sphinx-autobuild** — live reload during development

All of these are declared in the `[dependency-groups].docs` section of `pyproject.toml` (PEP 735).

## Install

```bash
cd lens
pip install --group docs -e .
```

This installs lens in editable mode plus all Sphinx dependencies. You only need to run this once (or after pulling changes to `pyproject.toml`). Requires pip 25.1+; with `uv`, use `uv pip install --group docs -e .`.

## Build commands

A `Makefile` in `docs/` wraps the common operations:

| Command | What it does | Typical time |
|---|---|---|
| `make html` | Full build with API reference | ~15-20 s |
| `make fast` | Skip autodoc2 (prose-only rebuild) | ~3 s |
| `make serve` | Live-reload server on :8000 | (runs until killed) |
| `make clean` | Remove `_build/` | instant |
| `make linkcheck` | Validate external links | ~10 s + network |

All commands run from `docs/`. Output lands in `docs/_build/html/`.

### One-shot full build

```bash
cd docs
make html
```

Open `docs/_build/html/index.html` in a browser.

### Fast iteration on prose

When you're editing prose (not adding features that change docstrings), skip autodoc2 — it has to re-scan the whole source tree:

```bash
cd docs
make fast
```

~5× faster per build. API reference pages are missing until you do a full build again.

### Live reload while editing

The most ergonomic workflow for writing docs:

```bash
cd docs
make serve
```

This starts `sphinx-autobuild` on `http://localhost:8000` (auto-opens your browser) and rebuilds on every file save. Keep the terminal running while you edit; the browser refreshes automatically.

`make serve` uses `SKIP_AUTODOC=true` — if you're changing docstrings and want them reflected, kill the server and run `make html` manually.

### Validate external links

Before publishing or merging a major docs change:

```bash
cd docs
make linkcheck
```

Sphinx walks every external URL and reports broken ones. GitHub links are excluded (they rate-limit in CI). Output lands in `_build/linkcheck/output.txt`.

### Clean rebuild

```bash
cd docs
make clean
make html
```

Useful if Sphinx's incremental build gets confused (rare, but it happens).

## What gets built

```
docs/
├── _build/                  ← generated; gitignored
│   └── html/
│       ├── index.html
│       ├── get-started/
│       ├── user-guide/
│       ├── design/
│       ├── observability/
│       ├── developer/
│       ├── apidocs/         ← autodoc2 output
│       ├── _static/         ← theme assets
│       └── ...
└── apidocs/                 ← autodoc2 source; gitignored, regenerated each build
    └── (generated .rst files)
```

Both `_build/` and `apidocs/` are in `.gitignore`. Never commit either.

## The autodoc2 flow

When you run `make html` (or plain `sphinx-build`), autodoc2:

1. Scans `src/nemo/lens/` for every module, class, and function.
2. Reads their docstrings.
3. Writes `.rst` files into `docs/apidocs/` describing the API.
4. Sphinx then renders those `.rst` files alongside your hand-written Markdown.

Skip with `SKIP_AUTODOC=true` when you don't need API pages — the `make fast` target does this automatically.

Autodoc2 config lives in `docs/conf.py` under the `autodoc2_*` settings.

## Writing conventions

- **Format**: Markdown (`.md`) everywhere. MyST parses it; ReST is fine too but not used.
- **Filenames**: lowercase-with-hyphens, e.g. `context-propagation.md`.
- **Single H1 per file**: the page title. Subsections use `##`, `###`.
- **Code blocks**: fenced with triple backticks + language tag (` ```python `, ` ```bash `, ` ```yaml `).
- **Cross-references**: relative Markdown links to other `.md` files, e.g. `[sampling](../user-guide/sampling.md)`. Sphinx rewrites these to `.html` at build time.
- **Tables**: GitHub-flavoured Markdown tables.
- **Admonitions** (when needed): MyST-style `:::{note}` ... `:::` blocks.

## Adding a new page

1. Create the `.md` file under the appropriate section (e.g. `docs/user-guide/my-feature.md`).
2. Add it to the relevant `{toctree}` block in `docs/index.md`.
3. Run `make html` and verify the page appears in the sidebar.

If you skip step 2, the page builds but isn't linked from anywhere — Sphinx warns about "document isn't included in any toctree".

## Debugging build failures

### "toctree contains reference to nonexisting document X"

You added an entry to a toctree but the file doesn't exist. Check spelling and path; entries are relative to the file they're in, without the `.md` extension.

### "document isn't included in any toctree"

A `.md` file exists but nothing references it. Add it to a toctree or delete it.

### "duplicate label" / "duplicate anchor"

Two pages use the same heading text. MyST generates anchors from headings; duplicates collide. Rename one.

### Blank pages or broken navigation

The theme cached assets don't match. `make clean && make html`.

### Autodoc2 errors about missing modules

Your source file has a syntax error, or an import in `src/nemo/lens/` that can't be resolved. Run `pytest` first — if tests pass, docs should too.

## CI

The docs are not currently built in CI. When that's added, the expected command will be:

```bash
pip install --group docs -e .
cd docs && make html
```

A CI failure would indicate:

- A toctree references a missing page.
- An autodoc2 import fails.
- A `make linkcheck` failure (if wired in).

Treat docs build failures as blocking — bad docs are worse than missing docs.

## Working on Megatron's docs

Megatron-LM has its own Sphinx docs at `../Megatron-LM/docs/`. The build command there is different (uses `uv` by default):

```bash
cd Megatron-LM
uv run --group docs sphinx-build docs docs/_build/html

# Or with SKIP_AUTODOC to skip megatron-core API scan (which is slow)
SKIP_AUTODOC=true uv run --group docs sphinx-build docs docs/_build/html
```

See the Megatron-LM Observability docs at `../Megatron-LM/docs/user-guide/observability/` for the user-facing side of that integration.

## Publishing

Currently docs are built locally — there's no published site yet. When publishing is set up (GitHub Pages, Read the Docs, or internal hosting), the flow will be:

```bash
cd docs && make clean && make html && make linkcheck
# then deploy _build/html/ to the target
```

The `conf.py` already includes theme config (icon links, version switcher scaffold) to support a versioned published site.
