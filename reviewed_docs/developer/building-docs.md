# Build the Documentation

This page explains how to build and iterate on the NeMo Lens documentation locally.

## Toolchain

The docs use:

- **Sphinx**: The build engine
- **MyST Parser**: A parser that enables writing documentation in Markdown instead of reStructuredText
- **nvidia-sphinx-theme**: The themed Sphinx theme from NVIDIA, which is the same theme used by Megatron-LM
- **sphinx-autodoc2**: A tool that automatically generates API reference pages from the source tree
- **sphinx-copybutton**: A tool that adds a copy-to-clipboard button to code blocks
- **sphinx-autobuild**: A tool that provides live reloading during development

All of these dependencies are declared in the `[dependency-groups].docs` section of `pyproject.toml` (PEP 735).

## Install the Dependencies

```bash
cd lens
pip install --group docs -e .
```

This installs NeMo Lens in editable mode and all Sphinx dependencies. You only need to run this command once or after pulling changes to `pyproject.toml`. This requires pip version 25.1 or later; with `uv`, use `uv pip install --group docs -e .`.

## Build Commands

A `Makefile` in `docs/` wraps the common operations:

| Command | What it does | Typical time |
|---|---|---|
| `make html` | Full build with API reference | ~15-20 s |
| `make fast` | Skip autodoc2 (prose-only rebuild) | ~3 s |
| `make serve` | Live-reload server on :8000 | (runs until killed) |
| `make clean` | Remove `_build/` | instant |
| `make linkcheck` | Validate external links | ~10 s + network |

All commands must run from the `docs/` directory. Output is generated in `docs/_build/html/`.

### Perform a One-Shot Full Build

```bash
cd docs
make html
```

Open `docs/_build/html/index.html` in a browser.

### Use Fast Iteration on Prose

When you are editing prose and not adding features that modify docstrings, skip autodoc2, as it must scan the entire source tree:

```bash
cd docs
make fast
```

This method is approximately five times faster per build. API reference pages remain missing until you perform a full build.

### Enable Live Reload While Editing

The most ergonomic workflow for writing the documentation is as follows:

```bash
cd docs
make serve
```

This starts `sphinx-autobuild` on `http://localhost:8000`, automatically opens your browser, and rebuilds on every file save. Keep the terminal running while you edit; the browser refreshes automatically.

The `make serve` command uses `SKIP_AUTODOC=true`. If you are changing docstrings and want those changes reflected in the documentation, stop the server and run `make html` manually.

### Validate External Links

Before publishing or merging a major documentation change, run:

```bash
cd docs
make linkcheck
```

Sphinx checks every external URL and reports any broken links. The output is generated in `_build/linkcheck/output.txt`.

### Perform a Clean Rebuild

```bash
cd docs
make clean
make html
```

This process is useful if the incremental build of Sphinx becomes out of sync, which is rare.

## What Gets Built

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

Both `_build/` and `apidocs/` are specified in `.gitignore`. Do not commit either directory.

## The Autodoc2 Flow

When you run `make html` or a plain `sphinx-build` command, autodoc2 performs the following steps:

1. Scans `src/nemo/lens/` for every module, class, and function.
2. Reads their docstrings.
3. Writes `.rst` files into `docs/apidocs/` to describe the API.
4. Sphinx renders those `.rst` files alongside the manually written Markdown.

You can skip this process by setting `SKIP_AUTODOC=true` when you do not require API pages; the `make fast` target handles this automatically.

The autodoc2 configuration is located in `docs/conf.py` under the `autodoc2_*` settings.

## Writing Conventions

- **Format**: Markdown (`.md`) everywhere. MyST parses it; ReST is fine too but not used.
- **Filenames**: Use lowercase letters with hyphens (for example, `context-propagation.md`). 
- **Headings**: Use a single H1 (`#`) heading per file for the page title. Subsections must use H2 (`##`) and H3 (`###`) headings.
- **Code blocks**: Fence code blocks with triple backticks and a language tag (for example, ` ```python `, ` ```bash `, or ` ```yaml `).
- **Cross-references**: Use relative Markdown links to other `.md` files (for example, `[sampling](../user-guide/sampling.md)`). Sphinx rewrites these links to `.html` at build time.
- **Tables**: Use GitHub-flavored Markdown tables.
- **Admonitions**: When required, use MyST-style `:::{note}` ... `:::` blocks. 

## Add a New Page

1. Create the `.md` file under the appropriate section (e.g., `docs/user-guide/my-feature.md`).
2. Add the file to the relevant `{toctree}` block in `docs/index.md`.
3. Run `make html` and verify that the page appears in the sidebar.

If you skip step 2, the page builds but is not linked from any location. In this case, Sphinx warns that the "document isn't included in any toctree."

## Debug Build Failures

### "toctree contains reference to nonexisting document X"

You added an entry to a toctree but the file does not exist. Check spelling and path; entries are relative to the file they're in, without the `.md` extension.

### "document isn't included in any toctree"

A `.md` file exists, but no pages reference it. Add the file to a toctree or delete it.

### "duplicate label" / "duplicate anchor"

Two pages use the exact same heading text. MyST generates anchors from headings, which causes duplicates to collide. Rename one of the headings.

### Blank Pages or Broken Navigation

The cached assets of the theme do not match. Run `make clean && make html` to resolve this issue.

### Autodoc2 Errors About Missing Modules

Your source file has a syntax error or contains an import in `src/nemo/lens/` that cannot be resolved. Run `pytest` first; if the tests pass, the documentation should build successfully as well.

## Continuous Integration

The docs are built in CI by `.github/workflows/build-docs.yml`. The build runs on every push to `main`, on `pull-request/[0-9]+` branches, and on every pull request. The job delegates to the shared reusable workflow `NVIDIA-NeMo/FW-CI-templates/.github/workflows/_build_docs.yml@v0.67.0`, which encapsulates the install and Sphinx build steps (so the exact commands are defined in that template, not in this repo).

A CI failure indicates one of the following issues:

- A toctree references a missing page.
- An autodoc2 import fails.
- A `make linkcheck` failure (if wired in).

Treat docs build failures as blocking; bad docs are worse than missing docs.

## About Working on the Megatron-LM Documentation

Megatron-LM has its own Sphinx documentation at `../Megatron-LM/docs/`. The build command for that repository is different and uses `uv` by default:

```bash
cd Megatron-LM
uv run --group docs sphinx-build docs docs/_build/html

# Or with SKIP_AUTODOC to skip megatron-core API scan (which is slow)
SKIP_AUTODOC=true uv run --group docs sphinx-build docs docs/_build/html
```

Refer to the Megatron-LM Observability documentation at `../Megatron-LM/docs/user-guide/observability/` for the user-facing details of that integration.

## Publish the Documentation

Currently, the documentation is built locally, and there is no published website yet. When publishing is established using GitHub Pages, Read the Docs, or internal hosting, the process will be as follows:

```bash
cd docs && make clean && make html && make linkcheck
# then deploy _build/html/ to the target
```

The `conf.py` file already includes theme configuration, such as a GitHub icon link using `html_theme_options["icon_links"]`, to prepare for a published site. However, a version switcher must still be added.
