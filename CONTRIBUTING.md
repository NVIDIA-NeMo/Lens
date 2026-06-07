# Contributing to nemo-lens

Thanks for your interest in contributing to nemo-lens!

## Reporting Issues

Bug reports, feature requests, and questions are tracked on the [GitHub issues page](https://github.com/NVIDIA-NeMo/Lens/issues). Before opening a new issue, please search existing issues to avoid duplicates. When reporting a bug, include a minimal reproducer, the lens version, and the relevant environment variables (`NEMO_LENS_*`, `OTEL_*`).

## Development Setup

Requires Python 3.13+.

```bash
git clone https://github.com/NVIDIA-NeMo/Lens.git
cd lens
pip install -e '.[dev]'
pre-commit install
```

Run the test suite and linters:

```bash
pytest
ruff check src tests --fix
ruff format src tests
```

## Pull Request Workflow

1. Fork the repository and create a feature branch off `main`:
   ```bash
   git checkout -b your-feature-name
   ```
2. Make your changes. Keep PRs focused on a single concern — split unrelated changes into separate PRs.
3. Add or update tests. New public APIs must be covered by tests in `tests/`.
4. Update documentation in `docs/` and the `README.md` when behavior, configuration, or public API changes.
5. Run `pytest` and `pre-commit run --all-files` locally; both must pass.
6. Commit with a sign-off (see [Signing Your Work](#signing-your-work)) and push to your fork:
   ```bash
   git commit -s -m "Short imperative summary"
   git push origin your-feature-name
   ```
7. Open a pull request against `main`. A maintainer will review and may request changes.

### Guidelines

- Follow existing conventions in the file and module you're touching.
- Keep the no-op hot path cheap — don't introduce work before the span-group gate check.
- Don't import from `opentelemetry.sdk.*` outside `providers.py`.
- Changes to the public API in `src/nemo/lens/__init__.py` or `fallbacks.py` must keep the consumer fallback files in `Megatron-LM/`, `RL/`, and `Gym/` ... in lockstep.
- The PR *title* must follow Conventional Commits (e.g. `feat: ...`, `fix: ...`, `docs: ...`, `ci: ...`). The "Validate PR title" CI check rejects non-conforming titles. Allowed type prefixes: `feat`, `fix`, `chore`, `docs`, `style`, `refactor`, `test`, `ci`, `build`, `perf`, `revert`.

## Signing Your Work

We require all contributors to sign off on their commits. Sign-off certifies that you wrote the change or otherwise have the right to submit it under the project's open source license. Any commit that is not signed off will not be accepted.

To sign off, use the `--signoff` (or `-s`) flag when committing:

```bash
git commit -s -m "Add cool feature."
```

This appends the following trailer to your commit message:

```
Signed-off-by: Your Name <your@email.com>
```

The name and email must match the author of the commit. By signing off, you certify the following Developer Certificate of Origin:

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
1 Letterman Drive
Suite D4700
San Francisco, CA, 94129

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```
