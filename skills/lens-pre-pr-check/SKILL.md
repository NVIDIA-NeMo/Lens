---
name: lens-pre-pr-check
description: Runs the full local gate before opening or updating a pull request in the NeMo Lens repo — tests, lint, pre-commit, copyright headers, docs registration, commit sign-off, and PR title conventions. Catches the failures this repo's CI is known to reject.
license: Apache-2.0
when_to_use: Before pushing a branch or opening a pull request in this repo; checking whether CI will pass; finishing a change; 'ready to push', 'prep this PR', 'check before I open the PR', 'will CI pass', 'am I done'.
user_invocable: true
metadata:
  author: Ahmad Kiswani <akiswani@nvidia.com>
---

# Pre-PR gate

Everything CI will check, run locally first, plus the conventions CI cannot
check but reviewers will. Work top to bottom; stop and fix rather than
collecting a list.

State the real result of each step. A skipped or failing step reported as passing
is worse than no check at all.

## 1. Know what you are shipping

```bash
git status --short
git log --oneline main..HEAD
git diff main...HEAD --stat
```

Confirm the diff is only what the change is about. Stray edits to `uv.lock`,
generated pages under `docs/fern/product-docs/`, `.env`, or a compose file you
toggled while testing should come out now.

## 2. Lint and format

```bash
ruff check src tests --fix
ruff format src tests
```

Then run what CI's lint job actually runs, which is the pre-commit hooks over
the whole tree:

```bash
pre-commit run --all-files
```

If pre-commit modified files, stage them and re-run until clean.

## 3. Tests

```bash
pytest -q
```

The full suite is fast; run all of it rather than a subset. If anything fails,
fix it — do not rationalize a pre-existing failure without first checking
whether it also fails on `main`.

For a change with meaningful coverage implications:

```bash
pytest --cov=nemo.lens --cov-report=term-missing
```

Codecov comments on PRs and compares against the base commit. `semconv.py` is
excluded by config; every other new line should be exercised.

## 4. Copyright headers

New Python modules and new `.github/workflows/*.yml` files need the SPDX +
Apache-2.0 header; a copyright-check workflow runs on every PR. Other config
YAML — compose files, `observability/*` — carries none, so match the directory
you are adding to rather than applying the header blindly. List new files and
confirm:

```bash
git diff main...HEAD --name-only --diff-filter=A
head -3 <each new file>
```

Copy the header verbatim from a neighbouring file in the same directory — the
year and wording must match what CI expects.

## 5. Docs

If the change is user-visible — a new or renamed env var, span group, metric,
public symbol, or any behavior change a user could notice — the matching page
under `docs/` needs updating. `AGENTS.md`'s routing table says which page.

Then:

- A **new** page must be registered in `docs/fern/versions/nightly.yml`, or it
  will build fine and simply never appear in the sidebar.
- Doc edits belong in `docs/` (nightly). Changes landing under
  `docs/fern/versions/0.1.0/pages/` are a backport to a frozen release — make
  sure that is deliberate.
- MDX requires self-closing `<img ... />`; CI hard-fails on a bare `<img>`.
- CI runs an **offline** link check over `docs/**/*.mdx`, so relative links must
  resolve on disk. A link to a `.md` file that is actually `.mdx` will fail.

If Node is available and docs changed:

```bash
npm --prefix docs/fern run check
```

## 6. Commits

```bash
git log main..HEAD --format='%H%n%an <%ae>%n%(trailers:key=Signed-off-by)%n---'
```

Every commit needs a `Signed-off-by` trailer matching the commit author — DCO is
enforced and a single unsigned commit blocks the PR. Fix with
`git rebase --signoff main` (or `git commit --amend -s` for a single commit).

Commit *messages* are ordinary; the strict convention is on the PR title.

## 7. PR title and body

- **Title must follow Conventional Commits.** A CI check rejects anything else.
  Allowed types: `feat`, `fix`, `chore`, `docs`, `style`, `refactor`, `test`,
  `ci`, `build`, `perf`, `revert`. Mark a breaking change with `!` or a
  `BREAKING CHANGE:` trailer.
- **Do not open as a draft.** The CI workflow carries an explicit fail-on-draft
  guard. This is the opposite of the Megatron-LM convention.
- Fill in the PR template's checkboxes honestly, including the cross-repo
  `_fallbacks.py` box. Ticking it means the PR description names which consumer
  files need the matching edit — `Megatron-LM/megatron/core/telemetry/`,
  `RL/nemo_rl/telemetry/`, `Gym/nemo_gym/telemetry/` — and what the new
  signature is. None of those repos are in this checkout and their CI does not
  run against your branch, so an unticked or hand-waved box is how consumers
  find out from a bug report.

CI runs against a `pull-request/NNN` mirror branch created by NVIDIA's
copy-pr-bot rather than against your branch directly, so allow for the mirror
step before expecting checks to appear.

## 8. Final read

Re-read your own diff once, as a reviewer would. Look for debugging leftovers,
commented-out code, a `TODO` you meant to resolve, and — specific to this repo —
an `opentelemetry.sdk` import that wandered outside `providers.py`.

```bash
git diff main...HEAD
grep -rn "opentelemetry\.sdk" src/ | grep -v providers.py
```

For a substantial change, apply `skills/lens-review/SKILL.md` to the diff before
pushing. In Claude Code, delegate it to the `lens-reviewer` subagent.

## Report

Summarize: what passed, what you fixed, what still fails and why, and anything
you consciously skipped. Then state whether the branch is ready to push.
