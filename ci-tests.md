# CI Test Plan

A scratchpad of scenarios to validate every workflow on the CI-test branch before merging the CI to `main`. Delete this file before the final merge.

[v] linting workflow
[v] tests workflow
[v] PR title validation
[v] copyright check
[v] secret detection

[] building docs
[] PR semantics has two
[] Build wheel has 3 ?
[] DCO

Conventions:
- **Setup** — the change to push
- **Expect** — what the PR's "Checks" tab should show
- **Reset** — how to undo before testing the next scenario

Run each scenario as a separate commit (or a separate PR) so the signal is isolated. Where possible, push to the same PR and watch checks transition red → green.

---

## 1. DCO check (GitHub App, not a workflow)

### 1a. Unsigned commit fails
- **Setup**: `git commit -m "test: unsigned"` (no `-s`)
- **Expect**: ❌ `DCO` status check fails with a link to the bot's explanation
- **Reset**: `git reset --soft HEAD~1 && git commit -s -m "test: signed"` and force-push

### 1b. Signed commit passes
- **Setup**: amend with `git commit --amend -s --no-edit` and force-push
- **Expect**: ✅ `DCO` check goes green

### 1c. Mixed PR (one signed, one not) fails
- **Setup**: push 2 commits, only sign the first
- **Expect**: ❌ DCO check fails, bot identifies the offending commit by SHA

**If DCO doesn't appear at all**, the DCO GitHub App isn't installed on the repo (or org-wide). Install from https://github.com/apps/dco.

---

## 2. Copyright check (`copyright-check.yml`)

### 2a. New file without header fails
- **Setup**: create `src/nemo/lens/_ci_probe.py` containing just `x = 1`
- **Expect**: ❌ `copyright-check` fails, naming the file

### 2b. New file with wrong-format header fails
- **Setup**: same file, but with `# (c) Me 2026`
- **Expect**: ❌ still fails — format must match the existing pattern

### 2c. New file with correct header passes
- **Setup**: prepend `# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.`
- **Expect**: ✅ passes

### 2d. Confirm existing files still pass
- The workflow should green on a no-op commit (sanity that the v0.66.7 template understands the existing headers in `src/nemo/lens/*.py`).

**Reset**: `git rm src/nemo/lens/_ci_probe.py`

---

## 3. Code linting (`code-linting.yml` → pre-commit + ruff)

### 3a. Unused import fails
- **Setup**: in any existing file, add `import sys` at the top (unused)
- **Expect**: ❌ ruff F401

### 3b. Formatting violation fails
- **Setup**: add a line like `x=1+2` (missing spaces) somewhere
- **Expect**: ❌ ruff-format finds a diff

### 3c. Autofix is not silently applied
- **Setup**: same as 3a/3b
- **Expect**: CI **fails** rather than committing a fix — `--exit-non-zero-on-fix` is set in `.pre-commit-config.yaml`. Catches the bug where local pre-commit auto-fixes and developers forget to amend.

### 3d. Clean code passes
- Revert and verify green.

---

## 4. Unit tests (`unit-tests.yml`)

### 4a. Failing test fails CI
- **Setup**: add `def test_ci_probe(): assert False` to `tests/test_config.py`
- **Expect**: ❌ pytest reports the failure with a clear traceback

### 4b. Source regression breaks tests
- **Setup**: change `NEMO_LENS_ENABLED` default in `src/nemo/lens/config.py` to something obviously wrong
- **Expect**: ❌ existing tests fail (not just the new one)

### 4c. Draft PR is blocked
- **Setup**: open the PR as a draft
- **Expect**: ❌ `Test` step short-circuits with "Failing on draft PR to enforce this check to run" (the `fail-on-draft` guard)
- **Reset**: mark "Ready for review" → check should re-run and pass

### 4d. Clean run reports 148+ tests
- After reverting probes, verify the test count in the run logs matches local `pytest` output.

### 4e. Coverage line is present
- The workflow runs with `--cov=nemo.lens --cov-report=term-missing`. Confirm a coverage table appears in the job log.

---

## 5. Build wheel (`build-wheel.yml`)

This is the highest-risk workflow because lens uses `setuptools+versioningit` but we passed `packaging: uv` to the FW-CI template (matching Gym).

### 5a. Dry-run completes on PR
- **Setup**: any commit; the workflow runs on `pull_request`
- **Expect**: ✅ wheel built, `twine check` passes inside the template
- **Watch for**:
  - Versioningit must see git tags. If the template doesn't fetch full history, version may be `0.0.0` (acceptable for dry-run, but flag for the real release flow).
  - If `packaging: uv` doesn't work with `setuptools.build_meta` backend, swap to a self-contained `python -m build` workflow.

### 5b. Broken pyproject.toml fails
- **Setup**: introduce a syntax error in `pyproject.toml` (e.g. unclosed bracket)
- **Expect**: ❌ build fails early
- **Reset**: revert

### 5c. Tagged build produces correct version
- **Setup** (optional, harder): `git tag v0.0.1-test && git push --tags`, then push a commit
- **Expect**: built wheel filename includes `0.0.1-test`
- **Reset**: `git tag -d v0.0.1-test && git push origin :refs/tags/v0.0.1-test`

---

## 6. Build docs (`build-docs.yml`)

This workflow runs on `push: main` and `push: pull-request/[0-9]+` (the copy-pr-bot mirror). To trigger it on this PR, the copy-pr-bot must have mirrored the branch.

### 6a. Clean build produces artifact
- **Expect**: ✅ Sphinx build completes, HTML artifact uploaded by the FW-CI template

### 6b. Broken cross-reference fails
- **Setup**: in any `.md` file under `docs/`, add `[text](nonexistent-page.md)` or a malformed `{ref}` directive
- **Expect**: ❌ Sphinx exits non-zero (depending on `-W` flag in the template; if warnings aren't errors, this may only print a warning — check the template's behavior)

### 6c. Malformed Markdown fails
- **Setup**: add a `.md` file under `docs/` with a missing closing code fence
- **Expect**: ❌ MyST parser errors

### 6d. Missing toctree entry fails or warns
- **Setup**: create `docs/orphan.md` not referenced from any toctree
- **Expect**: ⚠️ Sphinx warning ("document isn't included in any toctree") — escalates to failure only if `-W` is set in the template

### 6e. If `build-docs` doesn't run at all
- copy-pr-bot didn't mirror this PR. Check:
  - PR is not a draft (`auto_sync_draft: false`)
  - PR is ready (`auto_sync_ready: true`)
  - copy-pr-bot app is installed on the repo
- As a workaround for testing only, temporarily add `pull_request:` to `build-docs.yml` triggers, test, then revert.

---

## 7. Secrets detector (`secrets-detector.yml`)

### 7a. Fake AWS key fails
- **Expect**: ❌ detector flags the line

### 7b. Fake GitHub token fails
- **Setup**: same file with `ghp_` + 36 random chars
- **Expect**: ❌ detector flags it

### 7c. Real-looking-but-allowlisted strings pass
- **Setup**: revert the probes
- **Expect**: ✅ clean

**Reset**: `git rm _secret_probe.txt`

---

## 8. Semantic PR title (`semantic-pull-request.yml`)

### 8a. Non-conventional title fails
- **Setup**: edit the PR title to `add new feature`
- **Expect**: ❌ `Validate PR title` fails

### 8b. Wrong type prefix fails
- **Setup**: PR title `random: something`
- **Expect**: ❌ unrecognized type

### 8c. Valid title passes
- **Setup**: rename to e.g. `ci: add github actions workflows` or `feat: support custom rank strategy`
- **Expect**: ✅ passes

The allowed types follow Conventional Commits: `feat`, `fix`, `chore`, `docs`, `style`, `refactor`, `test`, `ci`, `build`, `perf`, `revert`.

---

## 9. Stale bot (`close-inactive-issue-pr.yml`)

Hard to test on a branch — it runs on `schedule: cron 30 1 * * *`. Smoke test:
- Trigger manually via Actions UI → "Run workflow" if `workflow_dispatch:` is added temporarily, or
- Just inspect the next scheduled run's log to confirm it ran and considered 0 issues (clean repo).

Don't merge a `workflow_dispatch:` trigger — only enable it temporarily for the test.

---

## 10. PR template (`PULL_REQUEST_TEMPLATE.md`)

- **Setup**: open a fresh PR from this branch
- **Expect**: the description box is pre-filled with the template (sections: What does this PR do?, Issues, Usage, Before your PR is "Ready for review", Additional information)

---

## 11. CODEOWNERS

- **Setup**: open a PR touching `src/nemo/lens/handle.py`
- **Expect**: `@ahmadki` is auto-requested as a reviewer
- **Setup**: open a PR touching `.github/workflows/unit-tests.yml`
- **Expect**: `@ahmadki` is auto-requested (from the `/.github/` rule)

---

## 12. copy-pr-bot

Best tested with a PR from a real fork (not a branch in the upstream repo). For now:
- Verify a `pull-request/N` branch appears in the upstream repo's branch list after the PR is opened from a fork
- Verify `build-docs.yml` runs on that branch
- Verify subsequent pushes to the PR update the mirror

If `copy-pr-bot.yaml` is misconfigured, the bot ignores the PR silently — there is no error message in the PR.

---

## 13. Final sanity sweep (after all individual checks pass)

- Push one final commit that does nothing CI-relevant (e.g. a doc typo fix). Verify:
  - Every PR check is green
  - No workflow ran that shouldn't have
  - The run times are reasonable (lint < 1 min, tests < 3 min, build-wheel < 2 min, build-docs < 3 min)
- Squash-merge into `main` and verify push-triggered workflows fire correctly:
  - `build-wheel.yml` (on `push: main`)
  - `build-docs.yml` (on `push: main`)
