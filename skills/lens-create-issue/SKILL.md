---
name: lens-create-issue
description: Investigate a failing GitHub Actions run or job in the NeMo Lens repo and file a well-structured GitHub issue for the failure. Covers log extraction for each of this repo's CI jobs, duplicate detection, triggering-PR resolution, and assignee lookup.
license: Apache-2.0
when_to_use: User shares a GitHub Actions URL and wants a bug filed; triaging a red CI run; 'create an issue for this failure', 'file a bug for this CI run', 'triage this GitHub Actions failure', 'CI is red, open an issue'.
user_invocable: true
argument: "GitHub Actions run or job URL"
metadata:
  author: Ahmad Kiswani <akiswani@nvidia.com>
---

# Triage a CI failure into a GitHub issue

Investigate a failing GitHub Actions job, extract the root cause from the logs,
and file a bug against `NVIDIA-NeMo/Lens`.

Everything here quotes the actual log. Do not guess a root cause — if the logs
do not show one, say so and file what is known.

## 1. Parse the URL

The argument is a GitHub Actions URL, in one of two shapes:

- **Job**: `https://github.com/<owner>/<repo>/actions/runs/<run_id>/job/<job_id>`
- **Run**: `https://github.com/<owner>/<repo>/actions/runs/<run_id>`

Extract `run_id` and, if present, `job_id`.

## 2. Identify the failed jobs

With a `job_id`, use it directly. With only a `run_id`, list what failed:

```bash
gh run view <run_id> --repo NVIDIA-NeMo/Lens --json jobs \
  --jq '[.jobs[] | select(.conclusion == "failure") | {id: .databaseId, name: .name, url: .url}]'
```

If several jobs failed, ask which to triage — unless the user already said to
take all of them.

## 3. Fetch the failure logs

```bash
gh api repos/NVIDIA-NeMo/Lens/actions/jobs/<job_id>/logs 2>&1 \
  | grep -E "(FAILED|ERROR|\bError\b|assert|Traceback|Exception|##\[error\])" \
  | head -200

gh run view --job <job_id> --repo NVIDIA-NeMo/Lens --json name --jq .name
```

If the grep comes back sparse, pull the full log and look for the pytest
`FAILURES` section or the last non-zero exit.

### What failure looks like per job

This repo's CI jobs fail in distinct ways, and only one of them produces pytest
node IDs. Identify the job first, then read for the right signal:

Job names below are the display names as they appear in the Actions UI, not the
workflow filenames.

| Job | Reads as | Signal to extract |
|---|---|---|
| `Test (Python 3.12)` / `(Python 3.13)` | pytest | `FAILED tests/...::...` node IDs and the assertion or traceback |
| `Lint check` | pre-commit | the failing hook id and its diff or error block |
| `Copyright check` | template workflow | the file paths missing an SPDX header |
| `Secrets detector` | template workflow | the flagged path — **do not quote the matched secret in the issue** |
| `Validate PR title` | Conventional Commits lint | not a code failure; the PR title is malformed. Tell the user to retitle rather than filing a bug |
| `Fern Docs CI` | docs build / link check | the MDX file and the broken link or tag |

The test matrix is Python 3.12 and 3.13. A failure on **both** is a real
regression; a failure on one is version-specific and worth saying so in the
issue, since it changes who needs to look at it.

## 4. Resolve the triggering PR

CI runs against `pull-request/NNN` mirror branches created by copy-pr-bot, not
against the PR branch, so the head branch carries the PR number:

```bash
gh run view <run_id> --repo NVIDIA-NeMo/Lens --json headBranch,event --jq '{branch: .headBranch, event: .event}'
# branch → e.g. "pull-request/42"

gh pr view <pr_number> --repo NVIDIA-NeMo/Lens --json number,title,url \
  --jq '{number: .number, title: .title, url: .url}'
```

**If `event` is `schedule`**, this is the nightly run against `main` and there is
no triggering PR. Say so in the issue, drop the PR row from the table, and note
that the failure is on `main` — that raises its severity, not lowers it.

## 5. Find an assignee

The author of whichever source or test file the failure points at:

```bash
gh api "repos/NVIDIA-NeMo/Lens/commits?path=<file-path>&sha=main&per_page=1" \
  --jq '.[0] | {login: .author.login, name: .commit.author.name, sha: .sha}'
```

If empty — the file was introduced by the PR itself — query the PR's commits:

```bash
gh api "repos/NVIDIA-NeMo/Lens/pulls/<pr_number>/commits" \
  --jq '[.[] | select(.files? // [] | any(.filename == "<file-path>"))] | .[0].author.login'
```

If the lookup fails or returns a bot, skip `--assignee` and say so under
Additional context rather than guessing.

## 6. Check for duplicates first

```bash
gh issue list --repo NVIDIA-NeMo/Lens --state open \
  --search "<failing-test-name-or-file>" --json number,title,url --limit 10
```

If an open issue already covers it, **do not file another**. Report the existing
issue and stop.

## 7. File the issue

The repo's templates set labels: `bug`, `enhancement`, `question`. A CI failure
is always `bug`.

```bash
gh issue create \
  --repo NVIDIA-NeMo/Lens \
  --title "🐛 CI failure: <failed-test-node-id-or-job-name>" \
  --label bug \
  --assignee "<login>" \
  --body "..."
```

Match `.github/ISSUE_TEMPLATE/bug_report.md`'s section order so the issue reads
like a hand-filed one:

````markdown
**Describe the bug**

CI job [`<job-name>`](<job-url>) failed on `<failed-test-node-id>`.

| Field | Value |
|-------|-------|
| PR    | [#<pr_number>: <pr_title>](<pr_url>) |
| Run   | [<run_id>](<run_url>) |
| Job   | [<job_name>](<job_url>) |

```
<core error message / traceback — 30 lines max>
```

**Steps/Code to reproduce bug**

Re-run the linked CI job, or locally:

```bash
uv venv && uv pip install -e . --group dev
pytest <failed-test-node-id>
```

**Expected behavior**

The test passes, as it does on `main`.

**Environment details**

- CI runner: `ubuntu-latest`, Python `<matrix-version>`
- nemo-lens: `<commit-sha>`

**Additional context**

Triaged automatically from the CI run linked above.
````

For a non-pytest job, replace the reproduce block with that job's local
equivalent — `pre-commit run --all-files` for Lint check, `npm --prefix
docs/fern run check` for the docs jobs.

If several tests failed in one job, bullet each under "Describe the bug" with a
combined error snippet, and assign on the first file in the list.

## 8. Report back

Print the URL of the new issue — or of the duplicate you found instead.

## Guidelines

- Never open a duplicate. Link the existing issue.
- Keep the error snippet to ~30 lines; the job URL carries the full log.
- Quote the log verbatim. Do not infer a cause the log does not state.
- Never paste a matched credential from a `Secrets detector` failure into a
  public issue — reference the file and line only.
- If the run is still in progress or the logs have expired, say so and ask the
  user to retry rather than filing a thin issue.
- A `Validate PR title` failure is a malformed PR title, not a bug. Say that and
  skip the issue.
