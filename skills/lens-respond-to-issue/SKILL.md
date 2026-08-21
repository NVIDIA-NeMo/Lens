---
name: lens-respond-to-issue
description: Research and draft a maintainer response to a GitHub issue or community question on the NeMo Lens repo. Grounds the answer in the actual code and docs, verifies every citation, and hands the draft back for approval rather than posting it.
license: Apache-2.0
when_to_use: User shares a GitHub issue URL or number and wants a reply; answering a community question; 'respond to this issue', 'draft a reply', 'answer this GitHub question', 'what should I tell them'.
user_invocable: true
argument: "<github-issue-url-or-number>"
metadata:
  author: Ahmad Kiswani <akiswani@nvidia.com>
---

# Respond to a GitHub issue

Draft a high-quality maintainer response to an issue on `NVIDIA-NeMo/Lens`,
grounded in what the code actually does.

Most questions here come from users of a library they did not choose directly —
lens arrives as an optional dependency of Megatron-LM, NeMo-RL, or NeMo-Gym. Assume
the reporter knows their own framework well and lens barely at all.

## 1. Read the issue

```bash
gh issue view <number> --repo NVIDIA-NeMo/Lens --json title,body,comments,labels,state,author
```

Read the body and every existing comment before forming a view. Classify it —
the templates label them `bug`, `enhancement`, or `question` — and note which
consumer repo the reporter is coming from, since that usually determines the
answer.

## 2. Route it

Most questions land in one of a few buckets. Check the likely cause before
reading code at random:

| Symptom | Look first at |
|---|---|
| "No spans / nothing exported" | `NEMO_LENS_ENABLED` is `false` by default; then export strategy — `single_rank` means only the last rank exports |
| "Works locally, nothing in my backend" | `OTEL_EXPORTER_OTLP_*` is the SDK's business, not lens's; check their endpoint before reading lens code |
| "Some spans missing" | span groups — `default` is only `job`, `checkpoint`, `evaluate`. They likely want `per_step` or `all` |
| "ImportError / no-op behavior" | the optional-dependency contract — `docs/design/optional-dependency.mdx`, `fallbacks.py` |
| "Attribute or metric I expected isn't there" | `semconv.py` for the real name; then whether it's classified as a resource attribute, span attribute, or metric |
| "Colliding trace IDs across ranks" | `SeedIndependentIdGenerator` — a known, fixed bug worth linking |
| "How do I add X to my repo" | it's a consumer-side question; point at their `telemetry/` package, not lens |

## 3. Research the code

- Search `src/nemo/lens/` for the relevant module — the layout table in
  `AGENTS.md` says which file owns what.
- Read the source. Do not answer a behavior question from the docs alone; the
  docs are authoritative for intent, the code for what shipped.
- `git log --oneline -20 -- <files>` — was this recently changed or already fixed?
- `git log -S "<symbol>" --oneline` — when did this appear or disappear? Useful
  for "why is this gone" and "was this ever supported".
- Check whether a PR is already in flight:
  `gh pr list --repo NVIDIA-NeMo/Lens --search "<keywords>" --limit 5`
- Check for related issues:
  `gh issue list --repo NVIDIA-NeMo/Lens --search "<keywords>" --limit 5 --state all`

If the report involves a released version, check whether the behavior differs
from `main` — `package_info.py` carries the version, and the docs are split
between nightly (`docs/`) and the frozen `0.1.0` snapshot.

## 4. Verify before citing

Every concrete claim in the draft gets checked:

- Citing a file and line? Re-read it and confirm the line says what you claim.
- Citing a commit? `git show <sha> --stat` and confirm it does what you say.
- Claiming something is unused, missing, or unsupported? Grep thoroughly enough
  to be sure — a wrong "that's not supported" from a maintainer is expensive.
- Quoting an env var? Confirm the exact name and prefix in `config.py`. The
  `USER_ID` key maps to the `user` field; that mismatch is easy to get wrong.
- Recommending a config? Confirm the default in the `from_env()` table.

## 5. Draft

Write a reply that:

- Answers the actual question first, in the first line or two.
- Cites `file:line` where it helps the reader verify you.
- Acknowledges a real bug or gap plainly when the reporter found one. They did
  you a favor; say so.
- Gives the workaround if one exists, even a clumsy one.
- Says whether a fix is planned, and that a PR would be welcome if it would be.
- Stays short. Contributors want an answer, not an essay.
- Stays welcoming — many reporters are first-time contributors to an NVIDIA repo.

If the honest answer is "that's a Megatron-LM / NeMo-RL / NeMo-Gym question",
say so and point at the right repo, but answer the lens-shaped part first rather
than bouncing them cold.

## 6. Offer the follow-up

If the issue exposes something cleanly actionable — a doc page that is wrong, a
missing fallback signature, a one-line fix — tell the maintainer and offer to
open a branch and PR. Don't stop at drafting a comment when the fix is small.

If it touches the public surface — `__all__`, a `fallbacks.py` signature, a
`NemoLensConfig` field, a `SpanRegistry` method — say so explicitly. The
consumer mirror obligation applies (Megatron-LM, NeMo-RL, NeMo-Gym each carry
their own copy of that contract) and it is easy to forget from an issue thread.

## 7. Hand the draft back

Show the draft to the maintainer as a fenced markdown block they can copy.
**Do not post it.** Posting is the maintainer's decision, always.

## Guidelines

- Never post or edit anything on GitHub without explicit approval.
- Flag your own uncertainty in the draft rather than smoothing it over — the
  maintainer needs to know which parts they are vouching for.
- If the code cannot settle the question, say what you established and what
  remains open. That is a useful answer; a confident wrong one is not.
- Don't promise a timeline, a release, or a roadmap position on the project's behalf.
