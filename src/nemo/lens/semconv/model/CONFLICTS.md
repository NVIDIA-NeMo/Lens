# Spelling conflicts — a reviewable register, not a set of renames

Some names in this registry are spelled one way by **Lens** (what
`../attributes.py` emits today), another by the **standalone registry**
([`nv-training-semconv`](https://github.com/NVIDIA/nv-training-semconv), which is
settling canonical names across the NeMo DL ecosystem), and a third by the
**producer** that actually emits them today (Megatron-LM / `nvidia-resiliency-ext`).
Same concept, different word. None of them is wrong; the ecosystem just has not
picked one yet.

This file **records** those conflicts so the owners can decide against a concrete
list. It does **not** rename anything. Changing a name here is a separate,
deliberate act with a cross-repo blast radius — see
[What resolving a row costs](#what-resolving-a-row-costs) below.

> **This registry keeps Lens's current spelling.** Where a row below names a
> different canonical, that is the open question, not a decision already taken.
> `stability: development` on every record is what reserves the right to change
> it once a row is resolved.

## Tier 1 — Lens emits a conflicting spelling today (2)

These are the load-bearing ones: Lens **already** defines the constant, so
resolving the conflict is a **rename in Lens** plus the cross-repo follow-through.
Both are `int` **attributes** (not spans — see the cost section).

| Concept | Kind | Lens spelling (this registry) | Standalone-registry canonical | Producer emits today | Producer location |
|---|---|---|---|---|---|
| Configured micro-batch size | attr | `nv.dl.training.config.micro_batch_size` | `nv.dl.training.config.microbatch_size` | `megatron.micro_batch_size` | `nv-mistralai-megatron/.../global_vars.py:425` |
| Target training iterations | attr | `nv.dl.training.target.train_iters` | `nv.dl.training.target.train_iterations` | `megatron.train_iters` | `nv-mistralai-megatron/.../global_vars.py:424` |

Both Lens records carry an inline note flagging the conflict and deferring it to
`model/README.md` (which routes here) — see `nv_dl.yaml`. The Lens spelling
tracks the **producer** word (`micro_batch_size`, `train_iters`); the standalone
registry chose the expanded form (`microbatch_size`, `train_iterations`). Picking
either means one side renames.

## Tier 2 — Lens does not emit these yet; decide the spelling *before* it adopts them (11)

Lens does not define these names. They are forks **between the standalone
registry and the producer**, and Lens is the likely future consumer. Settling the
spelling now is free; settling it after Lens adopts a side turns into a Tier-1
rename. Deciding not to adopt at all is also a valid outcome — these are not Lens
commitments.

| Concept | Kind | Standalone-registry canonical | Producer emits today | Producer location |
|---|---|---|---|---|
| Per-iteration step counter | attr | `nv.dl.training.step` | `megatron.iteration` | `nv-mistralai-megatron/.../training.py:4681` |
| Training-loop pass span | span | `span.nv.dl.training.loop_pass` | `megatron.train` | `nv-mistralai-megatron/.../training.py:615` |
| Async-checkpoint save spans (systematic `ckpt` → `checkpoint`) | span ×9 | `span.nv.nvrx.checkpoint.save.{request,schedule,stage,stage_wait,preload,shm_drain,write,finalize,completion_sync}` | `nvrx.ckpt.save.{…}` | `nvidia-resiliency-ext/.../async_ckpt/` — 7 in `core.py`, 2 (`shm_drain`, `stage`) in `filesystem_async.py` |

The nine `ckpt` → `checkpoint` spans are **one** decision, not nine: a single
word choice applied across the async-save span tree.

> **`iter_block` is informational, not a third fork.** The live loop-pass fork is
> `loop_pass` (registry canonical) vs `megatron.train` (producer). `iter_block`
> is an alternative the standalone registry **already rejected** in favour of
> `loop_pass` (its own prose: *"Named `loop_pass` rather than `iter_block`"*, in
> the standalone registry's `model/nv/nv.dl.yaml` — not a Lens file). It is noted
> only so Lens does not reach for `iter_block` by reflex; it is not an open choice.

## What resolving a row costs

The cost of a rename differs by **kind**, and it is not uniform across the rows
above — so do not treat every row as equally risky.

**Span renames (Tier 2: `loop_pass` + the 9 `nvrx.checkpoint.save.*`).** These are
the hazardous ones. `lens-monitor` classifies every **span** through a
hand-written name → bucket map (see `../README.md` rule 3). A span name the map
does not know is silently skipped and its time lands in `unobserved`, charged as
**BAD** — so a span rename not mirrored in the map *lowers measured goodput with
no error surfaced anywhere*. This cuts **both** directions: at cut-over the map
must learn the **new** name (or the producer's fresh emissions go to BAD) **and**
keep the **old** one as an alias (or in-flight data is reclassified). The map
already carries `# old name` entries for exactly this.

**Attribute renames (both Tier-1 rows + Tier-2 `step`).** These are `int`
attributes, not spans; they do **not** pass through the goodput bucket map, so a
rename cannot reclassify time to BAD. The cost is instead **dashboard and query
breakage** — anything that reads the old attribute key (Kibana queries, saved
searches) must move to the new one. Real, cross-repo, but a different and lower
blast radius than a span rename.

So resolving any row is a **coordinated cross-repo change**, in this order:

1. The namespace owner picks the winning spelling (see ownership below).
2. Edit the model here and regenerate (`../README.md` → Regenerating), in one PR.
3. **For a span rename only:** in the same change window, update the lens-monitor
   name → bucket map to map **both** the new and the old spelling to the right
   bucket, so neither fresh nor in-flight span time lands in BAD.
4. **For an attribute rename:** update the dashboard/query consumers of the old
   key in the same window.
5. The producer (Megatron / resiliency-ext) and any downstream consumer move
   together with the change.

## Ownership (who decides each row)

The tier split is about **which repo currently carries the name**, not divided
authority over the namespace. `nv.dl.*` has one owner; it is the producer side
that differs.

| Rows | Decision owner | Co-ordinates with |
|---|---|---|
| Tier 1 + Tier 2 `nv.dl.training.*` (`micro_batch_size`, `train_iters`, `step`, `loop_pass`) | `nv.dl.*` owner — Lens / NVIDIA DL-telemetry | Megatron producer |
| Tier 2 `nv.nvrx.checkpoint.*` | `nvidia-resiliency-ext` owner (`nv.nvrx.*`) | — |

Source of the producer spellings and locations: the standalone registry's
`MIGRATION.yaml` (`RENAME` entries) and `docs/decisions/NAMESPACE_MAP.md`.
This register is a snapshot; re-derive from those files before acting on a row.
