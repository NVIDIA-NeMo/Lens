# `model/` — the semantic-convention registry (source of truth for attribute, span and event names)

This directory **is** the definition of every attribute, span and event name Lens
exposes. One YAML record per name; the Python constants in `../attributes.py`,
`../spans.py` and `../events.py` are **generated from here** by
[weaver](https://github.com/open-telemetry/weaver). Editing a name
is a reviewable diff against the registry, not an untracked edit to a Python
module nobody reviews.

The registry uses the OTel semconv shape: each file is one `attribute_group`
(`id: registry.<ns>`) whose `attributes:` list carries the names. Each attribute
pins the Python constant it generates via `annotations.constant`, so the emitted
name is exact (e.g. `gen_ai.operation.name` → `GENAI_OPERATION_NAME`, which a
mechanical `screaming_snake_case` would get wrong).

Span and event names live in `nv_dl_spans.yaml` as `type: span` / `type: event`
groups (`id: span.<name>` / `id: event.<name>`), each with its own
`annotations.constant`. Their `attributes:` are `ref`s to attributes defined in
the attribute groups, so a span can only carry names this registry defines.
They generate into separate modules (`nemo.lens.semconv.spans`,
`nemo.lens.semconv.events`), not into the `nemo.lens.semconv` star-export.

## Where the names live now

```
model/*.yaml                     ← source of truth (this dir)
  │  weaver registry generate
  ▼
_generated/{attributes,spans,events}.py   ← GENERATED constants (do not edit)
  │  from ._generated.<kind> import *
  ▼
attributes.py                    ← thin shim: re-exports + owns SEMCONV_VERSION
spans.py, events.py              ← thin shims: re-export + ALL_SPANS / ALL_EVENTS
```

## The rules

1. **The model is the source of truth.** `../attributes.py` and
   `../_generated/attributes.py` are derived. Never hand-edit either.

2. **To add or change a name, edit the model and regenerate** (below), in the
   **same PR**. Two tests enforce this:
   - `tests/test_semconv_generated.py` regenerates and asserts the committed
     `_generated/attributes.py` is byte-identical (runs where weaver is present;
     `REQUIRE_WEAVER=1` makes a missing weaver a failure, not a skip).
   - `tests/test_semconv_model_parity.py` is weaver-independent and always runs:
     it asserts the model and the importable constants are the same name set.

3. **Renaming a name is not a local change.** `lens-monitor` classifies spans
   through a hand-written name → bucket map. A name missing from that map is
   silently skipped and its time lands in `unobserved`, charged as **BAD** — so a
   rename *lowers measured goodput with no error surfaced anywhere*. The map
   already carries `# old name` entries; this has happened before. A rename is a
   cross-repo change with an owner conversation.

4. **Namespace ownership** (who a change must go through):

   | Namespace(s) | Owner | Briefs |
   |---|---|---|
   | `nv.dl.*`, `nv.gpu.*`, `slurm.*`, `inverted.*` | Lens / NVIDIA | authored |
   | `nemo.*`, `wandb.*` | Lens-internal | `TODO` (intentional) |
   | `dl.*` | legacy, future undecided | `TODO` (intentional) |
   | `rl.*`, `gym.*` | **NeMo-RL** — RL *metrics* are declared in that tree, not here | `TODO` (intentional) |
   | `gen_ai.*`, `k8s.*`, `host.*` | **upstream OTel** | pointer brief only |

5. **`stability: development` on every record means no compatibility promise
   yet.** Names may still change while the spelling conflicts (below) are settled.

## Regenerating

Weaver is a Rust binary, pinned to the version the CI job installs:

```bash
# install weaver (macOS arm64 shown; see the OTel weaver releases for other OSes)
curl -sL https://github.com/open-telemetry/weaver/releases/download/v0.25.1/weaver-aarch64-apple-darwin.tar.xz \
  | tar xJ --strip-components=1 -C "$HOME/.local/weaver"

# regenerate after ANY model edit, then commit the result
weaver registry generate -r src/nemo/lens/semconv/model -t templates python /tmp/gen
cp /tmp/gen/*.py src/nemo/lens/semconv/_generated/
```

The templates are `templates/registry/python/{attributes,spans,events}.py.j2`
(+ `weaver.yaml`).

## Inventory

Twelve namespaces, **137 records** across 12 files (+ `manifest.yaml`, which carries
the registry `schema_url` and version), plus **31 spans and 3 events** in
`nv_dl_spans.yaml`.

- **Spans and events (`nv_dl_spans.yaml`)** are spelled as producers emit them
  today (Megatron and the launch scripts), verified against production traces.
  Three startup spans (`initialize_megatron`, `jit_fusion_options`,
  `weight_hash_check`) are Megatron-specific; their notes say so, and they are
  slated to move to `nv.mlm.*`.

- **Authored (`nv.dl.*`, `nv.gpu.*`, `slurm.*`, `inverted.*`):** no `TODO` remains.
- **Upstream (`gen_ai.*`, `k8s.*`, `host.*`):** standard OTel names, not Lens
  semantics. Their brief is a pointer to upstream (not a local description that
  could drift); they track OTel semconv **v1.29.0** (`SEMCONV_VERSION`).
- **`TODO` on purpose (`dl.*`, `rl.*`, `gym.*`, `nemo.*`, `wandb.*` — 29
  records):** the correct outcome, not unfinished. `dl.*` is legacy; `rl.*`/`gym.*`
  are NeMo-RL's; `nemo.*`/`wandb.*` are Lens-internal. **Do not guess a brief** —
  blank signals they need an owner decision. These still carry a placeholder
  `type: string`; settle the type when the brief is written.

## Notes for the next reader

- **`inverted.*` (4 records)** were added by `fix: preserve explicit spans across
  clock inversions (#75)`. Their `*_seconds` values are string-encoded on the wire
  (`span_utilities.py`), so they are `type: string`; `inverted.skew` is `boolean`.
- **Three `slurm.*` names are RETIRED** — `slurm.cluster`, `slurm.nodelist`,
  `slurm.torchelastic.restart_count` are kept as constants (in
  `SLURM_RETIRED_RESOURCE_ATTRIBUTE_KEYS`, `../resources.py`) but no longer emitted.
  Use `slurm.cluster.name`, not `slurm.cluster`.

## Deliberately deferred

- **Spelling conflicts.** Lens spells some names one way, the standalone registry
  or Megatron another (`micro_batch_size` vs `microbatch_size`, `train_iters` vs
  `train_iterations`, `nv.nvrx.ckpt.*`
  vs `nv.nvrx.checkpoint.*`). The registry records **Lens's current spelling**;
  choosing the winner is an owner conversation held against these files. Resolving
  one is a rename — see rule 3.
