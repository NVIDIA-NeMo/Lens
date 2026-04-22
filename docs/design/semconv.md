# Semantic Conventions

`nemo.lens.semconv` centralises every attribute name constant lens emits. Two categories of attribute coexist:

1. **Standard OTel semconv** — `gen_ai.*`, `k8s.*`, `service.*`, `host.*`. These track upstream OTel specs.
2. **NeMo custom namespaces** — `dl.*` (distributed learning), `rl.*`, `gym.*`, `slurm.*`, `nemo.*`, `wandb.*`. These are NeMo-specific extensions that don't exist upstream.

## Why constants instead of strings

Three reasons:

1. **Grep-ability**: renaming an attribute across the codebase means changing one constant, not every call site.
2. **Type safety** (weak but real): `DL_RANK` is an exported name; typos become `ImportError`s. `"dl.rank"` typos become silent data loss.
3. **Central registry**: one file lists every attribute lens might emit. Easy to review, easy to document.

Callers who want the string can use `DL_RANK` directly — Python strings-as-constants have no boxing cost.

## Version tracking

```python
SEMCONV_VERSION = "1.29.0"
```

This documents which upstream OTel semconv version the standard namespaces (`gen_ai.*`, `k8s.*`) are aligned with. Upstream bumps namespace conventions periodically — `gen_ai.*` graduated from experimental to stable around 1.30. When upgrading lens to a new semconv version:

1. Review the upstream changelog for renamed or removed attributes.
2. Update `SEMCONV_VERSION`.
3. Update any changed constants.
4. Note the version bump in lens's changelog so consumers know to update.

## Stability markers

From `semconv.py`:

```
# gen_ai.*  — Experimental (upstream, stabilising in semconv 1.30+)
# k8s.*     — Stable (upstream)
# dl.*      — NeMo custom (stable within NeMo ecosystem)
# rl.*      — NeMo custom (stable within NeMo ecosystem)
# gym.*     — NeMo custom (stable within NeMo ecosystem)
# slurm.*   — NeMo custom (stable within NeMo ecosystem)
# nemo.*    — NeMo custom (stable within NeMo ecosystem)
# wandb.*   — NeMo custom (stable within NeMo ecosystem)
```

"Stable" for custom namespaces means "we commit to not renaming these in minor releases." Breaking changes bump the major version.

"Experimental" for `gen_ai.*` matches upstream — OTel considers them stable-in-practice but reserves the right to tweak until the full semconv 1.30 stabilisation.

## Namespace conventions

### Standard (upstream) namespaces

Follow upstream OTel spec exactly. Don't re-define, don't rename. If upstream says `gen_ai.request.model`, lens uses `gen_ai.request.model`.

### `dl.*` — distributed learning

Shared across Megatron-LM, NeMo-RL, NeMo-Gym. Anything a distributed training job needs:

```
dl.rank                      — global rank
dl.world_size                — total ranks
dl.local_rank                — rank on this node
dl.tensor_parallel.{rank,size}
dl.pipeline_parallel.{rank,size}
dl.data_parallel.{rank,size}
dl.iteration                 — training iteration
dl.microbatch_id             — microbatch index
dl.batch_size, dl.sequence_length
dl.loss, dl.grad_norm, dl.learning_rate
dl.throughput_tflops, dl.throughput_tokens_per_sec
```

### `<project>.*` — project-specific

For attributes that don't generalise across the ecosystem:

- `megatron.*` — model architecture (`num_layers`, `hidden_size`), iteration-level (`skipped`, `update_successful`)
- `rl.*` — RL-specific (`reward`, `kl_divergence`, `policy_loss`)
- `gym.*` — Gym server-specific (`verify.success_rate`, `rollout.batch_size`)

### `nemo.*` — NeMo-wide identification

- `nemo.run.id` — unique run identifier, shared across all ranks
- `nemo.user.id` — optional team/user label

### Environment namespaces

- `slurm.*` — SLURM job attributes
- `k8s.*` — Kubernetes pod/node attributes (standard OTel)
- `wandb.*` — W&B Weave integration metadata

## Adding new attributes

Checklist before adding a constant:

1. **Is there an upstream OTel semconv name?** Use it.
2. **Is this shared across consumers?** Use `dl.*` or another shared namespace.
3. **Is this project-specific?** Use `<project>.*`.
4. **Is this a metric or a span attribute?** Metric instruments go in `instruments/`, span attributes go in `semconv.py`.
5. **Is the attribute always available?** If not, document it as "optional" so query authors know to handle missing values.

Don't add attributes speculatively. A constant with no call site is dead code that rots.

## Attribute vs metric decision

| Value | Goes where |
|---|---|
| Categorical, per-span (which iteration, what kind of request) | Span attribute |
| Stable for the process lifetime (rank, parallelism config) | Resource attribute |
| Numerical, varies over time (loss, duration) | Metric |
| Text payload (prompt, completion) | Span event, or don't emit (PII risk) |

Putting a continuously-varying value like loss on a span attribute produces thousands of span-attribute time series in Jaeger that can't be aggregated. Use a metric.

## Attribute cardinality

OTel metric attributes have a cardinality budget — each distinct combination of attribute values creates a new time series. `gen_ai.operation.name="text_completion"` is fine (one value). `gen_ai.request.model="llama-3-8b"` is fine (a handful of values). `user.session.id="abc-123"` is dangerous (unbounded cardinality — one time series per user session).

Rule of thumb: metric attributes should have **tens to hundreds** of distinct values, not millions. Use span attributes for high-cardinality values.
