# Semantic Conventions

`nemo.lens.semconv` centralizes every attribute name constant NeMo Lens emits. Two categories of attribute coexist:

1. **Standard OTel semconv**: `gen_ai.*` and `k8s.*`. These track upstream OTel specs. (Resource attributes like `service.*` and `host.*` are also standard OTel names but are set as literals in `providers.py` and `resources/local.py` rather than mirrored as `semconv.py` constants.)
2. **NeMo custom namespaces**: `dl.*` (distributed learning), `rl.*`, `gym.*`, `slurm.*`, `nemo.*`, `wandb.*`. These are NeMo-specific extensions that do not exist upstream.

## Why Constants Instead of Strings

Using constants instead of raw strings provides three key benefits:

1. **Grep-ability**: renaming an attribute across the codebase means changing one constant, not every call site.
2. **Type safety** (weak but real): `DL_RANK` is an exported name; typos become `ImportError`s. `"dl.rank"` typos become silent data loss.
3. **Central registry**: one file lists every attribute lens might emit. Easy to review, easy to document.

Callers who want the string can use `DL_RANK` directly; Python strings-as-constants have no boxing cost.

## Version Tracking

```python
SEMCONV_VERSION = "1.29.0"
```

This constant documents which upstream OTel semconv version the standard namespaces (`gen_ai.*`, `k8s.*`) are aligned with. Upstream bumps namespace conventions periodically; `gen_ai.*` graduated from experimental to stable around 1.30.

When upgrading NeMo Lens to a new semconv version, complete these tasks:

1. Review the upstream changelog for renamed or removed attributes.
2. Update `SEMCONV_VERSION`.
3. Update any changed constants.
4. Add the version bump to the NeMo Lens changelog so consumers know to update.

## Stability Markers

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

"Stable" for custom namespaces means these names will not change in minor releases. Breaking changes bump the major version.

"Experimental" for `gen_ai.*` matches upstream — OTel considers them stable-in-practice but reserves the right to tweak until the full semconv 1.30 stabilisation.

## Namespace Conventions

NeMo Lens organizes attributes into standard upstream namespaces and NeMo-specific custom namespaces.

### Standard Upstream Namespaces

Follow upstream OTel spec exactly. Don't re-define, don't rename. If upstream says `gen_ai.request.model`, lens uses `gen_ai.request.model`.

### `dl.*` Distributed Learning

Shared across Megatron-LM, NeMo RL, NeMo Gym. Anything a distributed training job needs:

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

### `<project>.*` Project-Specific Attributes

Use project-specific namespaces for attributes that apply to one consumer rather than the full NeMo ecosystem:

- `megatron.*`: Megatron-LM consumer-specific span attributes (e.g., model architecture `megatron.num_layers`/`megatron.hidden_size`, iteration-level `megatron.skipped`/`megatron.update_successful`). These are set as inline string attributes in the Megatron-LM fork and are **not** defined as constants in `nemo.lens.semconv`. The central registry only holds the shared and standard namespaces (`dl.*`, `gen_ai.*`, `rl.*`, `gym.*`, `slurm.*`, `nemo.*`, `wandb.*`, `k8s.*`).
- `rl.*` — RL-specific (`reward`, `kl_divergence`, `policy_loss`)
- `gym.*` — Gym server-specific (`verify.success_rate`, `rollout.batch_size`)

### `nemo.*` NeMo-Wide Identification

- `nemo.run.id` — unique run identifier, shared across all ranks
- `nemo.user.id` — optional team/user label

### Environment Namespaces

- `slurm.*` — SLURM job attributes
- `k8s.*` — Kubernetes pod/node attributes (standard OTel)
- `wandb.*` — W&B Weave integration metadata

## Adding New Attributes

Checklist before adding a constant:

1. **Is there an upstream OTel semconv name?** Use it.
2. **Is this shared across consumers?** Use `dl.*` or another shared namespace.
3. **Is this project-specific?** Use `<project>.*`. Such namespaces are set inline in the consumer fork (e.g. Megatron-LM), not as constants in `semconv.py`.
4. **Is this a metric or a span attribute?** Metric instruments go in `instruments/`, span attributes go in `semconv.py`.
5. **Is the attribute always available?** If not, document it as "optional" so query authors know to handle missing values.

Do not add attributes speculatively. A constant with no call site is dead code that becomes stale.

## Attribute vs. Metric Decision

| Value | Goes where |
|---|---|
| Categorical, per-span (which iteration, what kind of request) | Span attribute |
| Stable for the process lifetime (rank, parallelism config) | Resource attribute |
| Numerical, varies over time (loss, duration) | Metric |
| Text payload (prompt, completion) | Span event, or don't emit (PII risk) |

Putting a continuously-varying value like loss on a span attribute produces thousands of span-attribute time series in Jaeger that can't be aggregated. Use a metric.

## Attribute Cardinality

OTel metric attributes have a cardinality budget — each distinct combination of attribute values creates a new time series. `gen_ai.operation.name="text_completion"` is fine (one value). `gen_ai.request.model="llama-3-8b"` is fine (a handful of values). `user.session.id="abc-123"` is dangerous (unbounded cardinality — one time series per user session).

Rule of thumb: metric attributes should have **tens to hundreds** of distinct values, not millions. Use span attributes for high-cardinality values.
