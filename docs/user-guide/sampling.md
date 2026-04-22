# Sampling

Sampling controls how much of the raw telemetry stream actually reaches your backend. Lens exposes two layers of control that compose:

1. **Export strategy** (rank-level): which ranks send data at all.
2. **Sampler** (SDK-level): which spans on an exporting rank are kept.

Understand the difference before picking a configuration.

## Export strategy

Configured via `NemoLensConfig.export_strategy` / `NEMO_LENS_EXPORT_STRATEGY`:

### `single_rank` (default)

Only one rank sends telemetry. The rest get no-op providers and create no spans at all — non-exporting ranks skip span creation entirely, not merely record no-ops.

```bash
NEMO_LENS_EXPORT_STRATEGY=single_rank
NEMO_LENS_EXPORT_RANK=-1    # last rank; 0 for first rank
```

**Use when**: typical production training, where one rank's view is representative (loss, iteration time, etc.) and exporting from 1000 ranks would overwhelm the backend.

### `all_ranks`

Every rank sends telemetry. Overhead multiplies by world size at the collector.

**Use when**: debugging a specific issue that might manifest on only some ranks (e.g. hang on rank 7, NaN on rank 42).

### `sampled`

Hash-based deterministic sampling of ranks. The `export_sample_rate` fraction of ranks export; the rest are no-op.

```bash
NEMO_LENS_EXPORT_STRATEGY=sampled
NEMO_LENS_EXPORT_SAMPLE_RATE=0.1    # 10% of ranks
```

Sampling is deterministic by rank — same rank + rate → same decision across re-runs.

**Use when**: large jobs (1000+ ranks) where `all_ranks` is too much but you want more than one rank's perspective.

## `RankAwareSampler`

For finer control at the SDK level, lens ships a `Sampler` implementation at `nemo.lens.sampling.RankAwareSampler`. It implements the OTel `Sampler` interface so it plugs into `TracerProvider(sampler=...)`.

Enable via config:

```bash
NEMO_LENS_SAMPLER_ENABLED=1
NEMO_LENS_EXPORT_SAMPLE_RATE=0.1
```

Or programmatically:

```python
cfg = NemoLensConfig(
    enabled=True,
    sampler_enabled=True,
    export_sample_rate=0.1,
)
```

### Behaviour

- Decision is **per-rank**, not per-span: once a rank is sampled, *all* its spans export; otherwise *none* do.
- Deterministic: same rank + rate → same decision.
- Composes with export strategy: `all_ranks` + `sampler_enabled=1` gives uniform random rank sampling at SDK level.

### When to use it vs export strategy

The export strategy works at provider construction: non-exporting ranks get no-op providers, so they don't even create spans. The sampler works at span creation: spans are created, then dropped.

| Goal | Better choice |
|---|---|
| 1000-rank training, only rank 0 matters | `single_rank` (no sampler; non-exporting ranks skip span creation entirely) |
| 1000-rank training, want random 10% of ranks | `all_ranks` + `sampler_enabled=1` with rate 0.1 |
| Debugging, want spans on all ranks for local inspection | `all_ranks` (no sampler) |

Prefer the export strategy when it fits — it's cheaper because non-exporting ranks skip span creation entirely, whereas the sampler still builds and then drops spans.

## OTel SDK samplers

For **per-trace** sampling (not per-rank), use standard OTel SDK samplers via env vars:

```bash
OTEL_TRACES_SAMPLER=parentbased_traceidratio
OTEL_TRACES_SAMPLER_ARG=0.1    # keep 10% of traces
```

This is orthogonal to lens's sampler — you can run both. Typical production setup: export from one rank (`single_rank`), sample 10% of traces at that rank (`parentbased_traceidratio`), because even one rank of `per_step` spans for a long job is a lot of data.

## Composing samplers

| Export strategy | `sampler_enabled` | `OTEL_TRACES_SAMPLER` | Result |
|---|---|---|---|
| `single_rank` | `0` | default (always_on) | All spans from one rank |
| `single_rank` | `0` | `traceidratio 0.1` | 10% of traces from one rank |
| `all_ranks` | `1` (rate 0.1) | default | All spans from 10% of ranks |
| `all_ranks` | `1` (rate 0.1) | `traceidratio 0.1` | 10% of traces from 10% of ranks (1% total) |

## Cost implications

- Span creation has a per-span cost that scales with attribute count and recording decisions; non-exporting ranks skip this entirely.
- `BatchSpanProcessor` adds queueing work on top of span creation (only on exporting ranks).
- Network egress scales with export volume: 1000 ranks × 100 spans/step × 1 step/sec = 100,000 spans/sec, which is more than most collectors want to handle. Sample.
