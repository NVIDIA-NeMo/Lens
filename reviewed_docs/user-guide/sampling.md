# Sampling

Sampling controls how much of the raw telemetry stream actually reaches your backend. NeMo Lens exposes two layers of control that work together:

1. **Export strategy (rank-level).** Controls which ranks send data.
2. **Sampler (SDK-level).** Controls which spans on an exporting rank are kept.

For strategy logic beyond the built-ins, see [Custom Strategies](custom-strategies.md).

Review the differences between these two layers before choosing a configuration.

## Choose an Export Strategy

Configure the export strategy through `NemoLensConfig.export_strategy` or the `NEMO_LENS_EXPORT_STRATEGY` env var:

### `single_rank` (Default)

Only one rank sends telemetry. The remaining ranks get no-op providers and create no spans; non-exporting ranks skip span creation entirely instead of recording no-op spans.

```bash
NEMO_LENS_EXPORT_STRATEGY=single_rank
NEMO_LENS_EXPORT_RANK=-1    # last rank; 0 for first rank
```

**Recommended use.** Use this strategy for typical production training, where one rank's view is representative of the whole run (such as loss or iteration time), and exporting from 1,000 ranks would overwhelm the backend.

### `all_ranks`

Every rank sends telemetry. The telemetry overhead multiplies by the world size at the collector.

**Recommended use.** Use this strategy when debugging a specific issue that might manifest on only some ranks (such as a hang on rank 7 or a `NaN` value on rank 42).

### `sampled`

This strategy performs hash-based deterministic sampling of ranks. A fraction of ranks defined by the `export_sample_rate` configuration value exports telemetry, while the remaining ranks are set to no-op.

```bash
NEMO_LENS_EXPORT_STRATEGY=sampled
NEMO_LENS_EXPORT_SAMPLE_RATE=0.1    # 10% of ranks
```

Sampling is deterministic by rank; the same combination of rank and sample rate yields the same decision across reruns.

**Recommended use.** Use this strategy for large jobs (over 1,000 ranks) where `all_ranks` is excessive, but you still require the perspective of more than a single rank.

### `first_rank_per_node`

One rank per node sends telemetry; specifically, the rank with `LOCAL_RANK=0` on each node. The library reads the `LOCAL_RANK` env var, which is typically configured by launchers such as torchrun and DeepSpeed. A missing `LOCAL_RANK` value is treated as `0`.

```bash
NEMO_LENS_EXPORT_STRATEGY=first_rank_per_node
```

**Recommended use.** Use this strategy when you require per-node visibility (such as attributing hangs or slow processes to a specific machine) without the volume of `all_ranks`. This is a common deployment for medium-scale jobs (8 to 128 nodes) where a single rank's view per machine provides the correct level of granularity.

## Implement the RankAwareSampler

For finer control at the SDK level, NeMo Lens ships a `Sampler` implementation at `nemo.lens.sampling.RankAwareSampler`. Because this sampler implements the OTel `Sampler` interface, you can plug it into `TracerProvider(sampler=...)`.

Enable the sampler through the configuration:

```bash
NEMO_LENS_SAMPLER_ENABLED=1
NEMO_LENS_EXPORT_SAMPLE_RATE=0.1
```

Alternatively, configure the sampler programmatically in your Python code:

```python
from nemo.lens import NemoLensConfig

cfg = NemoLensConfig(
    enabled=True,
    sampler_enabled=True,
    export_sample_rate=0.1,
)
```

### How It Works

- **Make per-rank decisions.** The sampling decision is made per-rank instead of per-span. When a rank is sampled, all its spans export; otherwise, no spans export.
- **Ensure deterministic decisions.** The same combination of rank and sample rate yields the same decision. The decision is made by comparing a hash of the rank (`md5(rank)`) against the `export_sample_rate` value. The `world_size` value is passed to the sampler but is not used in the decision.
- **Combine with export strategies.** The sampler works together with your export strategy. For example, combining `all_ranks` and `sampler_enabled=True` provides uniform random rank sampling at the SDK level.
- **Preserve the API without the SDK.** If the OTel SDK is not installed, the `should_sample` method returns a bare boolean value instead of a `SamplingResult` object, which preserves the API for non-SDK callers.

### Choose Between RankAwareSampler and Export Strategy

The export strategy operates during provider construction; non-exporting ranks receive no-op providers and skip span creation entirely. By contrast, the sampler operates during span creation, where spans are built and then discarded.

| Goal | Better Choice |
|---|---|
| 1000-rank training, only rank 0 matters | `single_rank` (no sampler; non-exporting ranks skip span creation entirely) |
| 1000-rank training, want random 10% of ranks | `all_ranks` and `sampler_enabled=1` with rate 0.1 |
| Debugging, want spans on all ranks for local inspection | `all_ranks` (no sampler) |

The export strategy is more efficient than the sampler when both approaches fit your requirements. Non-exporting ranks bypass span creation entirely, whereas the sampler still constructs and then discards spans.

## Configure OTel SDK Samplers

For **per-trace** sampling (instead of per-rank), use standard OTel SDK samplers through env vars:

```bash
OTEL_TRACES_SAMPLER=parentbased_traceidratio
OTEL_TRACES_SAMPLER_ARG=0.1    # keep 10% of traces
```

This configuration is independent of the NeMo Lens sampler; both can be run simultaneously. A typical production setup exports from a single rank using `single_rank` and then samples 10% of the traces at that rank using `parentbased_traceidratio`. This setup is recommended because even a single rank of `per_step` spans for a long job produces a large volume of data.


## Compose Samplers

| Export Strategy | `sampler_enabled` | `OTEL_TRACES_SAMPLER` | Result |
|---|---|---|---|
| `single_rank` | `0` | default (always_on) | All spans from one rank |
| `single_rank` | `0` | `traceidratio 0.1` | 10% of traces from one rank |
| `all_ranks` | `1` (rate 0.1) | default | All spans from 10% of ranks |
| `all_ranks` | `1` (rate 0.1) | `traceidratio 0.1` | 10% of traces from 10% of ranks (1% total) |

## Performance and Cost Implications

- **Understand span creation cost.** Span creation carries a per-span performance cost that scales with the number of attributes and recording decisions. Non-exporting ranks skip this entire process.
- **Account for queueing overhead.** The `BatchSpanProcessor` adds queueing work on top of span creation, which occurs only on exporting ranks.
- **Manage network egress.** Network egress scales with the total export volume. For example, exporting from 1,000 ranks with 100 spans per step at one step per second produces 100,000 spans per second, which exceeds what most collectors can handle. Always use sampling to manage this volume.
