# Production Checklist

Going from "telemetry works on my laptop" to "telemetry runs for the full training campaign" is a different exercise. This page is opinionated about which knobs matter in production and how to pick them deliberately rather than by accident.

If anything here conflicts with your site's ops conventions, trust ops.

## Pick an export strategy

Decision flow:

- One rank's view is representative and you want the simplest setup. Use `single_rank` (the default). Done.
- You need per-rank investigation — hang debugging, NaN hunting, suspected bad nodes. Use `all_ranks` for the duration of the investigation, then revert.
- Fleet-scale job where you want more than one rank's perspective without full export. Use `sampled` with `sampler_enabled=1`.

Full discussion in [Sampling](sampling.md). Don't leave this at default "because it's the default" — pick it because it fits the run.

## Pick a span group preset

- Quiet production telemetry: `default`. Job, checkpoint, evaluate. No per-step noise.
- Production profiling window: `per_step` combined with aggressive SDK trace sampling. Something like `OTEL_TRACES_SAMPLER=parentbased_traceidratio` with a low ratio keeps volume manageable while you gather a profile.
- Active debugging: `all`, and only for short-duration runs. Do not leave this on.

See [Span Groups](span-groups.md) for the full breakdown.

## Layer your sampling

Think of sampling as four composable layers:

1. **Export strategy** (`single_rank` / `all_ranks` / `sampled`) — decides which ranks emit at all. Non-exporting ranks are fully no-op.
2. **`RankAwareSampler`** (via `sampler_enabled=1`) — SDK-level per-rank decision on whether to keep spans on an exporting rank.
3. **OTel SDK trace sampler** (`OTEL_TRACES_SAMPLER`) — per-trace decision on whether to record.
4. **Collector-side tail sampling** (optional) — lets you make smart decisions after the fact: keep all error traces, sample successful ones.

Compose them. Layer 1 decides "who talks"; layers 2 and 3 decide "how much of what they have"; layer 4 decides "what's worth keeping". Trying to do everything at one layer either over-samples the interesting cases or under-samples and hides them.

## Use `nemo.run.id` as the partition key

Every backend worth using needs a partition key — a way to scope queries to a single run. Lens sets `nemo.run.id` on every span and metric, automatically derived from `NEMO_LENS_RUN_ID`, `SLURM_JOB_ID`, or a generated UUID.

Use it everywhere:

- Jaeger: tag filter `nemo.run.id=<value>`
- Grafana: dashboard variable `nemo_run_id` populated from `label_values(<metric>, nemo_run_id)`
- Honeycomb: filter facet `nemo.run.id`
- Datadog: facet `@nemo.run.id`
- Kibana: filter field `nemo.run.id`

Set `NEMO_LENS_RUN_ID` explicitly when you want a human-readable identifier (`llama3-pretrain-2026-04-22`) rather than a UUID. SLURM job IDs are fine too, but "the SLURM job that OOMed last Tuesday" is harder to grep for than "llama3-pretrain-2026-04-22".

## Use resource attributes for run comparison

Parallelism config, model architecture, precision, cluster name — anything stable for the process lifetime belongs in `resource_attributes`, not on individual spans. In Jaeger these become "Process" tags, filterable across every span and metric in the run without cluttering the span view.

```python
handle = setup_telemetry(
    config,
    rank=rank,
    world_size=world_size,
    resource_attributes={
        "dl.tensor_parallel.size": 4,
        "dl.pipeline_parallel.size": 2,
        "dl.data_parallel.size": 8,
        "megatron.num_layers": 80,
        "megatron.precision": "bf16",
    },
)
```

See [Resource Detection](resources.md) for the full picture and the `dl.*` vs project-scoped conventions.

## Size your collector

The collector has to keep up with your peak trace rate. If its receive queue backs up, the SDK's `BatchSpanProcessor` will eventually back up too, and spans will drop. Monitor the collector's own telemetry (Prometheus metrics on `:8888/metrics`) and alert on processor queue saturation — `otelcol_processor_batch_send_size`, exporter send failures, and queue depth are the usual suspects.

If a collector can't keep up, the honest fix is more collector capacity or more aggressive sampling, not muting alerts.

## Export only what you need

Resist the instinct to turn on `all` span groups "just in case". Span groups exist precisely so you can enable and disable without redeploying — keep `default` on by default and escalate when debugging. Every additional group is more collector volume, more backend cost, and more noise in the trace view.

If you find yourself needing `all` permanently, that's a signal to move some of those groups into `default` at the library level, not to ship `all` to every production run.

## Think before enabling logs

If you're using the log bridge (`NEMO_LENS_LOGS_ENABLED=1`), be deliberate about level and scope. `DEBUG` across a fleet of ranks will overwhelm any log store. Prefer bridging specific subsystems rather than the root logger:

```python
setup_logging_bridge(logger_name="megatron.training")
```

See [Logging Bridge](logging-bridge.md).

## Consider a backup sink

The Collector fans out cheaply. A common pattern is to route one OTLP stream from the application to the Collector, then have the Collector export to two backends — e.g. W&B Weave for trace UX and Prometheus for alerting. The application exports once; the plumbing is in the Collector config.

## Before going live

- [ ] `setup_telemetry` is called exactly once at process startup.
- [ ] `handle.shutdown()` runs in a `finally` block so traces flush on clean exit.
- [ ] `nemo.run.id` is set (or auto-derived) and visible in your dashboards.
- [ ] Jaeger / Grafana / your backend can filter by `nemo.run.id`.
- [ ] You've tested the OTLP endpoint with `curl` from a training host.
- [ ] Sampler and export strategy are chosen deliberately, not left at default.
- [ ] Dashboards have panels for step duration, loss, throughput, grad norm.
- [ ] Rollback plan exists: `NEMO_LENS_ENABLED=0` and restart, no code change required.
