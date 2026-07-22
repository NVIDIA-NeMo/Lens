# Lens tools

Standalone scripts for **emitting** and **analyzing** nemo-lens telemetry. They are not part
of the installed `nemo.lens` package (they live outside `src/`) — run them directly with a
python that has `nemo.lens` (reckoner) or just the stdlib (parsers) available.

| Script | Kind | Needs |
| --- | --- | --- |
| `slurm_telemetry_reckoner.py` | **emit** — backdate the SLURM job envelope into the goodput stream | `nemo.lens`, `sacct` |
| `goodput_report.py` | **analyze** — per-GPU goodput report + flame from lens spans | stdlib + `lens_analysis.py` |
| `json_to_perfetto.py` | **analyze** — convert lens OTel dumps → Perfetto/Chrome trace | stdlib + `lens_analysis.py` |
| `lens_analysis.py` | shared ingest plumbing (span/metric readers, timestamps, file discovery) | stdlib |
| `otelcol.yaml` | **collector** — example job-local `otelcol-contrib` config: two on-disk routings (`all/` + `goodput/`) | `otelcol-contrib` |
| `otelcol_honeycomb.yaml` | **collector** — overlay adding a live Honeycomb route to the goodput subset | `otelcol-contrib` |

All four are path-generic — no cluster paths, `#!/usr/bin/env python3`. Keep `lens_analysis.py`
next to the two parsers (they `import lens_analysis`).

## slurm_telemetry_reckoner.py

Out-of-band companion job that scrapes `sacct` after a run tears down and backdates real lens
spans onto the windows the training process can't emit for itself (queue wait, setup, teardown,
inter-attempt gaps). Uses the real Lens SDK, so its spans land in whatever backend the run's
inherited `NEMO_LENS_*` / `OTEL_*` env points at and correlate by `slurm.sluid`. See the guide
in `docs/observability/slurm-telemetry-reckoner.mdx`.

```bash
# usually submitted afterany:<jobid> on a cpu node so sacct's End is final
python slurm_telemetry_reckoner.py <jobid> [<lens_json_dir>]
```

## goodput_report.py

Per-GPU goodput report (scheduling × runtime × app factors, Drake-style) + optional HTML flame
and folded-stack output, computed from lens spans against the true SLURM allocation.

```bash
python goodput_report.py <lens_json_dir>... [--html out.html] [--fold out.folded]
```

The span→bucket mapping is the `CLASSIFY` dict at the top of the file. It is **tuned to the
megatron/slurm span taxonomy** this project emits (`megatron.*`, `slurm.*`, `nvrx.*`); the
machinery is generic but retune `CLASSIFY` for a different workload's span names.

## json_to_perfetto.py

Convert lens OTel span/metric dumps (per-rank `*.jsonl`) into the Chrome Trace Event Format that
[Perfetto](https://ui.perfetto.dev) and `chrome://tracing` load directly. Fully generic.

```bash
python json_to_perfetto.py <input_dir_or_files>... -o trace.json
```
