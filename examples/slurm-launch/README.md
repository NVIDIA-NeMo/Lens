# SLURM Launch Examples

These examples show how to emit Lens/OpenTelemetry launch spans from SLURM jobs.
They are intentionally small, so they can be copied into a cluster test directory
and edited for a site without pulling in a larger training stack.

Before submitting, edit the `#SBATCH` account, partition, qos, output path,
`TEL`, `CONTAINER_IMAGE`, and `CONTAINER_MOUNTS` values in the chosen script.
The placeholder values such as `example-account`, `example-qos`, and
`/path/to/example-telemetry` are not valid defaults.

The `nemo-lens` command must be on `PATH` in both the sbatch script context and
inside the container task context. The container must include SDK-capable Lens
dependencies because the examples call `nemo-lens emit-spans` and the mock
workload emits spans directly.

The launch pattern is:

1. The sbatch preamble records scheduler/script timestamps, queries `sacct` once
   for supplemental SLUID values, and runs
   `nemo-lens set-slurm-resource-attrs --stage sbatch`.
2. Each `srun` task runs `nemo-lens set-slurm-resource-attrs --stage task` inside
   the container so per-task values such as `host.name` and `slurm.topology.*`
   are produced in the task context.
3. Each task starts a local OpenTelemetry collector, emits `nv.dl.launch` spans
   with `nemo-lens emit-spans`, runs the workload, then gives the collector a
   short drain window before terminating it.

The mock workload emits `nv.dl.training.python_startup` and
`nv.dl.training.python_imports` spans. They stand in for training-process costs;
NVRx-specific startup spans should be emitted by NVRx itself.

`otel_minimal.sbatch` is the plain SLURM example. The `nvrx_*.sbatch` examples
assume NVRx is installed in the container and that `ft_launcher` is on `PATH`.
They do not mount or override an NVRx source checkout.

The array-spare example includes a minimal generation-teardown guard based on the
NVRx singleton wrapper pattern: task 0 writes `RDZV_CLOSED` and cancels the array
generation when it exits. A cold spare may therefore be cancelled before it emits
telemetry; that is expected and preferable to waiting out the TCPStore connect
timeout after the rendezvous host has exited.
