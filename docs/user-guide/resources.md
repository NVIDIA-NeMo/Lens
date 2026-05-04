# Resource Detection

OTel [Resources](https://opentelemetry.io/docs/specs/otel/resource/sdk/) describe the entity producing telemetry: service name, version, host, cloud provider, etc. Lens auto-detects a handful of environment-specific attributes so runs are filterable without manual config.

## Default attributes

Every exporter-rank process emits these attributes (set in `providers.py:build_providers`):

| Attribute | Source | Example |
|---|---|---|
| `service.name` | `config.service_name` or `OTEL_SERVICE_NAME` | `"megatron-lm"` |
| `service.version` | `nemo.lens.__version__` | `"0.1.0"`, `"0.1.0.post3+gabc1234"` |
| `service.instance.id` | `"{run_id}-rank{rank}"` | `"abc123-rank0"` |
| `dl.rank` | `rank` argument | `0` |
| `dl.world_size` | `world_size` argument | `8` |
| `nemo.run.id` | `config.run_id` (auto-generated if empty) | `"abc123"` |
| `nemo.user.id` | `config.user` (if set) | `"my-team"` |
| `deployment.environment` | `DEPLOYMENT_ENV` or `ENVIRONMENT` env var | `"production"` |

## Auto-detected attributes

`nemo.lens.resources.detect_resource()` merges attributes from three sources:

### `detect_local()` — local process

| Attribute | Description |
|---|---|
| `host.name` | Hostname from `socket.gethostname()` |
| `process.pid` | Python's `os.getpid()` |
| `dl.gpu.count` | GPU count from `CUDA_VISIBLE_DEVICES` or `nvidia-smi` |

### `detect_slurm()` — SLURM env

Active when `SLURM_JOB_ID` is set. Maps:

| Attribute | Source env var |
|---|---|
| `slurm.job.id` | `SLURM_JOB_ID` |
| `slurm.job.name` | `SLURM_JOB_NAME` |
| `slurm.nodelist` | `SLURM_JOB_NODELIST` |
| `slurm.nnodes` | `SLURM_NNODES` |
| `slurm.ntasks` | `SLURM_NTASKS` |
| `slurm.partition` | `SLURM_JOB_PARTITION` |
| `slurm.cluster.name` | `SLURM_CLUSTER_NAME` |

### `detect_kubernetes()` — K8s env

Active when `KUBERNETES_SERVICE_HOST` is set or `/var/run/secrets/kubernetes.io` exists. Maps:

| Attribute | Source env var |
|---|---|
| `k8s.namespace.name` | `K8S_NAMESPACE` / `POD_NAMESPACE` |
| `k8s.pod.name` | `K8S_POD_NAME` / `POD_NAME` |
| `k8s.pod.uid` | `K8S_POD_UID` / `POD_UID` |
| `k8s.node.name` | `K8S_NODE_NAME` / `NODE_NAME` |
| `k8s.container.name` | `K8S_CONTAINER_NAME` |
| `k8s.job.name` | `K8S_JOB_NAME` |

## Adding custom attributes

Pass `resource_attributes=` to `setup_telemetry`:

```python
handle = setup_telemetry(
    config,
    rank=rank,
    world_size=world_size,
    resource_attributes={
        'dl.tensor_parallel.size': 4,
        'dl.pipeline_parallel.size': 2,
        'dl.data_parallel.size': 8,
        'megatron.num_layers': 32,
        'megatron.precision': 'bf16',
    },
)
```

These merge with the auto-detected set. In Jaeger they appear as "Process" tags — filterable across every span in the run.

## Use cases

### Filter by rank

In Jaeger: `dl.rank=0`

### Compare two runs

In Grafana: use `nemo.run.id` as a dashboard variable, list all values, select the runs to compare.

### Filter by parallelism config

In Jaeger: `dl.tensor_parallel.size=4 AND dl.pipeline_parallel.size=2`

Because these are resource attributes (not span attributes), they apply to every span without cluttering the span view.

## Conventions

- Use standard OTel attribute names where they exist (`service.*`, `k8s.*`, `host.*`).
- Use `dl.*` (distributed learning) for training-specific attributes shared across consumers.
- Use `<project>.*` for project-specific attributes (`megatron.*`, `rl.*`, `gym.*`).

See [semconv](../design/semconv.md) for the full attribute namespace conventions.

## Detection order

`detect_resource()` merges in this order: local → SLURM → Kubernetes. If a key collides, the later source wins. In practice collisions are rare because each layer uses its own namespace.

## Running locally

On a developer machine with no SLURM or K8s, only `detect_local()` fires. Run IDs auto-generate to a UUID, so you can still filter by `nemo.run.id` to isolate a single local run.
