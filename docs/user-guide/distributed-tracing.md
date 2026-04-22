# Distributed Tracing

Instrumenting distributed training requires handling three specific challenges that don't exist in single-process applications:

1. Each rank is a separate Python process with its own OTel SDK state.
2. Cross-rank operations (all-reduce, pipeline send/recv) happen via `torch.distributed` — not HTTP, so carrier-based propagation doesn't apply.
3. Concurrent pipeline stages have lateral (not hierarchical) relationships that parent-child spans can't represent.

Lens's `nemo.lens.distributed` module addresses all three.

## `broadcast_trace_context`

```python
from nemo.lens.distributed import broadcast_trace_context

# Must be called on ALL ranks (collective).
carrier = broadcast_trace_context(
    rank=torch.distributed.get_rank(),
    src_rank=0,
)
```

### What it does

1. On `src_rank`: serialise the current trace context to a W3C carrier dict, JSON-encode it to bytes.
2. `torch.distributed.broadcast` the byte length (int64 tensor), then the bytes (uint8 tensor).
3. All ranks deserialise into a carrier dict.
4. Returns the carrier on every rank.

### When to use it

Once per step or once per iteration when you want all ranks to share the same trace ID. Call it **inside** a span on `src_rank` so there's a meaningful context to broadcast.

### Cost

Two small collective broadcasts (a length int64 plus ~200 bytes of payload). Runs once per iteration, not per microbatch. **Do not** call it per-microbatch — call it once per iteration.

### Collective correctness

`broadcast_trace_context` is a **collective operation**. If some ranks call it and others don't, you get a deadlock. Gate it on conditions that are the same on all ranks (e.g. "telemetry was initialised" — use `get_telemetry() is not None`, not `TelemetryHandle.is_exporting` which differs per rank).

## `create_linked_span`

```python
from nemo.lens.distributed import create_linked_span

span = create_linked_span(
    tracer,
    'pipeline.recv_forward.linked',
    remote_carrier=carrier,       # from broadcast_trace_context or another source
    rank=my_rank,
    from_rank=my_rank - 1,
)
# ... work ...
span.end()
```

### What it does

Creates a new span with an [OTel Link](https://opentelemetry.io/docs/concepts/signals/traces/#span-links) pointing at the span context encoded in `remote_carrier`. The new span is **not a child** of the remote span — it's an independent span that *references* the remote one.

### Why links instead of parent-child

Parent-child edges imply sequential dependency: "the parent was running, it spawned this child, then the parent continued." For pipeline-parallel stages, that model is wrong — stages run **concurrently**. Stage 1 doing forward on microbatch N doesn't "spawn" stage 2's forward on microbatch N-1; they happen at the same time with a tensor exchange between them.

Links model this correctly: "this span is related to that span, but no temporal ordering is implied." Jaeger renders links as clickable references in the span detail panel rather than timeline nesting.

### When to use it

- **Pipeline-parallel stage correlation**: link each stage's `recv_forward` to the sender's context so the stage structure is visible in Jaeger.
- **Cross-service async operations**: a message-queue handler linking to the producer's span.
- **Any cross-process correlation** where temporal order matters less than "these happened for the same logical request."

## Typical pattern

Combine both primitives to wire pipeline-parallel correlation:

```python
# Once per iteration, after a step span is started on rank 0:
carrier = broadcast_trace_context(rank, src_rank=0)

# Later, inside the pipeline schedule on non-first ranks:
if my_pp_rank != 0:
    span = create_linked_span(
        tracer,
        'pipeline.recv_forward.linked',
        remote_carrier=carrier,
        rank=my_pp_rank,
        microbatch_id=i,
    )
    do_recv()
    span.end()
```

The carrier is the same on every rank (it's what rank 0 broadcast). Each rank creates its own locally-rooted spans but with a visible link back to rank 0's step context.

## Storing the carrier across module boundaries

`broadcast_trace_context` returns a carrier, but the code that calls `create_linked_span` is often in a different module (e.g. a pipeline schedule that doesn't see the training-loop call site). Rather than threading the carrier through every function signature, use the module-level helpers in `nemo.lens.state`:

```python
from nemo.lens.state import set_pp_trace_carrier, get_pp_trace_carrier

# In the training loop:
carrier = broadcast_trace_context(rank, src_rank=0)
set_pp_trace_carrier(carrier)

# In the pipeline schedule (deep in the call stack):
carrier = get_pp_trace_carrier()
if carrier is not None:
    span = create_linked_span(tracer, 'pp.recv_forward.linked', remote_carrier=carrier, ...)
```

This keeps the carrier out of function signatures while still making it available to whoever needs it.

## Contrib helpers for specific transports

For transports where W3C headers don't apply natively:

- `nemo.lens.contrib.nccl` — serialise/deserialise carriers to bytes for piggy-backing on NCCL sends.
- `nemo.lens.contrib.ray` — Ray remote-call helpers that accept an `_otel_carrier` kwarg.

See [Contrib](contrib.md).

## What to instrument in distributed code

A pragmatic rule: **instrument the boundaries, not every hop**.

- Yes: pipeline-stage boundaries (recv_forward), data-loader boundaries, optimizer step.
- Sometimes: individual all-reduces (only with `communication` span group; too chatty for default).
- Rarely: every P2P send (overhead dominates signal).

Linked spans at stage boundaries combined with a shared trace ID across ranks gives you the 90% view. Save the fine-grained instrumentation for when you're actively debugging a specific issue.
