# Distributed Tracing

Instrumenting distributed training requires handling three specific challenges that do not exist in single-process applications:

1. Each rank is a separate Python process with its own OTel SDK state.
2. Cross-rank operations, such as all-reduce or pipeline send and receive operations, occur through `torch.distributed` instead of HTTP; therefore, carrier-based propagation does not apply.
3. Concurrent pipeline stages have lateral, rather than hierarchical, relationships that parent-child spans cannot represent.

NeMo Lens's `nemo.lens.distributed` module addresses all three challenges.

## `broadcast_trace_context`

```python
from nemo.lens.distributed import broadcast_trace_context

# Must be called on ALL ranks (collective).
carrier = broadcast_trace_context(
    rank=torch.distributed.get_rank(),
    src_rank=0,
)

# Check `if carrier is not None:` before using it — a `None` carrier passed
# to `create_linked_span` yields an unlinked span rather than an error.
```

### What It Does

1. Serialize the current trace context to a W3C carrier dictionary and JSON-encode it to bytes on `src_rank`.
2. Broadcast the byte length (as an `int64` tensor) and then broadcast the bytes (as a `uint8` tensor) using `torch.distributed.broadcast`.
3. Deserialize the bytes into a carrier dictionary on all ranks.
4. Return the carrier dictionary on every rank, or `None` if PyTorch or `torch.distributed` is unavailable, the process group is not initialized, or the broadcast payload is empty.

### When to Use It

Call this function once per step or once per iteration when you want all ranks to share the same trace ID. Call the function inside a span on `src_rank` so that a meaningful context exists to broadcast.

### Cost

This operation performs two small collective broadcasts: a byte-length `int64` tensor and a payload of approximately 200 bytes. The operation runs once per iteration rather than per microbatch. Do not call this function per microbatch; call it once per iteration.

### Collective Correctness

`broadcast_trace_context` is a collective operation. If some ranks call it and others do not, a deadlock occurs. Gate this call only on conditions that are identical on every rank; for example, `config.enabled` (where the same `NemoLensConfig` value is passed to `setup_telemetry()` on all ranks). Do not gate this call on `handle.is_exporting` because that flag is `True` only on exporting ranks; since it differs per rank, gating on it would cause a deadlock.

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

### What It Does

This function creates a new span with an [OTel Link](https://opentelemetry.io/docs/concepts/signals/traces/#span-links) pointing to the span context encoded in the `remote_carrier`. The new span is not a child of the remote span; rather, it is an independent span that references the remote span.

### Why Use Links Instead of Parent-Child Relationships

Parent-child edges imply a sequential dependency: "the parent was running, it spawned this child, and then the parent continued." For pipeline-parallel stages, that model is incorrect because stages run concurrently. Stage one performing forward propagation on microbatch N does not spawn stage two's forward propagation on microbatch N - 1; these operations occur at the same time with a tensor exchange between them.

Links represent this relationship correctly: "this span is related to that span, but no temporal ordering is implied." Jaeger renders links as clickable references in the span detail panel instead of nesting them in the timeline.

### When to Use It

- **Pipeline-parallel stage correlation.** Link each stage's `recv_forward` to the sender's context so that the stage structure is visible in Jaeger.
- **Cross-service asynchronous operations.** A message-queue handler linking to the span of the producer.
- **Any cross-process correlation.** Use this pattern when temporal order is less important than the fact that both events occurred for the same logical request.

## Typical Pattern

Combine both primitives to wire pipeline-parallel correlation:

```python
# Once per iteration, after a step span is started on rank 0:
carrier = broadcast_trace_context(rank, src_rank=0)

# Later, inside the pipeline schedule on non-first ranks:
if carrier is not None and my_pp_rank != 0:
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

The carrier is the same on every rank (it is the context that rank zero broadcast). Each rank creates its own locally rooted spans, but each span contains a visible link back to the step context of rank zero.

## Store the Carrier Across Module Boundaries

The `broadcast_trace_context` function returns a carrier, but the code that calls `create_linked_span` is often in a different module (such as a pipeline schedule that does not see the call site of the training loop). Instead of threading the carrier through every function signature, use the module-level helpers in `nemo.lens.state`:

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

## Use Contrib Helpers for Specific Transports

For transports where W3C headers do not apply natively, use these helpers:

- `nemo.lens.contrib.nccl`: Serialize and deserialize carriers to bytes to piggyback on NCCL send operations.
- `nemo.lens.contrib.ray`: Use Ray remote-call helpers that accept an `_otel_carrier` keyword argument.

See [Contrib Helpers](contrib.md).

## Determine What to Instrument in Distributed Code

As a pragmatic rule, instrument the boundaries instead of every individual hop.

- **Yes**: Instrument pipeline-stage boundaries. This includes `recv_forward`, data-loader boundaries, and optimizer steps.
- **Sometimes**: Instrument individual all-reduce operations selectively. Enable this only with the `communication` span group because it is too verbose for default tracking.
- **Rarely**: Instrument every point-to-point send operation. The performance overhead of doing so dominates the telemetry signal.

Linked spans at stage boundaries, combined with a shared trace ID across ranks, provide 90% of the necessary visibility. Save fine-grained instrumentation for scenarios where you are actively debugging a specific issue.
