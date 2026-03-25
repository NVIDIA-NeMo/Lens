# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Cross-rank trace context broadcast and linked spans.

Provides helpers for sharing a trace_id across all ranks (so that
distributed training appears as a single trace) and for creating
linked spans that reference remote contexts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from opentelemetry import trace

if TYPE_CHECKING:
    from opentelemetry import context


def broadcast_trace_context(
    rank: int,
    src_rank: int = 0,
) -> Optional[dict]:
    """Broadcast W3C trace context from src_rank to all other ranks.

    Must be called after ``torch.distributed.init_process_group()``.
    Returns a carrier dict with the broadcasted trace context.

    Args:
        rank: Current process rank.
        src_rank: Rank that holds the source context.

    Returns:
        A carrier dict with the broadcasted trace context, or None if
        torch.distributed is not available.
    """
    try:
        import torch
        import torch.distributed as dist
    except ImportError:
        return None

    if not dist.is_initialized():
        return None

    from nemo.lens.propagation import inject_context, extract_context

    if rank == src_rank:
        carrier: dict = {}
        inject_context(carrier)
        # Serialize to string for broadcast
        import json
        data = json.dumps(carrier).encode('utf-8')
    else:
        data = None

    # Broadcast the length first, then the data
    if rank == src_rank:
        length_tensor = torch.tensor([len(data)], dtype=torch.long)
    else:
        length_tensor = torch.tensor([0], dtype=torch.long)

    if torch.cuda.is_available():
        length_tensor = length_tensor.cuda()
    dist.broadcast(length_tensor, src=src_rank)

    length = length_tensor.item()
    if length == 0:
        return None

    if rank == src_rank:
        data_tensor = torch.tensor(list(data), dtype=torch.uint8)
    else:
        data_tensor = torch.zeros(length, dtype=torch.uint8)

    if torch.cuda.is_available():
        data_tensor = data_tensor.cuda()
    dist.broadcast(data_tensor, src=src_rank)

    import json
    carrier = json.loads(bytes(data_tensor.cpu().tolist()).decode('utf-8'))
    return carrier


def create_linked_span(
    tracer: trace.Tracer,
    name: str,
    remote_context: Optional['context.Context'] = None,
    remote_carrier: Optional[dict] = None,
    **attributes,
) -> trace.Span:
    """Create a span with a link to a remote span context.

    Use this for cross-rank or cross-service span correlation where
    parent-child relationships don't make sense (e.g., pipeline parallel
    stages).

    Args:
        tracer: OTel tracer.
        name: Span name.
        remote_context: An OTel Context from extract_context().
        remote_carrier: A W3C carrier dict (alternative to remote_context).
        **attributes: Span attributes.

    Returns:
        The created Span (already started, caller must end it).
    """
    links = []
    if remote_carrier is not None and remote_context is None:
        from nemo.lens.propagation import extract_context
        remote_context = extract_context(remote_carrier)

    if remote_context is not None:
        remote_span = trace.get_current_span(remote_context)
        span_ctx = remote_span.get_span_context()
        if span_ctx.is_valid:
            links.append(trace.Link(span_ctx))

    span = tracer.start_span(name, links=links)
    if attributes:
        from nemo.lens.helpers import safe_set_span_attributes
        safe_set_span_attributes(span, attributes)

    return span
