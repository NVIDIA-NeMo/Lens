# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""NCCL context propagation helpers.

Provides utilities for serializing/deserializing OTel trace context
alongside activation tensor transfers in pipeline parallel training.
"""

from __future__ import annotations

import json

from nemo.lens.propagation import extract_context, inject_context


def serialize_context() -> bytes:
    """Serialize current trace context to bytes for piggy-backing on NCCL sends.

    Returns:
        JSON-encoded W3C trace context as bytes.
    """
    carrier: dict = {}
    inject_context(carrier)
    return json.dumps(carrier).encode("utf-8")


def deserialize_context(data: bytes) -> dict | None:
    """Deserialize trace context from bytes received via NCCL.

    Args:
        data: JSON-encoded W3C trace context bytes.

    Returns:
        A carrier dict, or None if deserialization fails.
    """
    try:
        return json.loads(data.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def extract_nccl_context(data: bytes):
    """Extract OTel context from NCCL-received bytes.

    Convenience function combining deserialize + extract.

    Args:
        data: JSON-encoded W3C trace context bytes.

    Returns:
        An OTel Context (empty if deserialization fails).
    """
    carrier = deserialize_context(data)
    if carrier is None:
        from opentelemetry.context import get_current

        return get_current()
    return extract_context(carrier)
