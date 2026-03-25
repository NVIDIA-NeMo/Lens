# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Resource detection: auto-detect SLURM, K8s, and local environment attributes."""

from nemo.lens.resources.kubernetes import detect_kubernetes
from nemo.lens.resources.local import detect_local
from nemo.lens.resources.slurm import detect_slurm


def detect_resource() -> dict:
    """Detect deployment environment and return resource attributes.

    Checks SLURM, Kubernetes, and local environment in order.
    All detected attributes are merged.
    """
    attrs = {}
    attrs.update(detect_local())
    attrs.update(detect_slurm())
    attrs.update(detect_kubernetes())
    return attrs


__all__ = ["detect_resource", "detect_slurm", "detect_kubernetes", "detect_local"]
