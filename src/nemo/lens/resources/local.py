# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Local environment detection: hostname, PID, GPU count."""

import os
import socket


def detect_local() -> dict:
    """Detect local environment attributes."""
    attrs = {
        'host.name': socket.gethostname(),
        'process.pid': os.getpid(),
    }

    # GPU count detection (best-effort)
    gpu_count = _detect_gpu_count()
    if gpu_count is not None:
        attrs['host.gpu.count'] = gpu_count

    return attrs


def _detect_gpu_count() -> int | None:
    """Detect the number of GPUs available (best-effort)."""
    # Check CUDA_VISIBLE_DEVICES first
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if cuda_visible is not None:
        if cuda_visible.strip() == '':
            return 0
        return len(cuda_visible.split(','))

    # Try nvidia-smi
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=count', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            lines = [l.strip() for l in result.stdout.strip().split('\n') if l.strip()]
            if lines:
                return len(lines)
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass

    return None
