# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""SLURM environment detection."""

import os


def detect_slurm() -> dict:
    """Detect SLURM environment variables and return resource attributes.

    Returns an empty dict if not running under SLURM.
    """
    job_id = os.environ.get('SLURM_JOB_ID')
    if not job_id:
        return {}

    attrs = {'slurm.job.id': job_id}

    _mappings = {
        'SLURM_JOB_NAME': 'slurm.job.name',
        'SLURM_NODELIST': 'slurm.nodelist',
        'SLURM_NNODES': 'slurm.nnodes',
        'SLURM_NTASKS': 'slurm.ntasks',
        'SLURM_PARTITION': 'slurm.partition',
        'SLURM_CLUSTER_NAME': 'slurm.cluster.name',
    }
    for env_var, attr_name in _mappings.items():
        val = os.environ.get(env_var)
        if val:
            attrs[attr_name] = val

    return attrs
