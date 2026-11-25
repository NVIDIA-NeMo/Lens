# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# GENERATED FILE - DO NOT EDIT.
# ruff: noqa: E501  (registry ids set the line length; not ours to wrap)
#
# Regenerate after any model/ edit:
#   weaver registry generate -r src/nemo/lens/semconv/model -t templates \
#     python <out>   &&   cp <out>/*.py src/nemo/lens/semconv/_generated/
#
# tests/test_semconv_generated.py fails if this drifts from model/.
"""Span names, generated from ``semconv/model/``.

``semconv/spans.py`` re-exports these. Edit the model, not this file.
"""

NV_DL_LAUNCH = "nv.dl.launch"
NV_DL_LAUNCH_COLLECTOR_START = "nv.dl.launch.collector_start"
NV_DL_LAUNCH_CONTAINER_LOAD = "nv.dl.launch.container_load"
NV_DL_LAUNCH_SBATCH_SCRIPT = "nv.dl.launch.sbatch_script"
NV_DL_LAUNCH_SCHEDULER_DELAY = "nv.dl.launch.scheduler_delay"
NV_DL_LAUNCH_WORKLOAD_STEP_START = "nv.dl.launch.workload_step_start"
NV_DL_RESILIENCY_GPU_SNIFF_PERIODIC = "nv.dl.resiliency.gpu_sniff.periodic"
NV_DL_RESILIENCY_GPU_SNIFF_STARTUP = "nv.dl.resiliency.gpu_sniff.startup"
NV_DL_TRAINING_CHECKPOINT_EXPOSED_SAVE = "nv.dl.training.checkpoint.exposed_save"
NV_DL_TRAINING_CHECKPOINT_LOAD = "nv.dl.training.checkpoint.load"
NV_DL_TRAINING_CHECKPOINT_LOAD_IO_READ = "nv.dl.training.checkpoint.load.io_read"
NV_DL_TRAINING_CHECKPOINT_SAVE = "nv.dl.training.checkpoint.save"
NV_DL_TRAINING_CHECKPOINT_SAVE_FINALIZE = "nv.dl.training.checkpoint.save.finalize"
NV_DL_TRAINING_CHECKPOINT_SAVE_IO_WRITE = "nv.dl.training.checkpoint.save.io_write"
NV_DL_TRAINING_CHECKPOINT_SAVE_STATE_DICT = "nv.dl.training.checkpoint.save.state_dict"
NV_DL_TRAINING_EVALUATE = "nv.dl.training.evaluate"
NV_DL_TRAINING_EVALUATE_STEP = "nv.dl.training.evaluate.step"
NV_DL_TRAINING_ITER_BLOCK = "nv.dl.training.iter_block"
NV_DL_TRAINING_ITERATION = "nv.dl.training.iteration"
NV_DL_TRAINING_ITERATION_FORWARD_BACKWARD = "nv.dl.training.iteration.forward_backward"
NV_DL_TRAINING_ITERATION_OPTIMIZER_STEP = "nv.dl.training.iteration.optimizer_step"
NV_DL_TRAINING_STARTUP = "nv.dl.training.startup"
NV_DL_TRAINING_STARTUP_ARG_PARSE = "nv.dl.training.startup.arg_parse"
NV_DL_TRAINING_STARTUP_DATALOADER = "nv.dl.training.startup.dataloader"
NV_DL_TRAINING_STARTUP_IMPORTS = "nv.dl.training.startup.imports"
NV_DL_TRAINING_STARTUP_IN_JOB_SETUP = "nv.dl.training.startup.in_job_setup"
NV_DL_TRAINING_STARTUP_INITIALIZE_MEGATRON = "nv.dl.training.startup.initialize_megatron"
NV_DL_TRAINING_STARTUP_JIT_FUSION_OPTIONS = "nv.dl.training.startup.jit_fusion_options"
NV_DL_TRAINING_STARTUP_MODEL_INIT = "nv.dl.training.startup.model_init"
NV_DL_TRAINING_STARTUP_PYTHON = "nv.dl.training.startup.python"
NV_DL_TRAINING_STARTUP_WEIGHT_HASH_CHECK = "nv.dl.training.startup.weight_hash_check"

ALL_SPANS: frozenset[str] = frozenset({
    "nv.dl.launch",
    "nv.dl.launch.collector_start",
    "nv.dl.launch.container_load",
    "nv.dl.launch.sbatch_script",
    "nv.dl.launch.scheduler_delay",
    "nv.dl.launch.workload_step_start",
    "nv.dl.resiliency.gpu_sniff.periodic",
    "nv.dl.resiliency.gpu_sniff.startup",
    "nv.dl.training.checkpoint.exposed_save",
    "nv.dl.training.checkpoint.load",
    "nv.dl.training.checkpoint.load.io_read",
    "nv.dl.training.checkpoint.save",
    "nv.dl.training.checkpoint.save.finalize",
    "nv.dl.training.checkpoint.save.io_write",
    "nv.dl.training.checkpoint.save.state_dict",
    "nv.dl.training.evaluate",
    "nv.dl.training.evaluate.step",
    "nv.dl.training.iter_block",
    "nv.dl.training.iteration",
    "nv.dl.training.iteration.forward_backward",
    "nv.dl.training.iteration.optimizer_step",
    "nv.dl.training.startup",
    "nv.dl.training.startup.arg_parse",
    "nv.dl.training.startup.dataloader",
    "nv.dl.training.startup.imports",
    "nv.dl.training.startup.in_job_setup",
    "nv.dl.training.startup.initialize_megatron",
    "nv.dl.training.startup.jit_fusion_options",
    "nv.dl.training.startup.model_init",
    "nv.dl.training.startup.python",
    "nv.dl.training.startup.weight_hash_check",
})
