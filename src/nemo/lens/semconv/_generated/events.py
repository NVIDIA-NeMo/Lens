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
"""Event names, generated from ``semconv/model/``.

``semconv/events.py`` re-exports these. Edit the model, not this file.
"""

NV_DL_RESILIENCY_GPU_SNIFF_MEASUREMENT = "nv.dl.resiliency.gpu_sniff.measurement"
NV_DL_TRAINING_NUMERICS = "nv.dl.training.numerics"
NV_DL_TRAINING_OBJECTIVE = "nv.dl.training.objective"

ALL_EVENTS: frozenset[str] = frozenset({
    "nv.dl.resiliency.gpu_sniff.measurement",
    "nv.dl.training.numerics",
    "nv.dl.training.objective",
})
