# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""OTel metric instruments for the NeMo ecosystem."""

from nemo.lens.instruments.training import record_training_metrics
from nemo.lens.instruments.inference import record_inference_metrics

__all__ = ['record_training_metrics', 'record_inference_metrics']
