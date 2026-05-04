# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("nemo-lens")
except PackageNotFoundError:
    __version__ = "0.0.0"
