# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _dist_version

try:
    __version__ = _dist_version("nemo-lens")
except _PackageNotFoundError:
    try:
        from nemo.lens._version import __version__
    except ModuleNotFoundError:
        from setuptools_scm import get_version as _get_version

        __version__ = _get_version(root="../../..", relative_to=__file__)

__package_name__ = "nemo_lens"
__contact_names__ = "NVIDIA"
__contact_emails__ = "nemo-toolkit@nvidia.com"
__homepage__ = "https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/"
__repository_url__ = "https://github.com/NVIDIA-NeMo/Lens"
__download_url__ = "https://github.com/NVIDIA-NeMo/Lens/releases"
__description__ = "NeMo Lens"
__license__ = "Apache2"
__keywords__ = (
    "deep learning, machine learning, gpu, NLP, NeMo, Lens, nvidia, pytorch, torch, language"
)
