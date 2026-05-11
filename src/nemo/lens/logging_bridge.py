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

"""Python logging -> OTel LoggerProvider bridge.

Correlates Python log records with the active span's trace_id/span_id.
Gated by ``NEMO_LENS_LOGS_ENABLED=1``.
"""

from __future__ import annotations

import logging


def setup_logging_bridge(logger_name: str = "", level: int = logging.INFO) -> None:
    """Bridge Python logging to the OTel LoggerProvider.

    Adds an OTel log handler to the specified Python logger so that log
    records are exported as OTel log records with trace correlation.

    Args:
        logger_name: Python logger name (empty string = root logger).
        level: Minimum log level to bridge.
    """
    try:
        from opentelemetry._logs import get_logger_provider
        from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler

        logger_provider = get_logger_provider()
        if not isinstance(logger_provider, LoggerProvider):
            return

        handler = LoggingHandler(logger_provider=logger_provider)
        handler.setLevel(level)
        logging.getLogger(logger_name).addHandler(handler)
    except ImportError:
        # OTel logs SDK not installed; silently skip
        pass
    except Exception:
        # Don't break the application if logging bridge setup fails
        logging.getLogger(__name__).debug("Failed to set up OTel logging bridge", exc_info=True)
