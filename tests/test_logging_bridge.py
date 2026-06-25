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

"""Unit tests for Python logging bridge setup."""

import logging
import sys
import types

from nemo.lens.logging_bridge import setup_logging_bridge


def test_setup_logging_bridge_adds_handler_for_sdk_provider(monkeypatch):
    logs_module = types.ModuleType("opentelemetry._logs")
    sdk_logs_module = types.ModuleType("opentelemetry.sdk._logs")

    class LoggerProvider:
        pass

    class LoggingHandler(logging.Handler):
        def __init__(self, logger_provider):
            super().__init__()
            self.logger_provider = logger_provider

    provider = LoggerProvider()
    logs_module.get_logger_provider = lambda: provider
    sdk_logs_module.LoggerProvider = LoggerProvider
    sdk_logs_module.LoggingHandler = LoggingHandler
    monkeypatch.setitem(sys.modules, logs_module.__name__, logs_module)
    monkeypatch.setitem(sys.modules, sdk_logs_module.__name__, sdk_logs_module)

    logger = logging.getLogger("nemo.lens.test.logging_bridge")
    logger.handlers.clear()

    setup_logging_bridge(logger.name, level=logging.WARNING)

    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], LoggingHandler)
    assert logger.handlers[0].level == logging.WARNING
    assert logger.handlers[0].logger_provider is provider
    logger.handlers.clear()


def test_setup_logging_bridge_skips_non_sdk_provider(monkeypatch):
    logs_module = types.ModuleType("opentelemetry._logs")
    sdk_logs_module = types.ModuleType("opentelemetry.sdk._logs")

    class LoggerProvider:
        pass

    class LoggingHandler(logging.Handler):
        pass

    logs_module.get_logger_provider = object
    sdk_logs_module.LoggerProvider = LoggerProvider
    sdk_logs_module.LoggingHandler = LoggingHandler
    monkeypatch.setitem(sys.modules, logs_module.__name__, logs_module)
    monkeypatch.setitem(sys.modules, sdk_logs_module.__name__, sdk_logs_module)

    logger = logging.getLogger("nemo.lens.test.logging_bridge.skip")
    logger.handlers.clear()

    setup_logging_bridge(logger.name)

    assert logger.handlers == []
