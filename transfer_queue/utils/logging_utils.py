# Copyright 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2025 The TransferQueue Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os

_DEFAULT_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"


def get_logger(
    name: str | None,
    default_level: str = "WARNING",
) -> logging.Logger:
    """Create and configure a logger with consistent formatting.

    Creates a logger with the specified name, sets its level based on the
    TQ_LOGGING_LEVEL environment variable (or default), and adds a StreamHandler
    if no handlers exist. This is particularly useful for Ray Actor subprocesses
    that may not inherit logging configuration from the parent process.

    Args:
        name: The name for the logger, typically __name__.
        default_level: The default logging level if TQ_LOGGING_LEVEL is not set.

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(os.getenv("TQ_LOGGING_LEVEL", default_level))

    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_DEFAULT_LOG_FORMAT))
        logger.addHandler(handler)

    return logger
