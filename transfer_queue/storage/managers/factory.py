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

import warnings
from typing import Any

from transfer_queue.storage.managers.base import TransferQueueStorageManager
from transfer_queue.utils.zmq_utils import ZMQServerInfo


class TransferQueueStorageManagerFactory:
    """Factory that creates a StorageManager instance."""

    _registry: dict[str, type[TransferQueueStorageManager]] = {}

    @classmethod
    def register(cls, manager_type: str):
        """Register a TransferQueueStorageManager class."""

        def decorator(manager_cls: type[TransferQueueStorageManager]):
            if not issubclass(manager_cls, TransferQueueStorageManager):
                raise TypeError(
                    f"manager_cls {getattr(manager_cls, '__name__', repr(manager_cls))} must be "
                    f"a subclass of TransferQueueStorageManager"
                )
            cls._registry[manager_type] = manager_cls
            return manager_cls

        return decorator

    @classmethod
    def create(
        cls, manager_type: str, controller_info: ZMQServerInfo, config: dict[str, Any]
    ) -> TransferQueueStorageManager:
        """Create and return a TransferQueueStorageManager instance."""
        if manager_type not in cls._registry:
            if manager_type == "AsyncSimpleStorageManager":
                warnings.warn(
                    f"The manager_type {manager_type} will be deprecated in 0.1.7, please use SimpleStorage instead.",
                    category=DeprecationWarning,
                    stacklevel=2,
                )
                manager_type = "SimpleStorage"
            elif manager_type == "MooncakeStorageManager":
                warnings.warn(
                    f"The manager_type {manager_type} will be deprecated in 0.1.7, please use MooncakeStore instead.",
                    category=DeprecationWarning,
                    stacklevel=2,
                )
                manager_type = "MooncakeStore"
            elif manager_type == "YuanrongStorageManager":
                warnings.warn(
                    f"The manager_type {manager_type} will be deprecated in 0.1.7, please use Yuanrong instead.",
                    category=DeprecationWarning,
                    stacklevel=2,
                )
                manager_type = "Yuanrong"
            else:
                raise ValueError(
                    f"Unknown manager_type: {manager_type}. Supported managers include: {list(cls._registry.keys())}"
                )
        return cls._registry[manager_type](controller_info, config)
