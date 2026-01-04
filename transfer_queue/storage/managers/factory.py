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

from typing import Any

from transfer_queue.storage.managers.base import TransferQueueStorageManager


class TransferQueueStorageManagerFactory:
    """Factory that creates a StorageManager instance."""

    _registry: dict[str, type[TransferQueueStorageManager]] = {}

    @classmethod
    def register(cls, manager_type: str):
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
    def create(cls, manager_type: str, config: dict[str, Any]) -> TransferQueueStorageManager:
        if manager_type not in cls._registry:
            raise ValueError(
                f"Unknown manager_type: {manager_type}. Supported managers include: {list(cls._registry.keys())}"
            )
        return cls._registry[manager_type](config)
