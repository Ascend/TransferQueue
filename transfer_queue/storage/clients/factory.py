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


from transfer_queue.storage.clients.base import TransferQueueStorageKVClient


class StorageClientFactory:
    """
    Factory class for creating storage client instances.
    Uses a decorator-based registration mechanism to map client names to classes.
    """

    # Class variable: maps client names to their corresponding classes
    _registry: dict[str, type[TransferQueueStorageKVClient]] = {}

    @classmethod
    def register(cls, client_type: str):
        """
        Decorator to register a concrete client class with the factory.
        Args:
            client_type (str): The name used to identify the client
        Returns:
            Callable: The decorator function that returns the original class
        """

        def decorator(client_class: type[TransferQueueStorageKVClient]) -> type[TransferQueueStorageKVClient]:
            cls._registry[client_type] = client_class
            return client_class

        return decorator

    @classmethod
    def create(cls, client_type: str, config: dict) -> TransferQueueStorageKVClient:
        """
        Create and return an instance of the storage client by name.
        Args:
            client_type (str): The registered name of the client
        Returns:
            StorageClientFactory: An instance of the requested client
        Raises:
            ValueError: If no client is registered with the given name
        """
        if client_type not in cls._registry:
            raise ValueError(f"Unknown StorageClient: {client_type}")
        return cls._registry[client_type](config)
