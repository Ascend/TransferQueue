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

from abc import ABC, abstractmethod
from typing import Any, Optional


class TransferQueueStorageKVClient(ABC):
    """
    Abstract base class for storage client.
    Subclasses must implement the core methods: put, get, and clear.
    """

    @abstractmethod
    def put(self, keys: list[str], values: list[Any]) -> Optional[list[Any]]:
        """
        Store key-value pairs in the storage backend.
        Args:
            keys (list[str]): List of keys to store.
            values (list[Any]): List of any type to store.
        Returns:
            Optional[list[Any]]: Optional list of custom metadata from each storage backend.
        """
        raise NotImplementedError("Subclasses must implement put")

    @abstractmethod
    def get(self, keys: list[str], shapes=None, dtypes=None, custom_meta=None) -> list[Any]:
        """
        Retrieve values from the storage backend by key.
        Args:
            keys (list[str]): List of keys whose values should be retrieved.
            shapes: Optional shape information for the expected tensors. The
                structure and interpretation of this argument are determined
                by the concrete storage backend implementation.
            dtypes: Optional data type information for the expected tensors.
                The structure and interpretation of this argument are
                determined by the concrete storage backend implementation.
            custom_meta: Optional backend-specific metadata used to control
                or optimize the retrieval process. Its format is defined by
                the concrete storage backend implementation.
        Returns:
            list[Tensor]: List of tensors retrieved from the storage backend,
            in the same order as the provided keys.
        """
        raise NotImplementedError("Subclasses must implement get")

    @abstractmethod
    def clear(self, keys: list[str]) -> None:
        raise NotImplementedError("Subclasses must implement clear")
