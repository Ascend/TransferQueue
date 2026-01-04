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

from torch import Tensor


class TransferQueueStorageKVClient(ABC):
    """
    Abstract base class for storage client.
    Subclasses must implement the core methods: put, get, and clear.
    """

    @abstractmethod
    def put(self, keys: list[str], values: list[Tensor]) -> None:
        raise NotImplementedError("Subclasses must implement put")

    @abstractmethod
    def get(self, keys: list[str], shapes=None, dtypes=None) -> list[Tensor]:
        raise NotImplementedError("Subclasses must implement get")

    @abstractmethod
    def clear(self, keys: list[str]) -> None:
        raise NotImplementedError("Subclasses must implement clear")
