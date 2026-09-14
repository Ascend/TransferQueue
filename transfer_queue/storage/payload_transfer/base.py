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

"""High-level payload transfer contract used by SimpleStorage."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from transfer_queue.utils.zmq_utils import ZMQMessage


class PayloadTransferError(RuntimeError):
    """A payload transfer could not be completed safely."""


class PayloadTransfer(ABC):
    """Complete SimpleStorage payload strategy, including its wire protocol."""

    @abstractmethod
    async def put(
        self,
        *,
        control_socket: Any,
        sender_id: str,
        target_id: str,
        global_indexes: list[int],
        data: dict[str, Any],
        data_parser: Callable[[Any], Any] | None,
    ) -> None:
        """Put decoded storage data through this strategy."""

    @abstractmethod
    async def get(
        self,
        *,
        control_socket: Any,
        sender_id: str,
        target_id: str,
        global_indexes: list[int],
        fields: list[str],
    ) -> dict[str, Any]:
        """Get storage data through this strategy."""

    @abstractmethod
    def handle_request(
        self,
        request: ZMQMessage,
        *,
        storage_id: str,
        load_data: Callable[..., dict[str, Any]],
        store_data: Callable[..., None],
    ) -> ZMQMessage | None:
        """Handle a strategy-owned request on a SimpleStorageUnit."""

    def bootstrap_info(self) -> dict[str, Any] | None:
        """Return transport-specific metadata needed by peer instances."""
        return None

    def close(self) -> None:
        """Release transport resources."""
        return None
