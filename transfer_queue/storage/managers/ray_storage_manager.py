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
from typing import Any

from transfer_queue.storage.managers.base import KVStorageManager
from transfer_queue.storage.managers.factory import TransferQueueStorageManagerFactory
from transfer_queue.utils.zmq_utils import ZMQServerInfo

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("TQ_LOGGING_LEVEL", logging.WARNING))


@TransferQueueStorageManagerFactory.register("RayStore")
class RayStorageManager(KVStorageManager):
    """Storage manager for Ray-RDT backend."""

    def __init__(self, controller_info: ZMQServerInfo, config: dict[str, Any]):
        config = (config or {}).copy()
        if config.get("client_name") not in (None, "RayStorageClient"):
            raise ValueError(f"RayStorageManager only supports 'RayStorageClient', got: {config.get('client_name')}")
        super().__init__(controller_info, {**config, "client_name": "RayStorageClient"})
