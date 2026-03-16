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


@TransferQueueStorageManagerFactory.register("MooncakeStore")
class MooncakeStorageManager(KVStorageManager):
    """Storage manager for MooncakeStorage backend."""

    def __init__(self, controller_info: ZMQServerInfo, config: dict[str, Any]):
        logger.warning(
            "MooncakeStore backend doesn't support key update (upsert) for now. "
            "You must delete the key before updating it. "
            "Refer to https://github.com/kvcache-ai/Mooncake/issues/1645 for details."
        )

        config["client_name"] = "MooncakeStorageClient"
        super().__init__(controller_info, config)
