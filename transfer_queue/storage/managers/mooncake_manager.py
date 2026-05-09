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

from transfer_queue.storage.managers.base import KVStorageManager, StorageManagerFactory
from transfer_queue.utils.zmq_utils import ZMQServerInfo


@StorageManagerFactory.register("MooncakeStore")
class MooncakeStorageManager(KVStorageManager):
    """Storage manager for MooncakeStorage backend.

    Key update (upsert) is supported natively via the MooncakeStore client's
    ``batch_upsert_from`` (zero-copy tensor path) and ``upsert_batch`` (raw bytes
    path). See ``mooncake-integration/store/store_py.cpp`` upstream for the
    pybind bindings.
    """

    def __init__(self, controller_info: ZMQServerInfo, config: dict[str, Any]):
        config["client_name"] = "MooncakeStoreClient"
        super().__init__(controller_info, config)
