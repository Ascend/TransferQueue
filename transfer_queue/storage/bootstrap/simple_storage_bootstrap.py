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

import math
from typing import Any

from omegaconf import DictConfig

from transfer_queue.storage.bootstrap.provider import StorageBootstrapProvider
from transfer_queue.storage.simple_storage import SimpleStorageUnit
from transfer_queue.utils.common import get_placement_group
from transfer_queue.utils.logging_utils import get_logger
from transfer_queue.utils.zmq_utils import process_zmq_server_info

logger = get_logger(__name__)


@StorageBootstrapProvider.register_provider("SimpleStorage")
def initialize_simple_storage(conf: DictConfig) -> dict[str, Any]:
    """Initialize Simple storage with metastore mode."""

    simple_storage_handles = {}
    num_data_storage_units = conf.backend.SimpleStorage.num_data_storage_units
    total_storage_size = conf.backend.SimpleStorage.get("total_storage_size", None)
    storage_placement_group = get_placement_group(num_data_storage_units, num_cpus_per_actor=1)

    # Compute per-unit capacity: None means unlimited
    storage_unit_size = (
        math.ceil(total_storage_size / num_data_storage_units) if total_storage_size is not None else None
    )

    for storage_unit_rank in range(num_data_storage_units):
        storage_node = SimpleStorageUnit.options(  # type: ignore[attr-defined]
            placement_group=storage_placement_group,
            placement_group_bundle_index=storage_unit_rank,
            name=f"TransferQueueStorageUnit#{storage_unit_rank}",
        ).remote(
            storage_unit_size=storage_unit_size,
        )
        simple_storage_handles[f"TransferQueueStorageUnit#{storage_unit_rank}"] = storage_node
        logger.info(f"TransferQueueStorageUnit#{storage_unit_rank} has been created.")

    storage_zmq_info = process_zmq_server_info(simple_storage_handles)
    backend_name = conf.backend.storage_backend
    conf.backend[backend_name].zmq_info = storage_zmq_info

    return simple_storage_handles
