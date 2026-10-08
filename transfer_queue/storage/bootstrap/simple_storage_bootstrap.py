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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any
from uuid import uuid4

from omegaconf import DictConfig

from transfer_queue.storage.bootstrap.provider import StorageBootstrapProvider
from transfer_queue.storage.simple_storage import SimpleStorageUnit
from transfer_queue.utils.common import get_node_round_robin_scheduling_strategies
from transfer_queue.utils.logging_utils import get_logger
from transfer_queue.utils.zmq_utils import process_zmq_server_info

logger = get_logger(__name__)

DEFAULT_GLIBC_MMAP_THRESHOLD_BYTES = 1024 * 1024


@StorageBootstrapProvider.register_provider("SimpleStorage")
def initialize_simple_storage(conf: DictConfig) -> dict[str, Any]:
    """Initialize Simple storage with metastore mode."""

    simple_storage_handles = {}
    simple_storage_config = dict(conf.backend.SimpleStorage)
    simple_storage_config["_run_id"] = uuid4().hex
    num_data_storage_units = simple_storage_config["num_data_storage_units"]
    required_node_resource = simple_storage_config.get("required_node_resource")
    ssd_config = simple_storage_config.get("ssd_offload") or {}
    storage_actor_runtime_env = None
    if ssd_config.get("enabled", False):
        mmap_threshold = ssd_config.get("glibc_mmap_threshold_bytes", DEFAULT_GLIBC_MMAP_THRESHOLD_BYTES)
        if mmap_threshold is not None:
            mmap_threshold = int(mmap_threshold)
            if mmap_threshold <= 0:
                raise ValueError("glibc_mmap_threshold_bytes must be greater than zero or null")
            storage_actor_runtime_env = {"env_vars": {"MALLOC_MMAP_THRESHOLD_": str(mmap_threshold)}}
    scheduling_strategies = get_node_round_robin_scheduling_strategies(
        num_data_storage_units, required_node_resource=required_node_resource
    )

    for storage_unit_rank in range(num_data_storage_units):
        actor_options = {
            "scheduling_strategy": scheduling_strategies[storage_unit_rank],
            "name": f"TransferQueueStorageUnit#{storage_unit_rank}",
        }
        if storage_actor_runtime_env is not None:
            actor_options["runtime_env"] = storage_actor_runtime_env
        storage_node = SimpleStorageUnit.options(**actor_options).remote(  # type: ignore[attr-defined]
            config=simple_storage_config
        )
        simple_storage_handles[f"TransferQueueStorageUnit#{storage_unit_rank}"] = storage_node
        logger.info(
            f"TransferQueueStorageUnit#{storage_unit_rank} has been created "
            f"on node {scheduling_strategies[storage_unit_rank].node_id}."
        )

    storage_zmq_info = process_zmq_server_info(simple_storage_handles)
    backend_name = conf.backend.storage_backend
    conf.backend[backend_name].zmq_info = storage_zmq_info

    return simple_storage_handles
