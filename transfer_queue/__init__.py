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

import os

from .client import TransferQueueClient
from .controller import TransferQueueController
from .dataloader import StreamingDataLoader, StreamingDataset
from .interface import (
    async_clear_partition,
    async_clear_samples,
    async_get_data,
    async_get_meta,
    async_put,
    async_set_custom_meta,
    clear_partition,
    clear_samples,
    close,
    get_data,
    get_meta,
    init,
    put,
    set_custom_meta,
)
from .metadata import BatchMeta
from .sampler import BaseSampler
from .sampler.grpo_group_n_sampler import GRPOGroupNSampler
from .sampler.rank_aware_sampler import RankAwareSampler
from .sampler.sequential_sampler import SequentialSampler
from .storage import SimpleStorageUnit
from .utils.common import get_placement_group
from .utils.zmq_utils import ZMQServerInfo, process_zmq_server_info

__all__ = [
    "init",
    "get_meta",
    "get_data",
    "put",
    "set_custom_meta",
    "clear_samples",
    "clear_partition",
    "async_get_meta",
    "async_get_data",
    "async_put",
    "async_set_custom_meta",
    "async_clear_samples",
    "async_clear_partition",
    "close",
] + [
    "TransferQueueClient",
    "StreamingDataset",
    "StreamingDataLoader",
    "BatchMeta",
    "TransferQueueController",
    "SimpleStorageUnit",
    "ZMQServerInfo",
    "process_zmq_server_info",
    "get_placement_group",
    "BaseSampler",
    "GRPOGroupNSampler",
    "SequentialSampler",
    "RankAwareSampler",
]

version_folder = os.path.dirname(os.path.join(os.path.abspath(__file__)))

with open(os.path.join(version_folder, "version/version")) as f:
    __version__ = f.read().strip()
