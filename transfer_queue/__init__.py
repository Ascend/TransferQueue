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

from .client import (
    AsyncTransferQueueClient,
    TransferQueueClient,
    process_zmq_server_info,
)
from .controller import TransferQueueController
from .metadata import BatchMeta
from .sampler import BaseSampler
from .sampler.grpo_group_n_sampler import GRPOGroupNSampler
from .sampler.sequential_sampler import SequentialSampler
from .storage import SimpleStorageUnit
from .utils.utils import get_placement_group
from .utils.zmq_utils import ZMQServerInfo

__all__ = [
    "AsyncTransferQueueClient",
    "BatchMeta",
    "TransferQueueClient",
    "TransferQueueController",
    "SimpleStorageUnit",
    "ZMQServerInfo",
    "process_zmq_server_info",
    "get_placement_group",
    "BaseSampler",
    "GRPOGroupNSampler",
    "SequentialSampler",
]

version_folder = os.path.dirname(os.path.join(os.path.abspath(__file__)))

with open(os.path.join(version_folder, "version/version")) as f:
    __version__ = f.read().strip()
