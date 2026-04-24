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
from .dataloader import StreamingDataLoader, StreamingDataset
from .interface import (
    async_kv_batch_get,
    async_kv_batch_get_by_meta,
    async_kv_batch_put,
    async_kv_clear,
    async_kv_list,
    async_kv_put,
    close,
    get_client,
    get_metrics_endpoint,
    init,
    kv_batch_get,
    kv_batch_get_by_meta,
    kv_batch_put,
    kv_clear,
    kv_list,
    kv_put,
)
from .metadata import BatchMeta, KVBatchMeta
from .sampler import BaseSampler
from .sampler.grpo_group_n_sampler import GRPOGroupNSampler
from .sampler.rank_aware_sampler import RankAwareSampler
from .sampler.seqlen_balanced_sampler import SeqlenBalancedSampler
from .sampler.sequential_sampler import SequentialSampler

__all__ = (
    [
        # High-Level KV Interface
        "init",
        "close",
        "get_metrics_endpoint",
        "kv_put",
        "kv_batch_put",
        "kv_batch_get",
        "kv_batch_get_by_meta",
        "kv_list",
        "kv_clear",
        "async_kv_put",
        "async_kv_batch_put",
        "async_kv_batch_get",
        "async_kv_batch_get_by_meta",
        "async_kv_list",
        "async_kv_clear",
        "KVBatchMeta",
    ]
    + [
        # High-Level StreamingDataLoader Interface
        "StreamingDataset",
        "StreamingDataLoader",
    ]
    + [
        # Low-Level Native Interface
        "get_client",
        "BatchMeta",
        "TransferQueueClient",
    ]
    + [
        # Sampler
        "BaseSampler",
        "GRPOGroupNSampler",
        "SequentialSampler",
        "RankAwareSampler",
        "SeqlenBalancedSampler",
    ]
)

version_folder = os.path.dirname(os.path.join(os.path.abspath(__file__)))

with open(os.path.join(version_folder, "version/version")) as f:
    __version__ = f.read().strip()
