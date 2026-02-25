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

import sys
import time
from pathlib import Path

import numpy as np
import ray
import torch
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from tensordict import TensorDict

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from transfer_queue.client import TransferQueueClient  # noqa: E402
from transfer_queue.metadata import BatchMeta  # noqa: E402
from transfer_queue.storage.managers.base import KVStorageManager  # noqa: E402
from transfer_queue.storage.managers.factory import TransferQueueStorageManagerFactory  # noqa: E402
from transfer_queue.utils.zmq_utils import ZMQServerInfo  # noqa: E402

TEST_CONFIGS: list[tuple[tuple[int, int], torch.dtype]] = [
    ((5000, 5000), torch.float32),
    ((10000, 10000), torch.float32),
    ((20000, 20000), torch.float32),
    ((30000, 30000), torch.float32),
    ((40000, 40000), torch.float32),
    ((10000, 10000), torch.float16),
    ((20000, 20000), torch.float16),
    ((30000, 30000), torch.float16),
    ((40000, 40000), torch.float16),
    ((10000, 10000), torch.float64),
    ((20000, 20000), torch.float64),
    ((30000, 30000), torch.float64),
    ((40000, 40000), torch.float64),
]

# Step 1: Mock Controller Role
try:
    from transfer_queue.role import TransferQueueRole
except ImportError:
    from enum import Enum

    class TransferQueueRole(Enum):
        CONTROLLER = "controller"
        STORAGE = "storage"


def create_mock_controller():
    return ZMQServerInfo(
        role=TransferQueueRole.CONTROLLER,
        id="controller_0",
        ip="127.0.0.1",
        ports={
            "request_handle_socket": 9981,
            "data_status_update_socket": 9982,
            "handshake_socket": 9983,
        },
    )


# Step 2: Mock Storage Manager (Skip Controller Connect)
def ensure_mock_storage_manager_registered():
    """Ensure MockKVStorageManager is registered in current process."""

    if "KV_MOCK" not in TransferQueueStorageManagerFactory._registry:

        @TransferQueueStorageManagerFactory.register("KV_MOCK")
        class MockKVStorageManager(KVStorageManager):
            def _connect_to_controller(self):
                pass

            def _do_handshake_with_controller(self):
                pass

            async def notify_data_update(*args, **kwargs):
                return

        print("Registered KV_MOCK in current process")


ensure_mock_storage_manager_registered()


# Step 3: Define Writer and Reader Actors
@ray.remote
class WriterActor:
    def __init__(self, controller_info, config):
        ensure_mock_storage_manager_registered()

        self.client = TransferQueueClient(client_id=f"writer_{id(self)}", controller_info=controller_info)
        self.client.initialize_storage_manager("KV_MOCK", config)
        self.data = None
        self.meta = None

    def generate_data(
        self, partition_id, batch_size: int = 10000, seq_len: int = 10000
    ) -> tuple[TensorDict, BatchMeta, int]:
        data = TensorDict(
            {
                "input_ids": torch.randn(batch_size, seq_len, dtype=torch.float32),
            },
            batch_size=batch_size,
        )

        meta = BatchMeta(
            global_indexes=list(range(batch_size)),
            partition_ids=[partition_id] * batch_size,
            field_schema={
                "input_ids": {
                    "dtype": torch.float32,
                    "shape": (seq_len,),
                    "is_nested": False,
                    "is_non_tensor": False,
                }
            },
            production_status=np.zeros(batch_size, dtype=np.int8),
        )

        self.data = data
        self.meta = meta

        return meta

    def put_once(self) -> float:
        t0 = time.time()
        self.client.put(data=self.data, metadata=self.meta)
        return time.time() - t0


@ray.remote
class ReaderActor:
    def __init__(self, controller_info, config):
        ensure_mock_storage_manager_registered()

        self.client = TransferQueueClient(client_id=f"reader_{id(self)}", controller_info=controller_info)
        self.client.initialize_storage_manager("KV_MOCK", config)

    def get_once(self, metadata: BatchMeta):
        t0 = time.time()
        self.client.get_data(metadata)
        return time.time() - t0


# Step 4: Main Test Function
def main():
    if not ray.is_initialized():
        ray.init(address="auto")

    client = None
    controller_info = create_mock_controller()
    config = {
        "client_name": "RayStorageClient",
        "controller_info": controller_info,
    }

    client = TransferQueueClient(client_id="test_driver", controller_info=controller_info)
    client.initialize_storage_manager("KV_MOCK", config)

    print("Driver initialized (mocked)")

    nodes = ray.nodes()
    ip_to_nodeid = {}
    for n in nodes:
        addr = n.get("NodeManagerAddress") or n.get("node_ip_address") or n.get("NodeIP")
        node_id = n["NodeID"] if "NodeID" in n else n.get("NodeID") or n.get("node_id")
        if addr and node_id:
            ip_to_nodeid[addr] = node_id

    ip_A = "10.90.41.117"  # Writer
    ip_B = "10.90.41.116"  # Reader
    node_id_A = ip_to_nodeid.get(ip_A)
    node_id_B = ip_to_nodeid.get(ip_B)
    assert node_id_A and node_id_B, f"cannot find node ids for {ip_A}, {ip_B}: {ip_to_nodeid}"

    writer = WriterActor.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id_A, soft=False),
    ).remote(controller_info, config)
    reader = ReaderActor.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id_B, soft=False),
    ).remote(controller_info, config)

    partition_id = "train_step_0"
    meta = ray.get(writer.generate_data.remote(partition_id=partition_id, batch_size=4000, seq_len=4000))

    for i in range(1):
        cost = ray.get(writer.put_once.remote())
        print(f"[WriterActor] The time consumed by the {i}th put costs: {cost:.2f}s")

    for i in range(3):
        cost = ray.get(reader.get_once.remote(meta))
        print(f"[ReaderActor] The time consumed by the {i}th get costs: {cost:.2f}s")

    print("Actor-to-Actor communication works!")


if __name__ == "__main__":
    main()
