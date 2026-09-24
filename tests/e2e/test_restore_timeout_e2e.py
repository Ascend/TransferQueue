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

import threading
import time
from unittest.mock import patch

import pytest
import ray
import torch
import zmq
import zmq.asyncio
from omegaconf import OmegaConf
from tensordict import TensorDict

import transfer_queue as tq
from transfer_queue.storage.managers.simple_storage_manager import AsyncSimpleStorageManager
from transfer_queue.utils.zmq_utils import ZMQMessage


def test_cancelled_delayed_load_cannot_overwrite_reused_index(tmp_path):
    ray.init(namespace="review_timeout")
    tq.init(
        OmegaConf.create(
            {
                "backend": {
                    "storage_backend": "SimpleStorage",
                    "SimpleStorage": {"num_data_storage_units": 1, "total_storage_size": 20},
                }
            }
        )
    )
    try:
        tq.kv_batch_put(["key"], "p", TensorDict({"x": torch.tensor([[1]])}, batch_size=1))
        tq.dump_data_by_key(tmp_path / "dump", ["key"], "p")
        tq.kv_batch_put(["key"], "p", TensorDict({"x": torch.tensor([[2]])}, batch_size=1))
        manager = tq.get_client().storage_manager
        real_info = next(iter(manager.storage_unit_infos.values()))
        real_addr = real_info.to_addr("put_get_socket")
        # Relay the request immediately, but delay delivery to the unit to model a queued load.
        ready = threading.Event()
        finished = threading.Event()
        port = []

        def relay():
            ctx = zmq.Context()
            front = ctx.socket(zmq.ROUTER)
            front.setsockopt(zmq.RCVTIMEO, 10000)
            port.append(front.bind_to_random_port("tcp://127.0.0.1"))
            back = ctx.socket(zmq.DEALER)
            back.setsockopt(zmq.RCVTIMEO, 10000)
            back.setsockopt(zmq.IDENTITY, b"TQ_STORAGE_review_relay")
            back.connect(real_addr)
            ready.set()
            req = front.recv_multipart()
            time.sleep(2)
            back.send_multipart(req[1:])
            reply = back.recv_multipart()
            assert not ZMQMessage.deserialize(reply).body["success"]
            finished.set()
            front.close(linger=0)
            back.close(linger=0)
            ctx.term()

        thread = threading.Thread(target=relay, daemon=True)
        thread.start()
        assert ready.wait(10)

        async def timeout_load(shards, target_storage_unit, restore):
            ctx = zmq.asyncio.Context()
            sock = ctx.socket(zmq.DEALER)
            sock.setsockopt(zmq.RCVTIMEO, 100)
            sock.connect(f"tcp://127.0.0.1:{port[0]}")
            try:
                return await AsyncSimpleStorageManager._load_selected_rows.__wrapped__(
                    manager, shards, target_storage_unit, restore=restore, socket=sock
                )
            finally:
                sock.close(linger=0)
                ctx.term()

        with patch.object(manager, "_load_selected_rows", timeout_load):
            with pytest.raises(zmq.error.Again):
                tq.load_data_by_key(tmp_path / "dump")
        value = tq.kv_batch_get(["key"], "p", ["x"])["x"][0].item()

        assert value == 2
        tq.get_client().clear_partition("p")
        tq.kv_batch_put(["other"], "unrelated", TensorDict({"x": torch.tensor([[99]])}, batch_size=1))
        assert finished.wait(10)
        thread.join(timeout=10)
        later = tq.kv_batch_get(["other"], "unrelated", ["x"])["x"][0].item()

        assert later == 99
    finally:
        tq.close()
        ray.shutdown()
