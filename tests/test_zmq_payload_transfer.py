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


"""Exercise the field-data contract over the existing ZMQ multipart path."""

from uuid import uuid4

import pytest
import torch
import zmq.asyncio

from transfer_queue.storage.payload_transfer.base import Operation, RequestHeader
from transfer_queue.storage.payload_transfer.zmq import ZmqPayloadTransfer
from transfer_queue.storage.simple_storage import SimpleStorageUnit
from transfer_queue.utils.zmq_utils import STORAGE_MANAGER_IDENTITY_PREFIX, ZMQSocketPool


@pytest.mark.asyncio
async def test_zmq_field_data_roundtrip(monkeypatch):
    monkeypatch.setattr("transfer_queue.storage.simple_storage.get_node_ip_address", lambda: "127.0.0.1")
    unit = SimpleStorageUnit.__ray_metadata__.modified_class({})
    context = zmq.asyncio.Context()
    pool = ZMQSocketPool(context, f"{STORAGE_MANAGER_IDENTITY_PREFIX}{uuid4().hex[:8]}", "put_get_socket", timeout=3)
    client = ZmqPayloadTransfer(owner_id="test-manager", pool=pool)
    try:
        connection = await client.connect({"backend": "zmq", "address": unit.get_zmq_server_info()})
        put = RequestHeader(Operation.PUT, {"global_indexes": [0]})
        _, ack = await client.request(connection, put, send={"value": [torch.tensor([1, 2])], "text": ["first"]})
        assert ack is None

        update = RequestHeader(
            Operation.UPDATE,
            {
                "global_indexes": [0],
                "merge_fn": lambda old, new: torch.cat([old, new]),
                "merge_plan": {"value": {"held": [True], "dtype": torch.int64, "is_non_tensor": False}},
            },
        )
        reply, ack = await client.request(connection, update, send={"value": [torch.tensor([3])]})
        assert ack is None
        assert tuple(reply.body["stored_shapes"]["value"]["shapes"][0]) == (3,)

        get = RequestHeader(Operation.GET, {"global_indexes": [0], "fields": ["value", "text"]})
        _, data = await client.request(connection, get, recv=None)
        torch.testing.assert_close(data["value"][0], torch.tensor([1, 2, 3]))
        assert data["text"] == ["first"]

        with pytest.raises(NotImplementedError, match="receive destinations"):
            await client.request(connection, get, recv=(memoryview(bytearray(24)),))
        assert unit._arrivals_by_op["GET_DATA"] == 1
    finally:
        await client.close()
        pool.close()
        context.term()
        unit.shutdown()
