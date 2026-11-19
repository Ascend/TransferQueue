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

import asyncio
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
import torch
import zmq.asyncio

from transfer_queue.storage.managers.simple_storage_manager import AsyncSimpleStorageManager, StorageUnitTimeout
from transfer_queue.storage.payload_transfer.base import Operation, RequestHeader, TransferTimeout
from transfer_queue.storage.payload_transfer.zmq import ZmqPayloadTransfer
from transfer_queue.storage.simple_storage import SimpleStorageUnit
from transfer_queue.utils.enum_utils import Role
from transfer_queue.utils.zmq_utils import (
    STORAGE_MANAGER_IDENTITY_PREFIX,
    ZMQMessage,
    ZMQRequestType,
    ZMQServerInfo,
    ZMQSocketPool,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["send_timeout", "receive_timeout", "disconnect", "cancel", "receive_error"])
@pytest.mark.parametrize("operation", list(Operation))
async def test_payload_failure_discards_socket_and_releases_permit(monkeypatch, failure, operation):
    peer = ZMQServerInfo(role=Role.STORAGE, id="unit", ip="127.0.0.1", ports={"put_get_socket": 5555})
    failed = MagicMock(closed=False)
    failed.send_multipart = AsyncMock()
    failed.recv_multipart = AsyncMock()
    error = {
        "send_timeout": zmq.Again(), "receive_timeout": zmq.Again(),
        "disconnect": zmq.ZMQError(zmq.ENOTSOCK), "cancel": asyncio.CancelledError(),
        "receive_error": ValueError("invalid response"),
    }[failure]
    if failure == "send_timeout":
        failed.send_multipart.side_effect = error
    else:
        failed.recv_multipart.side_effect = error
    healthy = MagicMock(closed=False)
    healthy.send_multipart = AsyncMock()
    healthy.recv_multipart = AsyncMock(return_value=ZMQMessage.create(
        request_type=ZMQRequestType.GET_DATA_RESPONSE, sender_id="unit", body={"data": {"text": ["next"]}},
    ).serialize())
    pool = ZMQSocketPool(MagicMock(), "manager", "put_get_socket", maxsize=1)
    connect = MagicMock(side_effect=[failed, healthy])
    monkeypatch.setattr(pool, "_connect", connect)
    client = ZmqPayloadTransfer(owner_id="manager", pool=pool)
    header = RequestHeader(Operation.GET, {"global_indexes": [0], "fields": ["text"]})
    try:
        expected = TransferTimeout if isinstance(error, zmq.Again) else type(error)
        with pytest.raises(expected) as caught:
            await client.request(
                peer, RequestHeader(operation, header.body),
                send=None if operation == Operation.GET else {"text": ["input"]},
                recv=None if operation == Operation.GET else (),
            )
        if expected is TransferTimeout:
            assert isinstance(caught.value.__cause__, zmq.Again)
            assert (caught.value.wire_bytes > 0) == (operation != Operation.GET)
        failed.close.assert_called_once_with(linger=0)
        # A single permit forces recovery to exercise both socket eviction and permit return.
        for _ in range(2):
            _, data = await asyncio.wait_for(client.request(peer, header, recv=None), timeout=2)
            assert data == {"text": ["next"]}
        assert connect.call_count == 2
        healthy.close.assert_not_called()
    finally:
        client.shutdown()
        pool.close()


@pytest.mark.asyncio
async def test_update_lost_reply_is_not_replayed_through_payload_adapter(monkeypatch):
    monkeypatch.setattr("transfer_queue.storage.simple_storage.get_node_ip_address", lambda: "127.0.0.1")
    unit = SimpleStorageUnit.__ray_metadata__.modified_class({})
    unit.storage_data.put_data({"value": [torch.tensor([1])]}, [0])
    peer = unit.get_zmq_server_info()
    socket = MagicMock(closed=False)
    applied = []

    async def send(frames, **kwargs):
        applied.append(unit.payload_transfer.handle_legacy(ZMQMessage.deserialize(frames)))

    socket.send_multipart = AsyncMock(side_effect=send)
    socket.recv_multipart = AsyncMock(side_effect=zmq.Again())
    pool = ZMQSocketPool(MagicMock(), "manager", "put_get_socket", maxsize=1)
    monkeypatch.setattr(pool, "_connect", MagicMock(return_value=socket))
    manager = AsyncSimpleStorageManager.__new__(AsyncSimpleStorageManager)
    manager.storage_manager_id = "manager"
    manager.storage_unit_infos = {peer.id: peer}
    manager._payload_peers = {peer.id: {"backend": "zmq", "address": peer}}
    manager.payload_transfer = ZmqPayloadTransfer(owner_id="manager", pool=pool)
    manager.close = lambda: None
    merge = MagicMock(side_effect=lambda old, new: torch.cat([old, new]))
    try:
        with pytest.raises(StorageUnitTimeout):
            await manager._update_to_single_storage_unit(
                [0], {"value": [torch.tensor([2])]}, merge,
                {"value": {"held": [True], "dtype": torch.int64, "is_non_tensor": False}}, peer.id,
            )
        assert len(applied) == 1
        assert applied[0].request_type == ZMQRequestType.UPDATE_DATA_RESPONSE
        torch.testing.assert_close(unit.storage_data.get_data(["value"], [0])["value"][0], torch.tensor([1, 2]))
        socket.send_multipart.assert_awaited_once()
        socket.close.assert_called_once_with(linger=0)
    finally:
        manager.payload_transfer.shutdown()
        pool.close()
        unit.shutdown()


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
