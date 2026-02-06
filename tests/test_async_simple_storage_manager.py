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
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
import pytest_asyncio
import torch
import zmq
from tensordict import TensorDict

# Setup path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from transfer_queue.metadata import BatchMeta, FieldMeta, SampleMeta  # noqa: E402
from transfer_queue.storage import AsyncSimpleStorageManager  # noqa: E402
from transfer_queue.utils.enum_utils import TransferQueueRole  # noqa: E402
from transfer_queue.utils.zmq_utils import ZMQMessage, ZMQRequestType, ZMQServerInfo  # noqa: E402


@pytest_asyncio.fixture
async def mock_async_storage_manager():
    """Create a mock AsyncSimpleStorageManager for testing."""

    # Mock storage unit infos
    storage_unit_infos = {
        "storage_0": ZMQServerInfo(
            role=TransferQueueRole.STORAGE,
            id="storage_0",
            ip="127.0.0.1",
            ports={"put_get_socket": 12345},
        ),
        "storage_1": ZMQServerInfo(
            role=TransferQueueRole.STORAGE,
            id="storage_1",
            ip="127.0.0.1",
            ports={"put_get_socket": 12346},
        ),
    }

    # Mock controller info
    controller_info = ZMQServerInfo(
        role=TransferQueueRole.CONTROLLER,
        id="controller_0",
        ip="127.0.0.1",
        ports={"handshake_socket": 12347, "data_status_update_socket": 12348},
    )

    config = {
        "zmq_info": storage_unit_infos,
    }

    # Mock the handshake process entirely to avoid ZMQ complexity
    with patch(
        "transfer_queue.storage.managers.base.TransferQueueStorageManager._connect_to_controller"
    ) as mock_connect:
        # Mock the manager without actually connecting
        manager = AsyncSimpleStorageManager.__new__(AsyncSimpleStorageManager)
        manager.storage_manager_id = "test_storage_manager"
        manager.config = config
        manager.controller_info = controller_info
        manager.storage_unit_infos = storage_unit_infos
        manager.data_status_update_socket = None
        manager.controller_handshake_socket = None
        manager.zmq_context = None

        # Add mapping functions
        storage_unit_keys = list(storage_unit_infos.keys())
        manager.global_index_storage_unit_mapping = lambda x: storage_unit_keys[x % len(storage_unit_keys)]
        manager.global_index_local_index_mapping = lambda x: x // len(storage_unit_keys)

        # Mock essential methods
        manager._connect_to_controller = mock_connect

        yield manager


@pytest.mark.asyncio
async def test_async_storage_manager_initialization(mock_async_storage_manager):
    """Test AsyncSimpleStorageManager initialization."""
    manager = mock_async_storage_manager

    # Test basic properties
    assert len(manager.storage_unit_infos) == 2
    assert "storage_0" in manager.storage_unit_infos
    assert "storage_1" in manager.storage_unit_infos

    # Test mapping functions
    assert manager.global_index_storage_unit_mapping(0) == "storage_0"
    assert manager.global_index_storage_unit_mapping(1) == "storage_1"
    assert manager.global_index_local_index_mapping(0) == 0
    assert manager.global_index_local_index_mapping(3) == 1


@pytest.mark.asyncio
async def test_async_storage_manager_mock_operations(mock_async_storage_manager):
    """Test AsyncSimpleStorageManager operations with mocked ZMQ."""
    manager = mock_async_storage_manager

    # Create test metadata
    sample_metas = [
        SampleMeta(
            partition_id="0",
            global_index=0,
            fields={
                "test_field": FieldMeta(name="test_field", dtype=torch.float32, shape=(2,)),
            },
        ),
        SampleMeta(
            partition_id="0",
            global_index=1,
            fields={
                "test_field": FieldMeta(name="test_field", dtype=torch.float32, shape=(2,)),
            },
        ),
    ]
    batch_meta = BatchMeta(samples=sample_metas)

    # Create test data
    test_data = TensorDict(
        {
            "test_field": [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])],
        },
        batch_size=2,
    )

    manager._put_to_single_storage_unit = AsyncMock()
    manager._get_from_single_storage_unit = AsyncMock(
        return_value=(
            [0, 1],
            ["test_field"],
            {"test_field": [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]},
            b"this is the serialized message object.",
        )
    )
    manager._clear_single_storage_unit = AsyncMock()
    manager.notify_data_update = AsyncMock()

    # Test put_data (should not raise exceptions)
    await manager.put_data(test_data, batch_meta)
    manager.notify_data_update.assert_awaited_once()

    # Test get_data
    retrieved_data = await manager.get_data(batch_meta)
    assert "test_field" in retrieved_data

    # Test clear_data
    await manager.clear_data(batch_meta)


@pytest.mark.asyncio
async def test_async_storage_manager_mapping_functions():
    """Test AsyncSimpleStorageManager mapping functions."""

    # Mock storage unit infos
    storage_unit_infos = {
        "storage_0": ZMQServerInfo(
            role=TransferQueueRole.STORAGE,
            id="storage_0",
            ip="127.0.0.1",
            ports={"put_get_socket": 12345},
        ),
        "storage_1": ZMQServerInfo(
            role=TransferQueueRole.STORAGE,
            id="storage_1",
            ip="127.0.0.1",
            ports={"put_get_socket": 12346},
        ),
        "storage_2": ZMQServerInfo(
            role=TransferQueueRole.STORAGE,
            id="storage_2",
            ip="127.0.0.1",
            ports={"put_get_socket": 12347},
        ),
    }

    # Mock controller info
    controller_info = ZMQServerInfo(
        role=TransferQueueRole.CONTROLLER,
        id="controller_0",
        ip="127.0.0.1",
        ports={"handshake_socket": 12348, "data_status_update_socket": 12349},
    )

    config = {
        "zmq_info": storage_unit_infos,
    }

    # Mock ZMQ operations
    with (
        patch("transfer_queue.storage.managers.base.create_zmq_socket") as mock_create_socket,
        patch("zmq.Poller") as mock_poller,
    ):
        # Create mock socket with proper sync methods
        mock_socket = Mock()
        mock_socket.connect = Mock()  # sync method
        mock_socket.send = Mock()  # sync method
        mock_create_socket.return_value = mock_socket

        # Mock poller with sync methods
        mock_poller_instance = Mock()
        mock_poller_instance.register = Mock()  # sync method
        # Return mock socket in poll to simulate handshake response
        mock_poller_instance.poll = Mock(return_value=[(mock_socket, zmq.POLLIN)])  # sync method
        mock_poller.return_value = mock_poller_instance

        # Mock handshake response
        handshake_response = ZMQMessage.create(
            request_type=ZMQRequestType.HANDSHAKE_ACK,
            sender_id="controller_0",
            body={"message": "Handshake successful"},
        )
        mock_socket.recv_multipart = Mock(return_value=handshake_response.serialize())

        # Create manager
        manager = AsyncSimpleStorageManager(controller_info, config)

        # Test round-robin mapping for 3 storage units
        # global_index -> storage_unit mapping: 0->storage_0, 1->storage_1, 2->storage_2,
        # 3->storage_0, 4->storage_1, ...
        assert manager.global_index_storage_unit_mapping(0) == "storage_0"
        assert manager.global_index_storage_unit_mapping(1) == "storage_1"
        assert manager.global_index_storage_unit_mapping(2) == "storage_2"
        assert manager.global_index_storage_unit_mapping(3) == "storage_0"
        assert manager.global_index_storage_unit_mapping(4) == "storage_1"
        assert manager.global_index_storage_unit_mapping(5) == "storage_2"

        # global_index -> local_index mapping: global_index // num_storage_units
        assert manager.global_index_local_index_mapping(0) == 0
        assert manager.global_index_local_index_mapping(1) == 0
        assert manager.global_index_local_index_mapping(2) == 0
        assert manager.global_index_local_index_mapping(3) == 1
        assert manager.global_index_local_index_mapping(4) == 1
        assert manager.global_index_local_index_mapping(5) == 1


@pytest.mark.asyncio
async def test_async_storage_manager_error_handling():
    """Test AsyncSimpleStorageManager error handling."""

    # Mock storage unit infos
    storage_unit_infos = {
        "storage_0": ZMQServerInfo(
            role=TransferQueueRole.STORAGE,
            id="storage_0",
            ip="127.0.0.1",
            ports={"put_get_socket": 12345},
        ),
    }

    # Mock controller info
    controller_info = ZMQServerInfo(
        role=TransferQueueRole.CONTROLLER,
        id="controller_0",
        ip="127.0.0.1",
        ports={"handshake_socket": 12346, "data_status_update_socket": 12347},
    )

    config = {
        "zmq_info": storage_unit_infos,
    }

    # Mock ZMQ operations
    with (
        patch("transfer_queue.storage.managers.base.create_zmq_socket") as mock_create_socket,
        patch("zmq.Poller") as mock_poller,
    ):
        # Create mock socket with proper sync methods
        mock_socket = Mock()
        mock_socket.connect = Mock()  # sync method
        mock_socket.send = Mock()  # sync method
        mock_create_socket.return_value = mock_socket

        # Mock poller with sync methods
        mock_poller_instance = Mock()
        mock_poller_instance.register = Mock()  # sync method
        # Return mock socket in poll to simulate handshake response
        mock_poller_instance.poll = Mock(return_value=[(mock_socket, zmq.POLLIN)])  # sync method
        mock_poller.return_value = mock_poller_instance

        # Mock handshake response
        handshake_response = ZMQMessage.create(
            request_type=ZMQRequestType.HANDSHAKE_ACK,
            sender_id="controller_0",
            body={"message": "Handshake successful"},
        )
        mock_socket.recv_multipart = Mock(return_value=handshake_response.serialize())

        # Create manager
        manager = AsyncSimpleStorageManager(controller_info, config)

        # Mock operations that raise exceptions
        manager._put_to_single_storage_unit = AsyncMock(side_effect=RuntimeError("Mock PUT error"))
        manager._get_from_single_storage_unit = AsyncMock(side_effect=RuntimeError("Mock GET error"))
        manager._clear_single_storage_unit = AsyncMock(side_effect=RuntimeError("Mock CLEAR error"))
        manager.notify_data_update = AsyncMock()

        # Create test metadata
        sample_metas = [
            SampleMeta(
                partition_id="0",
                global_index=0,
                fields={
                    "test_field": FieldMeta(name="test_field", dtype=torch.float32, shape=(2,)),
                },
            ),
        ]
        batch_meta = BatchMeta(samples=sample_metas)

        # Create test data
        test_data = TensorDict(
            {
                "test_field": [torch.tensor([1.0, 2.0])],
            },
            batch_size=1,
        )

        # Test that exceptions are properly raised
        with pytest.raises(RuntimeError, match="Mock PUT error"):
            await manager.put_data(test_data, batch_meta)

        with pytest.raises(RuntimeError, match="Mock GET error"):
            await manager.get_data(batch_meta)

        # Note: clear_data uses return_exceptions=True, so it doesn't raise exceptions directly
        # Instead, we can verify that the clear operation was attempted
        await manager.clear_data(batch_meta)  # Should not raise due to return_exceptions=True
