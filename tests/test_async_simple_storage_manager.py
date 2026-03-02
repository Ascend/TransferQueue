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

import numpy as np
import pytest
import pytest_asyncio
import torch
import zmq
from tensordict import TensorDict

# Setup path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from transfer_queue.metadata import BatchMeta  # noqa: E402
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


@pytest.mark.asyncio
async def test_async_storage_manager_mock_operations(mock_async_storage_manager):
    """Test AsyncSimpleStorageManager operations with mocked ZMQ."""
    manager = mock_async_storage_manager

    # Create test metadata using columnar API
    batch_meta = BatchMeta(
        global_indexes=[0, 1],
        partition_ids=["0", "0"],
        field_schema={
            "test_field": {
                "dtype": torch.float32,
                "shape": (2,),
                "is_nested": False,
                "is_non_tensor": False,
            }
        },
        production_status=np.ones(2, dtype=np.int8),
        _custom_backend_meta={
            0: {"_su_id": "storage_0"},
            1: {"_su_id": "storage_1"},
        },
    )

    # Create test data
    test_data = TensorDict(
        {
            "test_field": torch.stack([torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]),
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

        # Create test metadata using columnar API
        batch_meta = BatchMeta(
            global_indexes=[0],
            partition_ids=["0"],
            field_schema={
                "test_field": {
                    "dtype": torch.float32,
                    "shape": (2,),
                    "is_nested": False,
                    "is_non_tensor": False,
                }
            },
            production_status=np.ones(1, dtype=np.int8),
            _custom_backend_meta={0: {"_su_id": "storage_0"}},
        )

        # Create test data
        test_data = TensorDict(
            {
                "test_field": torch.tensor([[1.0, 2.0]]),
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


@pytest.mark.asyncio
async def test_put_data_notifies_su_id():
    """put_data 调用 notify_data_update 时必须传入 custom_backend_meta 含 _su_id."""
    storage_unit_infos = {
        "storage_0": ZMQServerInfo(
            role=TransferQueueRole.STORAGE,
            id="storage_0",
            ip="127.0.0.1",
            ports={"put_get_socket": 19000},
        ),
        "storage_1": ZMQServerInfo(
            role=TransferQueueRole.STORAGE,
            id="storage_1",
            ip="127.0.0.1",
            ports={"put_get_socket": 19001},
        ),
    }

    with patch("transfer_queue.storage.managers.base.TransferQueueStorageManager._connect_to_controller"):
        manager = AsyncSimpleStorageManager.__new__(AsyncSimpleStorageManager)
        manager.storage_manager_id = "test_manager"
        manager.storage_unit_infos = storage_unit_infos
        manager.controller_info = None
        manager.data_status_update_socket = None
        manager.controller_handshake_socket = None
        manager.zmq_context = None

    manager._put_to_single_storage_unit = AsyncMock()
    notify_mock = AsyncMock()
    manager.notify_data_update = notify_mock

    batch_meta = BatchMeta(
        global_indexes=[0, 1, 2, 3],
        partition_ids=["p0"] * 4,
        field_schema={"f": {"dtype": torch.float32, "shape": (2,), "is_nested": False, "is_non_tensor": False}},
        production_status=np.ones(4, dtype=np.int8),
    )
    data = TensorDict({"f": torch.randn(4, 2)}, batch_size=4)

    await manager.put_data(data, batch_meta)

    # notify_data_update 必须被调用且 custom_backend_meta 不为 None
    notify_mock.assert_awaited_once()
    call_kwargs = notify_mock.call_args
    custom_backend_meta = call_kwargs.kwargs.get("custom_backend_meta") or (
        call_kwargs.args[-1] if call_kwargs.args else None
    )
    assert custom_backend_meta is not None, "custom_backend_meta 未传入 notify_data_update"
    # 每个 gi 都应有 _su_id
    for gi in [0, 1, 2, 3]:
        assert gi in custom_backend_meta, f"gi={gi} 不在 custom_backend_meta"
        assert "_su_id" in custom_backend_meta[gi], f"gi={gi} 缺少 _su_id"
        assert custom_backend_meta[gi]["_su_id"] in storage_unit_infos


@pytest.mark.asyncio
async def test_put_data_no_batch_counter():
    """put_data 不应存在 _batch_counter 属性（已删除）."""
    storage_unit_infos = {
        "storage_0": ZMQServerInfo(
            role=TransferQueueRole.STORAGE,
            id="storage_0",
            ip="127.0.0.1",
            ports={"put_get_socket": 19002},
        ),
    }
    with patch("transfer_queue.storage.managers.base.TransferQueueStorageManager._connect_to_controller"):
        manager = AsyncSimpleStorageManager.__new__(AsyncSimpleStorageManager)
        manager.storage_manager_id = "test_manager_2"
        manager.storage_unit_infos = storage_unit_infos
        manager.controller_info = None
        manager.data_status_update_socket = None
        manager.controller_handshake_socket = None
        manager.zmq_context = None

    assert not hasattr(manager, "_batch_counter"), "_batch_counter 应已删除"


@pytest.mark.asyncio
async def test_get_data_routes_from_custom_backend_meta():
    """get_data 应从 metadata._custom_backend_meta 读取 _su_id 做路由，不重算."""
    storage_unit_infos = {
        "storage_0": ZMQServerInfo(
            role=TransferQueueRole.STORAGE,
            id="storage_0",
            ip="127.0.0.1",
            ports={"put_get_socket": 19010},
        ),
        "storage_1": ZMQServerInfo(
            role=TransferQueueRole.STORAGE,
            id="storage_1",
            ip="127.0.0.1",
            ports={"put_get_socket": 19011},
        ),
    }
    with patch("transfer_queue.storage.managers.base.TransferQueueStorageManager._connect_to_controller"):
        manager = AsyncSimpleStorageManager.__new__(AsyncSimpleStorageManager)
        manager.storage_manager_id = "test_get"
        manager.storage_unit_infos = storage_unit_infos
        manager.controller_info = None
        manager.data_status_update_socket = None
        manager.controller_handshake_socket = None
        manager.zmq_context = None

    # gi=0,1 -> storage_0; gi=2,3 -> storage_1（通过 _custom_backend_meta 指定）
    batch_meta = BatchMeta(
        global_indexes=[0, 1, 2, 3],
        partition_ids=["p0"] * 4,
        field_schema={"f": {"dtype": torch.float32, "shape": (2,), "is_nested": False, "is_non_tensor": False}},
        production_status=np.ones(4, dtype=np.int8),
        _custom_backend_meta={
            0: {"_su_id": "storage_0"},
            1: {"_su_id": "storage_0"},
            2: {"_su_id": "storage_1"},
            3: {"_su_id": "storage_1"},
        },
    )

    # Mock _get_from_single_storage_unit to record which su_id and gi were requested
    called_with: dict[str, list] = {}

    async def fake_get(global_indexes, fields, target_storage_unit=None, **kwargs):
        su = target_storage_unit
        called_with[su] = list(global_indexes)
        tensors = [torch.zeros(2) for _ in global_indexes]
        return global_indexes, fields, {"f": tensors}, b""

    manager._get_from_single_storage_unit = fake_get

    await manager.get_data(batch_meta)

    assert "storage_0" in called_with, "storage_0 未被 get 调用"
    assert "storage_1" in called_with, "storage_1 未被 get 调用"
    assert set(called_with["storage_0"]) == {0, 1}
    assert set(called_with["storage_1"]) == {2, 3}


@pytest.mark.asyncio
async def test_clear_data_routes_from_custom_backend_meta():
    """clear_data 应从 metadata._custom_backend_meta 读取 _su_id 做路由."""
    storage_unit_infos = {
        "storage_0": ZMQServerInfo(
            role=TransferQueueRole.STORAGE,
            id="storage_0",
            ip="127.0.0.1",
            ports={"put_get_socket": 19020},
        ),
        "storage_1": ZMQServerInfo(
            role=TransferQueueRole.STORAGE,
            id="storage_1",
            ip="127.0.0.1",
            ports={"put_get_socket": 19021},
        ),
    }
    with patch("transfer_queue.storage.managers.base.TransferQueueStorageManager._connect_to_controller"):
        manager = AsyncSimpleStorageManager.__new__(AsyncSimpleStorageManager)
        manager.storage_manager_id = "test_clear"
        manager.storage_unit_infos = storage_unit_infos
        manager.controller_info = None
        manager.data_status_update_socket = None
        manager.controller_handshake_socket = None
        manager.zmq_context = None

    batch_meta = BatchMeta(
        global_indexes=[0, 1, 2, 3],
        partition_ids=["p0"] * 4,
        field_schema={"f": {"dtype": torch.float32, "shape": (2,), "is_nested": False, "is_non_tensor": False}},
        production_status=np.ones(4, dtype=np.int8),
        _custom_backend_meta={
            0: {"_su_id": "storage_0"},
            1: {"_su_id": "storage_0"},
            2: {"_su_id": "storage_1"},
            3: {"_su_id": "storage_1"},
        },
    )

    called_with: dict[str, list] = {}

    async def fake_clear(global_indexes, target_storage_unit=None, **kwargs):
        called_with[target_storage_unit] = list(global_indexes)

    manager._clear_single_storage_unit = fake_clear

    await manager.clear_data(batch_meta)

    assert set(called_with.get("storage_0", [])) == {0, 1}
    assert set(called_with.get("storage_1", [])) == {2, 3}
