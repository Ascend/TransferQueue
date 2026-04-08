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

from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest
import pytest_asyncio
import torch
import zmq
from tensordict import NonTensorStack, TensorDict

from transfer_queue.metadata import BatchMeta
from transfer_queue.storage import AsyncSimpleStorageManager
from transfer_queue.utils.enum_utils import TransferQueueRole
from transfer_queue.utils.zmq_utils import ZMQMessage, ZMQRequestType, ZMQServerInfo


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
            ["test_field"],
            {"test_field": [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]},
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
            request_type=ZMQRequestType.HANDSHAKE_ACK,  # type: ignore[arg-type]
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
async def test_get_data_routes_from_hash():
    """get_data should route using global_idx % num_su (hash routing)."""
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

    # global_index 0,2 → storage_0 (even % 2 = 0); 1,3 → storage_1 (odd % 2 = 1)
    batch_meta = BatchMeta(
        global_indexes=[0, 1, 2, 3],
        partition_ids=["p0"] * 4,
        field_schema={"f": {"dtype": torch.float32, "shape": (2,), "is_nested": False, "is_non_tensor": False}},
        production_status=np.ones(4, dtype=np.int8),
    )

    # Mock _get_from_single_storage_unit to record which su_id and global_index were requested
    called_with: dict[str, list] = {}

    async def fake_get(global_indexes, fields, target_storage_unit=None, **kwargs):
        su = target_storage_unit
        called_with[su] = list(global_indexes)
        tensors = [torch.zeros(2) for _ in global_indexes]
        return fields, {"f": tensors}

    manager._get_from_single_storage_unit = fake_get

    await manager.get_data(batch_meta)

    assert "storage_0" in called_with, "storage_0 was not called by get"
    assert "storage_1" in called_with, "storage_1 was not called by get"
    assert set(called_with["storage_0"]) == {0, 2}
    assert set(called_with["storage_1"]) == {1, 3}


@pytest.mark.asyncio
async def test_clear_data_routes_from_hash():
    """clear_data should route using global_idx % num_su (hash routing)."""
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

    # global_index 0,2 → storage_0 (even); 1,3 → storage_1 (odd)
    batch_meta = BatchMeta(
        global_indexes=[0, 1, 2, 3],
        partition_ids=["p0"] * 4,
        field_schema={"f": {"dtype": torch.float32, "shape": (2,), "is_nested": False, "is_non_tensor": False}},
        production_status=np.ones(4, dtype=np.int8),
    )

    called_with: dict[str, list] = {}

    async def fake_clear(global_indexes, target_storage_unit=None, **kwargs):
        called_with[target_storage_unit] = list(global_indexes)

    manager._clear_single_storage_unit = fake_clear

    await manager.clear_data(batch_meta)

    assert set(called_with.get("storage_0", [])) == {0, 2}
    assert set(called_with.get("storage_1", [])) == {1, 3}


@pytest.mark.asyncio
async def test_hash_routing_stable_across_batch_sizes():
    """Hash routing must produce the same SU assignment regardless of batch size.

    Put 10 samples in one batch vs two batches of 5 — each global_idx must route
    to the same SU in both cases.
    """
    storage_unit_infos = {
        "storage_0": ZMQServerInfo(
            role=TransferQueueRole.STORAGE,
            id="storage_0",
            ip="127.0.0.1",
            ports={"put_get_socket": 19030},
        ),
        "storage_1": ZMQServerInfo(
            role=TransferQueueRole.STORAGE,
            id="storage_1",
            ip="127.0.0.1",
            ports={"put_get_socket": 19031},
        ),
    }
    with patch("transfer_queue.storage.managers.base.TransferQueueStorageManager._connect_to_controller"):
        manager = AsyncSimpleStorageManager.__new__(AsyncSimpleStorageManager)
        manager.storage_manager_id = "test_hash_batch"
        manager.storage_unit_infos = storage_unit_infos
        manager.controller_info = None
        manager.data_status_update_socket = None
        manager.controller_handshake_socket = None
        manager.zmq_context = None

    all_indexes = list(range(10))
    full_routing = manager._group_by_hash(all_indexes)

    # Build per-index mapping from the full-batch result
    idx_to_su_full: dict[int, str] = {}
    for su_id, group in full_routing.items():
        for gi in group.global_indexes:
            idx_to_su_full[gi] = su_id

    # Route as two batches of 5
    batch_a_routing = manager._group_by_hash(all_indexes[:5])
    batch_b_routing = manager._group_by_hash(all_indexes[5:])

    idx_to_su_split: dict[int, str] = {}
    for su_id, group in batch_a_routing.items():
        for gi in group.global_indexes:
            idx_to_su_split[gi] = su_id
    for su_id, group in batch_b_routing.items():
        for gi in group.global_indexes:
            idx_to_su_split[gi] = su_id

    assert idx_to_su_full == idx_to_su_split, (
        f"Routing differs between full batch and split batches:\n  full:  {idx_to_su_full}\n  split: {idx_to_su_split}"
    )

    # Verify RoutingGroup carries correct batch_positions alongside global_indexes
    for su_id, group in full_routing.items():
        assert len(group.global_indexes) == len(group.batch_positions)
        for gi, pos in zip(group.global_indexes, group.batch_positions, strict=False):
            assert all_indexes[pos] == gi


@pytest.mark.asyncio
async def test_hash_routing_stable_reversed_order():
    """Hash routing must produce the same SU assignment regardless of key order.

    Forward order [0..9] and reversed order [9..0] must yield identical routing.
    """
    storage_unit_infos = {
        "storage_0": ZMQServerInfo(
            role=TransferQueueRole.STORAGE,
            id="storage_0",
            ip="127.0.0.1",
            ports={"put_get_socket": 19040},
        ),
        "storage_1": ZMQServerInfo(
            role=TransferQueueRole.STORAGE,
            id="storage_1",
            ip="127.0.0.1",
            ports={"put_get_socket": 19041},
        ),
    }
    with patch("transfer_queue.storage.managers.base.TransferQueueStorageManager._connect_to_controller"):
        manager = AsyncSimpleStorageManager.__new__(AsyncSimpleStorageManager)
        manager.storage_manager_id = "test_hash_order"
        manager.storage_unit_infos = storage_unit_infos
        manager.controller_info = None
        manager.data_status_update_socket = None
        manager.controller_handshake_socket = None
        manager.zmq_context = None

    forward = list(range(10))
    reversed_indexes = list(reversed(forward))

    routing_fwd = manager._group_by_hash(forward)
    routing_rev = manager._group_by_hash(reversed_indexes)

    # Build per-index mapping
    def _to_idx_map(routing):
        m = {}
        for su_id, group in routing.items():
            for gi in group.global_indexes:
                m[gi] = su_id
        return m

    assert _to_idx_map(routing_fwd) == _to_idx_map(routing_rev), "Hash routing should be order-independent"


class TestSelectByPositions:
    """Test _select_by_positions static method for all field types."""

    def test_regular_tensor(self):
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = AsyncSimpleStorageManager._select_by_positions(t, [0, 2])
        assert torch.equal(result, torch.tensor([[1.0, 2.0], [5.0, 6.0]]))

    def test_nested_tensor(self):
        t = torch.nested.as_nested_tensor(
            [torch.tensor([1.0]), torch.tensor([2.0, 3.0]), torch.tensor([4.0, 5.0, 6.0])],
            layout=torch.jagged,
        )
        result = AsyncSimpleStorageManager._select_by_positions(t, [0, 2])
        assert isinstance(result, list)
        assert len(result) == 2
        assert torch.equal(result[0], torch.tensor([1.0]))
        assert torch.equal(result[1], torch.tensor([4.0, 5.0, 6.0]))

    def test_non_tensor_stack(self):
        nts = NonTensorStack("a", "b", "c")
        result = AsyncSimpleStorageManager._select_by_positions(nts, [1, 2])
        assert isinstance(result, NonTensorStack)
        assert result.tolist() == ["b", "c"]

    def test_list(self):
        data = [{"x": 1}, {"x": 2}, {"x": 3}]
        result = AsyncSimpleStorageManager._select_by_positions(data, [0, 2])
        assert result == [{"x": 1}, {"x": 3}]

    def test_numpy_array(self):
        arr = np.array([10, 20, 30])
        result = AsyncSimpleStorageManager._select_by_positions(arr, [0, 2])
        np.testing.assert_array_equal(result, np.array([10, 30]))

    def test_regular_tensor_single_element(self):
        """Case 1: Single element selection returns a single-row view."""
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = AsyncSimpleStorageManager._select_by_positions(t, [1])
        assert result.shape == (1, 2)
        assert torch.equal(result, torch.tensor([[3.0, 4.0]]))

    def test_regular_tensor_strided_slice(self):
        """Case 2: Constant stride (step > 1) uses Python slicing for zero-copy view."""
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
        result = AsyncSimpleStorageManager._select_by_positions(t, [0, 2, 4])
        # positions form constant stride of 2
        expected = torch.tensor([[1.0, 2.0], [5.0, 6.0], [9.0, 10.0]])
        assert torch.equal(result, expected)

    def test_regular_tensor_irregular_indices_fallback(self):
        """Case 3: Irregular indices fall back to index_select to avoid ZMQ frame fragmentation."""
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        # positions [0, 2, 3] have irregular gaps (2, then 1) - not constant stride
        result = AsyncSimpleStorageManager._select_by_positions(t, [0, 2, 3])
        expected = torch.tensor([[1.0, 2.0], [5.0, 6.0], [7.0, 8.0]])
        assert torch.equal(result, expected)

    def test_regular_tensor_irregular_reverse_order(self):
        """Irregular indices in reverse order also falls back to index_select."""
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        result = AsyncSimpleStorageManager._select_by_positions(t, [3, 1, 0])
        expected = torch.tensor([[7.0, 8.0], [3.0, 4.0], [1.0, 2.0]])
        assert torch.equal(result, expected)

    def test_nested_tensor_single_element(self):
        """Single element from nested tensor uses the lambda path."""
        t = torch.nested.as_nested_tensor(
            [torch.tensor([1.0]), torch.tensor([2.0, 3.0]), torch.tensor([4.0, 5.0, 6.0])],
            layout=torch.jagged,
        )
        result = AsyncSimpleStorageManager._select_by_positions(t, [1])
        assert isinstance(result, list)
        assert len(result) == 1
        assert torch.equal(result[0], torch.tensor([2.0, 3.0]))

    def test_empty_positions_raises_error(self):
        """Empty positions list should raise ValueError."""
        t = torch.tensor([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="No positions specified"):
            AsyncSimpleStorageManager._select_by_positions(t, [])

    def test_regular_tensor_negative_stride_rejected(self):
        """Negative stride (reversed order) should fall back to index_select."""
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        # positions [2, 1, 0] have step = -1 (negative)
        result = AsyncSimpleStorageManager._select_by_positions(t, [2, 1, 0])
        expected = torch.tensor([[5.0, 6.0], [3.0, 4.0], [1.0, 2.0]])
        assert torch.equal(result, expected)


class TestPackFieldValues:
    """Test _pack_field_values static method packing logic."""

    def test_uniform_tensors_to_stack(self):
        """Same-shape tensors → torch.stack."""
        values = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
        result = AsyncSimpleStorageManager._pack_field_values(values)  # type: ignore[attr-defined]
        assert isinstance(result, torch.Tensor)
        assert not result.is_nested
        assert result.shape == (2, 2)

    def test_variable_length_tensors_to_nested(self):
        """Different-shape tensors → nested tensor."""
        values = [torch.tensor([1.0]), torch.tensor([2.0, 3.0])]
        result = AsyncSimpleStorageManager._pack_field_values(values)  # type: ignore[attr-defined]
        assert isinstance(result, torch.Tensor)
        assert result.is_nested

    def test_non_tensors_to_nontensorstack(self):
        """Non-tensor values → NonTensorStack."""
        values = ["hello", "world"]
        result = AsyncSimpleStorageManager._pack_field_values(values)  # type: ignore[attr-defined]
        assert isinstance(result, NonTensorStack)
        assert result.tolist() == ["hello", "world"]

    def test_mixed_tensors_and_none_to_nontensorstack(self):
        """Mixed tensor + None values should stay as NonTensorStack (no stacking)."""
        t0 = torch.tensor([1.0, 2.0])
        t2 = torch.tensor([3.0, 4.0])
        values = [t0, None, t2]

        result = AsyncSimpleStorageManager._pack_field_values(values)  # type: ignore[attr-defined]

        assert isinstance(result, NonTensorStack)
        unpacked = result.tolist()
        assert len(unpacked) == 3
        assert torch.equal(unpacked[0], t0)
        assert unpacked[1] is None
        assert torch.equal(unpacked[2], t2)

    def test_all_none_to_nontensorstack(self):
        """All-None values should be preserved in NonTensorStack."""
        values = [None, None]

        result = AsyncSimpleStorageManager._pack_field_values(values)  # type: ignore[attr-defined]

        assert isinstance(result, NonTensorStack)
        assert result.tolist() == [None, None]
