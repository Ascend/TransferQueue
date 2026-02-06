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
from threading import Thread
from unittest.mock import patch

import pytest
import torch
import zmq
from tensordict import NonTensorStack, TensorDict

# Import your classes here
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from transfer_queue import TransferQueueClient  # noqa: E402
from transfer_queue.metadata import (  # noqa: E402
    BatchMeta,
    FieldMeta,
    SampleMeta,
)
from transfer_queue.utils.enum_utils import TransferQueueRole  # noqa: E402
from transfer_queue.utils.zmq_utils import (  # noqa: E402
    ZMQMessage,
    ZMQRequestType,
    ZMQServerInfo,
)

TEST_DATA = TensorDict(
    {
        "log_probs": [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0, 6.0]), torch.tensor([7.0, 8.0, 9.0])],
        "variable_length_sequences": torch.nested.as_nested_tensor(
            [
                torch.tensor([-0.5, -1.2, -0.8]),
                torch.tensor([-0.3, -1.5, -2.1, -0.9]),
                torch.tensor([-1.1, -0.7]),
            ],
            layout=torch.jagged,
        ),
        "prompt_text": ["Hello world!", "This is a longer sentence for testing", "Test case"],
    },
    batch_size=[3],
)


# Mock Controller for Client Unit Testing
class MockController:
    def __init__(self, controller_id="controller_0"):
        self.controller_id = controller_id
        self.context = zmq.Context()

        # Socket for data requests
        self.request_socket = self.context.socket(zmq.ROUTER)
        self.request_port = self._bind_to_random_port(self.request_socket)

        self.zmq_server_info = ZMQServerInfo(
            role=TransferQueueRole.CONTROLLER,
            id=controller_id,
            ip="127.0.0.1",
            ports={
                "request_handle_socket": self.request_port,
            },
        )

        self.running = True
        self.request_thread = Thread(target=self._handle_requests, daemon=True)
        self.request_thread.start()

    def _bind_to_random_port(self, socket):
        port = socket.bind_to_random_port("tcp://127.0.0.1")
        return port

    def _handle_requests(self):
        poller = zmq.Poller()
        poller.register(self.request_socket, zmq.POLLIN)

        while self.running:
            try:
                socks = dict(poller.poll(100))  # 100ms timeout
                if self.request_socket in socks:
                    messages = self.request_socket.recv_multipart()
                    identity = messages.pop(0)
                    serialized_msg = messages
                    request_msg = ZMQMessage.deserialize(serialized_msg)

                    # Determine response based on request type
                    if request_msg.request_type == ZMQRequestType.GET_META:
                        response_body = self._mock_batch_meta(request_msg.body)
                        response_type = ZMQRequestType.GET_META_RESPONSE
                    elif request_msg.request_type == ZMQRequestType.CLEAR_META:
                        response_body = {"message": "clear meta ok"}
                        response_type = ZMQRequestType.CLEAR_META_RESPONSE
                    elif request_msg.request_type == ZMQRequestType.CLEAR_PARTITION:
                        response_body = {"message": "clear partition ok"}
                        response_type = ZMQRequestType.CLEAR_PARTITION_RESPONSE
                    elif request_msg.request_type == ZMQRequestType.GET_PARTITION_META:
                        # Mock partition metadata response
                        response_body = self._mock_batch_meta(request_msg.body)
                        response_type = ZMQRequestType.GET_PARTITION_META_RESPONSE
                    elif request_msg.request_type == ZMQRequestType.GET_CONSUMPTION:
                        # Mock consumption status check - all consumed
                        response_body = {
                            "partition_id": request_msg.body.get("partition_id"),
                            "global_index": torch.tensor([0, 1, 2]),
                            "consumption_status": torch.tensor([1, 1, 1]),
                        }
                        response_type = ZMQRequestType.CONSUMPTION_RESPONSE
                    elif request_msg.request_type == ZMQRequestType.GET_PRODUCTION:
                        # Mock production status check - all produced
                        response_body = {
                            "partition_id": request_msg.body.get("partition_id"),
                            "global_index": torch.tensor([0, 1, 2]),
                            "production_status": torch.tensor([[1, 1, 1], [1, 1, 1]]),
                        }
                        response_type = ZMQRequestType.PRODUCTION_RESPONSE
                    elif request_msg.request_type == ZMQRequestType.GET_LIST_PARTITIONS:
                        # Mock partition list
                        response_body = {
                            "partition_ids": ["partition_0", "partition_1", "test_partition"],
                        }
                        response_type = ZMQRequestType.LIST_PARTITIONS_RESPONSE
                    elif request_msg.request_type == ZMQRequestType.SET_CUSTOM_META:
                        response_body = {"message": "success"}
                        response_type = ZMQRequestType.SET_CUSTOM_META_RESPONSE
                    elif request_msg.request_type == ZMQRequestType.RESET_CONSUMPTION:
                        # Mock reset consumption - always succeed
                        response_body = {
                            "success": True,
                            "message": "Consumption reset successfully",
                        }
                        response_type = ZMQRequestType.RESET_CONSUMPTION_RESPONSE
                    else:
                        response_body = {"error": f"Unknown request type: {request_msg.request_type}"}
                        response_type = ZMQRequestType.CLEAR_META_RESPONSE

                    # Send response
                    response_msg = ZMQMessage.create(
                        request_type=response_type,
                        sender_id=self.controller_id,
                        receiver_id=request_msg.sender_id,
                        body=response_body,
                    )
                    self.request_socket.send_multipart([identity, *response_msg.serialize()])
            except zmq.Again:
                continue
            except Exception as e:
                print(f"MockController ERROR: {e}")
                raise

    def _mock_batch_meta(self, request_body):
        batch_size = request_body.get("batch_size", 1)
        data_fields = request_body.get("data_fields", [])

        samples = []
        for i in range(batch_size):
            fields = []
            for field_name in data_fields:
                field_meta = FieldMeta(
                    name=field_name,
                    dtype=None,
                    shape=None,
                    production_status=0,
                )
                fields.append(field_meta)
            sample = SampleMeta(
                partition_id="0",
                global_index=i,
                fields={field.name: field for field in fields},
            )
            samples.append(sample)
        metadata = BatchMeta(samples=samples)

        return {"metadata": metadata}

    def stop(self):
        self.running = False
        time.sleep(0.2)  # Give thread time to stop
        self.request_socket.close()
        self.context.term()


# Mock Storage for Client Unit Testing
class MockStorage:
    def __init__(self, storage_id="storage_0"):
        self.storage_id = storage_id
        self.context = zmq.Context()

        # Socket for data operations
        self.data_socket = self.context.socket(zmq.ROUTER)
        self.data_port = self._bind_to_random_port(self.data_socket)

        self.zmq_server_info = ZMQServerInfo(
            role=TransferQueueRole.STORAGE,
            id=storage_id,
            ip="127.0.0.1",
            ports={
                "put_get_socket": self.data_port,
            },
        )

        self.running = True
        self.data_thread = Thread(target=self._handle_data_requests, daemon=True)
        self.data_thread.start()

    def _bind_to_random_port(self, socket):
        port = socket.bind_to_random_port("tcp://127.0.0.1")
        return port

    def _handle_data_requests(self):
        poller = zmq.Poller()
        poller.register(self.data_socket, zmq.POLLIN)

        while self.running:
            try:
                socks = dict(poller.poll(100))  # 100ms timeout
                if self.data_socket in socks:
                    messages = self.data_socket.recv_multipart()
                    identity = messages.pop(0)
                    serialized_msg = messages
                    msg = ZMQMessage.deserialize(serialized_msg)

                    # Handle different request types
                    if msg.request_type == ZMQRequestType.PUT_DATA:
                        response_body = {"message": "Data stored successfully"}
                        response_type = ZMQRequestType.PUT_DATA_RESPONSE
                    elif msg.request_type == ZMQRequestType.GET_DATA:
                        response_body = self._handle_get_data(msg.body)
                        response_type = ZMQRequestType.GET_DATA_RESPONSE
                    elif msg.request_type == ZMQRequestType.CLEAR_DATA:
                        response_body = {"message": "Data cleared successfully"}
                        response_type = ZMQRequestType.CLEAR_DATA_RESPONSE

                    # Send response
                    response_msg = ZMQMessage.create(
                        request_type=response_type,
                        sender_id=self.storage_id,
                        receiver_id=msg.sender_id,
                        body=response_body,
                    )
                    self.data_socket.send_multipart([identity, *response_msg.serialize()])
            except zmq.Again:
                continue
            except Exception as e:
                if self.running:
                    print(f"MockStorage running exception: {e}")
                else:
                    print(f"MockStorage ERROR: {e}")
                    raise

    def _handle_get_data(self, request_body):
        """Handle GET_DATA request by retrieving stored data"""
        local_indexes = request_body.get("local_indexes", [])
        fields = request_body.get("fields", [])

        result: dict[str, list] = {}
        for field in fields:
            gathered_items = [TEST_DATA[field][i] for i in local_indexes]

            if gathered_items:
                all_tensors = all(isinstance(x, torch.Tensor) for x in gathered_items)
                if all_tensors:
                    result[field] = torch.nested.as_nested_tensor(gathered_items, layout=torch.jagged)
                else:
                    result[field] = NonTensorStack(*gathered_items)

        return {"data": TensorDict(result)}

    def stop(self):
        self.running = False
        time.sleep(0.2)  # Give thread time to stop
        self.data_socket.close()
        self.context.term()


# Test Fixtures
@pytest.fixture
def mock_controller():
    controller = MockController()
    yield controller
    controller.stop()


@pytest.fixture
def mock_storage():
    storage = MockStorage()
    yield storage
    storage.stop()


@pytest.fixture
def client_setup(mock_controller, mock_storage):
    # Create client with mock controller and storage
    client_id = "client_0"

    client = TransferQueueClient(
        client_id=client_id,
        controller_info=mock_controller.zmq_server_info,
    )

    # Mock the storage manager to avoid handshake issues but mock all data operations
    with patch(
        "transfer_queue.storage.managers.simple_backend_manager.AsyncSimpleStorageManager._connect_to_controller"
    ):
        config = {
            "controller_info": mock_controller.zmq_server_info,
            "zmq_info": {mock_storage.storage_id: mock_storage.zmq_server_info},
        }
        client.initialize_storage_manager(manager_type="SimpleStorage", config=config)

        # Mock all storage manager methods to avoid real ZMQ operations
        async def mock_put_data(data, metadata):
            pass  # Just pretend to store the data

        async def mock_get_data(metadata):
            # Return the test data when requested
            return TEST_DATA

        async def mock_clear_data(metadata):
            pass  # Just pretend to clear the data

        client.storage_manager.put_data = mock_put_data
        client.storage_manager.get_data = mock_get_data
        client.storage_manager.clear_data = mock_clear_data

    yield client, mock_controller, mock_storage


# Test basic functionality
def test_client_initialization(client_setup):
    """Test client initialization and connection setup"""
    client, mock_controller, mock_storage = client_setup

    assert client.client_id is not None
    assert client._controller is not None
    assert client._controller.id == mock_controller.controller_id


def test_put_and_get_data(client_setup):
    """Test basic put and get operations"""
    client, _, _ = client_setup

    # Test put operation
    client.put(data=TEST_DATA, partition_id="0")

    # Get metadata for retrieving data
    metadata = client.get_meta(
        data_fields=["log_probs", "variable_length_sequences", "prompt_text"], batch_size=2, partition_id="0"
    )

    # Test get operation
    result = client.get_data(metadata)

    # Verify result structure
    assert "log_probs" in result
    assert "variable_length_sequences" in result
    assert "prompt_text" in result

    torch.testing.assert_close(result["log_probs"][0], torch.tensor([1.0, 2.0, 3.0]))
    torch.testing.assert_close(result["log_probs"][1], torch.tensor([4.0, 5.0, 6.0]))
    torch.testing.assert_close(result["variable_length_sequences"][0], torch.tensor([-0.5, -1.2, -0.8]))
    torch.testing.assert_close(result["variable_length_sequences"][1], torch.tensor([-0.3, -1.5, -2.1, -0.9]))
    assert result["prompt_text"][0] == "Hello world!"
    assert result["prompt_text"][1] == "This is a longer sentence for testing"


def test_get_meta(client_setup):
    """Test metadata retrieval"""
    client, _, _ = client_setup

    # Test get_meta operation
    metadata = client.get_meta(data_fields=["tokens", "labels"], batch_size=10, partition_id="0")

    # Verify metadata structure
    assert hasattr(metadata, "global_indexes")
    assert hasattr(metadata, "field_names")
    assert hasattr(metadata, "size")
    assert len(metadata.global_indexes) == 10


# Test with single controller and multiple storage units
def test_single_controller_multiple_storages():
    """Test client with single controller and multiple storage units"""
    # Create single controller and multiple storage units
    controller = MockController("controller_0")
    storages = [MockStorage(f"storage_{i}") for i in range(3)]

    try:
        # Create client with single controller
        client_id = "client_test_single_controller"

        client = TransferQueueClient(client_id=client_id, controller_info=controller.zmq_server_info)

        # Mock the storage manager to avoid handshake issues but mock all data operations
        with patch(
            "transfer_queue.storage.managers.simple_backend_manager.AsyncSimpleStorageManager._connect_to_controller"
        ):
            config = {
                "controller_info": controller.zmq_server_info,
                "zmq_info": {s.storage_id: s.zmq_server_info for s in storages},
            }
            client.initialize_storage_manager(manager_type="SimpleStorage", config=config)

            # Mock all storage manager methods to avoid real ZMQ operations
            async def mock_put_data(data, metadata):
                pass  # Just pretend to store the data

            async def mock_get_data(metadata):
                # Return some test data when requested
                return TensorDict({"tokens": torch.randint(0, 100, (5, 128))}, batch_size=5)

            async def mock_clear_data(metadata):
                pass  # Just pretend to clear the data

            client.storage_manager.put_data = mock_put_data
            client.storage_manager.get_data = mock_get_data
            client.storage_manager.clear_data = mock_clear_data

        # Verify controller is set
        assert client._controller is not None
        assert client._controller.id == controller.controller_id

        # Test basic operation
        test_data = TensorDict({"tokens": torch.randint(0, 100, (5, 128))}, batch_size=5)

        # Test put operation
        client.put(data=test_data, partition_id="0")

    finally:
        # Clean up
        controller.stop()
        for s in storages:
            s.stop()


# Test error handling
def test_put_without_required_params(client_setup):
    """Test put operation without required parameters"""
    client, _, _ = client_setup

    # Create test data
    test_data = TensorDict({"tokens": torch.randint(0, 100, (5, 128))}, batch_size=5)

    # Test put without partition id (should fail)
    with pytest.raises(ValueError):
        client.put(data=test_data)


# Test new status checking methods
def test_check_consumption_status(client_setup):
    """Test consumption status checking"""
    client, _, _ = client_setup

    # Test synchronous check_consumption_status
    is_consumed = client.check_consumption_status(task_name="generate_sequences", partition_id="train_0")
    assert is_consumed is True


def test_check_production_status(client_setup):
    """Test production status checking"""
    client, _, _ = client_setup

    # Test synchronous check_production_status
    is_produced = client.check_production_status(data_fields=["prompt_ids", "attention_mask"], partition_id="train_0")
    assert is_produced is True


def test_get_consumption_status(client_setup):
    """Test get_consumption_status - returns global_index and consumption_status tensors"""
    client, _, _ = client_setup

    # Test synchronous get_consumption_status
    global_index, consumption_status = client.get_consumption_status(
        task_name="generate_sequences", partition_id="train_0"
    )

    # Verify return types
    assert global_index is not None
    assert consumption_status is not None

    # Verify global_index contains expected values
    assert torch.equal(global_index, torch.tensor([0, 1, 2], dtype=torch.long))

    # Verify consumption_status (mock returns all consumed)
    expected_status = torch.tensor([1, 1, 1], dtype=torch.int8)
    assert torch.equal(consumption_status, expected_status)

    print("✓ get_consumption_status returns correct global_index and consumption_status")


def test_get_production_status(client_setup):
    """Test get_production_status - returns global_index and production_status tensors"""
    client, _, _ = client_setup

    # Test synchronous get_production_status
    global_index, production_status = client.get_production_status(
        data_fields=["prompt_ids", "attention_mask"], partition_id="train_0"
    )

    # Verify return types
    assert global_index is not None
    assert production_status is not None

    # Verify global_index contains expected values
    assert torch.equal(global_index, torch.tensor([0, 1, 2], dtype=torch.long))

    # Verify production_status shape (mock returns 2x3 matrix)
    expected_status = torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.int8)
    assert torch.equal(production_status, expected_status)

    print("✓ get_production_status returns correct global_index and production_status")


def test_get_partition_list(client_setup):
    """Test partition list retrieval"""
    client, _, _ = client_setup

    # Test synchronous get_partition_list
    partition_list = client.get_partition_list()
    assert isinstance(partition_list, list)
    assert len(partition_list) > 0
    assert "partition_0" in partition_list
    assert "partition_1" in partition_list
    assert "test_partition" in partition_list


def test_reset_consumption(client_setup):
    """Test synchronous reset_consumption - resets consumption status for a partition"""
    client, _, _ = client_setup

    # Test synchronous reset_consumption with task_name
    success = client.reset_consumption(partition_id="train_0", task_name="generate_sequences")
    assert success is True

    print("✓ reset_consumption with task_name returns True")


def test_reset_consumption_all_tasks(client_setup):
    """Test synchronous reset_consumption without task_name (resets all tasks)"""
    client, _, _ = client_setup

    # Test synchronous reset_consumption without task_name (reset all tasks)
    success = client.reset_consumption(partition_id="train_0")
    assert success is True

    print("✓ reset_consumption without task_name (all tasks) returns True")


@pytest.mark.asyncio
async def test_async_reset_consumption(client_setup):
    """Test async reset_consumption - resets consumption status for a partition"""
    client, _, _ = client_setup

    # Test async_reset_consumption with task_name
    success = await client.async_reset_consumption(partition_id="train_0", task_name="generate_sequences")
    assert success is True

    print("✓ async_reset_consumption with task_name returns True")


@pytest.mark.asyncio
async def test_async_reset_consumption_all_tasks(client_setup):
    """Test async reset_consumption without task_name (resets all tasks)"""
    client, _, _ = client_setup

    # Test async_reset_consumption without task_name (reset all tasks)
    success = await client.async_reset_consumption(partition_id="train_0")
    assert success is True

    print("✓ async_reset_consumption without task_name (all tasks) returns True")


@pytest.mark.asyncio
async def test_async_check_consumption_status(client_setup):
    """Test async consumption status checking"""
    client, _, _ = client_setup

    # Test async_check_consumption_status
    is_consumed = await client.async_check_consumption_status(task_name="generate_sequences", partition_id="train_0")
    assert is_consumed is True


@pytest.mark.asyncio
async def test_async_check_production_status(client_setup):
    """Test async production status checking"""
    client, _, _ = client_setup

    # Test async_check_production_status
    is_produced = await client.async_check_production_status(
        data_fields=["prompt_ids", "attention_mask"], partition_id="train_0"
    )
    assert is_produced is True


@pytest.mark.asyncio
async def test_async_get_consumption_status(client_setup):
    """Test async get_consumption_status - returns global_index and consumption_status tensors"""
    client, _, _ = client_setup

    # Test async_get_consumption_status
    global_index, consumption_status = await client.async_get_consumption_status(
        task_name="generate_sequences", partition_id="train_0"
    )

    # Verify return types
    assert global_index is not None
    assert consumption_status is not None

    # Verify global_index contains expected values
    assert torch.equal(global_index, torch.tensor([0, 1, 2], dtype=torch.long))

    # Verify consumption_status (mock returns all consumed)
    expected_status = torch.tensor([1, 1, 1], dtype=torch.int8)
    assert torch.equal(consumption_status, expected_status)

    print("✓ async_get_consumption_status returns correct global_index and consumption_status")


@pytest.mark.asyncio
async def test_async_get_production_status(client_setup):
    """Test async get_production_status - returns global_index and production_status tensors"""
    client, _, _ = client_setup

    # Test async_get_production_status
    global_index, production_status = await client.async_get_production_status(
        data_fields=["prompt_ids", "attention_mask"], partition_id="train_0"
    )

    # Verify return types
    assert global_index is not None
    assert production_status is not None

    # Verify global_index contains expected values
    assert torch.equal(global_index, torch.tensor([0, 1, 2], dtype=torch.long))

    # Verify production_status shape (mock returns 2x3 matrix)
    expected_status = torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.int8)
    assert torch.equal(production_status, expected_status)

    print("✓ async_get_production_status returns correct global_index and production_status")


@pytest.mark.asyncio
async def test_async_get_partition_list(client_setup):
    """Test async partition list retrieval"""
    client, _, _ = client_setup

    # Test async_get_partition_list
    partition_list = await client.async_get_partition_list()
    assert isinstance(partition_list, list)
    assert len(partition_list) > 0
    assert "partition_0" in partition_list
    assert "partition_1" in partition_list
    assert "test_partition" in partition_list


# Test clear methods
@pytest.mark.asyncio
async def test_async_clear_partition(client_setup):
    """Test async clear partition operation"""
    client, _, _ = client_setup

    # Test async_clear_partition
    await client.async_clear_partition(partition_id="test_partition")

    # If no exception is raised, the test passes
    assert True


@pytest.mark.asyncio
async def test_async_clear_samples(client_setup):
    """Test async clear samples operation"""
    client, _, _ = client_setup

    # First get metadata to create a BatchMeta object
    metadata = await client.async_get_meta(data_fields=["tokens", "labels"], batch_size=2, partition_id="0")

    # Test async_clear_samples
    await client.async_clear_samples(metadata=metadata)

    # If no exception is raised, the test passes
    assert True


def test_clear_partition(client_setup):
    """Test synchronous clear partition operation"""
    client, _, _ = client_setup

    # Test synchronous clear_partition
    client.clear_partition(partition_id="test_partition")

    # If no exception is raised, the test passes
    assert True


def test_clear_samples(client_setup):
    """Test synchronous clear samples operation"""
    client, _, _ = client_setup

    # First get metadata to create a BatchMeta object
    metadata = client.get_meta(data_fields=["tokens", "labels"], batch_size=2, partition_id="0")

    # Test synchronous clear_samples
    client.clear_samples(metadata=metadata)

    # If no exception is raised, the test passes
    assert True


@pytest.mark.asyncio
async def test_async_clear_samples_with_empty_metadata(client_setup):
    """Test async_clear_samples with empty BatchMeta"""
    client, _, _ = client_setup

    # Create empty BatchMeta
    metadata = BatchMeta(samples=[])

    # The clear operation should complete without raising an exception
    # because the mock storage manager is configured to handle this
    await client.async_clear_samples(metadata=metadata)

    # If no exception is raised, the test passes
    assert True


@pytest.mark.asyncio
async def test_sync_methods_work_in_async_event_loop(client_setup):
    """Test all synchronous methods can be called from within an asyncio event loop.

    This test verifies that the sync methods can be called directly from an async
    function without causing "asyncio.run() cannot be called from a running loop" errors.
    """
    client, _, _ = client_setup

    test_data = TensorDict({"tokens": torch.randint(0, 100, (3, 64))}, batch_size=3)

    # Test sync put
    metadata = client.put(data=test_data, partition_id="0")
    assert metadata is not None

    # Test sync get_meta - use fields that mock returns
    metadata = client.get_meta(
        data_fields=["log_probs", "variable_length_sequences", "prompt_text"], batch_size=2, partition_id="0"
    )
    assert metadata is not None
    assert len(metadata.global_indexes) == 2

    # Test sync get_data - verify we get the expected fields from mock
    result = client.get_data(metadata)
    assert result is not None
    assert "log_probs" in result
    assert "prompt_text" in result

    # Test sync check_consumption_status
    is_consumed = client.check_consumption_status(task_name="generate_sequences", partition_id="train_0")
    assert isinstance(is_consumed, bool)

    # Test sync get_consumption_status
    global_index, consumption_status = client.get_consumption_status(
        task_name="generate_sequences", partition_id="train_0"
    )
    assert global_index is not None
    assert consumption_status is not None

    # Test sync check_production_status
    is_produced = client.check_production_status(data_fields=["log_probs", "prompt_text"], partition_id="train_0")
    assert isinstance(is_produced, bool)

    # Test sync get_production_status
    global_index, production_status = client.get_production_status(
        data_fields=["log_probs", "prompt_text"], partition_id="train_0"
    )
    assert global_index is not None
    assert production_status is not None

    # Test sync get_partition_list
    partition_list = client.get_partition_list()
    assert isinstance(partition_list, list)
    assert len(partition_list) > 0

    # Test sync clear_partition
    client.clear_partition(partition_id="test_partition")

    # Test sync clear_samples
    metadata = client.get_meta(data_fields=["log_probs", "prompt_text"], batch_size=2, partition_id="0")
    client.clear_samples(metadata=metadata)

    print("✓ All sync methods work correctly when called from within asyncio event loop")


@pytest.mark.asyncio
async def test_sync_and_async_methods_mixed_usage(client_setup):
    """Test mixing sync and async method calls within the same async context.

    This test verifies that async methods and sync methods can be used interchangeably
    without conflicts when called from an async function.
    """
    client, _, _ = client_setup

    test_data = TensorDict({"tokens": torch.randint(0, 100, (2, 32))}, batch_size=2)

    # Call sync method first
    sync_put_result = client.put(data=test_data, partition_id="0")
    assert sync_put_result is not None

    # Call async method
    async_metadata = await client.async_get_meta(data_fields=["tokens"], batch_size=2, partition_id="0")
    assert async_metadata is not None

    # Call sync method again
    sync_get_meta_result = client.get_meta(data_fields=["tokens"], batch_size=2, partition_id="0")
    assert sync_get_meta_result is not None

    # Call async method
    async_data = await client.async_get_data(sync_get_meta_result)
    assert async_data is not None

    print("✓ Mixed async and sync method calls work correctly")


# =====================================================
# Custom Meta Interface Tests
# =====================================================


class TestClientCustomMetaInterface:
    """Tests for client custom_meta interface methods."""

    def test_set_custom_meta_sync(self, client_setup):
        """Test synchronous set_custom_meta method."""
        client, _, _ = client_setup

        # Test synchronous set_custom_meta

        # First get metadata
        metadata = client.get_meta(data_fields=["input_ids"], batch_size=2, partition_id="0")
        # Set custom_meta on the metadata
        metadata.update_custom_meta(
            {
                0: {"input_ids": {"token_count": 100}},
                1: {"input_ids": {"token_count": 120}},
            }
        )

        # Call set_custom_meta with metadata (BatchMeta)
        client.set_custom_meta(metadata)
        print("✓ set_custom_meta sync method works")

    @pytest.mark.asyncio
    async def test_set_custom_meta_async(self, client_setup):
        """Test asynchronous async_set_custom_meta method."""
        client, _, _ = client_setup

        # First get metadata
        metadata = await client.async_get_meta(data_fields=["input_ids"], batch_size=2, partition_id="0")
        # Set custom_meta on the metadata
        metadata.update_custom_meta(
            {
                0: {"input_ids": {"token_count": 100}},
                1: {"input_ids": {"token_count": 120}},
            }
        )

        # Call async_set_custom_meta with metadata (BatchMeta)
        await client.async_set_custom_meta(metadata)
        print("✓ async_set_custom_meta async method works")
