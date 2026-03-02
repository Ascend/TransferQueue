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

import pytest
import ray
import tensordict
import torch
import zmq

# Setup path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from transfer_queue.storage.simple_backend import SimpleStorageUnit  # noqa: E402
from transfer_queue.utils.zmq_utils import ZMQMessage, ZMQRequestType  # noqa: E402


class MockStorageClient:
    """Mock client for testing storage unit operations."""

    def __init__(self, storage_put_get_address):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.DEALER)
        self.socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout
        self.socket.connect(storage_put_get_address)

    def send_put(self, client_id, global_indexes, field_data):
        msg = ZMQMessage.create(
            request_type=ZMQRequestType.PUT_DATA,
            sender_id=f"mock_client_{client_id}",
            body={"global_indexes": global_indexes, "data": field_data},
        )
        self.socket.send_multipart(msg.serialize())
        return ZMQMessage.deserialize(self.socket.recv_multipart())

    def send_get(self, client_id, global_indexes, fields):
        msg = ZMQMessage.create(
            request_type=ZMQRequestType.GET_DATA,
            sender_id=f"mock_client_{client_id}",
            body={"global_indexes": global_indexes, "fields": fields},
        )
        self.socket.send_multipart(msg.serialize())
        return ZMQMessage.deserialize(self.socket.recv_multipart())

    def send_clear(self, client_id, global_indexes):
        msg = ZMQMessage.create(
            request_type=ZMQRequestType.CLEAR_DATA,
            sender_id=f"mock_client_{client_id}",
            body={"global_indexes": global_indexes},
        )
        self.socket.send_multipart(msg.serialize())
        return ZMQMessage.deserialize(self.socket.recv_multipart())

    def close(self):
        self.socket.close()
        self.context.term()


@pytest.fixture(scope="session")
def ray_setup():
    """Initialize Ray for testing."""
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


@pytest.fixture
def storage_setup(ray_setup):
    """Set up storage unit for testing."""
    storage_size = 10000
    tensordict.set_list_to_stack(True).set()

    # Start Ray actor for SimpleStorageUnit
    storage_actor = SimpleStorageUnit.options(max_concurrency=50, num_cpus=1).remote(storage_unit_size=storage_size)

    # Get ZMQ server info from storage unit
    zmq_info = ray.get(storage_actor.get_zmq_server_info.remote())
    put_get_address = zmq_info.to_addr("put_get_socket")
    time.sleep(1)  # Wait for socket to be ready

    yield storage_actor, put_get_address

    # Cleanup
    ray.kill(storage_actor)


def test_put_get_single_client(storage_setup):
    """Test basic put and get operations with a single client."""
    _, put_get_address = storage_setup

    client = MockStorageClient(put_get_address)

    # PUT data
    local_indexes = [0, 1, 2]
    field_data = {
        "log_probs": [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0, 6.0]), torch.tensor([7.0, 8.0, 9.0])],
        "rewards": [torch.tensor([10.0]), torch.tensor([20.0]), torch.tensor([30.0])],
    }

    response = client.send_put(0, local_indexes, field_data)
    assert response.request_type == ZMQRequestType.PUT_DATA_RESPONSE

    # GET data
    response = client.send_get(0, [0, 1], ["log_probs", "rewards"])
    assert response.request_type == ZMQRequestType.GET_DATA_RESPONSE

    retrieved_data = response.body["data"]
    assert "log_probs" in retrieved_data
    assert "rewards" in retrieved_data
    assert len(retrieved_data["log_probs"]) == 2
    assert len(retrieved_data["rewards"]) == 2

    # Verify data correctness
    torch.testing.assert_close(retrieved_data["log_probs"][0], torch.tensor([1.0, 2.0, 3.0]))
    torch.testing.assert_close(retrieved_data["log_probs"][1], torch.tensor([4.0, 5.0, 6.0]))
    torch.testing.assert_close(retrieved_data["rewards"][0], torch.tensor([10.0]))
    torch.testing.assert_close(retrieved_data["rewards"][1], torch.tensor([20.0]))

    client.close()


def test_put_get_multiple_clients(storage_setup):
    """Test put and get operations with multiple clients."""
    _, put_get_address = storage_setup

    num_clients = 3
    clients = [MockStorageClient(put_get_address) for _ in range(num_clients)]

    # Each client puts unique data using different local_indexes
    for i, client in enumerate(clients):
        local_indexes = [i * 10 + 0, i * 10 + 1, i * 10 + 2]
        field_data = {
            "log_probs": [
                torch.tensor([i, i + 1, i + 2]),
                torch.tensor([i + 3, i + 4, i + 5]),
                torch.tensor([i + 6, i + 7, i + 8]),
            ],
            "rewards": [torch.tensor([i * 10]), torch.tensor([i * 10 + 10]), torch.tensor([i * 10 + 20])],
        }

        response = client.send_put(i, local_indexes, field_data)
        assert response.request_type == ZMQRequestType.PUT_DATA_RESPONSE

    # Test overlapping local indexes
    overlapping_client = MockStorageClient(put_get_address)
    overlap_local_indexes = [0]  # Overlaps with first client's index 0
    overlap_field_data = {"log_probs": [torch.tensor([999, 999, 999])], "rewards": [torch.tensor([999])]}
    response = overlapping_client.send_put(99, overlap_local_indexes, overlap_field_data)
    assert response.request_type == ZMQRequestType.PUT_DATA_RESPONSE

    # Each original client gets its own data (except for index 0 which was overwritten)
    for i, client in enumerate(clients):
        response = client.send_get(i, [i * 10 + 0, i * 10 + 1], ["log_probs", "rewards"])
        assert response.request_type == ZMQRequestType.GET_DATA_RESPONSE

        retrieved_data = response.body["data"]
        assert len(retrieved_data["log_probs"]) == 2
        assert len(retrieved_data["rewards"]) == 2

        # For index 0, expect data from overlapping_client; others from original client
        if i == 0:
            # Index 0 was overwritten
            torch.testing.assert_close(retrieved_data["log_probs"][0], torch.tensor([999, 999, 999]))
            torch.testing.assert_close(retrieved_data["rewards"][0], torch.tensor([999]))
            # Index 1 remains original
            torch.testing.assert_close(retrieved_data["log_probs"][1], torch.tensor([3, 4, 5]))
            torch.testing.assert_close(retrieved_data["rewards"][1], torch.tensor([10]))
        else:
            # All data remains original
            torch.testing.assert_close(retrieved_data["log_probs"][0], torch.tensor([i, i + 1, i + 2]))
            torch.testing.assert_close(retrieved_data["log_probs"][1], torch.tensor([i + 3, i + 4, i + 5]))
            torch.testing.assert_close(retrieved_data["rewards"][0], torch.tensor([i * 10]))
            torch.testing.assert_close(retrieved_data["rewards"][1], torch.tensor([i * 10 + 10]))

    # Cleanup
    for client in clients:
        client.close()
    overlapping_client.close()


def test_performance_basic(storage_setup):
    """Basic performance test with larger data volume."""
    _, put_get_address = storage_setup

    client = MockStorageClient(put_get_address)

    # PUT performance test
    put_latencies = []
    num_puts = 10  # Reduced for faster testing
    batch_size = 16  # Reduced for faster testing

    for i in range(num_puts):
        start = time.time()

        # Use batch size and index mapping
        local_indexes = list(range(i * batch_size, (i + 1) * batch_size))

        # Create tensor data
        log_probs_data = []
        rewards_data = []

        for _ in range(batch_size):
            # Smaller tensors for faster testing
            log_probs_tensor = torch.randn(100)
            rewards_tensor = torch.randn(100)
            log_probs_data.append(log_probs_tensor)
            rewards_data.append(rewards_tensor)

        field_data = {"log_probs": log_probs_data, "rewards": rewards_data}

        response = client.send_put(0, local_indexes, field_data)
        latency = time.time() - start
        put_latencies.append(latency)
        assert response.request_type == ZMQRequestType.PUT_DATA_RESPONSE

    # GET performance test
    get_latencies = []
    num_gets = 10

    for i in range(num_gets):
        start = time.time()
        # Retrieve batch of data
        local_indexes = list(range(i * batch_size, (i + 1) * batch_size))
        response = client.send_get(0, local_indexes, ["log_probs", "rewards"])
        latency = time.time() - start
        get_latencies.append(latency)
        assert response.request_type == ZMQRequestType.GET_DATA_RESPONSE

    avg_put_latency = sum(put_latencies) / len(put_latencies) * 1000  # ms
    avg_get_latency = sum(get_latencies) / len(get_latencies) * 1000  # ms

    # More lenient performance thresholds for testing environment
    assert avg_put_latency < 1500, f"Avg PUT latency {avg_put_latency}ms exceeds threshold"
    assert avg_get_latency < 1500, f"Avg GET latency {avg_get_latency}ms exceeds threshold"

    client.close()


def test_put_get_nested_tensor(storage_setup):
    """Test put and get operations with nested tensors."""
    _, put_get_address = storage_setup

    client = MockStorageClient(put_get_address)

    # PUT data with nested tensors
    local_indexes = [0, 1, 2]
    field_data = {
        "variable_length_sequences": [
            torch.tensor([-0.5, -1.2, -0.8]),
            torch.tensor([-0.3, -1.5, -2.1, -0.9]),
            torch.tensor([-1.1, -0.7]),
        ],
        "attention_mask": [torch.tensor([1, 1, 1]), torch.tensor([1, 1, 1, 1]), torch.tensor([1, 1])],
    }

    response = client.send_put(0, local_indexes, field_data)
    assert response.request_type == ZMQRequestType.PUT_DATA_RESPONSE

    # GET data
    response = client.send_get(0, [0, 2], ["variable_length_sequences", "attention_mask"])
    assert response.request_type == ZMQRequestType.GET_DATA_RESPONSE

    retrieved_data = response.body["data"]
    assert "variable_length_sequences" in retrieved_data
    assert "attention_mask" in retrieved_data
    assert len(retrieved_data["variable_length_sequences"]) == 2
    assert len(retrieved_data["attention_mask"]) == 2

    # Verify data correctness
    torch.testing.assert_close(retrieved_data["variable_length_sequences"][0], torch.tensor([-0.5, -1.2, -0.8]))
    torch.testing.assert_close(retrieved_data["variable_length_sequences"][1], torch.tensor([-1.1, -0.7]))
    torch.testing.assert_close(retrieved_data["attention_mask"][0], torch.tensor([1, 1, 1]))
    torch.testing.assert_close(retrieved_data["attention_mask"][1], torch.tensor([1, 1]))

    client.close()


def test_put_get_non_tensor_data(storage_setup):
    """Test put and get operations with non-tensor data (strings)."""
    _, put_get_address = storage_setup

    client = MockStorageClient(put_get_address)

    # PUT data with non-tensor data
    local_indexes = [0, 1, 2]
    field_data = {
        "prompt_text": ["Hello world!", "This is a longer sentence for testing", "Test case"],
        "response_text": ["Hi there!", "This is the response to the longer sentence", "Test response"],
    }

    response = client.send_put(0, local_indexes, field_data)
    assert response.request_type == ZMQRequestType.PUT_DATA_RESPONSE

    # GET data
    response = client.send_get(0, [0, 1, 2], ["prompt_text", "response_text"])
    assert response.request_type == ZMQRequestType.GET_DATA_RESPONSE

    retrieved_data = response.body["data"]
    assert "prompt_text" in retrieved_data
    assert "response_text" in retrieved_data

    # Verify data correctness
    assert isinstance(retrieved_data["prompt_text"][0], str)
    assert isinstance(retrieved_data["response_text"][0], str)

    assert retrieved_data["prompt_text"][0] == "Hello world!"
    assert retrieved_data["prompt_text"][1] == "This is a longer sentence for testing"
    assert retrieved_data["prompt_text"][2] == "Test case"
    assert retrieved_data["response_text"][0] == "Hi there!"
    assert retrieved_data["response_text"][1] == "This is the response to the longer sentence"
    assert retrieved_data["response_text"][2] == "Test response"

    client.close()


def test_put_get_single_item(storage_setup):
    """Test put and get operations for a single item."""
    _, put_get_address = storage_setup

    client = MockStorageClient(put_get_address)

    # PUT single item data
    field_data = {
        "prompt_text": ["Hello world!"],
        "attention_mask": [torch.tensor([1, 1, 1])],
    }
    response = client.send_put(0, [0], field_data)
    assert response.request_type == ZMQRequestType.PUT_DATA_RESPONSE

    # GET data
    response = client.send_get(0, [0], ["prompt_text", "attention_mask"])
    assert response.request_type == ZMQRequestType.GET_DATA_RESPONSE

    retrieved_data = response.body["data"]

    assert "prompt_text" in retrieved_data
    assert "attention_mask" in retrieved_data

    assert retrieved_data["prompt_text"][0] == "Hello world!"
    assert len(retrieved_data["attention_mask"]) == 1
    torch.testing.assert_close(retrieved_data["attention_mask"][0], torch.tensor([1, 1, 1]))

    client.close()


def test_clear_data(storage_setup):
    """Test clear operations."""
    _, put_get_address = storage_setup

    client = MockStorageClient(put_get_address)

    # PUT data first
    local_indexes = [0, 1, 2]
    field_data = {
        "log_probs": [torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])],
        "rewards": [torch.tensor([10.0]), torch.tensor([20.0]), torch.tensor([30.0])],
    }

    response = client.send_put(0, local_indexes, field_data)
    assert response.request_type == ZMQRequestType.PUT_DATA_RESPONSE

    # Verify data exists
    response = client.send_get(0, [0, 1, 2], ["log_probs"])
    assert response.request_type == ZMQRequestType.GET_DATA_RESPONSE
    assert len(response.body["data"]["log_probs"]) == 3

    # Clear data
    response = client.send_clear(0, [0, 2])  # Clear only indexes 0 and 2
    assert response.request_type == ZMQRequestType.CLEAR_DATA_RESPONSE

    # Verify some data is cleared (but index 1 should still exist)
    response = client.send_get(0, [1], ["log_probs"])
    assert response.request_type == ZMQRequestType.GET_DATA_RESPONSE
    assert len(response.body["data"]["log_probs"]) == 1
    torch.testing.assert_close(response.body["data"]["log_probs"][0], torch.tensor([2.0]))

    client.close()


def test_storage_unit_data_direct():
    """Test StorageUnitData class directly without ZMQ."""
    from transfer_queue.storage import StorageUnitData

    storage_data = StorageUnitData(storage_size=10)

    field_data = {
        "log_probs": [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])],
        "rewards": [torch.tensor([10.0]), torch.tensor([20.0])],
    }
    # global_indexes = gi values (e.g., 0 and 1)
    storage_data.put_data(field_data, [0, 1])

    result = storage_data.get_data(["log_probs", "rewards"], [0, 1])
    assert "log_probs" in result
    assert "rewards" in result
    assert len(result["log_probs"]) == 2
    assert len(result["rewards"]) == 2

    result_single = storage_data.get_data(["log_probs"], [0])
    torch.testing.assert_close(result_single["log_probs"][0], torch.tensor([1.0, 2.0]))

    # clear: key is removed (not set to None)
    storage_data.clear([0])
    assert 0 not in storage_data.field_data["log_probs"]  # key gone
    assert 1 in storage_data.field_data["log_probs"]  # other key intact


def test_storage_unit_data_dict_key():
    """StorageUnitData dict-key: gi 直接作为 key，clear 真正释放内存."""
    from transfer_queue.storage.simple_backend import StorageUnitData

    storage = StorageUnitData(storage_size=4)

    # put_data: 用 gi 列表 [10, 11] 作为 global_indexes
    storage.put_data(
        {"f": [torch.tensor([1.0]), torch.tensor([2.0])]},
        global_indexes=[10, 11],
    )
    assert len(storage.field_data["f"]) == 2

    # get_data: 通过 gi 读取
    result = storage.get_data(["f"], global_indexes=[10, 11])
    torch.testing.assert_close(result["f"][0], torch.tensor([1.0]))
    torch.testing.assert_close(result["f"][1], torch.tensor([2.0]))

    # clear: 真正删除 key，不是置 None
    storage.clear(keys=[10])
    assert 10 not in storage.field_data["f"]
    assert 11 in storage.field_data["f"]

    # capacity check: storage_size=4，已有 1 条，再放 4 条应失败
    with pytest.raises(ValueError, match="Storage capacity exceeded"):
        storage.put_data(
            {"f": [torch.tensor([i * 1.0]) for i in range(4)]},
            global_indexes=[20, 21, 22, 23],
        )


def test_storage_unit_data_partial_consume_safety():
    """部分消费后写入复用 gi，不应覆盖未消费数据."""
    from transfer_queue.storage.simple_backend import StorageUnitData

    storage = StorageUnitData(storage_size=4)
    storage.put_data({"f": [torch.tensor([0.0]), torch.tensor([1.0])]}, global_indexes=[0, 1])

    storage.clear(keys=[1])  # 只清除 gi=1
    assert 0 in storage.field_data["f"]
    assert 1 not in storage.field_data["f"]

    # 复用 gi=1 写入新数据，不影响 gi=0
    storage.put_data({"f": [torch.tensor([9.0])]}, global_indexes=[1])
    torch.testing.assert_close(storage.field_data["f"][0], torch.tensor([0.0]))
    torch.testing.assert_close(storage.field_data["f"][1], torch.tensor([9.0]))
