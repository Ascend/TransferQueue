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

import numpy as np
import pytest
import torch
from tensordict import TensorDict

# Import your classes here
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from transfer_queue.utils.serial_utils import MsgpackDecoder, MsgpackEncoder  # noqa: E402


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ],
)
def test_tensor_serialization(dtype):
    encoder = MsgpackEncoder()
    decoder = MsgpackDecoder()

    tensor = torch.randn(100, 10, dtype=dtype)
    serialized = encoder.encode(tensor)
    deserialized = decoder.decode(serialized)
    assert torch.allclose(tensor, deserialized)
    assert deserialized.shape == tensor.shape
    assert isinstance(deserialized.shape, torch.Size)


def test_zmq_msg_serialization():
    from transfer_queue.utils.zmq_utils import ZMQMessage, ZMQRequestType

    # construct complex msg body with nested tensor, jagged tensor, normal tensor, numpy array
    msg = ZMQMessage(
        request_type=ZMQRequestType.PUT_DATA,
        sender_id="test_sender",
        receiver_id="test_receiver",
        request_id="test_request",
        timestamp="test_timestamp",
        body={
            "data": TensorDict(
                {
                    "nested_tensor": torch.nested.as_nested_tensor(
                        [torch.randn(4, 3), torch.randn(2, 4)], layout=torch.strided
                    ),
                    "jagged_tensor": torch.nested.as_nested_tensor(
                        [torch.randn(4, 5), torch.randn(4, 54)], layout=torch.jagged
                    ),
                    "normal_tensor": torch.randn(2, 10, 3),
                    "numpy_array": torch.randn(2, 2).numpy(),
                },
                batch_size=2,
            )
        },
    )
    encoded_msg = msg.serialize()
    decoded_msg = ZMQMessage.deserialize(encoded_msg)
    assert decoded_msg.request_type == msg.request_type
    # TensorDict converts numpy arrays to Tensors on insertion,
    # so decoding yields a Tensor (not np.ndarray).
    assert torch.allclose(decoded_msg.body["data"]["numpy_array"], msg.body["data"]["numpy_array"])
    assert torch.allclose(decoded_msg.body["data"]["normal_tensor"], msg.body["data"]["normal_tensor"])
    assert msg.body["data"]["nested_tensor"].layout == decoded_msg.body["data"]["nested_tensor"].layout
    assert msg.body["data"]["jagged_tensor"].layout == decoded_msg.body["data"]["jagged_tensor"].layout
    for i in range(len(msg.body["data"]["nested_tensor"].unbind())):
        assert torch.allclose(
            decoded_msg.body["data"]["nested_tensor"][i],
            msg.body["data"]["nested_tensor"][i],
        )
    for i in range(len(msg.body["data"]["jagged_tensor"].unbind())):
        assert torch.allclose(
            decoded_msg.body["data"]["jagged_tensor"][i],
            msg.body["data"]["jagged_tensor"][i],
        )


@pytest.mark.parametrize(
    "make_view",
    [
        lambda x: x[:, :5],
        lambda x: x[::2],
        lambda x: x[..., 1:],
        lambda x: x.transpose(0, 1),
        lambda x: x[1:-1, 2:8:2],
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ],
)
def test_tensor_serialization_with_views(dtype, make_view):
    encoder = MsgpackEncoder()
    decoder = MsgpackDecoder()

    base = torch.randn(16, 16, dtype=dtype)
    view = make_view(base)

    print("is_view_like:", view._base is not None, "is_contiguous:", view.is_contiguous())

    serialized = encoder.encode(view)
    deserialized = decoder.decode(serialized)

    assert deserialized.shape == view.shape
    assert deserialized.dtype == view.dtype
    assert torch.allclose(view, deserialized)


def test_tensordict_nested_serialization():
    """Test serialization of deeply nested TensorDict structures."""
    from transfer_queue.utils.zmq_utils import ZMQMessage, ZMQRequestType

    # Create nested TensorDict - all tensors must match batch_size
    inner_td = TensorDict(
        {"level3_tensor": torch.randn(2, 3), "level3_data": torch.tensor([1, 2, 3]).expand(2, -1)}, batch_size=2
    )

    middle_td = TensorDict({"level2_inner": inner_td, "level2_tensor": torch.randn(2, 4, 5)}, batch_size=2)

    outer_td = TensorDict(
        {
            "level1_middle": middle_td,
            "level1_tensor": torch.randn(2, 10),
        },
        batch_size=2,
    )

    msg = ZMQMessage(
        request_type=ZMQRequestType.PUT_DATA,
        sender_id="test",
        receiver_id="test",
        request_id="test",
        timestamp=0.0,
        body={"data": outer_td},
    )

    encoded_msg = msg.serialize()
    decoded_msg = ZMQMessage.deserialize(encoded_msg)

    assert decoded_msg.body["data"].batch_size == outer_td.batch_size
    assert torch.allclose(decoded_msg.body["data"]["level1_tensor"], outer_td["level1_tensor"])
    assert (
        decoded_msg.body["data"]["level1_middle"]["level2_tensor"].shape
        == outer_td["level1_middle"]["level2_tensor"].shape
    )
    assert torch.allclose(
        decoded_msg.body["data"]["level1_middle"]["level2_inner"]["level3_tensor"],
        outer_td["level1_middle"]["level2_inner"]["level3_tensor"],
    )


def test_tensordict_with_mixed_batch_sizes():
    """Test TensorDict with different batch size configurations."""
    from transfer_queue.utils.zmq_utils import ZMQMessage, ZMQRequestType

    # Test with various batch sizes
    for batch_size in [1, 5, 10, 32]:
        td = TensorDict(
            {
                "data": torch.randn(batch_size, 10),
                "labels": torch.randint(0, 100, (batch_size,)),
                "metadata": torch.randn(batch_size, 5),
            },
            batch_size=batch_size,
        )

        msg = ZMQMessage(
            request_type=ZMQRequestType.PUT_DATA,
            sender_id="test",
            receiver_id="test",
            request_id="test",
            timestamp=0.0,
            body={"data": td},
        )

        encoded_msg = msg.serialize()
        decoded_msg = ZMQMessage.deserialize(encoded_msg)

        assert decoded_msg.body["data"].batch_size == td.batch_size
        assert torch.allclose(decoded_msg.body["data"]["data"], td["data"])
        assert torch.equal(decoded_msg.body["data"]["labels"], td["labels"])


def test_tensordict_empty_tensor():
    """Test TensorDict handling of empty tensor."""
    from transfer_queue.utils.zmq_utils import ZMQMessage, ZMQRequestType

    # Create TensorDict with some empty/zero fields
    td = TensorDict(
        {
            "normal_tensor": torch.randn(3, 5),
            "empty_tensor": torch.empty(3, 0),
            "zeros_tensor": torch.zeros(3, 10),
        },
        batch_size=3,
    )

    msg = ZMQMessage(
        request_type=ZMQRequestType.PUT_DATA,
        sender_id="test",
        receiver_id="test",
        request_id="test",
        timestamp=0.0,
        body={"data": td},
    )

    encoded_msg = msg.serialize()
    decoded_msg = ZMQMessage.deserialize(encoded_msg)

    assert decoded_msg.body["data"].batch_size == td.batch_size
    assert decoded_msg.body["data"]["empty_tensor"].shape == td["empty_tensor"].shape
    assert torch.allclose(decoded_msg.body["data"]["zeros_tensor"], td["zeros_tensor"])


def test_tensordict_with_various_tensor_layouts():
    """Test TensorDict with various tensor layouts (strided, jagged, etc.)."""
    from transfer_queue.utils.zmq_utils import ZMQMessage, ZMQRequestType

    # Create TensorDict with different layouts
    td = TensorDict(
        {
            "strided": torch.randn(2, 5, 3),
            "jagged": torch.nested.as_nested_tensor([torch.randn(3, 4), torch.randn(2, 4)], layout=torch.jagged),
            "nested": torch.nested.as_nested_tensor([torch.randn(4, 3), torch.randn(2, 4)], layout=torch.strided),
        },
        batch_size=2,
    )

    msg = ZMQMessage(
        request_type=ZMQRequestType.PUT_DATA,
        sender_id="test",
        receiver_id="test",
        request_id="test",
        timestamp=0.0,
        body={"data": td},
    )

    encoded_msg = msg.serialize()
    decoded_msg = ZMQMessage.deserialize(encoded_msg)

    assert decoded_msg.body["data"].batch_size == td.batch_size
    assert decoded_msg.body["data"]["strided"].shape == td["strided"].shape
    assert decoded_msg.body["data"]["jagged"].layout == td["jagged"].layout
    assert decoded_msg.body["data"]["nested"].layout == td["nested"].layout


def test_tensordict_with_scalar_tensors():
    """Test TensorDict containing scalar tensors."""
    from transfer_queue.utils.zmq_utils import ZMQMessage, ZMQRequestType

    td = TensorDict(
        {
            "scalar_float": torch.tensor(3.14).expand(5, 1),
            "scalar_int": torch.tensor(42).expand(5, 1),
            "vector": torch.randn(5, 1),
        },
        batch_size=5,
    )

    msg = ZMQMessage(
        request_type=ZMQRequestType.PUT_DATA,
        sender_id="test",
        receiver_id="test",
        request_id="test",
        timestamp=0.0,
        body={"data": td},
    )

    encoded_msg = msg.serialize()
    decoded_msg = ZMQMessage.deserialize(encoded_msg)

    assert decoded_msg.body["data"].batch_size == td.batch_size
    assert decoded_msg.body["data"]["scalar_float"].shape == td["scalar_float"].shape
    assert decoded_msg.body["data"]["scalar_int"].shape == td["scalar_int"].shape


def test_zero_copy_serialization_large_tensors():
    """Test zero-copy serialization with large tensors."""
    from transfer_queue.utils.zmq_utils import ZMQMessage, ZMQRequestType

    # Create large tensors - jagged tensor has 3 items, so batch_size should be 3
    # But we can't mix jagged with regular tensors in the same TensorDict with different batch sizes
    # So let's test them separately
    td = TensorDict(
        {
            "large_tensor": torch.randn(3, 100, 200),
        },
        batch_size=3,
    )

    msg = ZMQMessage(
        request_type=ZMQRequestType.PUT_DATA,
        sender_id="test",
        receiver_id="test",
        request_id="test",
        timestamp=0.0,
        body={"data": td},
    )

    encoded_msg = msg.serialize()
    decoded_msg = ZMQMessage.deserialize(encoded_msg)

    assert decoded_msg.body["data"].batch_size == td.batch_size
    assert decoded_msg.body["data"]["large_tensor"].shape == td["large_tensor"].shape

    # Also test jagged tensor separately
    td_jagged = TensorDict(
        {
            "large_jagged": torch.nested.as_nested_tensor(
                [torch.randn(50, 100), torch.randn(30, 100), torch.randn(40, 100)], layout=torch.jagged
            ),
        },
        batch_size=3,
    )

    msg_jagged = ZMQMessage(
        request_type=ZMQRequestType.PUT_DATA,
        sender_id="test",
        receiver_id="test",
        request_id="test",
        timestamp=0.0,
        body={"data": td_jagged},
    )

    encoded_msg_jagged = msg_jagged.serialize()
    decoded_msg_jagged = ZMQMessage.deserialize(encoded_msg_jagged)

    assert decoded_msg_jagged.body["data"].batch_size == td_jagged.batch_size


def test_zero_copy_serialization_dtype_preservation():
    """Test that zero-copy preserves all tensor dtypes."""
    from transfer_queue.utils.zmq_utils import ZMQMessage, ZMQRequestType

    # Use only float dtypes for randn, use appropriate functions for other types
    dtypes = [torch.float16, torch.float32, torch.float64]

    td_dict = {}
    for i, dtype in enumerate(dtypes):
        key = f"tensor_{str(dtype).replace('torch.', '')}"
        td_dict[key] = torch.randn(2, 3, dtype=dtype)

    # Add integer types using appropriate initializers
    td_dict["tensor_int8"] = torch.randint(-128, 127, (2, 3), dtype=torch.int8)
    td_dict["tensor_int16"] = torch.randint(-32768, 32767, (2, 3), dtype=torch.int16)
    td_dict["tensor_int32"] = torch.randint(-1000, 1000, (2, 3), dtype=torch.int32)
    td_dict["tensor_int64"] = torch.randint(-1000, 1000, (2, 3), dtype=torch.int64)
    td_dict["tensor_bool"] = torch.randint(0, 2, (2, 3), dtype=torch.bool)

    dtypes_all = list(dtypes) + [torch.int8, torch.int16, torch.int32, torch.int64, torch.bool]

    td = TensorDict(td_dict, batch_size=2)

    msg = ZMQMessage(
        request_type=ZMQRequestType.PUT_DATA,
        sender_id="test",
        receiver_id="test",
        request_id="test",
        timestamp=0.0,
        body={"data": td},
    )

    encoded_msg = msg.serialize()
    decoded_msg = ZMQMessage.deserialize(encoded_msg)

    for dtype in dtypes_all:
        key = f"tensor_{str(dtype).replace('torch.', '')}"
        assert decoded_msg.body["data"][key].dtype == td[key].dtype


# ============================================================================
# Edge Case and Error Handling Tests
# ============================================================================
def test_serialization_with_extreme_shapes():
    """Test serialization with extreme tensor shapes."""
    encoder = MsgpackEncoder()
    decoder = MsgpackDecoder()

    # Very thin tensors
    thin_tensor = torch.randn(1000, 1)
    serialized = encoder.encode(thin_tensor)
    deserialized = decoder.decode(serialized)
    assert torch.allclose(thin_tensor, deserialized)

    # Very wide tensors
    wide_tensor = torch.randn(1, 1000)
    serialized = encoder.encode(wide_tensor)
    deserialized = decoder.decode(serialized)
    assert torch.allclose(wide_tensor, deserialized)


def test_serialization_memory_contiguity():
    """Test that serialized tensors maintain proper memory layout."""
    encoder = MsgpackEncoder()
    decoder = MsgpackDecoder()

    # Create non-contiguous tensor
    base = torch.randn(10, 10)
    non_contiguous = base[::2, ::2]

    serialized = encoder.encode(non_contiguous)
    deserialized = decoder.decode(serialized)

    assert deserialized.shape == non_contiguous.shape
    assert torch.allclose(non_contiguous, deserialized)


@pytest.mark.parametrize("batch_size", [0, 1, 100])
def test_tensordict_boundary_batch_sizes(batch_size):
    """Test TensorDict with boundary batch sizes."""
    from transfer_queue.utils.zmq_utils import ZMQMessage, ZMQRequestType

    if batch_size == 0:
        # Empty TensorDict
        td = TensorDict({}, batch_size=0)
        msg = ZMQMessage(
            request_type=ZMQRequestType.PUT_DATA,
            sender_id="test",
            receiver_id="test",
            request_id="test",
            timestamp=0.0,
            body={"data": td},
        )
        encoded_msg = msg.serialize()
        decoded_msg = ZMQMessage.deserialize(encoded_msg)
        assert decoded_msg.body["data"].batch_size == torch.Size([0])
    else:
        td = TensorDict({"data": torch.randn(batch_size, 5)}, batch_size=batch_size)

        msg = ZMQMessage(
            request_type=ZMQRequestType.PUT_DATA,
            sender_id="test",
            receiver_id="test",
            request_id="test",
            timestamp=0.0,
            body={"data": td},
        )
        encoded_msg = msg.serialize()
        decoded_msg = ZMQMessage.deserialize(encoded_msg)

        assert decoded_msg.body["data"].batch_size == td.batch_size
        assert torch.allclose(decoded_msg.body["data"]["data"], td["data"])


def test_serialization_with_special_values():
    """Test serialization with special float values."""
    encoder = MsgpackEncoder()
    decoder = MsgpackDecoder()

    # Test with special values
    special_tensor = torch.tensor([[float("inf"), float("-inf"), float("nan")], [0.0, -0.0, 1e-10]])

    serialized = encoder.encode(special_tensor)
    deserialized = decoder.decode(serialized)

    # Check regular values
    assert torch.allclose(deserialized[1, :], special_tensor[1, :])
    # Check NaN (can't use allclose for NaN)
    assert torch.isnan(deserialized[0, 2]) and torch.isnan(special_tensor[0, 2])
    # Check infinities
    assert torch.isinf(deserialized[0, 0]) and deserialized[0, 0] > 0
    assert torch.isinf(deserialized[0, 1]) and deserialized[0, 1] < 0


def test_nested_jagged_tensor_serialization():
    """Test serialization of nested jagged tensors (challenging for zero-copy)."""
    from transfer_queue.utils.zmq_utils import ZMQMessage, ZMQRequestType

    # Create nested jagged structure
    inner_jagged1 = torch.nested.as_nested_tensor([torch.randn(3, 5), torch.randn(2, 5)], layout=torch.jagged)
    inner_jagged2 = torch.nested.as_nested_tensor([torch.randn(4, 5), torch.randn(1, 5)], layout=torch.jagged)

    outer_td = TensorDict(
        {
            "nested_jagged1": inner_jagged1,
            "nested_jagged2": inner_jagged2,
            "normal_tensor": torch.randn(2, 10),
        },
        batch_size=2,
    )

    msg = ZMQMessage(
        request_type=ZMQRequestType.PUT_DATA,
        sender_id="test",
        receiver_id="test",
        request_id="test",
        timestamp=0.0,
        body={"data": outer_td},
    )

    encoded_msg = msg.serialize()
    decoded_msg = ZMQMessage.deserialize(encoded_msg)

    assert decoded_msg.body["data"].batch_size == outer_td.batch_size
    assert decoded_msg.body["data"]["nested_jagged1"].layout == torch.jagged
    assert decoded_msg.body["data"]["nested_jagged2"].layout == torch.jagged

    # Verify individual components
    for i in range(len(outer_td["nested_jagged1"].unbind())):
        assert torch.allclose(decoded_msg.body["data"]["nested_jagged1"][i], outer_td["nested_jagged1"][i])


def test_single_nested_tensor_serialization():
    """Test serialization of nested tensor with only one element (edge case for zero-copy)."""
    from transfer_queue.utils.zmq_utils import ZMQMessage, ZMQRequestType

    # Create nested tensor with only one element
    # This is the critical edge case where a nested tensor with 1 element
    # must be distinguished from a regular tensor during deserialization
    single_nested = torch.nested.as_nested_tensor([torch.randn(4, 3)], layout=torch.strided)
    # For normal tensor, expand to batch_size=1 to match the nested tensor's batch dimension
    normal_tensor = torch.randn(1, 4, 3)

    # Create TensorDict with both types
    td = TensorDict(
        {
            "single_nested_tensor": single_nested,
            "normal_tensor": normal_tensor,
        },
        batch_size=1,
    )

    msg = ZMQMessage(
        request_type=ZMQRequestType.PUT_DATA,
        sender_id="test",
        receiver_id="test",
        request_id="test",
        timestamp=0.0,
        body={"data": td},
    )

    encoded_msg = msg.serialize()
    decoded_msg = ZMQMessage.deserialize(encoded_msg)

    # Verify batch sizes
    assert decoded_msg.body["data"].batch_size == td.batch_size

    # Verify normal tensor
    assert torch.allclose(decoded_msg.body["data"]["normal_tensor"], td["normal_tensor"])
    assert decoded_msg.body["data"]["normal_tensor"].shape == td["normal_tensor"].shape

    # Verify single nested tensor is properly reconstructed as nested
    assert decoded_msg.body["data"]["single_nested_tensor"].is_nested
    assert decoded_msg.body["data"]["single_nested_tensor"].layout == torch.strided
    assert len(decoded_msg.body["data"]["single_nested_tensor"].unbind()) == 1
    assert torch.allclose(decoded_msg.body["data"]["single_nested_tensor"][0], td["single_nested_tensor"][0])

    # Ensure the nested tensor with single element is correctly distinguished from regular tensor
    # Both should have the same data but different types
    assert not decoded_msg.body["data"]["normal_tensor"].is_nested
    assert decoded_msg.body["data"]["single_nested_tensor"].is_nested


def test_large_string_serialization():
    """Test serialization of large strings (>10KB).

    Note: msgpack natively handles str type, so enc_hook is not called for strings.
    This test verifies large strings are correctly serialized/deserialized.
    """
    encoder = MsgpackEncoder()
    decoder = MsgpackDecoder()

    # Create a string larger than 10KB
    large_string = "x" * 11000  # ~11KB

    serialized = encoder.encode({"text": large_string})

    # Verify content is correctly restored
    decoded = decoder.decode(serialized)
    assert decoded["text"] == large_string
    assert len(decoded["text"]) == len(large_string)


def test_large_string_in_zmq_message():
    """Test large string in ZMQMessage body."""
    from transfer_queue.utils.zmq_utils import ZMQMessage, ZMQRequestType

    large_text = "Hello World! " * 1000  # ~13KB

    msg = ZMQMessage(
        request_type=ZMQRequestType.PUT_DATA,
        sender_id="test",
        receiver_id="test",
        request_id="test",
        timestamp=0.0,
        body={
            "large_text": large_text,
            "tensor": torch.randn(10, 10),  # Combined with tensor
        },
    )

    encoded_msg = msg.serialize()
    decoded_msg = ZMQMessage.deserialize(encoded_msg)

    assert decoded_msg.body["large_text"] == large_text
    assert torch.allclose(decoded_msg.body["tensor"], msg.body["tensor"])


def test_non_ascii_large_string():
    """Test large string with non-ASCII characters (UTF-8 handling)."""
    encoder = MsgpackEncoder()
    decoder = MsgpackDecoder()

    # Create large string with various UTF-8 characters
    unicode_chars = "你好世界🌍🚀 émojis and ümläuts "
    large_unicode_string = unicode_chars * 500  # ~12KB

    serialized = encoder.encode({"unicode_text": large_unicode_string})
    decoded = decoder.decode(serialized)

    assert decoded["unicode_text"] == large_unicode_string


# ============================================================================
# Thread Safety Tests (ContextVar-based isolation)
# ============================================================================
class TestSerialThreadSafety:
    """Test thread safety of MsgpackEncoder/MsgpackDecoder with ContextVar.

    These tests verify that the ContextVar-based fix properly isolates
    aux_buffers across multiple threads, preventing buffer/metadata mismatch
    errors that previously occurred when multiple threads used the global
    _encoder/_decoder instances concurrently.

    Historical issue: Before the fix, aux_buffers was stored as instance
    variable, causing race conditions where int8 tensor buffers could be
    associated with long tensor metadata, resulting in:
    "self.size(-1) must be divisible by 8 to view Byte as Long"
    """

    @staticmethod
    def _create_test_message(thread_id: int, iteration: int) -> dict:
        """Create test message simulating GET_CONSUMPTION response structure.

        Uses different dtypes and varying sizes to maximize the chance of
        detecting buffer/metadata mismatches under concurrent access.
        """
        num_samples = 30 + (iteration % 10)
        # torch.long: 8 bytes per element
        global_index = torch.arange(num_samples, dtype=torch.long)
        # torch.int8: 1 byte per element
        consumption_status = torch.zeros(num_samples + iteration % 5, dtype=torch.int8)

        return {
            "request_type": "CONSUMPTION_RESPONSE",
            "sender_id": f"controller_{thread_id}",
            "receiver_id": f"client_{thread_id}",
            "request_id": f"req_{thread_id}_{iteration}",
            "body": {
                "partition_id": f"partition_{thread_id}",
                "global_index": global_index,
                "consumption_status": consumption_status,
            },
        }

    def test_global_encoder_thread_safety(self):
        """Test that global _encoder/_decoder instances are thread-safe.

        This test verifies the ContextVar-based fix by using the global
        shared encoder/decoder instances across multiple threads with
        concurrent serialize/deserialize operations.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from transfer_queue.utils.serial_utils import _decoder, _encoder

        num_threads = 8
        iterations_per_thread = 50
        errors: list[str] = []
        success_count = 0

        def worker(thread_id: int) -> tuple[int, list[str]]:
            """Worker function that uses global encoder/decoder."""
            local_success = 0
            local_errors: list[str] = []

            for i in range(iterations_per_thread):
                try:
                    msg = self._create_test_message(thread_id, i)

                    # Use global shared encoder (thread-safe with ContextVar)
                    serialized = list(_encoder.encode(msg))

                    # Use global shared decoder
                    deserialized = _decoder.decode(serialized)

                    # Verify data correctness
                    original_global_index = msg["body"]["global_index"]
                    decoded_global_index = deserialized["body"]["global_index"]

                    if not torch.equal(original_global_index, decoded_global_index):
                        raise ValueError(
                            f"Data mismatch! Original shape: {original_global_index.shape}, "
                            f"Decoded shape: {decoded_global_index.shape}"
                        )

                    original_status = msg["body"]["consumption_status"]
                    decoded_status = deserialized["body"]["consumption_status"]
                    if not torch.equal(original_status, decoded_status):
                        raise ValueError(
                            f"consumption_status mismatch! Original: {original_status.shape}, "
                            f"Decoded: {decoded_status.shape}"
                        )

                    local_success += 1

                except Exception as e:
                    local_errors.append(f"Thread {thread_id}, Iter {i}: {type(e).__name__}: {e}")

            return local_success, local_errors

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {executor.submit(worker, tid): tid for tid in range(num_threads)}

            for future in as_completed(futures):
                s, e = future.result()
                success_count += s
                errors.extend(e)

        # All operations should succeed with the ContextVar fix
        total_ops = num_threads * iterations_per_thread
        assert success_count == total_ops, (
            f"Thread safety test failed: {len(errors)} errors out of {total_ops} operations.\n"
            f"Sample errors: {errors[:5]}"
        )

    def test_mixed_dtype_concurrent_serialization(self):
        """Test concurrent serialization of tensors with different dtypes.

        This test specifically targets the historical bug where buffer index
        mismatches occurred between int8 and int64 tensors, causing view errors.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from transfer_queue.utils.serial_utils import _decoder, _encoder

        num_threads = 16
        iterations = 30

        # Use different dtypes with different byte sizes to maximize
        # the chance of triggering buffer/metadata mismatches
        dtype_configs = [
            (torch.int8, (50,)),  # 1 byte per element
            (torch.long, (50,)),  # 8 bytes per element
            (torch.float16, (50, 10)),  # 2 bytes per element
            (torch.float32, (50, 10)),  # 4 bytes per element
            (torch.bfloat16, (50, 10)),  # 2 bytes per element
        ]

        def worker(thread_id: int) -> tuple[int, list[str]]:
            local_success = 0
            local_errors: list[str] = []

            for i in range(iterations):
                try:
                    # Select dtype configuration based on thread_id and iteration
                    dtype, shape = dtype_configs[(thread_id + i) % len(dtype_configs)]

                    if dtype in (torch.int8, torch.long):
                        tensor = torch.randint(-128, 127, shape, dtype=dtype)
                    else:
                        tensor = torch.randn(*shape, dtype=dtype)

                    msg = {
                        "thread_id": thread_id,
                        "iteration": i,
                        "tensor": tensor,
                        "nested": {"inner_tensor": torch.randn(10, dtype=torch.float32)},
                    }

                    serialized = list(_encoder.encode(msg))
                    deserialized = _decoder.decode(serialized)

                    # Verify tensor correctness
                    if not torch.equal(deserialized["tensor"], tensor):
                        raise ValueError(f"Tensor mismatch for {dtype}")

                    if not torch.allclose(deserialized["nested"]["inner_tensor"], msg["nested"]["inner_tensor"]):
                        raise ValueError("Nested tensor mismatch")

                    local_success += 1

                except Exception as e:
                    local_errors.append(f"Thread {thread_id}, Iter {i}: {type(e).__name__}: {e}")

            return local_success, local_errors

        errors: list[str] = []
        success_count = 0

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {executor.submit(worker, tid): tid for tid in range(num_threads)}
            for future in as_completed(futures):
                s, e = future.result()
                success_count += s
                errors.extend(e)

        total_ops = num_threads * iterations
        assert success_count == total_ops, (
            f"Mixed dtype test failed: {len(errors)} errors out of {total_ops}.\nSample errors: {errors[:5]}"
        )


# ============================================================================
# Numpy Array Type Compatibility Tests
# ============================================================================
class TestNumpyArrayTypeCompatibility:
    """Test numpy array serialization with various dtypes.

    These tests verify the fix for the TypeError when using torch.from_numpy()
    with unsupported numpy dtypes (e.g., object arrays). The fix uses pickle
    fallback for incompatible types while maintaining zero-copy for numeric types.
    """

    def test_numpy_object_array_strings(self):
        """Test numpy object array with string elements."""
        import numpy as np

        encoder = MsgpackEncoder()
        decoder = MsgpackDecoder()

        # String array (dtype=object or unicode)
        str_arr = np.array(["hello", "world", "test"])

        serialized = encoder.encode(str_arr)
        deserialized = decoder.decode(serialized)

        assert np.array_equal(deserialized, str_arr)
        assert deserialized.dtype == str_arr.dtype

    def test_numpy_object_array_mixed_types(self):
        """Test numpy object array with mixed Python types."""
        import numpy as np

        encoder = MsgpackEncoder()
        decoder = MsgpackDecoder()

        # Mixed type array (explicitly object dtype)
        mixed_arr = np.array([1, "two", 3.0, None], dtype=object)

        serialized = encoder.encode(mixed_arr)
        deserialized = decoder.decode(serialized)

        assert np.array_equal(deserialized, mixed_arr)
        assert deserialized.dtype == np.object_

    def test_numpy_object_array_dicts(self):
        """Test numpy object array containing Python dicts."""
        import numpy as np

        encoder = MsgpackEncoder()
        decoder = MsgpackDecoder()

        # Array of dicts
        dict_arr = np.array([{"a": 1}, {"b": 2}, {"c": 3}], dtype=object)

        serialized = encoder.encode(dict_arr)
        deserialized = decoder.decode(serialized)

        assert len(deserialized) == len(dict_arr)
        for orig, decoded in zip(dict_arr, deserialized, strict=False):
            assert orig == decoded

    def test_numpy_numeric_arrays_zero_copy(self):
        """Test that numeric numpy arrays use zero-copy path and return np.ndarray."""
        import numpy as np

        encoder = MsgpackEncoder()
        decoder = MsgpackDecoder()

        numeric_dtypes = [
            np.float32,
            np.float64,
            np.int32,
            np.int64,
            np.int8,
            np.uint8,
            np.bool_,
        ]

        for dtype in numeric_dtypes:
            if dtype == np.bool_:
                arr = np.array([True, False, True], dtype=dtype)
            elif np.issubdtype(dtype, np.integer):
                arr = np.array([1, 2, 3], dtype=dtype)
            else:
                arr = np.array([1.0, 2.0, 3.0], dtype=dtype)

            serialized = encoder.encode(arr)

            # Zero-copy must produce multiple buffers (metadata + data buffer)
            assert len(serialized) > 1, f"Expected zero-copy for dtype {dtype}"

            deserialized = decoder.decode(serialized)

            # After the fix: deserialized must be np.ndarray, not torch.Tensor
            assert isinstance(deserialized, np.ndarray), (
                f"Expected np.ndarray but got {type(deserialized)} for dtype={dtype}"
            )
            assert deserialized.dtype == arr.dtype
            assert np.array_equal(deserialized, arr)

    def test_numpy_object_array_in_zmq_message(self):
        """Test numpy object array inside ZMQMessage."""
        import numpy as np

        from transfer_queue.utils.zmq_utils import ZMQMessage, ZMQRequestType

        # Create message with both object array and regular tensors
        obj_arr = np.array(["prompt_1", "prompt_2", "prompt_3"], dtype=object)

        msg = ZMQMessage(
            request_type=ZMQRequestType.PUT_DATA,
            sender_id="test",
            receiver_id="test",
            request_id="test",
            timestamp=0.0,
            body={
                "prompts": obj_arr,
                "tensor_data": torch.randn(3, 10),
            },
        )

        encoded_msg = msg.serialize()
        decoded_msg = ZMQMessage.deserialize(encoded_msg)

        # Verify object array
        assert np.array_equal(decoded_msg.body["prompts"], obj_arr)

        # Verify tensor (should work with zero-copy)
        assert torch.allclose(decoded_msg.body["tensor_data"], msg.body["tensor_data"])

    def test_numpy_unicode_string_array(self):
        """Test numpy unicode string array (dtype='<U...')."""
        import numpy as np

        encoder = MsgpackEncoder()
        decoder = MsgpackDecoder()

        # Unicode string array with Chinese characters
        unicode_arr = np.array(["你好", "世界", "测试"])

        serialized = encoder.encode(unicode_arr)
        deserialized = decoder.decode(serialized)

        assert np.array_equal(deserialized, unicode_arr)

    def test_numpy_bytes_array(self):
        """Test numpy bytes array (dtype='S...')."""
        import numpy as np

        encoder = MsgpackEncoder()
        decoder = MsgpackDecoder()

        # Bytes array
        bytes_arr = np.array([b"hello", b"world"], dtype="S10")

        serialized = encoder.encode(bytes_arr)
        deserialized = decoder.decode(serialized)

        assert np.array_equal(deserialized, bytes_arr)


# ============================================================================
# Numpy Native Serialization Tests (CUSTOM_TYPE_NUMPY)
# ============================================================================
class TestNumpyNativeSerialization:
    """Test that numpy arrays are serialized/deserialized natively.

    After the fix, numeric numpy arrays must:
    1. Round-trip as np.ndarray (not torch.Tensor).
    2. Preserve dtype and shape exactly.
    3. Use zero-copy (len(serialized) > 1).
    4. Produce correct values.
    """

    @pytest.mark.parametrize(
        "dtype",
        [
            # Numeric / bool / complex (original coverage)
            np.float16,
            np.float32,
            np.float64,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
            np.bool_,
            np.complex64,
            np.complex128,
            # Extended types now also covered via exclusion-based check
            np.datetime64,  # kind='M', stored as int64
            np.timedelta64,  # kind='m', stored as int64
            np.dtype("S10"),  # kind='S', fixed-length bytes
        ],
    )
    def test_numpy_roundtrip_preserves_type(self, dtype):
        """All buffer-compatible ndarrays must come back as np.ndarray, not torch.Tensor."""
        encoder = MsgpackEncoder()
        decoder = MsgpackDecoder()

        dtype = np.dtype(dtype)  # normalise in case a dtype instance was passed
        if dtype == np.dtype("bool"):
            arr = np.array([True, False, True, True], dtype=dtype)
        elif dtype.kind == "c":  # complex
            arr = np.array([1 + 2j, 3 + 4j], dtype=dtype)
        elif dtype.kind == "M":  # datetime64
            arr = np.array(["2024-01", "2024-02"], dtype=dtype)
        elif dtype.kind == "m":  # timedelta64
            arr = np.array([1, 2], dtype=dtype)
        elif dtype.kind == "S":  # fixed-length bytes
            arr = np.array([b"hello", b"world"], dtype=dtype)
        elif np.issubdtype(dtype, np.integer):
            arr = np.array([1, 2, 3, 4], dtype=dtype)
        else:
            arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=dtype)

        serialized = encoder.encode(arr)
        deserialized = decoder.decode(serialized)

        assert isinstance(deserialized, np.ndarray), f"Expected np.ndarray, got {type(deserialized)} for dtype={dtype}"
        assert deserialized.dtype == arr.dtype
        assert deserialized.shape == arr.shape
        assert np.array_equal(deserialized, arr)

    def test_numpy_zero_copy_uses_multiple_buffers(self):
        """Zero-copy path must produce len(serialized) > 1."""
        encoder = MsgpackEncoder()
        arr = np.arange(100, dtype=np.float32)
        serialized = encoder.encode(arr)
        assert len(serialized) > 1, "Expected zero-copy (aux buffer) for float32 ndarray"

    def test_numpy_non_contiguous_roundtrip(self):
        """Non-C-contiguous arrays must be made contiguous before serialization."""
        encoder = MsgpackEncoder()
        decoder = MsgpackDecoder()

        base = np.arange(100, dtype=np.float64).reshape(10, 10)
        arr = base[::2, ::2]  # non-contiguous view
        assert not arr.flags["C_CONTIGUOUS"]

        serialized = encoder.encode(arr)
        deserialized = decoder.decode(serialized)

        assert isinstance(deserialized, np.ndarray)
        assert np.array_equal(deserialized, arr)

    def test_numpy_multidim_shape_preserved(self):
        """Shape must survive a round-trip for multi-dimensional arrays."""
        encoder = MsgpackEncoder()
        decoder = MsgpackDecoder()

        arr = np.arange(60, dtype=np.int32).reshape(3, 4, 5)
        serialized = encoder.encode(arr)
        deserialized = decoder.decode(serialized)

        assert isinstance(deserialized, np.ndarray)
        assert deserialized.shape == (3, 4, 5)
        assert np.array_equal(deserialized, arr)

    def test_numpy_empty_array_roundtrip(self):
        """Empty arrays must round-trip correctly."""
        encoder = MsgpackEncoder()
        decoder = MsgpackDecoder()

        arr = np.empty((0,), dtype=np.float32)
        serialized = encoder.encode(arr)
        deserialized = decoder.decode(serialized)

        assert isinstance(deserialized, np.ndarray)
        assert deserialized.shape == (0,)
        assert deserialized.dtype == np.float32

    def test_numpy_object_array_still_uses_pickle(self):
        """Object arrays (kind='O' or hasobject) must fall back to pickle."""
        encoder = MsgpackEncoder()
        decoder = MsgpackDecoder()

        # dtype=object — kind 'O', cannot be viewed as a contiguous byte buffer
        arr = np.array(["a", "b", "c"], dtype=object)
        serialized = encoder.encode(arr)

        # Pickle-fallback produces a single buffer (no aux tensor buffer appended)
        assert len(serialized) == 1, "Object array should not use zero-copy path"

        deserialized = decoder.decode(serialized)
        assert isinstance(deserialized, np.ndarray)
        assert np.array_equal(deserialized, arr)
