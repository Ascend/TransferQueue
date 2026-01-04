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
from unittest.mock import patch

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
@pytest.mark.parametrize("enable_zero_copy", [True, False])
def test_tensor_serialization(dtype, enable_zero_copy):
    with patch("transfer_queue.utils.serial_utils.TQ_ZERO_COPY_SERIALIZATION", enable_zero_copy):
        encoder = MsgpackEncoder()
        decoder = MsgpackDecoder(torch.Tensor)

        tensor = torch.randn(100, 10, dtype=dtype)
        serialized = encoder.encode(tensor)
        deserialized = decoder.decode(serialized)
        assert torch.allclose(tensor, deserialized)
        assert deserialized.shape == tensor.shape
        assert isinstance(deserialized.shape, torch.Size)


@pytest.mark.parametrize("enable_zero_copy", [True, False])
def test_zmq_msg_serialization(enable_zero_copy):
    with patch("transfer_queue.utils.serial_utils.TQ_ZERO_COPY_SERIALIZATION", enable_zero_copy):
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
    decoder = MsgpackDecoder(torch.Tensor)

    base = torch.randn(16, 16, dtype=dtype)
    view = make_view(base)

    print("is_view_like:", view._base is not None, "is_contiguous:", view.is_contiguous())

    serialized = encoder.encode(view)
    deserialized = decoder.decode(serialized)

    assert deserialized.shape == view.shape
    assert deserialized.dtype == view.dtype
    assert torch.allclose(view, deserialized)


@pytest.mark.parametrize("enable_zero_copy", [True, False])
def test_tensordict_nested_serialization(enable_zero_copy):
    """Test serialization of deeply nested TensorDict structures."""
    with patch("transfer_queue.utils.serial_utils.TQ_ZERO_COPY_SERIALIZATION", enable_zero_copy):
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


@pytest.mark.parametrize("enable_zero_copy", [True, False])
def test_tensordict_with_mixed_batch_sizes(enable_zero_copy):
    """Test TensorDict with different batch size configurations."""
    with patch("transfer_queue.utils.serial_utils.TQ_ZERO_COPY_SERIALIZATION", enable_zero_copy):
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


@pytest.mark.parametrize("enable_zero_copy", [True, False])
def test_tensordict_empty_tensor(enable_zero_copy):
    """Test TensorDict handling of empty tensor."""
    with patch("transfer_queue.utils.serial_utils.TQ_ZERO_COPY_SERIALIZATION", enable_zero_copy):
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


@pytest.mark.parametrize("enable_zero_copy", [True, False])
def test_tensordict_with_various_tensor_layouts(enable_zero_copy):
    """Test TensorDict with various tensor layouts (strided, jagged, etc.)."""
    with patch("transfer_queue.utils.serial_utils.TQ_ZERO_COPY_SERIALIZATION", enable_zero_copy):
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


@pytest.mark.parametrize("enable_zero_copy", [True, False])
def test_tensordict_with_scalar_tensors(enable_zero_copy):
    """Test TensorDict containing scalar tensors."""
    with patch("transfer_queue.utils.serial_utils.TQ_ZERO_COPY_SERIALIZATION", enable_zero_copy):
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


@pytest.mark.parametrize("enable_zero_copy", [True, False])
def test_zero_copy_serialization_large_tensors(enable_zero_copy):
    """Test zero-copy serialization with large tensors."""
    with patch("transfer_queue.utils.serial_utils.TQ_ZERO_COPY_SERIALIZATION", enable_zero_copy):
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


@pytest.mark.parametrize("enable_zero_copy", [True, False])
def test_zero_copy_serialization_dtype_preservation(enable_zero_copy):
    """Test that zero-copy preserves all tensor dtypes."""
    with patch("transfer_queue.utils.serial_utils.TQ_ZERO_COPY_SERIALIZATION", enable_zero_copy):
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
    decoder = MsgpackDecoder(torch.Tensor)

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
    decoder = MsgpackDecoder(torch.Tensor)

    # Create non-contiguous tensor
    base = torch.randn(10, 10)
    non_contiguous = base[::2, ::2]

    serialized = encoder.encode(non_contiguous)
    deserialized = decoder.decode(serialized)

    assert deserialized.shape == non_contiguous.shape
    assert torch.allclose(non_contiguous, deserialized)


@pytest.mark.parametrize("batch_size", [0, 1, 100])
@pytest.mark.parametrize("enable_zero_copy", [True, False])
def test_tensordict_boundary_batch_sizes(batch_size, enable_zero_copy):
    """Test TensorDict with boundary batch sizes."""
    with patch("transfer_queue.utils.serial_utils.TQ_ZERO_COPY_SERIALIZATION", enable_zero_copy):
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
    decoder = MsgpackDecoder(torch.Tensor)

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


@pytest.mark.parametrize("enable_zero_copy", [True, False])
def test_nested_jagged_tensor_serialization(enable_zero_copy):
    """Test serialization of nested jagged tensors (challenging for zero-copy)."""
    with patch("transfer_queue.utils.serial_utils.TQ_ZERO_COPY_SERIALIZATION", enable_zero_copy):
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


@pytest.mark.parametrize("enable_zero_copy", [True, False])
def test_single_nested_tensor_serialization(enable_zero_copy):
    """Test serialization of nested tensor with only one element (edge case for zero-copy)."""
    with patch("transfer_queue.utils.serial_utils.TQ_ZERO_COPY_SERIALIZATION", enable_zero_copy):
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
