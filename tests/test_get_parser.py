# Copyright 2026 The TransferQueue Team
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

import re

import numpy as np
import pytest
import torch
from tensordict import NonTensorData, NonTensorStack

from transfer_queue.utils.get_parser import (
    apply_get_parser,
    check_parser_select_fields,
    select_list_items,
    slice_tensor,
    validate_get_parser,
)


def test_select_list_items_keeps_requested_order():
    items = [torch.full((2,), float(i)) for i in range(4)]

    selected = select_list_items(items, [3, 0, 3])

    assert [item[0].item() for item in selected] == [3.0, 0.0, 3.0]


def test_select_list_items_empty_selection_returns_empty_list():
    assert select_list_items([1, 2], []) == []


@pytest.mark.parametrize("indices", [[2], [-1]])
def test_select_list_items_out_of_range_raises(indices):
    with pytest.raises(IndexError, match=re.escape(f"indices {indices} are out of range for a list of 2 items")):
        select_list_items([1, 2], indices)


def test_select_list_items_non_list_value_raises():
    with pytest.raises(TypeError, match="needs a list or tuple value, got Tensor"):
        select_list_items(torch.zeros(3), [0])


def test_slice_tensor_prefix_returns_contiguous_view():
    value = torch.arange(12).view(3, 4)

    sliced = slice_tensor(value, 2)

    assert torch.equal(sliced, value[:2])
    assert sliced.data_ptr() == value.data_ptr()
    assert sliced.untyped_storage().data_ptr() == value.untyped_storage().data_ptr()
    assert sliced.is_contiguous()


def test_slice_tensor_accepts_slice_and_tuple_indexes():
    value = torch.arange(24).view(4, 3, 2)

    assert torch.equal(slice_tensor(value, slice(1, 3)), value[1:3])
    assert torch.equal(slice_tensor(value, (slice(None), slice(0, 2))), value[:, :2])
    assert slice_tensor(value, (1, slice(None))).data_ptr() == value[1].data_ptr()


def test_slice_tensor_empty_prefix_is_valid_view():
    value = torch.arange(6).view(3, 2)

    sliced = slice_tensor(value, 0)

    assert sliced.shape == (0, 2)
    assert sliced.untyped_storage().data_ptr() == value.untyped_storage().data_ptr()


@pytest.mark.parametrize("index", [-1, 4])
def test_slice_tensor_prefix_out_of_range_raises(index):
    with pytest.raises(IndexError, match=re.escape(f"prefix length {index} is out of range for dim 0 of size 3")):
        slice_tensor(torch.zeros(3, 2), index)


def test_slice_tensor_rejects_non_tensor_and_nested():
    with pytest.raises(TypeError, match="needs a torch.Tensor value, got list"):
        slice_tensor([1, 2, 3], 1)
    nested = torch.nested.as_nested_tensor([torch.ones(2), torch.ones(3)], layout=torch.jagged)
    with pytest.raises(TypeError, match="does not support nested tensors"):
        slice_tensor(nested, 1)


def test_slice_tensor_rejects_bool_and_bad_index_type():
    value = torch.zeros(3)
    with pytest.raises(TypeError, match="not bool"):
        slice_tensor(value, True)
    with pytest.raises(TypeError, match="got str"):
        slice_tensor(value, "0:2")


def test_slice_tensor_rejects_index_list_unlike_select_list_items():
    """slice_tensor is not advanced indexing; a list of indices is select_list_items-only."""
    value = torch.arange(6).view(3, 2)
    with pytest.raises(TypeError, match="does not take a list of indices"):
        slice_tensor(value, [0, 2])


def test_slice_tensor_restores_indexing_tuple_after_zmq_roundtrip():
    """msgpack turns indexing tuples into lists; RPC parser_args must still slice multi-dim."""
    from transfer_queue.utils.zmq_utils import ZMQMessage, ZMQRequestType

    index = (slice(None), slice(0, 2))
    msg = ZMQMessage.create(
        request_type=ZMQRequestType.GET_DATA,
        sender_id="test",
        body={"parser_args": [index, (1, slice(None))]},
    )
    decoded_args = ZMQMessage.deserialize(msg.serialize()).body["parser_args"]
    assert all(isinstance(arg, list) for arg in decoded_args)

    value = torch.arange(24).view(4, 3, 2)
    parsed = apply_get_parser(
        {"x": [value, value]},
        slice_tensor,
        decoded_args,
        global_indexes=[0, 1],
    )
    assert torch.equal(parsed["x"][0], value[:, :2])
    assert torch.equal(parsed["x"][1], value[1])

    # Pure int lists stay rejected even after the same wire path.
    int_list_msg = ZMQMessage.create(
        request_type=ZMQRequestType.GET_DATA,
        sender_id="test",
        body={"parser_args": [[0, 2]]},
    )
    int_list_args = ZMQMessage.deserialize(int_list_msg.serialize()).body["parser_args"]
    with pytest.raises(ValueError, match="does not take a list of indices"):
        apply_get_parser({"x": [value]}, slice_tensor, int_list_args, global_indexes=[0])


def test_apply_get_parser_slice_tensor_keeps_per_key_views():
    field_data = {"input_ids": [torch.arange(4), torch.arange(10, 14)]}

    parsed = apply_get_parser(field_data, slice_tensor, [2, 3], global_indexes=[0, 1])

    assert torch.equal(parsed["input_ids"][0], torch.tensor([0, 1]))
    assert torch.equal(parsed["input_ids"][1], torch.tensor([10, 11, 12]))
    assert parsed["input_ids"][0].data_ptr() == field_data["input_ids"][0].data_ptr()
    assert parsed["input_ids"][1].data_ptr() == field_data["input_ids"][1].data_ptr()


def test_slice_tensor_prefix_encode_sends_only_sliced_nbytes():
    from transfer_queue.utils.serial_utils import MsgpackDecoder, MsgpackEncoder

    value = torch.arange(12, dtype=torch.int64).view(3, 4)
    sliced = slice_tensor(value, 2)
    assert sliced.is_contiguous()
    buffers = MsgpackEncoder().encode(sliced)

    assert buffers[1].nbytes == sliced.nbytes
    assert buffers[1].nbytes < value.nbytes
    assert torch.equal(MsgpackDecoder().decode(buffers), sliced)


def test_slice_tensor_noncontiguous_slice_encodes_correct_values():
    """Column / stride slices are correct; MsgpackEncoder may .contiguous() them first."""
    from transfer_queue.utils.serial_utils import MsgpackDecoder, MsgpackEncoder

    value = torch.arange(12, dtype=torch.int64).view(3, 4)
    for index in ((slice(None), slice(0, 2)), slice(None, None, 2)):
        sliced = slice_tensor(value, index)
        assert not sliced.is_contiguous()
        assert torch.equal(MsgpackDecoder().decode(MsgpackEncoder().encode(sliced)), sliced)


def test_check_parser_select_fields_rejects_multi_field_without_select_fields():
    with pytest.raises(ValueError, match="omits select_fields but the keys have multiple fields"):
        check_parser_select_fields(select_list_items, None, ["images", "input_ids"])


def test_check_parser_select_fields_allows_select_fields_or_single_field():
    check_parser_select_fields(select_list_items, "images", ["images", "input_ids"])
    check_parser_select_fields(select_list_items, ["images"], ["images", "input_ids"])
    check_parser_select_fields(select_list_items, None, ["images"])
    check_parser_select_fields(None, None, ["images", "input_ids"])


def test_validate_get_parser_rejects_args_without_parser():
    with pytest.raises(ValueError, match="parser_args was given without a parser"):
        validate_get_parser(None, [[0]], num_keys=1)


def test_validate_get_parser_rejects_non_callable():
    with pytest.raises(TypeError, match="parser must be callable, got str"):
        validate_get_parser("select_list_items", None, num_keys=1)


def test_validate_get_parser_rejects_misaligned_args():
    with pytest.raises(ValueError, match="parser_args has 1 entries but 2 keys are fetched"):
        validate_get_parser(select_list_items, [[0]], num_keys=2)


def test_validate_get_parser_keeps_plain_int_list_and_slice_args():
    ints = [2, 3]
    assert validate_get_parser(slice_tensor, ints, num_keys=2) is ints
    slices = [slice(0, 2), (slice(None), slice(0, 1))]
    assert validate_get_parser(slice_tensor, slices, num_keys=2) is slices
    nested = [[0, 2], [1]]
    assert validate_get_parser(select_list_items, nested, num_keys=2) is nested


@pytest.mark.parametrize(
    "parser_args",
    [
        torch.tensor(2),
        torch.tensor([2]),
        np.int64(2),
        np.array(2),
        np.array([2]),
    ],
)
def test_validate_get_parser_coerces_scalar_length_for_slice_tensor(parser_args):
    coerced = validate_get_parser(slice_tensor, [parser_args], num_keys=1)

    assert coerced == [2]
    assert all(type(x) is int for x in coerced)
    assert torch.equal(slice_tensor(torch.arange(5), coerced[0]), torch.arange(2))


@pytest.mark.parametrize(
    "parser_args",
    [
        torch.tensor([2, 3]),
        np.array([2, 3]),
        [torch.tensor(2), np.int64(3)],
        [np.array(2), torch.tensor([3])],
    ],
)
def test_validate_get_parser_coerces_1d_lengths_to_list_int(parser_args):
    coerced = validate_get_parser(slice_tensor, parser_args, num_keys=2)

    assert coerced == [2, 3]
    assert all(type(x) is int for x in coerced)
    field_data = {"input_ids": [torch.arange(4), torch.arange(10, 14)]}
    parsed = apply_get_parser(field_data, slice_tensor, coerced, global_indexes=[0, 1])
    assert torch.equal(parsed["input_ids"][0], torch.tensor([0, 1]))
    assert torch.equal(parsed["input_ids"][1], torch.tensor([10, 11, 12]))


def test_validate_get_parser_rejects_bad_tensor_shapes_early():
    with pytest.raises(ValueError, match="parser_args tensor/array must be 1-d with length 2"):
        validate_get_parser(slice_tensor, torch.tensor([[2, 3]]), num_keys=2)
    with pytest.raises(ValueError, match=r"parser_args\[0\] must be a 0-d or single-element integer tensor"):
        validate_get_parser(slice_tensor, [torch.zeros(2, 2, dtype=torch.int64)], num_keys=1)
    with pytest.raises(ValueError, match=r"parser_args\[0\] must be a 0-d or single-element integer tensor"):
        validate_get_parser(slice_tensor, [torch.tensor([[2]])], num_keys=1)
    with pytest.raises(TypeError, match=r"parser_args\[0\] must be an integer tensor"):
        validate_get_parser(slice_tensor, [torch.tensor(2.0)], num_keys=1)


def test_validate_get_parser_coerces_0d_tensor_as_whole_args_for_one_key():
    assert validate_get_parser(slice_tensor, torch.tensor(2), num_keys=1) == [2]
    assert validate_get_parser(slice_tensor, np.array(2), num_keys=1) == [2]


def test_validate_get_parser_does_not_coerce_select_list_items_args():
    """Coerce is slice_tensor-only; index tensors must stay Sequence-like, not bare int."""
    single = [torch.tensor([2])]
    assert validate_get_parser(select_list_items, single, num_keys=1) is single
    assert select_list_items(["a", "b", "c"], single[0]) == ["c"]

    for indices in (torch.tensor([0, 2]), np.array([0, 2])):
        args = [indices]
        assert validate_get_parser(select_list_items, args, num_keys=1) is args
        assert select_list_items(["a", "b", "c"], indices) == ["a", "c"]


def test_validate_get_parser_custom_callable_skips_length_coerce():
    def custom(value, parser_arg):
        return value

    args = [torch.tensor([2])]
    assert validate_get_parser(custom, args, num_keys=1) is args


def test_apply_get_parser_uses_each_key_arg_on_every_field():
    field_data = {"images": [["a0", "a1", "a2"], ["b0"]], "frames": [["fa0", "fa1", "fa2"], ["fb0", "fb1"]]}

    parsed = apply_get_parser(field_data, select_list_items, [[2, 0], [0]], global_indexes=[5, 9])

    assert parsed == {"images": [["a2", "a0"], ["b0"]], "frames": [["fa2", "fa0"], ["fb0"]]}


def test_apply_get_parser_unwraps_non_tensor_values():
    field_data = {"images": [NonTensorStack("x", "y"), NonTensorData(["z0", "z1"])]}

    parsed = apply_get_parser(field_data, select_list_items, [[1], [0, 1]], global_indexes=[0, 1])

    assert parsed == {"images": [["y"], ["z0", "z1"]]}


def test_apply_get_parser_without_args_passes_none():
    seen_args = []

    def record_arg(value, parser_arg):
        seen_args.append(parser_arg)
        return value

    apply_get_parser({"x": [1, 2]}, record_arg, None, global_indexes=[0, 1])

    assert seen_args == [None, None]


def test_apply_get_parser_error_names_field_index_and_arg():
    with pytest.raises(ValueError, match=r"failed on field 'images' of global index 9 with parser_arg \[3\]") as e:
        apply_get_parser({"images": [["a0"], ["b0"]]}, select_list_items, [[0], [3]], global_indexes=[5, 9])

    assert isinstance(e.value.__cause__, IndexError)
