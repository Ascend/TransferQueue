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

import pytest
import torch
from tensordict import NonTensorData, NonTensorStack

from transfer_queue.utils.get_parser import apply_get_parser, select_list_items, slice_tensor, validate_get_parser


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
    with pytest.raises(TypeError, match="got list"):
        slice_tensor(value, [0, 1])


def test_apply_get_parser_slice_tensor_keeps_per_key_views():
    field_data = {"input_ids": [torch.arange(4), torch.arange(10, 14)]}

    parsed = apply_get_parser(field_data, slice_tensor, [2, 3], global_indexes=[0, 1])

    assert torch.equal(parsed["input_ids"][0], torch.tensor([0, 1]))
    assert torch.equal(parsed["input_ids"][1], torch.tensor([10, 11, 12]))
    assert parsed["input_ids"][0].data_ptr() == field_data["input_ids"][0].data_ptr()
    assert parsed["input_ids"][1].data_ptr() == field_data["input_ids"][1].data_ptr()


def test_slice_tensor_encode_sends_only_sliced_bytes_without_copy():
    from transfer_queue.utils.serial_utils import MsgpackDecoder, MsgpackEncoder

    value = torch.arange(12, dtype=torch.int64).view(3, 4)
    sliced = slice_tensor(value, 2)
    buffers = MsgpackEncoder().encode(sliced)

    assert buffers[1].nbytes == sliced.nbytes
    assert buffers[1].nbytes < value.nbytes
    assert torch.equal(MsgpackDecoder().decode(buffers), sliced)


def test_validate_get_parser_rejects_args_without_parser():
    with pytest.raises(ValueError, match="parser_args was given without a parser"):
        validate_get_parser(None, [[0]], num_keys=1)


def test_validate_get_parser_rejects_non_callable():
    with pytest.raises(TypeError, match="parser must be callable, got str"):
        validate_get_parser("select_list_items", None, num_keys=1)


def test_validate_get_parser_rejects_misaligned_args():
    with pytest.raises(ValueError, match="parser_args has 1 entries but 2 keys are fetched"):
        validate_get_parser(select_list_items, [[0]], num_keys=2)


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
