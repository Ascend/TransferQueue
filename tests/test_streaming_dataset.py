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

import torch
from tensordict import TensorDict

from transfer_queue.dataloader.streaming_dataset import chunk_batch_fn
from transfer_queue.metadata import BatchMeta, extract_field_schema


def test_chunk_batch_fn_dense():
    td = TensorDict(
        {"a": torch.arange(6).reshape(3, 2), "b": torch.arange(3).reshape(3, 1)},
        batch_size=(3,),
    )
    meta = BatchMeta(global_indexes=[0, 1, 2], partition_ids=["p"] * 3)
    chunks = chunk_batch_fn(td, meta, micro_batch_size=2)
    assert len(chunks) == 2
    assert chunks[0][0].batch_size == (2,)
    assert chunks[1][0].batch_size == (1,)
    assert torch.equal(chunks[0][0]["a"], torch.tensor([[0, 1], [2, 3]]))
    assert torch.equal(chunks[0][0]["b"], torch.tensor([[0], [1]]))
    assert torch.equal(chunks[1][0]["a"], torch.tensor([[4, 5]]))
    assert torch.equal(chunks[1][0]["b"], torch.tensor([[2]]))


def test_chunk_batch_fn_mixed_dense_and_jagged():
    values = [torch.tensor([1, 2]), torch.tensor([3, 4, 5]), torch.tensor([6])]
    nested = torch.nested.as_nested_tensor(values, layout=torch.jagged)
    td = TensorDict({"a": nested, "b": torch.arange(3)}, batch_size=(3,))
    meta = BatchMeta(global_indexes=[0, 1, 2], partition_ids=["p"] * 3, field_schema=extract_field_schema(td))
    chunks = chunk_batch_fn(td, meta, micro_batch_size=2)
    assert len(chunks) == 2
    assert chunks[0][0].batch_size == (2,)
    assert chunks[1][0].batch_size == (1,)
    assert chunks[0][0]["a"].is_nested
    assert torch.equal(chunks[0][0]["a"][0], torch.tensor([1, 2]))
    assert torch.equal(chunks[0][0]["a"][1], torch.tensor([3, 4, 5]))
    assert torch.equal(chunks[0][0]["b"], torch.tensor([0, 1]))
    assert chunks[1][0]["a"].is_nested
    assert torch.equal(chunks[1][0]["a"][0], torch.tensor([6]))
    assert torch.equal(chunks[1][0]["b"], torch.tensor([2]))
    assert [chunk_meta.global_indexes for _, chunk_meta in chunks] == [[0, 1], [2]]
    for chunk_td, chunk_meta in chunks:
        assert chunk_meta.get_shapes("a") == [tuple(value.shape) for value in chunk_td["a"].unbind()]
        assert chunk_meta.get_shapes("b") == [(1,)] * len(chunk_meta)


def test_chunk_batch_fn_strided():
    values = [torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6], [7, 8]])]
    nested = torch.nested.as_nested_tensor(values, layout=torch.strided)
    td = TensorDict({"a": nested}, batch_size=(2,))
    meta = BatchMeta(global_indexes=[0, 1], partition_ids=["p"] * 2)
    chunks = chunk_batch_fn(td, meta, micro_batch_size=1)
    assert len(chunks) == 2
    assert chunks[0][0].batch_size == (1,)
    assert chunks[0][0]["a"].is_nested
    assert torch.equal(chunks[0][0]["a"][0], torch.tensor([[1, 2], [3, 4]]))
    assert chunks[1][0].batch_size == (1,)
    assert chunks[1][0]["a"].is_nested
    assert torch.equal(chunks[1][0]["a"][0], torch.tensor([[5, 6], [7, 8]]))
