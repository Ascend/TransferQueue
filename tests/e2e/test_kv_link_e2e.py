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

"""End-to-end tests for kv_link: n keys reading one prompt payload without copies."""

import shutil

import pytest
import ray
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

import transfer_queue as tq

PARTITION = "link"
N = 3


@pytest.fixture(scope="module")
def tq_system():
    if not ray.is_initialized():
        ray.init(namespace="TestKVLinkE2E")
    tq.init(
        OmegaConf.create(
            {
                "controller": {"polling_mode": True},
                "backend": {
                    "storage_backend": "SimpleStorage",
                    "SimpleStorage": {"total_storage_size": 200, "num_data_storage_units": 2},
                },
            }
        )
    )
    yield
    tq.close()
    ray.shutdown()


@pytest.fixture(autouse=True)
def fresh_partition(tq_system):
    yield
    tq.get_client().clear_partition(PARTITION)


def _prompt(p: int) -> torch.Tensor:
    return torch.tensor([p, p + 1, p + 2])


def _put_group(p: int) -> list[str]:
    """Put prompt ``p`` once and link its N responses to it; return the response keys."""
    tq.kv_put(key=f"p{p}", partition_id=PARTITION, fields={"prompt": _prompt(p)})
    keys = [f"p{p}_r{i}" for i in range(N)]
    tq.kv_link(keys, PARTITION, f"p{p}", "prompt")
    return keys


def _links(key: str) -> dict:
    return tq.kv_describe(key, PARTITION)[key]["links"]


def _rows(field: torch.Tensor) -> list[list]:
    """A batched field comes back as a nested tensor, which has no ``tolist``."""
    return [row.tolist() for row in field.unbind()]


def test_each_partition_spreads_evenly_over_storage_units(tq_system):
    """Two storage units hold global_idx % 2; alternating puts used to send each partition to one."""
    other = f"{PARTITION}_other"
    for i in range(4):
        tq.kv_put(key=f"a{i}", partition_id=PARTITION, fields={"prompt": _prompt(i)})
        tq.kv_put(key=f"b{i}", partition_id=other, fields={"prompt": _prompt(i)})

    client = tq.get_client()
    for partition_id, prefix in ((PARTITION, "a"), (other, "b")):
        rows = client.kv_describe([f"{prefix}{i}" for i in range(4)], partition_id)["rows"]
        assert sorted(row["global_index"] % 2 for row in rows.values()) == [0, 0, 1, 1]
    client.clear_partition(other)


def test_linked_keys_read_the_source_payload(tq_system):
    keys = _put_group(0) + _put_group(10)
    tq.kv_batch_put(
        keys=keys,
        partition_id=PARTITION,
        fields=TensorDict({"response": torch.arange(len(keys)).unsqueeze(1)}, batch_size=len(keys)),
    )

    data = tq.kv_batch_get(keys, PARTITION)

    assert _rows(data["prompt"]) == [_prompt(0).tolist()] * N + [_prompt(10).tolist()] * N
    assert _rows(data["response"]) == [[i] for i in range(len(keys))]


def test_data_parser_refuses_linked_fields(tq_system):
    keys = _put_group(0)
    tq.kv_batch_put(keys, PARTITION, fields=TensorDict({"response": torch.arange(N).unsqueeze(1)}, batch_size=N))

    def identity(data, args):
        return data

    with pytest.raises(NotImplementedError, match="kv_link"):
        tq.kv_batch_get(keys, PARTITION, data_parser=identity)
    data = tq.kv_batch_get(keys, PARTITION, select_fields="response", data_parser=identity)
    assert _rows(data["response"]) == [[i] for i in range(N)]


def test_link_requires_a_produced_source_field(tq_system):
    tq.kv_put(key="p0", partition_id=PARTITION, fields={"prompt": _prompt(0)})

    with pytest.raises(RuntimeError, match="not found"):
        tq.kv_link("r", PARTITION, "missing", "prompt")
    with pytest.raises(RuntimeError, match="must have produced"):
        tq.kv_link("r", PARTITION, "p0", "image")
    with pytest.raises(ValueError, match="duplicates"):
        tq.kv_link(["r", "r"], PARTITION, "p0", "prompt")
    with pytest.raises(RuntimeError, match="cannot link to itself"):
        tq.kv_link(["r", "p0"], PARTITION, "p0", "prompt")
    assert tq.kv_list(PARTITION)[PARTITION].keys() == {"p0"}


def test_link_refuses_to_relink_a_source(tq_system):
    keys = _put_group(0)
    tq.kv_put(key="q0", partition_id=PARTITION, fields={"prompt": _prompt(9)})

    with pytest.raises(RuntimeError, match="linked by other keys"):
        tq.kv_link("p0", PARTITION, "q0", "prompt")

    assert _links("p0") == {}
    assert _rows(tq.kv_batch_get(["p0", *keys], PARTITION)["prompt"]) == [_prompt(0).tolist()] * (N + 1)


def test_link_refuses_a_source_being_cleared(tq_system):
    tq.kv_put(key="p0", partition_id=PARTITION, fields={"prompt": _prompt(0)})
    source = tq.get_client().kv_describe(["p0"], PARTITION)["rows"]["p0"]["global_index"]
    controller = ray.get_actor("TransferQueueController", namespace="transfer_queue")
    ray.get(controller.mark_clearing.remote([source], [PARTITION]))

    with pytest.raises(RuntimeError, match="must have produced"):
        tq.kv_link("r", PARTITION, "p0", "prompt")


def test_source_cannot_be_cleared_while_linked(tq_system):
    keys = _put_group(0)

    with pytest.raises(RuntimeError, match="linked by other keys"):
        tq.kv_clear("p0", PARTITION)
    with pytest.raises(RuntimeError, match="linked by other keys"):
        tq.kv_clear("p0", PARTITION, fields="prompt")
    assert _rows(tq.kv_batch_get(keys, PARTITION)["prompt"]) == [_prompt(0).tolist()] * N

    tq.kv_clear(keys[1:], PARTITION, fields="prompt")
    tq.kv_clear([keys[0], "p0"], PARTITION)

    assert tq.kv_list(PARTITION)[PARTITION].keys() == set(keys[1:])
    assert all(not _links(key) for key in keys[1:])


def test_put_replaces_the_link(tq_system):
    keys = _put_group(0)

    tq.kv_put(key=keys[0], partition_id=PARTITION, fields={"prompt": _prompt(7)})

    assert _links(keys[0]) == {}
    assert _rows(tq.kv_batch_get(keys, PARTITION)["prompt"]) == [_prompt(7).tolist()] + [_prompt(0).tolist()] * 2


def test_put_through_meta_reads_back_the_written_value(tq_system):
    keys = _put_group(0)
    client = tq.get_client()
    meta = client.kv_retrieve_meta(keys, PARTITION)

    client.put(TensorDict({"prompt": torch.stack([_prompt(7)] * N)}, batch_size=N), metadata=meta)

    assert _rows(client.get_data(meta)["prompt"]) == [_prompt(7).tolist()] * N
    assert _rows(tq.kv_batch_get("p0", PARTITION)["prompt"]) == [_prompt(0).tolist()]


def test_update_refuses_a_linked_field(tq_system):
    keys = _put_group(0)

    with pytest.raises(RuntimeError, match="cannot merge into fields linked"):
        tq.kv_batch_update(
            ["new", keys[0]],
            PARTITION,
            TensorDict({"prompt": torch.stack([_prompt(5)] * 2)}, batch_size=2),
            merge_fn=lambda old, new: new,
        )

    assert "new" not in tq.kv_list(PARTITION)[PARTITION]
    assert _rows(tq.kv_batch_get(keys, PARTITION)["prompt"]) == [_prompt(0).tolist()] * N


def test_unfinalized_source_update_is_hidden_from_linked_keys(tq_system):
    keys = _put_group(0)

    tq.kv_update("p0", PARTITION, {"prompt": _prompt(5)}, merge_fn=lambda old, new: new, finalize=False)
    with pytest.raises(ValueError, match="not ready"):
        tq.kv_batch_get(keys, PARTITION, select_fields="prompt")

    tq.kv_update("p0", PARTITION, {"prompt": _prompt(5)}, merge_fn=lambda old, new: new)
    assert _rows(tq.kv_batch_get(keys, PARTITION)["prompt"]) == [_prompt(5).tolist()] * N


def test_dump_and_load_relink(tq_system, dump_test_root):
    keys = _put_group(0)
    dump_dir = dump_test_root / "link_dump"

    with pytest.raises(ValueError, match="must be dumped"):
        tq.dump_data_by_key(dump_dir, keys, PARTITION)

    tq.kv_update("p0", PARTITION, {"prompt": _prompt(0)}, merge_fn=lambda old, new: new, finalize=False)
    with pytest.raises(ValueError, match="must be finalized"):
        tq.dump_data_by_key(dump_dir, ["p0", *keys], PARTITION)
    tq.kv_update("p0", PARTITION, {"prompt": _prompt(0)}, merge_fn=lambda old, new: new)

    tq.dump_data_by_key(dump_dir, ["p0", *keys], PARTITION)
    tq.get_client().clear_partition(PARTITION)
    tq.load_data_by_key(dump_dir)
    shutil.rmtree(dump_dir)

    assert all(_links(key) == {"prompt": "p0"} for key in keys)
    assert _rows(tq.kv_batch_get(keys, PARTITION)["prompt"]) == [_prompt(0).tolist()] * N
