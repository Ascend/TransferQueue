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

"""End-to-end tests for kv_link: n keys reading one PUBLIC_PARTITION payload without copies."""

import asyncio
import shutil

import pytest
import ray
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

import transfer_queue as tq
from transfer_queue.data_dump import _read_row_index

PUBLIC = tq.PUBLIC_PARTITION
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
def fresh_partitions(tq_system):
    yield
    # Linking keys first: a public key cannot be cleared while keys link to it.
    tq.get_client().clear_partition(PARTITION)
    tq.get_client().clear_partition(PUBLIC)


def _prompt(p: int, length: int = 3) -> torch.Tensor:
    return torch.arange(p, p + length)


def _put_group(p: int) -> list[str]:
    """Put prompt ``p`` once into the public partition and link N response keys to it."""
    tq.kv_put(key=f"p{p}", partition_id=PUBLIC, fields={"prompt": _prompt(p)})
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
    for i in range(4):
        tq.kv_put(key=f"p{i}", partition_id=PUBLIC, fields={"prompt": _prompt(i)})
        tq.kv_put(key=f"r{i}", partition_id=PARTITION, fields={"response": _prompt(i)})

    client = tq.get_client()
    for partition_id, prefix in ((PUBLIC, "p"), (PARTITION, "r")):
        rows = client.kv_describe([f"{prefix}{i}" for i in range(4)], partition_id)["rows"]
        assert sorted(row["global_index"] % 2 for row in rows.values()) == [0, 0, 1, 1]


def test_linked_keys_read_the_public_payload(tq_system):
    keys = _put_group(0) + _put_group(10)
    tq.kv_batch_put(
        keys=keys,
        partition_id=PARTITION,
        fields=TensorDict({"response": torch.arange(len(keys)).unsqueeze(1)}, batch_size=len(keys)),
    )

    data = tq.kv_batch_get(keys, PARTITION)

    assert _rows(data["prompt"]) == [_prompt(0).tolist()] * N + [_prompt(10).tolist()] * N
    assert _rows(data["response"]) == [[i] for i in range(len(keys))]
    assert tq.kv_describe(keys[0], PARTITION)[keys[0]] == {
        "fields": ["prompt", "response"],
        "unfinalized": [],
        "links": {"prompt": "p0"},
    }


def test_link_requires_a_produced_public_field(tq_system):
    tq.kv_put(key="p0", partition_id=PUBLIC, fields={"prompt": _prompt(0)})

    with pytest.raises(RuntimeError, match="not found"):
        tq.kv_link("r", PARTITION, "missing", "prompt")
    with pytest.raises(RuntimeError, match="must have produced"):
        tq.kv_link("r", PARTITION, "p0", "image")
    with pytest.raises(RuntimeError, match="cannot link"):
        tq.kv_link("r", PUBLIC, "p0", "prompt")
    with pytest.raises(ValueError, match="duplicates"):
        tq.kv_link(["r", "r"], PARTITION, "p0", "prompt")
    with pytest.raises(RuntimeError, match="must not be empty"):
        tq.kv_link([], PARTITION, "p0", "prompt")
    assert tq.kv_list().get(PARTITION, {}) == {}


def test_link_a_batch_of_public_keys_in_one_call(tq_system):
    for p in range(2):
        tq.kv_put(key=f"p{p}", partition_id=PUBLIC, fields={"prompt": _prompt(p)})
    keys = ["p0_r0", "p0_r1", "p1_r0"]

    with pytest.raises(RuntimeError, match="2 public keys for 3 keys"):
        tq.kv_link(keys, PARTITION, ["p0", "p1"], "prompt")
    # Atomic: one missing public key links and creates nothing.
    with pytest.raises(RuntimeError, match=r"keys \['missing'\] not found"):
        tq.kv_link(keys, PARTITION, ["p0", "p0", "missing"], "prompt")
    assert tq.kv_list().get(PARTITION, {}) == {}

    tq.kv_link(keys, PARTITION, ["p0", "p0", "p1"], "prompt")
    assert _rows(tq.kv_batch_get(keys, PARTITION)["prompt"]) == [_prompt(0).tolist()] * 2 + [_prompt(1).tolist()]
    assert [_links(key) for key in keys] == [{"prompt": "p0"}, {"prompt": "p0"}, {"prompt": "p1"}]


def test_exists_reports_ready_fields_and_omits_missing_keys(tq_system):
    assert tq.kv_exists(["p0"], "never_created") == {}
    tq.kv_put(key="p0", partition_id=PUBLIC, fields={"prompt": _prompt(0)})
    tq.kv_put(key="p0", partition_id=PUBLIC, fields={"meta": _prompt(1)})
    tq.kv_link("r0", PARTITION, "p0", "prompt")

    assert tq.kv_exists(["p0", "missing", "p0"], PUBLIC) == {"p0": ["meta", "prompt"]}
    assert tq.kv_exists("r0", PARTITION) == {"r0": ["prompt"]}
    assert asyncio.run(tq.async_kv_exists(["missing", "p0"], PUBLIC)) == {"p0": ["meta", "prompt"]}


def test_exists_reports_no_ready_field_for_a_key_being_cleared(tq_system):
    tq.kv_put(key="p0", partition_id=PUBLIC, fields={"prompt": _prompt(0)})
    source = tq.get_client().kv_describe(["p0"], PUBLIC)["rows"]["p0"]["global_index"]
    controller = ray.get_actor("TransferQueueController", namespace="transfer_queue")
    ray.get(controller.mark_clearing.remote([source], [PUBLIC]))

    assert tq.kv_exists("p0", PUBLIC) == {"p0": []}


def test_link_refuses_a_field_the_key_stores_itself(tq_system):
    tq.kv_put(key="p0", partition_id=PUBLIC, fields={"prompt": _prompt(0)})
    tq.kv_put(key="r0", partition_id=PARTITION, fields={"prompt": _prompt(7)})

    with pytest.raises(RuntimeError, match="already store"):
        tq.kv_link(["r0", "r1"], PARTITION, "p0", "prompt")
    assert tq.kv_list(PARTITION)[PARTITION].keys() == {"r0"}

    tq.kv_clear("r0", PARTITION, fields="prompt")
    tq.kv_link(["r0", "r1"], PARTITION, "p0", "prompt")
    assert _links("r0") == {"prompt": "p0"}


def test_link_requires_simple_storage_on_the_client_too(tq_system, monkeypatch):
    tq.kv_put(key="p0", partition_id=PUBLIC, fields={"prompt": _prompt(0)})
    client = tq.get_client()
    monkeypatch.setattr(client, "storage_manager", object())

    with pytest.raises(NotImplementedError, match="does not support kv_link"):
        client.kv_link(["r"], PARTITION, ["p0"], ["prompt"])
    with pytest.raises(NotImplementedError, match="does not support kv_link"):
        tq.kv_link("r", PARTITION, "p0", "prompt")
    monkeypatch.undo()
    assert tq.kv_list().get(PARTITION, {}) == {}


def test_link_with_conflicting_dtype_creates_no_keys(tq_system):
    tq.kv_put(key="a", partition_id=PARTITION, fields={"prompt": _prompt(0)})
    tq.kv_put(key="p0", partition_id=PUBLIC, fields={"prompt": _prompt(0).float()})

    with pytest.raises(RuntimeError, match="dtype mismatch"):
        tq.kv_link(["r0", "r1"], PARTITION, "p0", "prompt")
    assert tq.kv_list(PARTITION)[PARTITION].keys() == {"a"}


def test_link_a_tensor_into_a_field_put_as_non_tensor(tq_system):
    # A put of a non-tensor value turns the field non-tensor for the partition, and later tensor puts still land.
    tq.kv_put(key="text", partition_id=PARTITION, fields={"prompt": "no image"})
    tq.kv_put(key="p0", partition_id=PUBLIC, fields={"prompt": _prompt(0)})

    tq.kv_link(["r0", "r1"], PARTITION, "p0", "prompt")

    data = tq.kv_batch_get(["text", "r0", "r1"], PARTITION)["prompt"]
    assert data[0] == "no image"
    assert [row.tolist() for row in data[1:]] == [_prompt(0).tolist()] * 2


def test_link_refuses_a_public_key_being_cleared(tq_system):
    tq.kv_put(key="p0", partition_id=PUBLIC, fields={"prompt": _prompt(0)})
    source = tq.get_client().kv_describe(["p0"], PUBLIC)["rows"]["p0"]["global_index"]
    controller = ray.get_actor("TransferQueueController", namespace="transfer_queue")
    ray.get(controller.mark_clearing.remote([source], [PUBLIC]))

    with pytest.raises(RuntimeError, match="must have produced"):
        tq.kv_link("r", PARTITION, "p0", "prompt")


def test_link_made_while_its_public_key_is_cleared_is_dropped(tq_system):
    tq.kv_put(key="p0", partition_id=PUBLIC, fields={"prompt": _prompt(0)})
    source = tq.get_client().kv_describe(["p0"], PUBLIC)["rows"]["p0"]["global_index"]
    controller = ray.get_actor("TransferQueueController", namespace="transfer_queue")
    ray.get(controller.mark_clearing.remote([source], [PUBLIC]))
    # A concurrent put makes p0 ready again, so a link slips in before the clear finishes.
    tq.kv_put(key="p0", partition_id=PUBLIC, fields={"prompt": _prompt(0)})
    tq.kv_link("r", PARTITION, "p0", "prompt")

    ray.get(controller.clear_meta.remote([source], [PUBLIC]))

    assert _links("r") == {}
    assert "prompt" not in tq.kv_describe("r", PARTITION)["r"]["fields"]


def test_public_key_cannot_be_cleared_while_linked(tq_system):
    keys = _put_group(0)

    with pytest.raises(RuntimeError, match=r"linked by keys of other partitions \(3 keys, e.g. \['link:p0_r0'"):
        tq.kv_clear("p0", PUBLIC)
    with pytest.raises(RuntimeError, match="linked by keys of other partitions"):
        tq.kv_clear("p0", PUBLIC, fields="prompt")
    with pytest.raises(RuntimeError, match="linked by keys of other partitions"):
        tq.get_client().clear_partition(PUBLIC)
    assert _rows(tq.kv_batch_get(keys, PARTITION)["prompt"]) == [_prompt(0).tolist()] * N

    tq.kv_clear(keys[0], PARTITION)
    tq.kv_clear(keys[1:], PARTITION, fields="prompt")
    tq.kv_clear("p0", PUBLIC)

    assert tq.kv_list().get(PUBLIC, {}) == {}
    assert tq.kv_list(PARTITION)[PARTITION].keys() == set(keys[1:])
    assert all(not _links(key) for key in keys[1:])


def test_clear_unlinked_public_frees_only_keys_no_longer_linked(tq_system):
    keys0, keys1 = _put_group(0), _put_group(1)
    # Put but not linked yet, as by a producer between kv_put and kv_link.
    tq.kv_put(key="p2", partition_id=PUBLIC, fields={"prompt": _prompt(2)})
    tq.kv_clear(keys1[:2], PARTITION)
    tq.kv_clear(keys1[2:], PARTITION, fields="prompt")

    assert tq.kv_clear_unlinked_public() == ["p1"]

    assert tq.kv_list(PUBLIC)[PUBLIC].keys() == {"p0", "p2"}
    with pytest.raises(ValueError, match="not found"):
        tq.kv_batch_get("p1", PUBLIC)
    assert _rows(tq.kv_batch_get(keys0, PARTITION)["prompt"]) == [_prompt(0).tolist()] * N
    assert tq.kv_clear_unlinked_public() == []


def test_clear_unlinked_public_frees_stale_keys_no_key_links(tq_system):
    keys0 = _put_group(0)
    # Never linked, as when every waiter of a producer was cancelled before kv_link.
    tq.kv_put(key="p1", partition_id=PUBLIC, fields={"prompt": _prompt(1)})
    tq.kv_put(key="p2", partition_id=PUBLIC, fields={"prompt": _prompt(2)})

    assert tq.kv_clear_unlinked_public(stale_keys=["p0", "p1", "missing"]) == ["p1"]

    assert tq.kv_list(PUBLIC)[PUBLIC].keys() == {"p0", "p2"}
    assert _rows(tq.kv_batch_get(keys0, PARTITION)["prompt"]) == [_prompt(0).tolist()] * N
    assert asyncio.run(tq.async_kv_clear_unlinked_public(stale_keys=["p2"])) == ["p2"]


def test_link_refuses_a_key_marked_by_clear_unlinked_public(tq_system):
    keys = _put_group(0)
    tq.kv_clear(keys, PARTITION)
    controller = ray.get_actor("TransferQueueController", namespace="transfer_queue")
    ray.get(controller.mark_unlinked_public_clearing.remote())

    with pytest.raises(RuntimeError, match="must have produced"):
        tq.kv_link("r", PARTITION, "p0", "prompt")


def test_checkpoint_restores_links_and_their_clear_guard(tq_system, tmp_path):
    keys = _put_group(0)
    tq.save_checkpoint(tmp_path / "ck")
    tq.kv_clear(keys, PARTITION)
    tq.kv_clear("p0", PUBLIC)

    tq.load_checkpoint(tmp_path / "ck")

    assert _rows(tq.kv_batch_get(keys, PARTITION)["prompt"]) == [_prompt(0).tolist()] * N
    assert all(_links(key) == {"prompt": "p0"} for key in keys)
    with pytest.raises(RuntimeError, match="linked by keys of other partitions"):
        tq.kv_clear("p0", PUBLIC)
    tq.kv_clear(keys, PARTITION)
    assert tq.kv_clear_unlinked_public() == ["p0"]


def test_put_refuses_a_linked_field(tq_system):
    keys = _put_group(0)
    tq.kv_batch_put(keys, PARTITION, fields=TensorDict({"response": torch.arange(N).unsqueeze(1)}, batch_size=N))

    with pytest.raises(ValueError, match="linked by kv_link"):
        tq.kv_put(key=keys[0], partition_id=PARTITION, fields={"prompt": _prompt(7)})
    # A read-modify-write of the whole batch would turn every link into a private copy.
    data = tq.kv_batch_get(keys, PARTITION)
    data["response"] += 100
    with pytest.raises(ValueError, match="linked by kv_link"):
        tq.kv_batch_put(keys, PARTITION, fields=data)
    assert all(_links(key) == {"prompt": "p0"} for key in keys)

    tq.kv_batch_put(keys, PARTITION, fields=data.select("response"))
    tq.kv_clear(keys[0], PARTITION, fields="prompt")
    tq.kv_put(key=keys[0], partition_id=PARTITION, fields={"prompt": _prompt(7)})
    assert _links(keys[0]) == {}
    assert _rows(tq.kv_batch_get(keys, PARTITION)["prompt"]) == [_prompt(7).tolist()] + [_prompt(0).tolist()] * 2
    assert _rows(tq.kv_batch_get(keys, PARTITION)["response"]) == [[i + 100] for i in range(N)]


def test_put_through_meta_refuses_a_linked_field(tq_system):
    keys = _put_group(0)
    client = tq.get_client()
    meta = client.kv_retrieve_meta(keys, PARTITION)

    with pytest.raises(ValueError, match="linked by kv_link"):
        client.put(TensorDict({"prompt": torch.stack([_prompt(7)] * N)}, batch_size=N), metadata=meta)
    assert _rows(tq.kv_batch_get(keys, PARTITION)["prompt"]) == [_prompt(0).tolist()] * N


def _per_value(parse):
    """A data_parser applying ``parse(value, arg)`` to each value of every field."""
    return lambda data, args: {
        field: [parse(value, arg) for value, arg in zip(values, args, strict=True)] for field, values in data.items()
    }


def test_data_parser_slices_a_linked_field_per_key(tq_system):
    keys = _put_group(0)
    tq.kv_batch_put(keys, PARTITION, fields=TensorDict({"response": torch.arange(N).unsqueeze(1)}, batch_size=N))

    data = tq.kv_batch_get(
        keys, PARTITION, data_parser=_per_value(lambda value, length: value[:length]), parser_args=[1, 3, 1]
    )

    assert _rows(data["prompt"]) == [_prompt(0)[:1].tolist(), _prompt(0).tolist(), _prompt(0)[:1].tolist()]
    assert _rows(data["response"]) == [[i] for i in range(N)]


def test_data_parser_selects_elements_of_a_linked_field_per_key(tq_system):
    keys = _put_group(0) + _put_group(10)

    data = tq.kv_batch_get(
        keys,
        PARTITION,
        select_fields="prompt",
        data_parser=_per_value(lambda value, positions: value[positions]),
        parser_args=[[0, 2], [1], [2, 2, 0], [0], [0], [1, 2]],
    )

    assert _rows(data["prompt"]) == [[0, 2], [1], [2, 2, 0], [10], [10], [11, 12]]


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


def test_linked_public_key_is_read_only(tq_system):
    keys = _put_group(0)

    for write in (
        lambda: tq.kv_put(key="p0", partition_id=PUBLIC, fields={"prompt": _prompt(5, 5)}),
        lambda: tq.kv_put(key="p0", partition_id=PUBLIC, tag={"step": 1}),
        lambda: tq.kv_update("p0", PUBLIC, {"prompt": _prompt(5, 5)}, merge_fn=lambda old, new: new),
    ):
        with pytest.raises(RuntimeError, match="linked by keys of other partitions"):
            write()
    assert _rows(tq.kv_batch_get(keys, PARTITION)["prompt"]) == [_prompt(0).tolist()] * N

    client = tq.get_client()
    meta = client.kv_retrieve_meta(["p0"], PUBLIC)
    with pytest.raises(RuntimeError, match="linked by keys of other partitions"):
        client.put(TensorDict({"prompt": _prompt(5, 5).unsqueeze(0)}, batch_size=1), metadata=meta)
    assert _rows(tq.kv_batch_get(keys, PARTITION)["prompt"]) == [_prompt(0).tolist()] * N


def test_dump_and_load_relink(tq_system, dump_test_root):
    keys = _put_group(0)
    public_dir, linked_dir = dump_test_root / "public_dump", dump_test_root / "link_dump"
    tq.dump_data_by_key(public_dir, ["p0"], PUBLIC)
    tq.dump_data_by_key(linked_dir, keys, PARTITION)
    tq.get_client().clear_partition(PARTITION)
    tq.get_client().clear_partition(PUBLIC)

    with pytest.raises(RuntimeError, match="does not exist|not found"):
        tq.load_data_by_key(linked_dir)
    assert tq.kv_list().get(PARTITION, {}) == {}

    assert _read_row_index(linked_dir)["rows"][keys[0]]["fields"] == []
    tq.load_data_by_key(public_dir)
    tq.load_data_by_key(linked_dir)
    shutil.rmtree(public_dir)
    shutil.rmtree(linked_dir)

    assert all(_links(key) == {"prompt": "p0"} for key in keys)
    assert _rows(tq.kv_batch_get(keys, PARTITION)["prompt"]) == [_prompt(0).tolist()] * N


def test_load_with_an_incompatible_linked_field_writes_nothing(tq_system, dump_test_root):
    keys = _put_group(0)
    tq.kv_batch_put(keys, PARTITION, fields=TensorDict({"response": torch.arange(N).unsqueeze(1)}, batch_size=N))
    linked_dir = dump_test_root / "incompatible_link_dump"
    tq.dump_data_by_key(linked_dir, keys, PARTITION)
    tq.get_client().clear_partition(PARTITION)
    tq.kv_put(key="other", partition_id=PARTITION, fields={"prompt": _prompt(0).float()})

    with pytest.raises(RuntimeError, match="dtype mismatch"):
        tq.load_data_by_key(linked_dir)
    shutil.rmtree(linked_dir)

    assert tq.kv_list(PARTITION)[PARTITION].keys() == {"other"}


def _meta_group(p: int):
    """Put prompt ``p`` into the public partition and N response rows, all through a BatchMeta with no keys."""
    client = tq.get_client()
    public_meta = client.put(TensorDict({"prompt": _prompt(p).unsqueeze(0)}, batch_size=1), partition_id=PUBLIC)
    rows_meta = client.put(TensorDict({"response": torch.arange(N).unsqueeze(1)}, batch_size=N), partition_id=PARTITION)
    return public_meta, rows_meta


def test_batch_meta_rows_read_a_public_row_put_through_a_batch_meta(tq_system):
    client = tq.get_client()
    public_meta, rows_meta = _meta_group(0)

    linked = client.link(rows_meta, public_meta.select_samples([0] * N), "prompt")

    data = client.get_data(linked)
    assert _rows(data["prompt"]) == [_prompt(0).tolist()] * N
    assert _rows(data["response"]) == [[i] for i in range(N)]
    assert linked._links == [{"prompt": public_meta.global_indexes[0]}] * N


def test_batch_meta_put_refuses_a_linked_public_row_until_no_row_links_it(tq_system):
    client = tq.get_client()
    public_meta, rows_meta = _meta_group(0)
    linked = client.link(rows_meta, public_meta.select_samples([0] * N), "prompt")
    rewrite = TensorDict({"prompt": _prompt(7).unsqueeze(0)}, batch_size=1)

    with pytest.raises(RuntimeError, match="linked by keys of other partitions"):
        client.put(rewrite, metadata=public_meta)
    assert _rows(client.get_data(linked)["prompt"]) == [_prompt(0).tolist()] * N

    client.clear_samples(linked)
    client.put(rewrite, metadata=public_meta)
    assert _rows(client.get_data(public_meta)["prompt"]) == [_prompt(7).tolist()]


def test_unlinked_public_rows_without_keys_are_freed_once_their_rows_are_cleared(tq_system):
    client = tq.get_client()
    public_meta, rows_meta = _meta_group(0)
    linked = client.link(rows_meta, public_meta.select_samples([0] * N), "prompt")

    with pytest.raises(RuntimeError, match="linked by keys of other partitions"):
        client.clear_samples(public_meta)

    client.clear_samples(linked)
    assert tq.kv_clear_unlinked_public() == []
    _, more_rows = _meta_group(1)
    with pytest.raises(RuntimeError, match="not found"):
        client.link(more_rows, public_meta.select_samples([0] * N), "prompt")


def test_batch_meta_link_refuses_what_kv_link_refuses(tq_system):
    client = tq.get_client()
    public_meta, rows_meta = _meta_group(0)
    public_rows = public_meta.select_samples([0] * N)

    with pytest.raises(RuntimeError, match="must have produced"):
        client.link(rows_meta, public_rows, "response")
    with pytest.raises(RuntimeError, match="cannot link"):
        client.link(public_meta, public_meta, "prompt")
    with pytest.raises(RuntimeError, match=f"1 public rows for {N} rows"):
        client.link(rows_meta, public_meta, "prompt")
    with pytest.raises(ValueError, match=f"only partition {PUBLIC!r}"):
        client.link(rows_meta, rows_meta, "prompt")

    client.put(TensorDict({"prompt": _prompt(5).unsqueeze(0).repeat(N, 1)}, batch_size=N), metadata=rows_meta)
    with pytest.raises(RuntimeError, match="already store"):
        client.link(rows_meta, public_rows, "prompt")


def test_data_parser_sees_every_field_a_row_links_to_one_public_row(tq_system):
    client = tq.get_client()
    public_meta = client.put(
        TensorDict({"prompt": torch.tensor([[1, 2, 3]]), "length": torch.tensor([[2]])}, batch_size=1),
        partition_id=PUBLIC,
    )
    rows_meta = client.put(TensorDict({"response": torch.tensor([[0]])}, batch_size=1), partition_id=PARTITION)
    linked = client.link(rows_meta, public_meta, ["prompt", "length"]).select_fields(["prompt", "length"])

    def trim_prompt_to_length(data, args):
        prompts = zip(data["prompt"], data["length"], strict=True)
        return {"prompt": [value[: int(length.item())] for value, length in prompts], "length": data["length"]}

    data = client.get_data(linked, data_parser=trim_prompt_to_length)

    assert _rows(data["prompt"]) == [[1, 2]]
    assert _rows(data["length"]) == [[2]]


def test_batch_meta_link_keeps_the_callers_extra_info_and_custom_meta(tq_system):
    client = tq.get_client()
    public_meta, rows_meta = _meta_group(0)
    rows_meta.extra_info["epoch"] = 3
    rows_meta.update_custom_meta([{"group": f"g{i}"} for i in range(N)])

    linked = client.link(rows_meta, public_meta.select_samples([0] * N), "prompt")

    assert linked.extra_info == {"epoch": 3}
    assert linked.custom_meta == [{"group": f"g{i}"} for i in range(N)]
