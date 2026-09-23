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

"""End-to-end exercise of kv_update / kv_empty against a real Ray cluster.

kv_update rewrites fields of an existing key by running ``parser(old, new)`` on the
SimpleStorage unit that holds the row, so nothing here works unless the parser
actually travels to the unit and the resulting schema travels back to the
controller. Every check below reads its result back through the public API, and
the multi-unit ones also inspect controller metadata, which is the part a
data-only assertion cannot see.

Run it as a Ray job so the driver lives on the cluster:

    ray job submit --address http://<dashboard-host>:<port> --working-dir . \
        -- python recipe/kv_update_e2e.py

Locally, ``python recipe/kv_update_e2e.py`` starts its own cluster instead.
"""

import asyncio
import os
import traceback

# Disable Ray's cross-worker log deduplication before importing Ray itself,
# otherwise worker-side prints get folded into "[repeated Nx across cluster]".
os.environ.setdefault("RAY_DEDUP_LOGS", "0")

import ray
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

import transfer_queue as tq

PARTITION = "kv_update_e2e"

# Several units so the hash routing actually splits a batch; the cross-unit checks
# below are meaningless with a single unit.
CONFIG = {
    "controller": {"polling_mode": True},
    "backend": {
        "storage_backend": "SimpleStorage",
        "SimpleStorage": {"total_storage_size": 512, "num_data_storage_units": 4},
    },
}


def concat(old, new):
    """Append new tokens to what is already stored, keeping the field name."""
    return new if old is None else torch.cat([old, new])


# ==================== checks ====================


def check_concat_keeps_field_name():
    """The rollout case: prompt_ids in place, response_ids appended, one field."""
    key = "concat"
    tq.kv_put(key=key, partition_id=PARTITION, fields={"sequence_ids": torch.tensor([10, 11, 12])})

    tq.kv_update(
        key=key,
        partition_id=PARTITION,
        fields="sequence_ids",
        values=torch.tensor([20, 21]),
        parser=concat,
    )

    got = tq.kv_batch_get(keys=key, partition_id=PARTITION)
    assert torch.equal(got["sequence_ids"][0], torch.tensor([10, 11, 12, 20, 21])), got["sequence_ids"][0]
    assert list(got.keys()) == ["sequence_ids"], f"update must not add columns: {list(got.keys())}"


def check_parser_sees_none_for_a_new_field():
    """A field the key never held reaches the parser as old=None.

    The parser runs in the storage unit's process, so what it observed can only
    come back inside the value it returns; a closure variable would stay untouched
    here in the driver.
    """
    key = "fresh_field"
    tq.kv_put(key=key, partition_id=PARTITION, fields={"anchor": torch.tensor([1])})

    def report_old(old, new):
        return {"old_was_none": old is None, "new": new.tolist()}

    tq.kv_update(key=key, partition_id=PARTITION, fields="logprobs", values=torch.tensor([7, 8]), parser=report_old)

    got = tq.kv_batch_get(keys=key, partition_id=PARTITION, select_fields="logprobs")
    assert got["logprobs"][0] == {"old_was_none": True, "new": [7, 8]}, got["logprobs"][0]


def check_multiple_fields_in_one_call():
    key = "multi_field"
    tq.kv_put(
        key=key,
        partition_id=PARTITION,
        fields={"a": torch.tensor([1, 2]), "b": torch.tensor([10])},
    )

    tq.kv_update(
        key=key,
        partition_id=PARTITION,
        fields=["a", "b"],
        values={"a": torch.tensor([3]), "b": torch.tensor([20, 30])},
        parser=concat,
    )

    got = tq.kv_batch_get(keys=key, partition_id=PARTITION)
    assert torch.equal(got["a"][0], torch.tensor([1, 2, 3])), got["a"][0]
    assert torch.equal(got["b"][0], torch.tensor([10, 20, 30])), got["b"][0]


def check_non_tensor_payload():
    """Parsers are not tensor-only: a plain object round-trips the same way."""
    key = "non_tensor"
    tq.kv_put(key=key, partition_id=PARTITION, fields={"stats": {"step": 1}})

    tq.kv_update(
        key=key,
        partition_id=PARTITION,
        fields="stats",
        values={"reward": 0.5},
        parser=lambda old, new: {**old, **new},
    )

    got = tq.kv_batch_get(keys=key, partition_id=PARTITION, select_fields="stats")
    assert got["stats"][0] == {"step": 1, "reward": 0.5}, got["stats"][0]


def check_empty_keeps_the_key_and_updates_the_controller():
    """kv_empty stores None; the key stays produced and stops being a tensor column."""
    key = "to_empty"
    tq.kv_put(key=key, partition_id=PARTITION, fields={"tokens": torch.tensor([1, 2, 3])})

    tq.kv_empty(key=key, partition_id=PARTITION, fields="tokens")

    got = tq.kv_batch_get(keys=key, partition_id=PARTITION, select_fields="tokens")
    assert got["tokens"][0] is None, got["tokens"][0]
    assert key in tq.kv_list(partition_id=PARTITION)[PARTITION], "kv_empty must not delete the key"

    partition = partition_snapshot()
    col = partition.field_name_mapping["tokens"]
    row = partition.keys_mapping[key]
    assert partition.production_status[row, col] == 1, "an emptied field stays produced"
    assert partition.field_metadata["tokens"].is_non_tensor is True, "controller still calls the column a tensor"


def check_failed_parser_leaves_the_row_unchanged():
    """The unit computes every value before writing, so a raise is a no-op."""
    key = "atomic"
    tq.kv_put(key=key, partition_id=PARTITION, fields={"tokens": torch.tensor([1, 2, 3])})

    def boom(old, new):
        raise RuntimeError("parser failed on purpose")

    try:
        tq.kv_update(key=key, partition_id=PARTITION, fields="tokens", values=torch.tensor([4]), parser=boom)
    except Exception as e:
        assert "parser failed on purpose" in str(e), f"unexpected error: {e}"
    else:
        raise AssertionError("a raising parser must not succeed")

    got = tq.kv_batch_get(keys=key, partition_id=PARTITION, select_fields="tokens")
    assert torch.equal(got["tokens"][0], torch.tensor([1, 2, 3])), got["tokens"][0]


def check_rejected_arguments():
    key = "rejects"
    tq.kv_put(key=key, partition_id=PARTITION, fields={"tokens": torch.tensor([1])})

    def expect(exc_type, match, fn):
        try:
            fn()
        except exc_type as e:
            assert match in str(e), f"expected {match!r} in {e!r}"
        else:
            raise AssertionError(f"expected {exc_type.__name__} containing {match!r}")

    expect(
        ValueError,
        "must not specify values",
        lambda: tq.kv_update(key=key, partition_id=PARTITION, fields="tokens", values=torch.tensor([1]), empty=True),
    )
    expect(
        ValueError,
        "must not specify parser",
        lambda: tq.kv_update(key=key, partition_id=PARTITION, fields="tokens", parser=concat, empty=True),
    )
    expect(
        TypeError,
        "parser must be callable",
        lambda: tq.kv_update(key=key, partition_id=PARTITION, fields="tokens", values=torch.tensor([1])),
    )
    expect(
        ValueError,
        "requires values",
        lambda: tq.kv_update(key=key, partition_id=PARTITION, fields="tokens", parser=concat),
    )
    expect(
        ValueError,
        "same columns",
        lambda: tq.kv_update(
            key=key, partition_id=PARTITION, fields=["tokens", "other"], values={"tokens": 1}, parser=concat
        ),
    )
    expect(
        ValueError,
        "not found",
        lambda: tq.kv_update(
            key="never_written", partition_id=PARTITION, fields="tokens", values=torch.tensor([1]), parser=concat
        ),
    )


def check_async_variants():
    key = "async_key"
    tq.kv_put(key=key, partition_id=PARTITION, fields={"tokens": torch.tensor([1, 2])})

    asyncio.run(
        tq.async_kv_update(key=key, partition_id=PARTITION, fields="tokens", values=torch.tensor([3]), parser=concat)
    )
    got = tq.kv_batch_get(keys=key, partition_id=PARTITION, select_fields="tokens")
    assert torch.equal(got["tokens"][0], torch.tensor([1, 2, 3])), got["tokens"][0]

    asyncio.run(tq.async_kv_empty(key=key, partition_id=PARTITION, fields="tokens"))
    got = tq.kv_batch_get(keys=key, partition_id=PARTITION, select_fields="tokens")
    assert got["tokens"][0] is None, got["tokens"][0]


def check_multi_sample_update_across_units():
    """One update spanning several units must report per-sample shapes in batch order.

    Each key starts with a different length, so concatenating turns the column
    nested. The controller's per_sample_shapes is the only place a mis-ordered or
    overwritten merge shows up; the payload alone would still look correct.
    """
    keys = [f"span_{i}" for i in range(8)]
    lengths = [i + 1 for i in range(len(keys))]
    for key, length in zip(keys, lengths, strict=True):
        tq.kv_put(key=key, partition_id=PARTITION, fields={"tokens": torch.ones(length, dtype=torch.int64)})

    client = tq.get_client()
    metadata = client.kv_retrieve_meta(keys=keys, partition_id=PARTITION, create=False)
    # Same addition for every row, so this holds whichever order the metadata came back in.
    values = TensorDict({"tokens": torch.full((len(keys), 1), 9, dtype=torch.int64)}, batch_size=[len(keys)])
    client.update(metadata, ["tokens"], values=values, parser=concat)

    got = tq.kv_batch_get(keys=keys, partition_id=PARTITION, select_fields="tokens")
    for i, (key, length) in enumerate(zip(keys, lengths, strict=True)):
        row = got["tokens"][i]
        assert row.shape == (length + 1,), f"{key}: expected len {length + 1}, got {tuple(row.shape)}"
        assert row[-1].item() == 9, f"{key}: appended value missing"

    partition = partition_snapshot()
    tokens = partition.field_metadata["tokens"]
    assert tokens.is_nested is True, "mixed lengths must promote the column to nested"
    for key, length in zip(keys, lengths, strict=True):
        recorded = tuple(tokens.per_sample_shapes[partition.keys_mapping[key]])
        assert recorded == (length + 1,), f"{key}: controller recorded {recorded}, expected {(length + 1,)}"


@ray.remote(num_cpus=0)
class Updater:
    """A worker that attaches to the running TransferQueue and updates its own key."""

    def __init__(self):
        tq.init()

    def run(self, key: str, addition: int) -> str:
        tq.kv_update(
            key=key,
            partition_id=PARTITION,
            fields="tokens",
            values=torch.tensor([addition]),
            parser=concat,
        )
        return key


def check_updates_from_remote_workers():
    """Updates issued by other processes on the cluster, not just the driver."""
    keys = [f"worker_{i}" for i in range(4)]
    for i, key in enumerate(keys):
        tq.kv_put(key=key, partition_id=PARTITION, fields={"tokens": torch.tensor([i])})

    updaters = [Updater.remote() for _ in keys]
    ray.get([u.run.remote(key, 100 + i) for i, (u, key) in enumerate(zip(updaters, keys, strict=True))])
    for u in updaters:
        ray.kill(u)

    got = tq.kv_batch_get(keys=keys, partition_id=PARTITION, select_fields="tokens")
    for i, key in enumerate(keys):
        expected = torch.tensor([i, 100 + i])
        assert torch.equal(got["tokens"][i], expected), f"{key}: {got['tokens'][i]} != {expected}"


# ==================== harness ====================


def partition_snapshot():
    controller = ray.get_actor("TransferQueueController", namespace="transfer_queue")
    return ray.get(controller.get_partition_snapshot.remote(PARTITION))


CHECKS = [
    ("concat keeps the field name", check_concat_keeps_field_name),
    ("parser sees old=None for a new field", check_parser_sees_none_for_a_new_field),
    ("several fields in one call", check_multiple_fields_in_one_call),
    ("non-tensor payload", check_non_tensor_payload),
    ("kv_empty keeps the key, updates the controller", check_empty_keeps_the_key_and_updates_the_controller),
    ("a failed parser leaves the row unchanged", check_failed_parser_leaves_the_row_unchanged),
    ("rejected arguments", check_rejected_arguments),
    ("async_kv_update / async_kv_empty", check_async_variants),
    ("multi-sample update across units", check_multi_sample_update_across_units),
    ("updates from remote workers", check_updates_from_remote_workers),
]


def refuse_if_already_deployed() -> None:
    """Never attach to a TransferQueue somebody else is using.

    tq.init() silently joins an existing controller by name, so on a shared cluster
    this script would otherwise run its checks against a live deployment and clear
    its keys on the way out.
    """
    try:
        ray.get_actor("TransferQueueController", namespace="transfer_queue")
    except ValueError:
        return
    raise SystemExit(
        "A TransferQueueController is already running on this cluster. This script needs a "
        "deployment of its own; stop the owning job or run it on an idle cluster."
    )


def main() -> int:
    if not ray.is_initialized():
        ray.init(namespace="transfer_queue")

    print("=" * 78)
    print("kv_update end-to-end checks")
    print(f"cluster: {ray.get_runtime_context().gcs_address}  nodes: {len(ray.nodes())}")
    print("=" * 78)

    refuse_if_already_deployed()
    tq.init(OmegaConf.create(CONFIG))
    failures = []
    try:
        for name, check in CHECKS:
            try:
                check()
            except Exception as e:
                failures.append(name)
                print(f"  FAIL  {name}: {type(e).__name__}: {e}")
                traceback.print_exc()
            else:
                print(f"  ok    {name}")
            finally:
                # Each check owns its keys; clear between them so one failure cannot cascade.
                leftover = list(tq.kv_list(partition_id=PARTITION).get(PARTITION, {}))
                if leftover:
                    tq.kv_clear(keys=leftover, partition_id=PARTITION)
    finally:
        tq.close()

    print("=" * 78)
    print(f"{len(CHECKS) - len(failures)}/{len(CHECKS)} checks passed")
    if failures:
        print("failed: " + ", ".join(failures))
    print("=" * 78)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
