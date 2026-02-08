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

"""
End-to-end tests for the high-level KV interface.

These tests verify the full lifecycle of the KV API:
  kv_put → kv_batch_put → kv_batch_get → kv_list → kv_clear

Prerequisites:
  - Ray must be available
  - TransferQueue must be initializable (tq.init())

Run:
  pytest tests/e2e/test_kv_interface_e2e.py -v
"""

import sys
from pathlib import Path

import pytest
import ray
import torch

# Setup path
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

import transfer_queue as tq  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def setup_teardown():
    """Initialize and tear down TransferQueue for the entire test module."""
    if not ray.is_initialized():
        ray.init(namespace="test_kv_interface_e2e")
    tq.init()
    yield
    tq.close()
    ray.shutdown()


@pytest.fixture(autouse=True)
def clean_partition():
    """Clean up the test partition before each test."""
    yield
    # Best-effort cleanup after each test
    try:
        tq.kv_clear(partition_id="test_kv")
    except Exception:
        pass


class TestKVPut:
    """Tests for kv_put and kv_batch_put."""

    def test_kv_put_single(self):
        """kv_put should insert a single sample and make it retrievable."""
        tq.kv_put(
            key="k1",
            fields={"x": torch.tensor([1.0, 2.0])},
            partition_id="test_kv",
        )

        result = tq.kv_batch_get(keys=["k1"], partition_id="test_kv")
        assert "k1" in result
        assert torch.allclose(result["k1"]["x"].squeeze(), torch.tensor([1.0, 2.0]))

    def test_kv_put_with_tag(self):
        """kv_put with tag should store metadata alongside the sample."""
        tq.kv_put(
            key="tagged",
            fields={"v": torch.tensor([42])},
            partition_id="test_kv",
            tag={"status": "ready"},
        )

        entries = tq.kv_list(partition_id="test_kv")
        tagged_entry = [e for e in entries if e["key"] == "tagged"]
        assert len(tagged_entry) == 1
        assert tagged_entry[0]["tag"]["status"] == "ready"

    def test_kv_batch_put(self):
        """kv_batch_put should insert multiple samples in one call."""
        tq.kv_batch_put(
            kv_pairs={
                "a": {"val": torch.tensor([10])},
                "b": {"val": torch.tensor([20])},
                "c": {"val": torch.tensor([30])},
            },
            partition_id="test_kv",
        )

        result = tq.kv_batch_get(keys=["a", "b", "c"], partition_id="test_kv")
        assert result["a"]["val"].item() == 10
        assert result["b"]["val"].item() == 20
        assert result["c"]["val"].item() == 30

    def test_kv_batch_put_with_tags(self):
        """kv_batch_put should support per-key tags."""
        tq.kv_batch_put(
            kv_pairs={
                "x": {"d": torch.tensor([1])},
                "y": {"d": torch.tensor([2])},
            },
            partition_id="test_kv",
            tags={
                "x": {"label": "pos"},
                "y": {"label": "neg"},
            },
        )

        entries = tq.kv_list(partition_id="test_kv")
        tag_map = {e["key"]: e.get("tag", {}) for e in entries}
        assert tag_map["x"]["label"] == "pos"
        assert tag_map["y"]["label"] == "neg"


class TestKVGet:
    """Tests for kv_batch_get."""

    def test_get_all_fields(self):
        """kv_batch_get without field filter should return all fields."""
        tq.kv_batch_put(
            kv_pairs={
                "m1": {"f1": torch.tensor([1.0]), "f2": torch.tensor([2.0])},
            },
            partition_id="test_kv",
        )

        result = tq.kv_batch_get(keys=["m1"], partition_id="test_kv")
        assert "f1" in result["m1"].keys()
        assert "f2" in result["m1"].keys()

    def test_get_selected_fields(self):
        """kv_batch_get with field selection should return only requested fields."""
        tq.kv_batch_put(
            kv_pairs={
                "m2": {"alpha": torch.tensor([3.0]), "beta": torch.tensor([4.0])},
            },
            partition_id="test_kv",
        )

        result = tq.kv_batch_get(keys=["m2"], fields=["alpha"], partition_id="test_kv")
        assert "alpha" in result["m2"].keys()

    def test_get_missing_key_raises(self):
        """kv_batch_get should raise KeyError for non-existent keys."""
        with pytest.raises(KeyError, match="Keys not found in partition"):
            tq.kv_batch_get(keys=["nonexistent"], partition_id="test_kv")


class TestKVList:
    """Tests for kv_list."""

    def test_list_empty_partition(self):
        """kv_list on empty/unknown partition should return empty list."""
        entries = tq.kv_list(partition_id="empty_partition_xyz")
        assert entries == []

    def test_list_returns_all_keys(self):
        """kv_list should return all keys in the partition."""
        tq.kv_batch_put(
            kv_pairs={
                "p": {"v": torch.tensor([1])},
                "q": {"v": torch.tensor([2])},
                "r": {"v": torch.tensor([3])},
            },
            partition_id="test_kv",
        )

        entries = tq.kv_list(partition_id="test_kv")
        keys = {e["key"] for e in entries}
        assert keys == {"p", "q", "r"}


class TestKVClear:
    """Tests for kv_clear."""

    def test_clear_specific_keys(self):
        """kv_clear with keys should remove only specified entries."""
        tq.kv_batch_put(
            kv_pairs={
                "d1": {"v": torch.tensor([1])},
                "d2": {"v": torch.tensor([2])},
                "d3": {"v": torch.tensor([3])},
            },
            partition_id="test_kv",
        )

        tq.kv_clear(keys=["d1", "d3"], partition_id="test_kv")

        entries = tq.kv_list(partition_id="test_kv")
        keys = {e["key"] for e in entries}
        assert "d1" not in keys
        assert "d3" not in keys
        assert "d2" in keys

    def test_clear_entire_partition(self):
        """kv_clear without keys should wipe the entire partition."""
        tq.kv_batch_put(
            kv_pairs={
                "e1": {"v": torch.tensor([1])},
                "e2": {"v": torch.tensor([2])},
            },
            partition_id="test_kv",
        )

        tq.kv_clear(partition_id="test_kv")
        entries = tq.kv_list(partition_id="test_kv")
        assert entries == []


class TestPartitionIsolation:
    """Tests for partition namespace isolation."""

    def test_same_key_different_partitions(self):
        """Same key in different partitions should hold independent values."""
        tq.kv_put(key="shared", fields={"v": torch.tensor([100])}, partition_id="ns_1")
        tq.kv_put(key="shared", fields={"v": torch.tensor([200])}, partition_id="ns_2")

        r1 = tq.kv_batch_get(keys=["shared"], partition_id="ns_1")
        r2 = tq.kv_batch_get(keys=["shared"], partition_id="ns_2")

        assert r1["shared"]["v"].item() == 100
        assert r2["shared"]["v"].item() == 200

        tq.kv_clear(partition_id="ns_1")
        tq.kv_clear(partition_id="ns_2")
