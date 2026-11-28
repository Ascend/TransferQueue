# Copyright 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2026 The TransferQueue Team
# SPDX-License-Identifier: Apache-2.0

"""Runtime ownership across local clients and attached Ray workers."""

from urllib.request import ProxyHandler, build_opener
from uuid import uuid4

import pytest
import ray
import torch
from omegaconf import OmegaConf

import transfer_queue as tq


@pytest.fixture(scope="module")
def ray_cluster():
    started_here = not ray.is_initialized()
    if started_here:
        ray.init(address="local", num_cpus=4, include_dashboard=False)
    yield
    if started_here:
        ray.shutdown()


@pytest.fixture
def runtime(ray_cluster, monkeypatch):
    namespace = f"tq-lifecycle-{uuid4().hex}"
    monkeypatch.setenv("TQ_RAY_NAMESPACE", namespace)
    monkeypatch.delenv("TQ_ENABLE_KV_GLOBAL_LOCK", raising=False)
    conf = OmegaConf.create(
        {
            "backend": {"SimpleStorage": {"num_data_storage_units": 1, "total_storage_size": 32}},
            "metrics": {"enabled": True},
        }
    )
    try:
        yield tq.init(conf), namespace
    finally:
        tq.close()


@ray.remote(num_cpus=0, max_calls=1)
def _read_write_and_reconnect(explicit_init):
    try:
        if explicit_init:
            assert tq.init() is None
        value = tq.kv_batch_get("source", "lifecycle")["value"][0]
        client = tq.get_client()
        assert tq.init() is None
        assert tq.get_client() is client
        tq.kv_put("result", "lifecycle", fields={"value": value + 1})
        endpoint = tq.get_metrics_endpoint()
        tq.close()
        tq.close()

        result = tq.kv_batch_get("result", "lifecycle")["value"]
        assert tq.get_client() is not client
        return result, endpoint
    finally:
        tq.close()


@pytest.mark.parametrize("explicit_init", [True, False], ids=["explicit-init", "lazy-attach"])
def test_attached_close_preserves_data_and_allows_reconnect(runtime, explicit_init):
    conf, namespace = runtime
    owner = tq.get_client()
    value = torch.tensor([2, 4, 6])
    tq.kv_put("source", "lifecycle", fields={"value": value})

    worker = _read_write_and_reconnect.options(runtime_env={"env_vars": {"TQ_RAY_NAMESPACE": namespace}})
    result, endpoint = ray.get(worker.remote(explicit_init), timeout=60)
    assert len(result) == 1
    torch.testing.assert_close(result[0], value + 1)
    assert endpoint == conf.metrics.endpoint
    assert tq.get_client() is owner
    torch.testing.assert_close(tq.kv_batch_get("source", "lifecycle")["value"][0], value)
    torch.testing.assert_close(tq.kv_batch_get("result", "lifecycle")["value"][0], result[0])

    with build_opener(ProxyHandler({})).open(f"http://{endpoint}/metrics", timeout=5) as response:
        assert "tq_controller_request_total" in response.read().decode()


def test_owner_close_allows_fresh_init(runtime):
    conf, namespace = runtime
    owner = tq.get_client()
    controller = ray.get_actor("TransferQueueController", namespace=namespace)
    storage = ray.get_actor("TransferQueueStorageUnit#0")
    tq.kv_put("previous", "lifecycle", fields={"value": torch.tensor([1])})
    assert tq.init(conf) is conf
    assert tq.get_client() is owner

    tq.close()
    tq.close()
    with pytest.raises(ray.exceptions.RayActorError):
        ray.get(controller.get_config.remote(), timeout=10)
    with pytest.raises(ray.exceptions.RayActorError):
        ray.get(storage.get_bootstrap_info.remote(), timeout=10)

    tq.init(conf)
    assert tq.get_client() is not owner
    assert tq.kv_list() == {}
    value = torch.tensor([7, 8])
    tq.kv_put("fresh", "lifecycle", fields={"value": value})
    torch.testing.assert_close(tq.kv_batch_get("fresh", "lifecycle")["value"][0], value)
