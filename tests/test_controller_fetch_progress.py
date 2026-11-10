import asyncio
import time
from uuid import uuid4

import pytest
import ray
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

import transfer_queue as tq
import transfer_queue.client as client_module


@pytest.mark.asyncio
async def test_waiting_fetch_allows_put_and_notification(monkeypatch):
    namespace = f"tq-fetch-progress-{uuid4().hex}"
    monkeypatch.setenv("TQ_RAY_NAMESPACE", namespace)
    ray.init(namespace=namespace, num_cpus=3, include_dashboard=False)
    try:
        tq.init(
            OmegaConf.create(
                {
                    "controller": {"polling_mode": False},
                    "backend": {"SimpleStorage": {"num_data_storage_units": 1}},
                }
            )
        )
        client = tq.get_client()
        client.controller_rpc_pool._maxsize = 1
        monkeypatch.setattr(client_module, "TQ_CONTROLLER_GET_METADATA_TIMEOUT", 4)

        reader = asyncio.create_task(
            asyncio.to_thread(
                client.get_meta,
                data_fields=["input_ids"],
                batch_size=1,
                partition_id="waiting_fetch",
                task_name="trainer",
            )
        )
        await asyncio.sleep(0.2)
        data = TensorDict({"input_ids": torch.tensor([[1, 2]])}, batch_size=[1])
        await asyncio.wait_for(asyncio.to_thread(client.put, data, partition_id="waiting_fetch"), timeout=3)
        metadata = await asyncio.wait_for(reader, timeout=5)
        result = await asyncio.to_thread(client.get_data, metadata)
        assert metadata.is_ready
        assert torch.equal(result["input_ids"][0], data["input_ids"][0])

        monkeypatch.setattr(client_module, "TQ_CONTROLLER_GET_METADATA_TIMEOUT", 0.25)
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="Timeout while waiting for sufficient data"):
            await asyncio.to_thread(
                client.get_meta,
                data_fields=["input_ids"],
                batch_size=1,
                partition_id="never_produced",
                task_name="trainer",
            )
        assert time.monotonic() - started < 2
    finally:
        tq.close()
        ray.shutdown()


@pytest.mark.asyncio
async def test_polling_mode_returns_empty_immediately(monkeypatch):
    namespace = f"tq-polling-{uuid4().hex}"
    monkeypatch.setenv("TQ_RAY_NAMESPACE", namespace)
    ray.init(namespace=namespace, num_cpus=3, include_dashboard=False)
    try:
        tq.init(
            OmegaConf.create(
                {
                    "controller": {"polling_mode": True},
                    "backend": {"SimpleStorage": {"num_data_storage_units": 1}},
                }
            )
        )
        metadata = await asyncio.wait_for(
            asyncio.to_thread(
                tq.get_client().get_meta,
                data_fields=["input_ids"],
                batch_size=1,
                partition_id="empty",
                task_name="trainer",
            ),
            timeout=2,
        )
        assert metadata.size == 0
    finally:
        tq.close()
        ray.shutdown()
