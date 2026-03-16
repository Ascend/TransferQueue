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

import asyncio
import logging
import os
import random
import sys
import time
import uuid
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

import ray
import torch
from omegaconf import OmegaConf
from tensordict import NonTensorData, TensorDict

import transfer_queue as tq
from transfer_queue import BatchMeta, KVBatchMeta

TQ_INITIALIZED = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_DEBUG"] = "1"
ray.init()


async def async_kv_batch_meta2batch_meta(meta: KVBatchMeta) -> BatchMeta:
    global TQ_INITIALIZED
    if not TQ_INITIALIZED:
        tq.init()
        TQ_INITIALIZED = True
    tq_client = tq.get_client()
    batch_meta = await tq_client.async_kv_retrieve_meta(keys=meta.keys, partition_id=meta.partition_id, create=False)
    fields = meta.fields
    if fields is not None:
        if isinstance(fields, str):
            fields = [fields]
        batch_meta = batch_meta.select_fields(fields)

    batch_meta.extra_info = meta.extra_info
    return batch_meta


async def async_batch_meta2kv_batch_meta(meta: BatchMeta) -> KVBatchMeta:
    global TQ_INITIALIZED
    if not TQ_INITIALIZED:
        tq.init()
        TQ_INITIALIZED = True
    tq_client = tq.get_client()
    partition_id = meta.partition_ids[0]
    assert all([partition_id == pid for pid in meta.partition_ids])
    keys = await tq_client.async_kv_retrieve_keys(global_indexes=meta.global_indexes, partition_id=partition_id)

    kv_batch_meta = KVBatchMeta(
        keys=keys,
        tags=[{}] * meta.size,
        partition_id=partition_id,
        fields=meta.field_names,
        extra_info=meta.extra_info,
    )
    return kv_batch_meta


def generate_sequences(data):
    time.sleep(3)
    return data


def compute_old_log_prob(data1, _data2):
    time.sleep(3)
    return data1


def actor_rollout_wg_generate_sequences(kv_meta: KVBatchMeta) -> KVBatchMeta:
    # 1. Convert KVBatchMeta -> BatchMeta
    batch_meta = asyncio.run(async_kv_batch_meta2batch_meta(kv_meta))

    # 2. Pull real data from the storage plane through client based on batch_meta
    tq_client = tq.get_client()
    data = tq_client.get_data(batch_meta)
    logger.info(f"demo get data {data}")

    # 3. Get generate results
    output = generate_sequences(data["input_ids"])

    output = TensorDict(
        {
            "generate_sequences_ids": output,
            "non_tensor_data": torch.stack([NonTensorData("test_str") for _ in range(output.size(0))]),
            "nested_tensor": torch.nested.as_nested_tensor(
                [torch.randn(1, 2) for _ in range(output.size(0))], layout=torch.jagged
            ),
        },
        batch_size=output.size(0),
    )

    # 4. Write results back to the storage plane based on batch_meta
    tq_client.put(data=output, metadata=batch_meta)
    logger.info("demo put data to storages done")

    # 5. Convert BatchMeta -> KVBatchMeta and return for further usage
    batch_meta.add_fields(output)
    kv_meta = asyncio.run(async_batch_meta2kv_batch_meta(batch_meta))
    return kv_meta

def actor_rollout_wg_compute_old_log_prob(kv_meta: KVBatchMeta) -> KVBatchMeta:
    # 1. Convert KVBatchMeta -> BatchMeta
    batch_meta = asyncio.run(async_kv_batch_meta2batch_meta(kv_meta))

    # 2. Pull real data from the storage plane through client based on batch_meta
    tq_client = tq.get_client()
    data = tq_client.get_data(batch_meta)
    logger.info(f"demo get data {data}")

    # 3. Get generate results
    output = compute_old_log_prob(data["input_ids"], data["generate_sequences_ids"])

    output = TensorDict({"old_log_prob": output}, batch_size=output.size(0))

    # 4. Write results back to the storage plane based on batch_meta
    tq_client.put(data=output, metadata=batch_meta)
    logger.info("demo put data to storages done")

    # 5. Convert BatchMeta -> KVBatchMeta and return for further usage
    batch_meta.add_fields(output)
    kv_meta = asyncio.run(async_batch_meta2kv_batch_meta(batch_meta))
    return kv_meta


# Simulate the fit function of the trainer
def fit(config):
    for _epoch in range(1):
        train_dataloader = 1
        for step in range(train_dataloader):
            # ============================== Construct prompt batch data ==============================
            input_ids = (torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [100, 111]])) * (step + 1)
            input_ids_repeated = torch.repeat_interleave(input_ids, config.num_n_samples, dim=0)
            batch_keys = [str(uuid.uuid4()) for _ in range(len(input_ids_repeated))]
            prompt_batch = TensorDict(
                {
                    "input_ids": input_ids_repeated,
                    "attention_mask": input_ids_repeated
                 },
                batch_size=input_ids_repeated.size(0),
            )

            # ============================== Put prompts to TQ system ==============================
            tq.kv_batch_put(keys=batch_keys, partition_id=f"train_{step}", fields=prompt_batch)
            logger.info("demo put prompts ok! ")
            time.sleep(5)

            # ============================== Sample generate KVBatchMeta ==============================
            sampled_keys = random.sample(batch_keys, config.global_batch_size)
            gen_meta = KVBatchMeta(
                keys=sampled_keys,
                tags=[{} for _ in sampled_keys],
                partition_id=f"train_{step}",
                fields=["input_ids", "attention_mask"]
            )
            logger.info(f"demo get gen KVBatchMeta {gen_meta}")

            # ============================== Simulate generate sequences task ==============================
            gen_meta = actor_rollout_wg_generate_sequences(gen_meta)
            logger.info(f"demo get after gen KVBatchMeta {gen_meta}")

            # ============================== Create old log prob KVBatchMeta ==============================
            old_log_prob_meta = KVBatchMeta(
                keys=sampled_keys,
                tags=[{} for _ in sampled_keys],
                partition_id=f"train_{step}",
                fields=["input_ids", "attention_mask", "generate_sequences_ids"]
            )

            logger.info(f"demo get old log prob kv_batch_meta: {old_log_prob_meta}")

            # ============================== Simulate compute old log prob task ==============================
            old_log_prob_meta = actor_rollout_wg_compute_old_log_prob(old_log_prob_meta)
            logger.info(f"demo get after old log prob kv_batch_meta: {old_log_prob_meta}")

            # ============================== clear partition in TQ ==============================
            # For the master client, notify all controllers to clear data status, master returns metadata;
            # Client then notifies the storage plane to clear based on metadata
            # Client selects one master controller to get metadata,
            # other controllers directly clear without returning metadata
            tq_client = tq.get_client()
            tq_client.clear_partition(partition_id=f"train_{step}")
            logger.info("clear ok! ")
    logger.info("demo done!")


def main(config):
    # Initialize TransferQueue
    tq.init(conf=config)

    time.sleep(5)

    # Start training loop
    fit(config)

    # Cleanup resources
    data_system_client.close()


if __name__ == "__main__":
    config_str = """
      global_batch_size: 6
      num_global_batch: 1
      num_data_storage_units: 2
      num_n_samples: 2
    """
    dict_conf = OmegaConf.create(config_str)

    main(dict_conf)

    ray.shutdown()
