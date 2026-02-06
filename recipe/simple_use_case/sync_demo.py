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

import logging
import math
import os
import sys
import time
from pathlib import Path

import ray
import torch
from omegaconf import OmegaConf
from tensordict import NonTensorData, TensorDict

parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

from transfer_queue import (  # noqa: E402
    SimpleStorageUnit,
    TransferQueueClient,
    TransferQueueController,
    process_zmq_server_info,
)
from transfer_queue.utils.common import get_placement_group  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_DEBUG"] = "1"
ray.init()


def initialize_data_system(config):
    # 1. Initialize TransferQueueStorage
    total_storage_size = config.global_batch_size * config.num_global_batch * config.num_n_samples
    data_system_storage_units = {}
    storage_placement_group = get_placement_group(config.num_data_storage_units, num_cpus_per_actor=1)
    for storage_unit_rank in range(config.num_data_storage_units):
        storage_node = SimpleStorageUnit.options(
            placement_group=storage_placement_group, placement_group_bundle_index=storage_unit_rank
        ).remote(storage_unit_size=math.ceil(total_storage_size / config.num_data_storage_units))
        data_system_storage_units[storage_unit_rank] = storage_node
        logger.info(f"SimpleStorageUnit #{storage_unit_rank} has been created.")

    # 2. Initialize TransferQueueController (single controller only)

    # Sampler usage instructions:
    # For GRPO grouped sampling, you can initialize the controller with GRPOGroupNSampler:
    # Option 1: Pass sampler class (will be instantiated automatically)
    # data_system_controller = TransferQueueController.remote(sampler=GRPOGroupNSampler)

    # Option 2: Pass sampler instance (if you need custom configuration)
    # grpo_sampler = GRPOGroupNSampler()
    # data_system_controller = TransferQueueController.remote(sampler=grpo_sampler)

    # Then use sampling_config in get_meta calls:
    data_system_controller = TransferQueueController.remote()
    logger.info("TransferQueueController has been created.")

    # 3. Prepare necessary information
    data_system_controller_info = process_zmq_server_info(data_system_controller)
    data_system_storage_unit_infos = process_zmq_server_info(data_system_storage_units)

    tq_config = OmegaConf.create({}, flags={"allow_objects": True})  # Note: Need to generate a new DictConfig
    # with allow_objects=True to maintain ZMQServerInfo instance. Otherwise it will be flattened to dict
    tq_config.controller_info = data_system_controller_info
    tq_config.storage_unit_infos = data_system_storage_unit_infos
    config = OmegaConf.merge(tq_config, config)

    # 4. Create client
    data_system_client = TransferQueueClient(
        client_id="Trainer",
        controller_info=data_system_controller_info,
    )

    data_system_client.initialize_storage_manager(manager_type="AsyncSimpleStorageManager", config=config)

    return data_system_controller, data_system_storage_units, data_system_client


def generate_sequences(data):
    time.sleep(3)
    return data


def compute_old_log_prob(data1, _data2):
    time.sleep(3)
    return data1


def actor_rollout_wg_generate_sequences(data_meta, data_system_client):
    # 1. Pull real data from the storage plane through client based on data_meta
    data = data_system_client.get_data(data_meta)
    logger.info(f"demo get data {data}")

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

    # 2. Write results back to the storage plane based on data_meta
    data_system_client.put(data=output, metadata=data_meta)
    data_meta.add_fields(output)
    logger.info("demo put data to storages done")

    return data_meta


def actor_rollout_wg_compute_old_log_prob(data_meta, data_system_client):
    # 1. Pull real data from the storage plane through client based on data_meta
    data = data_system_client.get_data(data_meta)
    logger.info(f"demo get data {data}")

    output = compute_old_log_prob(data["input_ids"], data["generate_sequences_ids"])

    output = TensorDict({"old_log_prob": output}, batch_size=output.size(0))

    # 2. Write results back to the storage plane based on data_meta
    data_system_client.put(data=output, metadata=data_meta)
    data_meta.add_fields(output)
    logger.info("demo put data to storages done")

    return data_meta


# Simulate the fit function of the trainer
def fit(config, data_system_client):
    for _epoch in range(1):
        train_dataloader = 1
        for step in range(train_dataloader):
            input_ids = (torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [10, 11], [100, 111]])) * (step + 1)
            input_ids_repeated = torch.repeat_interleave(input_ids, config.num_n_samples, dim=0)
            prompt_batch = TensorDict(
                {"input_ids": input_ids_repeated, "attention_mask": input_ids_repeated},
                batch_size=input_ids_repeated.size(0),
            )

            data_system_client.put(data=prompt_batch, partition_id=f"train_{step}")
            logger.info("demo put prompts ok! ")
            time.sleep(5)

            batch_meta = data_system_client.get_meta(
                data_fields=["input_ids", "attention_mask"],
                batch_size=config.global_batch_size,
                partition_id=f"train_{step}",
                task_name="generate_sequences",
            )
            # Set output fields for RL training - in this case, we want to generate sequences from input_ids
            logger.info(f"demo get meta {batch_meta}")

            # Simulate calling the generate sequences task of the worker group
            batch_meta = actor_rollout_wg_generate_sequences(batch_meta, data_system_client)
            log_prob_meta = data_system_client.get_meta(
                data_fields=["input_ids", "attention_mask", "generate_sequences_ids"],
                batch_size=config.global_batch_size,
                partition_id=f"train_{step}",
                task_name="compute_old_log_prob",
            )
            # Set output fields for RL training - we want to compute log probs for the generated sequences
            logger.info(f"demo get log prob meta: {log_prob_meta}")

            # Simulate calling the compute old log prob task of the worker group
            old_log_prob_meta = actor_rollout_wg_compute_old_log_prob(log_prob_meta, data_system_client)

            batch_meta = batch_meta.union(old_log_prob_meta)

            # For the master client, notify all controllers to clear data status, master returns metadata;
            # Client then notifies the storage plane to clear based on metadata
            # Client selects one master controller to get metadata,
            # other controllers directly clear without returning metadata
            data_system_client.clear_partition(partition_id=f"train_{step}")
            logger.info("clear ok! ")
    logger.info("demo done!")


def main(config):
    # Initialize Data System: Launching the Controller and Storage based on Ray
    _data_system_controller, _data_system_storage_units, data_system_client = initialize_data_system(config)
    import time

    time.sleep(5)

    fit(config, data_system_client)

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
