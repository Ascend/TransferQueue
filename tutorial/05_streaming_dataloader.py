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
Tutorial 5: Streaming DataLoader for Distributed Training

This script demonstrates how to use StreamingDataset and StreamingDataLoader
for efficient streaming data loading in distributed training scenarios.

Key Components:
- StreamingDataset: PyTorch IterableDataset that integrates with TransferQueue
- StreamingDataLoader: DataLoader wrapper that yields (batch, batch_meta) tuples
- RankAwareSampler: Enables DP group coordination for consistent
  sampling across multiple ranks

Use Cases:
- Distributed training with multiple DP groups
- Fine-grained micro-batch-level data retrieval
"""

import os
import sys
import textwrap
import time
import warnings
from pathlib import Path

os.environ["RAY_DEDUP_LOGS"] = "0"

warnings.filterwarnings(
    action="ignore",
    message=r"The PyTorch API of nested tensors is in prototype stage*",
    category=UserWarning,
    module=r"torch\.nested",
)

warnings.filterwarnings(
    action="ignore",
    message=r"Tip: In future versions of Ray, Ray will no longer override accelerator visible "
    r"devices env var if num_gpus=0 or num_gpus=None.*",
    category=FutureWarning,
    module=r"ray\._private\.worker",
)


import ray  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from tensordict import TensorDict  # noqa: E402

# Add the parent directory to the path for imports
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))


from transfer_queue import (  # noqa: E402
    RankAwareSampler,
    SimpleStorageUnit,
    StreamingDataLoader,
    StreamingDataset,
    TransferQueueClient,
    TransferQueueController,
    process_zmq_server_info,
)


def setup_transfer_queue():
    """Setup TransferQueue components."""
    if not ray.is_initialized():
        ray.init()

    config = OmegaConf.create(
        {
            "num_data_storage_units": 2,
        }
    )

    storage_units = {}
    for i in range(config["num_data_storage_units"]):
        storage_units[i] = SimpleStorageUnit.remote(storage_unit_size=100)

    print("[Setup]: Setup TransferQueue components")
    print(
        "Note: Using RankAwareSampler when each rank retrieves data independently. It guarantees that "
        "The same DP rank receives the same sample indices."
    )
    print(
        "Note: When using streaming data retrieval, please set polling_mode=True when initializing "
        "TransferQueueController. In polling_mode, the controller will return empty BatchMeta when "
        "available data cannot meet the consumption requirements. User side need to retry later."
    )
    controller = TransferQueueController.remote(
        sampler=RankAwareSampler,  # RankAwareSampler enables consistent sampling for each DP rank
        polling_mode=True,  # Enable polling mode for streaming data retrieval
    )

    controller_info = process_zmq_server_info(controller)
    storage_unit_infos = process_zmq_server_info(storage_units)

    # Build the complete configuration
    tq_config = OmegaConf.create({}, flags={"allow_objects": True})
    tq_config.controller_info = controller_info
    tq_config.storage_unit_infos = storage_unit_infos
    config.storage_backend = "AsyncSimpleStorageManager"
    config = OmegaConf.merge(tq_config, config)

    return controller, storage_units, config


@ray.remote(num_cpus=0.1)
def generate_worker(rank_id: int, config: DictConfig, num_samples: int = 20):
    """
    Generate actor that produces training samples.

    This actor simulates a data producer that generates training samples
    and puts them into the TransferQueue for consumption by training actors.

    Args:
        rank_id: Unique identifier for this generator (used for sample indexing)
        config: TransferQueue configuration
        num_samples: Number of samples to generate

    Note:
        Each sample has a unique sequence ID calculated as: seq_id = i + (rank_id * 10000)
        This ensures global uniqueness across all generator actors.
    """
    # Create a client for interacting with TransferQueue
    client = TransferQueueClient(
        client_id=f"gen_worker_{rank_id}",
        controller_info=config.controller_info,
    )

    # Initialize the storage manager for this client
    client.initialize_storage_manager(manager_type=config.storage_backend, config=config)

    # Generate and put samples into the queue
    for i in range(num_samples):
        # Create unique sequence ID for this sample
        seq_id = i + (rank_id * 10000)

        # Create sample data as TensorDict
        data = TensorDict(
            {"input_ids": torch.full((1, 16), seq_id, dtype=torch.long), "meta_idx": torch.tensor([seq_id])},
            batch_size=1,
        )

        print(f"[Generate Worker@{rank_id}]: Putting sample {seq_id} into TransferQueue")

        # Put data into the specified partition
        client.put(data, partition_id="train")

    print(f"[Generate Worker@{rank_id}]: Complete putting samples into TransferQueue")


@ray.remote(num_cpus=0.1)
def update_worker(
    rank_id: int,
    dp_rank: int,
    config: DictConfig,
    max_steps: int = 5,
):
    """
    Update actor that retrieves and processes training batches.

    This actor simulates a training worker that consumes data from the
    TransferQueue using StreamingDataLoader. It demonstrates how to use
    the streaming data loading infrastructure in a distributed setting.

    Args:
        rank_id: Global rank identifier for logging and display purposes
        dp_rank: Data parallel rank ID that this worker belongs to
            The same Ranks receive the same data samples
        config: TransferQueue configuration
        max_steps: Maximum number of batches to consume

    Returns:
        dict: Contains dp_rank and consumed_ids

    Example:
        For a setup with 2 data rank (0 and 1):
        - Rank 0: receive identical samples
        - Rank 1: receive identical samples
        All ranks within the same rank index get the same global indexes.

    Note:
        The StreamingDataLoader yields tuples of (batch, batch_meta) where:
        - batch: TensorDict containing the requested data fields
        - batch_meta: Metadata for TransferQueue coordination (contains global_indexes)
    """

    # Step 1: Create StreamingDataset
    # This dataset integrates with TransferQueue and handles batch retrieval
    dataset = StreamingDataset(
        config=config,
        batch_size=2,
        micro_batch_size=2,  # Number of samples per micro-batch.
        data_fields=["meta_idx"],  # Fields to retrieve from storage. We can retrieve partial fields!
        partition_id="train",  # Data partition to consume from
        task_name="update_task",  # Unique task identifier
        dp_rank=dp_rank,
    )
    print(f"[Update Worker@{rank_id}] StreamingDataset created successfully")

    # Step 2: Create StreamingDataLoader
    # Wraps the dataset and provides PyTorch DataLoader-compatible interface
    dataloader = StreamingDataLoader(
        dataset=dataset,
        num_workers=2,  # We can enable parallel data retrieval and data pre-fetching!
        prefetch_factor=2,
    )
    print(
        f"[Update Worker@{rank_id}] StreamingDataLoader ready, enabling data pre-fetching through num_workers "
        f"and prefetch_factor."
    )

    # Step 3: Consume data batches
    print(f"[Update Worker@{rank_id}] Starting data consumption...")
    consumed_ids = []
    step = 0

    for batch, batch_meta in dataloader:
        # Extract sample IDs from the batch
        ids = batch["meta_idx"].view(-1).tolist()

        print(f"[Update Worker@{rank_id}]: dp_rank {dp_rank} retrieved samples: {ids}")
        consumed_ids.extend(ids)

        # Simulate processing time (remove in real training)
        time.sleep(5)

        step += 1
        if step >= max_steps:
            print(f"[Update Worker@{rank_id}] Reached max steps ({max_steps}), stopping...")
            break

    print(f"[Update Worker@{rank_id}] Completed {step} steps, consumed {len(consumed_ids)} samples")

    return {
        "dp_rank": dp_rank,
        "consumed_ids": consumed_ids,
    }


def start_all_generate_actors(config):
    """
    Launch generate_actors for producing training samples.
    """
    num_workers = 2
    handlers = []

    for i in range(num_workers):
        handlers.append(generate_worker.remote(rank_id=i, config=config, num_samples=20))

    return handlers


def start_all_update_actors(config):
    """
    Launch update_actors for consuming training samples.
    """

    # Define the distributed training topology
    rank_ids = [0, 1, 2, 3]
    dp_rank = [0, 0, 1, 1]  # First two ranks in group 0, last two in group 1

    print("Training topology configuration:")
    print(f"  - Total ranks: {len(rank_ids)}")
    print(f"  - Data parallel rank: {len(set(dp_rank))}")

    handlers = []
    for i in range(len(rank_ids)):
        handlers.append(
            update_worker.remote(
                rank_id=rank_ids[i],
                dp_rank=dp_rank[i],
                config=config,
            )
        )

    return handlers


def main():
    """
    Main function demonstrating end-to-end streaming data loading.

    This tutorial showcases:
    1. Setting up TransferQueue with streaming capabilities
    2. Launching data generation actors
    3. Launching data consumption actors with distributed training topology
    4. Verifying that ranks in the same group receive identical samples
    """

    print("=" * 80)
    print(
        textwrap.dedent(
            """
        TransferQueue Tutorial 5: StreamingDataLoader for Distributed Training

        This tutorial demonstrates the StreamingDataLoader interface for distributed
        training scenarios. It showcases how to use StreamingDataset and StreamingDataLoader
        to efficiently consume micro-batch of samples from TransferQueue with proper coordination 
        across multiple training ranks.

        Key Concepts:
        - StreamingDataset: PyTorch IterableDataset that integrates with TransferQueue
        - StreamingDataLoader: DataLoader wrapper yielding (batch, batch_meta) tuples
        - RankAwareSampler: Enables correct data consumption across DP ranks
        - DP Rank: Ranks that should receive identical data samples
        """
        )
    )
    print("=" * 80)

    # Step 1: Setup TransferQueue infrastructure
    print("\n[Phase 1] Setting up TransferQueue infrastructure...")
    print(
        "\nIn real-world usage, please export the environment variable of TQ_PRE_ALLOC_SAMPLE_NUM to "
        "global_batch_size to make sure consumers can accurately determine consumption status even before "
        "producers have generated the samples."
    )
    controller, storage_units, config = setup_transfer_queue()

    # Step 2: Launch data generation actors
    print("\n[Phase 2] Starting data generation...")
    generate_worker_handlers = start_all_generate_actors(config)

    # Step 3: Launch data consumption actors
    print("\n[Phase 3] Starting data consumption...")
    update_worker_handlers = start_all_update_actors(config)

    # Wait for completion
    print("\n[Phase 4] Waiting for actors to complete...")
    print("=" * 80)

    # Wait for generation to complete
    ray.get(generate_worker_handlers)
    print("✓ All generation actors completed")

    # Wait for consumption to complete
    update_results = ray.get(update_worker_handlers)
    print("✓ All update actors completed")

    # Display results summary
    print("\n" + "=" * 80)
    print("Results Summary")
    print("=" * 80)
    for result in update_results:
        print(f"  DP Rank {result['dp_rank']}: consumed {len(result['consumed_ids'])} samples")

    print("\n" + "=" * 80)
    print("Tutorial Complete!")
    print("=" * 80)
    print("Key Takeaways:")
    print("1. StreamingDataset provides PyTorch IterableDataset interface for TransferQueue")
    print("2. StreamingDataLoader wraps the dataset and yields (batch, batch_meta) tuples")
    print("3. Ranks with the same DP rank receive identical samples")
    print("4. The system enables efficient streaming capabilities")


if __name__ == "__main__":
    main()
