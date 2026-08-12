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

import os
from contextlib import contextmanager

import psutil
import ray
import torch
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from transfer_queue.utils.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_TORCH_NUM_THREADS = torch.get_num_threads()


def get_placement_group(num_ray_actors: int, num_cpus_per_actor: int = 1):
    """
    Create a placement group with SPREAD strategy for Ray actors.

    Args:
        num_ray_actors (int): Number of Ray actors to create.
        num_cpus_per_actor (int): Number of CPUs to allocate per actor.

    Returns:
        placement_group: The created placement group.
    """
    bundle = {"CPU": num_cpus_per_actor}
    placement_group = ray.util.placement_group([bundle for _ in range(num_ray_actors)], strategy="SPREAD")
    ray.get(placement_group.ready())
    return placement_group


def get_node_round_robin_scheduling_strategies(
    num_actors: int, required_node_resource: str | None = None
) -> list[NodeAffinitySchedulingStrategy]:
    """
    Compute one scheduling strategy per actor that round-robins actors across
    eligible alive Ray nodes, in order.

    Unlike a placement group with SPREAD (best-effort) or STRICT_SPREAD (fails when
    num_actors > num_nodes), this guarantees each node is assigned floor(num_actors /
    num_nodes) or ceil(num_actors / num_nodes) actors, regardless of how num_actors
    compares to the number of nodes.

    Args:
        num_actors (int): Number of Ray actors to schedule.
        required_node_resource (str | None): Optional Ray custom resource required on eligible nodes.

    Returns:
        list[NodeAffinitySchedulingStrategy]: One scheduling strategy per actor.
    """
    nodes = ray.nodes()
    alive_node_ids = sorted(
        node["NodeID"]
        for node in nodes
        if node.get("Alive", False)
        and (required_node_resource is None or node.get("Resources", {}).get(required_node_resource, 0) > 0)
    )
    if not alive_node_ids:
        if required_node_resource is not None:
            raise ValueError(
                f"No alive Ray nodes provide custom resource {required_node_resource!r}. "
                "Start eligible nodes with a positive resource capacity or unset "
                "backend.SimpleStorage.required_node_resource."
            )
        raise RuntimeError("No alive Ray nodes found. Is Ray initialized?")

    return [
        NodeAffinitySchedulingStrategy(node_id=alive_node_ids[i % len(alive_node_ids)], soft=False)
        for i in range(num_actors)
    ]


@contextmanager
def limit_pytorch_auto_parallel_threads(target_num_threads: int | None = None, info: str = ""):
    """Prevent PyTorch from overdoing the automatic parallelism during tensor aggregation operations."""
    pytorch_current_num_threads = torch.get_num_threads()
    physical_cores = psutil.cpu_count(logical=False)
    pid = os.getpid()
    if target_num_threads is None:
        # auto determine target_num_threads
        if physical_cores >= 16:
            target_num_threads = 16
        else:
            target_num_threads = physical_cores

    if target_num_threads > physical_cores:
        logger.warning(
            f"target_num_threads {target_num_threads} should not exceed total "
            f"physical CPU cores {physical_cores}. Setting to {physical_cores}."
        )
        target_num_threads = physical_cores

    try:
        torch.set_num_threads(target_num_threads)
        logger.debug(
            f"{info} (pid={pid}): torch.get_num_threads() is {pytorch_current_num_threads}, "
            f"setting to {target_num_threads}."
        )
        yield
    finally:
        # Restore the original number of threads
        torch.set_num_threads(DEFAULT_TORCH_NUM_THREADS)
        logger.debug(
            f"{info} (pid={pid}): torch.get_num_threads() is {torch.get_num_threads()}, "
            f"restoring to {DEFAULT_TORCH_NUM_THREADS}."
        )


def get_env_bool(env_key: str, default: bool = False) -> bool:
    """Robustly get a boolean from an environment variable."""
    env_value = os.getenv(env_key)

    if env_value is None:
        return default

    env_value_lower = env_value.strip().lower()

    true_values = {"true", "1", "yes", "y", "on"}
    return env_value_lower in true_values
