#!/usr/bin/env python3
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

import argparse
import csv
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

import ray
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

import transfer_queue as tq  # noqa: E402
from transfer_queue.storage.simple_backend import SimpleStorageUnit  # noqa: E402
from transfer_queue.utils.common import get_placement_group  # noqa: E402
from transfer_queue.utils.zmq_utils import process_zmq_server_info  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
NUM_TEST_ITERATIONS = 3


def create_test_case(
    batch_size: int | None = None,
    seq_length: int | None = None,
    field_num: int | None = None,
    device: str = "cpu",
) -> tuple[TensorDict, float]:
    """Create a test case with tensor fields only.

    Args:
        batch_size: Batch size for the test case
        seq_length: Sequence length for tensor fields
        field_num: Number of fields to create
        device: Device to create tensors on ("cpu", "npu", or "gpu")

    Returns:
        Tuple of (TensorDict, total_size_gb)
    """
    tensor_field_size_bytes = batch_size * seq_length * 4
    tensor_field_size_gb = tensor_field_size_bytes / (1024**3)

    total_size_gb = tensor_field_size_gb * field_num

    logger.info(f"Total data size: {total_size_gb:.6f} GB")

    # Determine torch device
    torch_device = None
    if device == "npu":
        torch_device = "npu:0"
    elif device == "gpu":
        torch_device = "cuda:0"

    fields = {}
    for i in range(field_num):
        field_name = f"field_{i}"
        tensor_data = torch.randn(batch_size, seq_length, dtype=torch.float32, device=torch_device)
        fields[field_name] = tensor_data

    batch_size_tuple = (batch_size,)
    prompt_batch = TensorDict(
        fields,
        batch_size=batch_size_tuple,
    )

    return prompt_batch, total_size_gb


@ray.remote
class TQClientActor:
    """Ray actor that uses tq.init(config) to initialize."""

    def __init__(self, base_config: dict[str, Any]):
        self.base_config = base_config
        self.test_data = None
        self.total_data_size_gb = 0.0
        self.test_keys = None

    def initialize(self, zmq_info: Any = None) -> None:
        """Initialize transfer_queue with the config."""
        config = OmegaConf.create(self.base_config, flags={"allow_objects": True})
        if zmq_info is not None and self.base_config["backend"]["storage_backend"] == "SimpleStorage":
            # Use dict-style assignment to avoid OmegaConf validation
            config["backend"]["SimpleStorage"]["zmq_info"] = zmq_info
        tq.init(config)

    def create_test_case(
        self,
        batch_size: int | None = None,
        seq_length: int | None = None,
        field_num: int | None = None,
        device: str = "cpu",
    ) -> tuple[list[str], float]:
        """Create test case on the actor."""
        self.test_data, self.total_data_size_gb = create_test_case(batch_size, seq_length, field_num, device)
        # Create keys for each sample in the batch
        self.test_keys = [f"test_key_{i}" for i in range(batch_size)]
        return list(self.test_data.keys()), self.total_data_size_gb

    def put(self, partition_id: str) -> None:
        """Put data to storage using kv_batch_put."""
        tq.kv_batch_put(keys=self.test_keys, partition_id=partition_id, fields=self.test_data)

    def list_keys(self, partition_id: str) -> list[str]:
        """List keys in a partition using kv_list."""
        partition_info = tq.kv_list(partition_id=partition_id)
        if partition_id in partition_info:
            return list(partition_info[partition_id].keys())
        return []

    def get_data(self, partition_id: str, keys: list[str] | None = None) -> None:
        """Get data from storage using kv_batch_get."""
        if keys is None:
            keys = self.test_keys
        tq.kv_batch_get(keys=keys, partition_id=partition_id)


class TQThroughputTester:
    """Main throughput tester for TransferQueue backends."""

    def __init__(
        self,
        backend: str,
        backend_config: dict[str, Any],
        device: str,
        global_batch_size: int,
        field_num: int,
        seq_len: int,
        num_global_batch: int,
        head_node_ip: str,
        worker_node_ip: str | None = None,
        output_csv: str | None = None,
    ):
        """Initialize the throughput tester.

        Args:
            backend: Backend type ("SimpleStorage", "Yuanrong", "MooncakeStore")
            backend_config: Backend configuration dictionary
            device: Device type ("cpu", "npu", "gpu")
            global_batch_size: Global batch size
            field_num: Number of fields
            seq_len: Sequence length
            num_global_batch: Number of global batches
            head_node_ip: Head node IP address
            worker_node_ip: Worker node IP address (required for Yuanrong inter_node)
            output_csv: Path to output CSV file (optional)
        """
        self.backend = backend
        self.backend_config = backend_config
        self.device = device
        self.global_batch_size = global_batch_size
        self.field_num = field_num
        self.seq_len = seq_len
        self.num_global_batch = num_global_batch
        self.head_node_ip = head_node_ip
        self.worker_node_ip = worker_node_ip
        self.output_csv = output_csv

        # Get client_placement from Yuanrong config, default to inter_node
        self.client_placement = (
            self.backend_config.get("client_placement", "inter_node") if self.backend == "Yuanrong" else "intra_node"
        )

        # Validate arguments
        self._validate_args()

        # Prepare full config for tq.init()
        self.base_config, self.zmq_info = self._prepare_configs()

        # Initialize the test infrastructure
        self._initialize_data_system()
        self._initialize_clients()

    def _validate_args(self) -> None:
        """Validate input arguments."""
        # Check worker_node_ip for Yuanrong inter_node
        if self.backend == "Yuanrong" and self.client_placement == "inter_node" and self.worker_node_ip is None:
            raise ValueError("worker_node_ip is required for Yuanrong with client_placement=inter_node")

    def _prepare_configs(self) -> tuple[dict[str, Any], Any]:
        """Prepare the base config and storage units.

        Returns:
            Tuple of (base_config, zmq_info)
        """
        total_storage_size = self.global_batch_size * self.num_global_batch

        config = {
            "controller": {
                "sampler": "SequentialSampler",
                "polling_mode": False,
            },
            "backend": {
                "storage_backend": self.backend,
            },
        }

        # Set client_name based on backend
        if self.backend == "Yuanrong":
            self.backend_config["client_name"] = "YuanrongStorageClient"
        elif self.backend == "MooncakeStore":
            self.backend_config["client_name"] = "MooncakeStoreClient"

        # Add backend-specific config
        if self.backend == "SimpleStorage":
            config["backend"]["SimpleStorage"] = {
                "total_storage_size": total_storage_size,
                "num_data_storage_units": self.backend_config.get("num_data_storage_units", 1),
            }
        elif self.backend == "Yuanrong":
            config["backend"]["Yuanrong"] = self.backend_config.copy()
            # Remove client_placement from the backend config passed to tq
            if "client_placement" in config["backend"]["Yuanrong"]:
                del config["backend"]["Yuanrong"]["client_placement"]
        elif self.backend == "MooncakeStore":
            config["backend"]["MooncakeStore"] = self.backend_config.copy()

        return config, None

    def _initialize_data_system(self) -> None:
        """Initialize controller and storage units if needed."""
        # For SimpleStorage, we need to manually create storage units with placement
        if self.backend == "SimpleStorage":
            self._initialize_storage_units()

    def _initialize_storage_units(self) -> None:
        """Initialize SimpleStorageUnits for SimpleStorage backend."""
        num_data_storage_units = self.backend_config.get("num_data_storage_units", 1)
        total_storage_size = self.global_batch_size * self.num_global_batch

        self.data_system_storage_units = {}

        storage_placement_group = get_placement_group(num_data_storage_units, num_cpus_per_actor=0.001)
        for storage_unit_rank in range(num_data_storage_units):
            storage_node = SimpleStorageUnit.options(
                placement_group=storage_placement_group,
                placement_group_bundle_index=storage_unit_rank,
            ).remote(storage_unit_size=NUM_TEST_ITERATIONS * math.ceil(total_storage_size / num_data_storage_units))
            self.data_system_storage_units[storage_unit_rank] = storage_node
        logger.info(f"StorageUnit #0 ~ #{num_data_storage_units - 1} has been created.")

        self.zmq_info = process_zmq_server_info(self.data_system_storage_units)

    def _initialize_clients(self) -> None:
        """Initialize writer and reader TQClientActors."""
        # Determine node placement
        if self.client_placement == "intra_node":
            writer_node = reader_node = self.head_node_ip
        else:
            writer_node = self.head_node_ip
            reader_node = self.worker_node_ip

        logger.info(f"Writer is on {writer_node}, Reader is on {reader_node}")

        # Prepare base options
        writer_options = {
            "num_cpus": 0.001,
            "resources": {f"node:{writer_node}": 0.001},
        }
        reader_options = {
            "num_cpus": 0.001,
            "resources": {f"node:{reader_node}": 0.001},
        }

        # Add device-specific options
        if self.device == "gpu":
            writer_options["num_gpus"] = 1
            reader_options["num_gpus"] = 1
        elif self.device == "npu":
            writer_options["resources"]["NPU"] = 1
            reader_options["resources"]["NPU"] = 1

        # Create writer and reader actors
        self.writer = TQClientActor.options(**writer_options).remote(self.base_config)
        self.reader = TQClientActor.options(**reader_options).remote(self.base_config)

        # Initialize transfer_queue
        logger.info(f"Using {self.backend} as storage backend.")

        w = self.writer.initialize.remote(self.zmq_info)
        r = self.reader.initialize.remote(self.zmq_info)
        ray.get([w, r])

    def run_throughput_test(self) -> dict[str, Any]:
        """Run the throughput test and print results.

        Returns:
            Dictionary with test results
        """
        logger.info("Creating large batch for throughput test...")
        start_create_data = time.perf_counter()
        data_fields, total_data_size_gb = ray.get(
            self.writer.create_test_case.remote(
                batch_size=self.global_batch_size,
                seq_length=self.seq_len,
                field_num=self.field_num,
                device=self.device,
            )
        )
        end_create_data = time.perf_counter()
        logger.info(f"Data creation time: {end_create_data - start_create_data:.8f}s")

        partition_id = "train_0"

        # PUT operation using kv_batch_put
        logger.info("Starting PUT operation (kv_batch_put)...")
        start_put = time.perf_counter()
        ray.get(self.writer.put.remote(partition_id=partition_id))
        end_put = time.perf_counter()
        put_time = end_put - start_put
        put_gbit_per_sec = (total_data_size_gb * 8) / put_time
        put_gbyte_per_sec = total_data_size_gb / put_time
        logger.info(f"put cost time: {put_time:.8f}s")
        logger.info(f"PUT Throughput: {put_gbit_per_sec:.8f} Gb/s ({put_gbyte_per_sec:.8f} GB/s)")

        time.sleep(2)

        # LIST_KEYS operation using kv_list
        logger.info("Starting LIST_KEYS operation (kv_list)...")
        start_list = time.perf_counter()
        keys = ray.get(self.reader.list_keys.remote(partition_id=partition_id))
        end_list = time.perf_counter()
        logger.info(f"list_keys cost time: {end_list - start_list:.8f}s")
        logger.info(f"Found {len(keys)} keys")

        time.sleep(2)

        # GET_DATA operation using kv_batch_get
        logger.info("Starting GET_DATA operation (kv_batch_get)...")
        start_get_data = time.perf_counter()
        ray.get(self.reader.get_data.remote(partition_id=partition_id, keys=keys))
        end_get_data = time.perf_counter()
        get_time = end_get_data - start_get_data
        get_gbit_per_sec = (total_data_size_gb * 8) / get_time
        get_gbyte_per_sec = total_data_size_gb / get_time

        logger.info(f"get_data cost time: {get_time:.8f}s")
        logger.info(f"GET Throughput: {get_gbit_per_sec:.8f} Gb/s ({get_gbyte_per_sec:.8f} GB/s)")

        # Print summary
        total_gbit_per_sec = (total_data_size_gb * 16) / (put_time + get_time)
        total_gbyte_per_sec = (total_data_size_gb * 2) / (put_time + get_time)

        logger.info("=" * 60)
        logger.info("THROUGHPUT TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Backend: {self.backend}")
        logger.info(f"Client Placement: {self.client_placement}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Total Data Size: {total_data_size_gb:.6f} GB")
        logger.info(f"PUT Time: {put_time:.8f}s")
        logger.info(f"GET Time: {get_time:.8f}s")
        logger.info(f"PUT Throughput: {put_gbit_per_sec:.8f} Gb/s ({put_gbyte_per_sec:.8f} GB/s)")
        logger.info(f"GET Throughput: {get_gbit_per_sec:.8f} Gb/s ({get_gbyte_per_sec:.8f} GB/s)")
        logger.info(f"Total Throughput: {total_gbit_per_sec:.8f} Gb/s ({total_gbyte_per_sec:.8f} GB/s)")
        logger.info("=" * 60)

        # Return results
        return {
            "backend": self.backend,
            "client_placement": self.client_placement,
            "device": self.device,
            "total_data_size_gb": total_data_size_gb,
            "put_time": put_time,
            "get_time": get_time,
            "put_gbit_per_sec": put_gbit_per_sec,
            "put_gbyte_per_sec": put_gbyte_per_sec,
            "get_gbit_per_sec": get_gbit_per_sec,
            "get_gbyte_per_sec": get_gbyte_per_sec,
            "total_gbit_per_sec": total_gbit_per_sec,
            "total_gbyte_per_sec": total_gbyte_per_sec,
        }


def load_backend_config(config_path: str | None, backend: str) -> dict[str, Any]:
    """Load backend config from YAML file or use defaults.

    Args:
        config_path: Path to YAML config file (optional)
        backend: Backend type for default config

    Returns:
        Backend configuration dictionary
    """
    if config_path is not None:
        config = OmegaConf.load(config_path)
        return OmegaConf.to_container(config, resolve=True)

    # Default configs
    if backend == "SimpleStorage":
        return {"num_data_storage_units": 1}
    elif backend == "Yuanrong":
        return {
            "host": "127.0.0.1",
            "port": 31501,
            "enable_yr_npu_transport": False,
            "client_placement": "inter_node",
        }
    elif backend == "MooncakeStore":
        return {
            "local_hostname": "127.0.0.1",
            "metadata_server": "127.0.0.1:8080",
            "master_server_address": "127.0.0.1:8081",
        }
    else:
        return {}


def write_results_to_csv(results: list[dict[str, Any]], output_path: str) -> None:
    """Write test results to CSV file.

    Args:
        results: List of result dictionaries
        output_path: Path to output CSV file
    """
    if not results:
        return

    fieldnames = list(results[0].keys())

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)

    logger.info(f"Results written to {output_path}")


def main() -> None:
    """Main entry point for the perftest script."""
    parser = argparse.ArgumentParser(description="TransferQueue Throughput Test")
    parser.add_argument(
        "--backend",
        type=str,
        default="SimpleStorage",
        choices=["SimpleStorage", "Yuanrong", "MooncakeStore"],
        help="Backend type to test (default: SimpleStorage)",
    )
    parser.add_argument(
        "--backend_config",
        type=str,
        default=None,
        help="Path to backend config YAML file (optional)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "npu", "gpu"],
        help="Device to use (default: cpu)",
    )
    parser.add_argument(
        "--global_batch_size",
        type=int,
        default=1024,
        help="Global batch size (default: 1024)",
    )
    parser.add_argument(
        "--field_num",
        type=int,
        default=10,
        help="Number of fields (default: 10)",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=8192,
        help="Sequence length (default: 8192)",
    )
    parser.add_argument(
        "--num_global_batch",
        type=int,
        default=1,
        help="Number of global batches (default: 1)",
    )
    parser.add_argument(
        "--head_node_ip",
        type=str,
        required=True,
        help="Head node IP address",
    )
    parser.add_argument(
        "--worker_node_ip",
        type=str,
        default=None,
        help="Worker node IP address (required for Yuanrong inter_node)",
    )
    parser.add_argument(
        "--ray_address",
        type=str,
        default="auto",
        help="Ray cluster address (default: auto)",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Path to output CSV file (optional)",
    )

    args = parser.parse_args()

    # Load backend config
    backend_config = load_backend_config(args.backend_config, args.backend)

    # Initialize Ray
    logger.info(f"Connecting to Ray cluster at {args.ray_address}")
    ray.init(address=args.ray_address)

    # Create and run tester
    tester = TQThroughputTester(
        backend=args.backend,
        backend_config=backend_config,
        device=args.device,
        global_batch_size=args.global_batch_size,
        field_num=args.field_num,
        seq_len=args.seq_len,
        num_global_batch=args.num_global_batch,
        head_node_ip=args.head_node_ip,
        worker_node_ip=args.worker_node_ip,
        output_csv=args.output_csv,
    )

    # Run test multiple times for consistent results using a for loop
    all_results = []
    for i in range(NUM_TEST_ITERATIONS):
        logger.info("-" * 60)
        logger.info(f"Iteration {i + 1}/{NUM_TEST_ITERATIONS}")
        logger.info("-" * 60)
        result = tester.run_throughput_test()
        all_results.append(result)

    # Write to CSV if output path is specified
    if args.output_csv:
        write_results_to_csv(all_results, args.output_csv)

    logger.info("Throughput test completed successfully!")


if __name__ == "__main__":
    main()
