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
import sys
import time
from pathlib import Path
from typing import Any

import ray
import torch
from omegaconf import OmegaConf
from tensordict import NonTensorStack, TensorDict

parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

import transfer_queue as tq  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_test_case(
    batch_size: int | None = None,
    seq_length: int | None = None,
    field_num: int | None = None,
    device: str = "cpu",
) -> tuple[TensorDict, float]:
    """Create a test case with complex data formats.

    Creates TensorDict with:
    - Regular tensors: (batch_size, seq_length) shape, each element is float32
    - Nested Tensors (non-NPU): variable-length sequences with lengths forming an
      arithmetic progression from 1 to seq_length (average length ≈ seq_length/2)
    - Nested Tensors (NPU): regular tensors of shape (batch_size, seq_length//2)
    - NonTensorStack wrapped strings: each string size ~= seq_length * 4 bytes
      (to match memory footprint of one tensor element)

    Args:
        batch_size: Batch size for the test case
        seq_length: Maximum sequence length (used for regular tensors and
            as upper bound for nested tensor lengths)
        field_num: Total number of fields to create (distributed across types)
        device: Device to create tensors on ("cpu", "npu", or "gpu")

    Returns:
        Tuple of (TensorDict, total_size_gb)
    """
    bytes_per_element = 4  # float32

    # Calculate field distribution (1/3 each type, last fields may be regular)
    num_regular_fields = (field_num + 2) // 3
    num_nested_fields = (field_num + 2) // 3
    num_nontensor_fields = field_num - num_regular_fields - num_nested_fields

    # Each regular tensor field: batch_size * seq_length * 4 bytes
    regular_field_size_bytes = batch_size * seq_length * bytes_per_element
    regular_field_size_gb = regular_field_size_bytes / (1024**3)

    # Nested tensor field: average length = (1 + seq_length) / 2 (arithmetic progression),
    # so avg size = batch_size * (1 + seq_length) / 2 * 4 bytes
    # For NPU, nested fields become regular tensors of seq_length // 2
    if device == "npu":
        avg_nested_length = seq_length // 2
        nested_field_size_bytes = int(batch_size * avg_nested_length * bytes_per_element)
    else:
        avg_nested_length = (1 + seq_length) / 2
        nested_field_size_bytes = int(batch_size * avg_nested_length * bytes_per_element)
    nested_field_size_gb = nested_field_size_bytes / (1024**3)

    # NonTensorStack string field: each string ~= seq_length * 4 bytes to match one tensor element
    # Total for field: batch_size strings * seq_length * 4 bytes each
    string_size_per_elem = seq_length * bytes_per_element
    nontensor_field_size_bytes = batch_size * string_size_per_elem
    nontensor_field_size_gb = nontensor_field_size_bytes / (1024**3)

    # Total size = sum of all field types
    total_size_gb = (
        regular_field_size_gb * num_regular_fields
        + nested_field_size_gb * num_nested_fields
        + nontensor_field_size_gb * num_nontensor_fields
    )

    logger.info(f"Total data size: {total_size_gb:.6f} GB")

    # Determine torch device
    torch_device = None
    if device == "npu":
        torch_device = "npu:0"
    elif device == "gpu":
        torch_device = "cuda:0"

    # Set seeds for reproducibility (within this process)
    # For non-NPU: arithmetic progression lengths from 1 to seq_length for each nested field
    # For NPU: nested fields become regular tensors of seq_length // 2

    batch_size_tuple = (batch_size,)

    prompt_batch = TensorDict(batch_size=batch_size_tuple)

    # 1. Regular tensor fields
    for i in range(num_regular_fields):
        field_name = f"field_{i}"
        tensor_data = torch.randn(batch_size, seq_length, dtype=torch.float32, device=torch_device)
        prompt_batch.set(field_name, tensor_data)

    # 2. Nested Tensor fields (variable-length sequences) or regular tensors for NPU
    for i in range(num_nested_fields):
        field_name = f"nested_field_{i}"

        if device == "npu":
            # For NPU: create a regular tensor of seq_length // 2
            tensor_data = torch.randn(batch_size, seq_length // 2, dtype=torch.float32, device=torch_device)
            prompt_batch.set(field_name, tensor_data)
        else:
            # For non-NPU: create nested tensor with arithmetic progression lengths
            # Lengths go from 1 to seq_length in equal increments
            step = (seq_length - 1) / (batch_size - 1) if batch_size > 1 else 0
            nested_list = []
            for j in range(batch_size):
                length = int(round(1 + j * step))
                length = max(1, min(length, seq_length))  # Clamp to [1, seq_length]
                seq_data = torch.arange(length, dtype=torch.float32, device=torch_device)
                nested_list.append(seq_data)

            nested_tensor = torch.nested.as_nested_tensor(nested_list, layout=torch.jagged)
            prompt_batch.set(field_name, nested_tensor)

    # 3. NonTensorStack wrapped strings
    # Each string ~= seq_length * 4 bytes to match one tensor element's memory footprint
    string_char_count = seq_length * bytes_per_element  # 4 bytes per char (unicode)
    string_template = "x" * string_char_count

    for i in range(num_nontensor_fields):
        field_name = f"nontensor_field_{i}"
        string_data = [string_template for _ in range(batch_size)]
        prompt_batch.set(field_name, NonTensorStack.from_list(string_data))

    return prompt_batch, total_size_gb


@ray.remote
class TQClientActor:
    """Ray actor that uses tq.init(config) to initialize."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.test_data = None
        self.total_data_size_gb = 0.0
        self.test_keys = None

    def initialize(self) -> None:
        """Initialize transfer_queue with the config."""
        tq.init(OmegaConf.create(self.config))

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

    def delete(self, partition_id: str, keys: list[str] | None = None) -> None:
        """Delete data from storage using kv_clear."""
        if keys is None:
            keys = self.test_keys
        tq.kv_clear(keys=keys, partition_id=partition_id)

    def close(self) -> None:
        """Close transfer_queue."""
        tq.close()


class TQThroughputTester:
    """Main throughput tester for TransferQueue backends."""

    def __init__(
        self,
        backend_config_path: str,
        device: str,
        global_batch_size: int,
        field_num: int,
        seq_len: int,
        num_test_iterations: int,
        head_node_ip: str,
        backend: str | None = None,
        worker_node_ip: str | None = None,
        output_csv: str | None = None,
    ):
        """Initialize the throughput tester.

        Args:
            backend_config_path: Path to backend config YAML file
            backend: Override storage_backend in config (e.g. "SimpleStorage")
            device: Device type ("cpu", "npu", "gpu")
            global_batch_size: Global batch size
            field_num: Number of fields
            seq_len: Sequence length
            num_test_iterations: Number of test iterations
            head_node_ip: Head node IP address
            worker_node_ip: Worker node IP address (required for Yuanrong)
            output_csv: Path to output CSV file (optional)
        """
        self.backend_config_path = backend_config_path
        self.backend_override = backend
        self.device = device
        self.global_batch_size = global_batch_size
        self.field_num = field_num
        self.seq_len = seq_len
        self.num_test_iterations = num_test_iterations
        self.head_node_ip = head_node_ip
        self.worker_node_ip = worker_node_ip
        self.output_csv = output_csv

        # Prepare full config for tq.init()
        self.full_config = self._prepare_config()

        # Get backend from config
        self.backend = self.full_config["backend"]["storage_backend"]

        # For Yuanrong, always use inter_node
        self.use_inter_node = self.backend == "Yuanrong"

        # Validate arguments
        self._validate_args()

        # Initialize clients
        self._initialize_clients()

    def _validate_args(self) -> None:
        """Validate input arguments."""
        # Check worker_node_ip for Yuanrong
        if self.use_inter_node and self.worker_node_ip is None:
            raise ValueError("worker_node_ip is required for Yuanrong backend")

    def _prepare_config(self) -> dict[str, Any]:
        """Prepare the config by directly reading the backend_config file.

        Returns:
            Configuration dictionary
        """
        # Directly read the backend_config file, no merging with default
        config = OmegaConf.load(self.backend_config_path)

        # Override storage_backend if specified via CLI
        if self.backend_override is not None:
            config.backend.storage_backend = self.backend_override
            logger.info(f"Overriding storage_backend to: {self.backend_override}")

        # If backend.storage_backend is SimpleStorage, override total_storage_size
        total_storage_size = self.global_batch_size * self.num_test_iterations
        if config.backend.storage_backend == "SimpleStorage":
            config.backend.SimpleStorage.total_storage_size = total_storage_size

        return OmegaConf.to_container(config, resolve=True)

    def _initialize_clients(self) -> None:
        """Initialize writer and reader TQClientActors."""
        # Determine node placement
        if self.use_inter_node:
            writer_node = self.head_node_ip
            reader_node = self.worker_node_ip
        else:
            writer_node = reader_node = self.head_node_ip

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

        # Prepare configs for writer and reader
        # For Yuanrong backend, set different hosts for writer and reader
        if self.backend == "Yuanrong":
            import copy

            writer_config = copy.deepcopy(self.full_config)
            reader_config = copy.deepcopy(self.full_config)
            writer_config["backend"]["Yuanrong"]["host"] = self.head_node_ip
            reader_config["backend"]["Yuanrong"]["host"] = self.worker_node_ip
            logger.info(f"Writer Yuanrong host: {self.head_node_ip}")
            logger.info(f"Reader Yuanrong host: {self.worker_node_ip}")
        else:
            writer_config = self.full_config
            reader_config = self.full_config

        # Create writer and reader actors
        self.writer = TQClientActor.options(**writer_options).remote(writer_config)
        self.reader = TQClientActor.options(**reader_options).remote(reader_config)

        # Initialize transfer_queue
        logger.info(f"Using {self.backend} as storage backend.")

        w = self.writer.initialize.remote()
        r = self.reader.initialize.remote()
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
        logger.info(f"Total Data Size: {total_data_size_gb:.6f} GB")
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

        time.sleep(2)

        # LIST_KEYS operation using kv_list
        logger.info("Starting LIST_KEYS operation (kv_list)...")
        keys = ray.get(self.reader.list_keys.remote(partition_id=partition_id))

        time.sleep(2)

        # GET_DATA operation using kv_batch_get
        logger.info("Starting GET_DATA operation (kv_batch_get)...")
        start_get_data = time.perf_counter()
        ray.get(self.reader.get_data.remote(partition_id=partition_id, keys=keys))
        end_get_data = time.perf_counter()
        get_time = end_get_data - start_get_data
        get_gbit_per_sec = (total_data_size_gb * 8) / get_time
        get_gbyte_per_sec = total_data_size_gb / get_time

        time.sleep(2)

        # DELETE operation using kv_clear
        logger.info("Starting DELETE operation (kv_clear)...")
        start_delete = time.perf_counter()
        ray.get(self.writer.delete.remote(partition_id=partition_id, keys=keys))
        end_delete = time.perf_counter()
        delete_time = end_delete - start_delete
        logger.info(f"DELETE Time: {delete_time:.8f}s")

        # Print summary
        total_gbit_per_sec = (total_data_size_gb * 16) / (put_time + get_time)
        total_gbyte_per_sec = (total_data_size_gb * 2) / (put_time + get_time)

        logger.info("=" * 60)
        logger.info("THROUGHPUT TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Backend: {self.backend}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Total Data Size: {total_data_size_gb:.6f} GB")
        logger.info(f"PUT Time: {put_time:.8f}s")
        logger.info(f"GET Time: {get_time:.8f}s")
        logger.info(f"PUT Throughput: {put_gbit_per_sec:.8f} Gb/s ({put_gbyte_per_sec:.8f} GB/s)")
        logger.info(f"GET Throughput: {get_gbit_per_sec:.8f} Gb/s ({get_gbyte_per_sec:.8f} GB/s)")
        logger.info(f"Total Throughput: {total_gbit_per_sec:.8f} Gb/s ({total_gbyte_per_sec:.8f} GB/s)")
        logger.info("=" * 60)

        # Return results (only Gb/s for CSV, not GB/s)
        return {
            "backend": self.backend,
            "device": self.device,
            "total_data_size_gb": total_data_size_gb,
            "put_time": put_time,
            "get_time": get_time,
            "put_gbit_per_sec": put_gbit_per_sec,
            "get_gbit_per_sec": get_gbit_per_sec,
            "total_gbit_per_sec": total_gbit_per_sec,
        }

    def close(self) -> None:
        """Close the transfer_queue clients."""
        ray.get([self.writer.close.remote(), self.reader.close.remote()])


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
        "--backend_config",
        type=str,
        required=True,
        help="Path to backend config YAML file",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="Override storage_backend in config (e.g. SimpleStorage, Yuanrong, MooncakeStore)",
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
        "--num_test_iterations",
        type=int,
        default=4,
        help="Number of test iterations (default: 4)",
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
        help="Worker node IP address (required for Yuanrong)",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Path to output CSV file (optional)",
    )

    args = parser.parse_args()

    # Create and run tester
    tester = TQThroughputTester(
        backend_config_path=args.backend_config,
        device=args.device,
        global_batch_size=args.global_batch_size,
        field_num=args.field_num,
        seq_len=args.seq_len,
        num_test_iterations=args.num_test_iterations,
        head_node_ip=args.head_node_ip,
        backend=args.backend,
        worker_node_ip=args.worker_node_ip,
        output_csv=args.output_csv,
    )

    # Run test multiple times for consistent results using a for loop
    all_results = []
    for i in range(args.num_test_iterations):
        logger.info("-" * 60)
        logger.info(f"Iteration {i + 1}/{args.num_test_iterations}")
        logger.info("-" * 60)
        result = tester.run_throughput_test()
        all_results.append(result)

    # Write to CSV if output path is specified
    if args.output_csv:
        write_results_to_csv(all_results, args.output_csv)

    # Close transfer_queue
    tester.close()

    logger.info("Throughput test completed successfully!")


if __name__ == "__main__":
    main()
