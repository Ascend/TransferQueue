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
from tensordict import NonTensorStack, TensorDict

parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

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
class RemoteDataStore:
    """Ray remote actor that stores and retrieves data directly (without ray.put)."""

    def __init__(self):
        self.stored_data = None

    def put_data(self, data: TensorDict) -> None:
        self.stored_data = data

    def get_data(self) -> TensorDict:
        return self.stored_data

    def clear_data(self) -> None:
        self.stored_data = None


class RayBaselineTester:
    """Ray baseline throughput tester - measures raw Ray data transfer performance."""

    def __init__(
        self,
        global_batch_size: int,
        field_num: int,
        seq_len: int,
        num_test_iterations: int,
        head_node_ip: str,
        worker_node_ip: str | None = None,
        output_csv: str | None = None,
    ):
        """Initialize the Ray baseline tester.

        Args:
            global_batch_size: Global batch size
            field_num: Number of fields
            seq_len: Sequence length
            num_test_iterations: Number of test iterations
            head_node_ip: Head node IP address
            worker_node_ip: Worker node IP address
            output_csv: Path to output CSV file (optional)
        """
        self.global_batch_size = global_batch_size
        self.field_num = field_num
        self.seq_len = seq_len
        self.num_test_iterations = num_test_iterations
        self.head_node_ip = head_node_ip
        self.worker_node_ip = worker_node_ip
        self.output_csv = output_csv

        # Initialize remote store on worker node
        self._initialize_remote_store()

    def _initialize_remote_store(self) -> None:
        """Initialize the RemoteDataStore actor on worker node."""
        writer_node = self.head_node_ip
        reader_node = self.worker_node_ip if self.worker_node_ip else self.head_node_ip

        logger.info(f"Writer is on {writer_node}, Reader is on {reader_node}")

        self.remote_store = RemoteDataStore.options(
            num_cpus=0.001,
            resources={f"node:{reader_node}": 0.001},
        ).remote()

        logger.info(f"RemoteDataStore created on {reader_node}")

    def run_throughput_test(self) -> dict[str, Any]:
        """Run the throughput test and print results.

        Returns:
            Dictionary with test results
        """
        # Create test data
        logger.info("Creating large batch for throughput test...")
        start_create_data = time.perf_counter()
        test_data, total_data_size_gb = create_test_case(
            batch_size=self.global_batch_size,
            seq_length=self.seq_len,
            field_num=self.field_num,
            device="cpu",
        )
        end_create_data = time.perf_counter()
        logger.info(f"Data creation time: {end_create_data - start_create_data:.8f}s")

        # PUT operation - pass data directly to remote actor
        logger.info("Starting PUT operation...")
        start_put = time.perf_counter()
        ray.get(self.remote_store.put_data.remote(test_data))
        end_put = time.perf_counter()
        put_time = end_put - start_put
        put_gbit_per_sec = (total_data_size_gb * 8) / put_time

        time.sleep(2)

        # GET operation - retrieve data from remote actor
        logger.info("Starting GET operation...")
        start_get = time.perf_counter()
        _ = ray.get(self.remote_store.get_data.remote())
        end_get = time.perf_counter()
        get_time = end_get - start_get
        get_gbit_per_sec = (total_data_size_gb * 8) / get_time

        # Clear data
        ray.get(self.remote_store.clear_data.remote())

        # Calculate total throughput
        total_gbit_per_sec = (total_data_size_gb * 16) / (put_time + get_time)

        # Print summary
        logger.info("=" * 60)
        logger.info("RAY BASELINE THROUGHPUT TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Data Size: {total_data_size_gb:.6f} GB")
        logger.info(f"PUT Time: {put_time:.8f}s")
        logger.info(f"GET Time: {get_time:.8f}s")
        logger.info(f"PUT Throughput: {put_gbit_per_sec:.8f} Gb/s")
        logger.info(f"GET Throughput: {get_gbit_per_sec:.8f} Gb/s")
        logger.info(f"Total Throughput (round-trip): {total_gbit_per_sec:.8f} Gb/s")
        logger.info("=" * 60)

        return {
            "backend": "RayBaseline",
            "device": "cpu",
            "total_data_size_gb": total_data_size_gb,
            "put_time": put_time,
            "get_time": get_time,
            "put_gbit_per_sec": put_gbit_per_sec,
            "get_gbit_per_sec": get_gbit_per_sec,
            "total_gbit_per_sec": total_gbit_per_sec,
        }


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
    """Main entry point for the Ray baseline perftest script."""
    parser = argparse.ArgumentParser(description="Ray Baseline Throughput Test")
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
        default=3,
        help="Number of test iterations (default: 3)",
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
        help="Worker node IP address (optional)",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Path to output CSV file (optional)",
    )

    args = parser.parse_args()

    # Create and run tester
    tester = RayBaselineTester(
        global_batch_size=args.global_batch_size,
        field_num=args.field_num,
        seq_len=args.seq_len,
        num_test_iterations=args.num_test_iterations,
        head_node_ip=args.head_node_ip,
        worker_node_ip=args.worker_node_ip,
        output_csv=args.output_csv,
    )

    # Run test multiple times
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

    logger.info("Ray baseline throughput test completed successfully!")


if __name__ == "__main__":
    main()
