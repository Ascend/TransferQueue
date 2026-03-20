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
import logging
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import ray
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from transfer_queue.client import TransferQueueClient  # noqa: E402
from transfer_queue.controller import TransferQueueController  # noqa: E402
from transfer_queue.storage.simple_backend import SimpleStorageUnit  # noqa: E402
from transfer_queue.utils.common import get_placement_group  # noqa: E402
from transfer_queue.utils.zmq_utils import process_zmq_server_info  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_complex_test_case(
    batch_size: int | None = None,
    seq_length: int | None = None,
    field_num: int | None = None,
    device: str = "cpu",
) -> tuple[TensorDict, float]:
    """Create a complex test case with tensor and non-tensor fields.

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

    num_tensor_fields = (field_num + 1) // 2
    num_nontensor_fields = field_num // 2

    total_tensor_size_gb = tensor_field_size_gb * num_tensor_fields
    total_nontensor_size_gb = (batch_size * 1024 / (1024**3)) * num_nontensor_fields
    total_size_gb = total_tensor_size_gb + total_nontensor_size_gb

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

        if i % 2 == 0:
            # Tensor field
            tensor_data = torch.randn(batch_size, seq_length, dtype=torch.float32, device=torch_device)
            fields[field_name] = tensor_data
        else:
            # NonTensorData field
            str_length = 1024
            non_tensor_data = [
                "".join(
                    random.choices(
                        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                        k=str_length,
                    )
                )
                for _ in range(batch_size)
            ]
            fields[field_name] = NonTensorData(data=non_tensor_data, batch_size=(batch_size,), device=None)

    batch_size_tuple = (batch_size,)
    prompt_batch = TensorDict(
        fields,
        batch_size=batch_size_tuple,
    )

    return prompt_batch, total_size_gb


@ray.remote
class TQClientActor:
    """Ray actor that holds a TransferQueueClient."""

    def __init__(self, client_id: str, controller_info: Any):
        self.client = TransferQueueClient(
            client_id=client_id,
            controller_info=controller_info,
        )
        self.prompt_meta = None
        self.test_data = None
        self.total_data_size_gb = 0.0

    def initialize_storage_manager(self, manager_type: str, config: dict[str, Any]) -> None:
        """Initialize the storage manager with given config."""
        self.client.initialize_storage_manager(manager_type=manager_type, config=config)

    def create_complex_test_case(
        self,
        batch_size: int | None = None,
        seq_length: int | None = None,
        field_num: int | None = None,
        device: str = "cpu",
    ) -> tuple[list[str], float]:
        """Create test case on the actor."""
        self.test_data, self.total_data_size_gb = create_complex_test_case(batch_size, seq_length, field_num, device)
        return list(self.test_data.keys()), self.total_data_size_gb

    def put(self, partition_id: str) -> None:
        """Put data to storage."""
        self.client.put(data=self.test_data, partition_id=partition_id)

    def get_meta(
        self,
        data_fields: list[str],
        batch_size: int,
        partition_id: str,
        task_name: str | None = None,
        sampling_config: dict[str, Any] | None = None,
    ) -> Any:
        """Get metadata from controller."""
        self.prompt_meta = self.client.get_meta(
            data_fields=data_fields,
            batch_size=batch_size,
            partition_id=partition_id,
            task_name=task_name,
            sampling_config=sampling_config,
        )
        return self.prompt_meta

    def get_data(self) -> None:
        """Get data from storage using cached metadata."""
        self.client.get_data(self.prompt_meta)


class TQThroughputTester:
    """Main throughput tester for TransferQueue backends."""

    def __init__(
        self,
        backend: str,
        client_placement: str,
        backend_config: dict[str, Any],
        device: str,
        global_batch_size: int,
        field_num: int,
        seq_len: int,
        num_global_batch: int,
        head_node_ip: str,
        worker_node_ip: str | None = None,
    ):
        """Initialize the throughput tester.

        Args:
            backend: Backend type ("default", "yuanrong", "mooncake")
            client_placement: Client placement mode ("intra_node" or "inter_node")
            backend_config: Backend configuration dictionary
            device: Device type ("cpu", "npu", "gpu")
            global_batch_size: Global batch size
            field_num: Number of fields
            seq_len: Sequence length
            num_global_batch: Number of global batches
            head_node_ip: Head node IP address
            worker_node_ip: Worker node IP address (required for inter_node)
        """
        self.backend = backend
        self.client_placement = client_placement
        self.backend_config = backend_config
        self.device = device
        self.global_batch_size = global_batch_size
        self.field_num = field_num
        self.seq_len = seq_len
        self.num_global_batch = num_global_batch
        self.head_node_ip = head_node_ip
        self.worker_node_ip = worker_node_ip

        # Validate arguments
        self._validate_args()

        # Determine manager type and prepare configs
        self.manager_type = self._get_manager_type()
        self.writer_config, self.reader_config = self._prepare_backend_configs()

        # Initialize the test infrastructure
        self._initialize_data_system()
        self._initialize_clients()

    def _validate_args(self) -> None:
        """Validate input arguments."""
        if self.client_placement == "inter_node" and self.worker_node_ip is None:
            raise ValueError("worker_node_ip is required for inter_node client placement")
        if self.backend == "default":
            storage_unit_placement = self.backend_config.get("storage_unit_placement", "normal")
            if storage_unit_placement == "remote" and self.worker_node_ip is None:
                raise ValueError("worker_node_ip is required for remote storage_unit_placement")

    def _get_manager_type(self) -> str:
        """Get the storage manager type based on backend."""
        if self.backend == "default":
            return "AsyncSimpleStorageManager"
        elif self.backend == "yuanrong":
            return "YuanrongStorageManager"
        elif self.backend == "mooncake":
            return "MooncakeStorageManager"
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _prepare_backend_configs(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Prepare writer and reader backend configs.

        Returns:
            Tuple of (writer_config, reader_config)
        """
        # Set client_name based on backend
        base_config = self.backend_config.copy()
        if self.backend == "yuanrong":
            base_config["client_name"] = "YuanrongStorageClient"
        elif self.backend == "mooncake":
            base_config["client_name"] = "MooncakeStoreClient"

        writer_config = base_config.copy()
        reader_config = base_config.copy()

        if self.client_placement == "inter_node":
            if self.backend == "yuanrong":
                writer_config["host"] = self.head_node_ip
                reader_config["host"] = self.worker_node_ip
            elif self.backend == "mooncake":
                writer_config["local_hostname"] = self.head_node_ip
                reader_config["local_hostname"] = self.worker_node_ip

        return writer_config, reader_config

    def _initialize_data_system(self) -> None:
        """Initialize controller and storage units if needed."""
        # Initialize controller
        self.data_system_controller = TransferQueueController.remote()
        logger.info("TransferQueueController has been created.")
        self.data_system_controller_info = process_zmq_server_info(self.data_system_controller)

        # Initialize storage units for default backend
        if self.backend == "default":
            self._initialize_storage_units()

    def _initialize_storage_units(self) -> None:
        """Initialize SimpleStorageUnits for default backend."""
        num_data_storage_units = self.backend_config.get("num_data_storage_units", 8)
        storage_unit_placement = self.backend_config.get("storage_unit_placement", "normal")
        total_storage_size = self.global_batch_size * self.num_global_batch

        self.data_system_storage_units = {}

        if storage_unit_placement == "remote":
            # Remote mode: create all storage units on worker node
            for storage_unit_rank in range(num_data_storage_units):
                storage_node = SimpleStorageUnit.options(
                    num_cpus=1,
                    resources={f"node:{self.worker_node_ip}": 0.001},
                ).remote(storage_unit_size=3 * math.ceil(total_storage_size / num_data_storage_units))
                self.data_system_storage_units[storage_unit_rank] = storage_node
            logger.info(
                f"StorageUnit #0 ~ #{num_data_storage_units - 1} has been created on worker node {self.worker_node_ip}."
            )
        else:
            # Normal mode: create storage units using placement group
            storage_placement_group = get_placement_group(num_data_storage_units, num_cpus_per_actor=1)
            for storage_unit_rank in range(num_data_storage_units):
                storage_node = SimpleStorageUnit.options(
                    placement_group=storage_placement_group,
                    placement_group_bundle_index=storage_unit_rank,
                ).remote(storage_unit_size=3 * math.ceil(total_storage_size / num_data_storage_units))
                self.data_system_storage_units[storage_unit_rank] = storage_node
            logger.info(f"StorageUnit #0 ~ #{num_data_storage_units - 1} has been created.")

        self.data_system_storage_unit_infos = process_zmq_server_info(self.data_system_storage_units)
        # Add storage unit infos to backend configs
        self.writer_config["zmq_info"] = self.data_system_storage_unit_infos
        self.reader_config["zmq_info"] = self.data_system_storage_unit_infos

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
            "resources": {f"node:{writer_node}": 0.001},
        }
        reader_options = {
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
        self.writer = TQClientActor.options(**writer_options).remote("writer", self.data_system_controller_info)
        self.reader = TQClientActor.options(**reader_options).remote("reader", self.data_system_controller_info)

        # Initialize storage managers
        logger.info(f"Using {self.manager_type} as storage backend.")

        w = self.writer.initialize_storage_manager.remote(manager_type=self.manager_type, config=self.writer_config)
        r = self.reader.initialize_storage_manager.remote(manager_type=self.manager_type, config=self.reader_config)
        ray.get([w, r])

    def run_throughput_test(self) -> None:
        """Run the throughput test and print results."""
        logger.info("Creating large batch for throughput test...")
        start_create_data = time.time()
        data_fields, total_data_size_gb = ray.get(
            self.writer.create_complex_test_case.remote(
                batch_size=self.global_batch_size,
                seq_length=self.seq_len,
                field_num=self.field_num,
                device=self.device,
            )
        )
        end_create_data = time.time()
        logger.info(f"Data creation time: {end_create_data - start_create_data:.8f}s")

        # PUT operation
        logger.info("Starting PUT operation...")
        start_put = time.time()
        ray.get(self.writer.put.remote(partition_id="train_0"))
        end_put = time.time()
        put_time = end_put - start_put
        put_throughput_gbps = (total_data_size_gb * 8) / put_time
        put_throughput_gbs = total_data_size_gb / put_time
        logger.info(f"put cost time: {put_time:.8f}s")
        logger.info(f"PUT Throughput: {put_throughput_gbps:.8f} Gb/s ({put_throughput_gbs:.8f} GB/s)")

        time.sleep(2)

        # GET_META operation
        logger.info("Starting GET_META operation...")
        start_get_meta = time.time()
        ray.wait(
            [
                self.reader.get_meta.remote(
                    data_fields=list(data_fields),
                    batch_size=self.global_batch_size,
                    partition_id="train_0",
                    task_name="generate_sequences",
                )
            ]
        )
        end_get_meta = time.time()
        logger.info(f"get_meta cost time: {end_get_meta - start_get_meta:.8f}s")

        time.sleep(2)

        # GET_DATA operation
        logger.info("Starting GET_DATA operation...")
        start_get_data = time.time()
        ray.get(self.reader.get_data.remote())
        end_get_data = time.time()
        get_time = end_get_data - start_get_data
        get_throughput_gbps = (total_data_size_gb * 8) / get_time
        get_throughput_gbs = total_data_size_gb / get_time

        logger.info(f"get_data cost time: {get_time:.8f}s")
        logger.info(f"GET Throughput: {get_throughput_gbps:.8f} Gb/s ({get_throughput_gbs:.8f} GB/s)")

        # Print summary
        total_throughput_gbps = (total_data_size_gb * 16) / (put_time + get_time)
        total_throughput_gbs = (total_data_size_gb * 2) / (put_time + get_time)

        logger.info("=" * 60)
        logger.info("THROUGHPUT TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Backend: {self.backend}")
        logger.info(f"Client Placement: {self.client_placement}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Total Data Size: {total_data_size_gb:.6f} GB")
        logger.info(f"PUT Time: {put_time:.8f}s")
        logger.info(f"GET Time: {get_time:.8f}s")
        logger.info(f"PUT Throughput: {put_throughput_gbps:.8f} Gb/s ({put_throughput_gbs:.8f} GB/s)")
        logger.info(f"GET Throughput: {get_throughput_gbps:.8f} Gb/s ({get_throughput_gbs:.8f} GB/s)")
        logger.info(f"Total Throughput: {total_throughput_gbps:.8f} Gb/s ({total_throughput_gbs:.8f} GB/s)")
        logger.info("=" * 60)


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
    if backend == "default":
        return {"num_data_storage_units": 1, "storage_unit_placement": "normal"}
    elif backend == "yuanrong":
        return {
            "host": "127.0.0.1",
            "port": 31501,
            "enable_yr_npu_transport": False,
        }
    elif backend == "mooncake":
        return {
            "local_hostname": "127.0.0.1",
            "metadata_server": "127.0.0.1:8080",
            "master_server_address": "127.0.0.1:8081",
        }
    else:
        return {}


def main() -> None:
    """Main entry point for the perftest script."""
    parser = argparse.ArgumentParser(description="TransferQueue Throughput Test")
    parser.add_argument(
        "--backend",
        type=str,
        default="default",
        choices=["default", "yuanrong", "mooncake"],
        help="Backend type to test (default: default)",
    )
    parser.add_argument(
        "--client_placement",
        type=str,
        default="intra_node",
        choices=["intra_node", "inter_node"],
        help="Client placement mode (default: intra_node)",
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
        help="Worker node IP address (required for inter_node)",
    )
    parser.add_argument(
        "--ray_address",
        type=str,
        default="auto",
        help="Ray cluster address (default: auto)",
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
        client_placement=args.client_placement,
        backend_config=backend_config,
        device=args.device,
        global_batch_size=args.global_batch_size,
        field_num=args.field_num,
        seq_len=args.seq_len,
        num_global_batch=args.num_global_batch,
        head_node_ip=args.head_node_ip,
        worker_node_ip=args.worker_node_ip,
    )

    # Run test multiple times for consistent results
    print("-" * 60)
    tester.run_throughput_test()
    print("-" * 60)
    tester.run_throughput_test()
    print("-" * 60)
    tester.run_throughput_test()

    logger.info("Throughput test completed successfully!")


if __name__ == "__main__":
    main()
