import argparse
import asyncio
import json
import logging
import math
import os
import random
import sys
import time
import numpy as np
import ray
import torch
from pathlib import Path
from omegaconf import OmegaConf
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData
from tensordict.utils import LinkedList


parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

from transfer_queue import (
    AsyncTransferQueueClient,
    SimpleStorageUnit,
    TransferQueueController,
    process_zmq_server_info,
)
from transfer_queue.utils.utils import get_placement_group

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =========================================================
# Configuration Map
# =========================================================
CONFIG_MAP = {
    "debug": {
        "global_batch_size": 32,
        "seq_length": 128,
        "field_num": 2,
        "desc": "Debug (~32KB)"
    },
    "tiny": {
        "global_batch_size": 64,
        "seq_length": 1024,
        "field_num": 4,
        "desc": "Tiny (~1MB)"
    },
    "small": {
        "global_batch_size": 512,
        "seq_length": 12800,
        "field_num": 4,
        "desc": "Small (~100MB)"
    },
    "medium": {
        "global_batch_size": 1024,
        "seq_length": 65536,
        "field_num": 4,
        "desc": "Medium (~1GB)"
    },
    "large": {
        "global_batch_size": 2048,
        "seq_length": 128000,
        "field_num": 5,
        "desc": "Large (~5GB)"
    },
    "xlarge": {
        "global_batch_size": 4096,
        "seq_length": 128000,
        "field_num": 5,
        "desc": "X-Large (~10GB)"
    },
    "huge": {
        "global_batch_size": 4096,
        "seq_length": 128000,
        "field_num": 10,
        "desc": "Huge (~20GB)"
    }
}


# =========================================================
# Helper Functions
# =========================================================
def calculate_stats(data: list) -> dict:
    """Calculate statistics"""
    if not data:
        return {"mean": 0.0, "max": 0.0, "min": 0.0, "p99": 0.0}
    return {
        "mean": float(np.mean(data)),
        "max": float(np.max(data)),
        "min": float(np.min(data)),
        "p99": float(np.percentile(data, 99))
    }


def create_complex_test_case(batch_size, seq_length, field_num):
    tensor_field_size_bytes = batch_size * seq_length * 4
    tensor_field_size_gb = tensor_field_size_bytes / (1024 ** 3)

    num_tensor_fields = (field_num + 1) // 2
    num_nontensor_fields = field_num // 2

    total_tensor_size_gb = tensor_field_size_gb * num_tensor_fields
    # Estimate String NonTensorData size (assuming 1024 bytes per sample)
    str_len = 1024
    total_nontensor_size_gb = (batch_size * str_len / (1024 ** 3)) * num_nontensor_fields
    total_size_gb = total_tensor_size_gb + total_nontensor_size_gb

    fields = {}
    for i in range(field_num):
        field_name = f"field_{i}"
        if i % 2 == 0:
            tensor_data = torch.randn(batch_size, seq_length, dtype=torch.float32)
            fields[field_name] = tensor_data
        else:
            non_tensor_data = [
                ''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=str_len))
                for _ in range(batch_size)
            ]
            fields[field_name] = NonTensorData(data=non_tensor_data, batch_size=(batch_size,), device=None)

    prompt_batch = TensorDict(
        fields,
        batch_size=(batch_size,),
        device=None,
    )
    return prompt_batch, total_size_gb


def remove_placement_group(placement_group):
    if placement_group is None:
        return

    ray.util.remove_placement_group(placement_group)


def check_data_consistency(original, retrieved, path="root"):
    """
    Data consistency verification
    """
    try:
        if isinstance(original, list) and isinstance(retrieved, LinkedList):
            retrieved = list(retrieved)

        if type(original) != type(retrieved):
            return False, f"[{path}] Type mismatch: {type(original)} vs {type(retrieved)}"

        if isinstance(original, TensorDict):
            if set(original.keys()) != set(retrieved.keys()):
                return False, f"[{path}] Keys mismatch: {original.keys()} vs {retrieved.keys()}"
            for key in original.keys():
                is_valid, msg = check_data_consistency(original[key], retrieved[key], path=f"{path}.{key}")
                if not is_valid:
                    return False, msg
            return True, "Passed"

        elif isinstance(original, torch.Tensor):
            if original.shape != retrieved.shape:
                return False, f"[{path}] Tensor shape mismatch: {original.shape} vs {retrieved.shape}"
            t1 = original.cpu()
            t2 = retrieved.cpu()
            if not torch.equal(t1, t2):
                return False, f"[{path}] Tensor values mismatch"
            return True, "Passed"

        elif 'NonTensorData' in str(type(original)):
            if original.data != retrieved.data:
                return False, f"[{path}] NonTensorData content mismatch"
            return True, "Passed"

        else:
            if original != retrieved:
                return False, f"[{path}] Value mismatch: {original} vs {retrieved}"
            return True, "Passed"

    except Exception as e:
        return False, f"[{path}] Exception during check: {str(e)}"


# =========================================================
# Core Tester Class
# =========================================================

def sync_stage(flag_to_create, flag_to_wait):
    """Profile sync helper function for synchronizing with external profiler process"""
    with open(flag_to_create, 'w') as f:
        f.write('1')
    while not os.path.exists(flag_to_wait):
        time.sleep(0.05)
    try:
        os.remove(flag_to_wait)
    except:
        pass


class TQBandwidthTester:
    def __init__(self, target_ip=None, storage_units=8, enable_profile=False):
        self.target_ip = target_ip
        self.num_storage_units = storage_units
        self.remote_mode = target_ip is not None
        self.enable_profile = enable_profile
        self.data_system_client = None
        self.tq_config = None
        self.data_system_controller = None
        self.data_system_storage_units = {}
        self.storage_placement_group = None

    def initialize_system(self, config_dict):
        """Initialize TransferQueue system based on current configuration"""
        # Basic config conversion
        self.tq_config = OmegaConf.create({
            "global_batch_size": config_dict["global_batch_size"],
            "num_global_batch": 1,
            "num_data_storage_units": self.num_storage_units,
            "num_data_controllers": 1
        })

        total_storage_size = self.tq_config.global_batch_size * 2

        logger.info(f"Initializing Storage Units (Remote={self.remote_mode}, Target={self.target_ip})...")

        if self.remote_mode:
            # Remote Mode: Force placement on specific worker IP
            for rank in range(self.num_storage_units):
                self.data_system_storage_units[rank] = SimpleStorageUnit.options(
                    num_cpus=1,
                    resources={f"node:{self.target_ip}": 0.001},
                    runtime_env={"env_vars": {"OMP_NUM_THREADS": "2"}},
                ).remote(
                    storage_unit_size=math.ceil(total_storage_size / self.num_storage_units)
                )
        else:
            # Local Mode: Use placement group
            self.storage_placement_group = get_placement_group(self.num_storage_units, num_cpus_per_actor=2)
            for rank in range(self.num_storage_units):
                self.data_system_storage_units[rank] = SimpleStorageUnit.options(
                    placement_group=self.storage_placement_group,
                    placement_group_bundle_index=rank,
                    runtime_env={"env_vars": {"OMP_NUM_THREADS": "2"}},
                ).remote(
                    storage_unit_size=math.ceil(total_storage_size / self.num_storage_units)
                )

        # Controller Init
        self.data_system_controller = TransferQueueController.remote()

        # Info Collection
        self.data_system_controller_info = process_zmq_server_info(self.data_system_controller)
        self.data_system_storage_unit_infos = process_zmq_server_info(self.data_system_storage_units)

        # Config Merge
        tq_internal_conf = OmegaConf.create({}, flags={"allow_objects": True})
        tq_internal_conf.controller_info = self.data_system_controller_info
        tq_internal_conf.storage_unit_infos = self.data_system_storage_unit_infos
        self.tq_config = OmegaConf.merge(tq_internal_conf, self.tq_config)

        # Client Init
        self.data_system_client = AsyncTransferQueueClient(
            client_id='Trainer',
            controller_info=self.data_system_controller_info
        )
        self.data_system_client.initialize_storage_manager(manager_type="AsyncSimpleStorageManager",
                                                           config=self.tq_config)

        return self.data_system_client

    def cleanup(self):
        """Explicitly cleanup Ray resources"""
        logger.info("Cleaning up previous Ray resources...")

        # 1. Kill Controller Actor
        if self.data_system_controller:
            ray.kill(self.data_system_controller)
            self.data_system_controller = None

        # 2. Kill Storage Unit Actors
        if self.data_system_storage_units:
            for unit in self.data_system_storage_units.values():
                ray.kill(unit)
            self.data_system_storage_units = {}

        # 3. Remove Placement Group (release reserved CPU/resource bundles)
        if self.storage_placement_group:
            remove_placement_group(self.storage_placement_group)
            self.storage_placement_group = None

        # 4. Force garbage collection
        import gc
        gc.collect()

        # 5. Wait for Ray scheduler to update state (prevent race condition)
        time.sleep(2)

    def run_benchmark_rounds(self, config_name, config, rounds):
        """Run multiple rounds of PUT/GET bandwidth tests"""
        logger.info(f"Generating test data [{config_name}]...")
        big_input_ids, total_gb = create_complex_test_case(
            batch_size=config["global_batch_size"],
            seq_length=config["seq_length"],
            field_num=config["field_num"]
        )
        logger.info(f"Data Size: {total_gb:.4f} GB")

        put_speeds = []
        get_speeds = []

        print(f"\n🚀 Running TQ Config: [{config_name}] | Size: {total_gb:.4f} GB | Rounds: {rounds}")

        for i in range(rounds):
            partition_key = f"bench_{config_name}_{i}"

            # PUT operation
            start_put = time.time()
            if i == 0 and self.enable_profile:
                sync_stage('init_ready.flag', 'put_start.flag')
            asyncio.run(self.data_system_client.async_put(data=big_input_ids, partition_id=partition_key))
            put_time = time.time() - start_put
            if i == 0 and self.enable_profile:
                sync_stage('put_done.flag', 'get_prepare.flag')

            put_gbps = (total_gb * 8) / put_time
            put_speeds.append(put_gbps)
            time.sleep(2)
            # Get metadata (required step for TQ flow)
            prompt_meta = asyncio.run(self.data_system_client.async_get_meta(
                data_fields=list(big_input_ids.keys()),
                batch_size=big_input_ids.size(0),
                partition_id=partition_key,
                task_name='generate_sequences',
            ))

            # GET operation
            start_get = time.time()
            if i == 0 and self.enable_profile:
                sync_stage('get_ready.flag', 'get_start.flag')
            retrieved_data = asyncio.run(self.data_system_client.async_get_data(prompt_meta))
            get_time = time.time() - start_get

            get_gbps = (total_gb * 8) / get_time
            get_speeds.append(get_gbps)

            print(f"\r  Round {i + 1}/{rounds}: PUT {put_gbps:.2f} Gbps | GET {get_gbps:.2f} Gbps", end="")

            # Data consistency verification (first and last round only)
            if i == 0 or i == rounds - 1:
                is_consistent, msg = check_data_consistency(big_input_ids, retrieved_data)
                if not is_consistent:
                    print(f" ❌ FAIL: {msg}")
                else:
                    print(f" ✅ PASS", end="")
            asyncio.run(self.data_system_client.async_clear_partition(partition_id=partition_key))
        print("\n")

        def make_result(op, speeds):
            """Construct result dictionary"""
            return {
                "scenario": "TransferQueue",
                "setting": f"{config_name} (Remote)" if self.remote_mode else f"{config_name} (Local)",
                "data_volume": f"{total_gb * 1024:.2f} MB" if total_gb * 1024 < 10 else f"{total_gb:.4f} GB",
                "operation": op,
                "payload_gb": total_gb,
                "stats_gbps": calculate_stats(speeds)
            }

        return [make_result("PUT", put_speeds), make_result("GET", get_speeds)]


# =========================================================
# Main Function
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="TransferQueue Bandwidth Benchmark")
    parser.add_argument("--ip", type=str, default=None, help="Worker node IP, local test if not set")
    parser.add_argument("--config", type=str, default=None, choices=list(CONFIG_MAP.keys()),
                        help="Specific config to run")
    parser.add_argument("--output", type=str, default="tq_benchmark_result.json", help="Output JSON file")
    parser.add_argument("--rounds", type=int, default=20, help="Test rounds per config (default: 20)")
    parser.add_argument("--shards", type=int, default=8, help="Number of storage units (default: 8)")
    parser.add_argument("--profile", action="store_true", help="Enable profile sync (requires external profiler)")

    args = parser.parse_args()

    # Initialize Ray
    current_working_dir = os.getcwd()
    if not ray.is_initialized():
        ray.init(
            address="auto" if args.ip else None,
            runtime_env={"working_dir": current_working_dir}
        )

    logger.info(f"Ray initialized. Target: {args.ip if args.ip else 'Local'}")

    # Create tester
    tester = TQBandwidthTester(target_ip=args.ip, storage_units=args.shards, enable_profile=args.profile)

    # Run tests
    run_list = [args.config] if args.config else list(CONFIG_MAP.keys())
    final_results = []

    try:
        for cfg_name in run_list:
            cfg = CONFIG_MAP[cfg_name]
            # Re-initialize system for each config to ensure correct storage_unit_size calculation
            tester.cleanup()
            tester.initialize_system(cfg)
            res = tester.run_benchmark_rounds(cfg_name, cfg, args.rounds)
            final_results.extend(res)

        # Save results
        with open(args.output, "w") as f:
            json.dump(final_results, f, indent=4)
        logger.info(f"💾 Results saved to {args.output}")

    except Exception as e:
        logger.error(f"❌ Critical error: {e}", exc_info=True)
    finally:
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    try:
        from transfer_queue.utils import serial_utils
        print(f'[Startup Check] TQ_ZERO_COPY_SERIALIZATION = {serial_utils.TQ_ZERO_COPY_SERIALIZATION}')
    except ImportError:
        print('[Startup Check] Could not import serial_utils to check flag')
    main()
