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
import subprocess
import time
from urllib.parse import urlparse

from omegaconf import DictConfig

from transfer_queue.storage.bootstrap.provider import StorageBootstrapProvider
from transfer_queue.utils.logging_utils import get_logger

logger = get_logger(__name__)


@StorageBootstrapProvider.register_provider("MooncakeStore")
def initialize_mooncake_storage(conf: DictConfig) -> subprocess.Popen | None:
    """
    Initialize Mooncake store backend.
    Args:
        conf (DictConfig): Configuration dictionary for the Mooncake store backend.
    Returns:
        subprocess.Popen | None: Process object for the Mooncake store backend process.
    Raises:
        ValueError: If the Mooncake store is not initialized successfully.
    """
    if not conf.backend.MooncakeStore.auto_init:
        return None

    # Try to kill existing mooncake_master processes before starting a new one to avoid potential conflicts
    check = subprocess.run(["pgrep", "-f", "mooncake_master"], stdout=subprocess.PIPE, text=True)
    if check.returncode == 0:
        pids = check.stdout.strip().replace("\n", ", ")
        logger.info(f"Find existing mooncake_master (PID: {pids}), try to kill first...")

        result = os.system('pkill -f "[m]ooncake_master"')
        if result == 0:
            logger.info("Successfully killed existing mooncake_master processes.")
        else:
            raise RuntimeError(f"Failed to kill existing mooncake_master processes (exit code: {result}).")

    # process metadata_server
    metadata_server_raw_address = conf.backend.MooncakeStore.metadata_server
    if "://" not in metadata_server_raw_address:
        metadata_server_raw_address = "//" + metadata_server_raw_address

    metadata_server_parsed = urlparse(metadata_server_raw_address)

    if not metadata_server_parsed.hostname or metadata_server_parsed.port is None:
        raise ValueError(
            f"Invalid metadata_server '{conf.backend.MooncakeStore.metadata_server}'. "
            f"Host and port are required (e.g., host:port)."
        )

    metadata_server_host = metadata_server_parsed.hostname
    metadata_server_port = str(metadata_server_parsed.port)

    # process master_server
    master_server_raw_address = conf.backend.MooncakeStore.master_server_address
    if "://" not in master_server_raw_address:
        master_server_raw_address = "//" + master_server_raw_address

    master_server_parsed = urlparse(master_server_raw_address)

    if not master_server_parsed.hostname or master_server_parsed.port is None:
        raise ValueError(
            f"Invalid master_server_address '{conf.backend.MooncakeStore.master_server_address}'. "
            f"Host and port are required (e.g., host:port)."
        )

    master_server_port = str(master_server_parsed.port)

    cmd = [
        "mooncake_master",
        "-client_ttl=30",
        "-default_kv_lease_ttl=999999",
        "-default_kv_soft_pin_ttl=999999",
        "--eviction_high_watermark_ratio=1.0",
        "--eviction_ratio=0.0",
        "--enable_http_metadata_server=true",
        "--allow_evict_soft_pinned_objects=false",
        f"--http_metadata_server_host={metadata_server_host}",
        f"--http_metadata_server_port={metadata_server_port}",
        f"--rpc_port={master_server_port}",
    ]

    log_file_path = "/tmp/mooncake_master.log"
    with open(log_file_path, "w") as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            start_new_session=True,
        )
        time.sleep(3)

    if process.poll() is None:
        logger.info(f"mooncake_master started, PID: {process.pid}. Logs are at: {os.path.abspath(log_file_path)}")
    else:
        error_msg = ""
        try:
            with open(log_file_path) as f:
                error_msg = f.read()
        except Exception as e:
            error_msg = f"Failed to read log file: {e}"

        raise RuntimeError(
            f"mooncake_master exited with error. Check {log_file_path} for detailed logs. Output:\n{error_msg}"
        )

    return process
