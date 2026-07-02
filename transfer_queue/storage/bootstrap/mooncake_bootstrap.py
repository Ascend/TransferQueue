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
def initialize_mooncake_storage(conf: DictConfig) -> subprocess.Popen | dict | None:
    """
    Initialize Mooncake store backend.

    Supports two metadata modes:
      - HTTP metadata server (default): metadata_server = "host:port"
      - P2P handshake (recommended for multi-NIC environments): metadata_server = "P2PHANDSHAKE"

    When ``offload.enabled`` is set to ``true`` in the config, this function also starts
    a standalone ``mooncake_client`` process that offloads data from DRAM to NVMe SSD.

    Args:
        conf (DictConfig): Configuration dictionary for the Mooncake store backend.
    Returns:
        subprocess.Popen | dict | None:
            - None if auto_init is disabled.
            - subprocess.Popen: master process (when offload is disabled).
            - dict: {"master_process": Popen, "offload_client_process": Popen} (when offload is enabled).
    Raises:
        ValueError: If the Mooncake store is not initialized successfully.
        RuntimeError: If mooncake_master or mooncake_client fails to start.
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

    # process metadata_server (normalize and validate)
    metadata_server_raw_address = str(conf.backend.MooncakeStore.metadata_server).strip()
    use_p2p_handshake = metadata_server_raw_address.upper() == "P2PHANDSHAKE"

    if use_p2p_handshake:
        metadata_server_host = None
        metadata_server_port = None
        logger.info("mooncake_master: Using P2PHANDSHAKE mode (no HTTP metadata server)")
    else:
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

    # Read offload configuration from config
    offload_conf = conf.backend.MooncakeStore.get("offload", {})
    enable_offload = offload_conf.get("enabled", False)

    cmd = [
        "mooncake_master",
        "-client_ttl=30",
        "-default_kv_lease_ttl=999999",
        "-default_kv_soft_pin_ttl=999999",
        "--allow_evict_soft_pinned_objects=false",
        f"--rpc_port={master_server_port}",
    ]

    if not use_p2p_handshake:
        # Enable HTTP metadata server for non-P2P mode
        cmd.extend(
            [
                "--enable_http_metadata_server=true",
                f"--http_metadata_server_host={metadata_server_host}",
                f"--http_metadata_server_port={metadata_server_port}",
            ]
        )

    if enable_offload:
        eviction_high_watermark = offload_conf.get("eviction_high_watermark_ratio", 0.9)
        eviction_ratio = offload_conf.get("eviction_ratio", 0.1)

        # Validate offload parameters
        if not (0.0 < eviction_high_watermark <= 1.0):
            raise ValueError(
                f"offload.eviction_high_watermark_ratio must be in (0.0, 1.0], got {eviction_high_watermark}"
            )
        if not (0.0 < eviction_ratio <= 1.0):
            raise ValueError(f"offload.eviction_ratio must be in (0.0, 1.0], got {eviction_ratio}")

        # Enable SSD offload: lower watermark to trigger eviction, offload on evict
        cmd.extend(
            [
                "--enable_offload=true",
                f"--eviction_high_watermark_ratio={eviction_high_watermark}",
                f"--eviction_ratio={eviction_ratio}",
                "--offload_on_evict=true",
                "--offload_force_evict=false",
                "--offloading_queue_limit=10000",
            ]
        )
        logger.info("mooncake_master: SSD offload enabled (offload_on_evict=true)")
    else:
        # Default: no eviction, no offload
        cmd.extend(
            [
                "--eviction_high_watermark_ratio=1.0",
                "--eviction_ratio=0.0",
            ]
        )

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

    # Start standalone mooncake_client for SSD offload if enabled
    if enable_offload:
        ssd_path = offload_conf.get("file_storage_path", "/tmp/mooncake_offload")
        client_port = str(offload_conf.get("client_port", 42052))
        local_buffer_size = str(offload_conf.get("local_buffer_size_bytes", 2147483648))
        use_uring = "1" if offload_conf.get("use_uring", False) else "0"
        heartbeat_interval = str(offload_conf.get("heartbeat_interval_seconds", 2))
        global_segment_size = str(conf.backend.MooncakeStore.get("global_segment_size", 4294967296))

        # Get local hostname
        local_hostname = conf.backend.MooncakeStore.get("local_hostname", "")
        if not local_hostname:
            try:
                from transfer_queue.utils.zmq_utils import get_node_ip_address

                local_hostname = get_node_ip_address()
            except Exception:
                import socket

                local_hostname = socket.gethostbyname(socket.gethostname())

        os.makedirs(ssd_path, exist_ok=True)

        client_env = os.environ.copy()
        client_env["MOONCAKE_OFFLOAD_FILE_STORAGE_PATH"] = ssd_path
        client_env["MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES"] = local_buffer_size
        client_env["MOONCAKE_OFFLOAD_USE_URING"] = use_uring
        client_env["MOONCAKE_OFFLOAD_HEARTBEAT_INTERVAL_SECONDS"] = heartbeat_interval

        if use_p2p_handshake:
            metadata_server_url = "P2PHANDSHAKE"
        else:
            metadata_server_url = f"http://{metadata_server_host}:{metadata_server_port}/metadata"
        master_address = f"{master_server_parsed.hostname}:{master_server_port}"

        client_cmd = [
            "mooncake_client",
            f"-host={local_hostname}",
            f"-global_segment_size={global_segment_size}",
            f"-master_server_address={master_address}",
            f"-metadata_server={metadata_server_url}",
            f"-protocol={conf.backend.MooncakeStore.get('protocol', 'tcp')}",
            "-enable_offload=true",
            f"-port={client_port}",
        ]

        client_log_path = "/tmp/mooncake_client.log"
        with open(client_log_path, "w") as client_log:
            client_process = subprocess.Popen(
                client_cmd,
                stdout=client_log,
                stderr=subprocess.STDOUT,
                env=client_env,
                start_new_session=True,
            )
            time.sleep(5)

        if client_process.poll() is None:
            logger.info(
                f"mooncake_client started for SSD offload, PID: {client_process.pid}. "
                f"SSD path: {ssd_path}. Logs: {client_log_path}"
            )
        else:
            # Offload client is a required component when offload is enabled.
            # Hard fail to prevent silent degradation.
            error_msg = ""
            try:
                with open(client_log_path) as f:
                    error_msg = f.read()
            except Exception as e:
                error_msg = f"Failed to read log file: {e}"

            # Terminate the master process since offload cannot work without the client
            process.terminate()
            raise RuntimeError(
                f"mooncake_client exited unexpectedly (exit code: {client_process.returncode}). "
                f"SSD offload is enabled but the offload client failed to start. "
                f"Check {client_log_path} for details. Output:\n{error_msg}"
            )

    # Return structured resources for lifecycle management
    if enable_offload:
        return {"master_process": process, "offload_client_process": client_process}
    return process
