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
import os
import shutil
import socket
import subprocess
from typing import Any

import ray
from omegaconf import DictConfig

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("TQ_LOGGING_LEVEL", logging.WARNING))


def get_local_ip_addresses() -> list[str]:
    """Get all local IP addresses including 127.0.0.1.

    Returns:
        List of local IP addresses, with 127.0.0.1 first.
    """
    ips = ["127.0.0.1"]

    try:
        hostname = socket.gethostname()
        # Add hostname resolution
        try:
            host_ip = socket.gethostbyname(hostname)
            if host_ip not in ips:
                ips.append(host_ip)
        except socket.gaierror:
            pass

        # Get all network interfaces
        import netifaces

        for interface in netifaces.interfaces():
            try:
                addrs = netifaces.ifaddresses(interface)
                if netifaces.AF_INET in addrs:
                    for addr_info in addrs[netifaces.AF_INET]:
                        ip = addr_info.get("addr")
                        if ip and ip not in ips:
                            ips.append(ip)
            except (ValueError, KeyError):
                continue
    except ImportError:
        # Fallback if netifaces is not available
        try:
            # Try to get IP by connecting to an external address
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                # Doesn't need to be reachable
                s.connect(("8.8.8.8", 80))
                ip = s.getsockname()[0]
                if ip not in ips:
                    ips.append(ip)
            except Exception:
                pass
            finally:
                s.close()
        except Exception:
            pass

    return ips


def check_port_connectivity(host: str, port: int, timeout: float = 2.0) -> bool:
    """Check if a TCP port is reachable on given host.

    Args:
        host: Host IP address to check
        port: Port number to check
        timeout: Connection timeout in seconds

    Returns:
        True if port is reachable, False otherwise
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


def find_reachable_host(port: int, timeout: float = 1.0) -> str | None:
    """Find a reachable local host IP address for given port.

    Tries all local IP addresses in order and returns the first one
    that has the given port open.

    Args:
        port: Port number to check
        timeout: Connection timeout in seconds per check

    Returns:
        The first reachable host IP address, or None if none found.
    """
    local_ips = get_local_ip_addresses()
    logger.info(f"Checking port {port} on local IPs: {local_ips}")

    for ip in local_ips:
        if check_port_connectivity(ip, port, timeout):
            logger.info(f"Found reachable host: {ip}:{port}")
            return ip

    logger.warning(f"No reachable host found for port {port}")
    return None


def _parse_remote_h2d_device_ids(worker_args: str) -> str | None:
    """Parse --remote_h2d_device_ids parameter from worker_args string.

    Args:
        worker_args: Worker arguments string, e.g., "--arg1 value1 --remote_h2d_device_ids 0,1,2,3"

    Returns:
        The device IDs string if found and valid, None otherwise.

    Raises:
        RuntimeError: If --remote_h2d_device_ids flag is found but has invalid format.
    """
    if not worker_args:
        return None

    args_list = worker_args.split()

    # Find the index of --remote_h2d_device_ids
    try:
        idx = args_list.index("--remote_h2d_device_ids")
    except ValueError:
        return None

    # Check if there's a value after the flag
    if idx + 1 >= len(args_list):
        raise RuntimeError("--remote_h2d_device_ids flag found but no value provided")

    device_ids = args_list[idx + 1]

    # Validate the format: comma-separated digits
    if not device_ids:
        raise RuntimeError("Empty device IDs value after --remote_h2d_device_ids")

    # Validate each segment is a digit
    parts = device_ids.split(",")
    for part in parts:
        if not part.isdigit():
            raise RuntimeError(
                f"Invalid device ID format: '{device_ids}'. Expected comma-separated digits (e.g., '0,1,2,3')."
            )

    return device_ids


def start_datasystem_worker(
    worker_address: str,
    metastore_address: str,
    is_head: bool,
    worker_args: str = "",
) -> None:
    """Start Yuanrong datasystem worker in metastore mode.

    Args:
        worker_address: Worker address in format host:port
        metastore_address: Metastore address in format host:port
        is_head: Whether this node should start metastore service
        worker_args: Additional arguments to append to dscli start command

    Raises:
        RuntimeError: If dscli command fails
    """
    if not shutil.which("dscli"):
        raise RuntimeError("dscli executable not found in PATH. Please run `pip install openyuanrong-datasystem`.")

    cmd = ["dscli", "start", "-w", "--worker_address", worker_address]
    cmd.extend(["--metastore_address", metastore_address])
    if is_head:
        cmd.extend(["--start_metastore_service", "true"])

    # Built-in default options
    cmd.extend(["--arena_per_tenant", "1", "--enable_worker_worker_batch_get", "true"])

    # Append worker_args if provided
    if worker_args:
        cmd.extend(worker_args.split())

    node_type = "head node" if is_head else "worker node"
    logger.info(f"Starting Yuanrong datasystem ({node_type}) at {worker_address}, worker_args={worker_args}")

    # Build environment with ASCEND_RT_VISIBLE_DEVICES if specified
    env = None
    device_ids = _parse_remote_h2d_device_ids(worker_args)
    if device_ids:
        env = os.environ.copy()
        env["ASCEND_RT_VISIBLE_DEVICES"] = device_ids
        logger.info(
            f"Setting ASCEND_RT_VISIBLE_DEVICES={device_ids} for dscli subprocess ({node_type} at {worker_address})"
        )

    try:
        ds_result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=90,
            env=env,
        )
    except subprocess.TimeoutExpired as err:
        raise RuntimeError(f"dscli start timed out: {err}") from err

    if ds_result.returncode == 0 and "[  OK  ]" in ds_result.stdout:
        logger.info(
            f"dscli started Yuanrong datasystem ({node_type}, metastore mode) at {worker_address} successfully."
        )
    else:
        raise RuntimeError(
            f"Failed to start datasystem ({node_type}, metastore mode) at {worker_address}. "
            f"Return code: {ds_result.returncode}, Output: {ds_result.stdout}"
        )


def stop_datasystem_worker(worker_address: str) -> None:
    """Stop Yuanrong datasystem worker.

    Args:
        worker_address: Worker address in format host:port
    """
    if worker_address:
        try:
            result = subprocess.run(
                ["dscli", "stop", "--worker_address", worker_address],
                timeout=90,
                capture_output=True,
            )
            if result.returncode == 0:
                logger.info(f"Stopped datasystem worker at {worker_address} via dscli stop")
            else:
                error_msg = (result.stderr or result.stdout or b"").decode()
                logger.warning(
                    f"Failed to stop datasystem worker at {worker_address}. "
                    f"Return code: {result.returncode}, Error: {error_msg}"
                )
        except subprocess.TimeoutExpired as err:
            logger.warning(f"dscli stop timed out for {worker_address}: {err}")
        except Exception as e:
            logger.warning(f"Failed to stop datasystem worker via dscli: {e}")


@ray.remote(num_cpus=0.1)
class YuanrongWorkerActor:
    """Ray actor to manage Yuanrong datasystem worker on a node.

    This actor runs on each node in the Ray cluster and is responsible for
    starting and stopping the Yuanrong datasystem worker process on that node.

    The actor determines its own rank and role (head or worker) by finding the
    intersection of local IP addresses with the provided node IPs.
    """

    def __init__(self, node_ips: list[str], worker_port: int, metastore_port: int, worker_args: str = ""):
        """Initialize the Yuanrong worker actor.

        Args:
            node_ips: List of all node IPs in the Ray cluster
            worker_port: Port for the datasystem worker
            metastore_port: Port for the metastore service (on head node)
            worker_args: Additional arguments to append to dscli start command

        Raises:
            RuntimeError: If cannot determine this node's IP from node_ips
        """
        local_ips = get_local_ip_addresses()
        self.my_ip = None

        # Find the intersection between local IPs and node_ips
        for ip in node_ips:
            if ip in local_ips:
                self.my_ip = ip
                break

        if self.my_ip is None:
            raise RuntimeError(f"Cannot determine local node IP. Local IPs: {local_ips}, Cluster node IPs: {node_ips}")

        self.node_ips = node_ips
        self.worker_port = worker_port
        self.metastore_port = metastore_port
        self.worker_address = f"{self.my_ip}:{worker_port}"
        self.worker_args = worker_args

        # First node in the list is assumed to be the head node.
        # This assumption is based on how interface.py constructs node_ips from ray.nodes().
        self.head_node_ip = node_ips[0]
        self.metastore_address = f"{self.head_node_ip}:{metastore_port}"
        self.is_head = self.my_ip == self.head_node_ip

        logger.info(
            f"YuanrongWorkerActor initialized on node {self.my_ip}: "
            f"worker_address={self.worker_address}, "
            f"metastore_address={self.metastore_address}, is_head={self.is_head}, worker_args={self.worker_args}"
        )

    def start(self) -> str:
        """Start the datasystem worker on this node.

        Returns:
            The worker address.

        Raises:
            RuntimeError: If dscli command fails
        """
        logger.info(f"Starting datasystem worker at {self.worker_address}...")
        start_datasystem_worker(
            self.worker_address,
            metastore_address=self.metastore_address,
            is_head=self.is_head,
            worker_args=self.worker_args,
        )
        logger.info(f"Datasystem worker started successfully at {self.worker_address}")
        return self.worker_address

    def get_metastore_address(self) -> str:
        """Get the metastore address.

        Returns:
            The metastore address in format host:port
        """
        return self.metastore_address

    def get_node_ip(self) -> str:
        """Return the IP address of the node this actor is running on."""
        assert self.my_ip is not None
        return self.my_ip

    def stop(self) -> None:
        """Stop the datasystem worker on this node."""
        logger.info(f"Stopping datasystem worker at {self.worker_address}...")
        stop_datasystem_worker(self.worker_address)
        logger.info(f"Datasystem worker stopped successfully at {self.worker_address}")


def _kill_actors_and_placement_group(worker_actors: list, placement_group: Any) -> None:
    """Kill actors and remove placement group without stopping workers.

    Args:
        worker_actors: List of Yuanrong worker actors to kill
        placement_group: Placement group to remove
    """
    for actor in worker_actors:
        try:
            ray.kill(actor)
        except Exception:
            pass
    if placement_group:
        try:
            ray.util.remove_placement_group(placement_group)
        except Exception:
            pass


def cleanup_yuanrong_resources(storage_value: Any) -> None:
    """Stop Yuanrong workers and cleanup resources.

    Args:
        storage_value: Yuanrong storage dict containing worker_actors and placement_group
    """
    if not isinstance(storage_value, dict):
        logger.warning(f"Unexpected Yuanrong storage value: {storage_value}")
        return

    worker_actors = storage_value.get("worker_actors", [])
    placement_group = storage_value.get("placement_group")

    try:
        if worker_actors:
            logger.info(f"Cleaning up Yuanrong backend (stopping {len(worker_actors)} workers)...")

            # Stop worker nodes (all except head node 0) in parallel first
            stop_exceptions = []
            if len(worker_actors) > 1:
                logger.info(f"Stopping {len(worker_actors) - 1} worker nodes (excluding head) in parallel...")
                stop_refs = [actor.stop.remote() for actor in worker_actors[1:]]
                for idx, stop_ref in enumerate(stop_refs, start=1):
                    try:
                        ray.get(stop_ref)
                    except Exception as e:
                        stop_exceptions.append(e)
                        logger.warning(f"Failed to stop worker node actor {idx}: {e}")
                if len(stop_exceptions) < len(stop_refs):
                    logger.info("Completed stop requests for non-head worker nodes")

            # Then stop head node (actor 0) which runs metastore service
            logger.info("Stopping head node with metastore service...")
            try:
                ray.get(worker_actors[0].stop.remote())
                logger.info("Head node stopped successfully")
            except Exception as e:
                stop_exceptions.append(e)
                logger.warning(f"Failed to stop head node actor: {e}")

            if stop_exceptions:
                logger.warning(f"Encountered {len(stop_exceptions)} errors while stopping workers")
    finally:
        # Kill actors and remove placement group even if graceful stop fails.
        _kill_actors_and_placement_group(worker_actors, placement_group)
        if placement_group:
            logger.info("Removed Yuanrong placement group")


def initialize_yuanrong_backend(conf: DictConfig) -> dict[str, Any]:
    """Initialize Yuanrong backend with metastore mode.

    This function sets up the Yuanrong datasystem workers across all Ray nodes
    using placement groups and actors.

    Args:
        conf: Configuration containing Yuanrong settings

    Returns:
        Dict containing worker_actors, metastore_address, and placement_group

    Raises:
        RuntimeError: If Ray nodes not found or initialization fails
    """
    # Get Ray cluster information
    nodes = ray.nodes()
    if not nodes:
        raise RuntimeError("No Ray nodes found. Is Ray initialized?")

    # Filter to only alive nodes and get their IPs
    alive_nodes = [node for node in nodes if node.get("Alive", False)]
    if not alive_nodes:
        raise RuntimeError("No alive Ray nodes found")

    # Get driver node IP to use as head node
    driver_ip = ray.util.get_node_ip_address()
    head_node = None
    other_nodes = []

    # Separate head node (driver) from other nodes
    for node in alive_nodes:
        node_ip = node["NodeManagerAddress"]
        if node_ip == driver_ip:
            head_node = node
        else:
            other_nodes.append(node)

    if head_node is None:
        raise RuntimeError(f"Driver node {driver_ip} not found in alive nodes")

    # Reorder nodes: head node first, then others
    ordered_nodes = [head_node] + other_nodes

    # Extract node IPs in deterministic order
    node_ips = [node["NodeManagerAddress"] for node in ordered_nodes]
    worker_port = conf.backend.Yuanrong.worker_port
    metastore_port = conf.backend.Yuanrong.metastore_port
    worker_args = conf.backend.Yuanrong.get("worker_args", "")

    logger.info(f"Found {len(ordered_nodes)} alive Ray nodes: {node_ips}")

    # Create placement group using STRICT_SPREAD to ensure each bundle is on a distinct node
    bundles = [{"CPU": 0.1} for _ in ordered_nodes]

    pg = ray.util.placement_group(bundles, strategy="STRICT_SPREAD")
    try:
        ray.get(pg.ready(), timeout=60)
    except ray.exceptions.GetTimeoutError as e:
        try:
            ray.util.remove_placement_group(pg)
        except Exception as cleanup_error:
            logger.warning(f"Failed to remove placement group after readiness timeout: {cleanup_error}")
        raise RuntimeError(
            "Timed out waiting for Yuanrong placement group to become ready. "
            f"Requested strategy=STRICT_SPREAD, bundles={bundles}. "
            "This may be due to insufficient cluster capacity."
        ) from e
    except Exception as e:
        try:
            ray.util.remove_placement_group(pg)
        except Exception as cleanup_error:
            logger.warning(f"Failed to remove placement group after scheduling failure: {cleanup_error}")
        raise RuntimeError(
            f"Failed to create Yuanrong placement group. Requested strategy=STRICT_SPREAD, bundles={bundles}."
        ) from e

    logger.info(f"Created placement group with {len(bundles)} bundles using STRICT_SPREAD")

    try:
        # Create all worker actors using placement group
        # Without node resources, actor scheduling order is not guaranteed to match node order
        # We'll identify head node actor by checking which node it runs on
        worker_actors = []
        for rank in range(len(ordered_nodes)):
            actor = YuanrongWorkerActor.options(  # type: ignore[attr-defined]
                placement_group=pg,
                placement_group_bundle_index=rank,
            ).remote(node_ips, worker_port, metastore_port, worker_args)
            worker_actors.append(actor)

        logger.info(f"Created {len(worker_actors)} YuanrongWorkerActor instances")

        # Find which actor is running on the head node (driver IP)
        # The head node actor needs to start first to initialize metastore service
        head_actor_index = None
        for idx, actor in enumerate(worker_actors):
            try:
                node_ip = ray.get(actor.get_node_ip.remote())
                if node_ip == driver_ip:
                    head_actor_index = idx
                    break
            except Exception:
                pass

        if head_actor_index is None:
            logger.warning("Could not identify head node actor, using actor 0 as default")
            head_actor_index = 0

        logger.info(f"Head node actor identified: actor {head_actor_index}")

        # Start head worker first to initialize metastore service
        logger.info("Starting head worker to initialize metastore...")
        ray.get(worker_actors[head_actor_index].start.remote())
        metastore_address = ray.get(worker_actors[head_actor_index].get_metastore_address.remote())
        logger.info(f"Head worker started, metastore address: {metastore_address}")

        # Start remaining worker actors in parallel
        other_actors = [worker_actors[i] for i in range(len(worker_actors)) if i != head_actor_index]
        if other_actors:
            logger.info(f"Starting {len(other_actors)} worker actors in parallel...")
            ray.get([actor.start.remote() for actor in other_actors])

        logger.info(
            f"Yuanrong backend started successfully: metastore at {metastore_address}, workers on {len(node_ips)} nodes"
        )

        return {
            "worker_actors": worker_actors,
            "metastore_address": metastore_address,
            "placement_group": pg,
        }
    except Exception as e:
        # Cleanup on initialization failure: attempt graceful stop of started workers first
        logger.error(f"Failed to start Yuanrong workers: {e}, cleaning up...")

        # Try to gracefully stop workers that may have already started
        if worker_actors:
            stop_exceptions = []
            # Stop worker nodes (all except head node 0) first
            if len(worker_actors) > 1:
                stop_refs = [actor.stop.remote() for actor in worker_actors[1:]]
                for idx, stop_ref in enumerate(stop_refs, start=1):
                    try:
                        ray.get(stop_ref, timeout=30)
                    except Exception as stop_e:
                        stop_exceptions.append(stop_e)
                        logger.warning(f"Failed to stop worker node actor {idx}: {stop_e}")
            # Stop head node (actor 0)
            try:
                ray.get(worker_actors[0].stop.remote(), timeout=30)
            except Exception as stop_e:
                stop_exceptions.append(stop_e)
                logger.warning(f"Failed to stop head node actor: {stop_e}")

            if stop_exceptions:
                logger.warning(f"Encountered {len(stop_exceptions)} errors during graceful worker stop")

        # Then kill actors and remove placement group
        _kill_actors_and_placement_group(worker_actors, pg)
        raise
