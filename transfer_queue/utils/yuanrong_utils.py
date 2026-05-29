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
import socket
from typing import Any

import ray

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


def kill_actors_and_placement_group(worker_actors: list, placement_group: Any) -> None:
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
    head_actor_index = storage_value.get("head_actor_index", 0)

    try:
        if worker_actors:
            logger.info(f"Cleaning up Yuanrong backend (stopping {len(worker_actors)} workers)...")

            # Stop worker nodes (non-head) in parallel first, then head node (metastore service)
            stop_exceptions = []
            other_indices = [i for i in range(len(worker_actors)) if i != head_actor_index]
            if other_indices:
                logger.info(f"Stopping {len(other_indices)} worker nodes in parallel...")
                stop_refs = [worker_actors[idx].stop.remote() for idx in other_indices]
                for idx, stop_ref in zip(other_indices, stop_refs, strict=False):
                    try:
                        ray.get(stop_ref)
                    except Exception as e:
                        stop_exceptions.append(e)
                        logger.warning(f"Failed to stop worker node actor {idx}: {e}")
                if len(stop_exceptions) < len(stop_refs):
                    logger.info("Completed stop requests for non-head worker nodes")

            # Then stop head node which runs metastore service
            logger.info("Stopping head node with metastore service...")
            try:
                ray.get(worker_actors[head_actor_index].stop.remote())
                logger.info("Head node stopped successfully")
            except Exception as e:
                stop_exceptions.append(e)
                logger.warning(f"Failed to stop head node actor: {e}")

            if stop_exceptions:
                logger.warning(f"Encountered {len(stop_exceptions)} errors while stopping workers")
    finally:
        # Kill actors and remove placement group even if graceful stop fails.
        kill_actors_and_placement_group(worker_actors, placement_group)
        if placement_group:
            logger.info("Removed Yuanrong placement group")
