"""NIXL UCX agent, registered buffers, and one-sided READ completion."""

from __future__ import annotations

import ctypes
import os
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from transfer_queue.storage.payload_transfer.base import TransferError
from transfer_queue.utils.logging_utils import get_logger

logger = get_logger(__name__)
MAX_REGISTERED_BYTES = 1024**3
MAX_REGISTERED_REGIONS = 256
# An agent whose native request cannot be released must retain its memory owners.
_UNDRAINED_AGENTS: list[NixlRuntime] = []


class ReadAccessUnknown(TransferError):
    """A READ may still access both buffers."""


@dataclass(eq=False)
class _Region:
    buffer: bytearray
    address: int
    registration: Any


class NixlRuntime:
    def __init__(self, ucx_env=None, timeout=180):
        os.environ.update({str(key): str(value) for key, value in (ucx_env or {}).items()})
        try:
            from nixl import nixl_agent, nixl_agent_config
        except ImportError as exc:
            raise TransferError("install NIXL Python bindings with the UCX plugin") from exc
        self.name = f"tq-{socket.gethostname()}-{os.getpid()}-{uuid4().hex[:12]}"
        self._agent = nixl_agent(
            self.name,
            nixl_agent_config(enable_prog_thread=True, backends=["UCX"]),
        )
        if "UCX" not in self._agent.backends:
            raise TransferError("NIXL UCX plugin is unavailable")
        self._timeout = timeout
        self._lock = threading.RLock()
        self._read_lock = threading.Lock()
        self._regions: dict[int, _Region] = {}
        self._registered_bytes = 0
        self._peers: set[str] = set()
        self._free_exports: list[_Region] = []
        self._unreleased: list[tuple[Any, _Region]] = []
        self._blocked_peers: set[str] = set()

    def endpoint(self):
        with self._lock:
            return {"agent": self.name, "metadata": self._agent.get_agent_metadata()}

    def connect(self, peer):
        with self._lock:
            if peer["agent"] not in self._peers:
                self._load_peer(peer["agent"], peer["metadata"])

    def _load_peer(self, name, metadata):
        if name in self._blocked_peers:
            raise TransferError(f"cannot replace metadata for {name} while a READ may be active")
        # Load partial metadata without disconnecting the notification endpoint.
        loaded = self._agent.add_remote_agent(metadata)
        if isinstance(loaded, bytes):
            loaded = loaded.decode()
        if loaded != name:
            self._agent.remove_remote_agent(loaded)
            raise TransferError("NIXL endpoint agent does not match its metadata")
        self._peers.add(name)

    def send_notification(self, peer, message):
        with self._lock:
            self.connect(peer)
            # The agent has only UCX. NIXL 1.5's explicit backend argument is incompatible
            # with its Python binding, so let the agent select its single plugin.
            self._agent.send_notif(peer["agent"], message)

    def poll_notifications(self):
        with self._lock:
            return self._agent.get_new_notifs(backends=["UCX"])

    def _allocate(self, size):
        if type(size) is not int or not 0 < size <= MAX_REGISTERED_BYTES:
            raise TransferError("invalid NIXL buffer size")
        if self._registered_bytes + size > MAX_REGISTERED_BYTES or len(self._regions) >= MAX_REGISTERED_REGIONS:
            raise TransferError("NIXL registered buffer budget exhausted")
        buffer = bytearray(size)
        address = ctypes.addressof(ctypes.c_ubyte.from_buffer(buffer))
        registration = self._agent.register_memory([(address, size, 0, "")], mem_type="DRAM", backends=["UCX"])
        if registration is None:
            raise TransferError("NIXL memory registration failed")
        region = _Region(buffer, address, registration)
        self._regions[address] = region
        self._registered_bytes += size
        return region

    def publish(self, size, fill):
        """Publish private bytes whose registration stays live until release."""
        with self._lock:
            if type(size) is not int or not 0 < size <= MAX_REGISTERED_BYTES:
                raise TransferError("invalid NIXL export size")
            region = min(
                (region for region in self._free_exports if len(region.buffer) >= size),
                key=lambda region: len(region.buffer),
                default=None,
            )
            if region is None:
                region = self._allocate(1 << (size - 1).bit_length())
            else:
                self._free_exports.remove(region)
            try:
                fill(region.buffer)
                metadata = self._agent.get_partial_agent_metadata(
                    region.registration, inc_conn_info=True, backends=["UCX"]
                )
            except BaseException:
                self.release_export(region.address)
                raise
            return {"address": region.address, "size": size, "metadata": metadata}

    def release_export(self, address):
        with self._lock:
            self._free_exports.append(self._regions[address])

    def release(self, address):
        with self._lock:
            region = self._regions[address]
            try:
                self._agent.deregister_memory(region.registration, backends=["UCX"])
            except Exception:
                logger.exception("NIXL deregistration failed. Retaining the buffer until agent shutdown.")
                return
            del self._regions[address]
            self._registered_bytes -= len(region.buffer)

    def read(self, peer, offer):
        """READ into a private buffer that can outlive the transport after deregistration."""
        size, address = offer["size"], offer["address"]
        if type(address) is not int or address <= 0:
            raise TransferError("invalid NIXL remote buffer address")
        name = peer["agent"]
        # Updating a peer's metadata while its READ is active invalidates that request.
        with self._read_lock:
            with self._lock:
                if name in self._blocked_peers:
                    raise TransferError(f"NIXL peer {name} has an unresolved READ")
                self._load_peer(name, offer["metadata"])
                region = self._allocate(size)
            handle = None
            try:
                with self._lock:
                    local = self._agent.get_xfer_descs([(region.address, size, 0)], mem_type="DRAM")
                    remote = self._agent.get_xfer_descs([(address, size, 0)], mem_type="DRAM")
                    handle = self._agent.initialize_xfer("READ", local, remote, name, backends=["UCX"])
                    status = self._agent.transfer(handle)
                deadline = time.monotonic() + self._timeout
                while status == "PROC":
                    if time.monotonic() >= deadline:
                        raise TransferError("NIXL READ timed out")
                    time.sleep(0.001)
                    with self._lock:
                        status = self._agent.check_xfer_state(handle)
                if status != "DONE":
                    raise TransferError(f"NIXL READ failed with status {status!r}")
                return region.buffer
            finally:
                with self._lock:
                    if handle is not None:
                        try:
                            self._agent.release_xfer_handle(handle)
                        except Exception as exc:
                            self._unreleased.append((handle, region))
                            self._blocked_peers.add(name)
                            raise ReadAccessUnknown(f"NIXL could not release the READ to {name}") from exc
                    self.release(region.address)

    def close(self):
        with self._read_lock, self._lock:
            if self._agent is None:
                return
            remaining = []
            for handle, region in self._unreleased:
                try:
                    self._agent.release_xfer_handle(handle)
                except Exception:
                    remaining.append((handle, region))
            self._unreleased = remaining
            if remaining:
                if self not in _UNDRAINED_AGENTS:
                    _UNDRAINED_AGENTS.append(self)
                logger.error("NIXL agent still has active READs. Retaining it and its buffers until process exit.")
                return
            # Keep all exported buffers alive while native teardown revokes their registrations.
            self._agent = None
            self._free_exports.clear()
            self._regions.clear()
            self._registered_bytes = 0
            self._peers.clear()
