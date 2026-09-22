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

"""NIXL runtime for the SimpleStorage host payload fast path."""

from __future__ import annotations

import ctypes
import os
import socket
import threading
import time
import weakref
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import numpy as np

from transfer_queue.storage.payload_transfer.base import PayloadTransferError
from transfer_queue.utils.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_NIXL_TRANSFER_TIMEOUT_SECONDS = 180
_POLL_INTERVAL_SECONDS = 0.0005


class NixlError(PayloadTransferError):
    """A NIXL runtime operation failed."""


@dataclass
class _RegisteredReceiveBuffer:
    buffer: bytearray
    registration: Any
    address: int
    lease_finalizer: weakref.finalize | None = None

    @property
    def capacity(self) -> int:
        return len(self.buffer)


@dataclass
class _FrameSource:
    owner: bytearray | memoryview
    registration: Any | None
    address: int
    size: int


def _buffer_address(buffer: bytearray | memoryview) -> int:
    """Return the address of a non-empty writable contiguous host buffer."""
    view = memoryview(buffer)
    if not view.nbytes:
        raise NixlError("NIXL does not support an empty memory region")
    if view.readonly or not view.c_contiguous:
        raise NixlError("NIXL memory regions must be writable and C-contiguous")
    return ctypes.addressof(ctypes.c_ubyte.from_buffer(view))


def _frame_regions(address: int, frame_sizes: tuple[int, ...]) -> list[tuple[int, int, int]]:
    """Build non-empty frame regions without narrowing cumulative offsets."""
    regions = []
    offset = 0
    for size in frame_sizes:
        if size:
            regions.append((address + offset, size, 0))
        offset += size
    return regions


def _frame_views(owner: Any, frame_sizes: tuple[int, ...]) -> tuple[memoryview, ...]:
    """Split a contiguous owner into frame views, preserving empty frames."""
    view = memoryview(owner).cast("B")
    frames = []
    offset = 0
    for size in frame_sizes:
        frames.append(view[offset : offset + size])
        offset += size
    return tuple(frames)


def _configure_ucx_environment(ucx_env_vars: dict[str, object] | None) -> dict[str, str]:
    """Apply YAML UCX settings before the NIXL agent reads its environment."""
    config = {str(key): str(value) for key, value in (ucx_env_vars or {}).items()}
    os.environ.update(config)
    return config


def _warn_if_tcp_fallback_possible() -> None:
    """Warn when the configured UCX transports allow a TCP payload fallback."""
    ucx_tls = os.environ.get("UCX_TLS")
    if not ucx_tls:
        return

    transports = {item.strip() for item in ucx_tls.split(",") if item.strip()}
    if "tcp" not in transports and "all" not in transports:
        return

    logger.warning("NIXL-UCX may fall back to TCP; check UCX logs for the actual transport (UCX_TLS=%s)", ucx_tls)


class NixlRuntime:
    """Own the NIXL agent, registered memory, and transfer completion."""

    def __init__(self, ucx_env_vars: dict[str, object] | None = None):
        _configure_ucx_environment(ucx_env_vars)
        try:
            from nixl import nixl_agent, nixl_agent_config
        except Exception as exc:  # pragma: no cover - optional dependency
            raise NixlError("NIXL Python bindings are unavailable; install a NIXL build with the UCX backend") from exc

        self._agent_name = self._make_agent_name()
        try:
            config = nixl_agent_config(
                enable_prog_thread=True,
                enable_listen_thread=True,
                listen_port=0,
                backends=["UCX"],
            )
            self._agent = nixl_agent(self._agent_name, config)
        except Exception as exc:  # pragma: no cover - native runtime dependent
            raise NixlError(f"failed to initialize the NIXL UCX backend: {exc}") from exc

        if "UCX" not in getattr(self._agent, "backends", {}):
            raise NixlError("NIXL UCX backend is not available")

        _warn_if_tcp_fallback_possible()
        logger.info(
            "SimpleStorage payload transfer selected: nixl-ucx device=%s gid_index=%s tls=%s",
            os.environ.get("UCX_NET_DEVICES", "ucx-auto"),
            os.environ.get("UCX_IB_GID_INDEX", "ucx-auto"),
            os.environ.get("UCX_TLS", "ucx-auto"),
        )

        self._receives: dict[str, _RegisteredReceiveBuffer | None] = {}
        self._idle_receive_buffers: list[_RegisteredReceiveBuffer] = []
        self._leased_receive_buffers: dict[int, _RegisteredReceiveBuffer] = {}
        self._quarantined_receive_buffers: list[_RegisteredReceiveBuffer] = []
        self._retained_resources: list[tuple[Any | None, list[_FrameSource]]] = []
        self._registered_sources: dict[int, _FrameSource] = {}
        self._remote_metadata: dict[str, bytes] = {}
        self._failed_peers: set[str] = set()
        self._peer_executors: dict[str, ThreadPoolExecutor] = {}
        self._lock = threading.RLock()
        self._closing = threading.Event()
        self._closed = False
        self._timeout_seconds = DEFAULT_NIXL_TRANSFER_TIMEOUT_SECONDS
        self._registered_bytes = 0
        self._quarantined_bytes = 0
        self._registration_seconds = 0.0
        self._data_transfer_seconds = 0.0
        self._total_seconds = 0.0

    @staticmethod
    def _make_agent_name() -> str:
        return f"tq-{socket.gethostname()}-{os.getpid()}-{uuid4().hex[:12]}"

    @property
    def agent_name(self) -> str:
        return self._agent_name

    @property
    def diagnostics(self) -> dict[str, float | int]:
        """Return the small runtime-only metric set from the design."""
        with self._lock:
            return {
                "registration_seconds": self._registration_seconds,
                "data_transfer_seconds": self._data_transfer_seconds,
                "total_seconds": self._total_seconds,
                "registered_bytes": self._registered_bytes,
                "quarantined_bytes": self._quarantined_bytes,
            }

    def endpoint_metadata(self) -> bytes:
        with self._lock:
            self._ensure_open()
            return self._agent.get_agent_metadata()

    def prepare_receive(self, descriptor: Any) -> dict[str, Any]:
        """Prepare frame-native remote descriptors and publish full metadata."""
        descriptor.validate()
        with self._lock:
            self._ensure_open()
            if descriptor.transfer_id in self._receives:
                raise NixlError(f"duplicate NIXL receive: {descriptor.transfer_id}")

            scratch = None
            try:
                if descriptor.payload_bytes:
                    scratch = self._acquire_receive_buffer(descriptor.payload_bytes)
                    regions = _frame_regions(scratch.address, descriptor.frame_sizes)
                    remote_descs = self._agent.get_xfer_descs(regions, mem_type="DRAM")
                    serialized = self._agent.get_serialized_descs(remote_descs)
                else:
                    serialized = b""
                metadata = self._agent.get_agent_metadata()
                self._receives[descriptor.transfer_id] = scratch
            except Exception as exc:
                if scratch is not None:
                    self._idle_receive_buffers.append(scratch)
                raise NixlError(f"failed to prepare NIXL receive buffer: {exc}") from exc

        return {
            "agent_name": self._agent_name,
            "agent_metadata": metadata,
            "frame_remote_descs": serialized,
            "payload_bytes": descriptor.payload_bytes,
        }

    def send(
        self,
        endpoint: dict[str, Any],
        token: dict[str, Any],
        descriptor: Any,
        frames: tuple[bytes | bytearray | memoryview, ...],
    ) -> Future[None]:
        """Submit one transfer to the remote peer's single-worker executor."""
        descriptor.validate()
        if tuple(memoryview(frame).nbytes for frame in frames) != descriptor.frame_sizes:
            raise NixlError(f"frame lengths do not match descriptor for {descriptor.transfer_id}")
        remote_name, metadata = self._validate_send_metadata(endpoint, token, descriptor)
        with self._lock:
            self._ensure_open()
            if remote_name in self._failed_peers:
                raise NixlError(f"NIXL peer session {remote_name!r} has failed")
            if not descriptor.payload_bytes:
                future: Future[None] = Future()
                future.set_result(None)
                return future
            executor = self._peer_executors.get(remote_name)
            if executor is None:
                executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"tq-nixl-{remote_name}")
                self._peer_executors[remote_name] = executor
            return executor.submit(
                self._send_scatter,
                remote_name,
                metadata,
                tuple(frames),
                token["frame_remote_descs"],
            )

    def receive(self, descriptor: Any) -> Future[tuple[memoryview, ...]]:
        """Return zero-copy frame views and lease the receive MR to their owner."""
        descriptor.validate()
        future: Future[tuple[memoryview, ...]] = Future()
        with self._lock:
            try:
                self._ensure_open()
                scratch = self._receives.pop(descriptor.transfer_id)
                if scratch is None:
                    future.set_result(tuple(memoryview(b"") for _ in descriptor.frame_sizes))
                    return future

                owner = np.frombuffer(scratch.buffer, dtype=np.uint8, count=descriptor.payload_bytes)
                key = id(scratch)
                self._leased_receive_buffers[key] = scratch
                scratch.lease_finalizer = weakref.finalize(owner, self._release_lease, key)
                future.set_result(_frame_views(owner, descriptor.frame_sizes))
            except KeyError:
                future.set_exception(NixlError(f"no prepared NIXL receive for {descriptor.transfer_id}"))
            except Exception as exc:
                future.set_exception(NixlError(f"failed to expose NIXL receive frames: {exc}"))
        return future

    def cancel_receive(self, transfer_id: str) -> None:
        """Release a receive that was never exposed to a possible WRITE."""
        with self._lock:
            scratch = self._receives.pop(transfer_id, None)
            if scratch is not None:
                self._idle_receive_buffers.append(scratch)

    def quarantine_receive(self, transfer_id: str) -> None:
        """Keep an exposed receiver out of the reuse pool until agent teardown."""
        with self._lock:
            scratch = self._receives.pop(transfer_id, None)
            if scratch is not None:
                self._quarantined_receive_buffers.append(scratch)
                self._quarantined_bytes += scratch.capacity

    def close(self) -> None:
        """Stop submissions, stop polling, then tear down the agent before owners."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._closing.set()
            executors = list(self._peer_executors.values())
            self._peer_executors.clear()

        for executor in executors:
            executor.shutdown(wait=True, cancel_futures=True)

        with self._lock:
            agent = self._agent
            self._agent = None
            retained = self._retained_resources
            self._retained_resources = []
            retained_handles = [handle for handle, _ in retained if handle is not None]
            retained_sources = [source for _, sources in retained for source in sources]
            retained.clear()

        # NIXL handles keep their agent alive. Drop them while every registered
        # owner is still retained, then tear down the agent before those owners.
        retained_handles.clear()
        del agent

        with self._lock:
            self._receives.clear()
            self._idle_receive_buffers.clear()
            self._leased_receive_buffers.clear()
            self._quarantined_receive_buffers.clear()
            self._registered_sources.clear()
            self._remote_metadata.clear()
            self._registered_bytes = 0
            self._quarantined_bytes = 0
        retained_sources.clear()

    def _acquire_receive_buffer(self, required_capacity: int) -> _RegisteredReceiveBuffer:
        for index, scratch in enumerate(self._idle_receive_buffers):
            if scratch.capacity >= required_capacity:
                return self._idle_receive_buffers.pop(index)

        buffer = bytearray(required_capacity)
        address = _buffer_address(buffer)
        started = time.monotonic()
        registration = self._agent.register_memory(
            [(address, required_capacity, 0, "")], mem_type="DRAM", backends=["UCX"]
        )
        self._registration_seconds += time.monotonic() - started
        if registration is None:
            raise NixlError("failed to register NIXL receive buffer")
        self._registered_bytes += required_capacity
        return _RegisteredReceiveBuffer(buffer, registration, address)

    def _register_frame_source(self, owner: bytearray | memoryview) -> _FrameSource:
        """Register one writable contiguous source while the runtime lock is held."""
        address = _buffer_address(owner)
        size = memoryview(owner).nbytes
        started = time.monotonic()
        registration = self._agent.register_memory([(address, size, 0, "")], mem_type="DRAM", backends=["UCX"])
        self._registration_seconds += time.monotonic() - started
        if registration is None:
            raise NixlError("failed to register NIXL source frame")
        self._registered_bytes += size
        source = _FrameSource(owner, registration, address, size)
        self._registered_sources[address] = source
        return source

    def _acquire_frame_source(self, frame: bytes | bytearray | memoryview) -> _FrameSource:
        view = memoryview(frame)

        # Staging can copy large payloads. Keep that work outside the runtime
        # lock so an unrelated peer is not blocked on Python memory copies.
        if view.readonly or not view.c_contiguous:
            owner: bytearray | memoryview = bytearray(view)
            with self._lock:
                self._ensure_open()
                return self._register_frame_source(owner)

        owner = view.cast("B")
        address = _buffer_address(owner)
        size = memoryview(owner).nbytes
        with self._lock:
            self._ensure_open()
            for scratch in self._leased_receive_buffers.values():
                if scratch.address <= address and address + size <= scratch.address + scratch.capacity:
                    return _FrameSource(owner, None, address, size)

            # NIXL resolves registrations by address. An overlapping external
            # source needs an independent staging owner before it can be
            # registered and cleaned up by this transfer.
            overlaps = any(
                address < source.address + source.size and source.address < address + size
                for source in self._registered_sources.values()
            )
            if not overlaps:
                return self._register_frame_source(owner)

        owner = bytearray(owner)
        with self._lock:
            self._ensure_open()
            return self._register_frame_source(owner)

    def _send_scatter(
        self,
        remote_name: str,
        metadata: bytes,
        frames: tuple[bytes | bytearray | memoryview, ...],
        serialized_remote_descs: bytes,
    ) -> None:
        started_total = time.monotonic()
        sources: list[_FrameSource] = []
        handle = None
        error_message = None
        try:
            with self._lock:
                self._ensure_open()
                if remote_name in self._failed_peers:
                    raise NixlError(f"NIXL peer session {remote_name!r} has failed")

            for frame in frames:
                if memoryview(frame).nbytes:
                    sources.append(self._acquire_frame_source(frame))

            with self._lock:
                self._ensure_open()
                if remote_name in self._failed_peers:
                    raise NixlError(f"NIXL peer session {remote_name!r} has failed")
                self._load_remote_metadata(remote_name, metadata)
                local_descs = self._agent.get_xfer_descs(
                    [(source.address, source.size, 0) for source in sources], mem_type="DRAM"
                )
                remote_descs = self._agent.deserialize_descs(serialized_remote_descs)
                handle = self._agent.initialize_xfer("WRITE", local_descs, remote_descs, remote_name, backends=["UCX"])
                started_transfer = time.monotonic()
                status = self._agent.transfer(handle)

            deadline = started_transfer + self._timeout_seconds
            while status == "PROC":
                if self._closing.is_set():
                    raise NixlError("NIXL runtime is closing")
                if time.monotonic() >= deadline:
                    raise NixlError(f"NIXL WRITE timed out after {self._timeout_seconds:g}s")
                with self._lock:
                    status = self._agent.check_xfer_state(handle)
                if status == "PROC":
                    time.sleep(_POLL_INTERVAL_SECONDS)
            with self._lock:
                self._data_transfer_seconds += time.monotonic() - started_transfer
            if status != "DONE":
                raise NixlError(f"NIXL WRITE failed with status {status!r}")
        except Exception as exc:
            error_message = f"NIXL H2H scatter WRITE failed: {exc}"
            with self._lock:
                self._failed_peers.add(remote_name)
                if handle is None:
                    self._retain_cleanup_failures(None, self._cleanup_sources(sources))
                else:
                    self._retained_resources.append((handle, sources))
                self._total_seconds += time.monotonic() - started_total
            handle = None
            sources = []

        if error_message is not None:
            # A retained Future must not keep the native handle and agent alive.
            raise NixlError(error_message)

        with self._lock:
            cleanup_failed = False
            try:
                self._agent.release_xfer_handle(handle)
                handle = None
            except Exception as exc:
                logger.warning("failed to release completed NIXL transfer handle: %s", exc)
                self._retained_resources.append((handle, sources))
                cleanup_failed = True
            if not cleanup_failed:
                failed_sources = self._cleanup_sources(sources)
                self._retain_cleanup_failures(None, failed_sources)
                cleanup_failed = bool(failed_sources)
            if cleanup_failed:
                self._failed_peers.add(remote_name)
            self._total_seconds += time.monotonic() - started_total

    def _load_remote_metadata(self, remote_name: str, metadata: bytes) -> None:
        previous = self._remote_metadata.get(remote_name)
        if previous == metadata:
            return
        if previous is not None:
            self._agent.remove_remote_agent(remote_name)
        loaded_name = self._agent.add_remote_agent(metadata)
        if isinstance(loaded_name, bytes):
            loaded_name = loaded_name.decode()
        if loaded_name != remote_name:
            raise NixlError(f"NIXL remote agent name mismatch: expected {remote_name!r}, got {loaded_name!r}")
        self._remote_metadata[remote_name] = metadata

    def _cleanup_sources(self, sources: list[_FrameSource]) -> list[_FrameSource]:
        failed = []
        for source in sources:
            if source.registration is None:
                continue
            try:
                self._agent.deregister_memory(source.registration, backends=["UCX"])
                self._registered_bytes -= source.size
                self._registered_sources.pop(source.address)
            except Exception as exc:
                logger.warning("failed to deregister NIXL source memory: %s", exc)
                failed.append(source)
        return failed

    def _retain_cleanup_failures(self, handle: Any | None, sources: list[_FrameSource]) -> None:
        if handle is not None or sources:
            self._retained_resources.append((handle, sources))

    def _release_lease(self, key: int) -> None:
        with self._lock:
            scratch = self._leased_receive_buffers.get(key)
            if scratch is None or self._closed:
                return
            self._leased_receive_buffers.pop(key)
            scratch.lease_finalizer = None
            self._idle_receive_buffers.append(scratch)

    def _ensure_open(self) -> None:
        if self._closed:
            raise NixlError("NIXL runtime is closed")

    @staticmethod
    def _validate_send_metadata(
        endpoint: dict[str, Any],
        token: dict[str, Any],
        descriptor: Any,
    ) -> tuple[str, bytes]:
        remote_name = str(token.get("agent_name") or endpoint.get("agent_name") or "")
        metadata = token.get("agent_metadata")
        if not remote_name or not isinstance(metadata, bytes):
            raise NixlError("NIXL receive token is missing current agent metadata")
        if int(token.get("payload_bytes", -1)) != descriptor.payload_bytes:
            raise NixlError("NIXL receive token length does not match descriptor")
        if not isinstance(token.get("frame_remote_descs"), bytes):
            raise NixlError("NIXL receive token is missing frame descriptors")
        return remote_name, metadata
