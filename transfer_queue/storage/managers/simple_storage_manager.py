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

import asyncio
import os
import time
import warnings
from collections import defaultdict
from collections.abc import Mapping
from functools import partial
from operator import itemgetter
from pathlib import Path
from typing import Any, Callable, NamedTuple

import torch
import zmq
from omegaconf import DictConfig
from tensordict import NonTensorStack, TensorDict

from transfer_queue.metadata import BatchMeta, extract_field_schema
from transfer_queue.storage.managers.base import StorageManager, StorageManagerFactory
from transfer_queue.storage.simple_storage import (
    KEY_NOT_FOUND_MARKER,
    StorageKeyNotFoundError,
)
from transfer_queue.utils.common import log_heavy_operation
from transfer_queue.utils.logging_utils import get_logger
from transfer_queue.utils.zmq_utils import (
    TQ_SOCKET_POOL_SIZE,
    ZMQMessage,
    ZMQRequestType,
    ZMQServerInfo,
    ZMQSocketPool,
    frame_nbytes,
    with_zmq_socket,
)

logger = get_logger(__name__)

TQ_SIMPLE_STORAGE_SEND_RECV_TIMEOUT = int(os.environ.get("TQ_SIMPLE_STORAGE_SEND_RECV_TIMEOUT", 200))  # seconds

# Attempts per storage-unit request, including the first. Each one must use a new connection: a
# DEALER silently queues messages for a peer it never reached, so no timeout can catch that.
TQ_SIMPLE_STORAGE_MAX_ATTEMPTS = int(os.environ.get("TQ_SIMPLE_STORAGE_MAX_ATTEMPTS", 3))

# Timeout for the post-failure probe, which only has to answer whether the unit still serves.
TQ_SIMPLE_STORAGE_PROBE_TIMEOUT = int(os.environ.get("TQ_SIMPLE_STORAGE_PROBE_TIMEOUT", 10))

# A selective dump or load sends one request per unit covering all of its rows, so its duration
# grows with the selection and shared-filesystem speed rather than with a batch.
TQ_SIMPLE_STORAGE_DUMP_TIMEOUT = int(os.environ.get("TQ_SIMPLE_STORAGE_DUMP_TIMEOUT", 3600))


class StorageUnitTimeout(RuntimeError):
    """A storage unit did not answer within the send/recv timeout.

    Distinct from an error the unit reported: only a missing answer is worth a new connection.
    """


def _describe_unit_state(body: dict[str, Any]) -> str:
    """Summarize a successful probe: the unit is serving again, plus its own counters.

    Draws no conclusion about the timed-out request: the counters are cumulative per
    operation and carry no request identity, so no value of them locates one request.
    """
    parts = [
        f"requests_arrived={body.get('requests_arrived')}",
        f"arrivals_by_op={body.get('arrivals_by_op')}",
        f"active_keys={body.get('active_keys')}",
        f"rss_gb={body.get('process_rss_bytes', 0) / 2**30:.2f}",
    ]
    op_stats = body.get("op_stats") or {}
    if op_stats:
        parts.append(f"completed={ {op: stats.get('request_count') for op, stats in op_stats.items()} }")
    else:
        # Only with Prometheus enabled; an empty dict would read as "served nothing".
        parts.append("completed=unavailable(prometheus_disabled)")
    return f"verdict=unit_serving_again ({' '.join(parts)})"


def _build_update_field_schema(
    global_indexes: list[int],
    per_unit: list[tuple[list[int], dict[str, dict[str, Any]]]],
) -> dict[str, dict[str, Any]]:
    """Assemble one batch-level field_schema from each unit's per-sample description.

    A unit only describes its own rows, so shapes are collected per global index and
    then ordered to match ``global_indexes``: the controller reads per_sample_shapes
    positionally against the indexes it is notified about.
    """
    dtypes: dict[str, Any] = {}
    shapes: dict[str, dict[int, tuple]] = defaultdict(dict)
    non_tensor: set[str] = set()
    for unit_indexes, described in per_unit:
        for field_name, description in described.items():
            unit_shapes = description["shapes"]
            if unit_shapes is None:
                non_tensor.add(field_name)
                continue
            dtypes[field_name] = description["dtype"]
            # Serialization turns the unit's tuples into lists; shapes are compared and hashed here.
            shapes[field_name].update(zip(unit_indexes, (tuple(s) for s in unit_shapes), strict=True))

    field_schema: dict[str, dict[str, Any]] = {}
    for field_name in non_tensor | set(shapes):
        # A column is only a tensor column when every unit stored tensors for it.
        if field_name in non_tensor:
            field_schema[field_name] = {"dtype": None, "shape": None, "is_nested": False, "is_non_tensor": True}
            continue
        ordered = [shapes[field_name][gi] for gi in global_indexes]
        is_nested = len(set(ordered)) > 1
        field_schema[field_name] = {
            "dtype": dtypes[field_name],
            "shape": None if is_nested else ordered[0],
            "is_nested": is_nested,
            "is_non_tensor": False,
        }
        if is_nested:
            field_schema[field_name]["per_sample_shapes"] = ordered
    return field_schema


_SU_SUBDIR = "simple_storage"
_SU_INFO_FILE = "storage_unit_info.json"

# Pre-bound decorator for storage-unit socket operations.
with_storage_unit_socket = with_zmq_socket(
    get_peer=lambda self, target: self.storage_unit_infos[target],
    get_pool=lambda self: self.storage_rpc_pool,
    resolve_target=lambda args, kwargs: kwargs.get("target_storage_unit"),
)

# Same endpoint as above but with the short diagnostic timeout, used only after a failure.
with_storage_unit_probe_socket = with_zmq_socket(
    get_peer=lambda self, target: self.storage_unit_infos[target],
    get_pool=lambda self: self.storage_probe_pool,
    resolve_target=lambda args, kwargs: kwargs.get("target_storage_unit"),
)

with_storage_unit_dump_socket = with_zmq_socket(
    get_peer=lambda self, target: self.storage_unit_infos[target],
    get_pool=lambda self: self.storage_dump_pool,
    resolve_target=lambda args, kwargs: kwargs.get("target_storage_unit"),
)


class RoutingGroup(NamedTuple):
    """Routing result for a single storage unit."""

    global_indexes: list[int]  # global indexes routed to this SU
    batch_positions: list[int]  # corresponding positions in the original batch


@StorageManagerFactory.register("SimpleStorage")
class AsyncSimpleStorageManager(StorageManager):
    """Asynchronous storage manager that handles multiple storage units.

    This manager provides async put/get/clear operations across multiple SimpleStorageUnit
    instances using ZMQ communication and dynamic socket management.
    """

    def __init__(
        self,
        controller_info: ZMQServerInfo,
        config: DictConfig,
        zmq_context: zmq.asyncio.Context | None = None,
    ):
        super().__init__(controller_info, config, zmq_context=zmq_context)
        # Storage-unit RPC, on whichever context the base class settled on.
        self.storage_rpc_pool = ZMQSocketPool(
            self.zmq_context,
            self.storage_manager_id,
            "put_get_socket",
            timeout=TQ_SIMPLE_STORAGE_SEND_RECV_TIMEOUT,
        )
        self.storage_probe_pool = ZMQSocketPool(
            self.zmq_context,
            f"{self.storage_manager_id}_probe",
            "put_get_socket",
            timeout=TQ_SIMPLE_STORAGE_PROBE_TIMEOUT,
            maxsize=1,
        )
        self.storage_dump_pool = ZMQSocketPool(
            self.zmq_context,
            f"{self.storage_manager_id}_dump",
            "put_get_socket",
            timeout=TQ_SIMPLE_STORAGE_DUMP_TIMEOUT,
        )

        self.config = config
        server_infos: ZMQServerInfo | dict[str, ZMQServerInfo] | None = config.get("zmq_info", None)

        if server_infos is None:
            server_infos = config.get("storage_unit_infos", None)
            if server_infos is not None:
                warnings.warn(
                    "The config entry `storage_unit_infos` will be deprecated in 0.1.7, please use `zmq_info` instead.",
                    category=DeprecationWarning,
                    stacklevel=2,
                )

        if server_infos is None:
            raise ValueError("AsyncSimpleStorageManager requires non-empty 'zmq_info' in config.")

        self.storage_unit_infos = self._register_servers(server_infos)
        self._warn_if_pool_can_exhaust_context(len(self.storage_unit_infos))

    def _warn_if_pool_can_exhaust_context(self, num_units: int) -> None:
        """Warn when the pool may park more sockets than the context can hold.

        The cap is per (owner, address), so it multiplies by storage-unit count while the
        context ceiling does not. At a few thousand units the product passes ZMQ_MAX_SOCKETS,
        where a lease fails with EMFILE instead of degrading.
        """
        try:
            budget = self.zmq_context.get(zmq.MAX_SOCKETS)
        except zmq.ZMQError:  # pragma: no cover - context already terminating
            return
        worst_case = (TQ_SOCKET_POOL_SIZE + 1) * num_units
        if worst_case > budget:
            logger.warning(
                f"[{self.storage_manager_id}]: storage RPC pool may hold up to "
                f"({TQ_SOCKET_POOL_SIZE} + 1 probe) x {num_units} = {worst_case} idle sockets, above this "
                f"context's ZMQ_MAX_SOCKETS ({budget}). Concurrent requests can then fail to "
                f"open a socket. Lower TQ_SOCKET_POOL_SIZE or raise TQ_CLIENT_ZMQ_MAX_SOCKETS "
                f"(with enough file descriptors, see ulimit -n)."
            )

    def _register_servers(self, server_infos: "ZMQServerInfo | dict[Any, ZMQServerInfo]"):
        """Register and validate server information.

        Args:
            server_infos: ZMQServerInfo | dict[Any, ZMQServerInfo])
                ZMQServerInfo or dict of server infos to register.

        Returns:
            Dictionary with server IDs as keys and ZMQServerInfo objects as values.

        Raises:
            ValueError: If server_infos format is invalid.
        """
        server_infos_transform = {}

        if isinstance(server_infos, ZMQServerInfo):
            server_infos_transform[server_infos.id] = server_infos
        elif isinstance(server_infos, Mapping):
            for k, v in server_infos.items():
                if not isinstance(v, ZMQServerInfo):
                    raise ValueError(f"Invalid server info for key {k}: {v}")
                server_infos_transform[v.id] = v
        else:
            raise ValueError(f"Invalid server infos: {server_infos}")

        return server_infos_transform

    def _group_by_hash(self, global_indexes: list[int]) -> dict[str, RoutingGroup]:
        """Group samples by global_idx % num_su, return {storage_id: RoutingGroup}.

        Routing depends solely on global_idx, independent of batch_size, key ordering,
        or number of calls. The same global_idx always routes to the same SU across
        put/get/clear operations.

        NOTE: Dynamic SU scaling requires a data migration mechanism (not yet supported).
        """
        storage_unit_keys = list(self.storage_unit_infos.keys())
        num_units = len(storage_unit_keys)
        gi_lists: dict[str, list[int]] = defaultdict(list)
        pos_lists: dict[str, list[int]] = defaultdict(list)
        for pos, global_idx in enumerate(global_indexes):
            key = storage_unit_keys[global_idx % num_units]
            gi_lists[key].append(global_idx)
            pos_lists[key].append(pos)
        return {key: RoutingGroup(gi_lists[key], pos_lists[key]) for key in gi_lists}

    def _describe_storage_unit(self, storage_unit_id: str) -> str:
        """Return ``ip:port`` for a storage unit, for use in diagnostics.

        The unit id is a random uuid4 fragment that carries no location, so a bare id in an
        error message cannot be traced back to a node. Never raises: it is only ever called
        while reporting another failure, and must not mask it.
        """
        info = self.storage_unit_infos.get(storage_unit_id)
        if info is None:
            return "endpoint unknown (unit not registered with this manager)"
        return f"{info.ip}:{info.ports.get('put_get_socket')}"

    @with_storage_unit_probe_socket
    async def _probe_storage_unit(self, target_storage_unit: str, socket: zmq.Socket = None) -> dict[str, Any]:
        """Ask a storage unit for its own counters over a brand-new socket.

        Served by the same worker thread as put and get, so an answer proves the unit is serving.
        """
        request_msg = ZMQMessage.create(
            request_type=ZMQRequestType.GET_METRICS,  # type: ignore[arg-type]
            sender_id=f"{self.storage_manager_id}_probe",
            receiver_id=target_storage_unit,
            body={},
        )
        await socket.send_multipart(request_msg.serialize())
        messages = await socket.recv_multipart(copy=False)
        response_msg = ZMQMessage.deserialize(messages)
        if response_msg.request_type != ZMQRequestType.METRICS_RESPONSE:
            raise RuntimeError(f"unexpected probe response type {response_msg.request_type}")
        return response_msg.body

    async def _diagnose_storage_unit(self, target_storage_unit: str) -> str:
        """Report whether the unit is reachable and serving after a request to it timed out.

        Returns one log line and never raises: it runs while another failure is being reported.
        Says nothing about where that request went; no per-request state exists to show it.
        """
        info = self.storage_unit_infos.get(target_storage_unit)
        if info is None:
            return "verdict=unknown(unit_not_registered)"

        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(info.ip, info.ports.get("put_get_socket")), timeout=5
            )
            writer.close()
            tcp = "tcp=up"
        except Exception as e:
            tcp = f"tcp=down({type(e).__name__})"

        try:
            body = await self._probe_storage_unit(target_storage_unit=target_storage_unit)
        except zmq.error.Again:
            return f"{tcp} verdict=unit_not_serving (no probe answer in {TQ_SIMPLE_STORAGE_PROBE_TIMEOUT}s)"
        except Exception as e:
            return f"{tcp} verdict=unknown (probe failed: {type(e).__name__}: {e})"

        return f"{tcp} {_describe_unit_state(body)}"

    async def _request_with_retry(
        self,
        operation: str,
        target_storage_unit: str,
        request_context: str,
        make_request: Callable[[], Any],
        max_attempts: int | None = None,
    ):
        """Run one storage-unit request, retrying a missing answer on a fresh connection.

        Args:
            operation: Operation name for logs, e.g. ``get`` or ``put``.
            target_storage_unit: Unit this request is routed to.
            request_context: Request shape, so both ends of a failure can be correlated.
            make_request: Zero-arg callable returning a coroutine for one attempt. Must build a
                new socket per call, which ``with_storage_unit_socket`` does.
            max_attempts: Attempts allowed. Defaults to ``TQ_SIMPLE_STORAGE_MAX_ATTEMPTS``; pass 1
                for a request that must not be replayed.
        """
        # Floored at one: a nonpositive count would skip the request and report success, which for
        # put would publish metadata for data that was never sent.
        attempts_allowed = max(1, TQ_SIMPLE_STORAGE_MAX_ATTEMPTS if max_attempts is None else max_attempts)
        for attempt in range(1, attempts_allowed + 1):
            try:
                return await make_request()
            except StorageUnitTimeout as e:
                # The exception already names the unit, its endpoint and the timeout, so these
                # lines only add what it cannot know: which attempt this was, and the shape.
                if attempt < attempts_allowed:
                    logger.warning(
                        f"[{self.storage_manager_id}]: {operation} retry {attempt + 1}/{attempts_allowed} "
                        f"on a new connection. {request_context} {e}"
                    )
                    continue
                logger.error(
                    f"[{self.storage_manager_id}]: {operation} failed after {attempt} attempts. "
                    f"{request_context} {e} {await self._diagnose_storage_unit(target_storage_unit)}"
                )
                raise

    @staticmethod
    def _select_by_positions(field_data, positions: list[int]):
        """Slice a single field's data by non-contiguous batch positions.

        This method optimizes selection to minimize memory overhead and network fragmentation:
        - Nested tensors: Unbinds into a list of views (end-to-end zero-copy).
        - Regular tensors (step == 1): Returns a contiguous slice (end-to-end zero-copy).
        - Regular tensors (step > 1): Returns a strided view (shares storage). Note that
          downstream serialization will force a `.contiguous()` copy, but slicing is still
          faster than `index_select` and the peak memory period is reduced.
        - Regular tensors (irregular): Falls back to `index_select` to assemble a single
          contiguous tensor, preventing excessive ZMQ multipart frames.
        - NonTensorStack: tolist → select → re-wrap.
        - List: Direct index selection via `itemgetter`.
        - Numpy arrays / Others: Advanced indexing (memory copy).
        """

        n = len(positions)
        if n == 0:
            raise ValueError("No positions specified for selection.")

        # --- Handle PyTorch Tensors ---
        if isinstance(field_data, torch.Tensor):
            if field_data.is_nested:
                # Nested tensors cannot be directly sliced into a single tensor view.
                # Unbinding and selecting returns a list of individual views (zero-copy),
                # which is acceptable for nested structures.
                unbound = field_data.unbind()
                getter = itemgetter(*positions) if len(positions) > 1 else lambda seq: (seq[positions[0]],)
                selected = getter(unbound)
                return list(selected)
            else:
                # --- Smart Slicing for Regular Tensors ---
                # Goal: Return a single underlying memory view (zero-copy) to avoid both
                # memory allocation overhead and downstream ZMQ frame fragmentation.

                # Case 1: Single element selection (returns a single-row view)
                if n == 1:
                    # Single element is natively contiguous
                    return field_data[positions[0] : positions[0] + 1]

                # Case 2: Check if positions form a constant-stride sequence
                step = positions[1] - positions[0]
                is_constant_stride = True
                for i in range(2, n):
                    if positions[i] - positions[i - 1] != step:
                        is_constant_stride = False
                        break

                # If perfectly regular (e.g., [0, 2, 4]), use Python slicing to get a view
                if is_constant_stride and step > 0:
                    # Note:
                    # A strided slice (step > 1) creates a non-contiguous view.
                    # While it shares storage here, the downstream MsgpackEncoder will force
                    # a .contiguous() copy before extracting the buffer. However, this pure
                    # Python slicing is still more efficient than falling back to index_select,
                    # and it reduces memory peak period.
                    return field_data[positions[0] : positions[-1] + 1 : step]

                # Case 3: Fallback for irregular indices (Typically this will not happen!)
                # We intentionally accept a memory copy here to assemble a single contiguous
                # tensor. Returning a list of individual views for irregular indices would
                # generate excessive multipart ZMQ frames, severely degrading network performance.
                else:
                    idx_tensor = torch.tensor(positions, device=field_data.device)
                    return torch.index_select(field_data, dim=0, index=idx_tensor)

        # --- Handle Non-Tensor Types ---
        elif isinstance(field_data, NonTensorStack):
            items = field_data.tolist()
            getter = itemgetter(*positions) if len(positions) > 1 else lambda seq: (seq[positions[0]],)
            selected = getter(items)
            return NonTensorStack(*selected)
        elif isinstance(field_data, list):
            getter = itemgetter(*positions) if len(positions) > 1 else lambda seq: (seq[positions[0]],)
            selected = getter(field_data)
            return list(selected)
        else:
            return field_data[positions]

    async def put_data(
        self, data: TensorDict, metadata: BatchMeta, data_parser: Callable[[Any], Any] | None = None
    ) -> None:
        """
        Send data to remote StorageUnit based on metadata.

        Routes each sample to its target SU using global_idx % num_su (hash routing).
        Complexity: O(F) for schema extraction + O(S) for data distribution.

        Args:
            data: TensorDict containing the data to store.
            metadata: BatchMeta containing storage location information.
            data_parser: Optional callable to parse reference data (e.g., URLs) into real
                         content. The input is a plain dict (not TensorDict) mapping
                         field_name -> batched values. For a regular tensor column the
                         value is a batched tensor; for nested tensors (jagged or strided)
                         and NonTensorStack columns the values are extracted into a list.
                         It must modify values in-place based on the original keys; do not
                         add or remove keys. The number of elements per column must also
                         remain unchanged. Do not change the inner order of values within
                         each column. Executed distributedly on each SimpleStorageUnit.
        """

        logger.debug(f"[{self.storage_manager_id}]: receive put_data request, putting {metadata.size} samples.")

        batch_size = metadata.size

        if batch_size == 0:
            return

        field_schema = extract_field_schema(data)

        routing = self._group_by_hash(metadata.global_indexes)
        # Parser-backed puts are not replayed: the public API does not constrain parser side effects.
        max_attempts = 1 if data_parser is not None else None
        tasks = []
        for su_id, group in routing.items():
            storage_data = {f: self._select_by_positions(data[f], group.batch_positions) for f in data.keys()}
            tasks.append(
                self._request_with_retry(
                    "put",
                    su_id,
                    f"samples={len(group.global_indexes)} fields={list(storage_data.keys())}",
                    partial(
                        self._put_to_single_storage_unit,
                        group.global_indexes,
                        storage_data,
                        target_storage_unit=su_id,
                        data_parser=data_parser,
                    ),
                    max_attempts=max_attempts,
                )
            )

        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            # The offending unit is named in the error itself; the full routing list can run to
            # hundreds of ids, which buries it.
            logger.error(
                f"[{self.storage_manager_id}]: put_data failed. "
                f"partition_id={metadata.partition_ids[0]}, "
                f"num_samples={metadata.size}, "
                f"num_storage_units={len(routing)}, "
                f"error={type(e).__name__}: {e}"
            )
            raise

        partition_id = metadata.partition_ids[0]
        await self.notify_data_update(
            partition_id,
            metadata.global_indexes,
            field_schema,
        )

    async def update_data(
        self,
        metadata: BatchMeta,
        field_names: list[str],
        values: TensorDict | None = None,
        parser: Callable[[Any, Any], Any] | None = None,
        empty: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """Send an update to each hashed storage unit and notify the controller.

        The unit reads the old value, applies empty or parser(old, new), and reports
        the shape it stored per sample; this assembles one batch-level field_schema
        so production status stays ready.
        """
        logger.debug(f"[{self.storage_manager_id}]: receive update_data request, updating {metadata.size} samples.")

        if metadata.size == 0:
            return {}
        if empty:
            if values is not None or parser is not None:
                raise ValueError("empty update must not include values or a parser")
        else:
            if values is None or parser is None:
                raise ValueError("update requires values and a parser unless empty is set")
            if values.batch_size[0] != metadata.size:
                raise ValueError(
                    f"Batch size of values ({values.batch_size[0]}) does not match metadata size ({metadata.size})"
                )

        routing = self._group_by_hash(metadata.global_indexes)
        # Parser-backed updates are not replayed: parser(old, new) reads the stored value, so a
        # second attempt after a lost answer would fold the new value in twice. Empty is a
        # plain overwrite and stays retryable.
        max_attempts = 1 if parser is not None else None
        tasks = []
        for su_id, group in routing.items():
            storage_data = None
            if values is not None:
                storage_data = {f: self._select_by_positions(values[f], group.batch_positions) for f in field_names}
            tasks.append(
                self._request_with_retry(
                    "update",
                    su_id,
                    f"samples={len(group.global_indexes)} fields={field_names}",
                    partial(
                        self._update_to_single_storage_unit,
                        group.global_indexes,
                        field_names,
                        storage_data,
                        parser,
                        empty,
                        target_storage_unit=su_id,
                    ),
                    max_attempts=max_attempts,
                )
            )

        try:
            described = await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(
                f"[{self.storage_manager_id}]: update_data failed. "
                f"partition_id={metadata.partition_ids[0]}, "
                f"num_samples={metadata.size}, "
                f"num_storage_units={len(routing)}, "
                f"error={type(e).__name__}: {e}"
            )
            raise

        field_schema = _build_update_field_schema(
            metadata.global_indexes, list(zip([g.global_indexes for g in routing.values()], described, strict=True))
        )

        await self.notify_data_update(
            metadata.partition_ids[0],
            metadata.global_indexes,
            field_schema,
        )
        return field_schema

    @with_storage_unit_socket
    async def _put_to_single_storage_unit(
        self,
        global_indexes: list[int],
        storage_data: dict[str, Any],
        target_storage_unit: str,
        data_parser: Callable[[Any], Any] | None = None,
        socket: zmq.Socket = None,
    ):
        """
        Send data to a specific storage unit.
        """

        request_msg = ZMQMessage.create(
            request_type=ZMQRequestType.PUT_DATA,  # type: ignore[arg-type]
            sender_id=self.storage_manager_id,
            receiver_id=target_storage_unit,
            body={"global_indexes": global_indexes, "data": storage_data, "data_parser": data_parser},
        )

        serialized_bytes = 0
        started = time.perf_counter()
        try:
            frames = request_msg.serialize()
            serialized_bytes = sum(frame_nbytes(frame) or 0 for frame in frames)
            await socket.send_multipart(frames, copy=False)
            messages = await socket.recv_multipart(copy=False)
            response_msg = ZMQMessage.deserialize(messages)

            if response_msg.request_type != ZMQRequestType.PUT_DATA_RESPONSE:
                raise RuntimeError(
                    f"Failed to put data to storage unit {target_storage_unit}: "
                    f"{response_msg.body.get('message', 'Unknown error')}"
                )
            # This end serializes the put payload and waits out the round trip, so it holds both
            # the true wire size and the end-to-end latency.
            log_heavy_operation(
                self.storage_manager_id,
                "put",
                time.perf_counter() - started,
                serialized_bytes,
                f"to {target_storage_unit} at {self._describe_storage_unit(target_storage_unit)} "
                f"samples={len(global_indexes)} fields={list(storage_data.keys())}",
            )
        except zmq.error.Again as e:
            raise StorageUnitTimeout(
                f"no answer in {TQ_SIMPLE_STORAGE_SEND_RECV_TIMEOUT}s during put to storage unit "
                f"{target_storage_unit} at {self._describe_storage_unit(target_storage_unit)}; "
                f"serialized_mb={serialized_bytes / 2**20:.1f}"
            ) from e
        except Exception as e:
            logger.error(
                f"[{self.storage_manager_id}]: Unexpected error during put to storage unit "
                f"{target_storage_unit}: {type(e).__name__}: {e}"
            )
            raise RuntimeError(f"Error in put to storage unit {target_storage_unit}: {type(e).__name__}: {e}") from e

    @with_storage_unit_socket
    async def _update_to_single_storage_unit(
        self,
        global_indexes: list[int],
        field_names: list[str],
        storage_data: dict[str, Any] | None,
        parser: Callable[[Any, Any], Any] | None,
        empty: bool,
        target_storage_unit: str,
        socket: zmq.Socket = None,
    ) -> dict[str, dict[str, Any]]:
        """Send an update to one storage unit and return what it stored, per field."""
        body: dict[str, Any] = {
            "global_indexes": global_indexes,
            "fields": field_names,
            "empty": empty,
        }
        if not empty:
            body["data"] = storage_data
            body["parser"] = parser

        request_msg = ZMQMessage.create(
            request_type=ZMQRequestType.UPDATE_DATA,  # type: ignore[arg-type]
            sender_id=self.storage_manager_id,
            receiver_id=target_storage_unit,
            body=body,
        )

        serialized_bytes = 0
        started = time.perf_counter()
        try:
            frames = request_msg.serialize()
            serialized_bytes = sum(frame_nbytes(frame) or 0 for frame in frames)
            await socket.send_multipart(frames, copy=False)
            messages = await socket.recv_multipart(copy=False)
            response_msg = ZMQMessage.deserialize(messages)

            if response_msg.request_type != ZMQRequestType.UPDATE_DATA_RESPONSE:
                raise RuntimeError(
                    f"Failed to update data on storage unit {target_storage_unit}: "
                    f"{response_msg.body.get('message', 'Unknown error')}"
                )
            log_heavy_operation(
                self.storage_manager_id,
                "update",
                time.perf_counter() - started,
                serialized_bytes,
                f"to {target_storage_unit} at {self._describe_storage_unit(target_storage_unit)} "
                f"samples={len(global_indexes)} fields={field_names}",
            )
            return response_msg.body.get("stored_shapes", {})
        except zmq.error.Again as e:
            raise StorageUnitTimeout(
                f"no answer in {TQ_SIMPLE_STORAGE_SEND_RECV_TIMEOUT}s during update to storage unit "
                f"{target_storage_unit} at {self._describe_storage_unit(target_storage_unit)}; "
                f"serialized_mb={serialized_bytes / 2**20:.1f}"
            ) from e
        except Exception as e:
            logger.error(
                f"[{self.storage_manager_id}]: Unexpected error during update to storage unit "
                f"{target_storage_unit}: {type(e).__name__}: {e}"
            )
            raise RuntimeError(f"Error in update to storage unit {target_storage_unit}: {type(e).__name__}: {e}") from e

    @staticmethod
    def _pack_field_values(values: list) -> torch.Tensor | NonTensorStack:
        """
        Pack a list of per-sample values into a batched container.

        For pure tensor lists (no None), this tries nested tensor
        (jagged layout first, then strided fallback), then falls back to
        ``NonTensorStack``. Scalar tensors are stacked densely.
        Mixed types, non-tensor values, or lists containing None placeholders
        are grouped into a ``NonTensorStack``.

        Args:
            values: List of per-sample values to pack. May contain None for
                unfilled batch positions.

        Returns:
            A ``torch.Tensor`` (nested or dense) when all values are tensors,
            otherwise a ``NonTensorStack``.

        Raises:
            ValueError: If *values* is empty.
        """
        if not values:
            raise ValueError("_pack_field_values received empty values list; caller should filter empty batches")
        non_none = [v for v in values if v is not None]
        if non_none and all(isinstance(v, torch.Tensor) for v in non_none):
            if len(non_none) == len(values):
                # Scalar tensors cannot be represented as jagged nested tensors;
                # stack them densely to avoid noisy fallback warnings.
                if all(v.dim() == 0 for v in non_none):
                    return torch.stack(non_none)
                # Pure tensor list — try nested tensor
                try:
                    return torch.nested.as_nested_tensor(values, layout=torch.jagged)
                except (RuntimeError, TypeError) as e:
                    logger.warning(
                        f"Failed to pack nested tensor with jagged layout. "
                        f"Falling back to strided layout. Detailed error: {e}"
                    )
                    try:
                        return torch.nested.as_nested_tensor(values, layout=torch.strided)
                    except (RuntimeError, TypeError) as e2:
                        logger.warning(
                            f"Failed to pack nested tensor with strided layout. "
                            f"Falling back to NonTensorStack. Detailed error: {e2}"
                        )
                        return NonTensorStack(*values)
            # Mixed tensor + None — cannot create nested tensor, fall through to NonTensorStack
        return NonTensorStack(*values)

    async def get_data(self, metadata: BatchMeta) -> TensorDict:
        """
        Retrieve data from remote StorageUnit based on metadata.

        Routes to each SU using global_idx % num_su (hash routing).

        Args:
            metadata: BatchMeta that contains metadata for data retrieval.

        Returns:
            TensorDict containing the retrieved data.
        """

        logger.debug(f"[{self.storage_manager_id}]: receive get_data request, getting {metadata.size} samples.")

        if metadata.size == 0:
            return TensorDict({}, batch_size=0)

        routing = self._group_by_hash(metadata.global_indexes)

        tasks = [
            self._request_with_retry(
                "get",
                su_id,
                f"samples={len(group.global_indexes)} fields={list(metadata.field_names)}",
                partial(
                    self._get_from_single_storage_unit,
                    group.global_indexes,
                    metadata.field_names,
                    target_storage_unit=su_id,
                ),
            )
            for su_id, group in routing.items()
        ]
        try:
            results = await asyncio.gather(*tasks)
        except Exception as e:
            # The offending unit is named in the error itself; the full routing list can run to
            # hundreds of ids, which buries it.
            log = logger.debug if isinstance(e, StorageKeyNotFoundError) else logger.error
            log(
                f"[{self.storage_manager_id}]: get_data failed. "
                f"partition_id={metadata.partition_ids[0]}, "
                f"num_samples={metadata.size}, "
                f"num_storage_units={len(routing)}, "
                f"error={type(e).__name__}: {e}"
            )
            raise

        # Scatter results directly to batch positions — no intermediate per-sample dict
        n = len(metadata.global_indexes)
        ordered_data: dict[str, list] = {field: [None] * n for field in metadata.field_names}

        for (su_id, group), (fields, su_data) in zip(routing.items(), results, strict=True):
            for field in fields:
                for i, pos in enumerate(group.batch_positions):
                    ordered_data[field][pos] = su_data[field][i]

        tensor_data = {field: self._pack_field_values(v) for field, v in ordered_data.items()}

        return TensorDict(tensor_data, batch_size=len(metadata))

    @with_storage_unit_socket
    async def _get_from_single_storage_unit(
        self,
        global_indexes: list[int],
        fields: list[str],
        target_storage_unit: str,
        socket: zmq.Socket = None,
    ):
        """Get data from a single SU by global index keys."""
        request_msg = ZMQMessage.create(
            request_type=ZMQRequestType.GET_DATA,  # type: ignore[arg-type]
            sender_id=self.storage_manager_id,
            receiver_id=target_storage_unit,
            body={"global_indexes": global_indexes, "fields": fields},
        )
        try:
            await socket.send_multipart(request_msg.serialize())
            messages = await socket.recv_multipart(copy=False)
            response_msg = ZMQMessage.deserialize(messages)

            if response_msg.request_type == ZMQRequestType.GET_DATA_RESPONSE:
                storage_unit_data = response_msg.body["data"]
                return fields, storage_unit_data
            else:
                message = response_msg.body.get("message", "Unknown error")
                error_type = StorageKeyNotFoundError if KEY_NOT_FOUND_MARKER in message else RuntimeError
                raise error_type(f"Failed to get data from storage unit {target_storage_unit}: {message}")
        except zmq.error.Again as e:
            raise StorageUnitTimeout(
                f"no answer in {TQ_SIMPLE_STORAGE_SEND_RECV_TIMEOUT}s during get from storage unit "
                f"{target_storage_unit} at {self._describe_storage_unit(target_storage_unit)}"
            ) from e
        except StorageKeyNotFoundError:
            # Already logged at debug by the storage unit; propagate for the caller to classify.
            raise
        except Exception as e:
            logger.error(
                f"[{self.storage_manager_id}]: Unexpected error from storage unit "
                f"{target_storage_unit}: {type(e).__name__}: {e}"
            )
            raise RuntimeError(
                f"Error getting data from storage unit {target_storage_unit}: {type(e).__name__}: {e}"
            ) from e

    async def clear_data(self, metadata: BatchMeta) -> None:
        """Clear data in remote StorageUnit.

        Routes to each SU using global_idx % num_su (hash routing).

        Args:
            metadata: BatchMeta that contains metadata for data clearing.
        """

        logger.debug(f"[{self.storage_manager_id}]: receive clear_data request, clearing {metadata.size} samples.")

        if metadata.size == 0:
            return

        routing = self._group_by_hash(metadata.global_indexes)

        tasks = [
            self._clear_single_storage_unit(group.global_indexes, target_storage_unit=su_id)
            for su_id, group in routing.items()
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"[{self.storage_manager_id}]: Error in clear operation task {i}: {result}")

    @with_storage_unit_socket
    async def _clear_single_storage_unit(self, global_indexes, target_storage_unit=None, socket=None):
        try:
            request_msg = ZMQMessage.create(
                request_type=ZMQRequestType.CLEAR_DATA,
                sender_id=self.storage_manager_id,
                receiver_id=target_storage_unit,
                body={"global_indexes": global_indexes},
            )

            await socket.send_multipart(request_msg.serialize())
            messages = await socket.recv_multipart(copy=False)
            response_msg = ZMQMessage.deserialize(messages)

            if response_msg.request_type != ZMQRequestType.CLEAR_DATA_RESPONSE:
                raise RuntimeError(
                    f"Failed to clear storage {target_storage_unit}: "
                    f"{response_msg.body.get('message', 'Unknown error')}"
                )

        except Exception as e:
            logger.error(f"[{self.storage_manager_id}]: Error clearing storage unit {target_storage_unit}: {str(e)}")
            raise

    def get_zmq_server_info(self) -> dict[str, ZMQServerInfo]:
        """Get ZMQ server information for all storage units.

        Returns:
            Dictionary mapping storage unit IDs to their ZMQServerInfo.
        """
        return self.storage_unit_infos

    @with_storage_unit_socket
    async def _save_single_storage_unit(
        self,
        path: str,
        target_storage_unit: str,
        socket: zmq.Socket = None,
    ):
        try:
            request_msg = ZMQMessage.create(
                request_type=ZMQRequestType.SAVE_STORAGE_CHECKPOINT,  # type: ignore[arg-type]
                sender_id=self.storage_manager_id,
                receiver_id=target_storage_unit,
                body={"path": path},
            )
            await socket.send_multipart(request_msg.serialize(), copy=False)
            messages = await socket.recv_multipart(copy=False)
            response_msg = ZMQMessage.deserialize(messages)
            if (
                response_msg.request_type != ZMQRequestType.SAVE_STORAGE_CHECKPOINT_RESPONSE
                or not response_msg.body.get("success")
            ):
                raise RuntimeError(
                    f"Storage unit {target_storage_unit} failed to save checkpoint to {path}: "
                    f"{response_msg.body.get('message', 'unknown error')}"
                )
        except Exception as e:
            raise RuntimeError(
                f"[{self.storage_manager_id}]: Error dumping storage unit {target_storage_unit}: {str(e)}"
            ) from e

    @with_storage_unit_socket
    async def _load_single_storage_unit(
        self,
        path: str,
        target_storage_unit: str,
        socket: zmq.Socket = None,
    ):
        try:
            request_msg = ZMQMessage.create(
                request_type=ZMQRequestType.LOAD_STORAGE_CHECKPOINT,  # type: ignore[arg-type]
                sender_id=self.storage_manager_id,
                receiver_id=target_storage_unit,
                body={"path": path},
            )
            await socket.send_multipart(request_msg.serialize(), copy=False)
            messages = await socket.recv_multipart(copy=False)
            response_msg = ZMQMessage.deserialize(messages)
            if (
                response_msg.request_type != ZMQRequestType.LOAD_STORAGE_CHECKPOINT_RESPONSE
                or not response_msg.body.get("success")
            ):
                raise RuntimeError(
                    f"Storage unit {target_storage_unit} failed to load checkpoint from {path}: "
                    f"{response_msg.body.get('message', 'unknown error')}"
                )
        except Exception as e:
            raise RuntimeError(
                f"[{self.storage_manager_id}]: Error restoring for storage unit {target_storage_unit}: {str(e)}"
            ) from e

    @with_storage_unit_dump_socket
    async def _dump_single_shard(
        self,
        path: str,
        target_storage_unit: str,
        fields_by_index: dict[int, list[str]],
        socket: zmq.Socket = None,
    ) -> dict[str, Any]:
        """Ask one storage unit to write the rows it owns into a shard file."""
        try:
            request_msg = ZMQMessage.create(
                request_type=ZMQRequestType.DUMP_ROWS,  # type: ignore[arg-type]
                sender_id=self.storage_manager_id,
                receiver_id=target_storage_unit,
                body={"path": path, "fields_by_index": fields_by_index},
            )
            await socket.send_multipart(request_msg.serialize(), copy=False)
            messages = await socket.recv_multipart(copy=False)
            response_msg = ZMQMessage.deserialize(messages)
            if response_msg.request_type != ZMQRequestType.DUMP_ROWS_RESPONSE or not response_msg.body.get("success"):
                raise RuntimeError(
                    f"Storage unit {target_storage_unit} failed to dump rows to {path}: "
                    f"{response_msg.body.get('message', 'unknown error')}"
                )
            missing_rows = response_msg.body["missing_rows"]
            if missing_rows:
                # The controller reported these rows as produced, so a unit that has no
                # data for them means the two disagree. Never write a half table.
                raise RuntimeError(
                    f"Storage unit {target_storage_unit} holds no data for requested rows: {missing_rows[:20]}"
                )
            return response_msg.body
        except Exception as e:
            raise RuntimeError(
                f"[{self.storage_manager_id}]: Error dumping shard from storage unit {target_storage_unit}: {str(e)}"
            ) from e

    async def dump_rows_by_index(
        self,
        shard_dir: str,
        fields_by_index: dict[int, list[str]],
    ) -> dict[str, Any]:
        """Dump the given rows into one shard per storage unit, in parallel.

        Each unit pickles its own rows in its own process, so the payload never passes
        through the caller. A unit that owns none of the rows is skipped rather than
        writing an empty shard.

        Args:
            shard_dir: Directory to write shard files into.
            fields_by_index: Global index of each row to dump -> its produced fields.

        Returns:
            ``{"shards", "row_schema"}``: one ``{"file", "storage_unit_id", "rows",
            "row_offsets"}`` entry per written shard, and each dumped row's
            ``{field: (dtype, shape) or None}`` as the owner unit holds it.

        Raises:
            RuntimeError: A unit holds no data for a row it was asked to dump.
        """
        shard_dir_path = Path(shard_dir)
        shard_dir_path.mkdir(parents=True, exist_ok=True)

        routing = self._group_by_hash(sorted(fields_by_index))
        targets = [(su_id, group.global_indexes) for su_id, group in routing.items()]
        paths = [str(shard_dir_path / f"shard_{pos}_{su_id}.pkl") for pos, (su_id, _) in enumerate(targets)]

        results = await asyncio.gather(
            *(
                self._dump_single_shard(
                    path,
                    target_storage_unit=su_id,
                    fields_by_index={index: fields_by_index[index] for index in indexes},
                )
                for path, (su_id, indexes) in zip(paths, targets, strict=True)
            ),
            return_exceptions=True,
        )
        shards = []
        row_schema = {}
        total_rows = 0
        for path, (su_id, _), result in zip(paths, targets, results, strict=True):
            if isinstance(result, BaseException):
                raise result
            offsets = result["row_offsets"]
            total_rows += len(offsets)
            row_schema.update(result["row_schema"])
            shards.append(
                {
                    "file": Path(path).name,
                    "storage_unit_id": su_id,
                    "rows": len(offsets),
                    "row_offsets": offsets,
                }
            )

        logger.info(
            f"[{self.storage_manager_id}]: dumped {total_rows} rows across {len(targets)} shards to {shard_dir_path}"
        )
        return {"shards": shards, "row_schema": row_schema}

    async def load_rows_by_index(self, partition_id: str, shards: list[dict[str, Any]]) -> int:
        """Have current owner units read assigned byte ranges concurrently, then publish metadata.

        Metadata is published only after every unit succeeded, so writes to new keys stay
        invisible after a failure, while fields existing keys already produced are
        overwritten in place. Retrying rewrites the same target indexes.

        Returns:
            Payload bytes read by the units.
        """
        assignments = defaultdict(list)
        for shard in shards:
            rows = shard["records"]
            for unit_id, group in self._group_by_hash([row["target_index"] for row in rows]).items():
                assignments[unit_id].append(
                    {
                        **shard,
                        "records": [rows[pos] for pos in group.batch_positions],
                    }
                )
        results = await asyncio.gather(
            *(self._load_selected_rows(shards, target_storage_unit=unit_id) for unit_id, shards in assignments.items()),
            return_exceptions=True,
        )
        loaded = []
        for result in results:
            if isinstance(result, BaseException):
                raise result
            loaded.append(result)
        # One update per (shard, unit, batch, field set): sequential round trips would
        # dominate a wide cross-topology load. The notify pool bounds the concurrency.
        await asyncio.gather(
            *(
                self.notify_data_update(partition_id, update["global_indexes"], update["field_schema"])
                for result in loaded
                for update in result["updates"]
            )
        )
        bytes_read = sum(result["bytes_read"] for result in loaded)
        logger.info(
            "[%s]: loaded %s bytes across %s units",
            self.storage_manager_id,
            bytes_read,
            len(assignments),
        )
        return bytes_read

    @with_storage_unit_dump_socket
    async def _load_selected_rows(
        self,
        shards: list[dict[str, Any]],
        target_storage_unit: str,
        socket: zmq.Socket = None,
    ) -> dict[str, Any]:
        request = ZMQMessage.create(
            request_type=ZMQRequestType.LOAD_ROWS,
            sender_id=self.storage_manager_id,
            receiver_id=target_storage_unit,
            body={"shards": shards},
        )
        await socket.send_multipart(request.serialize(), copy=False)
        response = ZMQMessage.deserialize(await socket.recv_multipart(copy=False))
        if response.request_type != ZMQRequestType.LOAD_ROWS_RESPONSE or not response.body.get("success"):
            raise RuntimeError(
                f"Storage unit {target_storage_unit} failed to load rows: {response.body.get('message')}"
            )
        return response.body

    async def save_checkpoint(self, checkpoint_dir: str) -> None:
        """Dump all storage units to the storage_units/ subdirectory of checkpoint_dir.

        Writes a storage_units/su_info.json manifest for load_checkpoint to use.
        Validates current SU count matches the manifest on load.
        """
        import json

        su_dir = Path(checkpoint_dir) / _SU_SUBDIR
        su_dir.mkdir(parents=True, exist_ok=True)

        su_ids = list(self.storage_unit_infos.keys())
        paths = [str(su_dir / f"su_{pos}_{su_id}.pkl") for pos, su_id in enumerate(su_ids)]
        tasks = [
            self._save_single_storage_unit(path, target_storage_unit=su_id)
            for su_id, path in zip(su_ids, paths, strict=False)
        ]
        await asyncio.gather(*tasks)

        su_info_list = [{"position": pos, "storage_unit_id": su_id} for pos, su_id in enumerate(su_ids)]
        with open(su_dir / _SU_INFO_FILE, "w") as f:
            json.dump(su_info_list, f)

        logger.info(f"[{self.storage_manager_id}]: saved {len(su_ids)} storage units to {su_dir}")

    async def load_checkpoint(self, checkpoint_dir: str) -> None:
        """Restore all storage units from the storage_units/ subdirectory of checkpoint_dir."""
        import json

        su_dir = Path(checkpoint_dir) / _SU_SUBDIR
        su_info_path = su_dir / _SU_INFO_FILE
        if not su_info_path.exists():
            raise FileNotFoundError(f"Storage unit manifest not found: {su_info_path}")

        with open(su_info_path) as f:
            su_info_list = json.load(f)

        su_ids = list(self.storage_unit_infos.keys())
        if len(su_ids) != len(su_info_list):
            raise ValueError(
                f"Storage unit count mismatch: checkpoint has {len(su_info_list)}, current system has {len(su_ids)}."
            )

        entries = sorted(su_info_list, key=lambda e: e["position"])
        tasks = [
            self._load_single_storage_unit(
                str(su_dir / f"su_{entry['position']}_{entry['storage_unit_id']}.pkl"),
                target_storage_unit=su_ids[entry["position"]],
            )
            for entry in entries
        ]
        await asyncio.gather(*tasks)

        logger.info(f"[{self.storage_manager_id}]: restored {len(su_ids)} storage units from {su_dir}")

    def close(self) -> None:
        """Close all ZMQ sockets and context to prevent resource leaks."""
        # Before super(), which may destroy the context these sockets live on. Absent when
        # the base constructor raised (a failed handshake does), and __del__ still calls this.
        pool = getattr(self, "storage_rpc_pool", None)
        if pool is not None:
            pool.close()
        probe_pool = getattr(self, "storage_probe_pool", None)
        if probe_pool is not None:
            probe_pool.close()
        dump_pool = getattr(self, "storage_dump_pool", None)
        if dump_pool is not None:
            dump_pool.close()
        super().close()
