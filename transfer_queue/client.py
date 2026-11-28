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
import threading
import weakref
from typing import Any, Callable

import torch
import zmq
import zmq.asyncio
from tensordict import TensorDict

from transfer_queue.controller import PUBLIC_PARTITION
from transfer_queue.metadata import BatchMeta
from transfer_queue.storage import StorageManagerFactory
from transfer_queue.storage.managers.simple_storage_manager import AsyncSimpleStorageManager
from transfer_queue.utils import serial_utils
from transfer_queue.utils.common import limit_pytorch_auto_parallel_threads
from transfer_queue.utils.logging_utils import get_logger
from transfer_queue.utils.zmq_utils import (
    TQ_SOCKET_POOL_SIZE,
    ZMQMessage,
    ZMQRequestType,
    ZMQServerInfo,
    ZMQSocketPool,
)

logger = get_logger(__name__)

TQ_NUM_THREADS = int(os.environ.get("TQ_NUM_THREADS", 8))
# Size of the client context's native I/O-thread pool, shared by all controller RPCs and,
# for SimpleStorage, storage-unit requests -- so this knob is client- not backend-scoped.
TQ_CLIENT_ZMQ_IO_THREADS = int(os.environ.get("TQ_CLIENT_ZMQ_IO_THREADS", 8))
# Per-context socket ceiling, shared by every in-flight socket on the client's context
# (including a borrowing storage manager's), so libzmq's own 1023 is reachable at scale.
# Raising it also needs enough file descriptors (``ulimit -n``).
TQ_CLIENT_ZMQ_MAX_SOCKETS = os.environ.get("TQ_CLIENT_ZMQ_MAX_SOCKETS") or None
DEFAULT_CLIENT_ZMQ_MAX_SOCKETS = 8192
# A DEALER drops a reply it can no longer route, so without a bound a lost response blocks
# the caller forever. Generous: a busy controller is normal, an unanswered request is not.
TQ_CONTROLLER_SEND_RECV_TIMEOUT = int(os.environ.get("TQ_CONTROLLER_SEND_RECV_TIMEOUT", 200))  # seconds


def _status_tensor(raw: bytes, shape: tuple[int, ...]) -> torch.Tensor:
    if not raw:
        return torch.zeros(shape, dtype=torch.int8)
    return torch.frombuffer(bytearray(raw), dtype=torch.int8).reshape(shape)


class AsyncTransferQueueClient:
    """Async access to controller metadata and storage payloads.

    Use one long-lived event loop to reuse pooled sockets. A new ``asyncio.run()``
    per call creates a new pool owner and connection. Use ``TransferQueueClient``
    when the caller has no event loop.

    The optional ``socket`` arguments remain for call compatibility. Controller
    RPCs always lease a socket from this client's pool.
    """

    def __init__(
        self,
        client_id: str,
        controller_info: ZMQServerInfo,
        zmq_io_threads: int | None = None,
        zmq_max_sockets: int | None = None,
    ):
        """Create a client with its own ZMQ context and controller socket pool.

        Args:
            client_id: Unique client identifier.
            controller_info: Controller endpoint.
            zmq_io_threads: Native I/O threads; defaults to ``TQ_CLIENT_ZMQ_IO_THREADS`` (8).
            zmq_max_sockets: Per-client socket limit. Defaults to
                ``TQ_CLIENT_ZMQ_MAX_SOCKETS``, or 8192 when unset. The default is
                clamped to libzmq's limit; an explicit value above that limit raises.
                Storage managers borrowing this context share the budget, and a
                PUT/GET may open a socket per storage unit. Large limits also need
                enough file descriptors (``ulimit -n``).
        """
        if controller_info is None:
            raise ValueError("controller_info cannot be None")
        if not isinstance(controller_info, ZMQServerInfo):
            raise TypeError(f"controller_info must be ZMQServerInfo, got {type(controller_info)}")
        self.client_id = client_id
        self._controller: ZMQServerInfo = controller_info
        # Check everything that does not need the context first: the finalizer is not armed
        # until __init__ returns, so raising after allocation leaks it and its I/O threads.
        io_threads = TQ_CLIENT_ZMQ_IO_THREADS if zmq_io_threads is None else zmq_io_threads
        if io_threads < 1:
            raise ValueError(f"Client ZMQ I/O thread pool size must be at least 1, got {io_threads}")
        if TQ_SOCKET_POOL_SIZE < 1:
            # Name the variable: the pool's own error cannot say which knob supplied the value.
            raise ValueError(
                f"TQ_SOCKET_POOL_SIZE must be at least 1, got {TQ_SOCKET_POOL_SIZE}. "
                f"The pool always reuses at least one socket per endpoint; it cannot be disabled."
            )

        max_sockets = zmq_max_sockets
        explicitly_requested = max_sockets is not None
        if max_sockets is None and TQ_CLIENT_ZMQ_MAX_SOCKETS is not None:
            try:
                max_sockets = int(TQ_CLIENT_ZMQ_MAX_SOCKETS)
            except ValueError as e:
                # Name the variable: a bare int() error gives no clue which knob is wrong.
                raise ValueError(
                    f"TQ_CLIENT_ZMQ_MAX_SOCKETS must be an integer, got {TQ_CLIENT_ZMQ_MAX_SOCKETS!r}"
                ) from e
            explicitly_requested = True
        if max_sockets is not None and max_sockets < 1:
            # The upper bound needs ZMQ_SOCKET_LIMIT, hence a live context, but the lower one
            # does not -- so reject it before allocating anything.
            raise ValueError(f"Client ZMQ max sockets must be at least 1, got {max_sockets}")
        if max_sockets is None:
            max_sockets = DEFAULT_CLIENT_ZMQ_MAX_SOCKETS

        self.zmq_context = zmq.asyncio.Context(io_threads=io_threads)
        try:
            # ZMQ_SOCKET_LIMIT is a property of the built context, so this last check cannot
            # be hoisted above it; destroy the context rather than leak it on failure.
            socket_limit = self.zmq_context.get(zmq.SOCKET_LIMIT)
            if explicitly_requested:
                # A value the caller asked for must not be silently reinterpreted.
                if max_sockets > socket_limit:
                    raise ValueError(
                        f"Client ZMQ max sockets must be between 1 and this build's "
                        f"ZMQ_SOCKET_LIMIT ({socket_limit}), got {max_sockets}"
                    )
            elif max_sockets > socket_limit:
                # Nobody asked for the default, so clamp instead of failing to construct on a
                # build whose ZMQ_SOCKET_LIMIT is below it.
                logger.debug(
                    f"[{client_id}]: Clamping default ZMQ max sockets {max_sockets} to this "
                    f"build's ZMQ_SOCKET_LIMIT ({socket_limit})."
                )
                max_sockets = socket_limit
            self.zmq_context.set(zmq.MAX_SOCKETS, max_sockets)
        except BaseException:
            self.zmq_context.destroy(linger=0)
            raise
        # Reused across requests, so the socket budget above is consumed by the concurrency
        # high-water mark rather than by request count.
        self.controller_rpc_pool = ZMQSocketPool(
            self.zmq_context,
            client_id,
            "request_handle_socket",
            timeout=TQ_CONTROLLER_SEND_RECV_TIMEOUT,
        )

        # Backstop for a client that is never closed, so the context and its I/O threads do
        # not leak for the process lifetime. finalize() (not __del__) also runs at
        # interpreter exit; close() detaches it. See _release_zmq_context().
        self._finalizer = weakref.finalize(self, self._release_zmq_context, self.zmq_context, client_id)
        logger.info(f"[{self.client_id}]: Registered Controller server {controller_info.id} at {controller_info.ip}")

    @staticmethod
    def _release_zmq_context(context: "zmq.asyncio.Context", client_id: str) -> None:
        """Destroy an open context without retaining the client.

        A bound callback would prevent finalization. The synchronous client's loop
        thread can also retain the client until exit, so callers should use ``close``.
        """
        try:
            if not context.closed:
                context.destroy(linger=0)
        except Exception as e:
            logger.warning(f"[{client_id}]: Error destroying zmq_context in finalizer: {e}")

    def initialize_storage_manager(
        self,
        manager_type: str,
        config: dict[str, Any],
    ):
        """Create a registered storage manager with the supplied backend configuration.

        Each manager chooses whether to borrow the client's ZMQ context or create
        its own, and owns the socket pools it creates on that context.

        Args:
            manager_type: Name registered with ``StorageManagerFactory``.
            config: Backend settings. SimpleStorage requires ``zmq_info`` with the
                storage-unit endpoints.
        """
        self.storage_manager = StorageManagerFactory.create(
            manager_type,
            controller_info=self._controller,
            config=config,
            zmq_context=self.zmq_context,
        )

    async def _request_controller(
        self,
        request_type: ZMQRequestType,
        response_type: ZMQRequestType,
        body: dict[str, Any],
    ) -> ZMQMessage:
        """Lease a socket for one RPC; discard it if transport or response validation fails."""
        async with self.controller_rpc_pool.alease(self._controller) as socket:
            request_msg = ZMQMessage.create(
                request_type=request_type,  # type: ignore[arg-type]
                sender_id=self.client_id,
                receiver_id=self._controller.id,
                body=body,
            )
            await socket.send_multipart(request_msg.serialize())
            response_serialized = await socket.recv_multipart(copy=False)
            response_msg = ZMQMessage.deserialize(response_serialized)
            logger.debug(
                f"[{self.client_id}]: Received {response_msg.request_type} from controller {self._controller.id}"
            )
            if response_msg.request_type != response_type:
                message = response_msg.body.get("message", "Unknown error")
                raise RuntimeError(
                    f"[{self.client_id}]: Expected {response_type}, got {response_msg.request_type} "
                    f"from controller {self._controller.id}: {message}"
                )
            return response_msg

    def _require_storage_manager(self, operation: str = "performing storage") -> None:
        """Require an initialized manager before the named storage operation."""
        if getattr(self, "storage_manager", None) is None:
            raise RuntimeError(
                f"[{self.client_id}]: Storage manager not initialized. "
                f"Call initialize_storage_manager() before {operation} operations."
            )

    # ==================== Basic API ====================
    def validate_tags(self, tags: list[dict[str, Any]]) -> None:
        """Validate that custom per-sample tags are ``str``-keyed ``dict``s of native MessagePack values."""
        serial_utils.validate_metadata_dicts(tags, "tags")

    async def async_get_meta(
        self,
        data_fields: list[str],
        batch_size: int,
        partition_id: str,
        mode: str = "fetch",
        task_name: str | None = None,
        sampling_config: dict[str, Any] | None = None,
        socket: zmq.asyncio.Socket | None = None,
    ) -> BatchMeta:
        """Request sample metadata from the controller.

        Args:
            data_fields: Fields to include.
            batch_size: Number of samples to request.
            partition_id: Partition to query.
            mode: ``fetch`` selects ready samples through the sampler. ``force_fetch``
                bypasses readiness, consumption filtering, and the sampler, so rows
                may be unready or already consumed. ``insert`` is for internal use.
            task_name: Consumer task name.
            sampling_config: Custom sampler options, such as
                ``{"n_samples_per_prompt": 4}`` for ``GRPOGroupNSampler``.
            socket: Compatibility argument; RPCs use the client's socket pool.

        Returns:
            BatchMeta containing sample identities, field metadata, and readiness.

        Raises:
            RuntimeError: If controller communication or response validation fails.

        Use ``await client.async_get_meta(...)`` with the arguments shown in
        ``TransferQueueClient.get_meta``'s sampling and inspection examples.
        """
        response_msg = await self._request_controller(
            request_type=ZMQRequestType.GET_META,
            response_type=ZMQRequestType.GET_META_RESPONSE,
            body={
                "data_fields": data_fields,
                "batch_size": batch_size,
                "partition_id": partition_id,
                "mode": mode,
                "task_name": task_name,
                "sampling_config": sampling_config,
            },
        )
        return response_msg.body["metadata"]

    async def async_set_custom_meta(
        self,
        metadata: BatchMeta,
        socket: zmq.asyncio.Socket | None = None,
        *,
        _validate_tags: bool = True,
    ) -> None:
        """Publish per-sample tags to the controller for subsequent metadata queries.

        Args:
            metadata: Samples and tags set with ``BatchMeta.update_custom_meta``.
            socket: Compatibility argument; RPCs use the client's socket pool.

        Raises:
            ValueError: If tags are not string-keyed dictionaries of native MessagePack values.
            RuntimeError: If controller communication or response validation fails.

        After updating the tags, call ``await client.async_set_custom_meta(meta)``.
        See ``TransferQueueClient.set_custom_meta`` for the complete example.
        """
        if not self._controller:
            raise RuntimeError("No controller registered")

        global_indexes = metadata.global_indexes
        custom_meta = metadata.custom_meta

        if not global_indexes or not custom_meta:
            logger.debug(f"[{self.client_id}]: Empty BatchMeta or custom_meta provided. No action taken.")
            return

        if _validate_tags:
            self.validate_tags(custom_meta)
        # Build partition_custom_meta in format: {partition_id: {global_index: {meta1:xxx, meta2:xxx}}}
        partition_custom_meta: dict[str, dict[int, dict[str, Any]]] = {pid: {} for pid in set(metadata.partition_ids)}
        for partition_id, index, tag in zip(metadata.partition_ids, global_indexes, custom_meta, strict=True):
            partition_custom_meta[partition_id][index] = tag if tag else {}

        await self._request_controller(
            request_type=ZMQRequestType.SET_CUSTOM_META,
            response_type=ZMQRequestType.SET_CUSTOM_META_RESPONSE,
            body={
                "partition_custom_meta": partition_custom_meta,
            },
        )

    async def async_put(
        self,
        data: TensorDict,
        metadata: BatchMeta | None = None,
        partition_id: str | None = None,
        data_parser: Callable[[Any], Any] | None = None,
        *,
        _validate_tags: bool = True,
    ) -> BatchMeta:
        """Write payloads, publish their tags, and return the metadata used for the write.

        Args:
            data: Batched fields to write as a TensorDict.
            metadata: Destination samples, or None to create them in ``insert`` mode.
            partition_id: Required when ``metadata`` is None.
            data_parser: SimpleStorage-only callback run on each storage unit. It
                receives that unit's slice as a plain field-to-batch dict: dense
                tensors remain tensors; nested tensors and NonTensorStack columns
                become lists. Modify values in place while preserving field names,
                sample count, and sample order. This can turn references such as
                URLs into payloads.

        Without metadata, start with an empty partition and submit the initial data
        in one batch to avoid overwriting unconsumed rows. Repeat rows with interleave
        when producing multiple samples per input. Concurrent producers may observe
        different sample ordering.

        Warning:
            The written fields are ``data.keys()``. Selecting fields on BatchMeta
            filters neither ``data`` nor the controller's links. Only links still
            present in ``metadata`` protect fields from writes; omitting a link
            from metadata can create a private copy and drop the controller link.
            Select the payload instead, e.g. ``data.select("response")``, before
            writing it with the original linked metadata.

            Do not overwrite public rows while another worker may link to them:
            the pre-write link check does not reserve rows. See ``async_link``.

        Returns:
            The supplied BatchMeta, or the BatchMeta created for this write.

        Raises:
            ValueError: If metadata is empty, the required partition is missing,
                tags are not string-keyed native MessagePack dictionaries, or a written field
                is marked as linked in metadata.
            RuntimeError: If storage fails or a public row still has incoming links.

        Await writes, e.g. ``meta = await client.async_put(output, metadata=meta)``.
        See ``TransferQueueClient.put`` for initial insertion and write-back examples.
        """

        self._require_storage_manager()

        for field_name, field_data in data.items():
            if isinstance(field_data, torch.Tensor) and field_data.ndim == 1:
                logger.info(
                    f"[{self.client_id}]: Data field '{field_name}' is a tensor with only one dimension. "
                    f"You may receive 2D tensors in key-value based backend."
                )

        inserted = metadata is None
        if metadata is None:
            if partition_id is None:
                raise ValueError("partition_id must be provided if metadata is not given")

            metadata = await self.async_get_meta(
                data_fields=list(data.keys()),
                batch_size=data.batch_size[0],
                partition_id=partition_id,
                mode="insert",
            )

        if not metadata or metadata.size == 0:
            raise ValueError("metadata cannot be none or empty")
        if _validate_tags and not inserted:
            self.validate_tags(metadata.custom_meta)
        # A read-modify-write would otherwise turn every link into a private copy without notice.
        linked = sorted({name for links in metadata._links for name in links if name in data.keys()})
        if linked:
            raise ValueError(
                f"fields {linked} are linked by kv_link on some samples; leave them out of the put, "
                "or kv_clear those fields first to store private values"
            )
        public_rows = [
            idx
            for idx, pid in zip(metadata.global_indexes, metadata.partition_ids, strict=True)
            if pid == PUBLIC_PARTITION
        ]
        if public_rows and not inserted:
            await self._check_public_write(public_rows)

        with limit_pytorch_auto_parallel_threads(
            target_num_threads=TQ_NUM_THREADS, info=f"[{self.client_id}] async_put"
        ):
            await self.storage_manager.put_data(data, metadata, data_parser=data_parser)

        await self.async_set_custom_meta(metadata, _validate_tags=False)

        logger.debug(
            f"[{self.client_id}]: partition {partition_id} put {metadata.size} samples to storage units successfully."
        )

        # update metadata after put
        metadata = metadata.add_fields(data)

        return metadata

    async def async_update(
        self,
        metadata: BatchMeta,
        values: TensorDict,
        merge_fn: Callable[[Any, Any], Any],
        merge_plan: dict[str, dict[str, Any]],
        finalize: bool = True,
    ) -> BatchMeta:
        """Merge values into SimpleStorage fields, inserting values where absent.

        Args:
            metadata: Samples already registered with the controller.
            values: New values in metadata order.
            merge_fn: ``merge_fn(old, new)`` for each existing sample-field value.
            merge_plan: Field types and held rows from ``async_kv_prepare_update``.
            finalize: Whether to make the updated fields available to consumers.

        Returns:
            The same BatchMeta with updated field schema and production status.

        Raises:
            NotImplementedError: If the backend is not SimpleStorage.
        """
        self._require_storage_manager()
        if not metadata or metadata.size == 0:
            raise ValueError("metadata cannot be none or empty")

        field_schema = await self.storage_manager.update_data(metadata, values, merge_fn, merge_plan, finalize=finalize)
        if not finalize:
            metadata.production_status[:] = 0
        return metadata.apply_field_schema(field_schema, set_all_ready=finalize)

    async def async_get_data(
        self,
        metadata: BatchMeta,
        data_parser: Callable[[Any, Any], Any] | None = None,
        parser_args: list | None = None,
    ) -> TensorDict:
        """Read the requested fields in metadata sample order as a TensorDict.

        Args:
            metadata: Sample locations, indexes, and fields to read.
            data_parser: Optional SimpleStorage callback ``data_parser(data, args)``
                applied before each unit replies; see ``transfer_queue.kv_batch_get``.
            parser_args: Per-sample arguments aligned with ``metadata.global_indexes``.

        After fetching metadata, call ``batch = await client.async_get_data(meta)``.
        See ``TransferQueueClient.get_data`` for the complete read sequence.
        """

        self._require_storage_manager()

        if not metadata or metadata.size == 0 or len(metadata.field_names) == 0:
            logger.warning(f"[{self.client_id}]: Empty BatchMeta provided to get_data. Returning empty TensorDict.")
            return TensorDict({}, batch_size=0)
        if parser_args is not None:
            if data_parser is None:
                raise ValueError("parser_args requires data_parser")
            if len(parser_args) != metadata.size:
                raise ValueError(f"parser_args has {len(parser_args)} entries for {metadata.size} samples")

        with limit_pytorch_auto_parallel_threads(
            target_num_threads=TQ_NUM_THREADS, info=f"[{self.client_id}] async_get_data"
        ):
            # Passed only when set, so custom managers with get_data(self, metadata) keep working.
            if data_parser is None:
                results = await self.storage_manager.get_data(metadata)
            else:
                results = await self.storage_manager.get_data(
                    metadata, data_parser=data_parser, parser_args=parser_args
                )

        logger.debug(f"[{self.client_id}]: get_data with {metadata.size} samples successfully.")

        return results

    async def async_clear_partition(self, partition_id: str):
        """Remove a partition's payloads and controller metadata.

        Rows are marked unready before their payloads and metadata are removed.
        A missing partition is a no-op. Raise RuntimeError if clearing fails.
        """
        try:
            self._require_storage_manager()

            if not self._controller:
                raise RuntimeError("No controller registered")

            metadata = await self._get_partition_meta(partition_id)

            if not metadata:
                logger.warning(
                    f"[{self.client_id}]: Trying to clear a non-existent partition {partition_id}. "
                    f"No action will be taken."
                )
                return

            # Mark as pending deletion so consumers cannot read stale data
            await self._mark_clearing_in_controller(metadata)

            # Clear storage unit data
            await self.storage_manager.clear_data(metadata)

            # Release controller metadata and indexes
            await self._clear_partition_in_controller(partition_id)

            logger.debug(f"[{self.client_id}]: Clear operation for partition_id {partition_id} completed.")
        except Exception as e:
            raise RuntimeError(f"Error in clear operation: {str(e)}") from e

    async def async_clear_samples(self, metadata: BatchMeta):
        """Remove the samples described by metadata from storage and the controller.

        Rows are marked unready before their payloads and metadata are removed.
        An empty batch is a no-op. Raise RuntimeError if clearing fails.
        """
        try:
            self._require_storage_manager()

            if metadata.size == 0:
                logger.warning(f"[{self.client_id}]: Empty BatchMeta provided to clear_samples. No action taken.")
                return

            if not self._controller:
                raise RuntimeError("No controller registered")

            # Mark as pending deletion so consumers cannot read stale data
            await self._mark_clearing_in_controller(metadata)

            # Clear storage unit data
            await self.storage_manager.clear_data(metadata)

            # Release controller metadata and indexes
            await self._clear_meta_in_controller(metadata)

            logger.debug(f"[{self.client_id}]: Clear operation for batch {metadata} completed.")
        except Exception as e:
            raise RuntimeError(f"Error in clear_samples operation: {str(e)}") from e

    async def _mark_clearing_in_controller(self, metadata: BatchMeta, socket=None):
        """Clear sample readiness before deletion; raise RuntimeError on an invalid reply."""
        await self._request_controller(
            request_type=ZMQRequestType.MARK_CLEARING,
            response_type=ZMQRequestType.MARK_CLEARING_RESPONSE,
            body={"global_indexes": metadata.global_indexes, "partition_ids": metadata.partition_ids},
        )

    async def _clear_meta_in_controller(self, metadata: BatchMeta, socket=None):
        """Remove sample metadata from the controller; raise RuntimeError on failure."""

        await self._request_controller(
            request_type=ZMQRequestType.CLEAR_META,
            response_type=ZMQRequestType.CLEAR_META_RESPONSE,
            body={"global_indexes": metadata.global_indexes, "partition_ids": metadata.partition_ids},
        )

    async def _get_partition_meta(self, partition_id: str, socket=None) -> BatchMeta:
        """Return a partition's BatchMeta; raise RuntimeError on an invalid controller reply."""
        response_msg = await self._request_controller(
            request_type=ZMQRequestType.GET_PARTITION_META,
            response_type=ZMQRequestType.GET_PARTITION_META_RESPONSE,
            body={"partition_id": partition_id},
        )
        return response_msg.body["metadata"]

    async def _clear_partition_in_controller(self, partition_id, socket=None):
        """Remove a partition's controller metadata; raise RuntimeError on failure."""

        await self._request_controller(
            request_type=ZMQRequestType.CLEAR_PARTITION,
            response_type=ZMQRequestType.CLEAR_PARTITION_RESPONSE,
            body={"partition_id": partition_id},
        )

    # ==================== Status Query API ====================
    async def async_get_consumption_status(
        self,
        task_name: str,
        partition_id: str,
        socket: zmq.asyncio.Socket | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return sample indexes and consumption flags as tensors for a task in a partition.

        Flags are 1 for consumed and 0 otherwise. Return ``(None, None)`` when
        the controller supplies no status. Raise RuntimeError on controller errors.
        ``socket`` is retained for compatibility; RPCs use the client's pool.

        Await this query; see ``TransferQueueClient.get_consumption_status`` for
        filtering sample indexes by their flags.
        """

        try:
            response_msg = await self._request_controller(
                request_type=ZMQRequestType.GET_CONSUMPTION,
                response_type=ZMQRequestType.CONSUMPTION_RESPONSE,
                body={
                    "partition_id": partition_id,
                    "task_name": task_name,
                },
            )
            global_index = response_msg.body.get("global_index")
            consumption_status = response_msg.body.get("consumption_status")
            if global_index is None or consumption_status is None:
                return None, None
            return (
                torch.tensor(global_index, dtype=torch.long),
                _status_tensor(consumption_status, (len(consumption_status),)),
            )
        except Exception as e:
            raise RuntimeError(f"[{self.client_id}]: Error in get_consumption_status: {str(e)}") from e

    async def async_get_production_status(
        self,
        data_fields: list[str],
        partition_id: str,
        socket: zmq.asyncio.Socket | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return sample indexes and readiness flags as tensors for fields in a partition.

        Flags are 1 for ready and 0 otherwise. Return ``(None, None)`` when
        the controller supplies no status. Raise RuntimeError on controller errors.
        ``socket`` is retained for compatibility; RPCs use the client's pool.

        Await this query; see ``TransferQueueClient.get_production_status`` for
        finding rows with incomplete fields.
        """
        try:
            response_msg = await self._request_controller(
                request_type=ZMQRequestType.GET_PRODUCTION,
                response_type=ZMQRequestType.PRODUCTION_RESPONSE,
                body={
                    "partition_id": partition_id,
                    "data_fields": data_fields,
                },
            )
            global_index = response_msg.body.get("global_index")
            production_status = response_msg.body.get("production_status")
            if global_index is None or production_status is None:
                return None, None
            return (
                torch.tensor(global_index, dtype=torch.long),
                _status_tensor(production_status, (len(global_index), len(set(data_fields)))),
            )
        except Exception as e:
            raise RuntimeError(f"[{self.client_id}]: Error in get_data_production_status: {str(e)}") from e

    async def async_check_consumption_status(
        self,
        task_name: str,
        partition_id: str,
    ) -> bool:
        """Return whether all samples in the partition are consumed by this task.

        Missing or empty consumption status returns False. Controller errors raise RuntimeError.

        See ``TransferQueueClient.get_partition_list`` for a partition scan;
        await both async calls when adapting that example.
        """

        _, consumption_status = await self.async_get_consumption_status(
            task_name=task_name,
            partition_id=partition_id,
        )

        if consumption_status is None or consumption_status.numel() == 0:
            return False
        return torch.all(consumption_status == 1).item()

    async def async_check_production_status(
        self,
        data_fields: list[str],
        partition_id: str,
    ) -> bool:
        """Return whether all queried field-production flags are ready.

        Missing production status returns False. Controller errors raise RuntimeError.

        Use ``await client.async_check_production_status(...)`` with the arguments
        shown in ``TransferQueueClient.check_production_status``.
        """
        _, production_status = await self.async_get_production_status(
            data_fields=data_fields,
            partition_id=partition_id,
        )

        if production_status is None:
            return False
        return torch.all(production_status == 1).item()

    async def async_reset_consumption(
        self,
        partition_id: str,
        task_name: str | None = None,
        socket: zmq.asyncio.Socket | None = None,
    ) -> bool:
        """Reset consumption so a partition's samples can be consumed again.

        ``task_name=None`` resets all tasks; otherwise only the named task is reset.
        Return the controller's success flag, or raise RuntimeError on communication
        or response errors. ``socket`` is retained for compatibility.

        See ``TransferQueueClient.reset_consumption`` for consuming data again;
        await both the reset and the subsequent metadata fetch.
        """
        body = {"partition_id": partition_id}
        if task_name is not None:
            body["task_name"] = task_name
        try:
            response_msg = await self._request_controller(
                request_type=ZMQRequestType.RESET_CONSUMPTION,
                response_type=ZMQRequestType.RESET_CONSUMPTION_RESPONSE,
                body=body,
            )
            success = response_msg.body.get("success", False)
            if not success:
                logger.warning(f"[{self.client_id}]: Reset consumption failed: {response_msg.body.get('message')}")
            return success
        except Exception as e:
            raise RuntimeError(f"[{self.client_id}]: Error in reset_consumption: {str(e)}") from e

    async def async_get_partition_list(
        self,
        socket: zmq.asyncio.Socket | None = None,
    ) -> list[str]:
        """Return partition IDs managed by the controller.

        ``socket`` is retained for compatibility; RPCs use the client's pool.

        See ``TransferQueueClient.get_partition_list`` for a partition scan;
        await both async calls when adapting that example.
        """
        try:
            response_msg = await self._request_controller(
                request_type=ZMQRequestType.GET_LIST_PARTITIONS,
                response_type=ZMQRequestType.LIST_PARTITIONS_RESPONSE,
                body={},
            )
            return response_msg.body.get("partition_ids", [])
        except Exception as e:
            raise RuntimeError(f"[{self.client_id}]: Error in get_partition_list: {str(e)}") from e

    # ==================== KV Interface API ====================
    async def async_kv_retrieve_meta(
        self,
        keys: list[str] | str,
        partition_id: str,
        create: bool = False,
        socket: zmq.asyncio.Socket | None = None,
    ) -> BatchMeta:
        """Return metadata for keys in a partition, optionally registering missing keys.

        Args:
            keys: One key or a nonempty list of string keys.
            partition_id: Partition containing the keys.
            create: Whether to register missing keys.
            socket: Compatibility argument; RPCs use the client's socket pool.

        Raises:
            TypeError: If keys is neither a string nor a list of strings.
            ValueError: If keys is an empty list.
        """
        if isinstance(keys, str):
            keys = [keys]
        elif isinstance(keys, list):
            if len(keys) < 1:
                raise ValueError("Received an empty list as keys.")
            # validate all the elements are str
            if not all(isinstance(k, str) for k in keys):
                raise TypeError("Not all elements in `keys` are strings.")
        else:
            raise TypeError("Only string or list of strings are allowed as `keys`.")

        try:
            response_msg = await self._request_controller(
                request_type=ZMQRequestType.KV_RETRIEVE_META,
                response_type=ZMQRequestType.KV_RETRIEVE_META_RESPONSE,
                body={
                    "keys": keys,
                    "partition_id": partition_id,
                    "create": create,
                },
            )
            return response_msg.body.get("metadata", BatchMeta.empty())
        except Exception as e:
            raise RuntimeError(f"[{self.client_id}] Failed in async_kv_retrieve_meta: {e}") from e

    async def async_kv_prepare_update(
        self, keys: list[str], partition_id: str, fields: list[str], socket=None
    ) -> tuple[BatchMeta, dict[str, dict[str, Any]]]:
        """Register missing keys and describe each field's type and which rows hold it.

        Return the keys' BatchMeta and the merge plan consumed by ``async_update``.
        """
        response_msg = await self._request_controller(
            request_type=ZMQRequestType.KV_PREPARE_UPDATE,
            response_type=ZMQRequestType.KV_PREPARE_UPDATE_RESPONSE,
            body={"keys": keys, "partition_id": partition_id, "fields": fields},
        )
        merge_plan = serial_utils.decode_field_schema(response_msg.body["merge_plan"])
        return response_msg.body["metadata"], merge_plan

    async def async_kv_clear_fields(self, keys: list[str], partition_id: str, fields: list[str]) -> None:
        """Clear fields of existing keys while keeping the keys, tags and consumption status.

        The controller marks the fields as unproduced before storage deletes them, so no
        reader sees a ready field whose value is gone.
        """
        metadata = await self._kv_clear_fields_in_controller(keys, partition_id, fields)
        if metadata.size > 0:
            await self.storage_manager.clear_data(metadata, fields=metadata.field_names)

    async def _kv_clear_fields_in_controller(
        self, keys: list[str], partition_id: str, fields: list[str], socket=None
    ) -> BatchMeta:
        """Clear fields in the controller and return the cleared cells' metadata."""
        response_msg = await self._request_controller(
            request_type=ZMQRequestType.KV_CLEAR_FIELDS,
            response_type=ZMQRequestType.KV_CLEAR_FIELDS_RESPONSE,
            body={"keys": keys, "partition_id": partition_id, "fields": fields},
        )
        return response_msg.body["metadata"]

    async def async_kv_retrieve_keys(
        self,
        global_indexes: list[int] | int,
        partition_id: str,
        socket: zmq.asyncio.Socket | None = None,
    ) -> list[str]:
        """Return keys for global indexes in a partition.

        Args:
            global_indexes: One index or a nonempty list of integer indexes.
            partition_id: Partition to query.
            socket: Compatibility argument; RPCs use the client's socket pool.

        Raises:
            TypeError: If indexes is neither an integer nor a list of integers.
            ValueError: If indexes is an empty list.
            RuntimeError: If any requested index has no key or the controller request fails.
        """

        if isinstance(global_indexes, int):
            global_indexes = [global_indexes]
        elif isinstance(global_indexes, list):
            if len(global_indexes) < 1:
                raise ValueError("Received an empty list as `global_indexes`.")
            # validate all the elements are int
            if not all(isinstance(idx, int) for idx in global_indexes):
                raise TypeError("Not all elements in `global_indexes` are int.")
        else:
            raise TypeError("Only int or list of int are allowed as `global_indexes`.")

        try:
            response_msg = await self._request_controller(
                request_type=ZMQRequestType.KV_RETRIEVE_KEYS,
                response_type=ZMQRequestType.KV_RETRIEVE_KEYS_RESPONSE,
                body={"global_indexes": global_indexes, "partition_id": partition_id},
            )
            keys = response_msg.body.get("keys", [])
            if len(keys) != len(global_indexes):
                raise RuntimeError("Some global_indexes have no corresponding keys!")
            return keys
        except Exception as e:
            raise RuntimeError(f"[{self.client_id}]: Error in kv_retrieve_indexes: {str(e)}") from e

    async def async_kv_list(
        self,
        partition_id: str | None = None,
        socket: zmq.asyncio.Socket | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Return ``{partition_id: {key: tags}}`` for one or all partitions.

        ``partition_id=None`` queries all partitions. ``socket`` is retained for
        compatibility; RPCs use the client's pool.
        """

        try:
            response_msg = await self._request_controller(
                request_type=ZMQRequestType.KV_LIST,
                response_type=ZMQRequestType.KV_LIST_RESPONSE,
                body={"partition_id": partition_id},
            )
            return response_msg.body.get("partition_info", {})
        except Exception as e:
            raise RuntimeError(f"[{self.client_id}]: Error in kv_list: {str(e)}") from e

    async def async_kv_describe(
        self,
        keys: list[str],
        partition_id: str,
        socket: zmq.asyncio.Socket | None = None,
    ) -> dict[str, Any]:
        """Describe each key's row and declared field types without reading payloads.

        Each row retains its own produced fields. Unlike ``kv_retrieve_meta``,
        missing keys raise instead of being omitted.

        Returns:
            ``{"rows": {key: {"global_index", "fields", "unfinalized", "links", "tag"}},
            "field_schema": {field: {"is_nested", "is_non_tensor"}}}``.
        """
        response = await self._request_controller(
            request_type=ZMQRequestType.KV_DESCRIBE,
            response_type=ZMQRequestType.KV_DESCRIBE_RESPONSE,
            body={"keys": keys, "partition_id": partition_id},
        )
        return {"rows": response.body["rows"], "field_schema": response.body["field_schema"]}

    async def async_kv_exists(
        self,
        keys: list[str],
        partition_id: str,
        socket: zmq.asyncio.Socket | None = None,
    ) -> dict[str, list[str]]:
        """Map existing keys to their ready fields; see ``transfer_queue.kv_exists``."""
        response = await self._request_controller(
            request_type=ZMQRequestType.KV_EXISTS,
            response_type=ZMQRequestType.KV_EXISTS_RESPONSE,
            body={"keys": keys, "partition_id": partition_id},
        )
        return response.body["ready"]

    async def async_kv_link(
        self,
        keys: list[str],
        partition_id: str,
        public_keys: list[str],
        fields: list[str],
        socket: zmq.asyncio.Socket | None = None,
    ) -> None:
        """Link fields of each key to its public key's payload; see ``transfer_queue.kv_link``."""
        # Only SimpleStorage resolves links on read; other backends would read the linking key's own cells.
        if not isinstance(self.storage_manager, AsyncSimpleStorageManager):
            raise NotImplementedError(f"{type(self.storage_manager).__name__} does not support kv_link")
        await self._request_controller(
            request_type=ZMQRequestType.KV_LINK,
            response_type=ZMQRequestType.KV_LINK_RESPONSE,
            body={"keys": keys, "partition_id": partition_id, "public_keys": public_keys, "fields": fields},
        )

    async def async_link(
        self,
        metadata: BatchMeta,
        public_metadata: BatchMeta,
        fields: list[str] | str,
        socket: zmq.asyncio.Socket | None = None,
    ) -> BatchMeta:
        """Link sample fields to the corresponding public rows without requiring keys.

        Args:
            metadata: Rows in one non-public partition that will read the linked fields.
            public_metadata: Aligned rows in ``PUBLIC_PARTITION`` that produced those fields.
            fields: One field name or a list of field names.

        A linked public row cannot be overwritten or cleared with ``clear_samples``.
        After all incoming links are removed, ``kv_clear_unlinked_public`` can free it.

        Warning:
            PUT checks links but does not reserve public rows. Overwriting while
            another worker links can expose both the old and new payloads. Write
            public rows before linking. To replace them, remove all links, finish
            the write, then link again.

        Returns:
            BatchMeta for the input rows with the union of existing and linked fields,
            carrying the new links and preserving ``extra_info`` and ``custom_meta``.
            Subsequent reads and writes through it honor the links.

        Raises:
            ValueError: If metadata spans partitions or public_metadata contains non-public rows.
            RuntimeError: If the controller rejects the link; see ``TransferQueueController.link``.
            NotImplementedError: If the backend is not SimpleStorage.
        """
        if not isinstance(self.storage_manager, AsyncSimpleStorageManager):
            raise NotImplementedError(f"{type(self.storage_manager).__name__} does not support link")
        fields = [fields] if isinstance(fields, str) else list(fields)
        partition_ids = set(metadata.partition_ids)
        if len(partition_ids) != 1:
            raise ValueError(f"metadata must cover exactly one partition, got {sorted(partition_ids)}")
        if set(public_metadata.partition_ids) != {PUBLIC_PARTITION}:
            raise ValueError(f"public_metadata must cover only partition {PUBLIC_PARTITION!r}")
        response = await self._request_controller(
            request_type=ZMQRequestType.LINK,
            response_type=ZMQRequestType.LINK_RESPONSE,
            body={
                "global_indexes": metadata.global_indexes,
                "partition_id": partition_ids.pop(),
                "public_indexes": public_metadata.global_indexes,
                "fields": fields,
                "data_fields": list(dict.fromkeys([*metadata.field_names, *fields])),
            },
        )
        linked: BatchMeta = response.body["metadata"]
        # The controller knows only synced custom_meta, and extra_info never leaves the caller.
        linked.extra_info = dict(metadata.extra_info)
        linked.custom_meta = [dict(meta) for meta in metadata.custom_meta]
        return linked

    async def _check_public_write(self, global_indexes: list[int], socket=None) -> None:
        await self._request_controller(
            request_type=ZMQRequestType.CHECK_PUBLIC_WRITE,
            response_type=ZMQRequestType.CHECK_PUBLIC_WRITE_RESPONSE,
            body={"global_indexes": global_indexes},
        )

    async def async_clear_unlinked_public(self, stale_keys: list[str] | None = None) -> list[str]:
        """Delete unlinked public keys and return their names; see ``kv_clear_unlinked_public``."""
        metadata, keys = await self._mark_unlinked_public_clearing(stale_keys)
        if metadata.size > 0:
            await self.storage_manager.clear_data(metadata)
            await self._clear_meta_in_controller(metadata)
        return keys

    async def _mark_unlinked_public_clearing(
        self, stale_keys: list[str] | None, socket=None
    ) -> tuple[BatchMeta, list[str]]:
        response = await self._request_controller(
            request_type=ZMQRequestType.MARK_UNLINKED_PUBLIC_CLEARING,
            response_type=ZMQRequestType.MARK_UNLINKED_PUBLIC_CLEARING_RESPONSE,
            body={"stale_keys": stale_keys},
        )
        return response.body["metadata"], response.body["keys"]

    def close(self) -> None:
        """Close storage, pooled sockets, and the client's ZMQ context.

        The caller must first finish all RPCs: context destruction closes sockets
        and is not thread-safe. Subclasses must stop and join their socket-owning
        threads before calling this method; see ``TransferQueueClient.close``.
        """
        try:
            if hasattr(self, "storage_manager") and self.storage_manager:
                if hasattr(self.storage_manager, "close"):
                    self.storage_manager.close()
        except Exception as e:
            logger.warning(f"Error closing storage manager: {e}")

        # Tear down the shared context last; linger=0 so shutdown cannot hang.
        if not self._can_destroy_zmq_context():
            # Detach first, or the finalizer would later run the destroy() judged unsafe here.
            self._detach_finalizer()
            logger.warning(
                f"[{self.client_id}]: Skipping zmq_context.destroy() because a thread owning "
                f"sockets on it is still alive; destroy() is not thread-safe. Leaking the context."
            )
            return
        try:
            # Close pooled sockets before the context that owns them.
            self.controller_rpc_pool.close()
            if hasattr(self, "zmq_context") and self.zmq_context is not None:
                self.zmq_context.destroy(linger=0)
        except Exception as e:
            logger.warning(f"[{self.client_id}]: Error terminating zmq_context: {e}")
        finally:
            # This close() superseded the finalizer; disarm it so it cannot run twice.
            self._detach_finalizer()

    def _detach_finalizer(self) -> None:
        """Disarm the context finalizer, if one was armed."""
        finalizer = getattr(self, "_finalizer", None)
        if finalizer is not None:
            finalizer.detach()

    def _can_destroy_zmq_context(self) -> bool:
        """Return whether all threads borrowing the client's context have stopped.

        Consult a sharing storage manager for its notify thread. Subclasses add
        checks for their own socket-owning threads.
        """
        manager = getattr(self, "storage_manager", None)
        if manager is not None:
            can_destroy = getattr(manager, "can_destroy_zmq_context", None)
            # Only consult a manager that actually shares this context; one with its own
            # context has no say in when the client's is destroyed.
            if callable(can_destroy) and getattr(manager, "zmq_context", None) is self.zmq_context:
                if not can_destroy():
                    return False
        return True

    # ==================== Selective Data Dump API ====================
    async def async_validate_dump_schema(
        self,
        partition_id: str,
        field_schema: dict,
        socket: zmq.asyncio.Socket | None = None,
    ) -> None:
        """Reject incompatible destination fields before restoring payloads."""
        field_schema = serial_utils.encode_field_schema(field_schema)
        await self._request_controller(
            request_type=ZMQRequestType.VALIDATE_DUMP_SCHEMA,
            response_type=ZMQRequestType.VALIDATE_DUMP_SCHEMA_RESPONSE,
            body={"partition_id": partition_id, "field_schema": field_schema},
        )

    async def async_dump_rows_by_index(self, shard_dir: str, fields_by_index: dict[int, list[str]]) -> dict[str, Any]:
        """Dump selected rows and fields into per-storage-unit shards.

        Args:
            shard_dir: Output directory.
            fields_by_index: Global row indexes mapped to their produced field names.

        Returns:
            ``{"shards", "row_schema"}`` with each shard and the rows' stored field types.

        Raises:
            RuntimeError: If storage is uninitialized or a requested row has no stored data.
            NotImplementedError: If the backend does not support dumps.
        """
        self._require_storage_manager("dump")
        return await self.storage_manager.dump_rows_by_index(shard_dir, fields_by_index)

    async def async_load_rows_by_key(
        self,
        partition_id: str,
        rows: dict[str, dict[str, Any]],
        shards: list[dict[str, Any]],
    ) -> int:
        """Restore payloads at their keys' current indexes and return bytes read.

        As with ``kv_batch_put``, new keys remain registered after failure and
        writes may be partial. Retries are idempotent because keys keep their indexes.

        Raises:
            ValueError: If restored tags are not string-keyed native MessagePack dictionaries.
            RuntimeError: If storage is uninitialized or a unit fails.
            NotImplementedError: If the backend does not support selective loads.
        """
        self._require_storage_manager("load")
        if not rows:
            return 0
        tags = [row["tag"] for row in rows.values()]
        self.validate_tags(tags)
        metadata = await self.async_kv_retrieve_meta(list(rows), partition_id, create=True)
        target_indexes = dict(zip(rows, metadata.global_indexes, strict=True))
        for shard in shards:
            for record in shard["records"]:
                record["target_index"] = target_indexes[record["key"]]
        bytes_read = await self.storage_manager.load_rows_by_index(partition_id, shards)
        metadata.update_custom_meta(tags)
        await self.async_set_custom_meta(metadata, _validate_tags=False)
        return bytes_read

    # ==================== Checkpoint API ====================
    async def async_save_controller_checkpoint(
        self,
        path: str,
        socket: zmq.asyncio.Socket | None = None,
    ) -> None:
        """Save controller state to a file on the controller node and wait for acknowledgement.

        ``path`` must be an absolute .pkl path writable from that node. The controller
        serializes its own state. ``socket`` is retained for compatibility.
        Communication or response errors raise RuntimeError.
        """
        try:
            await self._request_controller(
                request_type=ZMQRequestType.SAVE_CONTROLLER_CHECKPOINT,
                response_type=ZMQRequestType.SAVE_CONTROLLER_CHECKPOINT_RESPONSE,
                body={"path": path},
            )
        except Exception as e:
            raise RuntimeError(f"[{self.client_id}]: Error in save_controller_checkpoint: {str(e)}") from e

    async def async_load_controller_checkpoint(
        self,
        path: str,
        socket: zmq.asyncio.Socket | None = None,
    ) -> None:
        """Restore controller state from a file and wait for acknowledgement.

        ``path`` must be an absolute .pkl path saved by ``async_save_controller_checkpoint``
        and readable from the controller node. The controller replaces its in-memory
        state. ``socket`` is retained for compatibility. Communication or response
        errors raise RuntimeError.
        """
        try:
            await self._request_controller(
                request_type=ZMQRequestType.LOAD_CONTROLLER_CHECKPOINT,
                response_type=ZMQRequestType.LOAD_CONTROLLER_CHECKPOINT_RESPONSE,
                body={"path": path},
            )
        except Exception as e:
            raise RuntimeError(f"[{self.client_id}]: Error in load_controller_checkpoint: {str(e)}") from e

    async def async_save_storage_checkpoint(self, checkpoint_dir: str) -> None:
        """Save storage through the manager, which requests all units concurrently.

        Unit files go under ``checkpoint_dir/storage_units/``. Raise RuntimeError if
        storage is uninitialized, or NotImplementedError if checkpoints are unsupported.
        """
        self._require_storage_manager("checkpoint")
        await self.storage_manager.save_checkpoint(checkpoint_dir)

    async def async_load_storage_checkpoint(self, checkpoint_dir: str) -> None:
        """Restore storage through the manager, requesting all units concurrently.

        Existing in-memory data is replaced. ``checkpoint_dir`` must contain the
        ``storage_units/`` directory and manifest saved by ``async_save_storage_checkpoint``.

        Raises:
            RuntimeError: If storage is uninitialized.
            FileNotFoundError: If the storage-unit manifest is missing.
            ValueError: If the number of storage units differs from the checkpoint.
        """
        self._require_storage_manager("checkpoint")
        await self.storage_manager.load_checkpoint(checkpoint_dir)


class TransferQueueClient(AsyncTransferQueueClient):
    """Blocking client backed by one persistent event loop in a background thread.

    Public methods keep explicit signatures and run their async counterparts on
    that loop. See those methods for shared arguments, results, and error semantics.
    """

    def __init__(
        self,
        client_id: str,
        controller_info: ZMQServerInfo,
        zmq_io_threads: int | None = None,
        zmq_max_sockets: int | None = None,
    ):
        """Create a client and start its background event loop.

        Args:
            client_id: Unique client identifier.
            controller_info: Controller endpoint.
            zmq_io_threads: Native I/O threads; defaults to ``TQ_CLIENT_ZMQ_IO_THREADS`` (8).
            zmq_max_sockets: Per-client socket limit; defaults to ``TQ_CLIENT_ZMQ_MAX_SOCKETS``
                or 8192. See ``AsyncTransferQueueClient`` for shared budgets and limits.
        """
        super().__init__(
            client_id,
            controller_info,
            zmq_io_threads,
            zmq_max_sockets,
        )

        # create new event loop in a separate thread
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._start_loop, daemon=True)
        self._thread.start()

    def _start_loop(self):
        """Start the synchronous loop."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run_coroutine(self, coro):
        """Run a coroutine on the client's background event loop."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    # ==================== Basic API ====================
    def get_meta(
        self,
        data_fields: list[str],
        batch_size: int,
        partition_id: str,
        mode: str = "fetch",
        task_name: str | None = None,
        sampling_config: dict[str, Any] | None = None,
    ) -> BatchMeta:
        """Fetch sample metadata; see ``async_get_meta`` for arguments and errors.

        Examples:
            With ``GRPOGroupNSampler`` configured, fetch a ready group of four
            adjacent samples from an existing partition:

            >>> meta = client.get_meta(
            ...     data_fields=["prompts"], batch_size=4, partition_id="train_0",
            ...     task_name="rollout", sampling_config={"n_samples_per_prompt": 4},
            ... )

            Inspect rows even if they are unready or already consumed:

            >>> inspected = client.get_meta(
            ...     data_fields=["prompts"], batch_size=4, partition_id="train_0",
            ...     mode="force_fetch",
            ... )
        """
        return self._run_coroutine(
            self.async_get_meta(
                data_fields=data_fields,
                batch_size=batch_size,
                partition_id=partition_id,
                mode=mode,
                task_name=task_name,
                sampling_config=sampling_config,
            )
        )

    def set_custom_meta(self, metadata: BatchMeta) -> None:
        """Publish per-sample tags; see ``async_set_custom_meta`` for validation rules.

        Example:
            Publish one tag dictionary per sample, in the existing metadata order:

            >>> meta.update_custom_meta([{"score": 0.9} for _ in range(meta.size)])
            >>> client.set_custom_meta(meta)
        """

        return self._run_coroutine(self.async_set_custom_meta(metadata=metadata))

    def put(
        self,
        data: TensorDict,
        metadata: BatchMeta | None = None,
        partition_id: str | None = None,
        data_parser: Callable[[Any], Any] | None = None,
    ) -> BatchMeta:
        """Write payloads and tags, returning the metadata used for the write.

        See ``async_put`` for parser and linked-field constraints. Without metadata,
        submit all initial data to an empty partition to avoid overwriting rows.

        Examples:
            With an initialized client, keep four samples per prompt adjacent:

            >>> import torch
            >>> from tensordict import TensorDict
            >>> prompts = torch.arange(6).reshape(2, 3).repeat_interleave(4, dim=0)
            >>> data = TensorDict({"prompts": prompts}, batch_size=[8])
            >>> meta = client.put(data, partition_id="train_0")

            Write only the new output field to those same rows:

            >>> output = TensorDict({"response": torch.ones(8, 2)}, batch_size=[8])
            >>> meta = client.put(output, metadata=meta)
        """
        return self._run_coroutine(
            self.async_put(data=data, metadata=metadata, partition_id=partition_id, data_parser=data_parser)
        )

    def update(
        self,
        metadata: BatchMeta,
        values: TensorDict,
        merge_fn: Callable[[Any, Any], Any],
        merge_plan: dict[str, dict[str, Any]],
        finalize: bool = True,
    ) -> BatchMeta:
        """Blocking counterpart of ``AsyncTransferQueueClient.async_update``."""
        return self._run_coroutine(self.async_update(metadata, values, merge_fn, merge_plan, finalize=finalize))

    def get_data(
        self,
        metadata: BatchMeta,
        data_parser: Callable[[Any, Any], Any] | None = None,
        parser_args: list | None = None,
    ) -> TensorDict:
        """Read fields in metadata sample order; see ``async_get_data`` for parser options.

        Example:
            Fetch metadata for four ready samples, then read their payloads:

            >>> meta = client.get_meta(
            ...     data_fields=["prompts"], batch_size=4, partition_id="train_0",
            ...     task_name="rollout",
            ... )
            >>> batch = client.get_data(meta)
        """
        return self._run_coroutine(
            self.async_get_data(metadata=metadata, data_parser=data_parser, parser_args=parser_args)
        )

    def clear_partition(self, partition_id: str):
        """Blocking counterpart of ``AsyncTransferQueueClient.async_clear_partition``."""
        return self._run_coroutine(self.async_clear_partition(partition_id=partition_id))

    def clear_samples(self, metadata: BatchMeta):
        """Blocking counterpart of ``AsyncTransferQueueClient.async_clear_samples``."""
        return self._run_coroutine(self.async_clear_samples(metadata=metadata))

    # ==================== Status Query API ====================
    def get_consumption_status(
        self,
        task_name: str,
        partition_id: str,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return sample indexes and per-task consumption flags; see ``async_get_consumption_status``.

        Example:
            Find rows the rollout task has not consumed (flag 0):

            >>> indexes, consumed = client.get_consumption_status(
            ...     task_name="rollout", partition_id="train_0",
            ... )
            >>> if consumed is not None:
            ...     pending_indexes = indexes[consumed == 0].tolist()
        """
        return self._run_coroutine(self.async_get_consumption_status(task_name, partition_id))

    def get_production_status(
        self,
        data_fields: list[str],
        partition_id: str,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return sample indexes and per-field readiness flags; see ``async_get_production_status``.

        Example:
            Flags have one row per sample and one column per distinct field.
            Find rows with at least one field still unready (flag 0):

            >>> indexes, ready = client.get_production_status(
            ...     data_fields=["prompts", "response"], partition_id="train_0",
            ... )
            >>> if ready is not None:
            ...     incomplete_indexes = indexes[(ready == 0).any(dim=1)].tolist()
        """
        return self._run_coroutine(self.async_get_production_status(data_fields=data_fields, partition_id=partition_id))

    def check_consumption_status(self, task_name: str, partition_id: str) -> bool:
        """Return whether this task consumed every row; missing or empty status returns False.

        See ``get_partition_list`` for an example using this check across partitions,
        and ``async_check_consumption_status`` for shared error semantics.
        """
        return self._run_coroutine(self.async_check_consumption_status(task_name=task_name, partition_id=partition_id))

    def check_production_status(self, data_fields: list[str], partition_id: str) -> bool:
        """Return whether all queried production flags are ready; see ``async_check_production_status``.

        Example:
            Check whether both fields are ready on every row of an existing partition:

            >>> all_ready = client.check_production_status(
            ...     data_fields=["prompts", "response"], partition_id="train_0",
            ... )
        """
        return self._run_coroutine(
            self.async_check_production_status(data_fields=data_fields, partition_id=partition_id)
        )

    def reset_consumption(self, partition_id: str, task_name: str | None = None) -> bool:
        """Allow a task to consume the partition again, or reset all tasks when ``task_name=None``.

        Return the controller's success flag; see ``async_reset_consumption`` for errors.

        Example:
            Consume ready samples again without rewriting their payloads:

            >>> reset = client.reset_consumption(partition_id="train_0", task_name="rollout")
            >>> if reset:
            ...     meta = client.get_meta(
            ...         data_fields=["prompts"], batch_size=4, partition_id="train_0",
            ...         task_name="rollout",
            ...     )
        """
        return self._run_coroutine(self.async_reset_consumption(partition_id, task_name))

    def get_partition_list(
        self,
    ) -> list[str]:
        """Return partition IDs managed by the controller; see ``async_get_partition_list``.

        Example:
            List partitions not yet fully consumed by the rollout task:

            >>> pending_partitions = []
            >>> for partition_id in client.get_partition_list():
            ...     if not client.check_consumption_status(task_name="rollout", partition_id=partition_id):
            ...         pending_partitions.append(partition_id)
        """
        return self._run_coroutine(self.async_get_partition_list())

    # ==================== KV Interface API ====================
    def kv_retrieve_meta(
        self,
        keys: list[str] | str,
        partition_id: str,
        create: bool = False,
    ) -> BatchMeta:
        """Blocking counterpart of ``AsyncTransferQueueClient.async_kv_retrieve_meta``."""
        return self._run_coroutine(self.async_kv_retrieve_meta(keys=keys, partition_id=partition_id, create=create))

    def kv_prepare_update(
        self, keys: list[str], partition_id: str, fields: list[str]
    ) -> tuple[BatchMeta, dict[str, dict[str, Any]]]:
        """Blocking counterpart of ``AsyncTransferQueueClient.async_kv_prepare_update``."""
        return self._run_coroutine(self.async_kv_prepare_update(keys, partition_id, fields))

    def kv_retrieve_keys(
        self,
        global_indexes: list[int] | int,
        partition_id: str,
    ) -> BatchMeta:
        """Blocking counterpart of ``AsyncTransferQueueClient.async_kv_retrieve_keys``."""

        return self._run_coroutine(
            self.async_kv_retrieve_keys(global_indexes=global_indexes, partition_id=partition_id)
        )

    def kv_list(
        self,
        partition_id: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Blocking counterpart of ``AsyncTransferQueueClient.async_kv_list``."""

        return self._run_coroutine(self.async_kv_list(partition_id=partition_id))

    def kv_describe(self, keys: list[str], partition_id: str) -> dict[str, Any]:
        """Blocking counterpart of ``AsyncTransferQueueClient.async_kv_describe``."""
        return self._run_coroutine(self.async_kv_describe(keys, partition_id))

    def kv_link(self, keys: list[str], partition_id: str, public_keys: list[str], fields: list[str]) -> None:
        """Blocking counterpart of ``AsyncTransferQueueClient.async_kv_link``."""
        return self._run_coroutine(self.async_kv_link(keys, partition_id, public_keys, fields))

    def link(self, metadata: BatchMeta, public_metadata: BatchMeta, fields: list[str] | str) -> BatchMeta:
        """Blocking counterpart of ``AsyncTransferQueueClient.async_link``."""
        return self._run_coroutine(self.async_link(metadata, public_metadata, fields))

    def clear_unlinked_public(self, stale_keys: list[str] | None = None) -> list[str]:
        """Blocking counterpart of ``AsyncTransferQueueClient.async_clear_unlinked_public``."""
        return self._run_coroutine(self.async_clear_unlinked_public(stale_keys))

    # ==================== Selective Data Dump API ====================

    def validate_dump_schema(self, partition_id: str, field_schema: dict) -> None:
        """Blocking counterpart of ``AsyncTransferQueueClient.async_validate_dump_schema``."""
        return self._run_coroutine(self.async_validate_dump_schema(partition_id, field_schema))

    def dump_rows_by_index(self, shard_dir: str, fields_by_index: dict[int, list[str]]) -> dict[str, Any]:
        """Blocking counterpart of ``AsyncTransferQueueClient.async_dump_rows_by_index``."""
        return self._run_coroutine(self.async_dump_rows_by_index(shard_dir, fields_by_index))

    def load_rows_by_key(self, partition_id: str, rows: dict, shards: list[dict]) -> int:
        """Blocking counterpart of ``AsyncTransferQueueClient.async_load_rows_by_key``."""
        return self._run_coroutine(self.async_load_rows_by_key(partition_id, rows, shards))

    # ==================== Checkpoint API ====================
    def save_controller_checkpoint(self, path: str) -> None:
        """Blocking counterpart of ``AsyncTransferQueueClient.async_save_controller_checkpoint``."""
        return self._run_coroutine(self.async_save_controller_checkpoint(path))

    def load_controller_checkpoint(self, path: str) -> None:
        """Blocking counterpart of ``AsyncTransferQueueClient.async_load_controller_checkpoint``."""
        return self._run_coroutine(self.async_load_controller_checkpoint(path))

    def save_storage_checkpoint(self, checkpoint_dir: str) -> None:
        """Blocking counterpart of ``AsyncTransferQueueClient.async_save_storage_checkpoint``."""
        return self._run_coroutine(self.async_save_storage_checkpoint(checkpoint_dir))

    def load_storage_checkpoint(self, checkpoint_dir: str) -> None:
        """Blocking counterpart of ``AsyncTransferQueueClient.async_load_storage_checkpoint``."""
        return self._run_coroutine(self.async_load_storage_checkpoint(checkpoint_dir))

    def close(self) -> None:
        """Stop and join the background loop before closing shared resources.

        ZMQ context destruction is not thread-safe. If the thread outlives the join
        timeout, ``_can_destroy_zmq_context`` prevents destroying its context.
        """

        if hasattr(self, "_loop") and self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)

            if hasattr(self, "_thread") and self._thread is not None:
                self._thread.join(timeout=5.0)
                if self._thread.is_alive():
                    logger.warning(f"[{self.client_id}]: Background thread did not stop within timeout")

            try:
                self._loop.close()
            except Exception as e:
                logger.warning(f"[{self.client_id}]: Error closing event loop: {e}")

        super().close()

    def _can_destroy_zmq_context(self) -> bool:
        """Allow destruction only after the client and borrowing manager threads stop."""
        thread = getattr(self, "_thread", None)
        if thread is not None and thread.is_alive():
            return False
        return super()._can_destroy_zmq_context()
