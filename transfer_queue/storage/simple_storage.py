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
import weakref
from threading import Event
from typing import Any
from uuid import uuid4

import ray
import zmq

from transfer_queue.utils.common import limit_pytorch_auto_parallel_threads
from transfer_queue.utils.enum_utils import Role
from transfer_queue.utils.logging_utils import get_logger
from transfer_queue.utils.perf_utils import IntervalPerfMonitor
from transfer_queue.utils.zmq_utils import (
    ZMQMessage,
    ZMQRequestType,
    ZMQServerInfo,
    ZMQServerTransport,
    create_zmq_socket,
    get_node_ip_address,
)

logger = get_logger(__name__)

TQ_STORAGE_POLLER_TIMEOUT = int(os.environ.get("TQ_STORAGE_POLLER_TIMEOUT", 5))  # in seconds
TQ_NUM_THREADS = int(os.environ.get("TQ_NUM_THREADS", 8))


class StorageUnitData:
    """Storage unit for managing 2D data structure (samples × fields).

    Uses dict-based storage keyed by global_index instead of pre-allocated list.
    This allows O(1) insert/delete without index translation and avoids capacity bloat.

    Data Structure Example:
        field_data = {
            "field_name1": {global_index_0: item1, global_index_3: item2, ...},
            "field_name2": {global_index_0: item3, global_index_3: item4, ...},
        }
    """

    def __init__(self, storage_size: int):
        # field_name -> {global_index: data} nested dict
        self.field_data: dict[str, dict] = {}
        # Capacity upper bound (not pre-allocated list length)
        self.storage_size = storage_size
        # Track active global_index keys for O(1) capacity checks
        self._active_keys: set = set()

    def get_data(self, fields: list[str], global_indexes: list) -> dict[str, list]:
        """Get data by global index keys.

        Args:
            fields: Field names used for getting data.
            global_indexes: Global indexes used as dict keys.

        Returns:
            dict with field names as keys, corresponding data list as values.
        """
        result: dict[str, list] = {}
        for field in fields:
            if field not in self.field_data:
                raise ValueError(
                    f"StorageUnitData get_data: field '{field}' not found. Available: {list(self.field_data.keys())}"
                )
            try:
                result[field] = [self.field_data[field][k] for k in global_indexes]
            except KeyError as e:
                raise KeyError(f"StorageUnitData get_data: key {e} not found in field '{field}'") from e
        return result

    def put_data(self, field_data: dict[str, Any], global_indexes: list) -> None:
        """Put data into storage.

        Args:
            field_data: Dict with field names as keys, data list as values.
            global_indexes: Global indexes to use as dict keys.
        """
        # Capacity is enforced per unique sample key, not counted per-field
        new_global_keys = [k for k in global_indexes if k not in self._active_keys]
        if len(self._active_keys) + len(new_global_keys) > self.storage_size:
            raise ValueError(
                f"Storage capacity exceeded: {len(self._active_keys)} existing + "
                f"{len(new_global_keys)} new > {self.storage_size}"
            )
        for f, values in field_data.items():
            if len(values) != len(global_indexes):
                raise ValueError(
                    f"StorageUnitData put_data: field '{f}' values length {len(values)} "
                    f"!= global_indexes length {len(global_indexes)}, length mismatch"
                )
            if f not in self.field_data:
                self.field_data[f] = {}
            for key, val in zip(global_indexes, values, strict=True):
                self.field_data[f][key] = val
        self._active_keys.update(global_indexes)

    def clear(self, keys: list[int]) -> None:
        """Remove data at given global index keys, immediately freeing memory.

        Args:
            keys: Global indexes to remove.
        """
        for f in self.field_data:
            for key in keys:
                self.field_data[f].pop(key, None)
        self._active_keys -= set(keys)


@ray.remote(num_cpus=1)
class SimpleStorageUnit:
    """A storage unit that provides distributed data storage functionality.

    This class represents a storage unit that can store data in a 2D structure
    (samples, data_fields) and provides ZMQ-based communication for put/get/clear operations.

    Note: We use Ray decorator (@ray.remote) only for initialization purposes.
    We do NOT use Ray's .remote() call capabilities - the storage unit runs
    as a standalone process with its own ZMQ server socket.

    Attributes:
        storage_unit_id: Unique identifier for this storage unit.
        storage_unit_size: Maximum number of elements that can be stored.
        storage_data: Internal StorageUnitData instance for data management.
        zmq_server_info: ZMQ connection information for clients.
    """

    def __init__(self, storage_unit_size: int):
        """Initialize a SimpleStorageUnit with the specified size.

        Args:
            storage_unit_size: Maximum number of elements that can be stored in this storage unit.
        """
        self.storage_unit_id = f"TQ_STORAGE_UNIT_{uuid4().hex[:8]}"
        self.storage_unit_size = storage_unit_size

        self.storage_data = StorageUnitData(self.storage_unit_size)

        # Shutdown event for graceful termination
        self._shutdown_event = Event()

        self._init_zmq_transport()
        self._start_process_put_get()

        # Register finalizer for graceful cleanup when garbage collected
        self._finalizer = weakref.finalize(
            self,
            self._shutdown,
            self._shutdown_event,
            self._transport,
        )

    def _init_zmq_transport(self) -> None:
        """
        Initialize ZMQ transport layer with ROUTER-DEALER proxy:
        - put_get_socket (ROUTER): Frontend for receiving client requests.
        - Backend DEALER: Bound to inproc, workers connect locally.
        """
        self._transport = ZMQServerTransport(node_ip=get_node_ip_address())
        self._transport.create_router_proxy("put_get_socket")

        self.zmq_server_info = self._transport.build_server_info(
            role=Role.STORAGE,
            server_id=str(self.storage_unit_id),
        )

    def _start_process_put_get(self) -> None:
        """Start worker threads for handling requests."""
        # Start worker thread (proxy is already started by create_router_proxy)
        self._transport.start_daemon_thread(
            target=self._worker_routine,
            name=f"StorageUnitWorkerThread-{self.storage_unit_id}",
        )

    def _worker_routine(self) -> None:
        """Worker thread for processing requests."""

        worker_socket = create_zmq_socket(self._transport.zmq_ctx, zmq.DEALER, self._transport.node_ip)
        worker_socket.connect(self._transport.get_inproc_addr("put_get_socket"))

        poller = zmq.Poller()
        poller.register(worker_socket, zmq.POLLIN)

        logger.info(f"[{self.storage_unit_id}]: worker thread started...")
        perf_monitor = IntervalPerfMonitor(caller_name=f"{self.storage_unit_id}")

        while not self._shutdown_event.is_set():
            try:
                socks = dict(poller.poll(TQ_STORAGE_POLLER_TIMEOUT * 1000))
            except zmq.error.ContextTerminated:
                # ZMQ context was terminated, exit gracefully
                logger.info(f"[{self.storage_unit_id}]: worker stopped gracefully (Context Terminated)")
                break
            except Exception as e:
                logger.warning(f"[{self.storage_unit_id}]: worker poll error: {e}")
                continue

            if self._shutdown_event.is_set():
                break

            if worker_socket in socks:
                # Messages received from proxy: [identity, serialized_msg_frame1, ...]
                messages = worker_socket.recv_multipart(copy=False)
                identity = messages[0]
                serialized_msg = messages[1:]

                request_msg = ZMQMessage.deserialize(serialized_msg)
                operation = request_msg.request_type

                try:
                    logger.debug(f"[{self.storage_unit_id}]: worker received operation: {operation}")

                    # Process request
                    if operation == ZMQRequestType.PUT_DATA:  # type: ignore[arg-type]
                        with perf_monitor.measure(op_type="PUT_DATA"):
                            response_msg = self._handle_put(request_msg)
                    elif operation == ZMQRequestType.GET_DATA:  # type: ignore[arg-type]
                        with perf_monitor.measure(op_type="GET_DATA"):
                            response_msg = self._handle_get(request_msg)
                    elif operation == ZMQRequestType.CLEAR_DATA:  # type: ignore[arg-type]
                        with perf_monitor.measure(op_type="CLEAR_DATA"):
                            response_msg = self._handle_clear(request_msg)
                    else:
                        response_msg = ZMQMessage.create(
                            request_type=ZMQRequestType.PUT_GET_OPERATION_ERROR,  # type: ignore[arg-type]
                            sender_id=self.storage_unit_id,
                            body={
                                "message": f"Storage unit id #{self.storage_unit_id} "
                                f"receive invalid operation: {operation}."
                            },
                        )
                except Exception as e:
                    logger.error(
                        f"[{self.storage_unit_id}]: worker error during {operation} "
                        f"from sender={request_msg.sender_id}: {type(e).__name__}: {e}"
                    )
                    response_msg = ZMQMessage.create(
                        request_type=ZMQRequestType.PUT_GET_ERROR,  # type: ignore[arg-type]
                        sender_id=self.storage_unit_id,
                        body={
                            "message": f"{self.storage_unit_id}, worker encountered error "
                            f"during operation {operation}: {str(e)}."
                        },
                    )

                # Send response back with identity for routing
                worker_socket.send_multipart([identity] + response_msg.serialize(), copy=False)

        logger.info(f"[{self.storage_unit_id}]: worker stopped.")
        poller.unregister(worker_socket)
        worker_socket.close(linger=0)

    def _handle_put(self, data_parts: ZMQMessage) -> ZMQMessage:
        """
        Handle put request, add or update data into storage unit.

        Args:
            data_parts: ZMQMessage from client.

        Returns:
            Put data success response ZMQMessage.
        """
        try:
            global_indexes = data_parts.body["global_indexes"]
            field_data = data_parts.body["data"]  # field_data should be a dict.
            data_parser = data_parts.body.get("data_parser", None)

            with limit_pytorch_auto_parallel_threads(
                target_num_threads=TQ_NUM_THREADS, info=f"[{self.storage_unit_id}] _handle_put"
            ):
                if data_parser is not None:
                    if not callable(data_parser):
                        raise TypeError(f"data_parser must be callable, got {type(data_parser).__name__}")

                    original_keys = set(field_data.keys())
                    original_lengths = {}
                    for k, v in field_data.items():
                        if hasattr(v, "shape") and isinstance(v.shape, tuple | list) and len(v.shape) > 0:
                            original_lengths[k] = v.shape[0]
                        else:
                            try:
                                original_lengths[k] = len(v)
                            except Exception:
                                original_lengths[k] = None

                    field_data = data_parser(field_data)

                    if not isinstance(field_data, dict):
                        raise TypeError(f"data_parser must return a dict, got {type(field_data).__name__}")

                    new_keys = set(field_data.keys())
                    if new_keys != original_keys:
                        raise ValueError(
                            f"data_parser must not change dict keys. "
                            f"Original keys: {sorted(original_keys)}, got: {sorted(new_keys)}"
                        )

                    for k, v in field_data.items():
                        if hasattr(v, "shape") and isinstance(v.shape, tuple | list) and len(v.shape) > 0:
                            new_len = v.shape[0]
                        else:
                            try:
                                new_len = len(v)
                            except Exception:
                                new_len = None

                        orig_len = original_lengths[k]
                        if orig_len is not None and new_len is not None and orig_len != new_len:
                            raise ValueError(
                                f"data_parser changed the number of elements for key '{k}': "
                                f"expected {orig_len}, got {new_len}"
                            )
                self.storage_data.put_data(field_data, global_indexes)

            # After put operation finish, send a message to the client
            response_msg = ZMQMessage.create(
                request_type=ZMQRequestType.PUT_DATA_RESPONSE,  # type: ignore[arg-type]
                sender_id=self.storage_unit_id,
                body={},
            )

            return response_msg
        except Exception as e:
            return ZMQMessage.create(
                request_type=ZMQRequestType.PUT_ERROR,  # type: ignore[arg-type]
                sender_id=self.storage_unit_id,
                body={
                    "message": f"Failed to put data into storage unit id "
                    f"#{self.storage_unit_id}, detail error message: {str(e)}"
                },
            )

    def _handle_get(self, data_parts: ZMQMessage) -> ZMQMessage:
        """
        Handle get request, return data from storage unit.

        Args:
            data_parts: ZMQMessage from client.

        Returns:
            Get data success response ZMQMessage, containing target data.
        """
        try:
            fields = data_parts.body["fields"]
            global_indexes = data_parts.body["global_indexes"]

            with limit_pytorch_auto_parallel_threads(
                target_num_threads=TQ_NUM_THREADS, info=f"[{self.storage_unit_id}] _handle_get"
            ):
                result_data = self.storage_data.get_data(fields, global_indexes)

            response_msg = ZMQMessage.create(
                request_type=ZMQRequestType.GET_DATA_RESPONSE,  # type: ignore[arg-type]
                sender_id=self.storage_unit_id,
                body={
                    "data": result_data,
                },
            )
        except Exception as e:
            logger.error(
                f"[{self.storage_unit_id}]: _handle_get error, "
                f"fields={fields}, global_indexes={global_indexes}: {type(e).__name__}: {e}"
            )
            response_msg = ZMQMessage.create(
                request_type=ZMQRequestType.GET_ERROR,  # type: ignore[arg-type]
                sender_id=self.storage_unit_id,
                body={
                    "message": f"Failed to get data from storage unit id #{self.storage_unit_id}, "
                    f"detail error message: {str(e)}"
                },
            )
        return response_msg

    def _handle_clear(self, data_parts: ZMQMessage) -> ZMQMessage:
        """
        Handle clear request, clear data in storage unit according to given global_indexes.

        Args:
            data_parts: ZMQMessage from client, including target global_indexes.

        Returns:
            Clear data success response ZMQMessage.
        """
        try:
            global_indexes = data_parts.body["global_indexes"]

            with limit_pytorch_auto_parallel_threads(
                target_num_threads=TQ_NUM_THREADS, info=f"[{self.storage_unit_id}] _handle_clear"
            ):
                self.storage_data.clear(global_indexes)

            response_msg = ZMQMessage.create(
                request_type=ZMQRequestType.CLEAR_DATA_RESPONSE,  # type: ignore[arg-type]
                sender_id=self.storage_unit_id,
                body={"message": f"Clear data in storage unit id #{self.storage_unit_id} successfully."},
            )
        except Exception as e:
            response_msg = ZMQMessage.create(
                request_type=ZMQRequestType.CLEAR_DATA_ERROR,  # type: ignore[arg-type]
                sender_id=self.storage_unit_id,
                body={
                    "message": f"Failed to clear data in storage unit id #{self.storage_unit_id}, "
                    f"detail error message: {str(e)}"
                },
            )
        return response_msg

    @staticmethod
    def _shutdown(
        shutdown_event: Event,
        transport: ZMQServerTransport | None,
    ) -> None:
        """Clean up resources on garbage collection."""
        logger.info("Shutting down SimpleStorageUnit resources...")

        # Signal all threads to stop
        shutdown_event.set()

        # Terminate ZMQ context to unblock proxy and workers
        if transport and transport.zmq_ctx:
            transport.zmq_ctx.term()

        # Wait for threads to finish (with timeout)
        if transport:
            for t in transport.threads:
                if t.is_alive():
                    t.join(timeout=5)

        logger.info("SimpleStorageUnit resources shutdown complete.")

    def get_zmq_server_info(self) -> ZMQServerInfo:
        """Get the ZMQ server information for this storage unit."""
        return self.zmq_server_info
