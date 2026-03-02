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
import time
import weakref
from threading import Event, Thread
from typing import Any, Optional
from uuid import uuid4

import ray
import zmq
from ray.util import get_node_ip_address

from transfer_queue.utils.common import limit_pytorch_auto_parallel_threads
from transfer_queue.utils.enum_utils import TransferQueueRole
from transfer_queue.utils.perf_utils import IntervalPerfMonitor
from transfer_queue.utils.zmq_utils import ZMQMessage, ZMQRequestType, ZMQServerInfo, create_zmq_socket, get_free_port

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("TQ_LOGGING_LEVEL", logging.WARNING))

# Ensure logger has a handler (for Ray Actor subprocess)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    logger.addHandler(handler)

TQ_STORAGE_POLLER_TIMEOUT = int(os.environ.get("TQ_STORAGE_POLLER_TIMEOUT", 5))  # in seconds
TQ_NUM_THREADS = int(os.environ.get("TQ_NUM_THREADS", 8))


class StorageUnitData:
    """Storage unit for managing 2D data structure (samples × fields).

    Uses dict-based storage keyed by global_index (gi) instead of pre-allocated list.
    This allows O(1) insert/delete without index translation and avoids capacity bloat.

    Data Structure Example:
        field_data = {
            "field_name1": {gi0: item1, gi3: item2, ...},
            "field_name2": {gi0: item3, gi3: item4, ...},
        }
    """

    def __init__(self, storage_size: int):
        # field_name -> {gi: data} nested dict
        self.field_data: dict[str, dict] = {}
        # Capacity upper bound (not pre-allocated list length)
        self.storage_size = storage_size

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
        existing_keys: set = set()
        for fd in self.field_data.values():
            existing_keys.update(fd.keys())
        new_global_keys = [k for k in global_indexes if k not in existing_keys]
        if len(existing_keys) + len(new_global_keys) > self.storage_size:
            raise ValueError(
                f"Storage capacity exceeded: {len(existing_keys)} existing + "
                f"{len(new_global_keys)} new > {self.storage_size}"
            )
        for f, values in field_data.items():
            if f not in self.field_data:
                self.field_data[f] = {}
            for key, val in zip(global_indexes, values, strict=False):
                self.field_data[f][key] = val

    def clear(self, keys: list[int]) -> None:
        """Remove data at given global index keys, immediately freeing memory.

        Args:
            keys: Global indexes to remove.
        """
        for f in self.field_data:
            for key in keys:
                self.field_data[f].pop(key, None)


@ray.remote(num_cpus=1)
class SimpleStorageUnit:
    """A storage unit that provides distributed data storage functionality.

    This class represents a storage unit that can store data in a 2D structure
    (samples × data fields) and provides ZMQ-based communication for put/get/clear operations.

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

        # Internal communication address for proxy and workers
        self._inproc_addr = f"inproc://simple_storage_workers_{self.storage_unit_id}"

        # Shutdown event for graceful termination
        self._shutdown_event = Event()

        # Placeholder for zmq_context, proxy_thread and worker_threads
        self.zmq_context: Optional[zmq.Context] = None
        self.put_get_socket: Optional[zmq.Socket] = None
        self.proxy_thread: Optional[Thread] = None
        self.worker_thread: Optional[Thread] = None

        self._init_zmq_socket()
        self._start_process_put_get()

        # Register finalizer for graceful cleanup when garbage collected
        self._finalizer = weakref.finalize(
            self,
            self._shutdown_resources,
            self._shutdown_event,
            self.worker_thread,
            self.proxy_thread,
            self.zmq_context,
            self.put_get_socket,
        )

    def _init_zmq_socket(self) -> None:
        """
        Initialize ZMQ socket connections between storage unit and controller/clients:
        - put_get_socket (ROUTER): Handle put/get requests from clients.
        - worker_socket (DEALER): Backend socket for worker communication.
        """
        self.zmq_context = zmq.Context()

        # Frontend: ROUTER for receiving client requests
        self.put_get_socket = create_zmq_socket(self.zmq_context, zmq.ROUTER)
        self._node_ip = get_node_ip_address()

        while True:
            try:
                self._put_get_socket_port = get_free_port()
                self.put_get_socket.bind(f"tcp://{self._node_ip}:{self._put_get_socket_port}")
                break
            except zmq.ZMQError:
                logger.warning(f"[{self.storage_unit_id}]: Try to bind ZMQ sockets failed, retrying...")
                continue

        # Backend: DEALER for worker communication (connected via zmq.proxy)
        self.worker_socket = create_zmq_socket(self.zmq_context, zmq.DEALER)
        self.worker_socket.bind(self._inproc_addr)

        self.zmq_server_info = ZMQServerInfo(
            role=TransferQueueRole.STORAGE,
            id=str(self.storage_unit_id),
            ip=self._node_ip,
            ports={"put_get_socket": self._put_get_socket_port},
        )

    def _start_process_put_get(self) -> None:
        """Start worker threads and ZMQ proxy for handling requests."""

        # Start worker thread
        self.worker_thread = Thread(
            target=self._worker_routine,
            name=f"StorageUnitWorkerThread-{self.storage_unit_id}",
            daemon=True,
        )
        self.worker_thread.start()

        time.sleep(0.5)  # make sure worker thread is ready before zmq.proxy forwarding messages

        # Start proxy thread (ROUTER <-> DEALER)
        self.proxy_thread = Thread(
            target=self._proxy_routine,
            name=f"StorageUnitProxyThread-{self.storage_unit_id}",
            daemon=True,
        )
        self.proxy_thread.start()

    def _proxy_routine(self) -> None:
        """ZMQ proxy for message forwarding between frontend ROUTER and backend DEALER."""
        logger.info(f"[{self.storage_unit_id}]: start ZMQ proxy...")
        try:
            zmq.proxy(self.put_get_socket, self.worker_socket)
        except zmq.ContextTerminated:
            logger.info(f"[{self.storage_unit_id}]: ZMQ Proxy stopped gracefully (Context Terminated)")
        except Exception as e:
            if self._shutdown_event.is_set():
                logger.info(f"[{self.storage_unit_id}]: ZMQ Proxy shutting down...")
            else:
                logger.error(f"[{self.storage_unit_id}]: ZMQ Proxy unexpected error: {e}")

    def _worker_routine(self) -> None:
        """Worker thread for processing requests."""
        # Each worker must have its own socket
        worker_socket = create_zmq_socket(self.zmq_context, zmq.DEALER)
        worker_socket.connect(self._inproc_addr)

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
                messages = worker_socket.recv_multipart()
                identity = messages[0]
                serialized_msg = messages[1:]

                request_msg = ZMQMessage.deserialize(serialized_msg)
                operation = request_msg.request_type

                try:
                    logger.debug(f"[{self.storage_unit_id}]: worker received operation: {operation}")

                    # Process request
                    if operation == ZMQRequestType.PUT_DATA:
                        with perf_monitor.measure(op_type="PUT_DATA"):
                            response_msg = self._handle_put(request_msg)
                    elif operation == ZMQRequestType.GET_DATA:
                        with perf_monitor.measure(op_type="GET_DATA"):
                            response_msg = self._handle_get(request_msg)
                    elif operation == ZMQRequestType.CLEAR_DATA:
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
            field_data = data_parts.body["data"]  # field_data should be a TensorDict.
            with limit_pytorch_auto_parallel_threads(
                target_num_threads=TQ_NUM_THREADS, info=f"[{self.storage_unit_id}] _handle_put"
            ):
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
    def _shutdown_resources(
        shutdown_event: Event,
        worker_thread: Optional[Thread],
        proxy_thread: Optional[Thread],
        zmq_context: Optional[zmq.Context],
        put_get_socket: Optional[zmq.Socket],
    ) -> None:
        """Clean up resources on garbage collection."""
        logger.info("Shutting down SimpleStorageUnit resources...")

        # Signal all threads to stop
        shutdown_event.set()

        # Terminate put_get_socket
        if put_get_socket:
            put_get_socket.close(linger=0)

        # Terminate ZMQ context to unblock proxy and workers
        if zmq_context:
            zmq_context.term()

        # Wait for threads to finish (with timeout)
        if worker_thread and worker_thread.is_alive():
            worker_thread.join(timeout=5)
        if proxy_thread and proxy_thread.is_alive():
            proxy_thread.join(timeout=5)

        logger.info("SimpleStorageUnit resources shutdown complete.")

    def get_zmq_server_info(self) -> ZMQServerInfo:
        """Get the ZMQ server information for this storage unit.

        Returns:
            ZMQServerInfo containing connection details for this storage unit.
        """
        return self.zmq_server_info
