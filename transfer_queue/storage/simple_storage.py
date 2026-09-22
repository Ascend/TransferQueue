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
import pickle
import socket
import time
import weakref
from collections.abc import Mapping
from concurrent.futures import Future
from queue import Empty, SimpleQueue
from threading import Event, Thread
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import psutil
import ray
import zmq

from transfer_queue.storage.payload_transfer import DeferredResponse, PayloadTransfer, create_payload_transfer
from transfer_queue.utils.common import limit_pytorch_auto_parallel_threads
from transfer_queue.utils.enum_utils import Role
from transfer_queue.utils.logging_utils import get_logger
from transfer_queue.utils.perf_utils import IntervalPerfMonitor
from transfer_queue.utils.zmq_utils import (
    ZMQMessage,
    ZMQRequestType,
    ZMQServerInfo,
    create_zmq_socket,
    format_zmq_address,
    get_free_port,
    get_node_ip_address,
)

if TYPE_CHECKING:
    from transfer_queue.metrics import TQMetricsExporter

logger = get_logger(__name__)

TQ_STORAGE_POLLER_TIMEOUT = int(os.environ.get("TQ_STORAGE_POLLER_TIMEOUT", 5))  # in seconds
TQ_NUM_THREADS = int(os.environ.get("TQ_NUM_THREADS", 8))

# Marks a GET_ERROR reply as "the key is gone" so the caller can tell it apart from a real fault.
KEY_NOT_FOUND_MARKER = "TQKeyNotFound"


def _queue_deferred_response(
    identity: bytes,
    response: DeferredResponse,
    completions: SimpleQueue[tuple[bytes, Future[ZMQMessage]]],
    wakeup: socket.socket,
    shutdown_event: Event,
) -> None:
    """Wake the owning worker when an asynchronous response becomes ready."""
    routing_identity = bytes(identity)

    def ready(future: Future[ZMQMessage]) -> None:
        if shutdown_event.is_set():
            return
        completions.put((routing_identity, future))
        try:
            wakeup.send(b"\0")
        except (BlockingIOError, OSError):
            pass

    response.future.add_done_callback(ready)


def _drain_deferred_responses(
    completions: SimpleQueue[tuple[bytes, Future[ZMQMessage]]],
    worker_socket: zmq.Socket,
    storage_id: str,
) -> None:
    """Send completed responses from the only thread that owns the ZMQ socket."""
    while True:
        try:
            identity, future = completions.get_nowait()
        except Empty:
            return
        try:
            response = future.result()
        except Exception as exc:
            response = ZMQMessage.create(
                request_type=ZMQRequestType.PUT_GET_ERROR,
                sender_id=storage_id,
                body={"message": f"{storage_id}, deferred response failed: {exc}."},
            )
        worker_socket.send_multipart([identity] + response.serialize(), copy=False)


class StorageKeyNotFoundError(KeyError):
    """Raised when a requested global index is absent from a storage unit.

    Reads and ``clear`` are concurrent by design, so a key returned by ``kv_retrieve_meta`` can be
    cleared before ``get_data`` reaches the storage unit. Callers that tolerate that race catch this
    instead of matching on message text.
    """


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

    def __init__(self, storage_size: int | None = None):
        # field_name -> {global_index: data} nested dict
        self.field_data: dict[str, dict] = {}
        # Capacity upper bound (None means unlimited)
        self.storage_size = storage_size
        # Track active global_index keys for O(1) capacity checks
        self._active_keys: set = set()

    @property
    def active_key_count(self) -> int:
        """Number of active keys currently stored."""
        return len(self._active_keys)

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
                raise StorageKeyNotFoundError(f"StorageUnitData get_data: key {e} not found in field '{field}'") from e
        return result

    def put_data(self, field_data: dict[str, Any], global_indexes: list) -> None:
        """Put data into storage.

        Args:
            field_data: Dict with field names as keys, data list as values.
            global_indexes: Global indexes to use as dict keys.
        """
        # Capacity is enforced per unique sample key, not counted per-field
        if self.storage_size is not None:
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
            field_dict = self.field_data[f]
            for key, val in zip(global_indexes, values, strict=True):
                field_dict[key] = val
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

    def __init__(
        self,
        storage_unit_size: int | None = None,
        payload_transfer: Mapping[str, object] | None = None,
    ):
        """Initialize a SimpleStorageUnit with the specified size.

        Args:
            storage_unit_size: Maximum number of elements that can be stored in this storage unit.
                If None, the storage unit has unlimited capacity.
            payload_transfer: Backend and optional backend-specific settings.
        """
        self.storage_unit_id = f"TQ_STORAGE_UNIT_{uuid4().hex[:8]}"
        self.storage_unit_size = storage_unit_size

        self.storage_data = StorageUnitData(self.storage_unit_size)
        self.payload_transfer = create_payload_transfer(payload_transfer)

        # Internal communication address for proxy and workers
        self._inproc_addr = f"inproc://simple_storage_workers_{self.storage_unit_id}"

        # Shutdown event for graceful termination
        self._shutdown_event = Event()

        # Placeholder for zmq_context, proxy_thread and worker_threads
        self.zmq_context: zmq.Context | None = None
        self.put_get_socket: zmq.Socket | None = None
        self.proxy_thread: Thread | None = None
        self.worker_thread: Thread | None = None

        self._metrics: TQMetricsExporter | None = None

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
            self.payload_transfer,
        )

    def _init_zmq_socket(self) -> None:
        """
        Initialize ZMQ socket connections between storage unit and controller/clients:
        - put_get_socket (ROUTER): Handle put/get requests from clients.
        - worker_socket (DEALER): Backend socket for worker communication.
        """
        self.zmq_context = zmq.Context()
        self._node_ip = get_node_ip_address()

        # Frontend: ROUTER for receiving client requests
        self.put_get_socket = create_zmq_socket(self.zmq_context, zmq.ROUTER, self._node_ip)

        while True:
            try:
                self._put_get_socket_port = get_free_port(ip=self._node_ip)
                self.put_get_socket.bind(format_zmq_address(self._node_ip, self._put_get_socket_port))
                break
            except zmq.ZMQError:
                logger.warning(f"[{self.storage_unit_id}]: Try to bind ZMQ sockets failed, retrying...")
                continue

        # Backend: DEALER for worker communication (connected via zmq.proxy)
        self.worker_socket = create_zmq_socket(self.zmq_context, zmq.DEALER, self._node_ip)
        self.worker_socket.bind(self._inproc_addr)

        self.zmq_server_info = ZMQServerInfo(
            role=Role.STORAGE,
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

        worker_socket = create_zmq_socket(self.zmq_context, zmq.DEALER, self._node_ip)
        worker_socket.connect(self._inproc_addr)

        poller = zmq.Poller()
        poller.register(worker_socket, zmq.POLLIN)
        wakeup_reader, wakeup_writer = socket.socketpair()
        wakeup_reader.setblocking(False)
        wakeup_writer.setblocking(False)
        wakeup_fd = wakeup_reader.fileno()
        poller.register(wakeup_fd, zmq.POLLIN)
        deferred_completions: SimpleQueue[tuple[bytes, Future[ZMQMessage]]] = SimpleQueue()

        logger.info(f"[{self.storage_unit_id}]: worker thread started...")
        perf_monitor = IntervalPerfMonitor(caller_name=f"{self.storage_unit_id}")

        while not self._shutdown_event.is_set():
            monitor = self._metrics if self._metrics is not None else perf_monitor
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

            if wakeup_fd in socks:
                try:
                    while wakeup_reader.recv(4096):
                        pass
                except BlockingIOError:
                    pass
                try:
                    _drain_deferred_responses(deferred_completions, worker_socket, self.storage_unit_id)
                except zmq.ZMQError as exc:
                    logger.warning(f"[{self.storage_unit_id}]: deferred response send failed: {exc}")
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

                    metric_op = {
                        ZMQRequestType.PUT_DATA: "PUT_DATA",
                        ZMQRequestType.GET_DATA: "GET_DATA",
                        ZMQRequestType.PUT_DATA_COMMIT: "PUT_DATA",
                        ZMQRequestType.GET_DATA_COMMIT: "GET_DATA",
                    }.get(operation)
                    if metric_op is not None:
                        with monitor.measure(op_type=metric_op):
                            response_msg = self.payload_transfer.handle_request(
                                request_msg,
                                storage_id=self.storage_unit_id,
                                load_data=self._load_data,
                                store_data=self._put_decoded_data,
                            )
                    else:
                        response_msg = self.payload_transfer.handle_request(
                            request_msg,
                            storage_id=self.storage_unit_id,
                            load_data=self._load_data,
                            store_data=self._put_decoded_data,
                        )

                    if response_msg is None and operation == ZMQRequestType.CLEAR_DATA:  # type: ignore[arg-type]
                        with monitor.measure(op_type="CLEAR_DATA"):
                            response_msg = self._handle_clear(request_msg)
                    elif response_msg is None and operation == ZMQRequestType.GET_METRICS:  # type: ignore[arg-type]
                        response_msg = self._handle_get_metrics()
                    elif response_msg is None and operation == ZMQRequestType.SAVE_STORAGE_CHECKPOINT:  # type: ignore[arg-type]
                        response_msg = self._handle_save_checkpoint(request_msg)
                    elif response_msg is None and operation == ZMQRequestType.LOAD_STORAGE_CHECKPOINT:  # type: ignore[arg-type]
                        response_msg = self._handle_load_checkpoint(request_msg)
                    elif response_msg is None:
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

                if isinstance(response_msg, DeferredResponse):
                    _queue_deferred_response(
                        bytes(identity),
                        response_msg,
                        deferred_completions,
                        wakeup_writer,
                        self._shutdown_event,
                    )
                else:
                    worker_socket.send_multipart([identity] + response_msg.serialize(), copy=False)

        logger.info(f"[{self.storage_unit_id}]: worker stopped.")
        poller.unregister(wakeup_fd)
        poller.unregister(worker_socket)
        wakeup_reader.close()
        wakeup_writer.close()
        worker_socket.close(linger=0)

    def _put_decoded_data(self, global_indexes: list[int], field_data: dict[str, Any], data_parser: Any) -> None:
        """Validate parsed data and store it."""
        if data_parser is not None:
            if not callable(data_parser):
                raise TypeError(f"data_parser must be callable, got {type(data_parser).__name__}")
            original_keys = set(field_data)
            original_lengths = {key: self._field_length(value) for key, value in field_data.items()}
            field_data = data_parser(field_data)
            if not isinstance(field_data, dict):
                raise TypeError(f"data_parser must return a dict, got {type(field_data).__name__}")
            if set(field_data) != original_keys:
                raise ValueError(
                    f"data_parser must not change dict keys. Original keys: {sorted(original_keys)}, "
                    f"got: {sorted(field_data)}"
                )
            for key, value in field_data.items():
                original_length = original_lengths[key]
                new_length = self._field_length(value)
                if original_length is not None and new_length is not None and original_length != new_length:
                    raise ValueError(
                        f"data_parser changed the number of elements for key '{key}': "
                        f"expected {original_length}, got {new_length}"
                    )
        self.storage_data.put_data(field_data, global_indexes)

    @staticmethod
    def _field_length(value: Any) -> int | None:
        if hasattr(value, "shape") and isinstance(value.shape, tuple | list) and len(value.shape) > 0:
            return value.shape[0]
        try:
            return len(value)
        except Exception:
            return None

    def _load_data(self, fields: list[str], global_indexes: list[int]) -> dict[str, list]:
        try:
            return self.storage_data.get_data(fields, global_indexes)
        except StorageKeyNotFoundError as exc:
            raise StorageKeyNotFoundError(f"{KEY_NOT_FOUND_MARKER}: {exc}") from exc

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

    def _handle_get_metrics(self) -> ZMQMessage:
        """Handle GET_METRICS request by returning storage unit statistics.

        Returns:
            ZMQMessage containing storage unit ID, capacity, active keys,
            process RSS memory, and per-operation request stats.
        """
        try:
            process_rss = psutil.Process().memory_info().rss
        except Exception:
            process_rss = 0

        metrics = {
            "storage_unit_id": self.storage_unit_id,
            "capacity": self.storage_unit_size,
            "active_keys": self.storage_data.active_key_count,
            "process_rss_bytes": process_rss,
        }

        # Include per-operation stats if Prometheus metrics are enabled
        if self._metrics is not None:
            op_stats = {}
            for op_type in ("PUT_DATA", "GET_DATA", "CLEAR_DATA"):
                try:
                    hist = self._metrics.request_duration.labels(op_type=op_type)
                    counter = self._metrics.request_total.labels(op_type=op_type)
                    duration_sum = hist._sum.get()
                    # Build cumulative counts once, reuse for total and quantiles
                    cumulative_counts = self._cumulative_bucket_counts(hist)
                    duration_count = cumulative_counts[-1] if cumulative_counts else 0
                    op_stats[op_type] = {
                        "request_count": counter._value.get(),
                        "latency_avg": duration_sum / duration_count if duration_count > 0 else 0,
                        "latency_p50": self._quantile_from_cumulative(hist, cumulative_counts, 0.50),
                        "latency_p99": self._quantile_from_cumulative(hist, cumulative_counts, 0.99),
                    }
                except (AttributeError, TypeError, ZeroDivisionError) as e:
                    logger.debug(f"[{self.storage_unit_id}]: Failed to extract metrics for {op_type}: {e}")
            if op_stats:
                metrics["op_stats"] = op_stats

        return ZMQMessage.create(
            request_type=ZMQRequestType.METRICS_RESPONSE,  # type: ignore[arg-type]
            sender_id=self.storage_unit_id,
            body=metrics,
        )

    def _handle_save_checkpoint(self, data_parts) -> ZMQMessage:
        """Serialize storage unit data directly to a file.

        Args:
            data_parts: ZMQMessage from client, containing ``path`` in body:
                absolute path for the output .pkl file. The caller must ensure
                this path is reachable from the node running this actor
                (shared filesystem required for multi-node setups).

        Returns:
            ZMQMessage with ``success=True`` on success, or ``success=False``
            and ``message`` containing the error string on failure.
        """
        path = data_parts.body["path"]
        try:
            state = {
                "storage_unit_id": self.storage_unit_id,
                "storage_unit_size": self.storage_unit_size,
                "field_data": self.storage_data.field_data,
                "active_keys": self.storage_data._active_keys,
            }
            with open(path, "wb") as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"[{self.storage_unit_id}]: saved checkpoint to {path}")
            return ZMQMessage.create(
                request_type=ZMQRequestType.SAVE_STORAGE_CHECKPOINT_RESPONSE,  # type: ignore[arg-type]
                sender_id=self.storage_unit_id,
                body={"success": True},
            )
        except Exception as e:
            logger.error(f"[{self.storage_unit_id}]: save checkpoint failed: {e}")
            return ZMQMessage.create(
                request_type=ZMQRequestType.SAVE_STORAGE_CHECKPOINT_RESPONSE,  # type: ignore[arg-type]
                sender_id=self.storage_unit_id,
                body={"success": False, "message": str(e)},
            )

    def _handle_load_checkpoint(self, data_parts) -> ZMQMessage:
        """Restore storage unit data directly from a file.

        Args:
            data_parts: ZMQMessage from client, containing ``path`` in body:
                absolute path to a .pkl file previously written by
                ``_handle_save_checkpoint``. The caller must ensure this path
                is reachable from the node running this actor
                (shared filesystem required for multi-node setups).

        Returns:
            ZMQMessage with ``success=True`` on success, or ``success=False``
            and ``message`` containing the error string on failure.
        """
        path = data_parts.body["path"]
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)

            if data["storage_unit_size"] != self.storage_unit_size:
                logger.warning(
                    f"[{self.storage_unit_id}]: storage_unit_size mismatch — "
                    f"checkpoint={data['storage_unit_size']}, current={self.storage_unit_size}"
                )

            if self.storage_data._active_keys:
                logger.warning(
                    f"[{self.storage_unit_id}]: overwriting {len(self.storage_data._active_keys)} "
                    f"existing keys with checkpoint data from {path}"
                )
            self.storage_data.field_data.clear()
            self.storage_data._active_keys.clear()
            self.storage_data.field_data = data["field_data"]
            self.storage_data._active_keys = data["active_keys"]

            logger.info(
                f"[{self.storage_unit_id}]: loaded checkpoint from {path} — "
                f"{len(data['active_keys'])} keys, {len(data['field_data'])} fields"
            )
            return ZMQMessage.create(
                request_type=ZMQRequestType.LOAD_STORAGE_CHECKPOINT_RESPONSE,  # type: ignore[arg-type]
                sender_id=self.storage_unit_id,
                body={"success": True},
            )

        except Exception as e:
            logger.error(f"[{self.storage_unit_id}]: load checkpoint failed: {e}")
            return ZMQMessage.create(
                request_type=ZMQRequestType.LOAD_STORAGE_CHECKPOINT_RESPONSE,  # type: ignore[arg-type]
                sender_id=self.storage_unit_id,
                body={"success": False, "message": str(e)},
            )

    @staticmethod
    def _cumulative_bucket_counts(hist) -> list[float]:
        """Build cumulative counts from a prometheus_client Histogram's non-cumulative buckets."""
        cumulative = 0.0
        counts = []
        for bucket in hist._buckets:
            cumulative += bucket.get()
            counts.append(cumulative)
        return counts

    @staticmethod
    def _quantile_from_cumulative(hist, cumulative_counts: list[float], q: float) -> float:
        """Estimate a quantile using pre-computed cumulative bucket counts.

        Uses linear interpolation matching Prometheus histogram_quantile() logic.
        """
        total = cumulative_counts[-1] if cumulative_counts else 0
        if total == 0:
            return 0.0
        target = q * total
        prev_bound = 0.0
        prev_cumulative = 0.0
        for bound, cum_count in zip(hist._upper_bounds, cumulative_counts, strict=False):
            if cum_count >= target:
                fraction = (
                    (target - prev_cumulative) / (cum_count - prev_cumulative) if cum_count > prev_cumulative else 0
                )
                return prev_bound + (bound - prev_bound) * fraction
            prev_bound = bound
            prev_cumulative = cum_count
        return prev_bound

    @staticmethod
    def _shutdown_resources(
        shutdown_event: Event,
        worker_thread: Thread | None,
        proxy_thread: Thread | None,
        zmq_context: zmq.Context | None,
        put_get_socket: zmq.Socket | None,
        payload_transfer: PayloadTransfer,
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

        payload_transfer.close()

        logger.info("SimpleStorageUnit resources shutdown complete.")

    def start_metrics(self, port: int = 0) -> str:
        """Initialize and start the Prometheus metrics exporter for this storage unit.

        When enabled, replaces ``IntervalPerfMonitor`` for request latency/throughput
        tracking with Prometheus counters and histograms.

        Args:
            port: HTTP port for the /metrics endpoint (0 = auto-assign).

        Returns:
            The metrics endpoint address in ``host:port`` format.
        """
        if self._metrics is not None:
            return self._metrics.endpoint
        from transfer_queue.metrics import TQMetricsExporter

        self._metrics = TQMetricsExporter(role="storage")
        endpoint = self._metrics.start(node_ip=self._node_ip, port=port)
        logger.info(f"[{self.storage_unit_id}]: Prometheus metrics exporter started on {endpoint}")
        return endpoint

    def get_zmq_server_info(self) -> ZMQServerInfo:
        """Get the ZMQ server information for this storage unit.

        Returns:
            ZMQServerInfo containing connection details for this storage unit.
        """
        return self.zmq_server_info

    def get_payload_transfer_info(self) -> dict[str, Any] | None:
        """Return bootstrap-safe transfer endpoint metadata."""
        info = self.payload_transfer.bootstrap_info()
        if info is None:
            return None
        return {"id": self.storage_unit_id, **info}
