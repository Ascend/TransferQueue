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

"""Legacy inline ZMQ payload transfer strategy for SimpleStorage."""

from __future__ import annotations

import os
from typing import Any, Callable, NoReturn

import zmq
import zmq.asyncio

from transfer_queue.storage.payload_transfer.base import PayloadTransfer
from transfer_queue.utils.common import limit_pytorch_auto_parallel_threads
from transfer_queue.utils.logging_utils import get_logger
from transfer_queue.utils.zmq_utils import ZMQMessage, ZMQRequestType

logger = get_logger(__name__)
TQ_NUM_THREADS = int(os.environ.get("TQ_NUM_THREADS", 8))
TQ_SIMPLE_STORAGE_SEND_RECV_TIMEOUT = int(os.environ.get("TQ_SIMPLE_STORAGE_SEND_RECV_TIMEOUT", 200))
KEY_NOT_FOUND_MARKER = "TQKeyNotFound"


class ZmqPayloadTransfer(PayloadTransfer):
    """Keep the original decoded-data ZMQ request protocol behind the contract."""

    async def put(
        self,
        *,
        control_socket: zmq.asyncio.Socket,
        sender_id: str,
        target_id: str,
        global_indexes: list[int],
        data: dict[str, Any],
        data_parser: Callable[[Any], Any] | None,
    ) -> None:
        request = ZMQMessage.create(
            request_type=ZMQRequestType.PUT_DATA,
            sender_id=sender_id,
            receiver_id=target_id,
            body={"global_indexes": global_indexes, "data": data, "data_parser": data_parser},
        )
        try:
            await control_socket.send_multipart(request.serialize(), copy=False)
            response = ZMQMessage.deserialize(await control_socket.recv_multipart(copy=False))
            if response.request_type != ZMQRequestType.PUT_DATA_RESPONSE:
                raise RuntimeError(
                    f"Failed to put data to storage unit {target_id}: {response.body.get('message', 'Unknown error')}"
                )
        except zmq.error.Again as exc:
            self._raise_timeout(sender_id, target_id, "put", exc)
        except Exception as exc:
            logger.error(
                f"[{sender_id}]: Unexpected error during put to storage unit {target_id}: {type(exc).__name__}: {exc}"
            )
            raise RuntimeError(f"Error in put to storage unit {target_id}: {type(exc).__name__}: {exc}") from exc

    async def get(
        self,
        *,
        control_socket: zmq.asyncio.Socket,
        sender_id: str,
        target_id: str,
        global_indexes: list[int],
        fields: list[str],
    ) -> dict[str, Any]:
        request = ZMQMessage.create(
            request_type=ZMQRequestType.GET_DATA,
            sender_id=sender_id,
            receiver_id=target_id,
            body={"global_indexes": global_indexes, "fields": fields},
        )
        try:
            await control_socket.send_multipart(request.serialize())
            response = ZMQMessage.deserialize(await control_socket.recv_multipart(copy=False))
            if response.request_type != ZMQRequestType.GET_DATA_RESPONSE:
                raise RuntimeError(
                    f"Failed to get data from storage unit {target_id}: {response.body.get('message', 'Unknown error')}"
                )
            return response.body["data"]
        except zmq.error.Again as exc:
            self._raise_timeout(sender_id, target_id, "get", exc)
        except Exception as exc:
            logger.error(f"[{sender_id}]: Unexpected error from storage unit {target_id}: {type(exc).__name__}: {exc}")
            raise RuntimeError(
                f"Error getting data from storage unit {target_id}: {type(exc).__name__}: {exc}"
            ) from exc

    def handle_request(
        self,
        request: ZMQMessage,
        *,
        storage_id: str,
        load_data: Callable[..., dict[str, Any]],
        store_data: Callable[..., None],
    ) -> ZMQMessage | None:
        if request.request_type == ZMQRequestType.PUT_DATA:
            try:
                with limit_pytorch_auto_parallel_threads(
                    target_num_threads=TQ_NUM_THREADS, info=f"[{storage_id}] _handle_put"
                ):
                    store_data(
                        request.body["global_indexes"],
                        request.body["data"],
                        request.body.get("data_parser"),
                    )
                return self._response(ZMQRequestType.PUT_DATA_RESPONSE, storage_id)
            except Exception as exc:
                return ZMQMessage.create(
                    request_type=ZMQRequestType.PUT_ERROR,
                    sender_id=storage_id,
                    body={
                        "message": f"Failed to put data into storage unit id "
                        f"#{storage_id}, detail error message: {str(exc)}"
                    },
                )

        if request.request_type == ZMQRequestType.GET_DATA:
            try:
                fields = request.body["fields"]
                global_indexes = request.body["global_indexes"]
                with limit_pytorch_auto_parallel_threads(
                    target_num_threads=TQ_NUM_THREADS, info=f"[{storage_id}] _handle_get"
                ):
                    data = load_data(fields, global_indexes)
                return self._response(ZMQRequestType.GET_DATA_RESPONSE, storage_id, {"data": data})
            except Exception as exc:
                log = logger.debug if KEY_NOT_FOUND_MARKER in str(exc) else logger.error
                log(
                    f"[{storage_id}]: _handle_get error, "
                    f"fields={fields}, global_indexes={global_indexes}: {type(exc).__name__}: {exc}"
                )
                return ZMQMessage.create(
                    request_type=ZMQRequestType.GET_ERROR,
                    sender_id=storage_id,
                    body={
                        "message": f"Failed to get data from storage unit id #{storage_id}, "
                        f"detail error message: {str(exc)}"
                    },
                )

        return None

    @staticmethod
    def _response(request_type: ZMQRequestType, sender_id: str, body: dict[str, Any] | None = None) -> ZMQMessage:
        return ZMQMessage.create(request_type=request_type, sender_id=sender_id, body=body or {})

    @staticmethod
    def _raise_timeout(sender_id: str, target_id: str, operation: str, error: Exception) -> NoReturn:
        timeout = TQ_SIMPLE_STORAGE_SEND_RECV_TIMEOUT
        if operation == "put":
            logger.error(
                f"[{sender_id}]: ZMQ recv timeout ({timeout}s) during put to storage unit {target_id}. "
                "The storage unit may be overloaded or crashed."
            )
            raise RuntimeError(f"ZMQ recv timeout ({timeout}s) during put to storage unit {target_id}") from error

        logger.error(
            f"[{sender_id}]: ZMQ recv timeout ({timeout}s) from storage unit {target_id}. "
            "The storage unit may be overloaded or crashed."
        )
        raise RuntimeError(f"ZMQ recv timeout ({timeout}s) from storage unit {target_id}") from error
