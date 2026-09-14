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

"""A key removed by a concurrent ``clear`` must stay distinguishable from a real storage fault."""

import logging

import pytest
import torch

from transfer_queue.storage.payload_transfer.zmq import ZmqPayloadTransfer
from transfer_queue.storage.simple_storage import (
    KEY_NOT_FOUND_MARKER,
    StorageKeyNotFoundError,
    StorageUnitData,
)
from transfer_queue.utils.zmq_utils import ZMQMessage, ZMQRequestType


@pytest.fixture
def storage_data():
    data = StorageUnitData()
    data.put_data({"log_probs": [torch.tensor([1.0]), torch.tensor([2.0])]}, [0, 1])
    return data


def test_cleared_key_raises_key_not_found(storage_data):
    storage_data.clear([1])

    with pytest.raises(StorageKeyNotFoundError, match="key 1 not found in field 'log_probs'"):
        storage_data.get_data(["log_probs"], [0, 1])


def test_key_not_found_is_a_key_error(storage_data):
    # Subclassing KeyError keeps callers that already catch KeyError working.
    storage_data.clear([0])

    with pytest.raises(KeyError):
        storage_data.get_data(["log_probs"], [0])


def test_unknown_field_still_raises_value_error(storage_data):
    # A field absent from the schema is a caller bug, not the clear race.
    with pytest.raises(ValueError, match="field 'missing' not found"):
        storage_data.get_data(["missing"], [0])


def test_surviving_key_is_unaffected(storage_data):
    storage_data.clear([1])

    torch.testing.assert_close(storage_data.get_data(["log_probs"], [0])["log_probs"][0], torch.tensor([1.0]))


def test_get_error_reply_is_marked_and_logged_at_debug(storage_data, caplog):
    # The reply crosses ZMQ as text, so the marker is what lets the caller rebuild the type.
    storage_data.clear([1])
    transfer = ZmqPayloadTransfer()
    request = ZMQMessage.create(
        request_type=ZMQRequestType.GET_DATA,
        sender_id="client_0",
        body={"fields": ["log_probs"], "global_indexes": [1]},
    )

    def load_data(fields, global_indexes):
        try:
            return storage_data.get_data(fields, global_indexes)
        except StorageKeyNotFoundError as exc:
            raise StorageKeyNotFoundError(f"{KEY_NOT_FOUND_MARKER}: {exc}") from exc

    with caplog.at_level(logging.DEBUG, logger="transfer_queue.storage.payload_transfer.zmq"):
        reply = transfer.handle_request(
            request,
            storage_id="storage_unit_0",
            load_data=load_data,
            store_data=lambda *_: None,
        )

    assert reply.request_type == ZMQRequestType.GET_ERROR
    assert KEY_NOT_FOUND_MARKER in reply.body["message"]
    assert [record for record in caplog.records if record.levelno >= logging.ERROR] == []
