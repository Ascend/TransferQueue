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

"""Payload transfer contract and NIXL-UCX configuration tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from transfer_queue.storage.payload_transfer import (
    PayloadTransferError,
    create_payload_transfer,
    parse_payload_transfer_config,
)
from transfer_queue.storage.payload_transfer.nixl import PayloadDescriptor
from transfer_queue.storage.payload_transfer.nixl_ucx_runtime import _configure_ucx_environment
from transfer_queue.storage.payload_transfer.zmq import ZmqPayloadTransfer
from transfer_queue.utils.zmq_utils import ZMQMessage, ZMQRequestType


def test_payload_descriptor_preserves_frame_layout():
    descriptor = PayloadDescriptor("framed", 4 + 8 * 2 + 5, (2, 3))
    descriptor.validate()
    assert PayloadDescriptor.from_dict(descriptor.to_dict()) == descriptor

    with pytest.raises(PayloadTransferError, match="packed payload length"):
        PayloadDescriptor("framed", 5, (2, 3)).validate()


def test_payload_descriptor_requires_frame_layout_and_rejects_negative_lengths():
    with pytest.raises(KeyError, match="frame_sizes"):
        PayloadDescriptor.from_dict({"transfer_id": "payload", "payload_bytes": 3})

    with pytest.raises(PayloadTransferError, match="negative"):
        PayloadDescriptor.from_dict({"transfer_id": "payload", "payload_bytes": -1, "frame_sizes": [1]})


def test_yaml_ucx_settings_override_process_environment(monkeypatch):
    monkeypatch.setenv("UCX_TLS", "sm")
    monkeypatch.delenv("UCX_IB_GID_INDEX", raising=False)

    configured = _configure_ucx_environment(
        {
            "UCX_TLS": "tcp",
            "UCX_IB_GID_INDEX": 3,
        }
    )

    assert configured == {"UCX_TLS": "tcp", "UCX_IB_GID_INDEX": "3"}
    assert os.environ["UCX_TLS"] == "tcp"
    assert os.environ["UCX_IB_GID_INDEX"] == "3"


def test_empty_yaml_ucx_settings_preserve_process_environment(monkeypatch):
    monkeypatch.setenv("UCX_NET_DEVICES", "custom_hca:2")

    assert _configure_ucx_environment({}) == {}
    assert os.environ["UCX_NET_DEVICES"] == "custom_hca:2"


def test_payload_transfer_rejects_unsupported_backend():
    with pytest.raises(ValueError, match="expected 'zmq' or 'nixl-ucx'"):
        create_payload_transfer({"backend": "unsupported"})


def test_factory_returns_zmq_payload_transfer():
    transfer = create_payload_transfer({"backend": "zmq"})

    assert isinstance(transfer, ZmqPayloadTransfer)
    assert transfer.bootstrap_info() is None


def test_nixl_factory_rejects_incomplete_peer_endpoints():
    with pytest.raises(RuntimeError, match="payload transfer endpoints are missing"):
        create_payload_transfer(
            {"backend": "nixl-ucx"},
            control_peer_infos={"storage": object()},
        )


def test_zmq_payload_transfer_handles_put_and_get_requests():
    stored = {}
    transfer = ZmqPayloadTransfer()

    put_request = ZMQMessage.create(
        request_type=ZMQRequestType.PUT_DATA,
        sender_id="manager",
        receiver_id="storage",
        body={"global_indexes": [1], "data": {"value": [42]}},
    )
    put_response = transfer.handle_request(
        put_request,
        storage_id="storage",
        load_data=lambda fields, indexes: {field: stored[field] for field in fields},
        store_data=lambda indexes, data, parser: stored.update(data),
    )

    assert put_response.request_type == ZMQRequestType.PUT_DATA_RESPONSE
    assert stored == {"value": [42]}

    get_request = ZMQMessage.create(
        request_type=ZMQRequestType.GET_DATA,
        sender_id="manager",
        receiver_id="storage",
        body={"global_indexes": [1], "fields": ["value"]},
    )
    get_response = transfer.handle_request(
        get_request,
        storage_id="storage",
        load_data=lambda fields, indexes: {field: stored[field] for field in fields},
        store_data=lambda indexes, data, parser: stored.update(data),
    )

    assert get_response.request_type == ZMQRequestType.GET_DATA_RESPONSE
    assert get_response.body["data"] == {"value": [42]}


def test_payload_transfer_config_extracts_ucx_settings():
    assert parse_payload_transfer_config(
        {
            "backend": "nixl-ucx",
            "ucx_env_vars": {"UCX_TLS": "rc", "UCX_IB_GID_INDEX": 3},
        }
    ) == ("nixl-ucx", {"ucx_env_vars": {"UCX_TLS": "rc", "UCX_IB_GID_INDEX": 3}})


def test_public_config_defaults_to_zmq_payload_transfer():
    config = OmegaConf.load(Path(__file__).parents[1] / "transfer_queue/config.yaml")

    assert config.backend.SimpleStorage.payload_transfer.backend == "zmq"
