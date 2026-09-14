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

"""Construction helper for SimpleStorage payload transfer strategies."""

from collections.abc import Mapping

from transfer_queue.storage.payload_transfer.base import PayloadTransfer
from transfer_queue.utils.zmq_utils import ZMQServerInfo

_SUPPORTED_TRANSFERS = frozenset({"nixl-ucx", "zmq"})


def parse_payload_transfer_config(value: object | None = None) -> tuple[str, dict[str, object]]:
    """Return the backend name and options from the payload transfer block."""
    config = {"backend": "zmq"} if value is None else value
    if not isinstance(config, Mapping):
        raise TypeError("SimpleStorage.payload_transfer must be a mapping")

    config = dict(config)
    if "backend" not in config:
        raise ValueError("SimpleStorage.payload_transfer.backend is required")
    backend = str(config.pop("backend")).strip().lower()
    if backend not in _SUPPORTED_TRANSFERS:
        raise ValueError(f"unsupported SimpleStorage payload transfer: {backend!r}; expected 'zmq' or 'nixl-ucx'")
    return backend, config


def create_payload_transfer(
    value: object | None = None,
    *,
    peer_infos: Mapping[str, object] | None = None,
    control_peer_infos: Mapping[str, ZMQServerInfo] | None = None,
) -> PayloadTransfer:
    """Create the configured SimpleStorage payload transfer strategy."""
    backend, options = parse_payload_transfer_config(value)
    if backend == "zmq":
        from transfer_queue.storage.payload_transfer.zmq import ZmqPayloadTransfer

        return ZmqPayloadTransfer()
    if backend == "nixl-ucx":
        from transfer_queue.storage.payload_transfer.nixl import NixlPayloadTransfer

        ucx_env_vars = options.get("ucx_env_vars")
        if ucx_env_vars is not None and not isinstance(ucx_env_vars, Mapping):
            raise TypeError("SimpleStorage.payload_transfer.ucx_env_vars must be a mapping")
        return NixlPayloadTransfer(
            ucx_env_vars=None if ucx_env_vars is None else dict(ucx_env_vars),
            peer_infos=peer_infos,
            control_peer_infos=control_peer_infos,
        )

    raise RuntimeError(f"unhandled SimpleStorage payload transfer: {backend!r}")
