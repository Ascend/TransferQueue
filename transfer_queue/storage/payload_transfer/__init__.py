"""Selectable SimpleStorage payload transports."""

from collections.abc import Mapping

from transfer_queue.storage.payload_transfer.base import (
    Operation,
    PayloadTransfer,
    RequestHeader,
    StorageRejected,
    TransferTimeout,
    decode_spans,
    encode_spans,
)


def payload_backend(config) -> str:
    if config is None:
        return "zmq"
    if not isinstance(config, Mapping):
        raise TypeError("SimpleStorage.payload_transfer must be a mapping")
    backend = config.get("backend", "zmq")
    if backend not in ("zmq", "nixl-ucx"):
        raise ValueError(f"unsupported SimpleStorage payload backend: {backend!r}")
    return backend


def create_payload_transfer(config, *, owner_id, pool=None, timeout=200) -> PayloadTransfer:
    if payload_backend(config) == "zmq":
        from transfer_queue.storage.payload_transfer.zmq import ZmqPayloadTransfer

        return ZmqPayloadTransfer(owner_id=owner_id, pool=pool)

    from transfer_queue.storage.payload_transfer.nixl import NixlPayloadTransfer

    ucx_env = config.get("ucx_env_vars")
    if ucx_env is not None and not isinstance(ucx_env, Mapping):
        raise TypeError("SimpleStorage.payload_transfer.ucx_env_vars must be a mapping")
    return NixlPayloadTransfer(ucx_env=ucx_env, timeout=timeout)


__all__ = [
    "Operation",
    "PayloadTransfer",
    "RequestHeader",
    "StorageRejected",
    "TransferTimeout",
    "create_payload_transfer",
    "decode_spans",
    "encode_spans",
    "payload_backend",
]
