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

"""Unit tests for `close()` ownership: only the process that created TransferQueue tears it down."""

from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

import transfer_queue.interface as iface

STORED_CONF = OmegaConf.create(
    {"controller": {"zmq_info": None}, "backend": {"storage_backend": "SimpleStorage", "SimpleStorage": {}}}
)


@pytest.fixture
def fake(monkeypatch):
    for name in ("_TQ_CLIENT", "_TQ_STORAGE", "_TQ_CONTROLLER"):
        monkeypatch.setattr(iface, name, None)
    monkeypatch.setattr(iface, "_TQ_IS_OWNER", False)

    controller = MagicMock(name="controller")
    ray = MagicMock(name="ray")
    ray.get.return_value = STORED_CONF
    controller_cls = MagicMock(name="TransferQueueController")
    controller_cls.options.return_value.remote.return_value = controller
    monkeypatch.setattr(iface, "ray", ray)
    monkeypatch.setattr(iface, "TransferQueueController", controller_cls)
    monkeypatch.setattr(iface, "TransferQueueLockManager", MagicMock(name="TransferQueueLockManager"))
    monkeypatch.setattr(iface, "TransferQueueClient", MagicMock(name="TransferQueueClient"))
    monkeypatch.setattr(iface, "process_zmq_server_info", MagicMock(return_value=None))
    monkeypatch.setattr(iface, "_maybe_create_tq_storage", lambda conf: conf)
    return ray, controller, controller_cls


def test_owner_close_kills_controller(fake):
    ray, controller, _ = fake
    ray.get_actor.side_effect = ValueError("no controller yet")

    iface.init(OmegaConf.create({}))
    client, lock_managers = iface._TQ_CLIENT, iface._TQ_LOCK_MANAGERS
    assert iface._TQ_IS_OWNER and len(lock_managers) == 8

    iface.close()
    client.close.assert_called_once()
    assert [c.args for c in ray.kill.call_args_list] == [(controller,)] + [(m,) for m in lock_managers]
    assert (iface._TQ_CLIENT, iface._TQ_CONTROLLER, iface._TQ_IS_OWNER) == (None, None, False)
    assert iface._TQ_LOCK_MANAGERS == []


@pytest.mark.parametrize("num_lock_shards", [0, "8"])
def test_init_rejects_invalid_num_lock_shards(fake, num_lock_shards):
    ray, _, controller_cls = fake
    ray.get_actor.side_effect = ValueError("no controller yet")
    with pytest.raises(ValueError, match="num_lock_shards"):
        iface.init(OmegaConf.create({"controller": {"num_lock_shards": num_lock_shards}}))
    controller_cls.options.assert_not_called()


@pytest.mark.parametrize("lost_creation_race", [False, True], ids=["existing", "lost_creation_race"])
def test_attaching_close_keeps_controller_and_can_reattach(fake, lost_creation_race):
    ray, controller, controller_cls = fake
    if lost_creation_race:
        # Another process creates the named actor between our lookup and our creation attempt.
        ray.get_actor.side_effect = [ValueError("no controller yet"), controller]
        controller_cls.options.return_value.remote.side_effect = ValueError("name already taken")
    else:
        ray.get_actor.return_value = controller

    iface.init(OmegaConf.create({}))
    client = iface._TQ_CLIENT
    assert iface._TQ_CONTROLLER is controller and not iface._TQ_IS_OWNER

    iface.close()
    client.close.assert_called_once()
    ray.kill.assert_not_called()
    assert (iface._TQ_CLIENT, iface._TQ_CONTROLLER, iface._TQ_IS_OWNER) == (None, None, False)

    ray.get_actor.side_effect = None
    ray.get_actor.return_value = controller
    iface.init()
    assert iface._TQ_CONTROLLER is controller and iface._TQ_CLIENT is not None
    assert not iface._TQ_IS_OWNER
